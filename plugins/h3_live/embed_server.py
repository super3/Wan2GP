#!/usr/bin/env python3
# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""H3 prompt embeddings over HTTP: a prompt (and its start / end image) in, the embedding MiniMax H3 is conditioned on out.

WanGP runs headless and loads --model with one generation that does not end: H3 Live's hook turns it into an embedding
service, which encodes each request with the pipeline's own text encoder while the model stays loaded. A video generation
elsewhere then uses the embeddings instead of its text encoder (`embeddings.supply()`, or H3 Live's `embeddings_dir`).

    python plugins/h3_live/embed_server.py --port 8890 -- --attention sage2 --profile 1
    python plugins/h3_live/embed_server.py --dry-run          # no WanGP, no GPU: made-up embeddings of the right shape

POST /v1/embeddings  JSON {"prompt": "...", "images": ["<base64 PNG or JPEG>", ...], "width": 832, "height": 480, "format": "safetensors"}
    images: the start image, then the end image, as the video generation will get them (none for text only); width and
    height: the video's size, which the pictures are resized to. Answer: safetensors (application/octet-stream; tensors
    `prompt_embeds` bfloat16 [1, tokens, 5120] and `text_token_tags` int64 [tokens], the prompt and its picture kinds in the
    metadata), or with "format": "json" the same as JSON, `prompt_embeds` as base64 of the raw little-endian bfloat16 values.
    Headers X-H3-Tokens, X-H3-Dim and X-H3-Encode-S give the shape and the encode time.
GET  /v1/health  {"ready": ..., "model": ..., "dim": 5120, "served": ...}

It listens on 127.0.0.1 unless --host says otherwise; --token (or H3_EMBED_TOKEN) then asks every request for
"Authorization: Bearer <token>".
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import json
import os
import signal
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "minimax_h3_fl2va_pruned_pi_dwd_2step"
MAX_IMAGES, MAX_SIDE = 4, 4096


def _embeddings():
    if __package__:
        from . import embeddings
        return embeddings
    sys.path.insert(0, str(HERE.parent))
    return __import__(HERE.name + ".embeddings", fromlist=["embeddings"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog="Arguments after -- go to WanGP (e.g. --attention sage2 --profile 1).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--wangp", default=".", help="WanGP's folder (default: the current folder)")
    parser.add_argument("--model", default=MODEL, help="WanGP model type whose text encoder makes the embeddings")
    parser.add_argument("--token", default=os.environ.get("H3_EMBED_TOKEN", ""), help="ask every request for this bearer token")
    parser.add_argument("--max-body-mb", type=float, default=64.0)
    parser.add_argument("--dry-run", action="store_true", help="no WanGP, no GPU: made-up embeddings of the right shape, to try the endpoint")
    argv = sys.argv[1:] if argv is None else argv
    wangp_args = []
    if "--" in argv:
        split = argv.index("--")
        argv, wangp_args = argv[:split], argv[split + 1:]
    args = parser.parse_args(argv)
    args.wangp_args = wangp_args
    return args


def _validated(body: dict):
    if not isinstance(body, dict):
        raise ValueError("the body must be a JSON object")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("'prompt' must be a non-empty string")
    images = body.get("images") or []
    if not isinstance(images, list) or len(images) > MAX_IMAGES:
        raise ValueError(f"'images' must be a list of at most {MAX_IMAGES} base64 images")
    try:
        if not all(isinstance(image, str) for image in images):
            raise ValueError
        images = [base64.b64decode("".join(image.split()), validate=True) for image in images]  # line breaks allowed
    except (binascii.Error, TypeError, ValueError):
        raise ValueError("'images' must hold base64 PNG or JPEG data") from None
    width, height = body.get("width", 832), body.get("height", 480)
    if not all(isinstance(value, int) and 16 <= value <= MAX_SIDE for value in (width, height)):
        raise ValueError(f"'width' and 'height' must be whole numbers from 16 to {MAX_SIDE}")
    fmt = body.get("format", "safetensors")
    if fmt not in ("safetensors", "json"):
        raise ValueError("'format' must be 'safetensors' or 'json'")
    return prompt, images, width, height, fmt


def make_handler(service, embeddings, token: str = "", max_body: int = 64 << 20):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "H3Embeddings/1"

        def log_message(self, format, *args):  # noqa: A002 - one line per request
            print(f"[H3 embeddings] {self.address_string()} {format % args}", flush=True)

        def _send(self, status, body: bytes, content_type: str, headers=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in (headers or {}).items():
                self.send_header(name, str(value))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, value, headers=None):
            self._send(status, json.dumps(value).encode(), "application/json", headers)

        def _error(self, status, message, headers=None):
            self._json(status, {"error": message}, headers)

        def _authorized(self) -> bool:
            if not token:
                return True
            if hmac.compare_digest(self.headers.get("Authorization", "").encode(), f"Bearer {token}".encode()):
                return True
            self._error(HTTPStatus.UNAUTHORIZED, "a bearer token is required", {"WWW-Authenticate": "Bearer"})
            return False

        def do_GET(self):
            if not self._authorized():
                return
            if self.path.split("?")[0] in ("/v1/health", "/health"):
                return self._json(HTTPStatus.OK, service.stats())
            self._error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self):
            if not self._authorized():
                return
            if self.path.split("?")[0] != "/v1/embeddings":
                return self._error(HTTPStatus.NOT_FOUND, "not found")
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length <= 0:
                return self._error(HTTPStatus.LENGTH_REQUIRED, "a JSON body with Content-Length is required")
            if length > max_body:
                return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"the body is larger than {max_body} bytes")
            try:
                prompt, images, width, height, fmt = _validated(json.loads(self.rfile.read(length)))
            except (ValueError, UnicodeDecodeError) as error:
                return self._error(HTTPStatus.BAD_REQUEST, str(error))
            try:
                embedding = service.request(prompt, images, width, height)
            except embeddings.ServiceNotReady as error:
                return self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(error), {"Retry-After": 10})
            except TimeoutError as error:
                return self._error(HTTPStatus.GATEWAY_TIMEOUT, str(error))
            except Exception as error:  # noqa: BLE001 - the caller gets the reason
                traceback.print_exc()
                return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(error).__name__}: {error}"[:500])
            headers = {"X-H3-Tokens": embedding.tokens, "X-H3-Dim": embedding.dim, "X-H3-Encode-S": f"{embedding.encode_s:.4f}"}
            if fmt == "json":
                return self._json(HTTPStatus.OK, embeddings.to_json(embedding), headers)
            self._send(HTTPStatus.OK, embeddings.to_bytes(embedding), "application/octet-stream", headers)

    return Handler


class DryRunPipeline:
    """Made-up embeddings of H3's shape: 30 tokens plus one per word of the prompt, and 390 image tokens and 7 markers and
    labels per picture, values seeded by the prompt."""

    _interrupt = False

    def _encode_prompt(self, prompt, presentation):
        import torch

        tokens = 30 + len(prompt.split()) + 397 * len(presentation)
        generator = torch.Generator().manual_seed(sum(prompt.encode()) % (2**31))
        tags = torch.ones(tokens, dtype=torch.int64)
        for index in range(len(presentation)):
            tags[30 + 397 * index:30 + 397 * (index + 1)] = 0
        return torch.randn(1, tokens, 5120, generator=generator).to(torch.bfloat16), tags


def simple_helpers():
    """The pipeline's picture preparation for a CPU without WanGP (the dry run): to [-1, 1] CTHW, bicubic resize, [0, 1] THWC."""
    import torch
    import torch.nn.functional as F
    import numpy as np

    def pil_to_video(image):
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(127.5).sub_(1.0).unsqueeze(1)

    def resize_video(video, height, width):
        if video.shape[-2:] == (height, width):
            return video
        return F.interpolate(video.permute(1, 0, 2, 3), size=(height, width), mode="bicubic", align_corners=False, antialias=True).permute(1, 0, 2, 3)

    def as_video(video):
        return video if video.ndim == 4 else video.unsqueeze(1)

    def qwen_frames(video):
        return video.permute(1, 2, 3, 0).add(1.0).mul_(0.5).clamp_(0.0, 1.0)

    return pil_to_video, resize_video, as_video, qwen_frames


class Runner:
    """The generation thread: WanGP's API, H3 Live's hook, and one never-ending generation that serves the embeddings."""

    def __init__(self, args, service):
        self.args, self.service = args, service
        self.stop = threading.Event()
        self.live = None

    def run(self) -> None:
        try:
            self._run()
        except Exception:  # noqa: BLE001 - printed; health then says not ready
            traceback.print_exc()

    def _run(self) -> None:
        if self.args.dry_run:
            while not self.stop.is_set():
                self.service.run(DryRunPipeline())
            return
        root = Path(self.args.wangp).resolve()
        os.chdir(root)
        sys.path[:0] = [str(root), str(root / "plugins")]
        from shared.api import init

        live = __import__(HERE.name + ".live", fromlist=["live"])
        session = init(root=str(root), output_dir=str(root / "outputs" / "h3_embeddings"), cli_args=self.args.wangp_args, console_output=False)
        session._ensure_runtime()
        problems = live.install()
        blocking = live.blocking_problems(problems)
        if blocking:
            raise RuntimeError("H3 Live's hooks do not fit this WanGP: " + "; ".join(blocking))
        live.configure(None)
        live.STATE["embed_service"] = self.service
        self.live = live
        task = {"model_type": self.args.model, "prompt": "H3 embedding service", "resolution": "832x480", "video_length": 17, "seed": 1,
                "prompt_enhancer": "", "image_prompt_type": "", "video_prompt_type": "", "audio_prompt_type": ""}
        while not self.stop.is_set():
            print(f"[H3 embeddings] loading {self.args.model}", flush=True)
            result = session.submit_task(task).result()  # returns when the service stops
            if not self.stop.is_set():
                print(f"[H3 embeddings] the service generation ended ({'ok' if result.success else result.errors}); again in 5 s", flush=True)
                self.stop.wait(5)

    def request_stop(self) -> None:
        self.stop.set()
        self.service.stop()


def main(argv=None) -> None:
    args = parse_args(argv)
    embeddings = _embeddings()
    service = embeddings.EmbeddingService(model="dry-run" if args.dry_run else args.model, helpers=simple_helpers() if args.dry_run else None)
    runner = Runner(args, service)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service, embeddings, args.token, int(args.max_body_mb * 2**20)))
    server.daemon_threads = True
    print(f"[H3 embeddings] http://{args.host}:{server.server_address[1]}/v1/embeddings ({'dry run' if args.dry_run else args.model})", flush=True)
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.token:
        print("[H3 embeddings] listening beyond this machine without --token: anyone who can reach the port can use the GPU", flush=True)

    def stop(*_):
        runner.request_stop()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    worker = threading.Thread(target=runner.run, name="h3-embeddings", daemon=True)
    worker.start()
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        runner.request_stop()
        worker.join(timeout=30)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
