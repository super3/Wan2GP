#!/usr/bin/env python3
# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""H3 Live demo: an endless real-time MiniMax H3 stream in the browser.

WanGP runs headless (its Python API). One generation becomes an endless live session of one prompt, a new seed for each clip,
paced to play in real time. A small web server shows it: `/` is the viewer (an HLS player and the generation's numbers),
`/live/<session>/live.m3u8` the stream, `/stats.json` the numbers.

Run it from WanGP's folder, with this plugin in `plugins/h3_live` and the pi-H3 DWD models available:

    python plugins/h3_live/demo/h3live_demo.py --port 8000 -- --attention sage2 --profile 1

then open http://<this machine>:8000. Real time needs a GPU that makes clips faster than they play (measured: 1.26x on an RTX
PRO 6000 at 832x480). Ctrl+C stops after the clip being generated.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "minimax_h3_fl2va_pruned_pi_dwd_2step"
PROMPT = (
    "integrated_multimodal_description: [Shot 1] A five-second continuous tracking shot of a male athlete in a red running singlet "
    "and black shorts, running at a steady pace along the outside lane of a red stadium running track on a sunny afternoon. The camera "
    "moves alongside him at the same speed, framing him from the waist up in profile, so he stays in the center of the frame while the "
    "white lane lines, the green infield and the empty stands slide past behind him. His arms swing in an even rhythm, his breathing is "
    "controlled, and his stride never changes from the first frame to the last.\n"
    "overall_soundscape: Rhythmic footsteps on the rubber track, steady breathing, a light wind, distant birds.\n"
    "non_diegetic_music: None."
)
SESSION_NAME = re.compile(r"^live_[0-9A-Za-z_-]+$")
STREAM_FILE = re.compile(r"^(live\.m3u8|clip_\d{5}\.ts)$")
CONTENT_TYPES = {".m3u8": "application/vnd.apple.mpegurl", ".ts": "video/mp2t", ".html": "text/html; charset=utf-8", ".json": "application/json"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog="Arguments after -- go to WanGP (e.g. --attention sage2 --profile 1).")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--wangp", default=".", help="WanGP's folder (default: the current folder)")
    parser.add_argument("--model", default=MODEL, help="WanGP model type")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--resolution", default="832x480")
    parser.add_argument("--frames", type=int, default=124, help="frames per clip (124 = 5.17 s at 24 fps)")
    parser.add_argument("--encoder", default="auto", choices=["auto", "nvenc", "x264"])
    parser.add_argument("--lead", type=float, default=12.0, help="at most this many seconds of video ready ahead of the stream's clock")
    parser.add_argument("--window", type=int, default=8, help="clips in the live playlist (older clip files are deleted)")
    parser.add_argument("--dry-run", action="store_true", help="no WanGP, no GPU: synthetic clips 1.3x faster than real time, to try the page")
    argv = sys.argv[1:] if argv is None else argv
    wangp_args = []
    if "--" in argv:
        split = argv.index("--")
        argv, wangp_args = argv[:split], argv[split + 1:]
    args = parser.parse_args(argv)
    args.wangp_args = wangp_args
    return args


class GpuStats:
    """nvidia-smi, read at most every 2 s whatever the number of viewers."""

    FIELDS = "name,utilization.gpu,memory.used,memory.total,power.draw,power.limit"

    def __init__(self):
        self.value, self.read_at, self.lock = {}, 0.0, threading.Lock()

    def get(self) -> dict:
        with self.lock:
            if time.monotonic() - self.read_at > 2:
                self.read_at = time.monotonic()
                try:
                    out = subprocess.run(["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits"], capture_output=True,
                                         text=True, timeout=5, check=False).stdout.splitlines()[0]
                    name, util, used, total, power, limit = [part.strip() for part in out.split(",")]
                    self.value = {"name": name, "util_pct": float(util), "memory_used_gb": round(float(used) / 1024, 1),
                                  "memory_total_gb": round(float(total) / 1024, 1), "power_w": round(float(power)), "power_limit_w": round(float(limit))}
                except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                    self.value = {}
            return dict(self.value)


class Demo:
    """The generation thread: WanGP's API, the H3 Live hook, and one endless session after another (a new one if a session ends)."""

    def __init__(self, args):
        self.args = args
        self.root = Path(args.wangp).resolve()
        self.state, self.message = "starting", "Starting WanGP"
        self.started = time.time()
        self.sessions = 0
        self.stop = threading.Event()
        self.live = None
        self.gpu = GpuStats()

    def run(self) -> None:
        try:
            self._run()
        except Exception as error:  # noqa: BLE001 - shown on the page
            traceback.print_exc()
            self.state, self.message = "error", f"{type(error).__name__}: {error}"[:400]

    def _settings(self, live):
        return live.LiveSettings(enabled=True, clips=-1, seed_step=1, encoder=self.args.encoder, keep_sessions=2, pace_lead_s=self.args.lead,
                                 playlist_window=self.args.window)

    def _dry_run(self) -> None:
        """Synthetic clips (a colour per seed, a moving bar, a tone) through the same live session, stream and page."""
        import numpy as np

        sys.path.insert(0, str(HERE.parent.parent))
        live = __import__(HERE.parent.name + ".live", fromlist=["live"])
        self.live = live
        width, height = (int(value) for value in self.args.resolution.split("x"))
        frames, fps = self.args.frames, 24

        def generate(pipe, input_prompt="", seed=0, frame_num=frames, fps=fps, **_):
            time.sleep(frame_num / fps / 1.3)
            rng = np.random.default_rng(seed)
            video = np.empty((3, frame_num, height, width), dtype=np.uint8)
            video[:] = rng.integers(40, 200, size=3, dtype=np.uint8)[:, None, None, None]
            for index in range(frame_num):
                x = int(index / frame_num * (width - 40))
                video[:, index, :, x:x + 40] = 240
            t = np.arange(round(frame_num / fps * 32000)) / 32000
            tone = (0.2 * np.sin(2 * np.pi * (220 + seed % 440) * t)).astype(np.float32)
            return {"x": video, "audio": np.stack([tone, tone], 1), "audio_sampling_rate": 32000}

        pipe = type("DryRunPipeline", (), {"_interrupt": False})()
        root = self.root / "outputs" / "h3_live_demo_dry_run"
        while not self.stop.is_set():
            self.sessions += 1
            self.state, self.message = "loading", "Starting a dry run"
            settings = self._settings(live)
            kwargs = {"input_prompt": self.args.prompt, "seed": random.randrange(1, 2**31 - 1 - 10**7), "frame_num": frames, "fps": fps}
            live.run_session(lambda pipe_, **kw: None if self.stop.is_set() else generate(pipe_, **kw), pipe, kwargs, settings, root=root)
        self.state, self.message = "stopped", "Stopped"

    def _run(self) -> None:
        if self.args.dry_run:
            return self._dry_run()
        os.chdir(self.root)
        sys.path[:0] = [str(self.root), str(self.root / "plugins")]
        from shared.api import init

        live = __import__(HERE.parent.name + ".live", fromlist=["live"])

        self.state, self.message = "loading", "Loading WanGP"
        session = init(root=str(self.root), output_dir=str(self.root / "outputs" / "h3_live_demo"), cli_args=self.args.wangp_args, console_output=False)
        session._ensure_runtime()
        problems = live.install()
        blocking = live.blocking_problems(problems)
        if blocking:
            raise RuntimeError("live mode is not available in this WanGP: " + "; ".join(blocking))
        live.configure(self._settings(live))
        self.live = live
        while not self.stop.is_set():
            seed = random.randrange(1, 2**31 - 1 - 10**7)  # clip i gets seed + i
            task = {"model_type": self.args.model, "prompt": self.args.prompt, "resolution": self.args.resolution, "video_length": self.args.frames,
                    "seed": seed, "prompt_enhancer": "", "image_prompt_type": "", "video_prompt_type": "", "audio_prompt_type": ""}
            self.sessions += 1
            self.state, self.message = "loading", "Loading the model" if self.sessions == 1 else "Starting a new session"
            print(f"[H3 Live demo] session {self.sessions}, seeds from {seed}", flush=True)
            result = session.submit_task(task).result()
            if self.stop.is_set():
                break
            errors = "; ".join(str(error) for error in result.errors)[:300]
            print(f"[H3 Live demo] the session ended ({'ok' if result.success else errors}); a new one in 5 s", flush=True)
            self.state, self.message = "restarting", errors or "The session ended; starting a new one"
            self.stop.wait(5)
        self.state, self.message = "stopped", "Stopped"

    def request_stop(self) -> None:
        self.stop.set()
        if self.live is not None:
            self.live.configure(None)  # a generation not yet started stays one clip
            self.live.request_stop()

    def current(self):
        return None if self.live is None else self.live.STATE["session"]

    def stats(self) -> dict:
        session = self.current()
        out = {"state": self.state, "message": self.message, "prompt": self.args.prompt, "model": self.args.model, "resolution": self.args.resolution,
               "frames": self.args.frames, "uptime_s": round(time.time() - self.started), "sessions": self.sessions, "gpu": self.gpu.get(),
               "session": None, "recent": []}
        if session is not None:
            summary = session.summary()
            if summary["clips"]:
                self.state, self.message = "live", ""
            out["session"] = {key: summary[key] for key in ("encoder", "clips", "errors", "seconds_of_video", "cycle_s", "realtime_x", "generate_s",
                                                             "generation_x", "first_clip_ready_s", "uptime_s")}
            out["session"]["id"] = session.dir.name
            out["recent"] = [{key: record.get(key) for key in ("clip", "segment", "seed", "generate_s", "seconds", "ready_at_s")}
                             for record in summary["records"][-24:] if "error" not in record]
            out["state"], out["message"] = self.state, self.message
        return out

    def stream_file(self, session_name: str, name: str) -> Path | None:
        session = self.current() or (None if self.live is None else self.live.STATE["last_session"])
        if session is None or not SESSION_NAME.match(session_name) or not STREAM_FILE.match(name):
            return None
        path = session.dir.parent / session_name / name
        return path if path.is_file() else None


def make_handler(demo: Demo):
    index = (HERE / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "H3LiveDemo/1"

        def log_message(self, format, *args):  # noqa: A002 - quiet: players poll the playlist every few seconds
            pass

        def _send(self, status, body: bytes, content_type: str, cache: str = "no-store", head: bool = False):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            if not head:
                self.wfile.write(body)

        def do_HEAD(self):
            self.do_GET(head=True)

        def do_GET(self, head: bool = False):
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/", "/index.html"):
                    return self._send(HTTPStatus.OK, index, CONTENT_TYPES[".html"], "no-cache", head)
                if path == "/stats.json":
                    return self._send(HTTPStatus.OK, json.dumps(demo.stats()).encode(), CONTENT_TYPES[".json"], "no-store", head)
                parts = path.split("/")
                if len(parts) == 4 and parts[1] == "live":
                    file = demo.stream_file(parts[2], parts[3])
                    if file is not None:
                        try:
                            body = file.read_bytes()
                        except OSError:  # deleted meanwhile (out of the window)
                            body = None
                        if body is not None:
                            playlist = file.suffix == ".m3u8"
                            return self._send(HTTPStatus.OK, body, CONTENT_TYPES[file.suffix], "no-store" if playlist else "public, max-age=3600", head)
                return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain", "no-store", head)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main(argv=None) -> None:
    args = parse_args(argv)
    demo = Demo(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(demo))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="h3-live-demo-http", daemon=True).start()
    print(f"[H3 Live demo] serving on http://{args.host}:{args.port}", flush=True)

    def stop(*_):
        if demo.stop.is_set():  # a second Ctrl+C: at once
            os._exit(1)
        print("[H3 Live demo] stopping after the current clip (Ctrl+C again: at once)", flush=True)
        demo.request_stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    worker = threading.Thread(target=demo.run, name="h3-live-demo", daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(0.5)
    if demo.state == "error":  # keep the page up to show the error until stopped
        while not demo.stop.wait(1):
            pass
    server.shutdown()


if __name__ == "__main__":
    main()
