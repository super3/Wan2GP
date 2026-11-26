# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""MiniMax H3 prompt embeddings, made apart from the video generation and handed back to it.

H3 is conditioned on its text encoder's output for the prompt (Qwen3-VL, the 32B truncated after layer 50 or the distilled 4B
through its connector): `prompt_embeds`, bfloat16 [1, tokens, 5120], and `text_token_tags`, int64 [tokens] (1 for a text token,
0 for an image token and the markers around it). With a start or end image the encoder also sees the picture, which adds its
image tokens: 390 for an 832x480 picture. So one embedding is 10,240 bytes per token in bfloat16, and its size depends on the
prompt and the pictures.

- `encode(pipe, prompt, images)`: the pipeline's own `_encode_prompt`, with each picture prepared the way the pipeline prepares
  a start or end image for the text encoder.
- `EmbeddingService`: requests answered inside one WanGP generation of the model, like a live session, so the model stays
  loaded between them (`embed_server.py` puts an HTTP endpoint in front of it).
- `to_bytes` / `from_bytes`, `to_json` / `from_json`: one embedding as safetensors (tensors `prompt_embeds` and
  `text_token_tags`, the names of WanGP's fixed prompts, plus the prompt in its metadata) or as JSON.
- `supply()` and `install()`: embeddings that the next generations use instead of running the text encoder, matched by prompt
  and by the kinds of pictures shown with it.
"""

from __future__ import annotations

import base64
import functools
import json
import queue
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

FORMAT = "h3-prompt-embedding/1"
DIM = 5120  # H3's conditioning width, both text encoders
DEFAULT_WIDTH, DEFAULT_HEIGHT = 832, 480


@dataclass
class PromptEmbedding:
    prompt: str
    prompt_embeds: torch.Tensor  # bfloat16 [1, tokens, 5120]
    text_token_tags: torch.Tensor  # int64 [tokens]
    presentation: tuple = ()  # the kinds of what the encoder was shown with the prompt, in order: ("image", "image") for a start and an end image
    presentation_digest: str = ""  # the pipeline's digest of those pictures (its text-encoder cache key)
    model: str = ""
    encode_s: float = 0.0
    width: int = 0
    height: int = 0

    @property
    def tokens(self) -> int:
        return int(self.prompt_embeds.shape[-2])

    @property
    def dim(self) -> int:
        return int(self.prompt_embeds.shape[-1])

    @property
    def nbytes(self) -> int:
        return self.prompt_embeds.numel() * self.prompt_embeds.element_size() + self.text_token_tags.numel() * self.text_token_tags.element_size()

    def info(self) -> dict:
        return {"format": FORMAT, "prompt": self.prompt, "tokens": self.tokens, "dim": self.dim, "dtype": str(self.prompt_embeds.dtype).replace("torch.", ""),
                "bytes": self.nbytes, "presentation": list(self.presentation), "presentation_digest": self.presentation_digest, "model": self.model,
                "encode_s": round(self.encode_s, 4), "width": self.width, "height": self.height}


# -- the pictures, as the pipeline shows them to the text encoder -------------------------------------------------------------

def _pipeline_helpers():
    from models.minimax_h3 import pipeline as h3

    return h3._pil_to_video, h3._resize_video, h3._as_video, h3._qwen_frames


def _image(value):
    from PIL import Image
    import io

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    return Image.open(value).convert("RGB")


def presentation_for(images, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, helpers=None) -> list[dict]:
    """A start (and end) image as the pipeline presents it to the text encoder: resized to the generation size, first frame,
    `_qwen_frames` (models/minimax_h3/pipeline.py, `prepare_keyframes` and `_add_image_condition`)."""
    pil_to_video, resize_video, as_video, qwen_frames = helpers or _pipeline_helpers()
    presentation = []
    for value in images or ():
        video = as_video(resize_video(pil_to_video(_image(value)), int(height), int(width)))[:, :1]
        presentation.append({"type": "image", "frames": qwen_frames(video.clone())})
    return presentation


def digest(pipe, prompt, presentation) -> str:
    key = getattr(pipe, "_prompt_cache_key", None)
    if not callable(key):
        return ""
    value = key(prompt, presentation)[-1]
    return value.hex() if isinstance(value, (bytes, bytearray)) else str(value)


def encode(pipe, prompt: str, images=(), width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, model: str = "", helpers=None) -> PromptEmbedding:
    """The prompt (and pictures) through the pipeline's own text encoding, the result on the CPU. Never answered from
    supplied embeddings."""
    presentation = presentation_for(images, width, height, helpers)
    encode_prompt = functools.partial(_ORIGINAL["encode"], pipe) if _ORIGINAL.get("encode") and _owns(pipe) else pipe._encode_prompt
    started = time.monotonic()
    with torch.inference_mode():
        context, tags = encode_prompt(prompt, presentation)
        context = context.detach().to("cpu")
        tags = tags.detach().to("cpu", torch.int64) if torch.is_tensor(tags) else torch.as_tensor(tags, dtype=torch.int64)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return PromptEmbedding(prompt, context.clone(), tags.clone(), tuple(item["type"] for item in presentation), digest(pipe, prompt, presentation), model,
                           time.monotonic() - started, int(width) if images else 0, int(height) if images else 0)


# -- files and JSON ---------------------------------------------------------------------------------------------------------------

def _metadata(embedding: PromptEmbedding) -> dict:
    info = embedding.info()
    return {key: json.dumps(value) if not isinstance(value, str) else value for key, value in info.items()}


def to_bytes(embedding: PromptEmbedding) -> bytes:
    from safetensors.torch import save

    return save({"prompt_embeds": embedding.prompt_embeds.contiguous(), "text_token_tags": embedding.text_token_tags.contiguous()}, metadata=_metadata(embedding))


def _safetensors_metadata(data: bytes) -> dict:
    size = struct.unpack("<Q", data[:8])[0]
    return json.loads(data[8:8 + size]).get("__metadata__", {}) or {}


def _from_info(info: dict, prompt_embeds, text_token_tags) -> PromptEmbedding:
    def get(key, default):
        value = info.get(key, default)
        if isinstance(value, str) and key not in ("prompt", "presentation_digest", "model", "format"):
            try:
                return json.loads(value)
            except ValueError:
                return default
        return value

    if get("format", FORMAT) != FORMAT:
        raise ValueError(f"not an H3 prompt embedding ({get('format', '')})")
    if prompt_embeds.ndim == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)
    if prompt_embeds.ndim != 3 or text_token_tags.ndim != 1 or prompt_embeds.shape[1] != text_token_tags.shape[0]:
        raise ValueError(f"mismatched shapes {tuple(prompt_embeds.shape)} and {tuple(text_token_tags.shape)}")
    return PromptEmbedding(str(get("prompt", "")), prompt_embeds, text_token_tags.to(torch.int64), tuple(get("presentation", [])),
                           str(get("presentation_digest", "")), str(get("model", "")), float(get("encode_s", 0.0)), int(get("width", 0)), int(get("height", 0)))


def from_bytes(data: bytes) -> PromptEmbedding:
    from safetensors.torch import load

    tensors = load(bytes(data))
    return _from_info(_safetensors_metadata(data), tensors["prompt_embeds"], tensors["text_token_tags"])


def save(embedding: PromptEmbedding, path) -> Path:
    path = Path(path)
    path.write_bytes(to_bytes(embedding))
    return path


def load(path) -> PromptEmbedding:
    return from_bytes(Path(path).read_bytes())


def to_json(embedding: PromptEmbedding) -> dict:
    """The embedding as JSON: `prompt_embeds` is base64 of the raw little-endian bfloat16 values (row-major, [1, tokens, 5120])."""
    raw = embedding.prompt_embeds.to(torch.bfloat16).contiguous().view(torch.int16).numpy().astype("<i2").tobytes()
    return {**embedding.info(), "prompt_embeds": base64.b64encode(raw).decode(), "text_token_tags": embedding.text_token_tags.tolist()}


def from_json(data: dict) -> PromptEmbedding:
    tags = torch.tensor(data["text_token_tags"], dtype=torch.int64)
    values = np.frombuffer(base64.b64decode(data["prompt_embeds"]), dtype="<i2").astype(np.int16)
    embeds = torch.from_numpy(values.copy()).view(torch.bfloat16).reshape(1, tags.shape[0], -1)
    info = {key: value for key, value in data.items() if key not in ("prompt_embeds", "text_token_tags")}
    return _from_info(info, embeds, tags)


# -- supplied embeddings: used by the next generations instead of the text encoder ------------------------------------------

SUPPLIED: dict = {}  # (prompt, presentation kinds) -> PromptEmbedding
STATS = {"hits": 0, "misses": 0, "digest_mismatches": 0}
_ORIGINAL: dict = {}
_LOCK = threading.Lock()
OPTIONS = {"strict": False}  # True: a supplied embedding is used only when its pictures' digest matches this generation's


def supply(embeddings, strict: bool | None = None) -> int:
    """Embeddings for the next generations, by prompt and picture kinds; a later one replaces an earlier one for the same key."""
    if isinstance(embeddings, PromptEmbedding):
        embeddings = [embeddings]
    count = 0
    with _LOCK:
        for embedding in embeddings:
            SUPPLIED[(embedding.prompt, tuple(embedding.presentation))] = embedding
            count += 1
        if strict is not None:
            OPTIONS["strict"] = bool(strict)
    return count


def supply_dir(folder) -> int:
    """Every `*.safetensors` embedding in a folder."""
    return supply([load(path) for path in sorted(Path(folder).glob("*.safetensors"))])


def clear() -> None:
    with _LOCK:
        SUPPLIED.clear()
        STATS.update(hits=0, misses=0, digest_mismatches=0)


def _owns(pipe) -> bool:
    return _ORIGINAL.get("cls") is not None and isinstance(pipe, _ORIGINAL["cls"])


def _supplied(self, prompt, presentation):
    with _LOCK:
        embedding = SUPPLIED.get((prompt, tuple(item.get("type") for item in presentation))) if SUPPLIED else None
    if embedding is None or getattr(self, "fixed_prompt", None) is not None:
        if SUPPLIED:
            STATS["misses"] += 1
        return None
    if OPTIONS["strict"] and embedding.presentation_digest and embedding.presentation_digest != digest(self, prompt, presentation):
        STATS["digest_mismatches"] += 1
        return None
    STATS["hits"] += 1
    device, dtype = getattr(self, "device", None), getattr(self, "dtype", None)
    return embedding.prompt_embeds.to(device=device, dtype=dtype or embedding.prompt_embeds.dtype), embedding.text_token_tags.to(device=device)


def install(pipeline_cls=None) -> None:
    """Wrap MiniMaxH3Pipeline._encode_prompt (once) so that supplied embeddings are used in its place."""
    if pipeline_cls is None:
        from models.minimax_h3.pipeline import MiniMaxH3Pipeline as pipeline_cls
    original = getattr(pipeline_cls, "_encode_prompt", None)
    if _ORIGINAL.get("cls") is not None or not callable(original):
        return

    @functools.wraps(original)
    def _encode_prompt(self, prompt, presentation):
        found = _supplied(self, prompt, presentation)
        return found if found is not None else original(self, prompt, presentation)

    _encode_prompt._h3_live_embeddings = True
    _ORIGINAL.update(cls=pipeline_cls, encode=original)
    pipeline_cls._encode_prompt = _encode_prompt


def uninstall() -> None:
    if _ORIGINAL.get("cls") is not None:
        _ORIGINAL["cls"]._encode_prompt = _ORIGINAL["encode"]
        _ORIGINAL.clear()


# -- the service: requests answered inside one generation, so that the model stays loaded -----------------------------------

@dataclass
class _Request:
    prompt: str
    images: tuple
    width: int
    height: int
    done: threading.Event = field(default_factory=threading.Event)
    result: PromptEmbedding | None = None
    error: BaseException | None = None


class ServiceNotReady(RuntimeError):
    pass


class EmbeddingService:
    """`run(pipe)` serves `request()` calls from other threads until `stop()`, on the thread of the generation that called
    it (WanGP's), one encode at a time."""

    def __init__(self, model: str = "", helpers=None):
        self.model, self.helpers = model, helpers
        self.requests: queue.Queue = queue.Queue()
        self.ready, self.stopped = threading.Event(), threading.Event()
        self.served, self.failed, self.encode_s = 0, 0, 0.0
        self.started_at = None

    @property
    def running(self) -> bool:
        return self.ready.is_set()

    def request(self, prompt: str, images=(), width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, timeout: float = 300.0) -> PromptEmbedding:
        if not self.ready.is_set():
            raise ServiceNotReady("the model is not loaded yet")
        item = _Request(prompt, tuple(images or ()), int(width), int(height))
        self.requests.put(item)
        if not item.done.wait(timeout):
            raise TimeoutError(f"no embedding after {timeout:.0f} s")
        if item.error is not None:
            raise item.error
        return item.result

    def run(self, pipe) -> None:
        self.started_at = time.time()
        self.ready.set()
        print("[H3 Live] embedding service ready", flush=True)
        try:
            while not self.stopped.is_set() and not getattr(pipe, "_interrupt", False):
                try:
                    item = self.requests.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    item.result = encode(pipe, item.prompt, item.images, item.width, item.height, self.model, self.helpers)
                    self.served += 1
                    self.encode_s += item.result.encode_s
                except BaseException as error:  # noqa: BLE001 - handed to the caller
                    item.error = error
                    self.failed += 1
                finally:
                    item.done.set()
        finally:
            self.ready.clear()
            while True:  # nobody is left waiting on a request
                try:
                    item = self.requests.get_nowait()
                except queue.Empty:
                    break
                item.error = ServiceNotReady("the embedding service stopped")
                item.done.set()
            print(f"[H3 Live] embedding service stopped after {self.served} embeddings", flush=True)

    def stop(self) -> None:
        self.stopped.set()

    def stats(self) -> dict:
        return {"ready": self.ready.is_set(), "model": self.model, "dim": DIM, "dtype": "bfloat16", "served": self.served, "failed": self.failed,
                "mean_encode_s": round(self.encode_s / self.served, 4) if self.served else None,
                "uptime_s": round(time.time() - self.started_at) if self.started_at else 0}


def fetch(url: str, prompt: str, images=(), width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, token: str = "", timeout: float = 300.0) -> PromptEmbedding:
    """One embedding from an `embed_server.py` endpoint (`url` is its base, e.g. http://host:8890)."""
    import urllib.request

    body = {"prompt": prompt, "width": int(width), "height": int(height), "format": "safetensors",
            "images": [base64.b64encode(Path(image).read_bytes() if not isinstance(image, (bytes, bytearray)) else bytes(image)).decode() for image in images or ()]}
    request = urllib.request.Request(url.rstrip("/") + "/v1/embeddings", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return from_bytes(response.read())
