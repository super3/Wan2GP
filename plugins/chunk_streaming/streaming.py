# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Chunk-streaming instrumentation for MiniMax H3 video generation.

The H3 video VAE decodes temporally: `AutoencoderKLMiniMaxH3._decode` walks the
latent video in chunks of `tokens_chunk_size` tokens, cross-fades the
`frame_overlap` boundary frames, and finalizes a run of pixel frames per chunk
(17 frames per chunk at the shipped geometry, ~0.7 s of video at 24 fps). Today
every caller waits for the whole tensor; this module exposes the per-chunk
completion boundary so finished frames can leave the worker while the tail of
the clip is still decoding.

`activate()` monkeypatches `_decode` with a copy whose only additions are
recorder callbacks; the returned tensor is bit-identical to the original.
`_decode` converts every finalized chunk with `_prepare_decoded_chunk` (in
`MiniMaxH3VideoVAE`: ImageNet de-normalization, clamp, and uint8 rounding when
the pipeline asks for it) and copies it into a CPU output buffer with a
blocking copy; the recorder reads its uint8 CPU frames from that buffer, so
they match the frames `MiniMaxH3VideoVAE.decode` returns and the timestamps
include the device-to-host cost. The audio VAE decode is
wrapped for timing and output capture: audio decoding is independent of video
decoding, so a streaming pipeline reorders it first and every video chunk
shifts later by its duration (serial execution makes that shift exact).

Above 480p one GPU decodes slower than playback. `set_decode_devices` (or
`activate(decode_devices=...)`) splits the chunks of each video decode
round-robin across several devices: the clips are independent, so they decode
concurrently, one thread per extra device holding its own copy of the decoder,
while the cross-fade, the conversion and the recorder callbacks stay in chunk
order on the calling thread. Only latents and overlap frames cross devices,
through the CPU, and the output is bit-identical to the single-device decode.

The pure-math helpers at the bottom model the player: `no_buffer_start` is the
oracle earliest start with zero rebuffering, `linear_estimate_start` is the
deployable policy (observe the first chunks, fit a linear completion rate, pad
it, commit to a start time) that the bench validates against actual arrivals.
"""

from __future__ import annotations

import concurrent.futures
import functools
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import wave
import weakref
from pathlib import Path

import torch

from shared.utils.phase_progress import PhaseProgress

from models.minimax_h3.components import video_autoencoder as _va


class StreamRecorder:
    """Collects chunk/audio events from one instrumented generation."""

    def __init__(self, store_frames: bool = True, live_dir: str | None = None,
                 audio_first: bool = False, fps: int = 24, sample_rate: int = 32000):
        self.store_frames = store_frames
        self.live_dir = live_dir
        self.audio_first = audio_first
        self.fps = fps
        self.sample_rate = sample_rate
        self.generation_index = 0
        self.live: "LiveSegmenter | None" = None
        self.reset()

    def reset(self) -> None:
        self.decode_start_t: float | None = None
        self.decode_end_t: float | None = None
        #: list of dicts: {chunk, frame_start, frame_end, t} (monotonic seconds)
        self.chunks: list[dict] = []
        #: uint8 CPU tensors (frames, H, W, 3), aligned with `chunks`
        self.frames: list[torch.Tensor] = []
        self.audio_start_t: float | None = None
        self.audio_end_t: float | None = None
        self.audio_out: torch.Tensor | None = None
        #: (latents, decoded) so an audio-first prefetch is never paid twice. It holds the latents
        #: tensor itself and matches by identity (`is`): an id() could be reused by a later tensor.
        self.audio_cache: tuple[object, object] | None = None
        #: True when the audio-first hook actually decoded audio before video
        self.audio_first_effective = False
        #: the last decode's multi-device plan: {"devices", "main", "fallback"}; None for a single-device decode
        self.decode_plan: dict | None = None

    # -- hooks ------------------------------------------------------------
    def on_decode_start(self) -> None:
        self.decode_start_t = time.monotonic()
        if self.live_dir:
            if self.live is not None:
                self.live.close()
            self.generation_index += 1
            self.live = LiveSegmenter(
                Path(self.live_dir) / f"gen_{self.generation_index:03d}",
                fps=self.fps, sample_rate=self.sample_rate)
            if self.audio_out is not None:
                self.live.set_audio(self.audio_out)

    def on_chunk(self, vae, decoded: torch.Tensor, frame_start: int, frame_end: int, chunk_index: int) -> None:
        if frame_end <= frame_start:
            return
        u8 = _to_uint8_frames(vae, decoded[:, :, frame_start:frame_end])  # (frames, H, W, 3)
        # `decoded` is the CPU buffer `_decode` fills with a blocking copy per
        # finalized chunk: the timestamp includes the decode of this chunk AND
        # the host copy a streaming mux would need.
        now = time.monotonic()
        meta = {"chunk": chunk_index, "frame_start": frame_start, "frame_end": frame_end, "t": now}
        self.chunks.append(meta)
        if self.store_frames:
            self.frames.append(u8)
        if self.live is not None:
            self.live.submit(meta, u8)

    def on_decode_end(self) -> None:
        self.decode_end_t = time.monotonic()

    # -- derived ----------------------------------------------------------
    @property
    def audio_s(self) -> float:
        if self.audio_start_t is None or self.audio_end_t is None:
            return 0.0
        return self.audio_end_t - self.audio_start_t

    def chunk_ready_rel(self, t0: float) -> list[float]:
        return [c["t"] - t0 for c in self.chunks]


def _to_uint8_frames(vae, sl: torch.Tensor) -> torch.Tensor:
    """A finalized slice of the `_decode` output as uint8 CPU frames (frames, H, W, 3).

    The output buffer holds whatever `_prepare_decoded_chunk` returned: uint8
    frames when the pipeline passes `uint8_rounding` (the normal video path:
    these are the exact output frames), [-1, 1] floats from the WanGP adapter
    (`MiniMaxH3VideoVAE`) otherwise, or the raw ImageNet-normalized decoder
    output when the base class's identity `_prepare_decoded_chunk` is in use.
    The result never aliases the buffer, which the caller may edit afterwards.
    """
    if sl.dtype != torch.uint8:
        # .float() is a no-op view when the buffer is already float32, so clone
        # explicitly: the in-place ops below must never touch the output buffer.
        sl = sl.detach().float().clone()
        if type(vae)._prepare_decoded_chunk is _va.AutoencoderKLMiniMaxH3._prepare_decoded_chunk:
            std = getattr(vae, "pixel_std", None)
            mean = getattr(vae, "pixel_mean", None)
            if std is not None and mean is not None:
                sl.mul_(std.to(sl)).add_(mean.to(sl))
            sl.clamp_(0.0, 1.0).mul_(255.0)
        else:
            sl.clamp_(-1.0, 1.0).add_(1.0).mul_(127.5)
        sl = sl.round_().to(torch.uint8)
    return sl[0].permute(1, 2, 3, 0).cpu().clone(memory_format=torch.contiguous_format)


_STATE: dict = {"active": False, "recorder": None, "orig_decode": None,
                "orig_audio_decode": None, "orig_video_decode": None, "decode_devices": None}


def recorder() -> StreamRecorder | None:
    return _STATE["recorder"]


def _pipeline_locals(frame, max_depth: int = 4) -> dict | None:
    """The locals of the H3 pipeline frame that called the video decode.

    The pipeline calls `self.vae.decode(...)` inside a list comprehension
    (`[self.vae.decode(video[item:item + 1], uint8_rounding) ... for item in
    ...]`). On Python 3.11, which WanGP installs, a comprehension runs in its own
    frame whose locals hold only what it references, so `audio` and
    `image_outputs` are one frame further up. Python 3.12+ inlines
    comprehensions (PEP 709). Walk up a few frames to the one holding the
    pipeline (`self` with an `audio_vae`) and the `audio` latents.
    """
    for _ in range(max_depth):
        if frame is None:
            return None
        found = frame.f_locals
        if torch.is_tensor(found.get("audio")) and getattr(found.get("self"), "audio_vae", None) is not None:
            return found
        frame = frame.f_back
    return None


def _maybe_prefetch_audio(frame, rec: StreamRecorder) -> bool:
    """Audio-first hook: decode audio before the video decode starts.

    The pipeline decodes video before audio (`self.vae.decode(video, ...)`
    then `self.audio_vae.decode(audio)`), which would leave live segments
    silent until the whole clip is decoded. When the video decode is entered,
    the pipeline frame (`_pipeline_locals`) already holds the pipeline and the
    audio latents, so decode them now; the audio wrapper caches the result by
    latents identity and the pipeline's own later call returns it without
    paying twice. Image outputs (`image_outputs`) return before any audio
    decode, so they are skipped. Every check fails closed: any mismatch with
    the expected call site just keeps the original ordering.
    """
    try:
        caller_locals = _pipeline_locals(frame)
        if caller_locals is None or caller_locals.get("image_outputs"):
            return False  # unknown call site, or an image generation that never decodes its audio latents
        audio = caller_locals["audio"]
        if rec.audio_cache is not None and rec.audio_cache[0] is audio:
            return True
        caller_locals["self"].audio_vae.decode(audio)
        return True
    except Exception:  # noqa: BLE001 - the hook must never break generation
        return False


# -- multi-device decode ----------------------------------------------------
#
# `_decode` runs one `_decode_clip` per temporal chunk on the latents
# z[:, :, i * tokens_chunk_size : i * tokens_chunk_size + tokens_chunk_size + token_overlap]
# and nothing else: no state flows from one clip to the next. What couples
# consecutive chunks is only the finalization that follows each clip: the
# first `frame_overlap` frames of chunk i are cross-faded (`_blend`) with the
# trailing `frame_overlap` frames of clip i-1 (its j == 1 part, the `overlap`),
# and the finalized frames are converted and written to the CPU output in
# order. So the clips can be decoded anywhere, concurrently, as long as the
# blend, the conversion, the CPU writes and the recorder callbacks run in chunk
# order on the calling thread. That is what `_ChunkDispatcher` does: chunk i
# decodes on decode device i % N, the calling thread decodes the chunks that
# fall on the VAE's own device itself (through the original, MMGP-managed
# module: offloading, progress and abort hooks unchanged) and finalizes every
# chunk in order exactly as the single-device loop does.
#
# Each other device gets a `_Replica`: a copy of the decode path only
# (`post_quant_conv` and `decoder`, plus the root's plain attributes) whose
# weights are copied once to that device and kept there. It is built from the
# live VAE module rather than reloaded from disk, so it has the same
# quantization, qkv splitting, dtype and patched forwards whatever loader and
# MMGP profile produced it. MMGP's per-module offload hooks are stripped (they
# would load the original weights onto the main GPU); see `_replicate_module`.

_DECODE_LOOKAHEAD = 2  # chunks queued per replica device ahead of the in-order finalization
_REPLICAS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()  # vae -> {(device, ordinal): _Replica}
_REPLICA_LOCK = threading.Lock()


class ReplicaUnsupported(RuntimeError):
    """The VAE carries a hook or wrapper that `_replicate_module` cannot copy faithfully."""


def _normalize_device(device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device() if torch.cuda.is_available() else 0)
    return device


def parse_decode_devices(devices) -> list[str] | None:
    """A decode device list from a list of names or a comma/space separated string; None when empty.

    Every entry must name a torch device ("cuda:1", "cuda", "cpu"); "cuda" means the current CUDA device.
    Entries may repeat (two slots on one device decode two chunks at once there, which the CPU tests use).
    """
    if devices is None:
        return None
    if isinstance(devices, str):
        devices = devices.replace(",", " ").split()
    names = [str(_normalize_device(str(d).strip())) for d in devices if str(d).strip()]
    return names or None


def set_decode_devices(devices) -> list[str] | None:
    """Set the multi-device decode list (None/empty: off, the single-device decode of today)."""
    names = parse_decode_devices(devices)
    if names != _STATE.get("decode_devices"):
        release_decode_replicas()
    _STATE["decode_devices"] = names
    return names


def decode_devices() -> list[str] | None:
    return _STATE.get("decode_devices")


def release_decode_replicas() -> None:
    """Drop every cached replica (their weights free on their devices once no decode is using them)."""
    with _REPLICA_LOCK:
        for cache in list(_REPLICAS.values()):
            for replica in cache.values():
                if isinstance(replica, _ProcessReplica):
                    replica.close()
        _REPLICAS.clear()


def _main_decode_device(z: torch.Tensor) -> torch.device:
    """The device the original VAE decodes on: the latents' device (MMGP moves them to the GPU before `_decode`)."""
    if z.device.type != "cpu":
        return _normalize_device(z.device)
    if torch.cuda.is_available():
        return _normalize_device("cuda")
    return torch.device("cpu")


def _move_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """`tensor` on `device`; between two GPUs through the CPU, so no peer-to-peer access (NVLink, P2P) is ever needed."""
    tensor = tensor.detach()
    if tensor.device == device:
        return tensor
    if tensor.device.type != "cpu" and device.type != "cpu":
        tensor = tensor.to("cpu")
    return tensor.to(device)


def _rebind(value, memo: dict):
    """A method bound to a module of the copied tree, bound to its copy instead; anything else unchanged."""
    owner = getattr(value, "__self__", None)
    if isinstance(value, types.MethodType) and id(owner) in memo:
        return types.MethodType(value.__func__, memo[id(owner)])
    return value


def _replicate_value(value, memo: dict, device: torch.device, where: str):
    """An attribute of a copied module, for its copy: tensors moved to `device`, modules and their bound methods replaced by their
    copies, through lists, tuples and dicts. A module outside the copied tree is refused: sharing it would run its weights, on their
    own device, against the replica's inputs (mmgp's row views keep their fused module in `__dict__`, not as a submodule)."""
    if torch.is_tensor(value):
        return _move_tensor(value, device)
    if isinstance(value, torch.nn.Module):
        if id(value) not in memo:
            raise ReplicaUnsupported(f"{where} refers to a {type(value).__name__} outside the copied modules")
        return memo[id(value)]
    if type(value) in (list, tuple):
        return type(value)(_replicate_value(item, memo, device, where) for item in value)
    if type(value) is dict:
        return {key: _replicate_value(item, memo, device, where) for key, item in value.items()}
    return _rebind(value, memo)


_HOOK_DICTS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks",
               "_state_dict_hooks", "_state_dict_pre_hooks", "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks")


def _replicate_module(module: torch.nn.Module, device: torch.device, keep_children=None, memo: dict | None = None) -> torch.nn.Module:
    """A copy of `module` with its parameters and buffers on `device`, without MMGP's offload hooks.

    MMGP hooks a module by replacing its instance `forward` with a partial that first makes the model's weights resident on the main
    GPU, keeping the previous forward in `_mm_forward` (`_mm_<method>` for other methods). The copy runs that previous forward:
    the class forward, or a forward the loader installed on the instance (an INT8 precision rule, a ConvRot embedding), which is
    rebound to the copy, as are other methods bound to the module. MMGP's attributes (`_mm_*`, `_hf_hook`) are dropped. A forward or a
    module hook of any other kind is refused (`ReplicaUnsupported`) rather than guessed at. Parameters and buffers are copied with
    `.to(device)` (quantized weight tensors move with their scales); on their own device they are shared, not copied.
    """
    memo = {} if memo is None else memo
    # first pass: one empty copy per module, so methods bound to any module of the tree can be rebound
    originals = []

    def allocate(mod, keep):
        clone = type(mod).__new__(type(mod))
        memo[id(mod)] = clone
        originals.append(mod)
        for name, child in mod._modules.items():
            if child is not None and (keep is None or name in keep):
                allocate(child, None)

    allocate(module, keep_children)
    for mod in originals:
        clone = memo[id(mod)]
        if getattr(mod, "_compiled_call_impl", None) is not None:
            raise ReplicaUnsupported(f"{type(mod).__name__} is torch.compile'd")
        state = {}
        for key, value in mod.__dict__.items():
            if key.startswith("_mm_") or key in ("_hf_hook", "forward", "_parameters", "_buffers", "_modules") or key in _HOOK_DICTS:
                continue
            state[key] = _replicate_value(value, memo, device, f"{type(mod).__name__}.{key}")
        forward = mod.__dict__.get("_mm_forward", mod.__dict__.get("forward"))
        if "_mm_lora_old_forward" in mod.__dict__:
            raise ReplicaUnsupported(f"{type(mod).__name__} carries LoRA hooks")
        if forward is not None:
            if not (isinstance(forward, types.MethodType) and forward.__self__ is mod):
                raise ReplicaUnsupported(f"{type(mod).__name__}.forward is wrapped by {type(forward).__name__}")
            if forward.__func__ is not type(mod).forward:
                state["forward"] = types.MethodType(forward.__func__, clone)
        for key in _HOOK_DICTS:
            hooks = mod.__dict__.get(key)
            if hooks is None:
                continue
            copied = type(hooks)()
            for hook_id, hook in hooks.items():
                rebound = _rebind(hook, memo)
                if rebound is hook:
                    raise ReplicaUnsupported(f"{type(mod).__name__} has a {key[1:]} entry that is not a method of the copied modules")
                copied[hook_id] = rebound
            state[key] = copied
        params = {}
        for name, param in mod._parameters.items():
            if param is None:
                params[name] = None
                continue
            moved = _move_tensor(param, device)
            params[name] = torch.nn.Parameter(moved, requires_grad=False) if isinstance(param, torch.nn.Parameter) else moved
        state["_parameters"] = params
        state["_buffers"] = {name: None if buf is None else _move_tensor(buf, device) for name, buf in mod._buffers.items()}
        state["_modules"] = {name: memo[id(child)] for name, child in mod._modules.items() if child is not None and id(child) in memo}
        clone.__dict__.update(state)
    return memo[id(module)]


_ROOT_DECODE_MODULES = ("post_quant_conv", "decoder")


class _Replica:
    """The decode path of one VAE on one extra device, with the CUDA stream its decodes run on."""

    def __init__(self, vae, device: torch.device):
        self.device = device
        # Built with inference mode off, as mmgp makes its Parameters (`_make_parameter`): the decode runs inside inference mode,
        # and detaching an mmgp-loaded INT8 (quanto) weight there yields an inference tensor that cannot share the weight's
        # version counter ("Cannot set version_counter for inference tensor": every decode fell back to one GPU on 2x RTX PRO
        # 6000). The copy is ordinary tensors, which the replica then decodes with under inference mode.
        with torch.inference_mode(False):
            self.module = _replicate_module(vae, device, keep_children=_ROOT_DECODE_MODULES)
        missing = [name for name in _ROOT_DECODE_MODULES if name not in self.module._modules]
        if missing:
            raise ReplicaUnsupported(f"the VAE has no {', '.join(missing)}")
        self.stream = torch.cuda.Stream(device) if device.type == "cuda" else None

    def sync_from(self, vae) -> None:
        """Mirror the root's plain settings (tiling, tile sizes) the UI may have changed since the copy, and clear a past abort."""
        for key, value in vae.__dict__.items():
            if (key.startswith("_mm_") or key in ("_hf_hook", "forward", "_parameters", "_buffers", "_modules")
                    or key in _HOOK_DICTS or torch.is_tensor(value) or isinstance(value, (types.MethodType, torch.nn.Module))):
                continue
            self.module.__dict__[key] = value
        self.set_interrupt(False)

    def set_interrupt(self, value: bool) -> None:
        decoder = self.module._modules["decoder"]
        decoder.__dict__["_interrupt"] = bool(value)

    def decode(self, z_part: torch.Tensor, grad_enabled: bool, inference: bool) -> torch.Tensor:
        """`_decode_clip` of one chunk on this device, on a worker thread; returns once the clip is computed."""
        mode = torch.inference_mode() if inference else torch.set_grad_enabled(grad_enabled)
        if self.stream is None:
            with mode:
                return self.module._decode_clip(z_part.to(self.device))
        torch.cuda.set_device(self.device)
        with mode, torch.cuda.stream(self.stream):
            clip = self.module._decode_clip(z_part.to(self.device))
            self.stream.synchronize()
        return clip


_RECIPE_ATTR = "_decode_worker_recipe"  # set on a VAE by the loader hook: how a decode worker loads the same VAE
_WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decode_worker.py")
_WORKER_BOOT_TIMEOUT_S = 900
_INHERITED_FLAGS = {  # the WanGP flags a decode worker shares with this process (the USP helper's set)
    "--config", "--loras", "--lora-config", "--profile", "--attention", "--compile", "--fp16", "--bf16",
    "--vae-config", "--perc-reserved-mem-max", "--vram-safety-coefficient", "--max-vram", "--max-ram",
    "--preload", "--offline", "--convrot", "--ram-allocator", "--vram-allocator", "--settings", "--verbose",
}


def install_recipe_hook() -> bool:
    """Record, on every H3 video VAE WanGP loads from now on, the loader call that rebuilds it (`_RECIPE_ATTR`), so a decode
    worker can load the same weights with the same quantization. False when the H3 loader cannot be imported."""
    if _STATE.get("recipe_hook") is not None:
        return _STATE["recipe_hook"]
    try:
        from models.minimax_h3 import minimax_h3_main as h3_main
    except Exception:  # noqa: BLE001 - outside WanGP (or its H3 loader cannot import here): in-process copies only
        _STATE["recipe_hook"] = False
        return False
    loader = h3_main._load_video_vae
    if getattr(loader, "_records_recipe", False):
        return True

    @functools.wraps(loader)
    def load_video_vae(*args, **kwargs):
        vae = loader(*args, **kwargs)
        vae.__dict__[_RECIPE_ATTR] = {"factory": "models.minimax_h3.minimax_h3_main:_load_video_vae", "args": list(args),
                                      "kwargs": dict(kwargs), "wangp": True}
        return vae

    load_video_vae._records_recipe = True
    h3_main._load_video_vae = load_video_vae
    _STATE["recipe_hook"] = True
    return True


def _inherited_cli_args(argv: list[str]) -> list[str]:
    out, index = [], 0
    while index < len(argv):
        token = argv[index]
        if token.split("=", 1)[0] in _INHERITED_FLAGS:
            out.append(token)
            if "=" not in token and index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                out.append(argv[index + 1])
                index += 1
        index += 1
    return out


def _worker_settings() -> dict:
    """What a decode worker's WanGP session shares with this one: folder, command-line flags and the INT8 / FP8 kernel choices."""
    wgp = sys.modules.get("wgp")
    root = os.path.dirname(os.path.abspath(wgp.__file__)) if getattr(wgp, "__file__", None) else os.getcwd()
    return {"root": root, "cli_args": _inherited_cli_args(sys.argv[1:]),
            "int8_kernels": getattr(wgp, "int8_kernels", None), "fp8_kernels": getattr(wgp, "fp8_kernels", None)}


def _physical_gpu(index: int) -> str:
    """The CUDA_VISIBLE_DEVICES entry of this process's GPU `index`."""
    visible = [entry.strip() for entry in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if entry.strip()]
    return visible[index] if visible else str(index)


def _plain_settings(vae) -> dict:
    """The root VAE's plain settings (tiling, tile sizes), which the UI may change between decodes."""
    plain = (bool, int, float, str, type(None))
    return {key: value for key, value in vae.__dict__.items()
            if not (key.startswith("_mm_") or key.startswith("_") and key not in ("_model_dtype",))
            and (isinstance(value, plain) or isinstance(value, tuple) and all(isinstance(item, plain) for item in value))}


class _ProcessReplica:
    """An extra decode device served by a decode worker process (decode_worker.py): one GPU and one thread per process for every
    kernel, so per-process kernel state (WGPKernels' INT8 GEMM setup and buffers) never meets a second GPU or a second thread."""

    def __init__(self, recipe: dict, device: torch.device, main: torch.device):
        from multiprocessing.connection import Listener

        self.device, self.main = device, main
        self.attrs, self.lock, self.proc, self.conn = {}, threading.Lock(), None, None
        settings = _worker_settings()
        authkey = os.urandom(16)
        self.listener = Listener(("127.0.0.1", 0), authkey=authkey)
        host, port = self.listener.address
        env = dict(os.environ, DECODE_WORKER_AUTHKEY=authkey.hex(), PYTHONUNBUFFERED="1")
        if device.type == "cuda":
            env["CUDA_VISIBLE_DEVICES"], worker_device = _physical_gpu(device.index), "cuda:0"
        else:
            env["CUDA_VISIBLE_DEVICES"], worker_device = "", "cpu"
        command = [sys.executable, _WORKER_PATH, "--address", f"{host}:{port}", "--device", worker_device]
        if recipe.get("wangp"):
            command += ["--root", settings["root"]]
        self.log_path = os.path.join(tempfile.gettempdir(), f"decode_worker_{os.getpid()}_{str(device).replace(':', '')}.log")
        self.log = open(self.log_path, "w")
        try:
            self.proc = subprocess.Popen(command, env=env, stdout=self.log, stderr=subprocess.STDOUT)
            self.conn = self._accept()
            self.conn.send(("load", recipe, settings))
            reply = self._recv(_WORKER_BOOT_TIMEOUT_S)
            if reply[0] != "ready":
                raise ReplicaUnsupported(f"decode worker on {device} could not load the VAE: {reply[1]}")
            self.info = reply[1]
        except BaseException:
            self.close()
            raise

    def _alive_or_raise(self):
        if self.proc is not None and self.proc.poll() is not None:
            raise ReplicaUnsupported(f"decode worker on {self.device} exited ({self.proc.returncode}): {self._tail()}")

    def _tail(self, lines: int = 12) -> str:
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as handle:
                return "".join(handle.readlines()[-lines:])[-800:]
        except OSError:
            return ""

    def _accept(self):
        result = {}
        thread = threading.Thread(target=lambda: result.setdefault("conn", self.listener.accept()), daemon=True)
        thread.start()
        deadline = time.monotonic() + _WORKER_BOOT_TIMEOUT_S
        while thread.is_alive() and time.monotonic() < deadline:
            thread.join(0.5)
            self._alive_or_raise()
        if "conn" not in result:
            raise ReplicaUnsupported(f"decode worker on {self.device} did not connect")
        return result["conn"]

    def _recv(self, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        while not self.conn.poll(0.5):
            self._alive_or_raise()
            if time.monotonic() > deadline:
                raise ReplicaUnsupported(f"decode worker on {self.device} did not answer in {timeout_s:.0f} s")
        return self.conn.recv()

    def sync_from(self, vae) -> None:
        self.attrs = _plain_settings(vae)

    def set_interrupt(self, value: bool) -> None:
        # a clip in flight finishes in the worker (one chunk); the dispatcher drops it
        pass

    def decode(self, z_part: torch.Tensor, grad_enabled: bool, inference: bool) -> torch.Tensor:
        from . import decode_worker as worker

        meta, buffer = worker.pack(z_part)
        with self.lock:
            self.conn.send(("decode", meta, self.attrs, grad_enabled, inference))
            self.conn.send_bytes(buffer)
            reply = self._recv(_WORKER_BOOT_TIMEOUT_S)
            if reply[0] != "clip":
                raise RuntimeError(f"decode worker on {self.device}: {reply[1]}\n{reply[2] if len(reply) > 2 else ''}")
            clip = worker.unpack(reply[1], self.conn.recv_bytes())
        return clip.to(self.main)

    def close(self) -> None:
        try:
            if self.conn is not None:
                self.conn.send(("exit",))
        except (OSError, EOFError, ValueError):
            pass
        if self.proc is not None:
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        for closable in (self.conn, self.listener, self.log):
            try:
                if closable is not None:
                    closable.close()
            except OSError:
                pass
        self.proc = self.conn = None

    def __del__(self):
        self.close()


def _replicas_for(vae, slots: list[torch.device], main: torch.device | None = None) -> list:
    """One replica per extra decode slot, built once per VAE and slot and cached while the device list is unchanged: a decode
    worker process when the VAE carries a loader recipe (WanGP's H3 VAEs, `install_recipe_hook`), else an in-process copy."""
    with _REPLICA_LOCK:
        cache = _REPLICAS.get(vae)
        if cache is None:
            cache = _REPLICAS[vae] = {}
        replicas, ordinal = [], {}
        for device in slots:
            key = (str(device), ordinal.setdefault(str(device), 0))
            ordinal[str(device)] += 1
            replica = cache.get(key)
            if replica is None:
                if device.type == "cuda":
                    torch.cuda.synchronize()  # no MMGP prefetch of these weights is still in flight while they are read
                recipe = vae.__dict__.get(_RECIPE_ATTR)
                if recipe is not None:
                    replica = cache[key] = _ProcessReplica(recipe, device, main if main is not None else _normalize_device("cuda"))
                else:
                    replica = cache[key] = _Replica(vae, device)
            replica.sync_from(vae)
            replicas.append(replica)
        return replicas


class _ChunkDispatcher:
    """Decodes the clips of `_decode`'s chunk loop across devices; `clip(i)` is called in chunk order by the calling thread.

    Slot k of the device list takes chunks k, k + N, k + 2N... The first slot on the VAE's own device is decoded by the calling
    thread with the original module; every other slot has a replica and a worker thread. At most `_DECODE_LOOKAHEAD` chunks per
    replica are queued ahead of the chunk being finalized, which bounds the decoded clips waiting on each device.
    """

    def __init__(self, vae, z: torch.Tensor, slots: list, num_chunks: int, tokens_chunk_size: int, clip_tokens: int):
        self.vae = vae
        self.z = z
        self.slots = slots  # None: the calling thread with the original module; else a _Replica
        self.num_chunks = num_chunks
        self.tokens_chunk_size = tokens_chunk_size
        self.clip_tokens = clip_tokens
        self.progress = None
        self.spatial_tiles = 1
        # only the latents leave the main device (a few MB), through the CPU
        self.z_host = z.detach().to("cpu")
        self.grad_enabled = torch.is_grad_enabled()
        self.inference = torch.is_inference_mode_enabled()
        self.executors = {}
        for k, slot in enumerate(slots):
            if slot is not None:
                self.executors[id(slot)] = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix=f"h3-decode-slot{k}-{slot.device}")
        self.futures: dict[int, concurrent.futures.Future] = {}
        self.next_submit = 0

    def _submit_until(self, limit: int) -> None:
        while self.next_submit < min(limit, self.num_chunks):
            i = self.next_submit
            slot = self.slots[i % len(self.slots)]
            if slot is not None:
                start = i * self.tokens_chunk_size
                z_part = self.z_host[:, :, start : start + self.clip_tokens]
                self.futures[i] = self.executors[id(slot)].submit(slot.decode, z_part, self.grad_enabled, self.inference)
            self.next_submit += 1

    def _interrupt_requested(self) -> bool:
        return bool(getattr(self.vae.decoder, "_interrupt", False))

    def clip(self, i: int) -> torch.Tensor:
        self._submit_until(i + 1 + _DECODE_LOOKAHEAD * len(self.slots))
        slot = self.slots[i % len(self.slots)]
        if slot is None:
            start = i * self.tokens_chunk_size
            return self.vae._decode_clip(self.z[:, :, start : start + self.clip_tokens])
        future = self.futures.pop(i)
        while True:
            try:
                clip = future.result(timeout=0.05)
                break
            except concurrent.futures.TimeoutError:
                if self._interrupt_requested():
                    for other in self.slots:
                        if other is not None:
                            other.set_interrupt(True)
        if clip.device.type == "cuda":
            # allocated on the replica's stream, finalized on this thread's: its memory must outlive this stream's reads
            clip.record_stream(torch.cuda.current_stream(clip.device))
        if self.progress is not None:
            self.progress.advance(self.spatial_tiles)  # the replica's decoder carries no progress hook
        return clip

    def close(self, failed: bool) -> None:
        """Stop and join the workers. On failure, abort in-flight replica decodes at their next block first."""
        for future in self.futures.values():
            future.cancel()
        if failed:
            for slot in self.slots:
                if slot is not None:
                    slot.set_interrupt(True)
        for executor in self.executors.values():
            executor.shutdown(wait=True)
        for slot in self.slots:
            if slot is not None:
                slot.set_interrupt(False)
        self.futures.clear()


def _wgp_int8_active() -> bool:
    """Whether WanGP runs its INT8 layers with WGPKernels (False outside WanGP or when the backend module is unavailable)."""
    try:
        from shared.kernels import int8_backend
    except Exception:  # noqa: BLE001 - outside WanGP, or its kernels cannot load: no WGPKernels INT8 in use
        return False
    return bool(int8_backend.wgp_enabled())


def _single_device_reason(vae) -> str | None:
    """Why this VAE's decode must stay on one GPU in this process, or None.

    WGPKernels 1.4.3 INT8 cannot serve a split decode: its GEMM sets the kernel's shared-memory attribute once per process, so the
    second GPU fails ("INT8 GEMM launch failed (CUDA error 1)", confirmed on 2x RTX PRO 6000), and its Python wrapper shares one
    workspace-size buffer and one autotuning buffer between threads (with that fixed, the split decode hit an illegal memory access).
    """
    decoder = vae._modules.get("decoder")
    if decoder is None or not _wgp_int8_active():
        return None
    if any(getattr(param, "_qtype", None) is not None for param in decoder.parameters()):
        return "WGPKernels INT8 decodes on one GPU per process: the INT8 VAE stays on its own GPU"
    return None


def _make_dispatcher(vae, z: torch.Tensor, num_chunks: int, tokens_chunk_size: int, clip_tokens: int,
                     rec: "StreamRecorder") -> "_ChunkDispatcher | None":
    """The multi-device dispatcher for this decode, or None for the single-device loop (option off, or nothing to split)."""
    names = _STATE.get("decode_devices")
    rec.decode_plan = None
    if not names or num_chunks < 2:
        return None
    main = _main_decode_device(z)
    devices = [_normalize_device(name) for name in names]
    if len(devices) == 1 and devices[0] == main:
        return None
    reason = None if _RECIPE_ATTR in vae.__dict__ else _single_device_reason(vae)
    if reason is not None:
        rec.decode_plan = {"devices": [str(main)], "fallback": reason}
        if not _STATE.get("single_device_noted"):
            _STATE["single_device_noted"] = True
            print(f"[Chunk Streaming] multi-device decode off: {reason}")
        return None
    main_slot = next((k for k, device in enumerate(devices) if device == main), None)
    extra = [device for k, device in enumerate(devices) if k != main_slot]
    try:
        replicas = iter(_replicas_for(vae, extra, main))
    except Exception as exc:  # noqa: BLE001 - fall back to the single-device decode, which is always correct
        where = " <- ".join(f"{frame.name}@{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno}"
                            for frame in reversed(traceback.extract_tb(exc.__traceback__)[-4:]))
        rec.decode_plan = {"devices": [str(main)], "fallback": f"{type(exc).__name__}: {exc}"[:300], "where": where[:400]}
        print(f"[Chunk Streaming] multi-device decode unavailable, decoding on {main} only: {exc}")
        return None
    slots = [None if k == main_slot else next(replicas) for k in range(len(devices))]
    rec.decode_plan = {"devices": [str(device) for device in devices], "main": str(main), "fallback": None}
    return _ChunkDispatcher(vae, z, slots, num_chunks, tokens_chunk_size, clip_tokens)


def _instrumented_decode(self, z: torch.Tensor, uint8_rounding: str | None = None) -> torch.Tensor:
    """`AutoencoderKLMiniMaxH3._decode` with recorder callbacks.

    The body is copied verbatim from components/video_autoencoder.py (upstream
    9f30419: single-frame decoder and single-latent still paths, CPU output
    buffer filled by `write_chunk`, `PhaseProgress`), its chunk loop moved into
    `_instrumented_decode_loop`; the callbacks only read finalized regions
    of that buffer, so the returned tensor is bit-identical to the original
    implementation's. With decode devices set (`set_decode_devices`), the
    clips of the chunk loop are decoded across them (`_ChunkDispatcher`) and
    still finalized here in chunk order, so the output is unchanged.
    """
    rec: StreamRecorder = _STATE["recorder"]
    rec.on_decode_start()

    tokens_chunk_size = self.tokens_chunk_size
    token_drop = self.config.token_drop
    temporal_ratio = self.temporal_compression_ratio
    chunk_num_frames = tokens_chunk_size * temporal_ratio
    frame_pre_padding = self.frame_pre_padding

    if self.decoder.patch_size_t == 1:
        # A single-frame decoder: one latent frame, one pixel frame.
        num_chunks, output_frames, frame_pre_padding = 0, 1, 0
    elif z.shape[2] == 1:
        # Fizgig H3 Still's decode recipe: five identical latents, then keep pixel frame 3.
        # https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still (MIT)
        z = z.repeat(1, 1, tokens_chunk_size, 1, 1)
        num_chunks, output_frames = 0, 1
    else:
        num_tokens = z.shape[2] + token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

        intra_tail = self.config.clip_length % temporal_ratio
        num_tokens_before_pad = z.shape[2] - pad_tokens
        pad_frames = sum(
            intra_tail if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0 else temporal_ratio
            for k in range(pad_tokens)
        )
        output_frames = num_chunks * (chunk_num_frames - self.frame_pre_padding) + self.frame_overlap - pad_frames
    spatial_tiles = 1
    if self.use_tiling:
        height, width = z.shape[-2] * self.spatial_compression_ratio, z.shape[-1] * self.spatial_compression_ratio
        rows = self._split_tiles(height, self.tile_sample_min_height, self.tile_sample_min_overlap_height)[0]
        cols = self._split_tiles(width, self.tile_sample_min_width, self.tile_sample_min_overlap_width)[0]
        spatial_tiles = len(rows) * len(cols)
    # multi-device decode (`set_decode_devices`): None keeps the single-device loop below exactly as upstream. Taken before the
    # progress hook is registered on the decoder, which the replicas must not copy.
    dispatcher = _make_dispatcher(self, z, num_chunks, tokens_chunk_size, tokens_chunk_size + self.token_overlap, rec)
    if dispatcher is None:
        return _instrumented_decode_loop(self, z, uint8_rounding, rec, None, num_chunks, output_frames, frame_pre_padding,
                                         spatial_tiles)
    failed = True
    try:
        decoded = _instrumented_decode_loop(self, z, uint8_rounding, rec, dispatcher, num_chunks, output_frames, frame_pre_padding,
                                            spatial_tiles)
        failed = False
        return decoded
    finally:
        dispatcher.close(failed)


def _instrumented_decode_loop(self, z, uint8_rounding, rec, dispatcher, num_chunks, output_frames, frame_pre_padding,
                              spatial_tiles):
    """The chunk loop of `_instrumented_decode`; with a dispatcher, each clip comes from `dispatcher.clip(i)`."""
    tokens_chunk_size = self.tokens_chunk_size
    token_drop = self.config.token_drop
    chunk_num_frames = tokens_chunk_size * self.temporal_compression_ratio
    with PhaseProgress(max(1, num_chunks) * spatial_tiles) as progress, progress.track(self.decoder, spatial_tiles):
        if dispatcher is not None:
            dispatcher.progress, dispatcher.spatial_tiles = progress, spatial_tiles
        decoded = None
        write_position = 0
        overlap = None
        reported = 0  # end of the frame range already handed to the recorder

        def write_chunk(chunk):
            # finalized pixels converted and copied to the CPU output by groups of frames (no full-clip conversion copies)
            nonlocal decoded, write_position
            copy_frames = min(chunk.shape[2], output_frames - write_position)
            for start in range(0, copy_frames, 4):
                part = self._prepare_decoded_chunk(chunk[:, :, start : min(start + 4, copy_frames)], uint8_rounding)
                if decoded is None:
                    decoded = torch.empty(*part.shape[:2], output_frames, *part.shape[3:], dtype=part.dtype, device="cpu")
                _va.transfer.copy_(decoded[:, :, write_position : write_position + part.shape[2]], part)  # mmgp.transfer, as upstream
                write_position += part.shape[2]
                del part

        if num_chunks == 0:
            # Short videos still need one decode, without an overlapping temporal chunk.
            write_chunk(self._decode_clip(z)[:, :, frame_pre_padding : frame_pre_padding + output_frames])
            rec.on_chunk(self, decoded, 0, write_position, 0)
            rec.on_decode_end()
            return decoded

        for i in range(num_chunks):
            if dispatcher is None:
                start = i * tokens_chunk_size
                clip = self._decode_clip(z[:, :, start : start + tokens_chunk_size + self.token_overlap])
            else:
                clip = dispatcher.clip(i)  # decoded on decode device i % N; finalized here, in chunk order
            for j in range(int(token_drop > 0) + 1):
                frame_start = j * chunk_num_frames
                chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
                chunk = chunk[:, :, self.frame_pre_padding :]
                if j == 0:
                    if overlap is not None:
                        if overlap.device != chunk.device:  # the previous clip was decoded on another device
                            overlap = _move_tensor(overlap, chunk.device)
                        chunk = self._blend(overlap, chunk, self.frame_overlap, dim=-3)
                    write_chunk(chunk)
                    if i < num_chunks - 1 and write_position > reported:
                        rec.on_chunk(self, decoded, reported, write_position, i)
                        reported = write_position
                else:
                    overlap = chunk.contiguous()
            del chunk, clip
        if overlap is not None:
            write_chunk(overlap)
        if write_position != output_frames:
            raise RuntimeError(f"MiniMax H3 VAE decoded {write_position} frames, expected {output_frames}")
        # The last chunk's frames plus the trailing overlap finalize together.
        rec.on_chunk(self, decoded, reported, write_position, num_chunks - 1)
        rec.on_decode_end()
        return decoded


def activate(store_frames: bool = True, live_dir: str | None = None,
             audio_first: bool = False, decode_devices=None) -> StreamRecorder:
    """Install the instrumented decode paths; returns the shared recorder.

    `live_dir` turns on the live segmenter: every finalized chunk is muxed to
    a standalone fMP4 segment in a background thread as it lands, under
    `live_dir/gen_NNN/`. `audio_first` additionally decodes the audio before
    the video decode starts (see `_maybe_prefetch_audio`), so live segments
    carry their audio slice from the first chunk onward. `decode_devices`
    (e.g. ["cuda:0", "cuda:1"] or "cuda:0,cuda:1") splits each video decode's
    chunks round-robin across those devices (see `set_decode_devices`);
    None keeps the single-device decode.
    """
    names = parse_decode_devices(decode_devices)  # validated before anything is patched
    if names:
        install_recipe_hook()  # VAEs loaded from now on can be decoded by worker processes
    if _STATE["active"]:
        rec = _STATE["recorder"]
        rec.store_frames = store_frames
        rec.live_dir = live_dir
        rec.audio_first = audio_first
        set_decode_devices(names)
        return rec
    from models.minimax_h3 import audio_vae as _audio_mod
    from models.minimax_h3 import video_vae as _video_mod

    rec = StreamRecorder(store_frames=store_frames, live_dir=live_dir, audio_first=audio_first)
    _STATE["recorder"] = rec
    set_decode_devices(names)
    _STATE["orig_decode"] = _va.AutoencoderKLMiniMaxH3._decode
    _va.AutoencoderKLMiniMaxH3._decode = _instrumented_decode

    orig_audio_decode = _audio_mod.MiniMaxH3AudioVAE.decode

    def timed_audio_decode(self, latents):
        if rec.audio_cache is not None and rec.audio_cache[0] is latents:
            return rec.audio_cache[1]
        rec.audio_start_t = time.monotonic()
        out = orig_audio_decode(self, latents)
        rec.audio_end_t = time.monotonic()
        rec.audio_cache = (latents, out)
        rec.audio_out = out.detach().float().cpu() if torch.is_tensor(out) else None
        if rec.live is not None and rec.audio_out is not None:
            rec.live.set_audio(rec.audio_out)
        return out

    _STATE["orig_audio_decode"] = orig_audio_decode
    _audio_mod.MiniMaxH3AudioVAE.decode = timed_audio_decode

    orig_video_decode = _video_mod.MiniMaxH3VideoVAE.decode

    def audio_first_video_decode(self, latents, *args, **kwargs):
        # The pipeline calls `self.vae.decode(video, uint8_rounding)`: forward
        # every extra argument untouched.
        # A new video decode starts a new generation's segments: drop the previous generation's audio, or
        # its segments would carry it (the UI never calls reset() between generations).
        rec.audio_out = rec.audio_cache = rec.audio_start_t = rec.audio_end_t = None
        rec.audio_first_effective = False
        if rec.audio_first:
            rec.audio_first_effective = _maybe_prefetch_audio(sys._getframe(1), rec)
        return orig_video_decode(self, latents, *args, **kwargs)

    _STATE["orig_video_decode"] = orig_video_decode
    _video_mod.MiniMaxH3VideoVAE.decode = audio_first_video_decode
    _STATE["active"] = True
    return rec


def deactivate() -> None:
    if not _STATE["active"]:
        return
    from models.minimax_h3 import audio_vae as _audio_mod
    from models.minimax_h3 import video_vae as _video_mod

    rec = _STATE["recorder"]
    if rec is not None and rec.live is not None:
        rec.live.close()
    _va.AutoencoderKLMiniMaxH3._decode = _STATE["orig_decode"]
    _audio_mod.MiniMaxH3AudioVAE.decode = _STATE["orig_audio_decode"]
    _video_mod.MiniMaxH3VideoVAE.decode = _STATE["orig_video_decode"]
    _STATE.update({"active": False, "recorder": None, "orig_decode": None,
                   "orig_audio_decode": None, "orig_video_decode": None})
    set_decode_devices(None)


# -- segment muxing -------------------------------------------------------

class LiveSegmenter:
    """Muxes finalized chunks into fMP4 segments as they land, off the GPU path.

    One instance per generation. `submit` is called from the decode hook with
    each chunk's metadata and uint8 CPU frames; a background thread encodes
    the segment (with its audio slice when `set_audio` has been called, which
    the audio-first hook guarantees happens before the first chunk) and stamps
    the wall-clock time the finished file hit disk. `close` drains the queue
    and returns the per-segment records.
    """

    def __init__(self, out_dir: str | Path, fps: int = 24, sample_rate: int = 32000):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.sample_rate = sample_rate
        self.audio: torch.Tensor | None = None
        self.results: list[dict] = []
        self._queue: queue.Queue = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_audio(self, audio: torch.Tensor) -> None:
        if audio is not None and audio.dim() == 3:
            audio = audio[0]
        self.audio = audio

    def submit(self, meta: dict, frames_u8: torch.Tensor) -> None:
        if not self._closed:
            self._queue.put((dict(meta), frames_u8))

    def _run(self) -> None:
        index = 0
        while True:
            item = self._queue.get()
            if item is None:
                return
            meta, frames = item
            seg_path = self.out_dir / f"segment_{index:03d}.mp4"
            record = {"segment": index, "frame_start": meta["frame_start"],
                      "frame_end": meta["frame_end"],
                      "duration_s": (meta["frame_end"] - meta["frame_start"]) / self.fps,
                      "chunk_t": meta["t"], "has_audio": False}
            try:
                wav = None
                if self.audio is not None:
                    s0 = int(round(meta["frame_start"] / self.fps * self.sample_rate))
                    s1 = min(int(round(meta["frame_end"] / self.fps * self.sample_rate)),
                             self.audio.shape[-1])
                    if s1 > s0:
                        wav = str(seg_path.with_suffix(".wav"))
                        write_wav(wav, self.audio[:, s0:s1], self.sample_rate)
                        record["has_audio"] = True
                record["mux_s"] = round(mux_segment(frames, wav, seg_path, fps=self.fps), 4)
                if wav:
                    Path(wav).unlink(missing_ok=True)
                record["path"] = str(seg_path)
            except Exception as exc:  # noqa: BLE001 - a bad segment must not kill the run
                record["error"] = repr(exc)[:300]
            record["t_done"] = time.monotonic()
            self.results.append(record)
            index += 1

    def close(self) -> list[dict]:
        if not self._closed:
            self._closed = True
            self._queue.put(None)
        self._thread.join(timeout=120)
        return self.results


def write_wav(path: str | Path, samples, sample_rate: int) -> None:
    """Write float32 stereo samples shaped (2, n) or (n, 2) as 16-bit WAV."""
    if torch.is_tensor(samples):
        samples = samples.detach().float().cpu()
        if samples.dim() == 2 and samples.shape[0] == 2:
            samples = samples.transpose(0, 1)
        samples = samples.numpy()
    import numpy as np

    arr = np.asarray(samples)
    if arr.ndim == 1:
        arr = arr[:, None].repeat(2, axis=1)
    if arr.shape[0] == 2 and arr.shape[1] != 2:
        arr = arr.T
    pcm = (arr.clip(-1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def mux_segment(frames_u8: torch.Tensor, audio_wav: str | None, out_path: str | Path, fps: int = 24) -> float:
    """Encode one finished chunk as a standalone fragmented MP4 segment.

    `frames_u8` is (frames, H, W, 3) uint8. Returns the encode wall seconds,
    which is the per-segment cost a streaming worker pays off the GPU path.
    """
    frames_u8 = frames_u8.contiguous()
    n, height, width, _ = frames_u8.shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-"]
    if audio_wav:
        # No -shortest: AAC frames quantize to 1024 samples, so the audio can
        # land a few ms short of the video and -shortest would drop a frame.
        cmd += ["-i", audio_wav, "-c:a", "aac", "-b:a", "128k"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof", str(out_path)]
    started = time.monotonic()
    proc = subprocess.run(cmd, input=frames_u8.numpy().tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"segment mux failed: {proc.stderr.decode(errors='replace')[-500:]}")
    return time.monotonic() - started


def mux_all_segments(rec: StreamRecorder, out_dir: str | Path, fps: int = 24,
                     sample_rate: int = 32000, total_frames: int | None = None) -> list[dict]:
    """Mux every recorded chunk with its audio slice; returns per-segment info."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio = rec.audio_out
    if audio is not None and audio.dim() == 3:
        audio = audio[0]
    results = []
    with tempfile.TemporaryDirectory() as td:
        for idx, (meta, frames) in enumerate(zip(rec.chunks, rec.frames)):
            start_f, end_f = meta["frame_start"], meta["frame_end"]
            if total_frames is not None:
                end_f = min(end_f, total_frames)
                if start_f >= end_f:
                    continue
                frames = frames[: end_f - start_f]
            wav = None
            if audio is not None:
                s0 = int(round(start_f / fps * sample_rate))
                s1 = int(round(end_f / fps * sample_rate))
                s1 = min(s1, audio.shape[-1])
                if s1 > s0:
                    wav = f"{td}/seg{idx}.wav"
                    write_wav(wav, audio[:, s0:s1], sample_rate)
            seg_path = out_dir / f"segment_{idx:03d}.mp4"
            mux_s = mux_segment(frames, wav, seg_path, fps=fps)
            results.append({"segment": idx, "frame_start": start_f, "frame_end": end_f,
                            "duration_s": (end_f - start_f) / fps, "mux_s": round(mux_s, 4),
                            "path": str(seg_path)})
    return results


# -- player schedule model ------------------------------------------------

def playback_offsets(durations: list[float]) -> list[float]:
    """Media-time offset at which each segment starts playing."""
    offsets, acc = [], 0.0
    for d in durations:
        offsets.append(acc)
        acc += d
    return offsets


def no_buffer_start(ready: list[float], durations: list[float]) -> float:
    """Oracle: earliest playback start with zero rebuffering (post-hoc)."""
    offsets = playback_offsets(durations)
    return max(r - o for r, o in zip(ready, offsets))


def linear_estimate_start(ready: list[float], durations: list[float],
                          observe: int = 2, pad: float = 1.15) -> dict:
    """The deployable policy: watch the first `observe` segment arrivals,
    assume the rest arrive at that linear rate padded by `pad`, and commit to
    the earliest start time that keeps predicted playback ahead of arrivals.

    Returns the committed start, whether the real arrivals would have caused a
    rebuffer, and the worst-case margin (min over segments of how many seconds
    each segment arrived before the player needed it; negative = stall).
    """
    n = len(ready)
    observe = max(1, min(observe, n))
    offsets = playback_offsets(durations)
    if observe >= 2:
        rate = (ready[observe - 1] - ready[0]) / (observe - 1)
    else:
        rate = ready[0]
    predicted = list(ready[:observe])
    for i in range(observe, n):
        predicted.append(ready[observe - 1] + (i - (observe - 1)) * rate * pad)
    start = max(p - o for p, o in zip(predicted, offsets))
    start = max(start, ready[0])
    margins = [start + o - r for r, o in zip(ready, offsets)]
    worst = min(margins) if margins else 0.0
    return {"start_s": start, "rate_est_s": rate, "would_rebuffer": worst < 0.0,
            "worst_margin_s": worst, "predicted": predicted}
