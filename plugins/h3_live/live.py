# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""H3 Live: MiniMax H3 clips generated back to back as one live stream.

A live session is one WanGP generation. `install()` wraps `MiniMaxH3Pipeline.generate`; while live mode is on, the one call
WanGP makes for an H3 video runs the pipeline once per clip of the session instead (each with its own prompt and seed), and
returns the last clip to WanGP, which saves it as usual. Between clips nothing leaves the GPU: no model unload, no LoRA reload,
no garbage collection, no save in the generation thread. Each clip goes to a background encoder, which writes it as an MP4 and
as a segment of a rolling HLS playlist that a player follows while the next clips render.

WanGP is patched from outside: the only hooks are `MiniMaxH3Pipeline.generate` (keyword arguments `input_prompt`, `seed`,
`frame_num`, `fps`; a result with `x`, `audio`, `audio_sampling_rate`), `MiniMaxH3Pipeline._encode_prompt(prompt,
presentation)` to encode a session's prompts once before its first clip (and, from embeddings.py, to use embeddings made
elsewhere in its place), and mmgp's `offloadobj.cotenants_map`, so that the audio decode does not evict the transformer
between clips. The same generate hook serves prompt embeddings instead of a video while an `embeddings.EmbeddingService` is
waiting for a model (embed_server.py). `compatibility_problems()` checks them; live mode refuses to start
when they do not match this WanGP, instead of breaking its generations.
"""

from __future__ import annotations

import functools
import inspect
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

GENERATE_PARAMS = ("input_prompt", "seed", "frame_num", "fps")  # read and set by the session loop
RESULT_KEYS = ("x", "audio", "audio_sampling_rate")
BIG_VRAM_BYTES = 60 * 2**30  # from here the transformer stays beside the text encoder; below, prompt changes may move it out
SESSION_PREFIX = "live_"
STREAM_START_S = 1.0  # where the stream's timestamps start
STATE: dict = {"installed": False, "pipeline_cls": None, "orig_generate": None, "settings": None, "session": None, "last_session": None,
               "embed_service": None}


@dataclass
class LiveSettings:
    enabled: bool = False
    prompts: list[str] = field(default_factory=list)  # one per clip, cycled; empty: the task's own prompt
    clips: int = 0  # 0: one clip per prompt; -1: until stopped
    seed_step: int = 1  # clip i renders with the task's seed + i * seed_step
    encoder: str = "auto"  # auto (NVENC where ffmpeg can use it), nvenc, x264
    keep_sessions: int = 5  # older session folders are deleted when a session starts
    pre_encode_prompts: bool = True  # encode every prompt of the session before its first clip
    pace_lead_s: float = 0.0  # > 0: never more than this much video ahead of the stream's clock (a broadcast); 0: as fast as it goes
    playlist_window: int = 0  # > 0: a live playlist of the last N clips, older clip files deleted (endless sessions); 0: keep them all
    embeddings_dir: str = ""  # a folder of prompt embeddings made elsewhere (*.safetensors, embed_server.py), used instead of the text encoder
    cache_image_encodes: bool = True  # a start / end image's VAE encoding made once per session (the same picture every clip)
    steady_vram_preload: bool = True  # the clips of a session share mmgp's VRAM preload measure, instead of one per prompt length

    @classmethod
    def from_dict(cls, data):
        known = {name: data[name] for name in cls.__dataclass_fields__ if name in (data or {})}
        return cls(**known)


# -- compatibility ---------------------------------------------------------------------------------------------------------------

def compatibility_problems(pipeline_cls) -> list[str]:
    """What keeps live mode from running on this WanGP: an empty list when the hooks match."""
    problems = []
    generate = getattr(pipeline_cls, "generate", None)
    if not callable(generate):
        return ["MiniMaxH3Pipeline.generate is missing"]
    try:
        parameters = inspect.signature(inspect.unwrap(generate)).parameters
    except (TypeError, ValueError):
        parameters = {}
    missing = [name for name in GENERATE_PARAMS if name not in parameters]
    if missing:
        problems.append(f"MiniMaxH3Pipeline.generate has no {', '.join(missing)} parameter")
    encode = getattr(pipeline_cls, "_encode_prompt", None)
    if not callable(encode) or list(inspect.signature(encode).parameters)[1:3] != ["prompt", "presentation"]:
        problems.append("MiniMaxH3Pipeline._encode_prompt(prompt, presentation) is missing: prompts are encoded per clip")
    return problems


def blocking_problems(problems) -> list[str]:
    """The problems that stop live mode (prompt pre-encoding is optional)."""
    return [problem for problem in problems if "_encode_prompt" not in problem]


# -- encoding --------------------------------------------------------------------------------------------------------------------

_NVENC_PROBE: dict = {}


def find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - no bundled ffmpeg either
        return None


def nvenc_available(ffmpeg: str) -> bool:
    """Whether this ffmpeg encodes H.264 with NVENC here (checked once per binary)."""
    if ffmpeg not in _NVENC_PROBE:
        probe = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2", "-c:v", "h264_nvenc", "-f", "null", "-"]
        try:
            _NVENC_PROBE[ffmpeg] = subprocess.run(probe, capture_output=True, timeout=60, check=False).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            _NVENC_PROBE[ffmpeg] = False
    return _NVENC_PROBE[ffmpeg]


def video_args(kind: str) -> list[str]:
    if kind == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "19", "-b:v", "0", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]


def choose_encoder(requested: str, ffmpeg: str) -> str:
    if requested == "x264":
        return "x264"
    if nvenc_available(ffmpeg):
        return "nvenc"
    if requested == "nvenc":
        print("[H3 Live] NVENC is not available to this ffmpeg: clips are encoded with x264")
    return "x264"


def frames_to_rgb(frames) -> np.ndarray:
    """A clip's frames as (F, H, W, 3) uint8: the pipeline's (C, F, H, W) uint8 video, or [-1, 1] floats."""
    try:
        import torch

        if isinstance(frames, torch.Tensor):
            if frames.dtype != torch.uint8:
                frames = frames.float().clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
            return frames.permute(1, 2, 3, 0).contiguous().cpu().numpy()
    except ImportError:
        pass
    array = np.asarray(frames)
    if array.dtype != np.uint8:
        array = np.clip(np.rint((array.astype(np.float32) + 1.0) * 127.5), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(np.transpose(array, (1, 2, 3, 0)))


def write_wav(path, audio, sample_rate: int, samples: int) -> None:
    """Stereo 16-bit WAV of `samples` samples (cut or padded with silence) from (n, 2) or (2, n) float audio."""
    data = np.zeros((samples, 2), dtype=np.float32)
    if audio is not None:
        audio = np.asarray(audio.detach().float().cpu() if hasattr(audio, "detach") else audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], 1)
        elif audio.shape[0] == 2 and audio.shape[1] != 2:
            audio = audio.T
        audio = audio[:, :2] if audio.shape[1] >= 2 else np.repeat(audio, 2, axis=1)
        count = min(samples, audio.shape[0])
        data[:count] = audio[:count]
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(int(sample_rate))
        writer.writeframes(pcm.tobytes())


def encode_clip(ffmpeg: str, rgb: np.ndarray, audio, sample_rate: int, fps: float, mp4_path: Path, ts_path: Path, offset_s: float, kind: str) -> None:
    """One clip as an MP4 (H.264 + AAC) and as an MPEG-TS segment whose timestamps continue the stream at offset_s."""
    count, height, width, _ = rgb.shape
    wav = mp4_path.with_suffix(".wav")
    write_wav(wav, audio, sample_rate, round(count / fps * sample_rate))
    try:
        encode = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
                  "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0", *video_args(kind), "-g", str(count), "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(mp4_path)]
        done = subprocess.run(encode, input=rgb.tobytes(), capture_output=True, check=False)
        if done.returncode != 0:
            raise RuntimeError(f"ffmpeg could not encode the clip: {done.stderr.decode(errors='replace')[-400:]}")
    finally:
        wav.unlink(missing_ok=True)
    segment = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(mp4_path), "-c", "copy", "-bsf:v", "h264_mp4toannexb",
               "-output_ts_offset", f"{offset_s:.6f}", "-muxdelay", "0", "-f", "mpegts", str(ts_path)]
    done = subprocess.run(segment, capture_output=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"ffmpeg could not write the stream segment: {done.stderr.decode(errors='replace')[-400:]}")


# -- the stream ------------------------------------------------------------------------------------------------------------------

class HlsPlaylist:
    """An HLS event playlist that grows by one segment per clip and ends with the session. `uri` maps a segment's file name to
    the URI written in the playlist (a plain name for players reading the folder; a web URL for the browser player)."""

    def __init__(self, path: Path, target_duration: int, uri=None, window: int = 0):
        self.path, self.target_duration, self.uri = Path(path), max(1, int(target_duration)), uri or (lambda name: name)
        self.window = max(0, int(window))  # > 0: a live playlist of the last `window` segments (older ones drop out)
        self.entries: list[tuple[str, float]] = []
        self.sequence = 0  # media sequence number of the first entry
        self.ended = False
        self.write()

    def add(self, name: str, duration: float) -> None:
        self.entries.append((name, duration))
        if self.window and len(self.entries) > self.window:
            self.sequence += len(self.entries) - self.window
            self.entries = self.entries[-self.window:]
        self.write()

    def finish(self) -> None:
        self.ended = True
        self.write()

    def text(self) -> str:
        lines = ["#EXTM3U", "#EXT-X-VERSION:3", f"#EXT-X-TARGETDURATION:{self.target_duration}", f"#EXT-X-MEDIA-SEQUENCE:{self.sequence}"]
        if not self.window:
            lines.append("#EXT-X-PLAYLIST-TYPE:EVENT")
        for name, duration in self.entries:
            lines += [f"#EXTINF:{duration:.3f},", self.uri(name)]
        if self.ended:
            lines.append("#EXT-X-ENDLIST")
        return "\n".join(lines) + "\n"

    def write(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(self.text())
        os.replace(temporary, self.path)


def prune_sessions(root: Path, keep: int) -> None:
    """Delete all but the `keep` newest session folders (made by this plugin) under root."""
    sessions = sorted((path for path in root.glob(SESSION_PREFIX + "*") if path.is_dir() and (path / "session.json").exists()), key=lambda path: path.name)
    for old in sessions[:max(0, len(sessions) - max(0, int(keep)))]:
        shutil.rmtree(old, ignore_errors=True)


class LiveSession:
    """One live session's clips: encoded in a background thread, in order, as they are generated. At most `max_waiting` clips
    wait for the encoder (each holds its frames); the generation thread blocks beyond that."""

    def __init__(self, root, fps: float, clip_seconds: float, encoder: str, ffmpeg: str, web_uri=None, keep_sessions: int = 5, max_waiting: int = 2,
                 window: int = 0, kept_records: int = 300):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        prune_sessions(root, max(0, keep_sessions - 1))
        self.dir = root / f"{SESSION_PREFIX}{time.strftime('%Y%m%d-%H%M%S')}"
        suffix = 1
        while self.dir.exists():
            suffix += 1
            self.dir = root / f"{SESSION_PREFIX}{time.strftime('%Y%m%d-%H%M%S')}_{suffix}"
        self.dir.mkdir()
        self.fps, self.encoder, self.ffmpeg = float(fps), encoder, ffmpeg
        target = math.ceil(clip_seconds) + 1
        self.window = max(0, int(window))
        self.playlist = HlsPlaylist(self.dir / "live.m3u8", target, window=self.window)
        self.web_playlist = HlsPlaylist(self.dir / "live_web.m3u8", target, lambda name: web_uri(self.dir / name), window=self.window) if web_uri else None
        self.records: deque = deque(maxlen=max(1, int(kept_records)))  # the latest clips' records (an endless session keeps no more)
        self.clip_count = self.error_count = 0
        self.seconds_total = 0.0
        self.first_ready_s: float | None = None
        self.offset_s = STREAM_START_S  # the stream's clock (> 0: a first segment with B-frames has no negative timestamps to shift)
        self.started = time.monotonic()
        self.stop_requested = False
        self.closed = False
        self.film: str | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, max_waiting))
        self._thread = threading.Thread(target=self._run, name="h3-live-encoder", daemon=True)
        self._thread.start()
        self._write_summary()

    def submit(self, index: int, prompt: str, seed: int, result: dict, generate_s: float) -> None:
        self._queue.put((index, prompt, seed, result, generate_s, time.monotonic()))

    def _run(self) -> None:
        while (item := self._queue.get()) is not None:
            index, prompt, seed, result, generate_s, generated_t = item
            record = {"clip": index, "prompt": prompt[:200], "seed": seed, "generate_s": round(generate_s, 3), "generated_at_s": round(generated_t - self.started, 3)}
            try:
                started = time.monotonic()
                rgb = frames_to_rgb(result["x"])
                seconds = rgb.shape[0] / self.fps
                name = f"clip_{index:05d}"
                encode_clip(self.ffmpeg, rgb, result.get("audio"), int(result.get("audio_sampling_rate") or 32000), self.fps,
                            self.dir / f"{name}.mp4", self.dir / f"{name}.ts", self.offset_s, self.encoder)
                self.offset_s += seconds
                self.playlist.add(f"{name}.ts", seconds)
                if self.web_playlist is not None:
                    self.web_playlist.add(f"{name}.ts", seconds)
                record.update(frames=int(rgb.shape[0]), seconds=round(seconds, 3), encode_s=round(time.monotonic() - started, 3),
                              ready_at_s=round(time.monotonic() - self.started, 3), mp4=f"{name}.mp4", segment=self.clip_count)  # its media sequence number
                self.clip_count += 1
                self.seconds_total += seconds
                if self.first_ready_s is None:
                    self.first_ready_s = record["ready_at_s"]
                if self.window:  # clips out of the playlist (with a margin for players still reading them) are deleted
                    for old in (self.dir / f"clip_{index - self.window - 3:05d}.ts", self.dir / f"clip_{index - self.window - 3:05d}.mp4"):
                        old.unlink(missing_ok=True)
            except Exception as error:  # noqa: BLE001 - one bad clip must not end the session
                record["error"] = repr(error)[:400]
                self.error_count += 1
            result = item = None
            self.records.append(record)
            self._write_summary()

    def close(self) -> None:
        """Wait for the clips being encoded, end the playlists, and join the clips into film.mp4."""
        if self.closed:
            return
        self.closed = True
        self._queue.put(None)
        self._thread.join()
        self.playlist.finish()
        if self.web_playlist is not None:
            self.web_playlist.finish()
        segments = [] if self.window else [self.dir / f"clip_{record['clip']:05d}.ts" for record in self.records if "error" not in record]
        if segments:
            film = self.dir / "film.mp4"
            join = [self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", "concat:" + "|".join(str(path) for path in segments), "-c", "copy",
                    "-bsf:a", "aac_adtstoasc", "-movflags", "+faststart", str(film)]
            if subprocess.run(join, capture_output=True, check=False).returncode == 0:
                self.film = str(film)
        self._write_summary()

    def summary(self, recent: int = 10) -> dict:
        """The session so far: `cycle_s` is the time between clips (paced sessions: the stream's pace), `generation_x` the clips' length
        over their generation time (how much faster than real time the GPU makes them), both over the `recent` latest clips."""
        done = [record for record in self.records if "error" not in record][-max(2, recent):]
        cycle = None
        if len(done) > 1:
            cycle = (done[-1]["generated_at_s"] - done[0]["generated_at_s"]) / (len(done) - 1)
        generate = sum(record["generate_s"] for record in done) / len(done) if done else None
        return {"dir": str(self.dir), "encoder": self.encoder, "clips": self.clip_count, "errors": self.error_count,
                "seconds_of_video": round(self.seconds_total, 3), "cycle_s": None if cycle is None else round(cycle, 3),
                "realtime_x": None if not cycle else round(done[-1]["seconds"] / cycle, 3),
                "generate_s": None if generate is None else round(generate, 3), "generation_x": None if not generate else round(done[-1]["seconds"] / generate, 3),
                "first_clip_ready_s": self.first_ready_s, "uptime_s": round(time.monotonic() - self.started, 1), "closed": self.closed, "film": self.film,
                "records": list(self.records)}

    def _write_summary(self) -> None:
        path = self.dir / "session.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.summary(), indent=1))
        os.replace(temporary, path)


# -- the generation hook ---------------------------------------------------------------------------------------------------------

def _wgp():
    return sys.modules.get("wgp")


def _gpu_bytes() -> int:
    try:
        import torch

        return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001
        return 0


@contextmanager
def transformer_resident(offloadobj=None, big_vram: bool | None = None):
    """mmgp keeps a model in VRAM while another loads only when they are co-tenants. For the session, the transformer becomes
    one: of every model (`"*"`) on large GPUs; on smaller ones, of the VAEs only, so that a text encoder load (a new prompt) can
    still move it out instead of running out of memory. The map is restored afterwards."""
    offloadobj = offloadobj if offloadobj is not None else getattr(_wgp(), "offloadobj", None)
    original = getattr(offloadobj, "cotenants_map", None)
    if not isinstance(original, dict):
        yield
        return
    big_vram = _gpu_bytes() >= BIG_VRAM_BYTES if big_vram is None else big_vram
    updated = dict(original)
    if big_vram:
        updated["transformer"] = "*"
    else:
        for name in ("audio_vae", "vae", "video_encoder"):
            current = updated.get(name)
            if current == "*":
                continue
            listed = list(current) if isinstance(current, (list, tuple)) else []
            updated[name] = listed if "transformer" in listed else listed + ["transformer"]
    offloadobj.cotenants_map = updated
    try:
        yield
    finally:
        offloadobj.cotenants_map = original


def _tensor_key(tensor) -> tuple:
    import hashlib

    import torch

    data = tensor.detach().to("cpu").contiguous()
    raw = data.view(torch.int16) if data.dtype == torch.bfloat16 else data  # numpy has no bfloat16
    return tuple(data.shape), str(data.dtype), hashlib.sha256(raw.numpy().tobytes()).hexdigest()


@contextmanager
def image_encode_cache(pipe):
    """A start or end image is the same picture in every clip of a session, so its VAE encoding (one frame, the pipeline's
    `_encode_video`) is made once and reused; any other encode runs as usual. Besides the encode itself, running the VAE
    encoder in every clip moved the audio VAE out of the GPU and back (measured on an RTX PRO 6000: a clip with a start and an
    end image from 5.81 to 4.66 s, 0.89x to 1.11x real time). The encode is deterministic, so the latents are the same."""
    encode = getattr(pipe, "_encode_video", None)
    if not callable(encode):
        yield None
        return
    cache, stats = {}, {"hits": 0, "misses": 0}
    had_own = "_encode_video" in getattr(pipe, "__dict__", {})
    previous = pipe.__dict__.get("_encode_video") if had_own else None

    def cached(video, keep_all_latents=False):
        if keep_all_latents or getattr(video, "ndim", 0) != 4 or video.shape[1] != 1:
            return encode(video, keep_all_latents)
        key = _tensor_key(video)
        if key not in cache:
            stats["misses"] += 1
            cache[key] = encode(video, keep_all_latents)
        else:
            stats["hits"] += 1
        return cache[key].clone()

    pipe._encode_video = cached  # the instance's attribute, in front of the class's method
    try:
        yield stats
    finally:
        if had_own:
            pipe._encode_video = previous
        else:
            pipe.__dict__.pop("_encode_video", None)


WORKLOAD_STEPS = 64  # a workload rounded up to 1/64 of its power of two


def bucket_workload(elements: int) -> int:
    """`elements` rounded up to a step of 1/64 of its power of two (at most 1.6% more): workloads a few tokens apart are one."""
    elements = int(elements)
    if elements <= WORKLOAD_STEPS:
        return elements
    step = (1 << (elements.bit_length() - 1)) // WORKLOAD_STEPS
    return -(-elements // step) * step


@contextmanager
def steady_preload(model_id: str = "transformer"):
    """mmgp's automatic VRAM preload (profiles 1 and 3, and Dynamic) keeps its measure of the VRAM a workload needs by the size of
    the largest input of a block, and runs the first step of a workload it has not measured with the budget's preload: most of the
    transformer streamed from RAM. In a live session every clip's prompt is a few tokens longer or shorter, so every clip was a new
    workload, and with DWD's two steps half of the denoising streamed the weights. For the session, the transformer's workloads
    are rounded up (bucket_workload): the clips share one measure, and after the first clip the transformer stays in VRAM."""
    try:
        from mmgp import auto_preload
    except ImportError:
        yield 0
        return
    patched = []
    for planner in list(getattr(auto_preload, "_planners", ()) or ()):
        plan = getattr(planner, "plans", {}).get(model_id)
        workload_of = getattr(plan, "workload_of", None)
        if callable(workload_of) and not getattr(workload_of, "_h3_live_bucketed", False):
            def bucketed(args, kwargs, original=workload_of):
                return bucket_workload(original(args, kwargs))

            bucketed._h3_live_bucketed = True
            plan.workload_of = bucketed
            patched.append((plan, workload_of))
    try:
        yield len(patched)
    finally:
        for plan, original in patched:
            plan.workload_of = original


def _live_root() -> Path:
    wgp = _wgp()
    save_path = getattr(wgp, "save_path", None) or "outputs"
    return Path(save_path) / "h3_live"


def _web_uri():
    try:
        from shared.gradio.strip_gallery import gradio_file_url
    except ImportError:
        return None
    return lambda path: gradio_file_url(str(path))


def _pre_encode(pipe, prompts) -> None:
    encode = getattr(pipe, "_encode_prompt", None)
    if not callable(encode) or getattr(pipe, "fixed_prompt", None) is not None:
        return
    import torch

    with torch.inference_mode():
        for prompt in dict.fromkeys(prompts):
            encode(prompt, [])  # text only: the same cache key as the pipeline's own call


class Pacer:
    """Keeps a session at most `lead_s` of video ahead of the stream's clock, which starts with the first clip: a broadcast that
    plays at 1x while the GPU, faster than real time, waits. lead_s <= 0: no waiting."""

    def __init__(self, lead_s: float, clock=time.monotonic, sleep=time.sleep):
        self.lead_s, self.clock, self.sleep, self.start = float(lead_s or 0), clock, sleep, None

    def started(self) -> None:
        if self.start is None:
            self.start = self.clock()

    def wait(self, produced_s: float, stopped) -> bool:
        """Wait until the video made so far (produced_s) is within lead_s of the stream's clock. False when stopped meanwhile."""
        if self.lead_s <= 0 or self.start is None:
            return True
        while produced_s - (self.clock() - self.start) > self.lead_s:
            if stopped():
                return False
            self.sleep(0.05)
        return True


def run_session(orig_generate, pipe, kwargs: dict, settings: LiveSettings, root=None, web_uri=None, ffmpeg=None):
    """The live session behind one WanGP generation: the pipeline once per clip, each clip to the encoder. Returns the last
    clip (WanGP saves it), or None when WanGP stopped the generation."""
    if settings.embeddings_dir:
        print(f"[H3 Live] {_embeddings().supply_dir(settings.embeddings_dir)} prompt embeddings from {settings.embeddings_dir}")
    prompts = [prompt for prompt in settings.prompts if prompt.strip()] or [kwargs.get("input_prompt", "")]
    count = math.inf if int(settings.clips) < 0 else int(settings.clips) if int(settings.clips) > 0 else len(prompts)
    fps = float(kwargs.get("fps") or 24)
    ffmpeg = ffmpeg or find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("H3 Live needs ffmpeg (on the PATH, or imageio-ffmpeg)")
    clip_seconds = int(kwargs.get("frame_num") or 124) / fps
    session = LiveSession(root or _live_root(), fps, clip_seconds, choose_encoder(settings.encoder, ffmpeg), ffmpeg,
                          web_uri=web_uri, keep_sessions=settings.keep_sessions, window=settings.playlist_window)
    STATE["session"] = STATE["last_session"] = session
    base_seed = int(kwargs.get("seed") or 0)
    status = kwargs.get("set_progress_status")
    last = None
    print(f"[H3 Live] session {session.dir.name}: {'endless' if count == math.inf else count} clips, encoder {session.encoder}")
    pacer = Pacer(settings.pace_lead_s)
    try:
        with transformer_resident(), (image_encode_cache(pipe) if settings.cache_image_encodes else _nothing()), \
                (steady_preload() if settings.steady_vram_preload else _nothing()):
            if settings.pre_encode_prompts and kwargs.get("image_start") is None and kwargs.get("image_end") is None:
                _pre_encode(pipe, prompts)
            index = 0
            while index < count:
                if session.stop_requested or getattr(pipe, "_interrupt", False):
                    break
                if not pacer.wait(index * clip_seconds, lambda: session.stop_requested or getattr(pipe, "_interrupt", False)):
                    break
                clip_kwargs = dict(kwargs, input_prompt=prompts[index % len(prompts)], seed=base_seed + index * int(settings.seed_step))
                if callable(status):
                    status(f"H3 Live: clip {index + 1}" + ("" if count == math.inf else f"/{count}"))
                started = time.monotonic()
                result = orig_generate(pipe, **clip_kwargs)
                if result is None:  # stopped from WanGP
                    last = None
                    break
                missing = [key for key in RESULT_KEYS if key not in result]
                if missing:
                    raise RuntimeError(f"H3 Live: this WanGP's MiniMax H3 pipeline returns no {', '.join(missing)}: live mode does not support it")
                session.submit(index, clip_kwargs["input_prompt"], clip_kwargs["seed"], result, time.monotonic() - started)
                pacer.started()
                last = result
                index += 1
    finally:
        STATE["session"] = None
        session.close()
        print(f"[H3 Live] session {session.dir.name} done: {json.dumps({k: v for k, v in session.summary().items() if k != 'records'})}")
    return last


@contextmanager
def _nothing():
    yield None


def continues_a_video(kwargs: dict) -> bool:
    """A sliding window after the first, or a video continued from a source: generated as usual (a live session per window
    would repeat the whole session). A start image is not one: WanGP hands it to the pipeline as a one-frame prefix (input_video
    of one frame, prefix_frames_count 1) beside image_start, and every clip of the session starts on it."""
    if int(kwargs.get("window_start_frame_no") or 0) > 0:
        return True
    prefix = int(kwargs.get("prefix_frames_count") or 0)
    if kwargs.get("image_start") is not None and prefix <= 1:
        return False
    return kwargs.get("input_video") is not None or prefix > 0


def _embeddings():
    """embeddings.py beside this file: the plugin's module in WanGP and the demo, loaded by path when live.py is on its own."""
    try:
        from . import embeddings
        return embeddings
    except ImportError:
        import importlib.util

        name = "h3_live_embeddings"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("embeddings.py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return sys.modules[name]


def _make_wrapper(orig_generate):
    @functools.wraps(orig_generate)
    def generate(self, *args, **kwargs):
        service = STATE.get("embed_service")
        if service is not None and not service.running and not args and int(kwargs.get("image_mode") or 0) == 0:
            service.run(self)  # until the service stops: this generation makes no video
            return None
        settings: LiveSettings | None = STATE["settings"]
        if (settings is None or not settings.enabled or STATE["session"] is not None or args or int(kwargs.get("image_mode") or 0) > 0
                or getattr(self, "audio_only", False) or continues_a_video(kwargs)):
            return orig_generate(self, *args, **kwargs)
        return run_session(orig_generate, self, kwargs, settings, web_uri=_web_uri())

    generate._h3_live = True
    return generate


def install(pipeline_cls=None) -> list[str]:
    """Wrap MiniMaxH3Pipeline.generate (once). Returns the problems that keep live mode from running on this WanGP."""
    if pipeline_cls is None:
        from models.minimax_h3.pipeline import MiniMaxH3Pipeline as pipeline_cls
    problems = compatibility_problems(pipeline_cls)
    if blocking_problems(problems):
        return problems
    if not STATE["installed"]:
        STATE.update(installed=True, pipeline_cls=pipeline_cls, orig_generate=pipeline_cls.generate)
        pipeline_cls.generate = _make_wrapper(pipeline_cls.generate)
        _embeddings().install(pipeline_cls)
    return problems


def uninstall() -> None:
    if STATE["installed"]:
        STATE["pipeline_cls"].generate = STATE["orig_generate"]
        STATE.update(installed=False, pipeline_cls=None, orig_generate=None)
        _embeddings().uninstall()


def configure(settings: LiveSettings | None) -> None:
    STATE["settings"] = settings


def request_stop() -> bool:
    session = STATE["session"]
    if session is None:
        return False
    session.stop_requested = True
    return True


def settings_dict(settings: LiveSettings) -> dict:
    return asdict(settings)
