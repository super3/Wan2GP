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

The pure-math helpers at the bottom model the player: `no_buffer_start` is the
oracle earliest start with zero rebuffering, `linear_estimate_start` is the
deployable policy (observe the first chunks, fit a linear completion rate, pad
it, commit to a start time) that the bench validates against actual arrivals.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import torch

from shared.utils.phase_progress import PhaseProgress

from .components import video_autoencoder as _va


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
        #: (id(latents), decoded) so an audio-first prefetch is never paid twice
        self.audio_cache: tuple[int, object] | None = None
        #: True when the audio-first hook actually decoded audio before video
        self.audio_first_effective = False

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
                "orig_audio_decode": None, "orig_video_decode": None}


def recorder() -> StreamRecorder | None:
    return _STATE["recorder"]


def _maybe_prefetch_audio(caller_locals: dict, rec: StreamRecorder) -> bool:
    """Audio-first hook: decode audio before the video decode starts.

    The pipeline decodes video before audio (`decoded_video = self.vae.decode(
    video, uint8_rounding)` then `self.audio_vae.decode(audio)`), which would
    leave live segments silent until the whole clip is decoded. At the moment
    the video decode is entered, the caller's frame already holds both the
    pipeline (`self`) and the audio latents (`audio`), so decode them now; the
    audio wrapper caches the result by latents identity and the pipeline's own
    later call returns it without paying twice. Image outputs (`image_outputs`)
    return before any audio decode, so they are skipped. Every check fails
    closed: any mismatch with the expected call site just keeps the original
    ordering.
    """
    try:
        if caller_locals.get("image_outputs"):
            return False  # image generations never decode their audio latents
        pipe = caller_locals.get("self")
        audio = caller_locals.get("audio")
        audio_vae = getattr(pipe, "audio_vae", None)
        if audio_vae is None or not torch.is_tensor(audio):
            return False
        if rec.audio_cache is not None:
            return True
        audio_vae.decode(audio)
        return True
    except Exception:  # noqa: BLE001 - the hook must never break generation
        return False


def _instrumented_decode(self, z: torch.Tensor, uint8_rounding: str | None = None) -> torch.Tensor:
    """`AutoencoderKLMiniMaxH3._decode` with recorder callbacks.

    The body is copied verbatim from components/video_autoencoder.py (upstream
    dee6635: single-latent still path, CPU output buffer filled by
    `write_chunk`, `PhaseProgress`); the callbacks only read finalized regions
    of that buffer, so the returned tensor is bit-identical to the original
    implementation's.
    """
    rec: StreamRecorder = _STATE["recorder"]
    rec.on_decode_start()

    tokens_chunk_size = self.tokens_chunk_size
    token_drop = self.config.token_drop
    temporal_ratio = self.temporal_compression_ratio
    chunk_num_frames = tokens_chunk_size * temporal_ratio

    if z.shape[2] == 1:
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
    with PhaseProgress(max(1, num_chunks) * spatial_tiles) as progress, progress.track(self.decoder, spatial_tiles):
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
                decoded[:, :, write_position : write_position + part.shape[2]].copy_(part, non_blocking=False)
                write_position += part.shape[2]
                del part

        if num_chunks == 0:
            # Short videos still need one decode, without an overlapping temporal chunk.
            write_chunk(self._decode_clip(z)[:, :, self.frame_pre_padding : self.frame_pre_padding + output_frames])
            rec.on_chunk(self, decoded, 0, write_position, 0)
            rec.on_decode_end()
            return decoded

        for i in range(num_chunks):
            start = i * tokens_chunk_size
            clip = self._decode_clip(z[:, :, start : start + tokens_chunk_size + self.token_overlap])
            for j in range(int(token_drop > 0) + 1):
                frame_start = j * chunk_num_frames
                chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
                chunk = chunk[:, :, self.frame_pre_padding :]
                if j == 0:
                    if overlap is not None:
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
             audio_first: bool = False) -> StreamRecorder:
    """Install the instrumented decode paths; returns the shared recorder.

    `live_dir` turns on the live segmenter: every finalized chunk is muxed to
    a standalone fMP4 segment in a background thread as it lands, under
    `live_dir/gen_NNN/`. `audio_first` additionally decodes the audio before
    the video decode starts (see `_maybe_prefetch_audio`), so live segments
    carry their audio slice from the first chunk onward.
    """
    if _STATE["active"]:
        rec = _STATE["recorder"]
        rec.store_frames = store_frames
        rec.live_dir = live_dir
        rec.audio_first = audio_first
        return rec
    from . import audio_vae as _audio_mod
    from . import video_vae as _video_mod

    rec = StreamRecorder(store_frames=store_frames, live_dir=live_dir, audio_first=audio_first)
    _STATE["recorder"] = rec
    _STATE["orig_decode"] = _va.AutoencoderKLMiniMaxH3._decode
    _va.AutoencoderKLMiniMaxH3._decode = _instrumented_decode

    orig_audio_decode = _audio_mod.MiniMaxH3AudioVAE.decode

    def timed_audio_decode(self, latents):
        if rec.audio_cache is not None and rec.audio_cache[0] == id(latents):
            return rec.audio_cache[1]
        rec.audio_start_t = time.monotonic()
        out = orig_audio_decode(self, latents)
        rec.audio_end_t = time.monotonic()
        rec.audio_cache = (id(latents), out)
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
        if rec.audio_first:
            rec.audio_first_effective = _maybe_prefetch_audio(sys._getframe(1).f_locals, rec)
        return orig_video_decode(self, latents, *args, **kwargs)

    _STATE["orig_video_decode"] = orig_video_decode
    _video_mod.MiniMaxH3VideoVAE.decode = audio_first_video_decode
    _STATE["active"] = True
    return rec


def deactivate() -> None:
    if not _STATE["active"]:
        return
    from . import audio_vae as _audio_mod
    from . import video_vae as _video_mod

    rec = _STATE["recorder"]
    if rec is not None and rec.live is not None:
        rec.live.close()
    _va.AutoencoderKLMiniMaxH3._decode = _STATE["orig_decode"]
    _audio_mod.MiniMaxH3AudioVAE.decode = _STATE["orig_audio_decode"]
    _video_mod.MiniMaxH3VideoVAE.decode = _STATE["orig_video_decode"]
    _STATE.update({"active": False, "recorder": None, "orig_decode": None,
                   "orig_audio_decode": None, "orig_video_decode": None})


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
