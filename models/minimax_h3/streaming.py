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
Finalized slices are normalized exactly the way `MiniMaxH3VideoVAE.decode`
would (ImageNet de-normalization, clamp to [0, 1]) and handed to the recorder
as uint8 CPU frames, which is also the copy a real streaming mux would need,
so the timestamps include the device-to-host cost. The audio VAE decode is
wrapped for timing and output capture: audio decoding is independent of video
decoding, so a streaming pipeline reorders it first and every video chunk
shifts later by its duration (serial execution makes that shift exact).

The pure-math helpers at the bottom model the player: `no_buffer_start` is the
oracle earliest start with zero rebuffering, `linear_estimate_start` is the
deployable policy (observe the first chunks, fit a linear completion rate, pad
it, commit to a start time) that the bench validates against actual arrivals.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
import wave
from pathlib import Path

import torch

from .components import video_autoencoder as _va


class StreamRecorder:
    """Collects chunk/audio events from one instrumented generation."""

    def __init__(self, store_frames: bool = True):
        self.store_frames = store_frames
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

    # -- hooks ------------------------------------------------------------
    def on_decode_start(self) -> None:
        self.decode_start_t = time.monotonic()

    def on_chunk(self, vae, decoded: torch.Tensor, frame_start: int, frame_end: int, chunk_index: int) -> None:
        if frame_end <= frame_start:
            return
        # .float() is a no-op view when decoded is already float32, so clone
        # explicitly: the in-place normalization below must never touch the
        # tensor the decode loop is still building.
        sl = decoded[:, :, frame_start:frame_end].detach().float().clone()
        std = getattr(vae, "pixel_std", None)
        mean = getattr(vae, "pixel_mean", None)
        if std is not None and mean is not None:
            sl.mul_(std.to(sl)).add_(mean.to(sl))
        u8 = sl.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
        u8 = u8[0].permute(1, 2, 3, 0).contiguous().cpu()  # (frames, H, W, 3)
        # .cpu() synchronizes this stream's work: the timestamp includes the
        # decode of this chunk AND the host copy a streaming mux would need.
        now = time.monotonic()
        self.chunks.append({"chunk": chunk_index, "frame_start": frame_start, "frame_end": frame_end, "t": now})
        if self.store_frames:
            self.frames.append(u8)

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


_STATE: dict = {"active": False, "recorder": None, "orig_decode": None, "orig_audio_decode": None}


def recorder() -> StreamRecorder | None:
    return _STATE["recorder"]


def _instrumented_decode(self, z: torch.Tensor) -> torch.Tensor:
    """`AutoencoderKLMiniMaxH3._decode` with recorder callbacks.

    The chunk loop is copied verbatim from components/video_autoencoder.py;
    the callbacks only read finalized regions, so the returned tensor is
    bit-identical to the original implementation's.
    """
    rec: StreamRecorder = _STATE["recorder"]
    rec.on_decode_start()

    tokens_chunk_size = self.tokens_chunk_size
    token_drop = self.config.token_drop
    temporal_ratio = self.temporal_compression_ratio
    chunk_num_frames = tokens_chunk_size * temporal_ratio

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
    decoded = None
    write_position = 0
    overlap = None
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
                if decoded is None:
                    decoded = torch.empty(*chunk.shape[:2], output_frames, *chunk.shape[3:],
                                          dtype=chunk.dtype, device=chunk.device)
                copy_frames = min(chunk.shape[2], output_frames - write_position)
                if copy_frames > 0:
                    decoded[:, :, write_position : write_position + copy_frames].copy_(chunk[:, :, :copy_frames])
                    previous_position = write_position
                    write_position += copy_frames
                    if i < num_chunks - 1:
                        rec.on_chunk(self, decoded, previous_position, write_position, i)
            else:
                overlap = chunk.contiguous()
        del clip
    final_start = len(rec.chunks) and rec.chunks[-1]["frame_end"] or 0
    if overlap is not None:
        copy_frames = min(overlap.shape[2], output_frames - write_position)
        if copy_frames > 0:
            decoded[:, :, write_position : write_position + copy_frames].copy_(overlap[:, :, :copy_frames])
            write_position += copy_frames
    if write_position != output_frames:
        raise RuntimeError(f"MiniMax H3 VAE decoded {write_position} frames, expected {output_frames}")
    # The last chunk's frames plus the trailing overlap finalize together.
    rec.on_chunk(self, decoded, final_start, write_position, num_chunks - 1)
    rec.on_decode_end()
    return decoded


def activate(store_frames: bool = True) -> StreamRecorder:
    """Install the instrumented decode paths; returns the shared recorder."""
    if _STATE["active"]:
        _STATE["recorder"].store_frames = store_frames
        return _STATE["recorder"]
    from . import audio_vae as _audio_mod

    rec = StreamRecorder(store_frames=store_frames)
    _STATE["recorder"] = rec
    _STATE["orig_decode"] = _va.AutoencoderKLMiniMaxH3._decode
    _va.AutoencoderKLMiniMaxH3._decode = _instrumented_decode

    orig_audio_decode = _audio_mod.MiniMaxH3AudioVAE.decode

    def timed_audio_decode(self, latents):
        rec.audio_start_t = time.monotonic()
        out = orig_audio_decode(self, latents)
        rec.audio_end_t = time.monotonic()
        rec.audio_out = out.detach().float().cpu() if torch.is_tensor(out) else None
        return out

    _STATE["orig_audio_decode"] = orig_audio_decode
    _audio_mod.MiniMaxH3AudioVAE.decode = timed_audio_decode
    _STATE["active"] = True
    return rec


def deactivate() -> None:
    if not _STATE["active"]:
        return
    from . import audio_vae as _audio_mod

    _va.AutoencoderKLMiniMaxH3._decode = _STATE["orig_decode"]
    _audio_mod.MiniMaxH3AudioVAE.decode = _STATE["orig_audio_decode"]
    _STATE.update({"active": False, "recorder": None, "orig_decode": None, "orig_audio_decode": None})


# -- segment muxing -------------------------------------------------------

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
