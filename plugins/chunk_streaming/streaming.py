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
Sol-H3-VAE's whole-frame decoder (VAE config `sol_int8`, upstream v20) has
its own `_decode` in `MiniMaxH3VideoVAE`, instrumented the same way: it
reports a chunk per native block of 5 latents (17 frames). The pi-H3 DWD
models (models/minimax_h3/pi_h3_dwd.py, where present) decode their frames
without a VAE, inside the transformer's last step, with the same Sol layers:
`pi_h3_dwd.decode` is instrumented the same way.
`_decode` converts every finalized chunk with `_prepare_decoded_chunk` (in
`MiniMaxH3VideoVAE`: ImageNet de-normalization, clamp, and uint8 rounding when
the pipeline asks for it) row of tiles by row of tiles and sends it to a CPU
output buffer through an mmgp `RamWriter` (the copies land without the decode
waiting for them); once every row of a clip has landed there, the recorder
reads its uint8 CPU frames from that buffer, so they match the frames
`MiniMaxH3VideoVAE.decode` returns and the timestamps include the
device-to-host cost. The audio VAE decode is
wrapped for timing and output capture: audio decoding is independent of video
decoding, so a streaming pipeline reorders it first and every video chunk
shifts later by its duration (serial execution makes that shift exact).

The pure-math helpers at the bottom model the player: `no_buffer_start` is the
oracle earliest start with zero rebuffering, `linear_estimate_start` is the
deployable policy (observe the first chunks, fit a linear completion rate, pad
it, commit to a start time) that the bench validates against actual arrivals.
"""

from __future__ import annotations

import collections
import functools
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

from models.minimax_h3.components import video_autoencoder as _va


class StreamRecorder:
    """Collects chunk/audio events from one instrumented generation."""

    def __init__(self, store_frames: bool = True, live_dir: str | None = None,
                 audio_first: bool = False, fps: int = 24, sample_rate: int = 32000,
                 segment_video_args: list[str] | None = None):
        self.store_frames = store_frames
        self.live_dir = live_dir
        #: ffmpeg video encoder arguments of the live segments (None: x264 veryfast CRF 18, a preview)
        self.segment_video_args = segment_video_args
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
        #: True when the live segments waited for the audio instead (pi-H3 DWD decodes the video before the audio is known):
        #: their landing times then include that wait, as with audio first
        self.segments_wait_audio = False

    # -- hooks ------------------------------------------------------------
    def on_decode_start(self) -> None:
        self.decode_start_t = time.monotonic()
        if self.live_dir:
            if self.live is not None:
                self.live.close()
            self.generation_index += 1
            self.live = LiveSegmenter(
                Path(self.live_dir) / f"gen_{self.generation_index:03d}",
                fps=self.fps, sample_rate=self.sample_rate, video_args=self.segment_video_args)
            if self.audio_out is not None:
                self.live.set_audio(self.audio_out)

    def on_chunk(self, vae, decoded: torch.Tensor, frame_start: int, frame_end: int, chunk_index: int) -> None:
        """Called in chunk order; on a GPU decode, from the RamWriter thread that landed the chunk's last frames."""
        if frame_end <= frame_start:
            return
        u8 = _to_uint8_frames(vae, decoded[:, :, frame_start:frame_end])  # (frames, H, W, 3)
        # `decoded` is the CPU buffer `_decode` fills per finalized chunk, called
        # once this chunk's frames have landed there: the timestamp includes the
        # decode of this chunk AND the host copy a streaming mux would need.
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
        if getattr(type(vae), "_prepare_decoded_chunk", None) is _va.AutoencoderKLMiniMaxH3._prepare_decoded_chunk:  # vae None: [-1, 1] (DWD)
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
                "orig_audio_decode": None, "orig_video_decode": None, "orig_sol_decode": None, "orig_dwd_decode": None}


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


class _LandedReports:
    """Recorder reports that wait for their frames to reach the CPU output.

    `_decode` sends its finalized pixels there through an mmgp `RamWriter`,
    which lands the pieces later, in the order they were written, from its own
    thread (at once, on the calling thread, for a blocking copy) and calls each
    piece's `done` callback (`landed_one`) after landing it. A report (`add`) waits
    for every piece written before it; reports run in order, under the lock,
    on whichever thread completes them, and never stop the decode loop for the
    GPU.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.written = 0  # pieces handed to the writer (decode thread only)
        self.landed = 0
        self.waiting: collections.deque = collections.deque()  # (pieces it waits for, report)

    def landed_one(self) -> None:
        with self.lock:
            self.landed += 1
            self._run()

    def add(self, report) -> None:
        with self.lock:
            self.waiting.append((self.written, report))
            self._run()

    def _run(self) -> None:
        while self.waiting and self.waiting[0][0] <= self.landed:
            self.waiting.popleft()[1]()


def _instrumented_decode(self, z: torch.Tensor, uint8_rounding: str | None = None) -> torch.Tensor:
    """`AutoencoderKLMiniMaxH3._decode` with recorder callbacks.

    The body is copied verbatim from components/video_autoencoder.py (upstream
    v20: each clip decoded by rows of tiles (`_decode_clip_rows`), the
    cross-fade frames kept per row, finalized pixels sent to the CPU output
    buffer by `write` through an mmgp `RamWriter`, single-frame decoder and
    single-latent still paths, `PhaseProgress`); the callbacks only read
    finalized regions of that buffer, so the returned tensor is bit-identical
    to the original implementation's. A clip is reported once all its rows are
    written and have landed in the buffer (`_LandedReports`).
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
    height, width = z.shape[-2] * self.spatial_compression_ratio, z.shape[-1] * self.spatial_compression_ratio
    reports = _LandedReports()
    with (PhaseProgress(max(1, num_chunks) * spatial_tiles) as progress, progress.track(self.decoder, lambda inputs: inputs[0].shape[0] // z.shape[0]),
          _va.transfer.RamWriter("decoded video", depth=64, max_bytes=256 << 20) as writer):  # mmgp.transfer, as upstream
        decoded = None
        overlap = None  # the last frames of the previous clip, which cross-fade with the next one: kept in the decoder's precision
        reported = 0  # end of the frame range already handed to the recorder

        def write(pixels, row, position):
            """Finalized pixels (pixel rows from `row` on, frames from output frame `position` on) converted and sent to the CPU
            output by groups of frames (no clip-sized conversion copies); returns the number of frames."""
            nonlocal decoded
            copy_frames = min(pixels.shape[2], output_frames - position)
            for start in range(0, copy_frames, 4):
                part = self._prepare_decoded_chunk(pixels[:, :, start : min(start + 4, copy_frames)], uint8_rounding)
                scale = part.shape[-2] // pixels.shape[-2]  # (2: a decoder that upsamples)
                if decoded is None:
                    decoded = torch.empty(*part.shape[:2], output_frames, height * scale, width * scale, dtype=part.dtype, device="cpu")
                target = decoded[:, :, position + start : position + start + part.shape[2], row * scale : row * scale + part.shape[-2]]
                if num_chunks:
                    # neither the GPU nor this thread waits for the RAM: a wait in the middle of a clip would leave the GPU idle
                    # while the next decoder call is issued
                    reports.written += 1
                    writer.write(target, part, reports.landed_one)
                else:
                    _va.transfer.copy_(target, part)
                del part
            return copy_frames

        if num_chunks == 0:
            # Short videos still need one decode, without an overlapping temporal chunk: a picture or a few frames, gathered on
            # the GPU (the frames that are kept only) and copied at once.
            frames = slice(frame_pre_padding, frame_pre_padding + output_frames)
            written = write(torch.cat([pixels[:, :, frames].clone() for _, pixels in self._decode_clip_rows(z)], dim=-2), 0, 0)
            rec.on_chunk(self, decoded, 0, written, 0)  # a blocking copy: already in the CPU output
            rec.on_decode_end()
            return decoded

        write_position = 0
        for i in range(num_chunks):
            start = i * tokens_chunk_size
            for row, pixels in self._decode_clip_rows(z[:, :, start : start + tokens_chunk_size + self.token_overlap]):
                rows = slice(row, row + pixels.shape[-2])
                chunk = pixels[:, :, self.frame_pre_padding : chunk_num_frames]
                if i > 0 and overlap is not None:
                    chunk = self._blend(overlap[..., rows, :], chunk, self.frame_overlap, dim=-3)
                written = write(chunk, row, write_position)
                if token_drop > 0:  # what the next clip cross-fades with, written over the rows just read
                    chunk = pixels[:, :, chunk_num_frames + self.frame_pre_padding : 2 * chunk_num_frames]
                    if overlap is None:
                        overlap = torch.empty(*chunk.shape[:3], height, width, dtype=chunk.dtype, device=chunk.device)
                    overlap[..., rows, :].copy_(chunk)
                del chunk, pixels
            write_position += written
            if i < num_chunks - 1 and write_position > reported:
                # every row of this clip is written: its frames go to the recorder once they have landed
                reports.add(functools.partial(rec.on_chunk, self, decoded, reported, write_position, i))
                reported = write_position
        if overlap is not None:
            write_position += write(overlap, 0, write_position)
        if write_position != output_frames:
            raise RuntimeError(f"MiniMax H3 VAE decoded {write_position} frames, expected {output_frames}")
        # The last chunk's frames plus the trailing overlap finalize together.
        reports.add(functools.partial(rec.on_chunk, self, decoded, reported, write_position, num_chunks - 1))
    rec.on_decode_end()  # the writer is closed: every report has run
    return decoded


def _instrumented_sol_decode(self, z: torch.Tensor, uint8_rounding: str | None = None) -> torch.Tensor:
    """`MiniMaxH3VideoVAE._decode` with recorder callbacks.

    v20's `MiniMaxH3VideoVAE` overrides `_decode` for Sol-H3-VAE's decoder (VAE config `sol_int8`), which decodes whole frames
    instead of 256-pixel tiles of clips; any other decoder goes on to `AutoencoderKLMiniMaxH3._decode`, which `activate()` replaces
    with `_instrumented_decode`. The Sol body is copied verbatim from models/minimax_h3/video_vae.py (upstream v20): the decoder
    emits the frames of one latent frame after another, each by bands of pixel rows, and `emit` converts every band and sends it
    to the CPU output through an mmgp `RamWriter`. The callbacks only read finalized regions of that output, so the returned tensor
    is bit-identical to upstream's. A chunk is reported at each native block of 5 latents (17 frames: its first latent keeps one
    frame), once the block's last band is written and every piece before it has landed in the buffer (`_LandedReports`); the
    last block, which may be shorter, after the decode. A picture (one latent: five copies, one frame kept, a blocking copy) and
    a batch of several videos report once, at the end.
    """
    from models.minimax_h3 import video_vae as _video_mod
    from models.minimax_h3.components.sol_decoder import BLOCK as SOL_BLOCK, MiniMaxH3SolDecoder3d

    if not isinstance(self.decoder, MiniMaxH3SolDecoder3d):
        return _STATE["orig_sol_decode"](self, z, uint8_rounding)  # upstream's super()._decode: the instrumented H3 path
    rec: StreamRecorder = _STATE["recorder"]
    rec.on_decode_start()
    transfer = _video_mod.transfer
    decoder, batch = self.decoder, z.shape[0]
    if z.shape[2] == 1:  # a lone latent decodes poorly alone (23 to 28 dB to the source, the full VAE 30 to 37): five copies, its first frame kept, as H3's VAE does
        z = z.repeat(1, 1, SOL_BLOCK, 1, 1)
        frames = 1
    else:
        frames = decoder.output_frames(z.shape[2])
    decoded, height = None, z.shape[-2] * decoder.patch_size
    reports = _LandedReports()
    reported, chunk_index = 0, 0  # end of the frame range already handed to the recorder, the next chunk's number
    per_block = batch == 1 and frames > 1
    with PhaseProgress(batch * decoder.steps(z.shape[2])) as progress, transfer.RamWriter("decoded video", depth=64, max_bytes=256 << 20) as writer:
        decoder.progress = progress
        try:
            for item in range(batch):
                position = 0
                latents = 0  # latent frames whose last band is written

                def emit(pixels, row, last, item=item):  # (item bound here: upstream reads the loop variable, called within its iteration)
                    """A band of pixel rows (from row on) of the frames of a latent frame, converted on the GPU and written to the RAM
                    output; last: the frame's last band."""
                    nonlocal decoded, position, latents, reported, chunk_index
                    if position >= frames:
                        return
                    part = self._prepare_decoded_chunk(pixels[:, :, :frames - position], uint8_rounding)
                    if decoded is None:
                        decoded = torch.empty(batch, part.shape[1], frames, height, part.shape[-1], dtype=part.dtype, device="cpu")
                    target = decoded[item:item + 1, :, position:position + part.shape[2], row:row + part.shape[-2]]
                    if frames > 1:
                        reports.written += 1
                        writer.write(target, part, reports.landed_one)
                    else:
                        transfer.copy_(target, part)  # one picture: one transfer
                    if last:
                        position += part.shape[2]
                        latents += 1
                        if per_block and latents % SOL_BLOCK == 0 and reported < position < frames:
                            # a native block is written: its frames go to the recorder once they have landed
                            reports.add(functools.partial(rec.on_chunk, self, decoded, reported, position, chunk_index))
                            reported, chunk_index = position, chunk_index + 1

                decoder(self.post_quant_conv(z[item:item + 1]), emit)
                if position != frames:
                    raise RuntimeError(f"MiniMax H3 Sol VAE decoded {position} frames, expected {frames}")
            if reported < frames:  # the last block, or the whole decode of a picture or a batch
                reports.add(functools.partial(rec.on_chunk, self, decoded, reported, frames, chunk_index))
        finally:
            decoder.progress = None
    rec.on_decode_end()  # the writer is closed: every report has run
    return decoded


def _instrumented_dwd_decode(refiner, features, token_grid, uint8_rounding=None, interrupt=None):
    """`models.minimax_h3.pi_h3_dwd.decode` with recorder callbacks (pi-H3 DWD models: the adapter's pixel head decodes the video
    inside the transformer's last step, without the VAE).

    The body is copied from pi_h3_dwd.decode: its refiner is WanGP's Sol decoder (components/dwd_refiner.py), which emits the
    frames of one latent frame after another by bands of pixel rows, and `emit` converts every band and sends it to the CPU output
    through an mmgp `RamWriter`. As for the Sol VAE, a chunk is reported at each native block of 5 latents (17 frames) once every
    band before it has landed, the last block after the decode; the returned tensor is bit-identical to pi_h3_dwd.decode's.

    The audio: DWD decodes the video before the last step's audio is stepped and decoded, so the audio-first prefetch cannot run.
    Each generation drops the previous one's audio here (the Sol VAE path does it in its video decode wrapper, which DWD never
    calls); with `audio_first`, the live segments wait for this generation's audio, set by the audio decode moments later, instead
    of being muxed silent.
    """
    from models.minimax_h3.components.sol_decoder import BLOCK as SOL_BLOCK
    from models.minimax_h3.video_vae import IMAGENET_MEAN, IMAGENET_STD

    transfer, SOL_PATCH = _va.transfer, _dwd_module().SOL_PATCH
    rec: StreamRecorder = _STATE["recorder"]
    rec.audio_out = rec.audio_cache = rec.audio_start_t = rec.audio_end_t = None
    rec.audio_first_effective = False
    rec.on_decode_start()
    rec.segments_wait_audio = bool(rec.audio_first and rec.live is not None)
    if rec.segments_wait_audio:
        rec.live.wait_for_audio()
    t, h, w = token_grid
    e_h, e_w = refiner.expansion
    frames, height, width = refiner.output_frames(t), h * e_h * SOL_PATCH, w * e_w * SOL_PATCH
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=features.device).view(1, 3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=features.device).view(1, 3, 1, 1, 1)
    decoded = torch.empty(1, 3, frames, height, width, dtype=torch.float32 if uint8_rounding is None else torch.uint8, device="cpu")
    position = latents = reported = chunk_index = 0
    reports = _LandedReports()

    def emit(pixels, row, last):
        nonlocal position, latents, reported, chunk_index
        part = pixels.float().mul_(std).add_(mean).clamp_(0.0, 1.0).mul_(2.0).sub_(1.0)
        if uint8_rounding == "round":
            part = part.add_(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)
        elif uint8_rounding == "truncate":
            part = part.to(torch.float16).sub_(-1).mul_(127.5).clamp_(0, 255).to(torch.uint8)
        reports.written += 1
        writer.write(decoded[:, :, position:position + part.shape[2], row:row + part.shape[-2]], part, reports.landed_one)
        if last:
            position += part.shape[2]
            latents += 1
            if latents % SOL_BLOCK == 0 and reported < position < frames:  # a native block is written: reported once it has landed
                reports.add(functools.partial(rec.on_chunk, None, decoded, reported, position, chunk_index))
                reported, chunk_index = position, chunk_index + 1

    with PhaseProgress(refiner.steps(t)) as progress, transfer.RamWriter("decoded video", depth=64, max_bytes=256 << 20) as writer:
        refiner.progress, refiner.interrupt = progress, interrupt
        try:
            refiner(features, token_grid, emit)
        finally:
            refiner.progress = refiner.interrupt = None
        if position == frames and reported < frames:  # the last block
            reports.add(functools.partial(rec.on_chunk, None, decoded, reported, frames, chunk_index))
    if position != frames:
        raise RuntimeError(f"pi-H3 DWD decoded {position} frames, expected {frames}")
    rec.on_decode_end()  # the writer is closed: every report has run
    return decoded


def activate(store_frames: bool = True, live_dir: str | None = None,
             audio_first: bool = False, segment_video_args: list[str] | None = None) -> StreamRecorder:
    """Install the instrumented decode paths; returns the shared recorder.

    `live_dir` turns on the live segmenter: every finalized chunk is muxed to
    a standalone fMP4 segment in a background thread as it lands, under
    `live_dir/gen_NNN/`. `audio_first` additionally decodes the audio before
    the video decode starts (see `_maybe_prefetch_audio`), so live segments
    carry their audio slice from the first chunk onward. `segment_video_args`
    sets the segments' video encoder (see `mux_segment`); `stitch.install`
    uses the output codec's, so the saved video can be joined from them.
    """
    if _STATE["active"]:
        rec = _STATE["recorder"]
        rec.store_frames = store_frames
        rec.live_dir = live_dir
        rec.audio_first = audio_first
        rec.segment_video_args = segment_video_args
        return rec
    from models.minimax_h3 import audio_vae as _audio_mod
    from models.minimax_h3 import video_vae as _video_mod

    rec = StreamRecorder(store_frames=store_frames, live_dir=live_dir, audio_first=audio_first,
                         segment_video_args=segment_video_args)
    _STATE["recorder"] = rec
    _STATE["orig_decode"] = _va.AutoencoderKLMiniMaxH3._decode
    _va.AutoencoderKLMiniMaxH3._decode = _instrumented_decode
    # Sol-H3-VAE's whole-frame decoder has its own _decode in the WanGP adapter class (v20): instrumented too; other decoders go
    # on from there to the instrumented H3 path above
    _STATE["orig_sol_decode"] = _video_mod.MiniMaxH3VideoVAE.__dict__["_decode"]
    _video_mod.MiniMaxH3VideoVAE._decode = _instrumented_sol_decode

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
        rec.audio_first_effective = rec.segments_wait_audio = False
        if rec.audio_first:
            rec.audio_first_effective = _maybe_prefetch_audio(sys._getframe(1), rec)
        return orig_video_decode(self, latents, *args, **kwargs)

    _STATE["orig_video_decode"] = orig_video_decode
    _video_mod.MiniMaxH3VideoVAE.decode = audio_first_video_decode
    dwd_mod = _dwd_module()
    if dwd_mod is not None:  # the transformer of a pi-H3 DWD model calls pi_h3_dwd.decode through its module at each generation
        _STATE["orig_dwd_decode"] = dwd_mod.decode
        dwd_mod.decode = _instrumented_dwd_decode
    _STATE["active"] = True
    return rec


def _dwd_module():
    """models.minimax_h3.pi_h3_dwd, None in a WanGP without the pi-H3 DWD models."""
    try:
        from models.minimax_h3 import pi_h3_dwd
    except ImportError:
        return None
    return pi_h3_dwd


def deactivate() -> None:
    if not _STATE["active"]:
        return
    from models.minimax_h3 import audio_vae as _audio_mod
    from models.minimax_h3 import video_vae as _video_mod

    rec = _STATE["recorder"]
    if rec is not None and rec.live is not None:
        rec.live.close()
    _va.AutoencoderKLMiniMaxH3._decode = _STATE["orig_decode"]
    _video_mod.MiniMaxH3VideoVAE._decode = _STATE["orig_sol_decode"]
    _audio_mod.MiniMaxH3AudioVAE.decode = _STATE["orig_audio_decode"]
    _video_mod.MiniMaxH3VideoVAE.decode = _STATE["orig_video_decode"]
    if _STATE["orig_dwd_decode"] is not None:
        _dwd_module().decode = _STATE["orig_dwd_decode"]
    _STATE.update({"active": False, "recorder": None, "orig_decode": None,
                   "orig_audio_decode": None, "orig_video_decode": None, "orig_sol_decode": None, "orig_dwd_decode": None})


# -- segment muxing -------------------------------------------------------

class LiveSegmenter:
    """Muxes finalized chunks into fMP4 segments as they land, off the GPU path.

    One instance per generation. `submit` is called from the decode hook with
    each chunk's metadata and uint8 CPU frames; a background thread encodes
    the segment (with its audio slice when `set_audio` has been called, which
    the audio-first hook guarantees happens before the first chunk; after
    `wait_for_audio`, the thread waits for it) and stamps the wall-clock time
    the finished file hit disk. `close` drains the queue and returns the
    per-segment records.
    """

    def __init__(self, out_dir: str | Path, fps: int = 24, sample_rate: int = 32000,
                 video_args: list[str] | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.video_args = video_args
        self.sample_rate = sample_rate
        self.audio: torch.Tensor | None = None
        self._audio_ready = threading.Event()
        self._audio_wait_s = 0.0
        self.results: list[dict] = []
        self._queue: queue.Queue = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_audio(self, audio: torch.Tensor) -> None:
        if audio is not None and audio.dim() == 3:
            audio = audio[0]
        self.audio = audio
        self._audio_ready.set()

    def wait_for_audio(self, timeout_s: float = 60.0) -> None:
        """Mux the segments with the audio `set_audio` gives later (pi-H3 DWD: the video is decoded before the audio is known):
        the background thread waits for it, at most timeout_s, or until `close`."""
        self._audio_wait_s = timeout_s

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
            if self._audio_wait_s and self.audio is None:
                self._audio_ready.wait(self._audio_wait_s)  # off the GPU path: the decode goes on
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
                record["mux_s"] = round(mux_segment(frames, wav, seg_path, fps=self.fps, video_args=self.video_args), 4)
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
            self._audio_ready.set()  # no audio is coming any more: mux what is left as it is
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


PREVIEW_SEGMENT_VIDEO_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]


def mux_segment(frames_u8: torch.Tensor, audio_wav: str | None, out_path: str | Path, fps: int = 24,
                video_args: list[str] | None = None) -> float:
    """Encode one finished chunk as a standalone fragmented MP4 segment.

    `frames_u8` is (frames, H, W, 3) uint8. `video_args` are the ffmpeg video
    encoder arguments (default: x264 veryfast CRF 18, a preview). Returns the
    encode wall seconds, which is the per-segment cost a streaming worker pays
    off the GPU path.
    """
    frames_u8 = frames_u8.contiguous()
    n, height, width, _ = frames_u8.shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-"]
    if audio_wav:
        # No -shortest: AAC frames quantize to 1024 samples, so the audio can
        # land a few ms short of the video and -shortest would drop a frame.
        cmd += ["-i", audio_wav, "-c:a", "aac", "-b:a", "128k"]
    cmd += [*(video_args or PREVIEW_SEGMENT_VIDEO_ARGS),
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
