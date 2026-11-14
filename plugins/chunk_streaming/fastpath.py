# SPDX-License-Identifier: Apache-2.0
"""Opt-in cuts of the fixed per-job overhead of a MiniMax H3 generation (prototypes, off by default).

keep_models_resident: the H3 pipeline unloads its shared components (text encoder, VAEs) before the denoising
(`MiniMaxH3Pipeline._use_transformer`) and the DiT before the decode (`_use_shared_components`), so every job moves the DiT and the
VAEs from RAM back to the GPU (MMGP reloads a model at its first forward). When the GPU has room for all of them, skipping these two
unloads keeps the models in VRAM from one job to the next; MMGP still evicts a model when an allocation needs its memory. The prompt
embeddings are already cached (`text_encoder_cache`), so a repeated prompt does not need the text encoder at all.

Turn it on with CHUNK_STREAMING_KEEP_RESIDENT=1, or `keep_models_resident(True)`.

save_from_segments: with chunk streaming, every chunk of the video is already encoded once (the live fMP4 segments: libx264, CRF 18,
preset veryfast), and WanGP then encodes all the frames again for its saved file (3.5-8 s on a 2x RTX PRO 4500 run). This writes
the saved file's video track by concatenating the segments instead (`-c copy`, about 0.1-0.25 s); WanGP adds its own audio track
afterwards as usual. It only steps in when the segments cover exactly the frames being saved (same count and size, mp4 container,
every segment written); otherwise WanGP encodes as before. The video is the segments' encode, not WanGP's chosen codec.

Turn it on with CHUNK_STREAMING_SAVE_FROM_SEGMENTS=1, or `save_from_segments(True)`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time

_STATE: dict = {"originals": None, "save_video": None, "last_save": None}


def keep_models_resident(enable: bool = True) -> bool:
    """Skip (enable=True) or restore (False) the pipeline's unloads between its stages; returns whether they are skipped."""
    from models.minimax_h3.pipeline import MiniMaxH3Pipeline

    if enable and _STATE["originals"] is None:
        _STATE["originals"] = (MiniMaxH3Pipeline._use_transformer, MiniMaxH3Pipeline._use_shared_components)

        def _use_transformer(self):  # the shared components stay where MMGP put them
            return None

        def _use_shared_components(self):  # the DiT stays where MMGP put it
            return None

        MiniMaxH3Pipeline._use_transformer, MiniMaxH3Pipeline._use_shared_components = _use_transformer, _use_shared_components
    elif not enable and _STATE["originals"] is not None:
        MiniMaxH3Pipeline._use_transformer, MiniMaxH3Pipeline._use_shared_components = _STATE["originals"]
        _STATE["originals"] = None
    return _STATE["originals"] is not None


def _segments_cover(records: list[dict], frames: int) -> bool:
    if not records or any(not r.get("path") or r.get("error") for r in records):
        return False
    position = 0
    for record in sorted(records, key=lambda r: r["frame_start"]):
        if record["frame_start"] != position:
            return False
        position = record["frame_end"]
    return position == frames


def save_video_from_segments(live, tensor, save_file: str, container: str, fingerprint=None) -> str | None:
    """The video track of `tensor` (C, T, H, W) written to `save_file` from the live segments that hold the same frames, or None.
    `fingerprint` (streaming.output_fingerprint of the decode the segments came from) must match the frames being saved: a video
    changed after the decode (film grain, a filter) is encoded by WanGP as usual."""
    from .streaming import matches_fingerprint

    if live is None or container != "mp4" or save_file is None or not hasattr(tensor, "shape") or tensor.dim() not in (4, 5):
        return None
    frames, height, width = tensor.shape[-3], tensor.shape[-2], tensor.shape[-1]
    if not matches_fingerprint(fingerprint, tensor):
        return None
    records = live.close()  # every chunk of this generation was finalized before WanGP saves: drain the muxing thread
    if getattr(live, "frame_size", None) != (height, width) or not _segments_cover(records, frames):
        return None
    out = save_file if save_file.endswith(".mp4") else os.path.splitext(save_file)[0] + ".mp4"
    started = time.monotonic()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as listing:
        for record in sorted(records, key=lambda r: r["frame_start"]):
            listing.write(f"file '{os.path.abspath(record['path'])}'\n")
    try:
        proc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing.name, "-map", "0:v:0",
                               "-c", "copy", "-movflags", "+faststart", out], capture_output=True, text=True)
    finally:
        os.unlink(listing.name)
    if proc.returncode != 0:
        return None
    _STATE["last_save"] = {"path": out, "seconds": round(time.monotonic() - started, 3), "segments": len(records)}
    return out


def save_from_segments(enable: bool = True, wgp=None) -> bool:
    """Write WanGP's saved video from the streamed segments when they hold the same frames (enable=True), or restore WanGP's
    encode (False); returns whether it is on. `wgp`: WanGP's module (default: the loaded `wgp`)."""
    wgp = wgp if wgp is not None else _STATE.get("wgp") or sys.modules.get("wgp")
    if wgp is None or not hasattr(wgp, "save_video"):
        return False
    if enable and _STATE["save_video"] is None:
        original = wgp.save_video

        def save_video(tensor, save_file=None, fps=30, codec_type="libx264_8", container="mp4", *args, **kwargs):
            from . import streaming

            rec = streaming.recorder()
            try:
                path = None if rec is None else save_video_from_segments(rec.live, tensor, save_file, container, rec.fingerprint)
            except Exception:  # noqa: BLE001 - the prototype must never cost a saved video
                path = None
            if path is not None:
                return path
            return original(tensor, save_file, fps, codec_type, container, *args, **kwargs)

        _STATE["save_video"], _STATE["wgp"] = original, wgp
        wgp.save_video = save_video
    elif not enable and _STATE["save_video"] is not None:
        wgp.save_video = _STATE["save_video"]
        _STATE["save_video"] = None
    return _STATE["save_video"] is not None


def last_save() -> dict | None:
    """Path, seconds and segment count of the last video saved from segments."""
    return _STATE["last_save"]


def apply_environment() -> dict:
    """Turn on the shortcuts their environment variables ask for; returns which are on."""
    return {"keep_models_resident": keep_models_resident(True) if os.environ.get("CHUNK_STREAMING_KEEP_RESIDENT") == "1" else False,
            "save_from_segments": save_from_segments(True) if os.environ.get("CHUNK_STREAMING_SAVE_FROM_SEGMENTS") == "1" else False}
