"""Real-time output: the saved video joined from the generation's live segments instead of encoded again after the decode.

With the live segmenter on (`streaming.activate(live_dir=...)`), every finalized chunk is encoded in a background thread while the
rest of the clip decodes. WanGP then encodes the whole clip once more in `save_video`, after the decode, on the job's critical path
(x264 CRF 10: about 2-6 s per 480p 5-10 s clip on a PRO 6000 host). `install(wgp)` makes the segments use WanGP's output codec
(`server_config["video_output_codec"]`, NVENC included, through `shared.utils.video_codecs`) and replaces `wgp.save_video`: when the
segments cover exactly the frames being saved, the video is their join with a stream copy (no encode), and WanGP adds the
soundtrack afterwards as usual. Anything else (other frames, sizes or rates, post-processing, sliding windows, images) is saved
the normal way.

The frames check: same frame count, size and rate, and the first and last frames identical to the recorded ones (the recorder must
store frames: `activate(store_frames=True)`)."""

from __future__ import annotations

import functools
import os
import subprocess
import tempfile
import time

import torch

if __package__:
    from . import streaming
else:  # loaded as a top-level module (scripts with plugins/chunk_streaming on sys.path)
    import streaming

_STATE: dict = {"wgp": None, "original": None, "last": None}


def segment_video_args(codec_key: str | None, container: str | None = "mp4") -> list[str]:
    """ffmpeg video encoder arguments of the live segments: WanGP's for its output codec, so their join is the saved video."""
    from shared.utils.video_codecs import get_video_encode_args

    return get_video_encode_args(codec_key, container)


def _frame(video, index: int) -> torch.Tensor | None:
    """Frame `index` of a video given to save_video as uint8 (H, W, 3), or None when it is not uint8 frames."""
    if torch.is_tensor(video):
        if video.dtype != torch.uint8 or video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
            return None
        return video[0, :, index].permute(1, 2, 0)
    read = getattr(video, "read", None)
    if read is not None and hasattr(video, "frames_count"):
        frames = read(index, 1)
        if torch.is_tensor(frames) and frames.dtype == torch.uint8 and frames.ndim == 4 and frames.shape[-1] == 3:
            return frames[0]
    return None


def _frame_count(video) -> int | None:
    if torch.is_tensor(video) and video.ndim == 5:
        return int(video.shape[2])
    return int(video.frames_count) if hasattr(video, "frames_count") else None


def covering_segments(rec, video, fps) -> tuple[list[str] | None, str]:
    """The live segment files whose frames are exactly `video`'s, in order, or None and why not."""
    if rec is None or rec.live is None:
        return None, "no live segments"
    results = sorted(rec.live.close(), key=lambda record: record["segment"])  # waits for the last segments
    frames = _frame_count(video)
    if frames is None or not results:
        return None, "nothing to compare"
    if round(float(fps), 6) != round(float(rec.fps), 6):
        return None, f"saved at {fps} fps, segments at {rec.fps}"
    position = 0
    for record in results:
        if record.get("error") or not record.get("path") or record["frame_start"] != position:
            return None, f"segment {record['segment']}: {record.get('error') or 'missing or out of order'}"
        position = record["frame_end"]
    if position != frames:
        return None, f"segments hold {position} frames, the video {frames}"
    if not rec.frames:
        return None, "the recorder keeps no frames to compare (store_frames=False)"
    first, last = _frame(video, 0), _frame(video, frames - 1)
    if first is None or last is None:
        return None, "the video is not uint8 frames"
    if first.shape != rec.frames[0][0].shape or not torch.equal(first.cpu(), rec.frames[0][0]) or not torch.equal(last.cpu(), rec.frames[-1][-1]):
        return None, "its frames differ from the decoded ones (post-processing)"
    return [record["path"] for record in results], ""


def join_segments(paths: list[str], out_path: str) -> None:
    """The segments' video streams one after the other in `out_path`, copied (no encode)."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as listing:
        for path in paths:
            listing.write("file '" + os.path.abspath(path).replace("'", "'\\''") + "'\n")
    try:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listing.name,
               "-map", "0:v:0", "-c", "copy", "-movflags", "+faststart", out_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"joining segments failed: {proc.stderr[-400:]}")
    finally:
        os.unlink(listing.name)


def _stitched(original):
    @functools.wraps(original)
    def save_video(tensor=None, save_file=None, fps=30, codec_type="libx264_8", container="mp4", *args, **kwargs):
        started = time.monotonic()
        paths, reason = None, "no file name"
        if save_file is not None:
            try:
                paths, reason = covering_segments(streaming.recorder(), tensor, fps)
            except Exception as error:  # noqa: BLE001 - a check that fails must never fail the save
                paths, reason = None, f"check failed: {error!r}"[:200]
        if paths:
            suffix = f".{container or 'mp4'}"
            out = save_file if save_file.endswith(suffix) else os.path.splitext(save_file)[0] + suffix
            try:
                join_segments(paths, out)
                _STATE["last"] = {"joined": True, "segments": len(paths), "seconds": round(time.monotonic() - started, 3)}
                return out
            except Exception as error:  # noqa: BLE001 - fall back to the normal save
                reason = repr(error)[:200]
        result = original(tensor, save_file, fps, codec_type, container, *args, **kwargs)
        _STATE["last"] = {"joined": False, "reason": reason, "seconds": round(time.monotonic() - started, 3)}
        return result

    save_video.__wrapped__ = original
    return save_video


def install(wgp) -> None:
    """Join `wgp`'s saved videos from the live segments when they match (see the module docstring). The segments then use WanGP's
    output codec: call after `streaming.activate(live_dir=..., store_frames=True)`, and again if the codec setting changes."""
    rec = streaming.recorder()
    if rec is None:
        raise RuntimeError("stitch.install needs streaming.activate(live_dir=..., store_frames=True) first")
    codec = wgp.server_config.get("video_output_codec", None)
    rec.segment_video_args = segment_video_args(codec, wgp.server_config.get("video_container", "mp4"))
    if _STATE["wgp"] is None:
        _STATE["wgp"], _STATE["original"] = wgp, wgp.save_video
        wgp.save_video = _stitched(wgp.save_video)


def uninstall() -> None:
    if _STATE["wgp"] is not None:
        _STATE["wgp"].save_video = _STATE["original"]
        rec = streaming.recorder()
        if rec is not None:
            rec.segment_video_args = None
    _STATE.update({"wgp": None, "original": None, "last": None})


def last_save() -> dict | None:
    """How the last save went: {"joined": True, "segments", "seconds"} or {"joined": False, "reason", "seconds"}."""
    return _STATE["last"]
