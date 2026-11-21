"""CPU tests for plugins/chunk_streaming/stitch.py: the saved video joined from the live segments (real ffmpeg, x264)."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import types

import pytest
import torch

import test_streaming  # noqa: F401  (its module setup: package stubs, then plugins.chunk_streaming.streaming)

streaming = importlib.import_module("plugins.chunk_streaming.streaming")
stitch = importlib.import_module("plugins.chunk_streaming.stitch")

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg")
FRAMES, H, W = 37, 64, 96  # a 17-frame clip, then 20: two segments, as H3 chunks


def video(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, 3, FRAMES, H, W), generator=generator, dtype=torch.uint8)


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A recorder with live segments of `video()`, as a decode leaves it, and a stand-in WanGP whose own save is recorded."""
    rec = streaming.StreamRecorder(store_frames=True, live_dir=str(tmp_path / "live"))
    monkeypatch.setitem(streaming._STATE, "recorder", rec)
    calls = []

    def original_save(tensor, save_file=None, fps=30, codec_type="libx264_8", container="mp4", *args, **kwargs):
        calls.append(save_file)
        return save_file

    wgp = types.SimpleNamespace(server_config={"video_output_codec": "libx264_8", "video_container": "mp4"}, save_video=original_save)
    stitch.install(wgp)
    yield rec, wgp, calls
    stitch.uninstall()


def decode_into_segments(rec, clip):
    rec.reset()
    rec.on_decode_start()
    for index, (start, end) in enumerate(((0, 17), (17, FRAMES))):
        rec.on_chunk(None, clip, start, end, index)
    rec.on_decode_end()


def probe(path, entries):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", f"stream={entries}",
                          "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True, check=True).stdout
    return out.split()


def test_join_replaces_the_second_encode(session, tmp_path):
    rec, wgp, calls = session
    assert rec.segment_video_args == ["-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p"]  # WanGP's output codec, not the preview's
    clip = video()
    decode_into_segments(rec, clip)
    out = wgp.save_video(tensor=clip, save_file=str(tmp_path / "out.mp4"), fps=24, nrow=1, normalize=True, value_range=(-1, 1),
                         codec_type="libx264_8", container="mp4")
    assert calls == [] and out == str(tmp_path / "out.mp4")
    assert stitch.last_save()["joined"] and stitch.last_save()["segments"] == 2
    codec, frames = probe(out, "codec_name,nb_read_frames")
    assert codec == "h264" and int(frames) == FRAMES
    assert subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", out],
                          capture_output=True, text=True).stdout.strip() == ""  # video only: WanGP adds the soundtrack itself


@pytest.mark.parametrize("change", ["other frames", "other rate", "fewer frames", "float frames"])
def test_anything_else_saves_normally(session, tmp_path, change):
    rec, wgp, calls = session
    clip = video()
    decode_into_segments(rec, clip)
    saved, fps = clip, 24
    if change == "other frames":
        saved = clip.clone()
        saved[0, :, -1] = 255 - saved[0, :, -1]  # post-processed last frame
    elif change == "other rate":
        fps = 30
    elif change == "fewer frames":
        saved = clip[:, :, :-3]
    elif change == "float frames":
        saved = clip.float() / 127.5 - 1
    wgp.save_video(tensor=saved, save_file=str(tmp_path / "out.mp4"), fps=fps, codec_type="libx264_8", container="mp4")
    assert calls == [str(tmp_path / "out.mp4")]
    assert not stitch.last_save()["joined"] and stitch.last_save()["reason"]


def test_failed_segment_saves_normally(session, tmp_path):
    rec, wgp, calls = session
    clip = video()
    decode_into_segments(rec, clip)
    results = rec.live.close()
    results[1]["error"] = "RuntimeError('segment mux failed')"
    wgp.save_video(tensor=clip, save_file=str(tmp_path / "out.mp4"), fps=24, codec_type="libx264_8", container="mp4")
    assert calls == [str(tmp_path / "out.mp4")] and "segment 1" in stitch.last_save()["reason"]


def test_without_segments_or_install(tmp_path, monkeypatch):
    monkeypatch.setitem(streaming._STATE, "recorder", None)
    with pytest.raises(RuntimeError):
        stitch.install(types.SimpleNamespace(server_config={}, save_video=lambda *a, **k: None))
    rec = streaming.StreamRecorder(store_frames=True, live_dir=None)
    paths, reason = stitch.covering_segments(rec, video(), 24)
    assert paths is None and reason == "no live segments"


def test_uninstall_restores_the_save(session):
    rec, wgp, calls = session
    stitch.uninstall()
    assert getattr(wgp.save_video, "__wrapped__", None) is None and rec.segment_video_args is None
