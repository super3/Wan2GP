# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""CPU tests of the NVENC video codecs (shared/utils/video_codecs.py), with stand-in ffmpeg executables:

    python -m pytest shared/utils/tests/test_video_codecs_nvenc.py -q
"""

import importlib.util
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "video_codecs.py"


@pytest.fixture
def codecs(monkeypatch):
    """A fresh copy of the module (its probe cache is per process), with no ffmpeg settings inherited from the environment."""
    for name in ("IMAGEIO_FFMPEG_EXE", "FFMPEG_BINARY"):
        monkeypatch.delenv(name, raising=False)
    spec = importlib.util.spec_from_file_location("video_codecs_under_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_ffmpeg(folder, name="ffmpeg", works=True):
    """An executable that records each call and succeeds (an NVENC encode works) or fails like an ffmpeg without the driver's
    video library."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    log = folder / f"{name}.calls"
    body = "exit 0" if works else 'echo "[h264_nvenc @ 0x1] Cannot load libnvidia-encode.so.1" >&2; exit 1'
    path.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\n{body}\n')
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path, log


def calls(log):
    return log.read_text().splitlines() if log.exists() else []


pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="stand-in ffmpeg executables are shell scripts")


def test_nvenc_where_it_works(codecs, tmp_path, monkeypatch):
    ffmpeg, log = fake_ffmpeg(tmp_path / "bin")
    monkeypatch.setenv("PATH", f"{ffmpeg.parent}{os.pathsep}{os.environ['PATH']}")
    assert codecs.get_video_encode_args("h264_nvenc", "mp4") == ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "14",
                                                                 "-b:v", "0", "-pix_fmt", "yuv420p"]
    params = codecs.get_imageio_codec_params("hevc_nvenc", "mkv")
    assert params["codec"] == "hevc_nvenc" and params["pixelformat"] == "yuv420p" and "-cq" in params["output_params"]
    assert os.environ["IMAGEIO_FFMPEG_EXE"] == str(ffmpeg)  # ImageIO writes with the ffmpeg that passed, not its own build
    probes = calls(log)
    assert len(probes) == 2 and "h264_nvenc" in probes[0] and "hevc_nvenc" in probes[1]  # one probe per encoder
    codecs.get_video_encode_args("h264_nvenc", "mov")
    assert len(calls(log)) == 2  # cached


def test_x264_fallback_where_nvenc_fails(codecs, tmp_path, monkeypatch, capsys):
    ffmpeg, log = fake_ffmpeg(tmp_path / "bin", works=False)
    monkeypatch.setenv("PATH", f"{ffmpeg.parent}{os.pathsep}{os.environ['PATH']}")
    assert codecs.get_video_encode_args("h264_nvenc", "mp4") == ["-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p"]
    assert codecs.get_imageio_codec_params("h264_nvenc", "mp4")["codec"] == "libx264"
    assert "IMAGEIO_FFMPEG_EXE" not in os.environ
    out = capsys.readouterr().out
    assert out.count("h264_nvenc is not available here") == 1 and "libnvidia-encode" in out  # one warning, with the reason
    assert len(calls(log)) == 1


def test_no_ffmpeg_on_path(codecs, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert codecs.get_video_encode_args("h264_nvenc", "mp4")[1] == "libx264"
    assert "no ffmpeg on PATH" in capsys.readouterr().out


def test_ffmpeg_binary_setting_and_an_explicit_imageio_ffmpeg(codecs, tmp_path, monkeypatch):
    path_ffmpeg, path_log = fake_ffmpeg(tmp_path / "path")
    chosen, chosen_log = fake_ffmpeg(tmp_path / "chosen", name="my-ffmpeg")
    imageio_choice, imageio_log = fake_ffmpeg(tmp_path / "imageio", name="imageio-ffmpeg", works=False)
    monkeypatch.setenv("PATH", f"{path_ffmpeg.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FFMPEG_BINARY", str(chosen))  # WanGP's subprocess encoders use it over PATH
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", str(imageio_choice))  # the user's own choice for ImageIO: kept, and probed too
    assert codecs.get_video_encode_args("h264_nvenc", "mp4")[1] == "libx264"
    assert os.environ["IMAGEIO_FFMPEG_EXE"] == str(imageio_choice)
    assert len(calls(chosen_log)) == 1 and len(calls(imageio_log)) == 1 and calls(path_log) == []


def test_other_codecs_unchanged(codecs, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))  # no probe: an ffmpeg call would fail the NVENC test, not these
    assert codecs.get_video_encode_args("libx264_8", "mp4") == ["-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p"]
    assert codecs.get_video_encode_args(None, None)[1] == "libx264"
    assert codecs.get_video_encode_args("libx265_18", "mkv")[1] == "libx265"
    assert codecs._nvenc_ready == {}
    keys = [key for _, key in codecs.SDR_VIDEO_CODEC_CHOICES]
    assert "h264_nvenc" in keys and "hevc_nvenc" in keys and len(keys) == len(set(keys))
    assert codecs.validate_video_output_settings("h264_nvenc", "mp4", "aac_128") is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg installed")
def test_real_ffmpeg_never_breaks_saving(codecs, tmp_path, monkeypatch):
    """With the real ffmpeg, NVENC is used only if it encodes here; on a machine without an NVIDIA GPU it falls back to x264."""
    codec = codecs.get_video_encode_args("h264_nvenc", "mp4")[1]
    assert codec in ("h264_nvenc", "libx264")
    assert (codec == "h264_nvenc") == codecs._nvenc_ready["h264_nvenc"]
