# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""CPU tests of the NVENC codecs (nvenc.py) on WanGP's own shared/utils/video_codecs.py, with stand-in ffmpeg executables:

    python -m pytest plugins/speedups/tests -q
"""

import importlib
import importlib.util
import os
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "plugins"))

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="stand-in ffmpeg executables are shell scripts")


@pytest.fixture
def codecs(monkeypatch):
    """A fresh copy of WanGP's video_codecs and of nvenc.py (its check is cached per process), installed, with no ffmpeg settings
    inherited from the environment."""
    for name in ("IMAGEIO_FFMPEG_EXE", "FFMPEG_BINARY"):
        monkeypatch.delenv(name, raising=False)
    # inside WanGP's shared.utils package: since v20.12 the module imports its sibling video_decode where it encodes
    spec = importlib.util.spec_from_file_location("shared.utils.video_codecs_under_test", ROOT / "shared" / "utils" / "video_codecs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "set_hardware_video_encoding"):  # v20.12's own NVENC (Hardware Video Encoding) off: the plugin's codecs alone
        module.set_hardware_video_encoding("off")
    nvenc = importlib.reload(importlib.import_module("speedups.nvenc"))
    nvenc.install(module)
    return module, nvenc


def fake_ffmpeg(folder, works=True):
    """An executable that records each call and succeeds (an NVENC encode works) or fails like an ffmpeg without the driver's video
    library."""
    folder.mkdir(parents=True, exist_ok=True)
    path, log = folder / "ffmpeg", folder / "ffmpeg.calls"
    body = "exit 0" if works else 'echo "[h264_nvenc @ 0x1] Cannot load libnvidia-encode.so.1" >&2; exit 1'
    path.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\n{body}\n')
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path, log


def calls(log):
    return log.read_text().splitlines() if log.exists() else []


def test_choices_listed_before_the_editing_codecs(codecs):
    module, nvenc = codecs
    keys = [key for _, key in module.SDR_VIDEO_CODEC_CHOICES]
    assert keys.index("h264_nvenc") < keys.index("hevc_nvenc") < keys.index("prores_422")
    nvenc.install(module)  # twice: listed once, wrapped once
    assert [key for _, key in module.SDR_VIDEO_CODEC_CHOICES].count("h264_nvenc") == 1
    assert not hasattr(module._get_video_codec_spec.__wrapped__, "_speedups_nvenc")


def test_nvenc_where_it_works(codecs, tmp_path, monkeypatch):
    module, _ = codecs
    ffmpeg, log = fake_ffmpeg(tmp_path / "bin")
    monkeypatch.setenv("PATH", f"{ffmpeg.parent}{os.pathsep}{os.environ['PATH']}")
    assert module.get_video_encode_args("h264_nvenc", "mp4") == ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "14",
                                                                 "-b:v", "0", "-pix_fmt", "yuv420p"]
    params = module.get_imageio_codec_params("hevc_nvenc", "mkv")
    assert params["codec"] == "hevc_nvenc" and "-cq" in params["output_params"]
    assert os.environ["IMAGEIO_FFMPEG_EXE"] == str(ffmpeg)  # ImageIO writes with the ffmpeg that passed, not its own build
    assert len(calls(log)) == 2  # one check per encoder
    module.get_video_encode_args("h264_nvenc", "mov")
    assert len(calls(log)) == 2  # cached
    assert module.validate_video_output_settings("h264_nvenc", "mp4", "aac_128") is None


def test_x264_crf_10_where_it_does_not(codecs, tmp_path, monkeypatch, capsys):
    module, _ = codecs
    ffmpeg, _ = fake_ffmpeg(tmp_path / "bin", works=False)
    monkeypatch.setenv("PATH", f"{ffmpeg.parent}{os.pathsep}{os.environ['PATH']}")
    assert module.get_video_encode_args("h264_nvenc", "mp4") == ["-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p"]
    assert "Cannot load libnvidia-encode.so.1" in capsys.readouterr().out
    assert "IMAGEIO_FFMPEG_EXE" not in os.environ


def test_an_explicit_imageio_ffmpeg_is_checked_and_kept(codecs, tmp_path, monkeypatch):
    module, _ = codecs
    path_ffmpeg, _ = fake_ffmpeg(tmp_path / "path")
    imageio_ffmpeg, imageio_log = fake_ffmpeg(tmp_path / "imageio", works=False)
    monkeypatch.setenv("PATH", f"{path_ffmpeg.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", str(imageio_ffmpeg))
    assert module.get_video_encode_args("h264_nvenc", "mp4")[1] == "libx264"  # ImageIO's own ffmpeg cannot: x264
    assert len(calls(imageio_log)) == 1 and os.environ["IMAGEIO_FFMPEG_EXE"] == str(imageio_ffmpeg)


def test_other_codecs_unchanged_and_uninstall(codecs, tmp_path, monkeypatch):
    module, nvenc = codecs
    ffmpeg, log = fake_ffmpeg(tmp_path / "bin")
    monkeypatch.setenv("PATH", f"{ffmpeg.parent}{os.pathsep}{os.environ['PATH']}")
    assert module.get_video_encode_args("libx264_8", "mp4") == ["-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p"]
    assert calls(log) == []  # no check for other codecs
    nvenc.uninstall(module)
    assert "h264_nvenc" not in [key for _, key in module.SDR_VIDEO_CODEC_CHOICES]
    assert module.get_video_encode_args("h264_nvenc", "mp4")[1] == "libx264"  # WanGP's fallback for a codec it does not know


def test_the_wrapper_takes_the_arguments_of_v20_12(codecs, tmp_path, monkeypatch):
    """v20.12 calls _get_video_codec_spec(codec_key, container, hardware, ffmpeg_exe): the wrapper passes them on to WanGP's."""
    module, _ = codecs
    ffmpeg, _ = fake_ffmpeg(tmp_path / "bin")
    monkeypatch.setenv("PATH", f"{ffmpeg.parent}{os.pathsep}{os.environ['PATH']}")
    assert module._get_video_codec_spec("libx264_8", "mp4", False, str(ffmpeg))[0] == "libx264"
    assert module._get_video_codec_spec("h264_nvenc", "mp4", None, str(ffmpeg))[0] == "h264_nvenc"
