# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""CPU tests for plugins/chunk_streaming/streaming.py.

Four properties are load-bearing enough to gate a GPU bench on:
1. The instrumented `_decode` returns a tensor bit-identical to the original
   and its finalized-frame events tile the output exactly (tiny CPU VAE).
2. The segment muxer produces playable fMP4 segments whose ffprobe frame
   counts match the finalized ranges.
3. The player schedule math: the oracle start never rebuffers, and the linear
   estimator flags exactly the arrival patterns that would stall.
4. The multi-device decode (chunks split across devices, here CPU slots on
   worker threads) is bit-identical to the single-device decode, finalizes and
   reports chunks in order whatever order they finish in, and propagates errors.
"""

from __future__ import annotations

import functools
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

_ROOT = str(Path(__file__).resolve().parents[3])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# models/minimax_h3/__init__.py drags in mmgp, which asserts on CPU-only
# torch at import time. streaming.py needs none of that, so register the
# package path without executing __init__ and import the submodules directly.
# The VAE now imports real `shared` submodules (shared.utils.phase_progress,
# shared.kernels), so `shared` is registered the same way instead of as an
# empty module.
import importlib
import types

for name, rel in (("models", "models"), ("models.minimax_h3", "models/minimax_h3"), ("shared", "shared"),
                  ("plugins", "plugins"), ("plugins.chunk_streaming", "plugins/chunk_streaming")):
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(Path(_ROOT) / rel)]
        sys.modules[name] = pkg

# shared/attention.py probes CUDA at import (get_device_capability), which
# raises on a CPU-only runner. Both the original and instrumented decode go
# through the same stub, so the bit-identity assertion is unaffected by it.
if "shared.attention" not in sys.modules:
    import torch.nn.functional as F

    shared_pkg = sys.modules.setdefault("shared", types.ModuleType("shared"))
    attn_mod = types.ModuleType("shared.attention")

    def _sdpa_pay_attention(qkv_list, causal=False, **_kw):
        q, k, v = qkv_list  # (B, L, H, D)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal)
        return out.transpose(1, 2)

    # shared/attention_kit.py (imported lazily by the VAE's CPU norm path for
    # rows_) also imports these two names. No SageAttention on CPU: staged
    # quantization is unavailable (None, as the real helper reports), and the
    # VAE never builds a Sage attention mask.
    def _no_sage_staged(*_args, **_kw):
        return None

    def _sage_attention_mask(*_args, **_kw):
        raise NotImplementedError("Sage attention masks are not stubbed in plugins/chunk_streaming/tests/test_streaming.py")

    attn_mod.pay_attention = _sdpa_pay_attention
    attn_mod.sage_staged = _no_sage_staged
    attn_mod.sage_attention_mask = _sage_attention_mask
    sys.modules["shared.attention"] = attn_mod
    shared_pkg.attention = attn_mod

# models/minimax_h3/video_vae.py (imported by streaming.activate) pulls
# shared.utils.utils for guide_to_float, and that module imports cv2, rembg,
# torchvision and more at import time. Only the encode path calls it, which
# these tests never exercise, so stand in for it when those are not installed.
if "shared.utils.utils" not in sys.modules:
    try:
        importlib.import_module("shared.utils.utils")
    except ImportError:
        utils_mod = types.ModuleType("shared.utils.utils")

        def _guide_to_float(_video):
            raise RuntimeError("shared.utils.utils is stubbed by plugins/chunk_streaming/tests/test_streaming.py")

        utils_mod.guide_to_float = _guide_to_float
        sys.modules["shared.utils.utils"] = utils_mod

streaming = importlib.import_module("plugins.chunk_streaming.streaming")
AutoencoderKLMiniMaxH3 = importlib.import_module(
    "models.minimax_h3.components.video_autoencoder"
).AutoencoderKLMiniMaxH3


def _tiny_vae() -> AutoencoderKLMiniMaxH3:
    torch.manual_seed(0)
    return AutoencoderKLMiniMaxH3(
        latent_channels=4,
        block_out_channels=(8, 8, 8, 8, 8, 8),
        layers_per_block=1,
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=1,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=1,
        decoder_ffn_mult=1,
    ).eval()


def _tiny_adapter_vae() -> AutoencoderKLMiniMaxH3:
    """`_tiny_vae` with the WanGP adapter's finalized-chunk conversion.

    `MiniMaxH3VideoVAE._prepare_decoded_chunk` (ImageNet de-normalization,
    [-1, 1] or uint8 rounding) is what the production decode stores in its
    CPU output buffer; the full adapter is too large to build on CPU.
    """
    video_vae = importlib.import_module("models.minimax_h3.video_vae")

    class TinyAdapterVAE(AutoencoderKLMiniMaxH3):
        upsampling = False
        _prepare_decoded_chunk = video_vae.MiniMaxH3VideoVAE._prepare_decoded_chunk

    torch.manual_seed(0)
    vae = TinyAdapterVAE(
        latent_channels=4,
        block_out_channels=(8, 8, 8, 8, 8, 8),
        layers_per_block=1,
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=1,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=1,
        decoder_ffn_mult=1,
    ).eval()
    vae.register_buffer("pixel_mean", torch.tensor(video_vae.IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
    vae.register_buffer("pixel_std", torch.tensor(video_vae.IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)
    return vae


@pytest.mark.parametrize("z_tokens", [5, 12, 37])
def test_instrumented_decode_bit_identical(z_tokens):
    vae = _tiny_vae()
    z = torch.randn(1, 4, z_tokens, 8, 8)
    with torch.no_grad():
        expected = AutoencoderKLMiniMaxH3._decode(vae, z)
    rec = streaming.activate(store_frames=True)
    try:
        rec.reset()
        with torch.no_grad():
            got = AutoencoderKLMiniMaxH3._decode(vae, z)
    finally:
        streaming.deactivate()
    assert torch.equal(expected, got), "instrumented decode diverged from original"
    # events tile [0, output_frames) exactly, in order, without gaps
    assert rec.chunks, "no chunk events recorded"
    assert rec.chunks[0]["frame_start"] == 0
    for prev, cur in zip(rec.chunks, rec.chunks[1:]):
        assert cur["frame_start"] == prev["frame_end"]
    assert rec.chunks[-1]["frame_end"] == expected.shape[2]
    # stored frames align with the ranges and are uint8 (frames, H, W, 3)
    for meta, frames in zip(rec.chunks, rec.frames):
        assert frames.dtype == torch.uint8
        assert frames.shape[0] == meta["frame_end"] - meta["frame_start"]
        assert frames.shape[3] == 3
    # timestamps are monotonic
    ts = [c["t"] for c in rec.chunks]
    assert ts == sorted(ts)


def test_deactivate_restores_original():
    original = AutoencoderKLMiniMaxH3._decode
    streaming.activate()
    assert AutoencoderKLMiniMaxH3._decode is not original
    streaming.deactivate()
    assert AutoencoderKLMiniMaxH3._decode is original


@pytest.mark.parametrize("uint8_rounding", ["truncate", "round", None])
def test_instrumented_decode_adapter_frames_match_output(uint8_rounding):
    """The production path: `decode(z, ..., uint8_rounding)` forwards the
    rounding positionally to `_decode`, and the recorder's uint8 frames are
    exactly the frames of the (converted) decode output."""
    vae = _tiny_adapter_vae()
    z = torch.randn(1, 4, 12, 8, 8)
    with torch.no_grad():
        expected = AutoencoderKLMiniMaxH3.decode(vae, z, False, uint8_rounding)[0]
    rec = streaming.activate(store_frames=True)
    try:
        rec.reset()
        with torch.no_grad():
            got = AutoencoderKLMiniMaxH3.decode(vae, z, False, uint8_rounding)[0]
    finally:
        streaming.deactivate()
    assert torch.equal(expected, got), "instrumented decode diverged from original"
    assert expected.dtype == (torch.float32 if uint8_rounding is None else torch.uint8)
    assert len(rec.chunks) > 1 and rec.chunks[-1]["frame_end"] == expected.shape[2]
    if uint8_rounding is None:
        reference = expected.clamp(-1.0, 1.0).add(1.0).mul(127.5).round().to(torch.uint8)
    else:
        reference = expected
    assert torch.equal(torch.cat(rec.frames), reference[0].permute(1, 2, 3, 0))


def test_instrumented_decode_single_latent_still():
    vae = _tiny_vae()
    z = torch.randn(1, 4, 1, 8, 8)
    with torch.no_grad():
        expected = AutoencoderKLMiniMaxH3._decode(vae, z)
    rec = streaming.activate(store_frames=True)
    try:
        rec.reset()
        with torch.no_grad():
            got = AutoencoderKLMiniMaxH3._decode(vae, z)
    finally:
        streaming.deactivate()
    assert torch.equal(expected, got)
    assert expected.shape[2] == 1
    assert [(c["frame_start"], c["frame_end"]) for c in rec.chunks] == [(0, 1)]
    assert rec.frames[0].shape[0] == 1


def test_instrumented_decode_single_frame_decoder():
    """H3 image models build the VAE with a single-frame decoder
    (decoder_patch_size_t=1): one latent frame decodes to one pixel frame,
    with no still-image repeat and no pre-padding."""
    torch.manual_seed(0)
    vae = AutoencoderKLMiniMaxH3(
        latent_channels=4,
        block_out_channels=(8, 8, 8, 8, 8, 8),
        layers_per_block=1,
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=1,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=1,
        decoder_ffn_mult=1,
        decoder_patch_size_t=1,
    ).eval()
    z = torch.randn(1, 4, 1, 8, 8)
    with torch.no_grad():
        expected = AutoencoderKLMiniMaxH3._decode(vae, z)
    rec = streaming.activate(store_frames=True)
    try:
        rec.reset()
        with torch.no_grad():
            got = AutoencoderKLMiniMaxH3._decode(vae, z)
    finally:
        streaming.deactivate()
    assert expected.shape[2] == 1
    assert torch.equal(expected, got), "instrumented decode diverged from original"
    assert [(c["frame_start"], c["frame_end"]) for c in rec.chunks] == [(0, 1)]


def test_audio_first_video_decode_forwards_arguments(monkeypatch):
    """The audio-first wrapper sits on `MiniMaxH3VideoVAE.decode`, which the
    pipeline calls as `self.vae.decode(video, uint8_rounding)`."""
    video_vae = importlib.import_module("models.minimax_h3.video_vae")
    calls = []

    def fake_video_decode(self, latents, uint8_rounding=None):
        calls.append(("video", uint8_rounding))
        return latents

    monkeypatch.setattr(video_vae.MiniMaxH3VideoVAE, "decode", fake_video_decode)

    class DummyAudioVAE:
        def decode(self, latents):
            calls.append(("audio",))
            return latents

    class DummyPipe:
        audio_vae = DummyAudioVAE()

        def generate(self, video, audio, image_outputs=False):
            uint8_rounding = "truncate"
            return video_vae.MiniMaxH3VideoVAE.decode(None, video, uint8_rounding)

    rec = streaming.activate(store_frames=False, audio_first=True)
    try:
        rec.reset()
        DummyPipe().generate(torch.zeros(1), torch.zeros(2))
        assert calls == [("audio",), ("video", "truncate")]
        assert rec.audio_first_effective
        # image outputs never decode their audio: no prefetch
        calls.clear()
        rec.reset()
        DummyPipe().generate(torch.zeros(1), torch.zeros(2), image_outputs=True)
        assert calls == [("video", "truncate")]
        assert not rec.audio_first_effective
    finally:
        streaming.deactivate()
    assert video_vae.MiniMaxH3VideoVAE.decode is fake_video_decode


def _ffprobe_frames(path: str) -> tuple[int, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,duration", "-of", "json", path],
        capture_output=True, check=True)
    data = json.loads(out.stdout)["streams"][0]
    return int(data["nb_read_frames"]), float(data.get("duration") or 0.0)


def test_mux_segments_playable(tmp_path):
    rec = streaming.StreamRecorder(store_frames=True)
    fps, frames_per_chunk = 24, 17
    for i in range(3):
        start, end = i * frames_per_chunk, (i + 1) * frames_per_chunk
        rec.chunks.append({"chunk": i, "frame_start": start, "frame_end": end, "t": float(i)})
        ramp = torch.linspace(0, 255, frames_per_chunk).view(-1, 1, 1, 1)
        rec.frames.append(ramp.expand(frames_per_chunk, 48, 64, 3).to(torch.uint8).contiguous())
    t = torch.linspace(0, 2 * math.pi * 220 * (3 * frames_per_chunk / fps), int(32000 * 3 * frames_per_chunk / fps))
    rec.audio_out = torch.sin(t).repeat(2, 1) * 0.3

    segments = streaming.mux_all_segments(rec, tmp_path, fps=fps, sample_rate=32000)
    assert len(segments) == 3
    for seg in segments:
        n, _dur = _ffprobe_frames(seg["path"])
        assert n == seg["frame_end"] - seg["frame_start"]
        assert seg["mux_s"] > 0


def test_mux_respects_total_frames(tmp_path):
    rec = streaming.StreamRecorder(store_frames=True)
    rec.chunks.append({"chunk": 0, "frame_start": 0, "frame_end": 20, "t": 0.0})
    rec.frames.append(torch.zeros(20, 48, 64, 3, dtype=torch.uint8))
    segments = streaming.mux_all_segments(rec, tmp_path, fps=24, total_frames=12)
    assert len(segments) == 1
    n, _ = _ffprobe_frames(segments[0]["path"])
    assert n == 12


def test_schedule_oracle_never_rebuffers():
    durations = [17 / 24.0] * 7
    ready = [10.0, 10.4, 10.8, 11.2, 11.6, 12.0, 12.4]
    start = streaming.no_buffer_start(ready, durations)
    offsets = streaming.playback_offsets(durations)
    assert all(start + o >= r - 1e-9 for o, r in zip(offsets, ready))
    # arrival rate (0.4 s/chunk) < playback rate (0.708 s/chunk): the first
    # segment is the binding constraint and playback starts right on it.
    assert start == pytest.approx(10.0)


def test_schedule_oracle_slow_arrivals():
    durations = [1.0, 1.0, 1.0]
    ready = [1.0, 3.0, 5.0]  # 2 s per 1 s segment: must delay start
    start = streaming.no_buffer_start(ready, durations)
    assert start == pytest.approx(3.0)  # last segment arrives at 5, plays at start+2


def test_linear_estimator_padding_prevents_rebuffer():
    durations = [1.0] * 6
    ready = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]  # exactly 1 s per 1 s segment
    verdict = streaming.linear_estimate_start(ready, durations, observe=2, pad=1.15)
    assert not verdict["would_rebuffer"]
    assert verdict["start_s"] >= 2.0
    offsets = streaming.playback_offsets(durations)
    assert all(verdict["start_s"] + o >= r - 1e-9 for o, r in zip(offsets, ready))


def test_linear_estimator_detects_stall_on_decelerating_arrivals():
    durations = [1.0] * 5
    # first two arrive fast (rate looks like 0.2 s), the tail crawls
    ready = [1.0, 1.2, 5.0, 9.0, 13.0]
    verdict = streaming.linear_estimate_start(ready, durations, observe=2, pad=1.15)
    assert verdict["would_rebuffer"]
    assert verdict["worst_margin_s"] < 0


def test_linear_estimator_start_never_precedes_first_segment():
    durations = [1.0] * 3
    ready = [4.0, 4.1, 4.2]
    verdict = streaming.linear_estimate_start(ready, durations)
    assert verdict["start_s"] >= 4.0


def test_audio_first_prefetch_helper():
    calls = []

    class DummyAudioVAE:
        def decode(self, latents):
            calls.append(latents)
            return latents

    class DummyPipe:
        pass

    pipe = DummyPipe()
    pipe.audio_vae = DummyAudioVAE()

    def call_site(self, audio, rec):  # the pipeline frame: `self` and `audio` in its locals
        return streaming._maybe_prefetch_audio(sys._getframe(), rec)

    def bare_site(rec):
        return streaming._maybe_prefetch_audio(sys._getframe(), rec)

    rec = streaming.StreamRecorder()
    latents = torch.zeros(2)
    assert call_site(pipe, latents, rec)
    assert calls == [latents]
    # a warm cache for these same latents reports success without decoding again
    rec.audio_cache = (latents, latents)
    assert call_site(pipe, latents, rec)
    assert len(calls) == 1
    # a cache for other latents is not reused, even with an equal id() after a free
    rec.audio_cache = (torch.zeros(2), latents)
    assert call_site(pipe, latents, rec)
    assert len(calls) == 2
    # unexpected call sites fall back cleanly
    fresh = streaming.StreamRecorder()
    assert not bare_site(fresh)
    assert not call_site(object(), latents, fresh)


def _audio_first_pipeline(monkeypatch):
    """A pipeline whose generate() decodes video exactly like the real one: inside a list comprehension."""
    video_vae = importlib.import_module("models.minimax_h3.video_vae")
    calls = []

    def fake_video_decode(self, latents, uint8_rounding=None):
        calls.append(("video", uint8_rounding))
        return latents

    monkeypatch.setattr(video_vae.MiniMaxH3VideoVAE, "decode", fake_video_decode)

    audio_mod = importlib.import_module("models.minimax_h3.audio_vae")

    def fake_audio_decode(self, latents):
        calls.append(("audio", float(latents.sum())))
        return latents.clone()

    # The plugin's caching wrapper wraps the real class's decode, so patch that class (before activate()).
    monkeypatch.setattr(audio_mod.MiniMaxH3AudioVAE, "decode", fake_audio_decode)

    class DummyPipe:
        audio_vae = audio_mod.MiniMaxH3AudioVAE.__new__(audio_mod.MiniMaxH3AudioVAE)
        vae = video_vae.MiniMaxH3VideoVAE.__new__(video_vae.MiniMaxH3VideoVAE)

        def generate(self, video, audio, image_outputs=False):
            uint8_rounding = "truncate"
            decoded = [self.vae.decode(video[item:item + 1], uint8_rounding) for item in range(video.shape[0])]
            if image_outputs:
                return decoded, None
            return decoded, self.audio_vae.decode(audio)

    return DummyPipe(), calls


def test_audio_first_reaches_the_pipeline_through_a_list_comprehension(monkeypatch):
    """On Python 3.11 the comprehension is its own frame: the hook must look past it (it used to fall back silently)."""
    pipe, calls = _audio_first_pipeline(monkeypatch)
    rec = streaming.activate(store_frames=False, audio_first=True)
    try:
        pipe.generate(torch.zeros(1, 1), torch.ones(2))
        assert rec.audio_first_effective
        assert calls == [("audio", 2.0), ("video", "truncate")]  # audio first, and decoded only once
    finally:
        streaming.deactivate()


def test_each_generation_gets_its_own_audio(monkeypatch):
    """The UI never calls reset(): a second generation must not reuse the first one's audio for its segments."""
    pipe, calls = _audio_first_pipeline(monkeypatch)
    rec = streaming.activate(store_frames=False, audio_first=True)
    try:
        pipe.generate(torch.zeros(1, 1), torch.ones(2))
        first = rec.audio_out
        pipe.generate(torch.zeros(1, 1), torch.full((2,), 3.0))
        assert rec.audio_first_effective
        assert calls == [("audio", 2.0), ("video", "truncate"), ("audio", 6.0), ("video", "truncate")]
        assert torch.equal(rec.audio_out, torch.full((2,), 3.0)) and not torch.equal(rec.audio_out, first)
        # without audio-first, a generation's segments stay silent instead of inheriting the last audio
        rec.audio_first = False
        pipe.generate(torch.zeros(1, 1), torch.full((2,), 5.0))
        assert not rec.audio_first_effective
        assert calls[-2:] == [("video", "truncate"), ("audio", 10.0)]
    finally:
        streaming.deactivate()


def test_live_segmenter_muxes_with_audio(tmp_path):
    seg = streaming.LiveSegmenter(tmp_path, fps=24, sample_rate=32000)
    n_frames, chunks = 17, 3
    t = torch.linspace(0, 500, int(32000 * chunks * n_frames / 24))
    seg.set_audio((torch.sin(t) * 0.3).repeat(2, 1))
    for i in range(chunks):
        meta = {"chunk": i, "frame_start": i * n_frames, "frame_end": (i + 1) * n_frames,
                "t": float(i)}
        frames = torch.full((n_frames, 48, 64, 3), i * 40, dtype=torch.uint8)
        seg.submit(meta, frames)
    results = seg.close()
    assert len(results) == chunks
    for r in results:
        assert "error" not in r, r
        assert r["has_audio"]
        assert r["t_done"] > 0
        n, _ = _ffprobe_frames(r["path"])
        assert n == n_frames
    times = [r["t_done"] for r in results]
    assert times == sorted(times)


def test_instrumented_decode_feeds_live_segmenter(tmp_path):
    vae = _tiny_vae()
    z = torch.randn(1, 4, 12, 8, 8)
    rec = streaming.activate(store_frames=False, live_dir=str(tmp_path))
    try:
        rec.reset()
        with torch.no_grad():
            decoded = AutoencoderKLMiniMaxH3._decode(vae, z)
        results = rec.live.close()
    finally:
        streaming.deactivate()
    assert len(results) == len(rec.chunks)
    total = sum(r["frame_end"] - r["frame_start"] for r in results)
    assert total == decoded.shape[2]
    for r in results:
        assert "error" not in r, r
        assert not r["has_audio"]  # no audio was set in this run


# -- multi-device decode ----------------------------------------------------
#
# A device list like ["cpu", "cpu", "cpu"] runs the real dispatcher: the first
# "cpu" is the VAE's own device (decoded on the calling thread), every other
# entry gets a replica of the decoder and a worker thread, so the chunks decode
# concurrently on CPU exactly as they would across GPUs.

_MULTI_LENGTHS = (2, 5, 8, 12, 21, 37)  # 0, 1, 1, 2, 4 and 7 chunks
_REFERENCE_CACHE: dict = {}


def _reference_decode(z_tokens, uint8_rounding):
    """Original `decode` output (no plugin) of the tiny adapter VAE for a seeded latent."""
    key = (z_tokens, uint8_rounding)
    if key not in _REFERENCE_CACHE:
        vae = _tiny_adapter_vae()
        z = torch.randn(1, 4, z_tokens, 8, 8, generator=torch.Generator().manual_seed(z_tokens))
        with torch.no_grad():
            _REFERENCE_CACHE[key] = (z, AutoencoderKLMiniMaxH3.decode(vae, z, False, uint8_rounding)[0])
    return _REFERENCE_CACHE[key]


def _decode_threads():
    return [t for t in threading.enumerate() if t.name.startswith("h3-decode")]


def _assert_tiling_events(rec, total_frames):
    assert [c["chunk"] for c in rec.chunks] == list(range(len(rec.chunks)))
    assert rec.chunks[0]["frame_start"] == 0
    for k in range(1, len(rec.chunks)):
        assert rec.chunks[k]["frame_start"] == rec.chunks[k - 1]["frame_end"]
    assert rec.chunks[-1]["frame_end"] == total_frames
    ts = [c["t"] for c in rec.chunks]
    assert ts == sorted(ts)


@pytest.mark.parametrize("uint8_rounding", ["truncate", "round", None])
@pytest.mark.parametrize("num_devices", [1, 2, 3, 4])
def test_multi_device_decode_bit_identical(num_devices, uint8_rounding):
    """Every latent length, 1-4 decode slots and every rounding mode give the original output, frames and in-order events."""
    vae = _tiny_adapter_vae()
    rec = streaming.activate(store_frames=True, decode_devices=["cpu"] * num_devices)
    try:
        for z_tokens in _MULTI_LENGTHS:
            z, expected = _reference_decode(z_tokens, uint8_rounding)
            rec.reset()
            with torch.no_grad():
                got = AutoencoderKLMiniMaxH3.decode(vae, z, False, uint8_rounding)[0]
            assert got.dtype == expected.dtype
            assert torch.equal(expected, got), f"multi-device decode diverged: {z_tokens} latents, {num_devices} devices"
            _assert_tiling_events(rec, expected.shape[2])
            frames = torch.cat(rec.frames)
            reference = expected if uint8_rounding else expected.clamp(-1.0, 1.0).add(1.0).mul(127.5).round().to(torch.uint8)
            assert torch.equal(frames, reference[0].permute(1, 2, 3, 0))
            split = num_devices > 1 and len(rec.chunks) > 1
            assert (rec.decode_plan is not None) == split
            if split:
                assert rec.decode_plan["devices"] == ["cpu"] * num_devices and rec.decode_plan["fallback"] is None
    finally:
        streaming.deactivate()
    assert not _decode_threads()


def test_multi_device_decode_raw_vae_and_tiling(monkeypatch):
    """The base class (identity `_prepare_decoded_chunk`) with spatial tiling: same output, same progress count."""
    vae = _tiny_vae()
    vae.tile_sample_min_height = vae.tile_sample_min_width = 64
    vae.tile_sample_min_overlap_height = vae.tile_sample_min_overlap_width = 32
    z = torch.randn(1, 4, 21, 8, 8)
    totals = []

    class CountingProgress(streaming.PhaseProgress):
        def __exit__(self, *exc):
            totals.append((self.completed, self.total))
            return super().__exit__(*exc)

    monkeypatch.setattr(streaming, "PhaseProgress", CountingProgress)

    class FakeGeneration:  # PhaseProgress counts only inside a generation
        pipeline = types.SimpleNamespace(_interrupt=False)
        callback = None
        set_status = None
        next_update = 0.0

        def check_abort(self):
            pass

    phase_progress = importlib.import_module("shared.utils.phase_progress")
    token = phase_progress._generation.set(FakeGeneration())
    try:
        results = []
        for devices in (None, ["cpu"] * 3):
            streaming.activate(store_frames=False, decode_devices=devices)
            try:
                with torch.no_grad():
                    results.append(AutoencoderKLMiniMaxH3._decode(vae, z))
            finally:
                streaming.deactivate()
    finally:
        phase_progress._generation.reset(token)
    assert torch.equal(results[0], results[1])
    assert totals[0] == totals[1] and totals[0][0] == totals[0][1] > 4  # 4 chunks x several tiles, all counted


def test_multi_device_decode_runs_concurrently(monkeypatch):
    """The first clip of every slot is in flight at the same time: a 3-party barrier only passes if they run concurrently."""
    vae = _tiny_vae()
    z = torch.randn(1, 4, 37, 8, 8)
    with torch.no_grad():
        expected = AutoencoderKLMiniMaxH3._decode(vae, z)
    original = AutoencoderKLMiniMaxH3._decode_clip
    barrier = threading.Barrier(3, timeout=20)
    seen = set()
    lock = threading.Lock()

    def gated(self, z_part):
        name = threading.current_thread().name
        with lock:
            first = name not in seen
            seen.add(name)
        if first:
            barrier.wait()
        return original(self, z_part)

    monkeypatch.setattr(AutoencoderKLMiniMaxH3, "_decode_clip", gated)
    streaming.activate(store_frames=False, decode_devices=["cpu", "cpu", "cpu"])
    try:
        with torch.no_grad():
            got = AutoencoderKLMiniMaxH3._decode(vae, z)
    finally:
        streaming.deactivate()
    assert torch.equal(expected, got)
    assert len(seen) == 3 and sum(name.startswith("h3-decode") for name in seen) == 2


def _chunk_index(z_part, z_padded, tokens_chunk_size):
    for i in range(z_padded.shape[2] // tokens_chunk_size):
        start = i * tokens_chunk_size
        candidate = z_padded[:, :, start : start + z_part.shape[2]]
        if candidate.shape == z_part.shape and torch.equal(candidate, z_part):
            return i
    raise AssertionError("unknown latent slice")


def test_multi_device_decode_finalizes_in_order_when_chunks_finish_out_of_order(monkeypatch, tmp_path):
    """A slow device finishes its chunks after later ones: output, events and live segments stay in chunk order."""
    vae = _tiny_vae()
    z = torch.randn(1, 4, 37, 8, 8)  # 7 chunks, no padding (37 + 3 is a multiple of 5)
    with torch.no_grad():
        expected = AutoencoderKLMiniMaxH3._decode(vae, z)
    original = AutoencoderKLMiniMaxH3._decode_clip
    finished = []
    lock = threading.Lock()
    slow_thread = []

    def uneven(self, z_part):
        name = threading.current_thread().name
        with lock:
            if name.startswith("h3-decode") and not slow_thread:
                slow_thread.append(name)
        if slow_thread and name == slow_thread[0]:
            time.sleep(0.4)
        clip = original(self, z_part)
        with lock:
            finished.append(_chunk_index(z_part, z, vae.tokens_chunk_size))
        return clip

    monkeypatch.setattr(AutoencoderKLMiniMaxH3, "_decode_clip", uneven)
    rec = streaming.activate(store_frames=True, live_dir=str(tmp_path), decode_devices=["cpu", "cpu", "cpu"])
    try:
        with torch.no_grad():
            got = AutoencoderKLMiniMaxH3._decode(vae, z)
        segments = rec.live.close()
    finally:
        streaming.deactivate()
    assert sorted(finished) == list(range(7))
    assert finished != sorted(finished), "the slow device should have finished after later chunks"
    assert torch.equal(expected, got)
    _assert_tiling_events(rec, expected.shape[2])
    assert [(s["frame_start"], s["frame_end"]) for s in segments] == [(c["frame_start"], c["frame_end"]) for c in rec.chunks]
    assert all("error" not in s for s in segments)


def test_multi_device_decode_option_off_is_unchanged(monkeypatch):
    """Without decode devices no dispatcher, replica or thread is created and the output is the original one."""
    created = []
    monkeypatch.setattr(streaming, "_ChunkDispatcher", lambda *a, **k: created.append(a))
    vae = _tiny_vae()
    z = torch.randn(1, 4, 21, 8, 8)
    with torch.no_grad():
        expected = AutoencoderKLMiniMaxH3._decode(vae, z)
    for devices in (None, [], "", ["cpu"]):  # a single slot on the VAE's own device is the plain decode too
        rec = streaming.activate(store_frames=True, decode_devices=devices)
        try:
            with torch.no_grad():
                got = AutoencoderKLMiniMaxH3._decode(vae, z)
        finally:
            streaming.deactivate()
        assert torch.equal(expected, got)
        assert rec.decode_plan is None
    assert not created and not _decode_threads()
    assert streaming.decode_devices() is None


@pytest.mark.parametrize("failing", ["worker", "main"])
def test_multi_device_decode_propagates_errors(monkeypatch, failing):
    """A decode that raises on a worker (or on the calling thread) surfaces its exception; workers are joined and reusable."""
    vae = _tiny_vae()
    z = torch.randn(1, 4, 37, 8, 8)
    with torch.no_grad():
        expected = AutoencoderKLMiniMaxH3._decode(vae, z)
    original = AutoencoderKLMiniMaxH3._decode_clip
    calls = {"worker": 0, "main": 0}

    def flaky(self, z_part):
        side = "worker" if threading.current_thread().name.startswith("h3-decode") else "main"
        calls[side] += 1
        if side == failing and calls[side] == 2:
            raise ValueError(f"boom on {side}")
        return original(self, z_part)

    rec = streaming.activate(store_frames=True, decode_devices=["cpu"] * 3)
    try:
        monkeypatch.setattr(AutoencoderKLMiniMaxH3, "_decode_clip", flaky)
        with pytest.raises(ValueError, match=f"boom on {failing}"), torch.no_grad():
            AutoencoderKLMiniMaxH3._decode(vae, z)
        assert not _decode_threads()
        assert rec.decode_end_t is None
        # the cached replicas are left usable (no stale abort flag) and the next decode is exact
        monkeypatch.setattr(AutoencoderKLMiniMaxH3, "_decode_clip", original)
        rec.reset()
        with torch.no_grad():
            got = AutoencoderKLMiniMaxH3._decode(vae, z)
        assert torch.equal(expected, got)
        _assert_tiling_events(rec, expected.shape[2])
    finally:
        streaming.deactivate()
    assert not _decode_threads()


def test_multi_device_decode_aborts_replicas_on_interrupt(monkeypatch):
    """The pipeline's abort flag on the original decoder reaches a replica mid-decode, which stops with GenerationInterrupted."""
    interrupt = importlib.import_module("models.minimax_h3.interrupt")
    vae = _tiny_vae()
    z = torch.randn(1, 4, 37, 8, 8)
    original = AutoencoderKLMiniMaxH3._decode_clip
    interrupted_on = []

    def abort_after_first(self, z_part):
        if threading.current_thread().name.startswith("h3-decode"):
            time.sleep(0.3)  # still decoding when the calling thread asks for this clip and sees the abort
            try:
                return original(self, z_part)
            except interrupt.GenerationInterrupted:
                interrupted_on.append(threading.current_thread().name)
                raise
        out = original(self, z_part)
        vae.decoder._interrupt = True  # what `MiniMaxH3VideoVAE._interrupt = True` does, after chunk 0
        return out

    monkeypatch.setattr(AutoencoderKLMiniMaxH3, "_decode_clip", abort_after_first)
    streaming.activate(store_frames=False, decode_devices=["cpu", "cpu"])  # chunk 0 on the calling thread, chunk 1 on a replica
    try:
        with pytest.raises(interrupt.GenerationInterrupted), torch.no_grad():
            AutoencoderKLMiniMaxH3._decode(vae, z)
        replica = next(iter(streaming._REPLICAS[vae].values()))
        assert replica.module.decoder._interrupt is False  # reset for the next decode
    finally:
        vae.decoder._interrupt = False
        streaming.deactivate()
    assert interrupted_on and not _decode_threads()


def test_multi_device_single_frame_and_still_paths_stay_on_one_device():
    torch.manual_seed(0)
    single = AutoencoderKLMiniMaxH3(
        latent_channels=4, block_out_channels=(8, 8, 8, 8, 8, 8), layers_per_block=1, norm_num_groups=4,
        decoder_num_layers=1, decoder_num_attention_heads=1, decoder_attention_head_dim=8,
        decoder_num_register_tokens=1, decoder_ffn_mult=1, decoder_patch_size_t=1).eval()
    for vae in (_tiny_vae(), single):
        z = torch.randn(1, 4, 1, 8, 8)
        with torch.no_grad():
            expected = AutoencoderKLMiniMaxH3._decode(vae, z)
        rec = streaming.activate(store_frames=True, decode_devices=["cpu"] * 3)
        try:
            with torch.no_grad():
                got = AutoencoderKLMiniMaxH3._decode(vae, z)
        finally:
            streaming.deactivate()
        assert torch.equal(expected, got) and expected.shape[2] == 1
        assert rec.decode_plan is None
        assert [(c["frame_start"], c["frame_end"]) for c in rec.chunks] == [(0, 1)]


def _mmgp_style_hooks(vae, calls):
    """Hook the decode path like MMGP's offload: an instance forward partial that would load the weights on the main GPU,
    the previous forward kept in `_mm_forward`, plus `_hf_hook` and `_mm_*` bookkeeping. A loader-installed instance forward
    (bound method, like the INT8 precision rules) sits under it on one linear."""
    linear = vae.decoder.proj_out
    linear._int8_precision_forward = linear.forward

    def precision_forward(module, input):
        calls.append(("precision", threading.current_thread().name, module is linear))
        return module._int8_precision_forward(input)

    linear.forward = types.MethodType(precision_forward, linear)
    for module in [vae.post_quant_conv, *vae.decoder.modules()]:
        previous = module.forward

        def check_load(module, *args, _previous=previous, **kwargs):
            calls.append(("mmgp", threading.current_thread().name))
            return _previous(*args, **kwargs)

        module._mm_id = "vae"
        module._mm_manager = object()
        module._mm_forward = previous
        module._hf_hook = object()
        module.forward = functools.update_wrapper(functools.partial(check_load, module), previous)


def test_replica_strips_mmgp_offload_hooks():
    vae = _tiny_vae()
    z = torch.randn(1, 4, 21, 8, 8)
    with torch.no_grad():
        expected = AutoencoderKLMiniMaxH3._decode(vae, z)
    calls = []
    _mmgp_style_hooks(vae, calls)
    rec = streaming.activate(store_frames=False, decode_devices=["cpu"] * 3)
    try:
        with torch.no_grad():
            got = AutoencoderKLMiniMaxH3._decode(vae, z)
        replica = next(iter(streaming._REPLICAS[vae].values())).module
    finally:
        streaming.deactivate()
    assert torch.equal(expected, got) and rec.decode_plan["fallback"] is None
    # MMGP's hook only ever runs on the calling thread; replicas run the loader's forward, bound to their own module
    assert {c[1] for c in calls if c[0] == "mmgp"} == {"MainThread"}
    worker_precision = [c for c in calls if c[0] == "precision" and c[1].startswith("h3-decode")]
    assert worker_precision and all(c[2] is False for c in worker_precision)
    for module in replica.modules():
        assert not any(key.startswith("_mm_") or key == "_hf_hook" for key in module.__dict__)
    assert replica.decoder.proj_out.forward.__self__ is replica.decoder.proj_out
    assert "encoder" not in replica._modules  # the decode path only
    # on its own device a replica shares the weights instead of copying them
    assert replica.decoder.proj_out.weight.data_ptr() == vae.decoder.proj_out.weight.data_ptr()


def test_replica_refuses_unknown_wrappers_and_falls_back():
    vae = _tiny_vae()
    z = torch.randn(1, 4, 21, 8, 8)
    with torch.no_grad():
        expected = AutoencoderKLMiniMaxH3._decode(vae, z)
    previous = vae.decoder.x_embedder.forward
    vae.decoder.x_embedder.forward = lambda x: previous(x)  # a closure: what it does cannot be copied safely
    rec = streaming.activate(store_frames=True, decode_devices=["cpu", "cpu"])
    try:
        with torch.no_grad():
            got = AutoencoderKLMiniMaxH3._decode(vae, z)
    finally:
        streaming.deactivate()
    assert torch.equal(expected, got)
    assert "ReplicaUnsupported" in rec.decode_plan["fallback"]
    _assert_tiling_events(rec, expected.shape[2])


def test_decode_device_parsing():
    assert streaming.parse_decode_devices("cpu, cpu cpu") == ["cpu", "cpu", "cpu"]
    assert streaming.parse_decode_devices(["cpu", " "]) == ["cpu"]
    assert streaming.parse_decode_devices("") is None and streaming.parse_decode_devices(None) is None
    with pytest.raises(RuntimeError):
        streaming.parse_decode_devices("gpu7")
    with pytest.raises(RuntimeError):
        streaming.activate(decode_devices="gpu7")
    assert not streaming._STATE["active"]
    rec = streaming.activate(decode_devices="cpu,cpu")
    try:
        assert streaming.decode_devices() == ["cpu", "cpu"]
        assert streaming.activate(decode_devices=None) is rec and streaming.decode_devices() is None
    finally:
        streaming.deactivate()


@pytest.mark.parametrize("decode_mode", ["no_grad", "inference_mode"])
def test_multi_device_decode_with_quantized_inference_weights(decode_mode):
    """INT8 (quanto) decoder weights loaded under inference mode, as WanGP's loader does: replicas build and stay bit-identical."""
    quanto = pytest.importorskip("optimum.quanto")
    vae = _tiny_adapter_vae()
    with torch.inference_mode():
        quanto.quantize(vae.decoder, weights=quanto.qint8)
        quanto.freeze(vae.decoder)
    z = torch.randn(1, 4, _MULTI_LENGTHS[-1], 8, 8, generator=torch.Generator().manual_seed(7))
    mode = torch.inference_mode if decode_mode == "inference_mode" else torch.no_grad
    with mode():
        expected = AutoencoderKLMiniMaxH3.decode(vae, z, False, "round")[0]
    rec = streaming.activate(store_frames=False, decode_devices=["cpu", "cpu"])
    try:
        with mode():
            got = AutoencoderKLMiniMaxH3.decode(vae, z, False, "round")[0]
        assert rec.decode_plan is not None and rec.decode_plan["fallback"] is None, rec.decode_plan
        assert torch.equal(expected, got)
    finally:
        streaming.deactivate()
    assert not _decode_threads()
