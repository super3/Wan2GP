# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""CPU tests for models/minimax_h3/streaming.py.

Three properties are load-bearing enough to gate a GPU bench on:
1. The instrumented `_decode` returns a tensor bit-identical to the original
   and its finalized-frame events tile the output exactly (tiny CPU VAE).
2. The segment muxer produces playable fMP4 segments whose ffprobe frame
   counts match the finalized ranges.
3. The player schedule math: the oracle start never rebuffers, and the linear
   estimator flags exactly the arrival patterns that would stall.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# models/minimax_h3/__init__.py drags in mmgp, which asserts on CPU-only
# torch at import time. streaming.py needs none of that, so register the
# package path without executing __init__ and import the submodules directly.
import importlib
import types

for name, rel in (("models", "models"), ("models.minimax_h3", "models/minimax_h3")):
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

    attn_mod.pay_attention = _sdpa_pay_attention
    sys.modules["shared.attention"] = attn_mod
    shared_pkg.attention = attn_mod

streaming = importlib.import_module("models.minimax_h3.streaming")
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
    rec = streaming.StreamRecorder()
    latents = torch.zeros(2)
    assert streaming._maybe_prefetch_audio({"self": pipe, "audio": latents}, rec)
    assert calls == [latents]
    # a warm cache reports success without decoding again
    rec.audio_cache = (id(latents), latents)
    assert streaming._maybe_prefetch_audio({"self": pipe, "audio": latents}, rec)
    assert len(calls) == 1
    # unexpected call sites fall back cleanly
    fresh = streaming.StreamRecorder()
    assert not streaming._maybe_prefetch_audio({"self": pipe}, fresh)
    assert not streaming._maybe_prefetch_audio({}, fresh)
    assert not streaming._maybe_prefetch_audio({"self": object(), "audio": latents}, fresh)


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
