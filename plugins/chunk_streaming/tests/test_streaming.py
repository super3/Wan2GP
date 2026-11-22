# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""CPU tests for plugins/chunk_streaming/streaming.py.

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
import queue
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

    # Imported by attention_kit since upstream v20: the attention kernel's launch waves on this GPU. None, as the real
    # helper reports when it cannot probe, makes attention_kit keep its default grouping.
    def _no_launch_waves(*_args, **_kw):
        return None

    attn_mod.pay_attention = _sdpa_pay_attention
    attn_mod.sage_staged = _no_sage_staged
    attn_mod.sage_attention_mask = _sage_attention_mask
    attn_mod.launch_waves = _no_launch_waves
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


class _DeferredRamWriter:
    """mmgp's `transfer.RamWriter` as it behaves with CUDA sources, on CPU: `write` returns at once, each piece lands later, in
    order, on a helper thread, which then calls its done(); close() waits for what is pending."""

    def __init__(self, *_args, **_kwargs):
        self.pieces: queue.Queue = queue.Queue()
        self.error = None
        self.thread = threading.Thread(target=self._land, name="DeferredRamWriter", daemon=True)
        self.thread.start()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def write(self, destination, source, done=None):
        self.pieces.put((destination, source.clone(), done))

    def _land(self):
        while (piece := self.pieces.get()) is not None:
            destination, source, done = piece
            time.sleep(0.002)  # the GPU has not passed the piece yet: the decode loop runs ahead
            try:
                destination.copy_(source)
                if done is not None:
                    done()
            except BaseException as error:  # noqa: BLE001 - reported by close(), as RamWriter does
                self.error = error

    def close(self):
        self.pieces.put(None)
        self.thread.join()
        if self.error is not None:
            raise self.error


@pytest.mark.parametrize("deferred", [False, True])
def test_instrumented_decode_reports_clips_once_their_rows_have_landed(monkeypatch, deferred):
    """Spatial tiling decodes each clip by rows of tiles (`_decode_clip_rows`, here 3 rows) whose pixels RamWriter lands in the
    CPU output later, from its own thread, on a GPU (`deferred`): a clip goes to the recorder only once all its rows are there."""
    vae = _tiny_adapter_vae()
    vae.tile_sample_min_height = vae.tile_sample_min_width = 64
    vae.tile_sample_min_overlap_height = vae.tile_sample_min_overlap_width = 32
    z = torch.randn(1, 4, 21, 8, 8)  # 4 clips
    with torch.no_grad():
        assert len(list(vae._decode_clip_rows(z[:, :, :7]))) == 3
        expected = AutoencoderKLMiniMaxH3.decode(vae, z, False, "round")[0]
    if deferred:
        monkeypatch.setattr(streaming._va.transfer, "RamWriter", _DeferredRamWriter)
    rec = streaming.activate(store_frames=True)
    seen = []
    on_chunk = rec.on_chunk

    def checked_on_chunk(vae_, decoded, frame_start, frame_end, chunk_index):
        landed = torch.equal(decoded[:, :, frame_start:frame_end], expected[:, :, frame_start:frame_end])
        seen.append((threading.current_thread().name, landed))
        on_chunk(vae_, decoded, frame_start, frame_end, chunk_index)

    monkeypatch.setattr(rec, "on_chunk", checked_on_chunk)
    try:
        rec.reset()
        with torch.no_grad():
            got = AutoencoderKLMiniMaxH3.decode(vae, z, False, "round")[0]
    finally:
        streaming.deactivate()
    assert torch.equal(expected, got), "instrumented decode diverged from original"
    assert [c["chunk"] for c in rec.chunks] == [0, 1, 2, 3]
    assert rec.chunks[0]["frame_start"] == 0 and rec.chunks[-1]["frame_end"] == expected.shape[2]
    assert all(rec.chunks[k]["frame_start"] == rec.chunks[k - 1]["frame_end"] for k in range(1, len(rec.chunks)))
    assert all(landed for _, landed in seen), "a clip was reported before all its rows landed"
    assert torch.equal(torch.cat(rec.frames), expected[0].permute(1, 2, 3, 0))
    if deferred:
        assert any(name == "DeferredRamWriter" for name, _ in seen)


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


# -- Sol-H3-VAE (VAE config sol_int8): MiniMaxH3VideoVAE's own whole-frame _decode ----------------------------------------

_video_vae = importlib.import_module("models.minimax_h3.video_vae")
_sol_mod = importlib.import_module("models.minimax_h3.components.sol_decoder")


class _TinySolVAE(_video_vae.MiniMaxH3VideoVAE):
    """The WanGP adapter class (its Sol `_decode`, `_prepare_decoded_chunk`) around a tiny H3 VAE whose decoder is a tiny
    `MiniMaxH3SolDecoder3d` (sol=False keeps the tiny H3 decoder: the non-Sol route through the same class)."""

    def __init__(self, sol: bool = True):
        AutoencoderKLMiniMaxH3.__init__(self, latent_channels=4, block_out_channels=(8, 8, 8, 8, 8, 8), layers_per_block=1,
                                        norm_num_groups=4, decoder_num_layers=1, decoder_num_attention_heads=1,
                                        decoder_attention_head_dim=8, decoder_num_register_tokens=1, decoder_ffn_mult=1)
        if sol:
            self.decoder = _sol_mod.MiniMaxH3SolDecoder3d(in_channels=4, out_channels=3, patch_size=2, patch_size_t=4, num_layers=2,
                                                          num_attention_heads=2, attention_head_dim=8, num_register_tokens=1, ffn_mult=1)
        self.upsampling = False
        self.register_buffer("pixel_mean", torch.tensor(_video_vae.IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(_video_vae.IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)
        self._interrupt = False


@pytest.fixture
def sol_cpu(monkeypatch):
    """CPU stand-ins for the Sol decoder's GPU-only sizing, the same for the upstream and the instrumented decode: one FFN chunk
    (kernels.mlp asks CUDA for the L2 size), and bands of 2 latent rows so each latent frame is emitted in several bands."""
    ops = importlib.import_module("shared.kernels.ops")
    monkeypatch.setattr(ops, "mlp_chunk_rows", lambda tokens, *_a, **_k: tokens)
    monkeypatch.setattr(_sol_mod, "EMIT_BYTES", 2 * 10 * _sol_mod.EMIT_TOKEN_BYTES)  # W = 10 latent columns below
    torch.manual_seed(0)
    return _TinySolVAE().eval()


def _sol_decode(vae, z, uint8_rounding=None):
    with torch.no_grad():
        return _video_vae.MiniMaxH3VideoVAE._decode(vae, z, uint8_rounding)


@pytest.mark.parametrize("latents,uint8_rounding", [(7, "round"), (12, None), (22, "truncate"), (37, "round")])
def test_sol_decode_bit_identical_and_reports_each_block(sol_cpu, latents, uint8_rounding):
    """37 latents are a 124-frame clip, as WanGP's 5 s generations: 7 blocks of 17 frames, then 5."""
    z = torch.randn(1, 4, latents, 6, 10)
    expected = _sol_decode(sol_cpu, z, uint8_rounding)
    rec = streaming.activate(store_frames=True)
    try:
        rec.reset()
        got = _sol_decode(sol_cpu, z, uint8_rounding)
    finally:
        streaming.deactivate()
    assert torch.equal(expected, got), "instrumented Sol decode diverged from upstream's"
    frames = sol_cpu.decoder.output_frames(latents)
    assert expected.shape[2] == frames
    blocks = -(-latents // _sol_mod.BLOCK)
    assert [(c["frame_start"], c["frame_end"]) for c in rec.chunks] == [(17 * b, min(17 * (b + 1), frames)) for b in range(blocks)]
    assert [c["chunk"] for c in rec.chunks] == list(range(blocks))
    reference = expected if uint8_rounding else expected.clamp(-1.0, 1.0).add(1.0).mul(127.5).round().to(torch.uint8)
    assert torch.equal(torch.cat(rec.frames), reference[0].permute(1, 2, 3, 0))


@pytest.mark.parametrize("deferred", [False, True])
def test_sol_decode_reports_blocks_once_they_have_landed(sol_cpu, monkeypatch, deferred):
    """On a GPU RamWriter lands each band later, from its own thread (`deferred`): a block goes to the recorder only once every
    band written before its report is in the CPU output."""
    z = torch.randn(1, 4, 22, 6, 10)
    expected = _sol_decode(sol_cpu, z, "round")
    if deferred:
        monkeypatch.setattr(streaming._va.transfer, "RamWriter", _DeferredRamWriter)
    rec = streaming.activate(store_frames=True)
    seen = []
    on_chunk = rec.on_chunk

    def checked_on_chunk(vae_, decoded, frame_start, frame_end, chunk_index):
        seen.append((threading.current_thread().name, torch.equal(decoded[:, :, :frame_end], expected[:, :, :frame_end])))
        on_chunk(vae_, decoded, frame_start, frame_end, chunk_index)

    monkeypatch.setattr(rec, "on_chunk", checked_on_chunk)
    try:
        rec.reset()
        got = _sol_decode(sol_cpu, z, "round")
    finally:
        streaming.deactivate()
    assert torch.equal(expected, got)
    assert len(seen) == 5 and all(landed for _, landed in seen), "a block was reported before its bands landed"
    if deferred:
        assert any(name == "DeferredRamWriter" for name, _ in seen)


def test_sol_decode_picture(sol_cpu):
    """One latent: five copies, its first frame kept, one blocking copy; one report."""
    z = torch.randn(1, 4, 1, 6, 10)
    expected = _sol_decode(sol_cpu, z, "round")
    rec = streaming.activate(store_frames=True)
    try:
        rec.reset()
        got = _sol_decode(sol_cpu, z, "round")
    finally:
        streaming.deactivate()
    assert torch.equal(expected, got) and expected.shape[2] == 1
    assert [(c["frame_start"], c["frame_end"]) for c in rec.chunks] == [(0, 1)]


def test_adapter_class_without_sol_decoder_streams_through_the_h3_path():
    """MiniMaxH3VideoVAE._decode hands any other decoder to AutoencoderKLMiniMaxH3._decode: the instrumented H3 path, once."""
    torch.manual_seed(0)
    vae = _TinySolVAE(sol=False).eval()
    z = torch.randn(1, 4, 12, 8, 8)
    expected = _sol_decode(vae, z, "round")
    rec = streaming.activate(store_frames=True)
    try:
        rec.reset()
        got = _sol_decode(vae, z, "round")
    finally:
        streaming.deactivate()
    assert torch.equal(expected, got)
    assert len(rec.chunks) > 1 and rec.chunks[-1]["frame_end"] == expected.shape[2]
    assert rec.decode_start_t is not None


def test_deactivate_restores_the_sol_decode():
    original = _video_vae.MiniMaxH3VideoVAE.__dict__["_decode"]
    streaming.activate()
    assert _video_vae.MiniMaxH3VideoVAE.__dict__["_decode"] is not original
    streaming.deactivate()
    assert _video_vae.MiniMaxH3VideoVAE.__dict__["_decode"] is original


# -- pi-H3 DWD (models/minimax_h3/pi_h3_dwd.py, where present): the transformer's pixel head decodes the video, without a VAE ---

try:
    _dwd = importlib.import_module("models.minimax_h3.pi_h3_dwd")
    _dwd_refiner = importlib.import_module("models.minimax_h3.components.dwd_refiner")
except ImportError:  # a WanGP without the pi-H3 DWD models
    _dwd = _dwd_refiner = None
needs_dwd = pytest.mark.skipif(_dwd is None, reason="no pi-H3 DWD (models/minimax_h3/pi_h3_dwd.py) in this WanGP")
DWD_GRID_H, DWD_GRID_W = 3, 6  # DiT tokens: 6 x 12 Sol tokens, 96 x 192 pixels


@pytest.fixture
def dwd_cpu(monkeypatch):
    """A one-layer DWD refiner (Sol width, DiT width 6) with random weights, and the Sol CPU stand-ins of `sol_cpu`: bands of 2
    Sol rows, so each latent frame is emitted in 3 bands."""
    ops = importlib.import_module("shared.kernels.ops")
    monkeypatch.setattr(ops, "mlp_chunk_rows", lambda tokens, *_a, **_k: tokens)
    monkeypatch.setattr(_dwd_refiner, "EMIT_BYTES", 2 * 2 * DWD_GRID_W * _sol_mod.EMIT_TOKEN_BYTES)
    torch.manual_seed(0)
    refiner = _dwd_refiner.MiniMaxH3DWDRefiner(6, 1, (2, 2)).eval()
    with torch.no_grad():
        for parameter in refiner.parameters():
            parameter.normal_(0.0, 0.05)
    return refiner


def _dwd_decode(refiner, latents, uint8_rounding):
    features = torch.randn(latents * DWD_GRID_H * DWD_GRID_W, 6, generator=torch.Generator().manual_seed(latents))
    with torch.no_grad():
        return _dwd.decode(refiner, features, (latents, DWD_GRID_H, DWD_GRID_W), uint8_rounding)  # through the module, as the transformer


@needs_dwd
@pytest.mark.parametrize("latents,uint8_rounding", [(7, "round"), (12, None), (22, "truncate")])
def test_dwd_decode_bit_identical_and_reports_each_block(dwd_cpu, latents, uint8_rounding):
    expected = _dwd_decode(dwd_cpu, latents, uint8_rounding)
    rec = streaming.activate(store_frames=True)
    try:
        rec.reset()
        got = _dwd_decode(dwd_cpu, latents, uint8_rounding)
    finally:
        streaming.deactivate()
    assert torch.equal(expected, got), "instrumented DWD decode diverged from pi_h3_dwd.decode"
    frames = dwd_cpu.output_frames(latents)
    assert expected.shape == (1, 3, frames, DWD_GRID_H * 32, DWD_GRID_W * 32)
    blocks = -(-latents // _sol_mod.BLOCK)
    assert [(c["frame_start"], c["frame_end"]) for c in rec.chunks] == [(17 * b, min(17 * (b + 1), frames)) for b in range(blocks)]
    reference = expected if uint8_rounding else expected.clamp(-1.0, 1.0).add(1.0).mul(127.5).round().to(torch.uint8)
    assert torch.equal(torch.cat(rec.frames), reference[0].permute(1, 2, 3, 0))
    assert rec.decode_start_t is not None and rec.decode_end_t is not None


@needs_dwd
@pytest.mark.parametrize("deferred", [False, True])
def test_dwd_decode_reports_blocks_once_they_have_landed(dwd_cpu, monkeypatch, deferred):
    expected = _dwd_decode(dwd_cpu, 12, "round")
    if deferred:
        monkeypatch.setattr(streaming._va.transfer, "RamWriter", _DeferredRamWriter)
    rec = streaming.activate(store_frames=True)
    seen = []
    on_chunk = rec.on_chunk

    def checked_on_chunk(vae_, decoded, frame_start, frame_end, chunk_index):
        seen.append((threading.current_thread().name, torch.equal(decoded[:, :, :frame_end], expected[:, :, :frame_end])))
        on_chunk(vae_, decoded, frame_start, frame_end, chunk_index)

    monkeypatch.setattr(rec, "on_chunk", checked_on_chunk)
    try:
        rec.reset()
        got = _dwd_decode(dwd_cpu, 12, "round")
    finally:
        streaming.deactivate()
    assert torch.equal(expected, got)
    assert len(seen) == 3 and all(landed for _, landed in seen), "a block was reported before its bands landed"
    if deferred:
        assert any(name == "DeferredRamWriter" for name, _ in seen)


@needs_dwd
def test_dwd_live_segments_wait_for_the_audio(dwd_cpu, tmp_path):
    """DWD decodes the video before the audio is known: with audio first, the segments wait for this generation's audio (never
    the previous generation's) instead of being muxed silent."""
    rec = streaming.activate(store_frames=False, live_dir=str(tmp_path), audio_first=True)
    try:
        rec.audio_out = torch.ones(2, 32000)  # the previous generation's
        decoded = _dwd_decode(dwd_cpu, 7, "round")
        assert rec.segments_wait_audio and not rec.audio_first_effective and rec.audio_out is None
        time.sleep(0.3)
        assert rec.live.results == []  # nothing muxed while the audio is pending
        frames = decoded.shape[2]
        audio = (torch.sin(torch.linspace(0, 400, 32000 * frames // 24)) * 0.3).repeat(2, 1)
        rec.live.set_audio(audio)  # what the audio decode wrapper does once the pipeline decodes the audio
        results = rec.live.close()
    finally:
        streaming.deactivate()
    assert [(r["frame_start"], r["frame_end"]) for r in results] == [(0, 17), (17, frames)]
    for r in results:
        assert "error" not in r, r
        assert r["has_audio"]
        assert _ffprobe_frames(r["path"])[0] == r["frame_end"] - r["frame_start"]


def test_live_segmenter_stops_waiting_for_audio_at_close(tmp_path):
    seg = streaming.LiveSegmenter(tmp_path, fps=24, sample_rate=32000)
    seg.wait_for_audio(60.0)
    seg.submit({"chunk": 0, "frame_start": 0, "frame_end": 17, "t": 0.0}, torch.zeros(17, 48, 64, 3, dtype=torch.uint8))
    started = time.monotonic()
    results = seg.close()
    assert time.monotonic() - started < 30
    assert len(results) == 1 and "error" not in results[0] and not results[0]["has_audio"]


@needs_dwd
def test_deactivate_restores_the_dwd_decode():
    original = _dwd.decode
    streaming.activate()
    assert _dwd.decode is not original
    streaming.deactivate()
    assert _dwd.decode is original


def test_a_wangp_without_dwd_still_activates(monkeypatch):
    monkeypatch.setattr(streaming, "_dwd_module", lambda: None)
    streaming.activate()
    try:
        assert streaming._STATE["orig_dwd_decode"] is None
    finally:
        streaming.deactivate()
