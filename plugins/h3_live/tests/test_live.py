# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""CPU tests of live.py (no WanGP, no GPU; ffmpeg on the PATH for the encoding tests):

    python -m pytest tests -q
"""

import importlib.util
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h3_live_under_test", ROOT / "live.py")
live = importlib.util.module_from_spec(spec)
sys.modules["h3_live_under_test"] = live
spec.loader.exec_module(live)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg and ffprobe on the PATH")
FPS, H, W, SR = 24, 48, 64, 32000


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries", "stream=codec_type,nb_read_frames,start_time", "-of", "json", str(path)],
                         capture_output=True, check=True, text=True)
    return {stream["codec_type"]: stream for stream in json.loads(out.stdout)["streams"]}


def clip(frames=17, value=0):
    video = torch.full((3, frames, H, W), value, dtype=torch.uint8)
    audio = (np.sin(np.linspace(0, 200, round(frames / FPS * SR) + 1000)) * 0.3).astype(np.float32)[:, None].repeat(2, 1)  # with a tail, as H3's
    return {"x": video, "audio": audio, "audio_sampling_rate": SR}


class FakePipeline:
    def __init__(self):
        self.calls, self.encoded, self._interrupt, self.audio_only, self.fixed_prompt = [], [], False, False, None
        self.stop_after = None

    def generate(self, input_prompt="", seed=0, frame_num=17, fps=24, image_mode=0, input_video=None, window_start_frame_no=0, prefix_frames_count=0, **kwargs):
        self.calls.append((input_prompt, seed))
        if self.stop_after is not None and len(self.calls) > self.stop_after:
            return None  # WanGP's abort: the pipeline returns None
        return clip(frame_num, value=(seed * 20) % 256)

    def _encode_prompt(self, prompt, presentation):
        self.encoded.append((prompt, tuple(presentation)))
        return None, None


def test_compatibility():
    assert live.compatibility_problems(FakePipeline) == []

    class Old:
        def generate(self, prompt, seed=0):
            return None

    problems = live.compatibility_problems(Old)
    assert any("input_prompt" in problem for problem in problems) and live.blocking_problems(problems)
    assert any("_encode_prompt" in problem for problem in problems)

    class NoEncode(FakePipeline):
        _encode_prompt = None

    problems = live.compatibility_problems(NoEncode)
    assert problems and not live.blocking_problems(problems)  # prompts are then encoded per clip, as WanGP does


def test_playlist(tmp_path):
    playlist = live.HlsPlaylist(tmp_path / "live.m3u8", 6, uri=lambda name: f"/files/{name}")
    playlist.add("clip_00000.ts", 5.167)
    text = (tmp_path / "live.m3u8").read_text()
    assert "#EXT-X-TARGETDURATION:6" in text and "#EXTINF:5.167,\n/files/clip_00000.ts" in text and "#EXT-X-ENDLIST" not in text
    playlist.finish()
    assert (tmp_path / "live.m3u8").read_text().endswith("#EXT-X-ENDLIST\n")


def test_frames_to_rgb():
    video = torch.zeros(3, 2, 4, 5)
    video[0] = 1.0  # red in [-1, 1]
    rgb = live.frames_to_rgb(video)
    assert rgb.shape == (2, 4, 5, 3) and rgb.dtype == np.uint8 and rgb[0, 0, 0].tolist() == [255, 128, 128]
    assert live.frames_to_rgb(torch.full((3, 2, 4, 5), 7, dtype=torch.uint8))[1, 3, 4].tolist() == [7, 7, 7]


@needs_ffmpeg
def test_encode_clip(tmp_path):
    item = clip(17, 100)
    live.encode_clip(shutil.which("ffmpeg"), live.frames_to_rgb(item["x"]), item["audio"], SR, FPS, tmp_path / "c.mp4", tmp_path / "c.ts", 10.0, "x264")
    mp4, ts = probe(tmp_path / "c.mp4"), probe(tmp_path / "c.ts")
    assert int(mp4["video"]["nb_read_frames"]) == 17 and "audio" in mp4
    assert int(ts["video"]["nb_read_frames"]) == 17 and "audio" in ts
    assert float(ts["video"]["start_time"]) == pytest.approx(10.0, abs=0.1)  # the stream's clock goes on from the previous clip
    assert not (tmp_path / "c.wav").exists()


@needs_ffmpeg
def test_session_writes_the_stream_and_the_film(tmp_path):
    session = live.LiveSession(tmp_path, FPS, 17 / FPS, "x264", shutil.which("ffmpeg"), web_uri=lambda path: "/web/" + Path(path).name)
    for index in range(3):
        session.submit(index, f"prompt {index}", index, clip(17, 60 * index), 0.5)
    session.close()
    summary = json.loads((session.dir / "session.json").read_text())
    assert summary["clips"] == 3 and summary["errors"] == 0 and summary["closed"] and summary["film"]
    assert [record["clip"] for record in summary["records"]] == [0, 1, 2]
    text = (session.dir / "live.m3u8").read_text()
    assert text.count("#EXTINF") == 3 and text.endswith("#EXT-X-ENDLIST\n") and "clip_00002.ts" in text
    assert "/web/clip_00001.ts" in (session.dir / "live_web.m3u8").read_text()
    film = probe(session.dir / "film.mp4")
    assert int(film["video"]["nb_read_frames"]) == 51 and "audio" in film
    assert all((session.dir / f"clip_{index:05d}.mp4").exists() for index in range(3))


def test_old_sessions_are_pruned(tmp_path):
    for name in ("live_20260101-000000", "live_20260102-000000", "live_20260103-000000", "other_folder"):
        (tmp_path / name).mkdir()
        if name.startswith("live_"):
            (tmp_path / name / "session.json").write_text("{}")
    (tmp_path / "live_20260104-000000").mkdir()  # not a session of this plugin (no session.json): kept
    live.prune_sessions(tmp_path, 1)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["live_20260103-000000", "live_20260104-000000", "other_folder"]


@needs_ffmpeg
def test_run_session_generates_every_clip_and_returns_the_last(tmp_path):
    pipe = FakePipeline()
    settings = live.LiveSettings(enabled=True, prompts=["a", "b"], clips=5, seed_step=3, encoder="x264")
    statuses = []
    result = live.run_session(FakePipeline.generate, pipe, {"input_prompt": "task", "seed": 10, "frame_num": 17, "fps": 24, "set_progress_status": statuses.append},
                              settings, root=tmp_path)
    assert pipe.calls == [("a", 10), ("b", 13), ("a", 16), ("b", 19), ("a", 22)]  # prompts cycled, seeds stepped
    assert pipe.encoded == [("a", ()), ("b", ()), ] and statuses[-1] == "H3 Live: clip 5/5"  # each prompt encoded once, before the first clip
    assert int(result["x"][0, 0, 0, 0]) == (22 * 20) % 256
    session = live.STATE["last_session"]
    assert session.closed and session.summary()["clips"] == 5 and live.STATE["session"] is None


@needs_ffmpeg
def test_run_session_uses_the_task_prompt_and_stops_on_request_or_abort(tmp_path):
    pipe = FakePipeline()
    result = live.run_session(FakePipeline.generate, pipe, {"input_prompt": "task", "seed": 1, "frame_num": 17, "fps": 24}, live.LiveSettings(enabled=True, encoder="x264"), root=tmp_path)
    assert pipe.calls == [("task", 1)] and result is not None  # no prompts, no count: one clip of the task's prompt
    pipe = FakePipeline()
    pipe.stop_after = 2
    result = live.run_session(FakePipeline.generate, pipe, {"input_prompt": "task", "seed": 1, "frame_num": 17, "fps": 24}, live.LiveSettings(enabled=True, clips=6, encoder="x264"), root=tmp_path)
    assert result is None and len(pipe.calls) == 3 and live.STATE["last_session"].summary()["clips"] == 2  # aborted: WanGP gets None, the clips made stay

    class Stopping(FakePipeline):
        def generate(self, *args, **kwargs):
            if len(self.calls) == 1:
                live.request_stop()
            return FakePipeline.generate(self, *args, **kwargs)

    pipe = Stopping()
    result = live.run_session(Stopping.generate, pipe, {"input_prompt": "task", "seed": 1, "frame_num": 17, "fps": 24}, live.LiveSettings(enabled=True, clips=6, encoder="x264"), root=tmp_path)
    assert len(pipe.calls) == 2 and result is not None  # stopped after the clip being generated


def test_run_session_refuses_an_unknown_result(tmp_path, monkeypatch):
    pipe = FakePipeline()
    monkeypatch.setattr(FakePipeline, "generate", lambda self, **kwargs: {"video": None})
    with pytest.raises(RuntimeError, match="returns no x"):
        live.run_session(FakePipeline.generate, pipe, {"input_prompt": "t", "frame_num": 17, "fps": 24}, live.LiveSettings(enabled=True, encoder="x264"),
                         root=tmp_path, ffmpeg=shutil.which("ffmpeg") or "ffmpeg")


def test_the_hook_passes_through_unless_live_mode_applies(monkeypatch):
    class Pipeline(FakePipeline):
        pass

    sessions = []
    monkeypatch.setattr(live, "run_session", lambda orig, pipe, kwargs, settings, **_: sessions.append(kwargs) or "session")
    assert live.install(Pipeline) == [] and Pipeline.generate._h3_live
    try:
        pipe = Pipeline()
        live.configure(None)
        assert pipe.generate(input_prompt="x")["x"] is not None and not sessions
        live.configure(live.LiveSettings(enabled=False))
        pipe.generate(input_prompt="x")
        live.configure(live.LiveSettings(enabled=True))
        for kwargs in ({"image_mode": 1}, {"input_video": torch.zeros(1)}, {"window_start_frame_no": 107}, {"prefix_frames_count": 5}):
            pipe.generate(input_prompt="x", **kwargs)  # images, continuations and later windows: as usual
        assert not sessions
        assert pipe.generate(input_prompt="x") == "session" and len(sessions) == 1
        # a start image arrives as a one-frame prefix beside image_start: a session, every clip starting on it
        start = {"image_start": torch.zeros(3, 1, 8, 8), "input_video": torch.zeros(3, 1, 8, 8), "prefix_frames_count": 1}
        assert pipe.generate(input_prompt="x", **start) == "session" and len(sessions) == 2
        pipe.generate(input_prompt="x", **dict(start, prefix_frames_count=17))  # a start image on a continued video: as usual
        assert len(sessions) == 2
        assert live.install(Pipeline) == [] and Pipeline.generate.__wrapped__ is live.STATE["orig_generate"]  # installed once
    finally:
        live.configure(None)
        live.uninstall()
    assert not hasattr(Pipeline.__dict__.get("generate"), "_h3_live")


def test_an_incompatible_pipeline_is_never_patched():
    class Old:
        def generate(self, prompt, seed=0):
            return None

    original = Old.generate
    assert live.blocking_problems(live.install(Old))
    assert Old.generate is original and not live.STATE["installed"]


@pytest.mark.parametrize("big", [True, False])
def test_transformer_stays_beside_the_vaes_during_a_session(big):
    original = {"text_encoder": ["vae", "text_encoder_2"], "text_encoder_2": ["vae", "text_encoder"]}
    offloadobj = types.SimpleNamespace(cotenants_map=original)
    with live.transformer_resident(offloadobj, big_vram=big):
        current = offloadobj.cotenants_map
        if big:
            assert current["transformer"] == "*"
        else:
            assert "transformer" not in current and all("transformer" in current[name] for name in ("audio_vae", "vae", "video_encoder"))
            assert current["text_encoder"] == ["vae", "text_encoder_2"]  # a text encoder load may still move the transformer out
    assert offloadobj.cotenants_map is original
    with live.transformer_resident(types.SimpleNamespace()):  # no mmgp map: nothing to do
        pass


def test_a_windowed_playlist_slides(tmp_path):
    playlist = live.HlsPlaylist(tmp_path / "live.m3u8", 6, window=2)
    for index in range(4):
        playlist.add(f"clip_{index:05d}.ts", 5.0)
    text = (tmp_path / "live.m3u8").read_text()
    assert "#EXT-X-MEDIA-SEQUENCE:2" in text and "clip_00001.ts" not in text and text.count("#EXTINF") == 2
    assert "#EXT-X-PLAYLIST-TYPE" not in text  # a live playlist, not an event


def test_the_pacer_keeps_a_session_at_most_lead_seconds_ahead():
    now = [100.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    pacer = live.Pacer(10.0, clock=lambda: now[0], sleep=sleep)
    assert pacer.wait(0.0, lambda: False) and not slept  # no clip yet: no clock
    pacer.started()
    assert pacer.wait(10.0, lambda: False) and not slept  # 10 s ahead: within the lead
    assert pacer.wait(15.0, lambda: False) and now[0] == pytest.approx(105.0, abs=0.06)  # waits until 10 s ahead
    assert not pacer.wait(100.0, lambda: True)  # stopped while waiting
    assert live.Pacer(0).wait(1e9, lambda: False)  # no lead: never waits


@needs_ffmpeg
def test_an_endless_session_keeps_a_window_and_stops_on_request(tmp_path):
    class Endless(FakePipeline):
        def generate(self, *args, **kwargs):
            if len(self.calls) == 6:
                live.request_stop()
            return FakePipeline.generate(self, *args, **kwargs)

    pipe = Endless()
    settings = live.LiveSettings(enabled=True, clips=-1, encoder="x264", playlist_window=2)
    statuses = []
    result = live.run_session(Endless.generate, pipe, {"input_prompt": "task", "seed": 5, "frame_num": 17, "fps": 24, "set_progress_status": statuses.append},
                              settings, root=tmp_path)
    assert result is not None and len(pipe.calls) == 7 and statuses[-1] == "H3 Live: clip 7"
    session = live.STATE["last_session"]
    summary = session.summary()
    assert summary["clips"] == 7 and summary["film"] is None  # an endless session is not joined into a film
    assert [record["segment"] for record in summary["records"]] == list(range(7))
    text = (session.dir / "live.m3u8").read_text()
    assert "#EXT-X-MEDIA-SEQUENCE:5" in text and "clip_00006.ts" in text and text.endswith("#EXT-X-ENDLIST\n")
    assert not (session.dir / "clip_00000.ts").exists() and not (session.dir / "clip_00001.mp4").exists()  # out of the window (+ 3)
    assert (session.dir / "clip_00002.ts").exists()


@needs_ffmpeg
@pytest.mark.parametrize("encoder", ["x264"])
def test_segments_continue_each_other(tmp_path, encoder):
    session = live.LiveSession(tmp_path, FPS, 17 / FPS, encoder, shutil.which("ffmpeg"))
    for index in range(3):
        session.submit(index, "p", index, clip(17, 50 * index), 0.1)
    session.close()
    starts = [float(probe(session.dir / f"clip_{index:05d}.ts")["video"]["start_time"]) for index in range(3)]
    assert starts[0] == pytest.approx(live.STREAM_START_S, abs=0.01)  # B-frames: no shifted first segment
    assert starts[1] - starts[0] == pytest.approx(17 / FPS, abs=0.01) and starts[2] - starts[1] == pytest.approx(17 / FPS, abs=0.01)


class EncodingPipeline(FakePipeline):
    """Encodes its start image every clip, as the H3 pipeline does for a start / end image."""

    def __init__(self):
        super().__init__()
        self.vae_encodes = 0

    def _encode_video(self, video, keep_all_latents=False):
        self.vae_encodes += 1
        return video.mean(dim=(2, 3), keepdim=True) + 1  # stands for the latents

    def generate(self, input_prompt="", seed=0, frame_num=17, fps=24, image_start=None, **kwargs):
        if image_start is not None:
            self.latents = self._encode_video(image_start[:, :1])
        return super().generate(input_prompt=input_prompt, seed=seed, frame_num=frame_num, fps=fps, **kwargs)


def test_the_image_encode_cache_encodes_each_picture_once():
    pipe = EncodingPipeline()
    one, other = torch.rand(3, 1, 8, 8), torch.rand(3, 1, 8, 8)
    with live.image_encode_cache(pipe) as stats:
        first = pipe._encode_video(one)
        again = pipe._encode_video(one.clone())  # the same picture, another tensor
        pipe._encode_video(other)
        pipe._encode_video(torch.rand(3, 5, 8, 8))  # not one frame: never cached
        pipe._encode_video(one, keep_all_latents=True)
        assert torch.equal(first, again) and again is not first
        again += 1  # a caller changing its latents does not change the cache
        assert torch.equal(pipe._encode_video(one), first)
    assert pipe.vae_encodes == 4 and stats == {"hits": 2, "misses": 2}
    assert "_encode_video" not in pipe.__dict__  # the method again
    pipe._encode_video(one)
    assert pipe.vae_encodes == 5


@needs_ffmpeg
@pytest.mark.parametrize("cache", [True, False])
def test_a_session_with_a_start_image_encodes_it_once(tmp_path, cache):
    pipe = EncodingPipeline()
    start = torch.rand(3, 1, 8, 8)
    settings = live.LiveSettings(enabled=True, prompts=["a", "b", "c"], encoder="x264", cache_image_encodes=cache)
    live.run_session(EncodingPipeline.generate, pipe, {"input_prompt": "task", "seed": 1, "frame_num": 17, "fps": 24, "image_start": start}, settings, root=tmp_path)
    assert pipe.vae_encodes == (1 if cache else 3) and len(pipe.calls) == 3
    assert "_encode_video" not in pipe.__dict__


def test_workloads_are_rounded_up_to_a_64th_of_their_power_of_two():
    assert live.bucket_workload(10) == 10
    for elements in (51_000 * 5120, 51_003 * 5120, 50_997 * 5120, 123_457, 2**20 + 1):
        bucketed = live.bucket_workload(elements)
        assert elements <= bucketed <= elements * 1.016 and live.bucket_workload(bucketed) == bucketed
    # prompts a few tokens apart in a clip of ~51,000 rows of 5120: one workload
    assert len({live.bucket_workload((51_000 + tokens) * 5120) for tokens in (166, 169, 173, 175)}) == 1


def test_steady_preload_buckets_the_transformer_plans_for_the_session(monkeypatch):
    plans = {"transformer": types.SimpleNamespace(workload_of=lambda args, kwargs: args[0]),
             "vae": types.SimpleNamespace(workload_of=lambda args, kwargs: args[0])}
    planner = types.SimpleNamespace(plans=plans)
    fake = types.ModuleType("mmgp.auto_preload")
    fake._planners = [planner]  # a WeakSet of AutoPreload objects in mmgp
    package = types.ModuleType("mmgp")
    package.auto_preload = fake
    monkeypatch.setitem(sys.modules, "mmgp", package)
    monkeypatch.setitem(sys.modules, "mmgp.auto_preload", fake)
    original = plans["transformer"].workload_of
    with live.steady_preload() as patched:
        assert patched == 1
        assert plans["transformer"].workload_of((51_169 * 5120,), {}) == plans["transformer"].workload_of((51_173 * 5120,), {})
        assert plans["vae"].workload_of((51_169,), {}) == 51_169  # other models as they are
        with live.steady_preload() as again:
            assert again == 0  # never bucketed twice
    assert plans["transformer"].workload_of is original


def test_steady_preload_without_mmgp_does_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, "mmgp", None)
    with live.steady_preload() as patched:
        assert patched == 0
