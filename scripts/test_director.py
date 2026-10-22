# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""CPU tests for scripts/director.py (no GPU, no WanGP import): python -m unittest scripts/test_director.py

The media tests need ffmpeg on PATH (or DIRECTOR_FFMPEG) and are skipped without it.
"""
from __future__ import annotations

import json
import re
import shutil
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import director as D  # noqa: E402

HAVE_FFMPEG = bool(shutil.which("ffmpeg") or D.os.environ.get("DIRECTOR_FFMPEG"))


def _plan(**overrides) -> dict:
    raw = {
        "title": "test",
        "resolution": "128x72",
        "clip_seconds": 10,
        "style": "a moody night drama",
        "subjects": [{"image": "a.png", "description": "the woman from {picture}"}, {"image": "b.png", "description": "the dog"}],
        "storyboard": {"shot_seconds": 1.0, "shots_per_board": 3},
        "shots": [{"keyframe": f"Shot {n} keyframe.", "action": f"action {n} happens."} for n in range(1, 6)],
    }
    raw.update(overrides)
    return raw


def _write_wav(path: Path, seconds: float, rate: int = 8000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(struct.pack("<h", i % 30000) for i in range(int(seconds * rate))))


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="director-test-"))
        from PIL import Image

        for name, color in (("a.png", (200, 50, 50)), ("b.png", (50, 50, 200))):
            Image.new("RGB", (64, 36), color).save(self.tmp / name)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def plan(self, **overrides) -> dict:
        return D.normalize_plan(_plan(**overrides), self.tmp)


class TimingTests(unittest.TestCase):
    def test_fit_frames_lands_on_h3_lengths(self):
        rule = D.H3_FRAMES
        self.assertEqual(D.fit_frames(120, rule), 124)
        self.assertEqual(D.fit_frames(240, rule), 243)
        self.assertEqual(D.fit_frames(360, rule), 362)
        self.assertEqual(D.fit_frames(10, rule), 107)
        self.assertEqual(D.fit_frames(125, rule, "floor"), 124)
        self.assertEqual(D.fit_frames(125, rule, "ceil"), 141)
        for requested in range(1, 500):
            for mode in ("nearest", "ceil", "floor"):
                frames = D.fit_frames(requested, rule, mode)
                self.assertGreaterEqual(frames, 107)
                self.assertEqual((frames - 5) % 17, 0, (requested, mode, frames))

    def test_shot_time(self):
        self.assertEqual(D.shot_time(0), "00:00.000")
        self.assertEqual(D.shot_time(1), "00:01.000")
        self.assertEqual(D.shot_time(65.5), "01:05.500")
        self.assertEqual(D.shot_time(0.3333), "00:00.333")

    def test_estimate_matches_measurements_and_grows(self):
        for seconds, measured in D.MEASURED_GEN_S:
            self.assertAlmostEqual(D.estimate_gen_s(seconds), measured)
        values = [D.estimate_gen_s(s) for s in (3, 5, 7, 10, 12, 15, 20)]
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(D.estimate_gen_s(10.13, 2.0), 62.6)


class PlanTests(TempDirTest):
    def test_defaults_and_paths(self):
        plan = self.plan()
        self.assertEqual(plan["model_type"], D.DEFAULT_MODEL)
        self.assertEqual(plan["subjects"][0]["image"], str(self.tmp / "a.png"))
        self.assertEqual(plan["missing_files"], [])
        self.assertEqual([shot["frames"] for shot in plan["shots"]], [243] * 5)
        self.assertEqual(plan["shots"][0]["subjects"], [1, 2])
        self.assertEqual(plan["audio"]["cut_fade_ms"], 40)

    def test_missing_files_are_listed_not_fatal(self):
        plan = self.plan(subjects=[{"image": "nope.png"}])
        self.assertEqual(len(plan["missing_files"]), 1)

    def test_invalid_plans(self):
        cases = {
            "no shots": {"shots": []},
            "unknown subject": {"shots": [{"keyframe": "k", "action": "a", "subjects": [3]}]},
            "kept without soundtrack": {"audio": {"mode": "kept"}},
            "bed without bed": {"audio": {"mode": "bed"}},
            "bad mode": {"audio": {"mode": "loud"}},
            "no keyframe": {"shots": [{"action": "a"}]},
            "no action": {"shots": [{"keyframe": "k"}]},
            "storyboard too long": {"storyboard": {"shot_seconds": 2.0, "shots_per_board": 8}},
            "shot too long": {"shots": [{"keyframe": "k", "action": "a", "seconds": 20}]},
            "short storyboard shots": {"storyboard": {"shot_seconds": 0.25}},
            "bad resolution": {"resolution": "large"},
            "bad end images": {"end_images": "always"},
            "bad number": {"seed": "forty-two"},
            "unlisted subject label": {"shots": [{"keyframe": "<Subject 2> waits", "action": "a", "subjects": [1]}]},
            "picture label": {"shots": [{"keyframe": "k", "action": "match <Picture 1>"}]},
        }
        for name, override in cases.items():
            with self.subTest(name), self.assertRaises(D.PlanError):
                self.plan(**override)


class BoardTests(TempDirTest):
    def test_boards_chain_and_keyframes_avoid_cuts(self):
        plan = self.plan()
        boards = D.plan_boards(plan)
        self.assertEqual([board["shots"] for board in boards], [[0, 1, 2], [3, 4]])
        self.assertIsNone(boards[0]["anchor_shot"])
        self.assertEqual(boards[1]["anchor_shot"], 2)
        for board in boards:
            self.assertLessEqual(board["frames"], D.WINDOW_MAX_FRAMES)
            for item in board["extract"]:
                self.assertLess(item["frame"], board["frames"])
                self.assertTrue(all(abs(item["frame"] - cut) >= D.CUT_GUARD_FRAMES for cut in board["cuts"]), (item, board["cuts"]))
        self.assertEqual([item["frame"] for item in boards[1]["extract"]], [36, 60])  # slot 0 holds the anchor

    def test_supplied_keyframes_skip_the_storyboard(self):
        shots = [{"keyframe": "k1", "action": "a"}, {"keyframe_image": "a.png", "action": "b"}, {"keyframe": "k3", "action": "c"}]
        boards = D.plan_boards(self.plan(shots=shots))
        self.assertEqual([board["shots"] for board in boards], [[0, 2]])

    def test_unlinked_boards(self):
        boards = D.plan_boards(self.plan(storyboard={"shot_seconds": 1.0, "shots_per_board": 2, "link_boards": False}))
        self.assertTrue(all(board["anchor_shot"] is None for board in boards))


class PromptTests(TempDirTest):
    def assert_h3_layout(self, prompt: str):
        self.assertNotIn("\n\n", prompt)  # a blank line would start a new sliding window
        headers = [line.split(":")[0] for line in prompt.split("\n") if line.endswith(":") or ": " in line]
        for section in ("subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music"):
            self.assertIn(section, headers)
        times = re.findall(r"\[Shot (\d+)\] At (\d\d):(\d\d\.\d\d\d)", prompt)
        seconds = [int(m) * 60 + float(s) for _, m, s in times]
        self.assertEqual(seconds, sorted(seconds))
        self.assertNotRegex(prompt, r"\[Shot 1\] At")

    def test_board_prompt(self):
        plan = self.plan()
        first, second = D.plan_boards(plan)
        prompt = D.board_prompt(plan, first)
        self.assert_h3_layout(prompt)
        self.assertIn("<Subject 1> is the woman from <Picture 1>.", prompt)
        self.assertIn("<Subject 2> is the dog from <Picture 2>.", prompt)
        self.assertIn("[Shot 2] At 00:01.000, cut to shot 2 keyframe.", prompt)
        linked = D.board_prompt(plan, second)
        self.assert_h3_layout(linked)
        self.assertIn("<Picture 1> is the first frame", linked)
        self.assertIn("<Subject 1> is the woman from <Picture 2>.", linked)  # references follow the start image
        self.assertIn("[Shot 1] <Picture 1> holds still.", linked)
        self.assertIn("[Shot 3] At 00:02.000, cut to shot 5 keyframe.", linked)

    def test_clip_prompt_picture_numbering(self):
        plan = self.plan()
        shot = plan["shots"][1]
        plain = D.clip_prompt(plan, shot, has_end=False, voices=[])
        self.assert_h3_layout(plain)
        self.assertEqual(sorted(set(re.findall(r"<Picture (\d+)>", plain))), ["1", "2", "3"])
        self.assertIn("<Subject 1> is the woman from <Picture 2>.", plain)
        ending = D.clip_prompt(plan, shot, has_end=True, voices=[])
        self.assertEqual(sorted(set(re.findall(r"<Picture (\d+)>", ending))), ["1", "2", "3", "4"])
        self.assertIn("<Subject 1> is the woman from <Picture 3>.", ending)
        self.assertIn("ends exactly on <Picture 2>", ending)
        self.assertIn("[Shot 1] Starting exactly from <Picture 1>, action 2 happens.", plain)
        self.assertIn("[keyframe completion + reference generation] Action 2 happens.", plain)

    def test_dialogue_by_audio_mode(self):
        line = [{"subject": 1, "text": "Is anyone there?"}]
        shots = [{"keyframe": "k", "action": "she speaks", "dialogue": line}]
        generated = self.plan(shots=shots)
        prompt = D.clip_prompt(generated, generated["shots"][0], False, [])
        self.assertIn("<Subject 1> says (S1) <d>[English] Is anyone there?</d>.", prompt)
        _write_wav(self.tmp / "song.wav", 1)
        kept = self.plan(shots=shots, audio={"mode": "kept", "soundtrack": "song.wav"})
        prompt = D.clip_prompt(kept, kept["shots"][0], False, [])
        self.assertIn("speaking with natural lip movements synchronized", prompt)
        self.assertNotIn("<d>", prompt)


class JobTests(TempDirTest):
    def test_job_graph(self):
        plan = self.plan(end_images="next", settings={"prompt": "ignored", "config": ",lightvae_int8"})
        out = self.tmp / "run"
        jobs, boards = D.build_jobs(plan, out)
        by_id = {job["id"]: job for job in jobs}
        self.assertEqual([job["id"] for job in jobs], ["board0", "board1", "clip0", "clip1", "clip2", "clip3", "clip4"])
        self.assertEqual(by_id["board1"]["deps"], ["board0"])
        self.assertEqual(by_id["board1"]["settings"]["image_start"], str(D.keyframe_path(out, 2)))
        self.assertEqual(by_id["clip2"]["deps"], ["board0", "board1"])  # ends on clip 4's keyframe from board 2
        clip = by_id["clip0"]["settings"]
        self.assertEqual(clip["image_prompt_type"], "SE")
        self.assertEqual(clip["image_end"], str(D.keyframe_path(out, 1)))
        self.assertEqual(clip["video_prompt_type"], "I")
        self.assertEqual(clip["image_refs"], [str(self.tmp / "a.png"), str(self.tmp / "b.png")])
        self.assertEqual(clip["config"], ",lightvae_int8")
        self.assertNotEqual(clip["prompt"], "ignored")  # structural keys win over the passthrough settings
        self.assertEqual(by_id["clip4"]["settings"]["image_prompt_type"], "S")  # the last clip has no next keyframe
        self.assertEqual(len(by_id["board0"]["extract"]), 3)
        D.Scheduler(jobs)  # every dependency exists

    def test_voices_and_kept_soundtrack(self):
        _write_wav(self.tmp / "voice.wav", 1)
        plan = self.plan(audio={"voices": [{"subject": 1, "audio": "voice.wav"}, {"subject": 2, "audio": "voice.wav"}]})
        jobs, _ = D.build_jobs(plan, self.tmp / "run")
        clip = next(job for job in jobs if job["id"] == "clip0")["settings"]
        self.assertEqual(clip["audio_prompt_type"], "AB")
        self.assertIn("<Audio 2> is the voice of <Subject 2>.", clip["prompt"])
        _write_wav(self.tmp / "song.wav", 25)
        kept = self.plan(audio={"mode": "kept", "soundtrack": "song.wav", "voices": [{"subject": 1, "audio": "voice.wav"}]})
        jobs, _ = D.build_jobs(kept, self.tmp / "run", soundtrack_seconds=25.0)
        clips = [job["settings"] for job in jobs if job["kind"] == "clip"]
        self.assertEqual([clip["audio_prompt_type"] for clip in clips], ["AS", "AS", "AS", "", ""])  # 25 s covers 3 starts
        self.assertNotIn("<Audio", clips[0]["prompt"])  # a kept soundtrack is not a reference

    def test_shot_without_subjects_has_no_references(self):
        plan = self.plan(shots=[{"keyframe": "k", "action": "a", "subjects": []}])
        jobs, _ = D.build_jobs(plan, self.tmp / "run")
        clip = jobs[-1]["settings"]
        self.assertEqual((clip["video_prompt_type"], clip.get("image_refs")), ("", None))


class SchedulingTests(unittest.TestCase):
    def jobs(self, clips=4):
        jobs = [{"id": "board0", "kind": "board", "priority": [0, 0], "deps": [], "seconds": 5.17}]
        jobs += [{"id": f"clip{k}", "kind": "clip", "index": k, "priority": [1, k], "deps": ["board0"], "seconds": 10.13} for k in range(clips)]
        return jobs

    def test_simulation(self):
        done = D.simulate(self.jobs(), 2, lambda job: 10.0 if job["kind"] == "board" else 30.0)
        self.assertEqual(done["board0"], 10.0)
        self.assertEqual([done[f"clip{k}"] for k in range(4)], [40.0, 40.0, 70.0, 70.0])
        cold = D.simulate(self.jobs(), 2, lambda job: 10.0, load_s=5.0)
        self.assertEqual(cold["board0"], 15.0)

    def test_scheduler_priority_and_unknown_deps(self):
        jobs = self.jobs(2) + [{"id": "board1", "kind": "board", "priority": [0, 1], "deps": [], "seconds": 5}]
        scheduler = D.Scheduler(jobs)
        self.assertEqual(scheduler.next_ready()["id"], "board0")
        scheduler.mark("board0", "done")
        self.assertEqual(scheduler.next_ready()["id"], "board1")
        with self.assertRaises(ValueError):
            D.Scheduler([{"id": "a", "priority": [0], "deps": ["missing"]}])

    def test_playback(self):
        play = D.playback([10, 5, 40], [10, 10, 10])  # clip 2 waits for clip 1; clip 3 is 10 s late
        self.assertEqual(play["first_frame_s"], 10)
        self.assertEqual(play["starts"], [10, 20, 40])
        self.assertEqual((play["stalls"], play["stall_s"]), (1, 10))
        buffered = D.playback([10, 5, 40], [10, 10, 10], prebuffer=3)
        self.assertEqual((buffered["first_frame_s"], buffered["stall_s"]), (40, 0))

    def test_four_gpus_stream_a_minute_in_realtime(self):
        plan = D.normalize_plan(_plan(subjects=[], shots=[{"keyframe": "k", "action": "a"}] * 6,
                                      storyboard={"shot_seconds": 1.0, "shots_per_board": 6}), Path("."))
        jobs, _ = D.build_jobs(plan, Path("run"))
        rows = {row["gpus"]: row for row in D.estimate(jobs, [3, 4])}
        self.assertGreater(rows[3]["stall_s"], 0)
        self.assertEqual(rows[4]["stall_s"], 0)


class AudioSliceTests(TempDirTest):
    def test_slices_are_contiguous_and_sample_exact(self):
        master = self.tmp / "master.wav"
        _write_wav(master, 2.5, rate=8000)
        slices = [(self.tmp / f"s{k}.wav", k * 1.0, 1.0) for k in range(4)]
        written = D.write_slices(master, slices)
        self.assertEqual(len(written), 3)  # the fourth starts after the end
        with wave.open(str(master)) as handle:
            original = handle.readframes(handle.getnframes())
        joined = b""
        for path in written:
            with wave.open(path) as handle:
                joined += handle.readframes(handle.getnframes())
        self.assertEqual(joined, original)

    def test_prepare_wav_copies_pcm16(self):
        source = self.tmp / "in.wav"
        _write_wav(source, 1.5)
        self.assertAlmostEqual(D.prepare_wav(None, source, self.tmp / "out.wav"), 1.5)


class FrameTests(TempDirTest):
    def test_save_frame_tensor(self):
        import numpy as np
        from PIL import Image

        frames = np.zeros((3, 4, 6, 8), dtype=np.uint8)
        frames[0, 2] = 255
        D.save_frame_tensor(frames, 2, self.tmp / "f.png")
        pixels = np.asarray(Image.open(self.tmp / "f.png"))
        self.assertEqual(pixels.shape, (6, 8, 3))
        self.assertTrue((pixels[..., 0] == 255).all() and (pixels[..., 1:] == 0).all())

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
    def test_extract_frame_file_is_zero_based(self):
        import numpy as np
        from PIL import Image

        ffmpeg = D.find_ffmpeg()
        for n in range(10):
            Image.new("RGB", (64, 48), (n * 25, 0, 0)).save(self.tmp / f"in_{n:02d}.png")
        video = self.tmp / "ramp.mp4"
        D.run_ffmpeg(ffmpeg, ["-framerate", 24, "-i", self.tmp / "in_%02d.png", "-c:v", "libx264", "-qp", 0, "-pix_fmt", "yuv444p", video])
        D.extract_frame_file(ffmpeg, video, 7, self.tmp / "out.png")
        red = float(np.asarray(Image.open(self.tmp / "out.png"))[..., 0].mean())
        self.assertLess(abs(red - 175), 6)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
class FakeRunTests(TempDirTest):
    def test_full_run_with_music_bed(self):
        _write_wav(self.tmp / "bed.wav", 70, rate=16000)
        plan = self.plan(storyboard={"shot_seconds": 1.0, "shots_per_board": 2}, end_images="next",
                         audio={"mode": "bed", "bed": "bed.wav", "cut_fade_ms": 50},
                         shots=[{"keyframe": f"k{n}", "action": f"a{n}", "seconds": 5} for n in range(4)])
        out = self.tmp / "run"
        logs = []
        report = D.run(plan, out, ["0", "0"], backend="fake", log=logs.append)
        self.assertEqual(sorted(p.name for p in (out / "keyframes").iterdir()), [f"kf_{k:03d}.png" for k in range(4)])
        playlist = (out / "stream" / "playlist.m3u8").read_text()
        self.assertEqual(playlist.count("#EXTINF:5.167,"), 4)
        self.assertTrue(playlist.rstrip().endswith("#EXT-X-ENDLIST"))
        self.assertEqual(playlist.count("#EXT-X-DISCONTINUITY"), 3)
        self.assertEqual([clip["index"] for clip in report["clips"]], [0, 1, 2, 3])
        lives = [clip["live_at_s"] for clip in report["clips"]]
        self.assertEqual(lives, sorted(lives))
        probe = D.subprocess.run([D.find_ffmpeg(), "-hide_banner", "-i", str(out / "film.mp4")], capture_output=True, text=True).stderr
        hours, minutes, seconds = re.search(r"Duration: (\d+):(\d+):([\d.]+)", probe).groups()
        self.assertLess(abs(int(hours) * 3600 + int(minutes) * 60 + float(seconds) - 4 * 124 / 24), 0.2)
        self.assertIn("Audio:", probe)
        saved = json.loads((out / "clips" / "clip_001.json").read_text())
        self.assertEqual(saved["image_end"], str(D.keyframe_path(out, 2)))
        self.assertTrue(any("stream: clip 4 is live" in line for line in logs))

    def test_first_job_runs_alone_unless_parallel_start(self):
        shots = [{"keyframe_image": "a.png", "action": f"a{n}", "seconds": 4.5} for n in range(2)]  # no storyboard
        for parallel, overlap in ((False, False), (True, True)):
            out = self.tmp / f"run{parallel}"
            report = D.run(self.plan(shots=shots), out, ["0", "0"], backend="fake", fake_seconds=1.0,
                           parallel_start=parallel, hls=False, log=lambda *_: None)
            first, second = report["jobs"]["clip0"], report["jobs"]["clip1"]
            overlapped = max(first["start_s"], second["start_s"]) < min(first["end_s"], second["end_s"])
            self.assertEqual(overlapped, overlap, report["jobs"])

    def test_failed_job_stops_the_run(self):
        plan = self.plan(shots=[{"keyframe_image": "missing.png", "action": "a"}])
        with self.assertRaises(D.RunError):
            D.run(plan, self.tmp / "run", ["0"], backend="fake", retries=0, log=lambda *_: None)


if __name__ == "__main__":
    unittest.main()
