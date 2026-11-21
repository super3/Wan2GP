# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""CPU tests of shared/utils/background_save.py (server_config "background_save") and of the API task worker that waits for it:

    python -m pytest shared/utils/tests/test_background_save.py -q
"""

import threading
import time
import types

import pytest

from shared.utils.background_save import BackgroundSaver, background_save_blockers, background_saves


def test_saves_run_in_order_after_submit_returns():
    saver, done = BackgroundSaver(), []
    gate = threading.Event()
    with saver.scope():
        started = time.monotonic()
        saver.submit("a", [gate.wait, lambda: done.append("a")])
        assert time.monotonic() - started < 0.5 and done == []  # the caller goes on while the file is written
        saver.submit("b", [lambda: done.append("b")])
        gate.set()
    assert done == ["a", "b"]  # leaving the scope waited for both, in order
    assert [record["label"] for record in saver.history] == ["a", "b"] and all(record["error"] is None for record in saver.history)


def test_generation_never_runs_more_than_one_save_ahead():
    saver, gate = BackgroundSaver(max_waiting=1), threading.Event()
    third_submitted = threading.Event()
    with saver.scope():
        saver.submit("writing", [gate.wait])  # being written
        time.sleep(0.05)
        saver.submit("waiting", [lambda: None])  # waits behind it

        def submit_third():
            saver.submit("blocked", [lambda: None])
            third_submitted.set()

        thread = threading.Thread(target=submit_third)
        thread.start()
        assert not third_submitted.wait(0.3)  # a third would hold a third clip's frames: the submit blocks
        gate.set()
        assert third_submitted.wait(5)
        thread.join()


def test_a_failed_step_skips_the_rest_and_is_reported():
    saver, done, reported, cleaned = BackgroundSaver(), [], [], []

    def encode():
        raise RuntimeError("encoder failed")

    with saver.scope(report=lambda label, error: reported.append((label, str(error)))):
        saver.submit("bad.mp4", [encode, lambda: done.append("published bad")], on_failure=lambda: cleaned.append("bad.mp4"))
        saver.submit("good.mp4", [lambda: done.append("published good")])
    assert done == ["published good"]  # no gallery entry for a file that was not written; the next save still runs
    assert cleaned == ["bad.mp4"] and reported == [("bad.mp4", "encoder failed")]
    assert saver.wait() == []  # reported once


def test_saves_are_deferred_only_inside_a_scope():
    saver = BackgroundSaver()
    assert not saver.accepting()  # a caller that does not wait (Deepy, the CLI queue): files are saved as before
    with saver.scope():
        assert saver.accepting()
    assert not saver.accepting()


def test_placeholder_removal_keeps_written_files(tmp_path):
    empty, written = tmp_path / "empty.mp4", tmp_path / "written.mp4"
    empty.touch()
    written.write_bytes(b"video")
    BackgroundSaver.remove_placeholder(str(empty))
    BackgroundSaver.remove_placeholder(str(written))
    BackgroundSaver.remove_placeholder(str(tmp_path / "missing.mp4"))
    assert not empty.exists() and written.exists()


PLAIN = dict(video_file=None, not_video=False, hdr=False, continued_video=None, windows=1, audio_tracks=0, audio_processing=False, flashvsr_continue=False)


def continued_video_class(monkeypatch):
    """shared.utils.video_continuation's ContinuedVideo, its media helpers stubbed (they pull in torchvision, imageio...): only its
    constructor and attributes are used here."""
    import importlib.util
    import sys
    from pathlib import Path

    stub = types.ModuleType("shared.utils.audio_video")
    stub.__getattr__ = lambda name: (lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "shared.utils.audio_video", stub)
    path = Path(__file__).resolve().parents[1] / "video_continuation.py"
    spec = importlib.util.spec_from_file_location("shared.utils.video_continuation_under_test", path)  # not registered: the real module stays
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ContinuedVideo


def test_a_plain_generated_video_may_be_saved_in_the_background(tmp_path, monkeypatch):
    ContinuedVideo = continued_video_class(monkeypatch)

    assert background_save_blockers(**PLAIN) == []
    # WanGP v20 gathers every plain video in a ContinuedVideo, even with one window: a fresh one does not block (on the first GPU run
    # this case was refused, so no file was ever saved in the background)
    fresh = ContinuedVideo(None, 0, 24, None, lambda frames, path: path, str(tmp_path), "mp4", [])
    assert background_save_blockers(**dict(PLAIN, continued_video=fresh)) == []
    fresh.file = str(tmp_path / "window1.mp4")  # a later window follows the video made so far: saved now
    assert background_save_blockers(**dict(PLAIN, continued_video=fresh)) == ["it continues a video"]
    source = ContinuedVideo(str(tmp_path / "source.mp4"), 48, 24, None, lambda frames, path: path, str(tmp_path), "mp4", [])
    assert background_save_blockers(**dict(PLAIN, continued_video=source)) == ["it continues a video"]


@pytest.mark.parametrize("change, reason", [({"video_file": "made.mp4"}, "the model wrote the video itself"), ({"not_video": True}, "not a video"),
                                            ({"hdr": True}, "HDR"), ({"windows": 2}, "several windows"), ({"audio_tracks": 1}, "source or control audio"),
                                            ({"audio_processing": True}, "soundtrack or voice processing"), ({"flashvsr_continue": True}, "FlashVSR continuation")])
def test_anything_else_is_saved_before_the_next_task(change, reason):
    assert background_save_blockers(**dict(PLAIN, **change)) == [reason]


class FakeWgp:
    """generate_media as WanGP's with background_save on: each task hands its save to the shared saver and returns."""

    def __init__(self, fail_task=None):
        self.events, self.fail_task, self.gate = [], fail_task, threading.Event()

    @staticmethod
    def validate_task(task, state):
        return dict(task["params"]), None

    def generate_media(self, task, send_cmd, plugin_data=None, **params):
        index = task["id"]
        self.events.append(("generated", index))
        assert background_saves.accepting()

        def encode():
            if index == 1:
                self.gate.wait(5)  # the first file is still being written when the second task starts
            if index == self.fail_task:
                raise OSError("disk full")
            self.events.append(("saved", index))

        background_saves.submit(f"clip{index}.mp4", [encode, lambda: send_cmd("output")])
        if index == 2:
            self.gate.set()
        return True

    @staticmethod
    def _is_edit_task_params(params):
        return False


def run_api_worker(wgp, count=2):
    from shared import api_cli

    pushed = []
    stream = types.SimpleNamespace(output_queue=types.SimpleNamespace(push=lambda command, data: pushed.append((command, data))))
    session = types.SimpleNamespace(_state={"gen": {"queue": []}}, _job_lock=threading.Lock(), _get_task_settings=lambda task: task["params"],
                                    _make_generation_error=lambda error, **kwargs: error, _build_progress_update=lambda data, **kwargs: data)
    job = types.SimpleNamespace(cancel_requested=False)
    tasks = [{"id": index, "params": {"client_id": f"c{index}"}} for index in range(1, count + 1)]
    summary = {"errors": [], "successful_tasks": 0, "failed_tasks": 0, "total_tasks": count}
    api_cli._run_tasks_worker(session, wgp, tasks, stream, job, summary)
    return summary, pushed


def test_api_job_overlaps_saves_and_waits_for_them():
    pytest.importorskip("shared.api_cli")
    wgp = FakeWgp()
    summary, pushed = run_api_worker(wgp)
    assert wgp.events.index(("generated", 2)) < wgp.events.index(("saved", 1))  # the second task ran while the first file was written
    assert ("saved", 2) in wgp.events  # the worker returned only once every file was written
    assert [command for command, _ in pushed].count("output") == 2
    assert summary["successful_tasks"] == 2 and summary["failed_tasks"] == 0 and summary["errors"] == []
    assert not background_saves.accepting()


def test_api_job_fails_the_task_whose_file_was_not_written():
    pytest.importorskip("shared.api_cli")
    summary, pushed = run_api_worker(FakeWgp(fail_task=2))
    assert summary["successful_tasks"] == 1 and summary["failed_tasks"] == 1
    assert len(summary["errors"]) == 1 and summary["errors"][0].stage == "save" and "clip2.mp4" in summary["errors"][0].message
    assert ("error", summary["errors"][0]) in pushed
