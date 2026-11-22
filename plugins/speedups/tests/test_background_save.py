# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""CPU tests of the background saver (background_save.py):

    python -m pytest plugins/speedups/tests -q
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from speedups.background_save import BackgroundSaver, remove_placeholder  # noqa: E402


def test_saves_run_in_order_after_submit_returns():
    saver, done = BackgroundSaver(), []
    started = time.monotonic()
    saver.submit("a", [lambda: time.sleep(0.2), lambda: done.append("a")])
    assert time.monotonic() - started < 0.1 and done == []
    saver.submit("b", [lambda: done.append("b")])
    assert saver.wait() == [] and done == ["a", "b"]
    assert [record["label"] for record in saver.history] == ["a", "b"]


def test_generation_never_runs_more_than_one_save_ahead():
    saver, release = BackgroundSaver(max_waiting=1), threading.Event()
    saver.submit("writing", [release.wait])
    time.sleep(0.05)  # the worker took it
    saver.submit("waiting", [lambda: None])
    third = threading.Thread(target=saver.submit, args=("blocked", [lambda: None]))
    third.start()
    time.sleep(0.1)
    assert third.is_alive()  # blocked behind the one waiting
    release.set()
    third.join(2)
    assert not third.is_alive() and saver.wait() == []


def test_a_failed_step_skips_the_rest_and_is_reported_once():
    saver, done, cleaned = BackgroundSaver(), [], []

    def fail():
        raise RuntimeError("disk full")

    saver.submit("clip", [fail, lambda: done.append("published")], on_failure=lambda: cleaned.append(True))
    failures = saver.wait()
    assert [(label, str(error)) for label, error in failures] == [("clip", "disk full")]
    assert done == [] and cleaned == [True] and saver.wait() == []


def test_placeholder_removal_keeps_written_files(tmp_path):
    empty, written = tmp_path / "empty.mp4", tmp_path / "written.mp4"
    empty.write_bytes(b"")
    written.write_bytes(b"video")
    remove_placeholder(str(empty))
    remove_placeholder(str(written))
    assert not empty.exists() and written.exists()
