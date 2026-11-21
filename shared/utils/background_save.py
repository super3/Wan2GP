# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Saving generated videos in the background (server_config "background_save"), so the next queued task starts at once.

Without it, a task ends only once its file is written: encode, audio mux, metadata, gallery entry. The GPU waits through all of
that before the next task of the queue starts. With it, generate_video hands those steps to one worker thread, in order, and
returns. The next task's GPU work then overlaps the previous file's save.

Only a caller that waits for the saves before it reports completion opens a `scope()`: the WebUI queue and the API's task worker.
Outside a scope (other callers of generate_media), `accepting()` is False and files are saved as before. At most `max_waiting`
saves wait behind the one being written: submitting another blocks, so generation never runs further ahead than that (each
waiting save holds its clip's frames in RAM)."""

from __future__ import annotations

import contextlib
import os
import queue
import threading
import time
import traceback
from typing import Callable


class BackgroundSaver:
    def __init__(self, max_waiting: int = 1):
        self._jobs: queue.Queue = queue.Queue(maxsize=max(1, int(max_waiting)))
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._pending = 0
        self._scopes = 0
        self._failures: list[tuple[str, BaseException]] = []
        self._thread: threading.Thread | None = None
        #: the last saves: {"label", "queued_t", "start_t", "end_t", "error"} (time.monotonic seconds)
        self.history: list[dict] = []

    def accepting(self) -> bool:
        """Whether a caller is waiting for the saves (an open scope): only then may a save be deferred."""
        with self._lock:
            return self._scopes > 0

    def submit(self, label: str, steps: list[Callable[[], object]], on_failure: Callable[[], object] | None = None) -> None:
        """Run `steps` in order on the worker thread, after the saves submitted before. A step that raises stops its save: the later
        steps (the gallery entry of a file that was not written) are skipped, `on_failure` runs, and the failure is reported when the
        scope closes."""
        with self._lock:
            self._pending += 1
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, daemon=True, name="wangp-background-save")
                self._thread.start()
        self._jobs.put((label, list(steps), on_failure, time.monotonic()))  # blocks while max_waiting saves wait

    def _run(self) -> None:
        while True:
            label, steps, on_failure, queued = self._jobs.get()
            record = {"label": label, "queued_t": queued, "start_t": time.monotonic(), "error": None}
            try:
                for step in steps:
                    step()
            except BaseException as error:  # noqa: BLE001 - reported to the scope, the worker goes on
                traceback.print_exc()
                record["error"] = repr(error)[:300]
                with self._lock:
                    self._failures.append((label, error))
                if on_failure is not None:
                    try:
                        on_failure()
                    except Exception:  # noqa: BLE001 - the save's own failure is the one reported
                        traceback.print_exc()
            record["end_t"] = time.monotonic()
            with self._lock:
                self.history.append(record)
                del self.history[:-64]
                self._pending -= 1
                if self._pending == 0:
                    self._idle.notify_all()

    def wait(self, timeout: float | None = None) -> list[tuple[str, BaseException]]:
        """Wait until every submitted save has run; returns the failures since the last wait, as (label, error)."""
        with self._lock:
            self._idle.wait_for(lambda: self._pending == 0, timeout)
            failures, self._failures = self._failures, []
        return failures

    @staticmethod
    def remove_placeholder(path: str) -> None:
        """Remove the empty file that reserved the name of a file whose save failed."""
        if os.path.isfile(path) and os.path.getsize(path) == 0:
            os.remove(path)

    @contextlib.contextmanager
    def scope(self, report: Callable[[str, BaseException], None] | None = None):
        """Saves may be deferred inside the block; leaving it waits for all of them and passes each failure to `report`."""
        with self._lock:
            self._scopes += 1
        try:
            yield self
        finally:
            with self._lock:
                self._scopes -= 1
            failures = self.wait()
            for label, error in failures:
                if report is not None:
                    report(label, error)
                else:
                    print(f"[Background save] {label}: {error!r}")


background_saves = BackgroundSaver()
