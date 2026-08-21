"""CPU tests for ``engine.run``'s drain loop and failure accounting.

``engine`` is the only module allowed to import WanGP, and it does so strictly
inside functions — ``boot()``, ``_runtime()``, ``_cuda_cleanup()``. Everything
below drives ``run()`` with ``sess=<fake>``, so no wgp, no torch and no CUDA are
ever touched, and the parts that are genuinely hard to get right (failure modes
11, 13, 14, 15, 16, 19) become testable on a laptop:

* the loop must be driven by ``events.get(timeout=...)``, never by
  ``SessionStream.iter``, so the wall clock is checked during a *silent*
  stretch (``shared/api.py:263-271``);
* termination requires ``job.done`` **and** ``events.closed`` **and** a drained
  queue, because ``_set_result`` fires before ``events.close()``
  (``shared/api_cli.py:94`` vs ``:112``) and the error text is in the last
  events;
* an overrun cancels cooperatively and, if the cancel does not land inside the
  grace window, latches the worker as poisoned rather than pretending the job
  merely failed.
"""

from __future__ import annotations

import queue
import threading

import pytest

from runpod_worker import config as C
from runpod_worker import engine
from runpod_worker.errors import BACKEND_FATAL, WORKER_BUSY, WorkerError


class FakeStream:
    """``shared.api.SessionStream`` as ``run()`` uses it."""

    def __init__(self, events=()):
        self._q: queue.Queue = queue.Queue()
        for event in events:
            self._q.put(event)
        self._closed = False

    def put(self, kind, data=None):
        self._q.put(Event(kind, data))

    def close(self):
        self._closed = True

    @property
    def closed(self):
        return self._closed and self._q.empty()

    def get(self, timeout=None):
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None


class Event:
    def __init__(self, kind, data=None):
        self.kind = kind
        self.data = data


class Stream:
    def __init__(self, stream, text):
        self.stream = stream
        self.text = text


class Progress:
    def __init__(self, phase, status="", progress=0, current_step=0, total_steps=0):
        self.phase = phase
        self.status = status
        self.progress = progress
        self.current_step = current_step
        self.total_steps = total_steps


class Err:
    def __init__(self, stage, message):
        self.stage = stage
        self.message = message


class FakeResult:
    def __init__(self, *, success=True, files=(), errors=(), cancelled=False):
        self.success = success
        self.generated_files = tuple(files)
        self.errors = tuple(errors)
        self.cancelled = cancelled


class FakeJob:
    def __init__(self, *, result=None, events=(), finish=True, honour_cancel=True):
        self.events = FakeStream(events)
        self._result = result or FakeResult()
        self.done = False
        self.cancelled = False
        self.released = []
        self._honour_cancel = honour_cancel
        if finish:
            self.finish()

    def finish(self):
        self.done = True
        self.events.close()

    def cancel(self):
        self.cancelled = True
        if self._honour_cancel:
            self.finish()

    def result(self, timeout=None):
        if not self.done:
            raise TimeoutError("still running")
        return self._result

    def release_output_payload(self):
        self.released.append("output")

    def release_input_payload(self):
        self.released.append("input")


class FakeSession:
    def __init__(self, job):
        self.job = job
        self.submitted = []
        self._state = {"gen": {"file_list": ["stale.mp4"], "file_settings_list": [{}],
                               "api_output_artifacts": {"old": 1}, "preview": object()}}

    def submit_task(self, settings):
        self.submitted.append(dict(settings))
        return self.job


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    engine.reset_failure_budget()
    engine.STATS["jobs_served"] = 0
    monkeypatch.setenv("WANGP_CANCEL_GRACE_S", "0")
    monkeypatch.setenv("WANGP_PROGRESS_INTERVAL_S", "0")
    monkeypatch.setattr(C, "CONFIG", C.WorkerConfig())
    yield
    engine.reset_failure_budget()


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_run_returns_a_runoutcome_and_drains_the_queue(tmp_path):
    video = tmp_path / "out.mp4"
    video.write_bytes(b"\x00" * 16)
    job = FakeJob(
        result=FakeResult(success=True, files=[str(video)]),
        events=[
            Event("started", {"tasks": 1}),
            Event("stream", Stream("stdout", "loading")),
            Event("progress", Progress("encoding_text", "Prompt 1/1", 12, 1, 4)),
            Event("progress", Progress("inference_stage_1", "Denoising", 44, 2, 4)),
            Event("status", "Loading model MiniMax H3"),
        ],
    )
    session = FakeSession(job)
    seen: list[dict] = []

    outcome = engine.run({"prompt": "x"}, budget_s=30, emit_progress=seen.append, sess=session)

    assert isinstance(outcome, engine.RunOutcome)
    assert outcome.success is True
    assert outcome.timed_out is False
    assert outcome.files == [str(video)]
    assert outcome.video_path == str(video)
    assert outcome.phase_marks.keys() == {"encoding_text", "inference_stage_1"}
    assert outcome.progress_events == 2
    assert outcome.stream_events == 1
    assert [event["phase"] for event in seen] == ["encoding_text", "inference_stage_1"]
    assert job.released == ["output", "input"]
    assert session.submitted == [{"prompt": "x"}]


def test_run_is_tuple_compatible_with_the_plan(tmp_path):
    video = tmp_path / "out.mp4"
    video.write_bytes(b"\x00" * 16)
    job = FakeJob(result=FakeResult(success=True, files=[str(video)]))
    result, timed_out, logs, phase_marks, gen_s = engine.run(
        {"prompt": "x"}, budget_s=30, sess=FakeSession(job)
    )
    assert result.success is True
    assert timed_out is False
    assert isinstance(logs, list) and isinstance(phase_marks, dict)
    assert isinstance(gen_s, float)


def test_between_job_state_is_truncated(tmp_path):
    video = tmp_path / "out.mp4"
    video.write_bytes(b"\x00" * 16)
    session = FakeSession(FakeJob(result=FakeResult(success=True, files=[str(video)])))
    engine.run({"prompt": "x"}, budget_s=30, sess=session)
    gen = session._state["gen"]
    assert gen["file_list"] == []
    assert gen["file_settings_list"] == []
    assert gen["api_output_artifacts"] == {}
    assert gen["preview"] is None


def test_jobs_served_is_counted(tmp_path):
    video = tmp_path / "out.mp4"
    video.write_bytes(b"\x00" * 16)
    before = engine.STATS.get("jobs_served", 0)
    engine.run({"prompt": "x"}, budget_s=30,
               sess=FakeSession(FakeJob(result=FakeResult(success=True, files=[str(video)]))))
    assert engine.STATS["jobs_served"] == before + 1


# ---------------------------------------------------------------------------
# Termination, cancellation, poisoning
# ---------------------------------------------------------------------------


def test_late_events_are_not_lost(tmp_path):
    """``_set_result`` fires before ``events.close()``; the error text is last."""
    job = FakeJob(result=FakeResult(success=False), finish=False)
    job.events.put("error", Err("validation", "frames_positions mismatch"))
    job.finish()
    outcome = engine.run({"prompt": "x"}, budget_s=30, sess=FakeSession(job))
    assert outcome.errors == ["[validation] frames_positions mismatch"]
    assert "validation" in outcome.stages


def test_budget_overrun_cancels_cooperatively():
    job = FakeJob(result=FakeResult(success=False, cancelled=True), finish=False)
    outcome = engine.run({"prompt": "x"}, budget_s=0, sess=FakeSession(job))
    assert job.cancelled is True
    assert outcome.timed_out is True
    assert outcome.cancelled is True


def test_cancel_that_never_lands_poisons_the_worker():
    job = FakeJob(finish=False, honour_cancel=False)
    with pytest.raises(WorkerError) as excinfo:
        engine.run({"prompt": "x"}, budget_s=0, sess=FakeSession(job))
    assert excinfo.value.code == BACKEND_FATAL
    assert excinfo.value.recycle is True
    assert engine.should_recycle() is True
    assert "cancel did not land" in (engine.recycle_reason() or "")


def test_cancel_check_can_stop_a_job_early():
    job = FakeJob(result=FakeResult(success=False, cancelled=True), finish=False)
    outcome = engine.run({"prompt": "x"}, budget_s=600, sess=FakeSession(job),
                         cancel_check=lambda: True)
    assert job.cancelled is True
    assert outcome.timed_out is True


def test_a_broken_cancel_check_does_not_fail_the_job(tmp_path):
    video = tmp_path / "out.mp4"
    video.write_bytes(b"\x00" * 16)
    job = FakeJob(result=FakeResult(success=True, files=[str(video)]))

    def boom():
        raise RuntimeError("cancel check exploded")

    outcome = engine.run({"prompt": "x"}, budget_s=30, sess=FakeSession(job), cancel_check=boom)
    assert outcome.success is True


def test_a_broken_progress_callback_does_not_fail_the_job(tmp_path):
    video = tmp_path / "out.mp4"
    video.write_bytes(b"\x00" * 16)
    job = FakeJob(result=FakeResult(success=True, files=[str(video)]),
                  events=[Event("progress", Progress("decoding", "", 90, 3, 4))])

    def boom(_payload):
        raise RuntimeError("progress exploded")

    assert engine.run({"prompt": "x"}, budget_s=30, sess=FakeSession(job),
                      emit_progress=boom).success is True


def test_second_concurrent_job_is_refused():
    engine._JOB_LOCK.acquire()
    try:
        with pytest.raises(WorkerError) as excinfo:
            engine.run({"prompt": "x"}, budget_s=1, sess=FakeSession(FakeJob()))
        assert excinfo.value.code == WORKER_BUSY
    finally:
        engine._JOB_LOCK.release()


def test_submit_runtime_error_becomes_worker_busy():
    class Refusing(FakeSession):
        def submit_task(self, settings):
            raise RuntimeError("a job is already running for this session")

    with pytest.raises(WorkerError) as excinfo:
        engine.run({"prompt": "x"}, budget_s=1, sess=Refusing(FakeJob()))
    assert excinfo.value.code == WORKER_BUSY


def test_the_job_lock_is_always_released():
    job = FakeJob(finish=False, honour_cancel=False)
    with pytest.raises(WorkerError):
        engine.run({"prompt": "x"}, budget_s=0, sess=FakeSession(job))
    assert engine._JOB_LOCK.acquire(blocking=False) is True
    engine._JOB_LOCK.release()


# ---------------------------------------------------------------------------
# Failure classification and the recycle budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["CUDA error: out of memory", "cuBLAS_STATUS_ALLOC_FAILED", "device-side assert triggered",
     "an illegal memory access was encountered", "NCCL communicator was aborted"],
)
def test_poison_markers_are_detected(text):
    assert engine.is_poison(text) is True


def test_a_plain_validation_message_is_not_poison():
    assert engine.is_poison("[validation] one position per Reference Image") is False


def test_a_poisoned_generation_latches_the_recycle_flag():
    job = FakeJob(result=FakeResult(success=False, errors=[Err("generation", "CUDA error: out of memory")]))
    outcome = engine.run({"prompt": "x"}, budget_s=30, sess=FakeSession(job))
    assert outcome.poisoned is True
    assert engine.should_recycle() is True


def test_success_without_a_file_is_not_poison(tmp_path):
    """Failure mode 10: a configuration refusal, not a dead process."""
    job = FakeJob(result=FakeResult(success=True, files=[]))
    outcome = engine.run({"prompt": "x"}, budget_s=30, sess=FakeSession(job))
    assert outcome.success is True
    assert outcome.files == []
    assert outcome.poisoned is False
    assert engine.should_recycle() is False


def test_failure_budget_trips_but_does_not_latch(monkeypatch):
    """The budget is a *consecutive*-failure counter, so a success clears it.

    Latching its verdict in ``_RECYCLE_REASON`` meant every later response --
    successes included -- carried ``refresh_worker: True``, i.e. one bad run
    bought a 150-250 s weight reload on every subsequent job forever.
    """
    monkeypatch.setenv("WORKER_FAILURE_BUDGET", "2")
    monkeypatch.setattr(C, "CONFIG", C.WorkerConfig())
    engine.reset_failure_budget()
    assert engine.note_failure("generation_failed") == 1
    assert engine.should_recycle() is False
    assert engine.note_failure("generation_failed") == 2
    assert engine.should_recycle() is True
    assert "consecutive failures" in (engine.recycle_reason() or "")

    engine.note_success()
    assert engine.should_recycle() is False
    assert engine.recycle_reason() is None


def test_a_validation_outcome_is_not_charged_to_the_worker(monkeypatch):
    """WanGP rejecting a request is the caller's fault, not the worker's."""
    monkeypatch.setenv("WORKER_FAILURE_BUDGET", "2")
    monkeypatch.setattr(C, "CONFIG", C.WorkerConfig())
    engine.reset_failure_budget()
    for _ in range(4):
        job = FakeJob(result=FakeResult(
            success=False, errors=[Err("validation", "reference video is too short")]))
        outcome = engine.run({"prompt": "x"}, budget_s=30, sess=FakeSession(job))
        assert outcome.stages == ("validation",)
    assert engine.stats()["consecutive_failures"] == 0
    assert engine.should_recycle() is False


def test_note_success_clears_the_run_but_not_the_poison_latch():
    engine.note_failure("generation_failed")
    engine.note_success()
    assert engine.stats()["consecutive_failures"] == 0
    engine.mark_poisoned("cuda fault")
    engine.note_success()
    assert engine.should_recycle() is True


def test_log_tail_is_bounded(monkeypatch, tmp_path):
    # _tail_size() floors at 20 so an error tail is never useless.
    monkeypatch.setenv("WANGP_LOG_TAIL", "25")
    video = tmp_path / "out.mp4"
    video.write_bytes(b"\x00" * 16)
    events = [Event("stream", Stream("stdout", f"line {i}")) for i in range(50)]
    job = FakeJob(result=FakeResult(success=True, files=[str(video)]), events=events)
    outcome = engine.run({"prompt": "x"}, budget_s=30, sess=FakeSession(job))
    assert len(outcome.logs) == 25
    assert outcome.logs[-1].endswith("line 49")


def test_eta_is_estimated_from_the_observed_rate():
    anchors: dict = {}
    assert engine._estimate_eta(anchors, "inference_stage_1", 1, 4, 100.0) is None
    eta = engine._estimate_eta(anchors, "inference_stage_1", 3, 4, 120.0)
    assert eta == pytest.approx(10.0, abs=0.5)


def test_video_is_picked_by_extension(tmp_path):
    assert engine._video_from(["a.png", "b.wav", "c.mp4"]) == "c.mp4"
    assert engine._video_from(["a.png"]) is None


def test_engine_module_imports_without_torch_or_wgp():
    import sys

    assert "torch" not in sys.modules
    assert "wgp" not in sys.modules
    assert threading.current_thread() is threading.main_thread()


def test_preview_events_ride_the_next_progress_frame(tmp_path, monkeypatch):
    """WanGP pushes raw denoising latents as 'preview' events; the engine
    encodes them (mocked here -- encoding needs wgp + torch) and attaches the
    JPEG to the NEXT progress payload, so preview traffic can never outpace
    progress. Encode failures degrade to progress without a preview."""
    video = tmp_path / "out.mp4"
    video.write_bytes(b"\x00" * 16)
    encoded = iter(["JPEG1", None])
    monkeypatch.setattr(engine, "_encode_preview", lambda payload: next(encoded))
    job = FakeJob(
        result=FakeResult(success=True, files=[str(video)]),
        events=[
            Event("progress", Progress("encoding_text", "Prompt 1/1", 12, 1, 4)),
            Event("preview", object()),
            Event("progress", Progress("inference_stage_1", "Denoising", 44, 2, 4)),
            Event("preview", object()),          # encoder returns None -> keep last
            Event("progress", Progress("inference_stage_1", "Denoising", 66, 3, 4)),
        ],
    )
    seen: list[dict] = []
    outcome = engine.run({"prompt": "x"}, budget_s=30, emit_progress=seen.append,
                         sess=FakeSession(job))
    assert outcome.success is True
    assert "preview_jpeg" not in seen[0]          # nothing encoded yet
    assert seen[1]["preview_jpeg"] == "JPEG1"
    assert seen[2]["preview_jpeg"] == "JPEG1"     # failed encode keeps the last good one
