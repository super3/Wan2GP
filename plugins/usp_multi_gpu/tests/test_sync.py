"""CPU tests for the USP plugin's lockstep protocol (sync.py): no torch, no GPU.

Rank 0 and one helper run as threads over a real ``multiprocessing`` pipe. A fake
pipeline makes ``generate`` calls and DiT forwards through the same gates the
plugin installs. Each test checks the property that matters: every forward one
side enters, the other enters too, and every job ends with both sides idle and
ready for the next one, whatever stopped it.
"""

from __future__ import annotations

import inspect
import sys
import threading
from multiprocessing import Pipe
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # plugins/

from usp_multi_gpu import sync  # noqa: E402


@pytest.fixture(autouse=True)
def _no_torch(monkeypatch):
    monkeypatch.setattr(sync, "_reset_rank_check", lambda: None)


class Interrupted(Exception):
    pass


class FakeModel:
    def __init__(self):
        self._interrupt = False
        self.forwards = 0

    def _check_interrupt(self):
        if self._interrupt:
            raise Interrupted()


def fake_generate(self, input_prompt, frame_num=124, height=480, width=832, sampling_steps=8, seed=0,
                  sample_solver="euler", guide_phases=1, image_mode=0, window_start_frame_no=0,
                  dialogue_segment=False, audio_prompt_type="", video_prompt_type="", **kwargs):
    pass


SIG = inspect.signature(fake_generate)


def gated_generate(gate, model, record, steps, prompt="p", seed=1, fail_before_step=None, abort_before_step=None, **call):
    """One generate call: the generate gate, then ``steps`` forwards through the forward gate."""
    bound, values = sync._call_values(SIG, (None, prompt), dict(seed=seed, sampling_steps=steps, **call))
    try:
        gate.before_generate(values, bound)
    except sync._Ended:
        record.append("ended-at-generate")
        return None
    record.append(("generate", bound.arguments["input_prompt"], bound.arguments["seed"],
                   bound.arguments.get("kwargs", {}).get("n_prompt")))
    try:
        for step in range(steps):
            if abort_before_step == step:
                model._interrupt = True
            if fail_before_step == step:
                raise RuntimeError("helper out of memory")
            gate.before_forward(model)
            model.forwards += 1
    except Interrupted:
        record.append("interrupted")
        return None
    record.append("done")
    return "video"


def run_pair(rank0_job, helper_job):
    """Run rank 0's job (a callable on the Leader) and a helper serving ``helper_job``."""
    leader_end, helper_end = Pipe()
    leader = sync.Leader([leader_end])
    follower = sync.Follower(helper_end)
    helper_log, errors = [], []

    def helper():
        try:
            sync.serve(follower, lambda settings: helper_job(follower, settings, helper_log))
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=helper, daemon=True)
    thread.start()
    try:
        results = rank0_job(leader)
    finally:
        leader.shutdown()
        thread.join(5)
    assert not thread.is_alive(), "helper did not return to idle and exit on shutdown"
    assert not errors, errors
    return results, helper_log


def queue_task(leader, body, params=None):
    """What the plugin's generate_media wrapper does around one queued task."""
    leader.begin_task(params or {"prompt": "p", "seed": -1})
    try:
        return body()
    finally:
        leader.end_task()


def helper_runner(steps_per_gen, gens=1, **options):
    def job(follower, settings, log):
        model = FakeModel()
        for _ in range(gens):
            try:
                result = gated_generate(follower, model, log, steps_per_gen, prompt="helper's own", seed=999, **options)
            except sync.HelperStopped as exc:
                return str(exc)
            except RuntimeError as exc:
                return str(exc)
            if result is None:
                return None
        log.append(("forwards", model.forwards))
        return None

    return job


def test_normal_job_adopts_rank0_prompt_and_seed_and_matches_forwards():
    model = FakeModel()
    log0 = []

    def rank0(leader):
        return [queue_task(leader, lambda: gated_generate(leader, model, log0, 3, prompt="enhanced", seed=42,
                                                           n_prompt="blurry")) for _ in range(2)]

    results, helper_log = run_pair(rank0, helper_runner(3))
    assert results == ["video", "video"]
    assert helper_log.count(("generate", "enhanced", 42, "blurry")) == 2
    assert helper_log.count(("forwards", 3)) == 2
    assert model.forwards == 6


def test_sliding_windows_each_pass_the_generate_gate():
    model = FakeModel()

    def rank0(leader):
        return queue_task(leader, lambda: [gated_generate(leader, model, [], 2, seed=5, window_start_frame_no=w)
                                           for w in (0, 97)])

    def helper(follower, settings, log):
        model_h = FakeModel()
        for w in (0, 97):
            gated_generate(follower, model_h, log, 2, window_start_frame_no=w)
        log.append(("forwards", model_h.forwards))

    results, helper_log = run_pair(rank0, helper)
    assert results == ["video", "video"]
    assert ("forwards", 4) in helper_log


def test_rank0_abort_inside_denoise_stops_the_helper_at_the_same_step():
    model = FakeModel()

    def rank0(leader):
        first = queue_task(leader, lambda: gated_generate(leader, model, [], 5, abort_before_step=2))
        model._interrupt = False  # wgp resets it for the next job
        second = queue_task(leader, lambda: gated_generate(leader, model, [], 2))
        return first, second

    steps = iter([5, 2])  # the helper runs the same two tasks

    def helper(follower, settings, log):
        model_h = FakeModel()
        gated_generate(follower, model_h, log, next(steps))
        log.append(("forwards", model_h.forwards))

    results, helper_log = run_pair(rank0, helper)
    assert results == (None, "video")
    assert ("forwards", 2) in helper_log, helper_log  # stopped before forward 3, like rank 0
    assert "interrupted" in helper_log
    assert helper_log[-1] == ("forwards", 2) and helper_log[-2] == "done"


def test_rank0_stopping_between_generate_calls_ends_the_helper_at_its_next_call():
    model = FakeModel()

    def rank0(leader):
        return queue_task(leader, lambda: gated_generate(leader, model, [], 2))  # one window, then the job ends

    results, helper_log = run_pair(rank0, helper_runner(2, gens=2))
    assert results == "video"
    assert "ended-at-generate" in helper_log


def test_rank0_abort_before_any_generate_never_reaches_the_helper():
    def rank0(leader):
        return queue_task(leader, lambda: None)

    calls = []
    results, helper_log = run_pair(rank0, lambda follower, settings, log: calls.append(settings))
    assert results is None and calls == []


def test_helper_failure_before_a_forward_fails_rank0_cleanly_and_both_recover():
    model = FakeModel()

    def rank0(leader):
        with pytest.raises(sync.HelperStopped, match="out of memory"):
            queue_task(leader, lambda: gated_generate(leader, model, [], 4))
        return queue_task(leader, lambda: gated_generate(leader, model, [], 2))

    jobs = iter([helper_runner(4, fail_before_step=1), helper_runner(2)])
    results, helper_log = run_pair(rank0, lambda follower, settings, log: next(jobs)(follower, settings, log))
    assert results == "video"
    assert ("forwards", 2) in helper_log


def test_fingerprint_mismatch_is_refused_on_both_sides():
    model = FakeModel()

    def rank0(leader):
        with pytest.raises(sync.HelperStopped, match="mismatch"):
            queue_task(leader, lambda: gated_generate(leader, model, [], 2, height=480))
        return queue_task(leader, lambda: gated_generate(leader, model, [], 2))

    def helper(follower, settings, log):
        size = 720 if not log else 480
        try:
            gated_generate(follower, FakeModel(), log, 2, height=size)
        except sync.HelperStopped as exc:
            log.append("refused")
            return str(exc)

    results, helper_log = run_pair(rank0, helper)
    assert results == "video"
    assert "refused" in helper_log and "done" in helper_log
    assert model.forwards == 2  # no forward ran for the refused job


def test_generate_outside_a_queued_task_is_refused_before_contacting_helpers():
    def rank0(leader):
        with pytest.raises(sync.HelperStopped, match="queue"):
            gated_generate(leader, FakeModel(), [], 2)
        return "refused"

    results, _ = run_pair(rank0, helper_runner(2))
    assert results == "refused"


def test_helper_settings_turn_off_rank0_only_work():
    params = {"prompt": "p", "prompt_enhancer": "TI", "spatial_upsampling": "lanczos2", "client_id": "x", "seed": 3}
    out = sync.helper_settings(params)
    assert out["prompt_enhancer"] == "" and out["spatial_upsampling"] == ""
    assert "client_id" not in out and out["seed"] == 3
    assert params["prompt_enhancer"] == "TI"  # rank 0's settings are untouched


def test_a_dead_helper_marks_the_leader_broken():
    leader_end, helper_end = Pipe()
    leader = sync.Leader([leader_end])
    helper_end.close()
    leader.begin_task({})
    with pytest.raises(sync.HelperStopped, match="restart"):
        gated_generate(leader, FakeModel(), [], 1)
    assert leader.broken
    leader.end_task()
    leader.begin_task({})
    with pytest.raises(sync.HelperStopped, match="restart"):
        gated_generate(leader, FakeModel(), [], 1)


# -- queue hook ---------------------------------------------------------------

def _generate_media(task, send_cmd, state, prompt, resolution="832x480", plugin_data=None):
    return {"state": state, "prompt": prompt, "resolution": resolution}


class _RecordingLeader:
    def __init__(self):
        self.events = []

    def begin_task(self, params):
        self.events.append(("begin", params))

    def end_task(self):
        self.events.append(("end",))


def test_queue_hook_keeps_the_signature_wgp_filters_params_by():
    # wgp keeps only the task params named in inspect.signature(generate_media): the wrapper must expose the original's.
    wrapped = sync.queue_hook(_generate_media, lambda: None)
    assert list(inspect.signature(wrapped).parameters) == list(inspect.signature(_generate_media).parameters)
    params = {"state": {"gen": 1}, "prompt": "p", "resolution": "1280x704", "not_an_arg": 3}
    expected = set(inspect.signature(wrapped).parameters)
    filtered = {k: v for k, v in params.items() if k in expected}
    assert filtered == {"state": {"gen": 1}, "prompt": "p", "resolution": "1280x704"}
    assert wrapped({"params": params}, None, plugin_data={}, **filtered)["state"] == {"gen": 1}


def test_queue_hook_brackets_each_job_and_ends_on_error():
    lead = _RecordingLeader()
    wrapped = sync.queue_hook(_generate_media, lambda: lead)
    wrapped({"params": {"prompt": "a"}}, None, state=None, prompt="a")
    assert lead.events == [("begin", {"prompt": "a"}), ("end",)]

    def boom(task, send_cmd, **kwargs):
        raise ValueError("x")

    lead.events.clear()
    with pytest.raises(ValueError):
        sync.queue_hook(boom, lambda: lead)({"params": {}}, None)
    assert lead.events == [("begin", {}), ("end",)]
