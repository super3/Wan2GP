# SPDX-License-Identifier: Apache-2.0
"""Lockstep between the WanGP process (rank 0) and its USP helper processes.

Under ``torchrun`` every rank runs the same script, so the ranks stay in step by
construction. In the WanGP UI only rank 0 sees the user: it picks the task,
resolves the seed, may run the prompt enhancer, and can be aborted at any of the
pipeline's ``_check_abort`` points. The helpers (``helper.py``) are headless WanGP
sessions that run the same task. This module keeps them in step through a
localhost control connection (``multiprocessing.connection``) per helper, so that
no rank ever waits inside an NCCL collective that another rank will not join.

Protocol (tuples; rank 0 -> helper / helper -> rank 0):

- ``("task", params)``: sent at the job's first ``MiniMaxH3Pipeline.generate``
  call, once rank 0 has loaded the model (so the weights are on disk and the
  helper never downloads them concurrently). The helper submits the task to its
  own session.
- ``("gen", payload)`` / ``("ready",)`` | ``("mismatch", why)`` | ``("failed", why)``:
  at every ``generate`` call (each sliding window, each dialogue segment). The
  payload carries rank 0's resolved prompt, negative/alt prompts and seed, which
  the helper adopts, plus a fingerprint (frames, size, steps, solver, model) that
  must match the helper's own call.
- ``("fwd",)`` / ``("ready",)`` | ``("failed", why)``: before every DiT forward
  (``usp._S.sync.before_forward``), ahead of its first collective. Rank 0 checks
  its abort flag first and sends nothing if it is set.
- ``("end",)``: rank 0 always sends it when a task that reached the helpers ends,
  however it ends. A helper waiting at a ``gen`` or ``fwd`` gate stops its job
  there; a helper that already stopped (error) answers every gate with
  ``failed`` until it reads ``end``.
- ``("shutdown",)``: the helper exits.

Every collective therefore starts only after all ranks said ``ready`` for that
same forward; a stopped rank answers instead of joining. What remains is a rank
failing *inside* a forward (out of memory mid-block): the others then wait in
NCCL until its timeout (``NCCL_TIMEOUT_S``), and the plugin must be restarted.
"""

from __future__ import annotations

import copy
import functools
import inspect
from typing import Any

#: ``generate`` arguments that must be identical on every rank: they set the
#: packed sequence length and the number of forwards.
FINGERPRINT_ARGS = (
    "frame_num", "height", "width", "sampling_steps", "sample_solver", "guide_phases",
    "image_mode", "window_start_frame_no", "dialogue_segment", "audio_prompt_type", "video_prompt_type",
)
#: ``generate`` arguments rank 0 resolves that a helper could resolve differently
#: (random seed, prompt enhancer, per-repeat seeds): the helper adopts rank 0's.
ADOPTED_ARGS = ("input_prompt", "seed")
ADOPTED_KWARGS = ("alt_prompt", "n_prompt")
#: Settings a helper turns off: work after the denoise that only rank 0's saved
#: file needs, and the prompt enhancer (rank 0's enhanced prompt arrives in ``gen``).
HELPER_SETTINGS_OFF = {
    "prompt_enhancer": "",
    "spatial_upsampling": "",
    "temporal_upsampling": "",
    "film_grain_intensity": 0,
}
NCCL_TIMEOUT_S = 600


class HelperStopped(RuntimeError):
    """A helper process failed, exited or fell out of step."""


class _Ended(Exception):
    """Rank 0 ended the job while this helper waited at a gate."""


def _call_values(sig: inspect.Signature, args: tuple, kwargs: dict) -> tuple[inspect.BoundArguments, dict]:
    bound = sig.bind(*args, **kwargs)
    extra = bound.arguments.get("kwargs", {})
    named = dict(bound.arguments)
    for name, parameter in sig.parameters.items():
        if name not in named and parameter.default is not inspect.Parameter.empty:
            named[name] = parameter.default
    return bound, {**named, **{f"kw:{key}": value for key, value in extra.items()}}


def fingerprint(values: dict) -> dict:
    out = {name: values.get(name) for name in FINGERPRINT_ARGS}
    out["model_type"] = values.get("kw:model_type")
    return {key: repr(value) for key, value in out.items()}


def helper_settings(params: dict) -> dict:
    """The task settings a helper runs: rank 0's, minus rank-0-only work."""
    settings = copy.deepcopy(params)
    for key in ("client_id", "priority", "workspace_id", "mcp_agent_id"):
        settings.pop(key, None)
    for key, value in HELPER_SETTINGS_OFF.items():
        if key in settings:
            settings[key] = value
    return settings


class Leader:
    """Rank 0: drives the helpers through each task."""

    def __init__(self, conns: list) -> None:
        self.conns = list(conns)
        self.params: dict | None = None
        self.task_sent = False
        self.broken: str | None = None

    # -- control connection ------------------------------------------------
    def _send_all(self, message: tuple) -> None:
        for index, conn in enumerate(self.conns, start=1):
            try:
                conn.send(message)
            except (OSError, EOFError) as exc:
                self.broken = f"GPU helper {index} is gone ({exc})"
                raise HelperStopped(self.broken + "; restart multi-GPU in the USP tab") from exc

    def _gather(self, what: str) -> None:
        problems = []
        for index, conn in enumerate(self.conns, start=1):
            try:
                reply = conn.recv()
            except (OSError, EOFError) as exc:
                self.broken = f"GPU helper {index} exited ({exc})"
                raise HelperStopped(self.broken + "; restart multi-GPU in the USP tab") from exc
            if reply[0] != "ready":
                problems.append(f"GPU helper {index}: {reply[0]}: {reply[1] if len(reply) > 1 else ''}")
        if problems:
            raise HelperStopped(f"Multi-GPU {what} refused. " + " | ".join(problems))

    # -- job boundaries (wrapped around wgp.generate_media) ----------------
    def begin_task(self, params: dict) -> None:
        self.params, self.task_sent = params, False

    def end_task(self) -> None:
        sent, self.params, self.task_sent = self.task_sent, None, False
        if sent and self.broken is None:
            try:
                self._send_all(("end",))
            except HelperStopped:
                pass  # reported by the next task

    # -- gates ----------------------------------------------------------------
    def before_generate(self, values: dict, bound: inspect.BoundArguments) -> None:
        if self.broken is not None:
            raise HelperStopped(self.broken + "; restart multi-GPU in the USP tab")
        if self.params is None:
            raise HelperStopped("Multi-GPU (USP) only runs generations started from the WanGP queue "
                                "(a MiniMax H3 call outside a queued task cannot reach the helper GPUs)")
        if not self.task_sent:
            self._send_all(("task", helper_settings(self.params)))
            self.task_sent = True
        payload = {
            "fingerprint": fingerprint(values),
            "args": {name: values.get(name) for name in ADOPTED_ARGS},
            "kwargs": {name: values.get(f"kw:{name}") for name in ADOPTED_KWARGS if f"kw:{name}" in values},
        }
        self._send_all(("gen", payload))
        self._gather("generation")
        _reset_rank_check()

    def before_forward(self, model) -> None:
        model._check_interrupt()  # abort here, before any rank commits to this forward
        self._send_all(("fwd",))
        self._gather("denoising step")

    def shutdown(self) -> None:
        for conn in self.conns:
            try:
                conn.send(("shutdown",))
            except (OSError, EOFError):
                pass


class Follower:
    """A helper: mirrors rank 0's generate calls and forwards."""

    def __init__(self, conn) -> None:
        self.conn = conn
        self.ended = False
        self.error: str | None = None

    def _recv(self) -> tuple:
        return self.conn.recv()

    def before_generate(self, values: dict, bound: inspect.BoundArguments) -> None:
        message = self._recv()
        if message[0] == "end":
            self.ended = True
            raise _Ended()
        if message[0] != "gen":
            self.conn.send(("failed", f"expected a generation, got {message[0]!r}"))
            raise HelperStopped(f"protocol: expected gen, got {message[0]!r}")
        payload = message[1]
        mine = fingerprint(values)
        if mine != payload["fingerprint"]:
            diff = {key: (payload["fingerprint"].get(key), mine.get(key)) for key in mine
                    if mine.get(key) != payload["fingerprint"].get(key)}
            self.conn.send(("mismatch", f"rank 0 vs this GPU: {diff}"))
            raise HelperStopped(f"generation mismatch {diff}")
        for name, value in payload["args"].items():
            bound.arguments[name] = value
        if payload["kwargs"]:
            extra = dict(bound.arguments.get("kwargs", {}))
            extra.update(payload["kwargs"])
            bound.arguments["kwargs"] = extra
        _reset_rank_check()
        self.conn.send(("ready",))

    def before_forward(self, model) -> None:
        message = self._recv()
        if message[0] == "fwd":
            self.conn.send(("ready",))
            return
        if message[0] == "end":
            self.ended = True
            model._interrupt = True
            model._check_interrupt()  # GenerationInterrupted: the pipeline returns None
        self.conn.send(("failed", f"expected a denoising step, got {message[0]!r}"))
        raise HelperStopped(f"protocol: expected fwd, got {message[0]!r}")

    def note_error(self, text: str) -> None:
        self.error = text

    def drain(self) -> str:
        """After a job: answer gates with ``failed`` until rank 0's ``end``.

        Returns ``"idle"`` or ``"shutdown"``."""
        if self.ended:
            self.ended, self.error = False, None
            return "idle"
        while True:
            message = self._recv()
            if message[0] == "end":
                self.error = None
                return "idle"
            if message[0] == "shutdown":
                return "shutdown"
            if message[0] in ("gen", "fwd"):
                self.conn.send(("failed", self.error or "this GPU finished the task before rank 0"))
            # a stray "task" cannot arrive before "end": rank 0 sends one task per job


def queue_hook(original, get_leader):
    """Wrap ``wgp.generate_media`` so each queue job is a lockstep job when multi-GPU is running.

    ``functools.wraps`` is load-bearing: WanGP's queue keeps only the task params that appear in
    ``inspect.signature(generate_media)``, and ``wraps`` makes that signature the original's. A bare
    ``(task, send_cmd, *args, **kwargs)`` wrapper would have every param (``state`` included) dropped.
    """

    @functools.wraps(original)
    def generate_media(task, send_cmd, *args, **kwargs):
        lead = get_leader()
        if lead is None:
            return original(task, send_cmd, *args, **kwargs)
        lead.begin_task(task.get("params", {}) if isinstance(task, dict) else {})
        try:
            return original(task, send_cmd, *args, **kwargs)
        finally:
            lead.end_task()

    return generate_media


def serve(follower: Follower, run_task, after_task=lambda: None) -> None:
    """A helper's main loop: run each task rank 0 sends, until ``shutdown``.

    ``run_task(settings)`` runs one task to its end and returns an error text, or
    None when it succeeded or was stopped; the gates above keep it in step."""
    conn = follower.conn
    while True:
        message = conn.recv()
        if message[0] == "shutdown":
            return
        if message[0] in ("gen", "fwd"):
            conn.send(("failed", "this GPU has no task"))
            continue
        if message[0] != "task":
            continue  # the "end" of a task this helper never started
        try:
            error = run_task(message[1])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        if error:
            follower.note_error(error)
        state = follower.drain()
        after_task()
        if state == "shutdown":
            return


def _reset_rank_check() -> None:
    from . import usp

    usp._S.checked_forward = False  # one cross-rank input checksum per generation


_STATE: dict[str, Any] = {"role": None, "orig_generate": None}


def install(role: Leader | Follower) -> None:
    """Gate ``MiniMaxH3Pipeline.generate`` and USP's forward through ``role``."""
    from models.minimax_h3.pipeline import MiniMaxH3Pipeline

    from . import usp

    if _STATE["orig_generate"] is None:
        original = MiniMaxH3Pipeline.generate
        signature = inspect.signature(original)
        missing = [name for name in (*ADOPTED_ARGS, *FINGERPRINT_ARGS) if name not in signature.parameters]
        if missing or "kwargs" not in signature.parameters:
            # A decorator without functools.wraps would hide the arguments: the helpers could not adopt rank 0's
            # prompt and seed, and the ranks would diverge. Refuse instead.
            raise RuntimeError(f"MiniMaxH3Pipeline.generate no longer exposes {missing or ['**kwargs']}; update sync.py")

        @functools.wraps(original)
        def generate(self, *args, **kwargs):
            gate = _STATE["role"]
            if gate is None or not usp.is_active():
                return original(self, *args, **kwargs)
            bound, values = _call_values(signature, (self, *args), kwargs)
            try:
                gate.before_generate(values, bound)
            except _Ended:
                return None  # rank 0 stopped: like an abort
            return original(*bound.args, **bound.kwargs)

        _STATE["orig_generate"] = original
        MiniMaxH3Pipeline.generate = generate
    _STATE["role"] = role
    usp._S.sync = role


def uninstall() -> None:
    from models.minimax_h3.pipeline import MiniMaxH3Pipeline

    from . import usp

    if _STATE["orig_generate"] is not None:
        MiniMaxH3Pipeline.generate = _STATE["orig_generate"]
        _STATE["orig_generate"] = None
    _STATE["role"] = None
    usp._S.sync = None
