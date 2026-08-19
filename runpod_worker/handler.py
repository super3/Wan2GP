#!/usr/bin/env python3
"""RunPod Serverless entrypoint for WanGP / MiniMax H3.

This module is the process's ``main``. Everything it does falls into four
phases, in this order:

1. **Import.** ``sys.path`` is fixed up, the RunPod SDK is imported, and the
   worker package is imported. Nothing here touches torch, CUDA or wgp.
2. **Boot** (:func:`bootstrap`, module scope). ``config.ensure_wgp_config()``
   writes the ``wgp_config.json`` that ``import wgp`` will *replace its own
   defaults with* (the ``attention_mode`` trap, ``wgp.py:2623`` / ``:3301``),
   then ``engine.boot()`` imports wgp through ``shared.api.init``, then
   ``engine.assert_weights_complete()`` proves the volume actually carries the
   weights, then an optional warm pass pre-writes ``settings/<mt>_settings.json``
   (``wgp.py:3174``) so the first request does not pay for it. Weights are NOT
   loaded here: that is minutes long and would push worker start past RunPod's
   7-minute unhealthy threshold.
3. **Fitness checks.** Registered with the SDK and run by ``run_worker`` before
   the first job (``runpod/serverless/worker.py:38`` ->
   ``rp_fitness.run_fitness_checks``). A failing check is ``os._exit(1)``, i.e.
   the worker is marked unhealthy and replaced. A boot failure is re-raised
   there on purpose: an unhealthy worker is strictly better than one that
   answers every job with the same error while the queue keeps feeding it.
4. **Serve.** ``runpod.serverless.start({"handler": handler})``, and only under
   ``__main__`` -- importing this module for tests must never start a server.

Concurrency is one job per process, permanently: ``shared/api_cli.py:29`` holds
the module-level ``_GENERATION_LOCK`` (``shared/api.py:27``) for the whole job
and ``:48`` installs a process-global ``redirect_stdout``. No
``concurrency_modifier`` is passed to ``start()``; scale with ``max_workers``.

WanGP disclosure (``docs/API.md:9``, ``LICENSE.txt:316``): any product built on
this endpoint must clearly disclose that it uses WanGP.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

# ---------------------------------------------------------------------------
# sys.path
#
# The image's CMD runs this file as a SCRIPT
# (``python3 -u /opt/wangp/runpod_worker/handler.py``), so sys.path[0] is
# ``/opt/wangp/runpod_worker`` -- neither the repo root (needed by engine for
# ``import shared.api``) nor the package's parent (needed for
# ``import runpod_worker.*``). Both are added by hand, before any package
# import.
#
# The package directory is deliberately NOT named ``runpod``: ``shared/api.py``
# inserts the repo root at ``sys.path[0]`` while importing wgp, which would
# shadow the ``runpod`` pip package for the rest of the process.
# ---------------------------------------------------------------------------
_HANDLER_DIR = Path(__file__).resolve().parent
_PACKAGE_PARENT = _HANDLER_DIR.parent

for _candidate in (os.environ.get("WANGP_ROOT") or "/opt/wangp", str(_PACKAGE_PARENT)):
    if _candidate and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

# ---------------------------------------------------------------------------
# The SDK. Imported defensively so the module stays importable on a plain CPU
# runner that has not installed it (the CPU test tier imports handler to poke
# at ``run_job`` with a stubbed engine). Serving without it is refused loudly in
# :func:`main`.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised by whichever environment lacks the SDK
    import runpod
except Exception as _exc:  # noqa: BLE001 - ImportError or a broken install
    runpod = None  # type: ignore[assignment]
    RUNPOD_IMPORT_ERROR: Exception | None = _exc
else:
    RUNPOD_IMPORT_ERROR = None

from runpod_worker import config as C  # noqa: E402
from runpod_worker import engine, media_in, media_out, schema  # noqa: E402
from runpod_worker.errors import (  # noqa: E402
    BACKEND_FATAL,
    GENERATION_CANCELLED,
    GENERATION_FAILED,
    GENERATION_TIMEOUT,
    INTERNAL_ERROR,
    NO_OUTPUT,
    OUTPUT_TOO_LARGE,
    WANGP_VALIDATION,
    WorkerError,
    default_retryable,
)
from runpod_worker.obs import LOG  # noqa: E402

__all__ = [
    "BOOT",
    "BootState",
    "bootstrap",
    "handler",
    "run_job",
    "error_response",
    "should_recycle",
    "consecutive_failures",
    "reset_failure_counter",
    "allowed_settings_for",
    "register_fitness_checks",
    "main",
]

#: Best-effort identity of this worker, echoed on every response and log line.
WORKER_ID = (
    os.environ.get("RUNPOD_POD_ID")
    or os.environ.get("RUNPOD_ENDPOINT_ID")
    or os.environ.get("HOSTNAME")
    or "local"
)

_PROCESS_START = time.monotonic()


# ---------------------------------------------------------------------------
# Small env helpers (no dependency on config.CONFIG so they work pre-boot)
# ---------------------------------------------------------------------------


def _flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in ("1", "true", "yes", "on")


def _int_env(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------


class BootState:
    """Outcome of the one-time boot. Read by the fitness check and every job."""

    __slots__ = ("ok", "error", "traceback", "attempts", "boot_ms", "session", "warm_ms")

    def __init__(self) -> None:
        self.ok: bool = False
        self.error: BaseException | None = None
        self.traceback: str = ""
        self.attempts: int = 0
        self.boot_ms: int = 0
        self.warm_ms: int = 0
        self.session: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "attempts": self.attempts,
            "boot_ms": self.boot_ms,
            "warm_ms": self.warm_ms,
            "error": None if self.error is None else f"{type(self.error).__name__}: {self.error}",
        }


#: Process-wide boot state.
BOOT = BootState()

_BOOT_LOCK = threading.Lock()
_SCHEMA_LOCK = threading.Lock()
#: ``model_type -> (default_settings, model_def, display_name)``. Caching matters:
#: ``get_default_settings`` json.dump()s ``settings/<model_type>_settings.json``
#: on first call (``wgp.py:3174``), which is a write we never want on the
#: request path.
_SCHEMA_CACHE: dict[str, tuple[dict[str, Any], dict[str, Any], str]] = {}


def _autoboot_enabled() -> bool:
    """Whether to boot at import.

    ``WANGP_EAGER_BOOT``: ``1`` / ``0`` / ``auto`` (default ``auto``).
    ``auto`` boots when this file is the program being run (the image's CMD) or
    when the RunPod job-fetch webhook is configured (i.e. we are on the real
    platform). Under pytest neither holds, so importing the module is free.
    """
    raw = str(os.environ.get("WANGP_EAGER_BOOT", "auto")).strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return __name__ == "__main__" or bool(os.environ.get("RUNPOD_WEBHOOK_GET_JOB"))


def allowed_settings_for(session: Any, model_type: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    """``(default_settings, model_def, display_name)`` for ``model_type``, cached.

    ``default_settings`` is what ``schema.parse`` takes as ``allowed_settings``
    (``session.get_default_settings`` -- ``shared/api.py:511``). It is NOT the
    settings universe on its own; ``schema.parse`` unions it with
    ``models/_settings.json``. Returns empty mappings when the backend does not
    know this ``model_type``, leaving the error to ``schema.parse`` so the
    client gets a ``bad_request`` instead of a stack trace.
    """
    key = str(model_type)
    with _SCHEMA_LOCK:
        cached = _SCHEMA_CACHE.get(key)
    if cached is not None:
        return cached

    defaults: dict[str, Any] = {}
    mdef: dict[str, Any] = {}
    name = key
    try:
        raw_def = session.get_model_def(key)
        if isinstance(raw_def, Mapping):
            mdef = dict(raw_def)
            name = str(mdef.get("name") or key)
        raw_defaults = session.get_default_settings(key)
        if isinstance(raw_defaults, Mapping):
            defaults = dict(raw_defaults)
    except Exception as exc:  # noqa: BLE001 - unknown model_type is a client error
        LOG.warn("model_schema_unavailable", model_type=key, exc=f"{type(exc).__name__}: {exc}")
        return {}, {}, key

    entry = (defaults, mdef, name)
    with _SCHEMA_LOCK:
        _SCHEMA_CACHE[key] = entry
    return entry


def _warm(session: Any) -> int:
    """Optional warm pass. Returns milliseconds spent.

    Two things happen here, both chosen because they are *cheap* and remove work
    from the first request:

    * ``get_default_settings(model_type)`` is called once. It json.dump()s
      ``settings/<model_type>_settings.json`` on first call (``wgp.py:3174``);
      doing that at boot keeps it off a billed request and off a slow volume.
    * stale job scratch dirs from a hard restart are swept.

    Weight preloading is opt-in via ``WANGP_WARM=1`` and only runs if the engine
    exposes a warm entry point. It is minutes long: enabling it trades
    first-request latency for the risk of tripping RunPod's 7-minute
    worker-start (unhealthy) threshold.

    ``WANGP_WARM_MODEL`` is a deprecated alias for ``WANGP_WARM``: two env vars
    for one behaviour meant the README had to explain that each was "the other
    one's equivalent". It is still honoured, with a warning.
    """
    started = time.monotonic()
    model_type = C.CONFIG.model_type
    defaults, mdef, name = allowed_settings_for(session, model_type)
    LOG.info(
        "warm_settings",
        model_type=model_type,
        model_name=name,
        default_keys=len(defaults),
        model_def_keys=len(mdef),
    )
    try:
        media_in.sweep(float(_int_env("WANGP_JOB_SWEEP_AGE_S", 3600)))
    except Exception as exc:  # noqa: BLE001 - a sweep failure must not fail boot
        LOG.warn("warm_sweep_failed", exc=f"{type(exc).__name__}: {exc}")

    if _flag("WANGP_WARM_MODEL", "0") and not _flag("WANGP_WARM", "0"):
        LOG.warn(
            "warm_env_deprecated",
            note="WANGP_WARM_MODEL is a deprecated alias for WANGP_WARM; set WANGP_WARM=1",
        )
    if _flag("WANGP_WARM", "0") or _flag("WANGP_WARM_MODEL", "0"):
        warm_fn = getattr(engine, "warm", None) or getattr(engine, "preload", None)
        if callable(warm_fn):
            with LOG.span("warm_model", model_type=model_type):
                warm_fn()
        else:
            LOG.warn(
                "warm_model_unsupported",
                note="WANGP_WARM=1 but engine exposes neither warm() nor preload()",
            )
    return int((time.monotonic() - started) * 1000)


def bootstrap(*, strict: bool = False) -> BootState:
    """Config -> ``import wgp`` -> weight gate -> warm. Idempotent.

    Never raises unless ``strict=True``: a boot failure is recorded on
    :data:`BOOT`, logged as one structured JSON line, turned into an unhealthy
    worker by :func:`_fitness_boot`, and turned into a ``backend_fatal`` +
    ``refresh_worker`` response by :func:`run_job` if a job somehow arrives
    first. Re-attempted on the next call while it has not succeeded, so a
    transient volume mount can recover.
    """
    if BOOT.ok:
        return BOOT
    with _BOOT_LOCK:
        if BOOT.ok:
            return BOOT
        BOOT.attempts += 1
        started = time.monotonic()
        LOG.info(
            "boot_start",
            attempt=BOOT.attempts,
            worker_id=WORKER_ID,
            model_type=C.CONFIG.model_type,
            wangp_root=str(C.WANGP_ROOT),
            volume_root=str(C.VOLUME_ROOT),
            cli_args=list(C.CONFIG.cli_args),
        )
        try:
            # 1) The config file wgp will REPLACE its defaults with. Written
            #    before the import, and again by engine.boot() (idempotent).
            cfg_path = C.ensure_wgp_config(C.CONFIG.cli_args)

            # 2) import wgp via shared.api.init. No weights are loaded.
            session = engine.boot()
            BOOT.session = session

            # 3) Prove the weights are actually here before anything can bill a
            #    GPU for downloading 27 GB on the clock (failure mode 3).
            if _flag("WORKER_SKIP_WEIGHT_CHECK", "0"):
                LOG.warn(
                    "weight_check_skipped",
                    model_type=C.CONFIG.model_type,
                    note="WORKER_SKIP_WEIGHT_CHECK=1; missing weights will download on the clock",
                )
            else:
                with LOG.span("assert_weights_complete", model_type=C.CONFIG.model_type):
                    engine.assert_weights_complete(C.CONFIG.model_type)

            # 4) Optional warm.
            BOOT.warm_ms = _warm(session)

            BOOT.ok = True
            BOOT.error = None
            BOOT.traceback = ""
            BOOT.boot_ms = int((time.monotonic() - started) * 1000)
            LOG.info(
                "boot_complete",
                boot_ms=BOOT.boot_ms,
                warm_ms=BOOT.warm_ms,
                config_path=str(cfg_path),
                model_type=C.CONFIG.model_type,
                checkpoints=C.checkpoint_paths(),
                loras_root=C.lora_root(),
                engine_stats=_engine_stats(),
            )
        except BaseException as exc:  # noqa: BLE001 - every failure must be visible
            BOOT.ok = False
            BOOT.error = exc
            BOOT.traceback = traceback.format_exc()
            BOOT.boot_ms = int((time.monotonic() - started) * 1000)
            LOG.error(
                "boot_failed",
                attempt=BOOT.attempts,
                boot_ms=BOOT.boot_ms,
                model_type=C.CONFIG.model_type,
                exc_type=type(exc).__name__,
                exc=str(exc),
                traceback=BOOT.traceback,
                note="worker is unhealthy; fitness checks will exit(1) on the platform",
            )
            if strict:
                raise
        return BOOT


# ---------------------------------------------------------------------------
# Fitness checks (platform only -- rp_fitness.run_fitness_checks is called from
# worker.run_worker, never from the local test path)
# ---------------------------------------------------------------------------


def _fitness_boot() -> None:
    """Fail the worker when module-scope boot failed, or boot now if deferred."""
    state = bootstrap()
    if state.error is not None:
        raise RuntimeError(
            f"worker boot failed after {state.attempts} attempt(s): "
            f"{type(state.error).__name__}: {state.error}"
        )


def _fitness_gpu() -> None:
    import torch  # local import: this file must stay torch-free at import time

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible to the worker")


def _fitness_weights() -> None:
    engine.assert_weights_complete(C.CONFIG.model_type)


def _fitness_transport() -> None:
    if _flag("REQUIRE_BUCKET", "0") and not C.CONFIG.bucket_configured:
        raise RuntimeError(
            "REQUIRE_BUCKET=1 but BUCKET_ENDPOINT_URL / BUCKET_ACCESS_KEY_ID / "
            "BUCKET_SECRET_ACCESS_KEY are not all set; every job would fall back to "
            "base64 or fail with output_too_large"
        )


#: Populated by :func:`register_fitness_checks`; read by :func:`main` so a boot
#: failure still ends the process when the SDK cannot run the checks for us.
_FITNESS_REGISTERED: list[str] = []


def register_fitness_checks() -> list[str]:
    """Register the startup health gates with the SDK. Returns their names.

    No-op (and harmless) when the SDK is absent, too old to expose
    ``register_fitness_check``, or when ``WORKER_FITNESS=0``.
    """
    if _FITNESS_REGISTERED:
        return list(_FITNESS_REGISTERED)
    if runpod is None or not _flag("WORKER_FITNESS", "1"):
        return []
    register = getattr(getattr(runpod, "serverless", None), "register_fitness_check", None)
    if not callable(register):
        LOG.warn(
            "fitness_checks_unsupported",
            note="this runpod SDK has no register_fitness_check; boot failures are "
            "reported per-job instead",
        )
        return []
    checks: list[Callable[[], None]] = [_fitness_boot, _fitness_gpu, _fitness_weights, _fitness_transport]
    if _flag("WORKER_SKIP_GPU_FITNESS", "0"):
        checks.remove(_fitness_gpu)
        # Dropping OUR check is only half the gate: rp_fitness.run_fitness_checks
        # auto-registers the SDK's own GPU check
        # (rp_fitness.py:242 -> rp_gpu_fitness.auto_register_gpu_check), which
        # self-detects via nvidia-smi, plus CUDA-version / CUDA-init / benchmark
        # checks. Without this the env var reads as an escape hatch that does not
        # open. setdefault so an explicit operator value always wins.
        os.environ.setdefault("RUNPOD_SKIP_GPU_CHECK", "true")
        LOG.warn(
            "gpu_fitness_skipped",
            note="WORKER_SKIP_GPU_FITNESS=1; RUNPOD_SKIP_GPU_CHECK defaulted to 'true'",
        )
    for check in checks:
        register(check)
        _FITNESS_REGISTERED.append(check.__name__)
    LOG.info("fitness_checks_registered", checks=list(_FITNESS_REGISTERED))
    return list(_FITNESS_REGISTERED)


# ---------------------------------------------------------------------------
# Failure accounting / recycling
# ---------------------------------------------------------------------------

_FAILURE_LOCK = threading.Lock()
_consecutive_failures = 0


def consecutive_failures() -> int:
    with _FAILURE_LOCK:
        return _consecutive_failures


def _note_failure() -> int:
    global _consecutive_failures
    with _FAILURE_LOCK:
        _consecutive_failures += 1
        return _consecutive_failures


def reset_failure_counter() -> None:
    global _consecutive_failures
    with _FAILURE_LOCK:
        _consecutive_failures = 0


def should_recycle(force: bool = False) -> bool:
    """Whether this process must not serve another job.

    Three independent sources, any of which is sufficient:

    * ``force`` -- the caller already knows (a poisoned generation, a cancel
      that never landed).
    * ``engine.should_recycle()`` -- the engine's own verdict (a held
      ``_GENERATION_LOCK``, VRAM creep past the threshold, ...).
    * the consecutive-failure budget (``WORKER_FAILURE_BUDGET``, default 3).

    The RunPod SDK pops ``refresh_worker`` off the handler's return value and
    turns it into ``stopPod: True`` (``rp_job.py:268``/``:273-274``).
    """
    if force:
        return True
    verdict = getattr(engine, "should_recycle", None)
    if callable(verdict):
        try:
            if bool(verdict()):
                return True
        except Exception as exc:  # noqa: BLE001 - never let this decide by crashing
            LOG.warn("engine_should_recycle_failed", exc=f"{type(exc).__name__}: {exc}")
    budget = int(getattr(C.CONFIG, "failure_budget", 3) or 3)
    return consecutive_failures() >= budget


def _engine_stats() -> dict[str, Any]:
    stats = getattr(engine, "STATS", None)
    return dict(stats) if isinstance(stats, Mapping) else {}


# ---------------------------------------------------------------------------
# Response envelopes
#
# rp_job.run_job (rp_job.py:266-274) POPS "error" and "refresh_worker" off the
# dict we return, puts everything that is left under "output", sets
# status: FAILED when "error" is truthy and stopPod: True when
# "refresh_worker" is truthy. So: the human message goes in "error" (the client
# sees it at the top level), and every machine-readable field has to survive in
# the remainder -- which is why "error_code" and "error_message" are separate
# keys rather than a nested object under "error".
# ---------------------------------------------------------------------------


def error_response(
    job_id: str,
    code: str,
    message: str,
    *,
    retryable: bool | None = None,
    details: Any = None,
    logs: Any = None,
    recycle: bool = False,
    req: Any = None,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The failure envelope. Never raises."""
    if retryable is None:
        retryable = default_retryable(code)
    detail_list = [str(item) for item in (details or [])]
    # `[-0:]` is the whole list, so a 0 here has to short-circuit rather than slice.
    tail_size = max(0, _int_env("WORKER_LOG_TAIL", 30))
    log_tail = [str(item) for item in (logs or [])][-tail_size:] if tail_size else []
    body: dict[str, Any] = {
        "status": "error",
        "error": message,
        # "error" is popped by the SDK; keep a copy so `output` is self-describing.
        "error_message": message,
        "error_code": code,
        "retryable": bool(retryable),
        "details": detail_list,
        "logs_tail": log_tail,
        "worker_id": WORKER_ID,
    }
    if _flag("WORKER_ERROR_OBJECT", "0"):
        # Opt-in shape for clients that would rather branch on a structured
        # top-level `error`. JSON-ENCODED, not a dict: rp_job.run_job assigns
        # `run_result["error"] = error_msg` with no str() coercion
        # (rp_job.py:266-273), unlike the streaming path (rp_job.py:205) which
        # does coerce -- so a dict would reach the result endpoint as a JSON
        # object in a field that is a string everywhere else in the API. The
        # default is still the plain string the response schema documents.
        body["error"] = json.dumps(
            {
                "code": code,
                "message": message,
                "retryable": bool(retryable),
                "details": detail_list,
            },
            ensure_ascii=False,
        )
    if req is not None:
        body["model_type"] = getattr(req, "model_type", None)
        seed = getattr(req, "settings", {}).get("seed") if getattr(req, "settings", None) else None
        if seed is not None:
            body["seed"] = seed
        warnings = list(getattr(req, "warnings", []) or [])
        if warnings:
            body["warnings"] = warnings
    if metrics:
        body["metrics"] = dict(metrics)
    if recycle:
        body["refresh_worker"] = True
    LOG.error(
        "job_failed",
        job_id=job_id,
        error_code=code,
        retryable=bool(retryable),
        recycle=bool(recycle),
        message=message,
        details=detail_list[:10],
    )
    return body


#: Spec spelling, kept so engine/tests written against the plan still resolve.
_fail = error_response


def _progress(job: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    """Push one progress frame. Fire-and-forget; failures are never fatal.

    ``runpod.serverless.progress_update(job, progress)``
    (``runpod/serverless/modules/rp_progress.py:47-54``, verified against the
    installed/downloaded runpod 1.12.0 wheel) starts a daemon thread that POSTs
    ``{"status": "IN_PROGRESS", "output": progress}``. Each frame OVERWRITES
    the previous one -- this is a status field, not an append-only log. It
    dereferences ``job["id"]``, so a job dict without an id would raise here.
    """
    if runpod is None:
        return
    try:
        runpod.serverless.progress_update(dict(job), dict(payload))
    except Exception as exc:  # noqa: BLE001 - progress must never fail a job
        LOG.warn("progress_update_failed", exc=f"{type(exc).__name__}: {exc}")


def _unpack_run_result(raw: Any) -> tuple[Any, bool, list[str], dict[str, Any], float]:
    """Normalize ``engine.run``'s return value.

    The contract is the 5-tuple from the plan:
    ``(result, timed_out, logs_tail, phase_marks, generate_s)``. An object with
    equivalent attributes is accepted too, so a later engine refactor to a
    dataclass does not break the handler.
    """
    if isinstance(raw, (tuple, list)):
        if len(raw) < 5:
            raise WorkerError(
                INTERNAL_ERROR,
                f"engine.run returned {len(raw)} values; expected "
                f"(result, timed_out, logs, phase_marks, generate_s)",
            )
        result, timed_out, logs, phase_marks, generate_s = raw[:5]
    else:
        result = getattr(raw, "result", None)
        if result is None:
            raise WorkerError(
                INTERNAL_ERROR,
                f"engine.run returned an unusable {type(raw).__name__}",
            )
        timed_out = getattr(raw, "timed_out", False)
        logs = getattr(raw, "logs", None) or getattr(raw, "tail", None) or []
        phase_marks = getattr(raw, "phase_marks", None) or {}
        # engine.RunOutcome spells it `gen_s`; the plan's sketch said
        # `generate_s`. Accept either, plus `duration_s`, so the metric is never
        # silently 0.0 (which then falls back to a wall clock that excludes the
        # queue wait).
        generate_s = next(
            (
                value
                for value in (
                    getattr(raw, "generate_s", None),
                    getattr(raw, "gen_s", None),
                    getattr(raw, "duration_s", None),
                )
                if value is not None
            ),
            0.0,
        )
    return (
        result,
        bool(timed_out),
        [str(line) for line in (logs or [])],
        dict(phase_marks or {}),
        float(generate_s or 0.0),
    )


def _classify_generation_failure(messages: list[str], stages: set[str]) -> tuple[str, bool]:
    """``(error_code, poisoned)`` for a failed ``GenerationResult``."""
    blob = " ".join(messages).lower()
    poisoned = any(marker in blob for marker in schema.POISON_MARKERS)
    if poisoned:
        # Failure mode 16: the device (or the process) is not trustworthy any
        # more. The plan's response schema keeps this under `generation_failed`
        # rather than the taxonomy's `oom`, so clients branch on one code.
        return GENERATION_FAILED, True
    if "validation" in stages:
        return WANGP_VALIDATION, False
    return GENERATION_FAILED, False


def _unlink_outputs(paths: list[str]) -> None:
    """Delete the files WanGP wrote for this job, once they are delivered.

    Restricted to ``WANGP_OUTPUT_DIR``: this runs in a ``finally`` over paths
    that came out of the backend, and an ``unlink`` there is not a place for
    optimism. ``WORKER_KEEP_OUTPUTS=1`` keeps them for debugging.
    """
    if not paths or _flag("WORKER_KEEP_OUTPUTS", "0"):
        return
    try:
        # Env first, like media_in._job_root()/media_out._volume_root(): the
        # module constant is frozen at import, so a process that had
        # WANGP_OUTPUT_DIR set later (or a test) would otherwise compare against
        # the wrong root and refuse every unlink.
        root = Path(os.path.realpath(os.environ.get("WANGP_OUTPUT_DIR") or C.OUTPUT_DIR))
    except OSError:  # pragma: no cover - realpath on a sane path does not fail
        return
    for raw in paths:
        try:
            target = Path(os.path.realpath(raw))
            if root not in target.parents:
                LOG.warn("output_unlink_refused", path=str(target), output_dir=str(root))
                continue
            target.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            LOG.warn("output_unlink_failed", path=str(raw), exc=str(exc))


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------


def _preflight_transport(req: Any) -> None:
    """Say, in the first second, when this endpoint cannot deliver a big file.

    The default chain is ``presigned,rp_bucket,base64``. On the documented
    phase-1 shape (network volume, no ``BUCKET_*`` credentials) and a request
    with no ``output.presigned_url``, that degrades to base64 under
    ``WANGP_B64_OUT_MAX`` (6 MiB) -- and a 5 s 832x480 clip with audio is
    plausibly 2-8 MB. ``media_out.deliver`` finds that out AFTER a multi-minute
    generation and raises ``output_too_large``, which is the most expensive
    possible moment to learn it.

    This cannot be a hard refusal by default: a small output really does fit
    inline, and rejecting every job on a volume-only endpoint would be worse
    than the disease. So it warns by default and refuses under
    ``WORKER_REQUIRE_DELIVERABLE=1`` -- and ``REQUIRE_BUCKET=1`` still fails the
    whole worker at fitness time, which is the right gate for an operator who
    knows the outputs are always large.
    """
    opts = getattr(req, "output", None) or {}
    mode = str(opts.get("mode") or "auto").strip().lower()
    if mode not in ("auto", "b64", "base64", "inline"):
        return
    if opts.get("presigned_url") or opts.get("url"):
        return
    chain = media_out.default_chain() if mode == "auto" else ["base64"]
    if "rp_bucket" in chain and C.CONFIG.bucket_configured:
        return
    if "volume" in chain:
        return
    cap = int(getattr(C.CONFIG, "b64_out_max", 0) or 0)
    message = (
        f"this endpoint can only return outputs under {cap} B: the only transport "
        f"left in the chain ({','.join(chain)}) is base64"
    )
    fixes = [
        "pass input.output.presigned_url (a presigned PUT URL)",
        "or set BUCKET_ENDPOINT_URL + BUCKET_ACCESS_KEY_ID + "
        "BUCKET_SECRET_ACCESS_KEY + BUCKET_NAME on the endpoint",
        "or add 'volume' to WANGP_OUTPUT_CHAIN (the response then carries "
        "volume_path, readable only by something with the same volume mounted)",
    ]
    LOG.warn("transport_preflight", chain=chain, b64_out_max=cap)
    if _flag("WORKER_REQUIRE_DELIVERABLE", "0"):
        raise WorkerError(OUTPUT_TOO_LARGE, message, details=fixes, retryable=False)
    req.warnings.append(message + "; " + fixes[0])


#: Request fields that do NOT change what gets generated, so they must not
#: change the idempotency digest: ``output`` is transport (a re-signed PUT URL
#: differs on every retry of the same request) and ``runtime`` is budget /
#: priority / the idempotency key itself.
_DIGEST_IGNORED_INPUT_KEYS: frozenset[str] = frozenset({"output", "runtime"})


def request_digest(job_input: Any, model_type: str) -> str:
    """A stable hash of everything about a request that decides its output.

    Computed from the RAW input, before schema resolution, so a replayed payload
    digests identically even though ``seed`` is resolved randomly per parse.
    """
    try:
        body = {
            key: value
            for key, value in dict(job_input or {}).items()
            if key not in _DIGEST_IGNORED_INPUT_KEYS
        }
    except (TypeError, ValueError):  # pragma: no cover - job_input is a Mapping
        body = {"_unhashable": repr(job_input)}
    body["_model_type"] = model_type
    blob = json.dumps(body, sort_keys=True, ensure_ascii=False, default=repr)
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()


def _idempotency_scope(req: Any, job_input: Any, job_id: str) -> str:
    """The job-id-shaped string :func:`media_out.object_key` namespaces by."""
    idem = getattr(req, "idempotency_key", None)
    if not idem:
        return job_id
    digest = request_digest(job_input, getattr(req, "model_type", ""))
    return f"{idem}-{digest[:16]}"


def _annotate_cached(req: Any, job_input: Any) -> None:
    """Keep a cache-hit envelope honest about the one field it cannot know.

    The digest guarantees the settings match, but a request that did not pin a
    seed had one resolved randomly for THIS parse, and the cached file was made
    with a different one. Drop it rather than report a seed the video was not
    generated with.
    """
    raw = job_input.get("settings") if isinstance(job_input, Mapping) else None
    pinned = False
    if isinstance(raw, Mapping) and raw.get("seed") not in (None, ""):
        try:
            pinned = int(raw["seed"]) >= 0
        except (TypeError, ValueError):
            pinned = False
    if pinned:
        return
    getattr(req, "settings", {}).pop("seed", None)
    getattr(req, "resolved", {}).pop("seed", None)
    req.warnings.append(
        "seed omitted: this is a cached delivery of an earlier identical request, "
        "which resolved its own random seed"
    )


def run_job(
    job: Mapping[str, Any], cancelled: threading.Event | None = None
) -> dict[str, Any]:
    """Synchronous job body. Runs on a worker thread, off the SDK event loop.

    ``cancelled`` is set by :func:`handler` when the platform cancels the job.
    It is polled by ``engine.run`` every 0.5 s; without it a ``/cancel`` frees
    the SDK's concurrency slot immediately while this thread keeps burning GPU
    for the rest of the budget.
    """
    job = dict(job or {})
    job_id = str(job.get("id") or "local_test")
    started = time.monotonic()
    metrics: dict[str, Any] = {}
    logs: list[str] = []
    outputs: list[str] = []
    req: Any = None
    delivered = False

    with LOG.bind_job(job_id, worker_id=WORKER_ID):
        try:
            LOG.info("job_started", jobs_served=_engine_stats().get("jobs_served", 0))

            state = bootstrap()
            if state.error is not None:
                # The traceback names container paths, env-derived directories
                # and internal frames; it goes to the structured log every time
                # and to the client only under WORKER_DEBUG_DETAILS=1.
                LOG.error("boot_failure_reported", traceback=state.traceback[-2000:])
                raise WorkerError(
                    BACKEND_FATAL,
                    f"worker failed to boot: {type(state.error).__name__}: {state.error}",
                    details=(
                        [line for line in state.traceback.splitlines()[-8:]]
                        if _flag("WORKER_DEBUG_DETAILS", "0")
                        else ["set WORKER_DEBUG_DETAILS=1 on the endpoint for the traceback"]
                    ),
                    retryable=True,
                    recycle=True,
                )
            session = state.session or engine.boot()

            # ---- validate -------------------------------------------------
            job_input = job.get("input")
            requested = C.CONFIG.model_type
            if isinstance(job_input, Mapping) and job_input.get("model_type"):
                requested = str(job_input["model_type"]).strip()
            defaults, mdef, model_name = allowed_settings_for(session, requested)
            req = schema.parse(
                job_input,
                model_type=C.CONFIG.model_type,
                allowed_settings=defaults or None,
                model_def=mdef or None,
                cfg=C.CONFIG,
                session=session,
            )
            metrics["validate_ms"] = int((time.monotonic() - started) * 1000)
            _preflight_transport(req)
            LOG.info(
                "request_validated",
                model_type=req.model_type,
                seed=req.settings.get("seed"),
                video_length=req.settings.get("video_length"),
                steps=req.settings.get("num_inference_steps"),
                resolution=req.settings.get("resolution"),
                profile=req.profile,
                budget_s=req.budget_s,
                media_keys=sorted(req.media),
                warnings=req.warnings,
            )

            # ---- idempotency: a retry must not cost GPU seconds ------------
            # The object key is namespaced by a digest of the request itself, not
            # by the caller-chosen idempotency key alone. Three things depend on
            # that: (1) an idempotency key is caller-chosen and the key namespace
            # is the whole endpoint, so a guessed or reused key would otherwise
            # return ANOTHER caller's video, sha256 and URL for free; (2) the
            # success envelope is built from the CURRENT request, so a hit on a
            # different request would describe yesterday's video with today's
            # parameters; (3) RunPod's own /retry replays the identical payload,
            # which is exactly the case this is for and which still hits.
            key = media_out.object_key(
                _idempotency_scope(req, job_input, job_id), "output.mp4",
                model_type=req.model_type,
            )
            # Probed with or without a caller key: without one the scope is the
            # platform's job id, which is not caller-chosen, so a hit can only
            # ever be this same job replayed (RunPod's /retry keeps the id).
            if _flag("WORKER_IDEMPOTENCY", "1"):
                cached = media_out.find_existing(
                    key, cfg=C.CONFIG, request_opts=req.output
                )
                if cached:
                    metrics["idempotent_hit"] = True
                    metrics["total_s"] = round(time.monotonic() - started, 2)
                    metrics.update(_engine_stats())
                    # Never log the URL: a presigned GET carries its signature in
                    # the query string, and obs.py serializes fields verbatim.
                    LOG.info("idempotent_hit", key=key,
                             transport=cached.get("transport"))
                    _annotate_cached(req, job_input)
                    return _success_response(req, cached, metrics, model_name)

            # ---- materialize inputs ---------------------------------------
            materialized = media_in.materialize(req.media, job_id=job_id, cfg=C.CONFIG)
            req.settings.update(materialized.settings)
            req.warnings.extend(materialized.warnings)
            metrics["inputs_ms"] = int((time.monotonic() - started) * 1000) - metrics.get(
                "validate_ms", 0
            )
            metrics["input_bytes"] = materialized.total_bytes
            metrics["input_files"] = len(materialized.items)

            # ---- generate --------------------------------------------------
            def emit(payload: Mapping[str, Any]) -> None:
                _progress(job, payload)

            generate_started = time.monotonic()
            result, timed_out, logs, phase_marks, generate_s = _unpack_run_result(
                engine.run(
                    req.settings,
                    budget_s=req.budget_s,
                    emit_progress=emit,
                    cancel_check=(cancelled.is_set if cancelled is not None else None),
                )
            )
            metrics["generate_s"] = generate_s or round(time.monotonic() - generate_started, 2)
            metrics["phase_marks_s"] = phase_marks

            if timed_out:
                failures = _note_failure()
                return error_response(
                    job_id,
                    GENERATION_TIMEOUT,
                    f"generation exceeded the {req.budget_s}s budget and was cancelled",
                    retryable=True,
                    details=[f"consecutive_failures={failures}"],
                    logs=logs,
                    recycle=should_recycle(),
                    req=req,
                    metrics=metrics,
                )

            if getattr(result, "cancelled", False):
                failures = _note_failure()
                return error_response(
                    job_id,
                    GENERATION_CANCELLED,
                    "the generation was cancelled before it produced a file",
                    retryable=True,
                    details=[f"consecutive_failures={failures}"],
                    logs=logs,
                    recycle=should_recycle(),
                    req=req,
                    metrics=metrics,
                )

            if not getattr(result, "success", False):
                errors = list(getattr(result, "errors", []) or [])
                messages = [
                    f"[{getattr(err, 'stage', None) or 'error'}] {getattr(err, 'message', err)}"
                    for err in errors
                ] or ["WanGP reported failure without an error message"]
                stages = {str(getattr(err, "stage", "") or "").lower() for err in errors}
                code, poisoned = _classify_generation_failure(messages, stages)
                # Same policy as the WorkerError path below: a request WanGP
                # rejected is the caller's fault and says nothing about this
                # worker's health. wangp_validation is precisely the class
                # schema.parse cannot pre-empt (reference-video / audio duration,
                # control-video soundtrack presence -- all need ffprobe or librosa
                # on the real file), so counting it let three bad client uploads
                # in a row kill a healthy worker.
                if code != WANGP_VALIDATION:
                    _note_failure()
                return error_response(
                    job_id,
                    code,
                    "; ".join(messages),
                    retryable=poisoned,
                    details=messages,
                    logs=logs,
                    recycle=should_recycle(force=poisoned),
                    req=req,
                    metrics=metrics,
                )

            outputs = [str(path) for path in (getattr(result, "generated_files", None) or [])]
            videos = [path for path in outputs if Path(path).suffix.lower() in schema.VIDEO_EXTS]
            if not videos:
                # NOT a poisoned process (failure mode 10). generate_media returns
                # True with no file on several *configuration* paths, e.g. an
                # unsupported attention mode: wgp.py:6815-6818 does
                # send_cmd("info", ...); send_cmd("exit"); return True, and "exit"
                # is unhandled by _handle_command (shared/api_cli.py:194-226), so
                # the task counts as successful. The "info" text is in logs_tail.
                _note_failure()
                return error_response(
                    job_id,
                    NO_OUTPUT,
                    "WanGP reported success but produced no video file; this usually means a "
                    "configuration was silently refused",
                    retryable=False,
                    details=[f"generated_files={outputs}"],
                    logs=logs,
                    recycle=should_recycle(),
                    req=req,
                    metrics=metrics,
                )
            if len(videos) > 1:
                req.warnings.append(
                    f"{len(videos)} video files were produced; delivering the first "
                    f"({Path(videos[0]).name})"
                )

            # ---- deliver ---------------------------------------------------
            upload_started = time.monotonic()
            video = media_out.deliver(
                Path(videos[0]),
                key=key,
                request_opts=req.output,
                cfg=C.CONFIG,
                job_id=job_id,
                model_type=req.model_type,
            )
            delivered = True
            metrics["upload_s"] = round(time.monotonic() - upload_started, 2)
            metrics["transport"] = video.get("transport")
            metrics["total_s"] = round(time.monotonic() - started, 2)
            metrics.update(_engine_stats())

            reset_failure_counter()
            return _success_response(req, video, metrics, model_name)

        except WorkerError as exc:
            # A rejected request is the caller's fault and says nothing about the
            # health of this worker, so only server-side failures count against
            # the recycle budget.
            if exc.recycle or exc.retryable:
                _note_failure()
            return error_response(
                job_id,
                exc.code,
                exc.message,
                retryable=exc.retryable,
                details=exc.details,
                logs=logs,
                recycle=should_recycle(force=exc.recycle),
                req=req,
                metrics=metrics,
            )
        except BaseException as exc:  # noqa: BLE001 - the SDK would swallow the code
            failures = _note_failure()
            LOG.exception("unhandled_job_exception", exc, job_id=job_id)
            return error_response(
                job_id,
                INTERNAL_ERROR,
                f"{type(exc).__name__}: {exc}",
                retryable=True,
                details=(
                    traceback.format_exc().splitlines()[-8:]
                    if _flag("WORKER_DEBUG_DETAILS", "0")
                    else ["set WORKER_DEBUG_DETAILS=1 on the endpoint for the traceback"]
                ),
                logs=logs,
                recycle=should_recycle(),
                req=req,
                metrics=metrics,
            )
        finally:
            media_in.cleanup(job_id)
            # Delivered or not, the file is unreachable from outside the
            # container and the next job must not inherit the disk usage.
            _unlink_outputs(outputs)
            LOG.info(
                "job_finished",
                total_s=round(time.monotonic() - started, 2),
                delivered=delivered,
                consecutive_failures=consecutive_failures(),
            )


#: Spec spelling.
_run = run_job


def _success_response(
    req: Any,
    video: Mapping[str, Any],
    metrics: Mapping[str, Any],
    model_name: str = "",
) -> dict[str, Any]:
    """The completed envelope (see "Response schema" in the plan)."""
    settings = getattr(req, "settings", {}) or {}
    resolved = dict(getattr(req, "resolved", None) or {})
    for key in schema.RESOLVED_ECHO_KEYS:
        if key in settings:
            resolved[key] = settings[key]
    body: dict[str, Any] = {
        "status": "completed",
        "model_type": req.model_type,
        "model": {
            "model_type": req.model_type,
            "name": model_name or req.model_type,
            "profile": getattr(req, "profile", None),
            "config": settings.get("config"),
        },
        "seed": settings.get("seed"),
        "video": dict(video),
        "resolved": resolved,
        "warnings": list(getattr(req, "warnings", []) or []),
        "metrics": dict(metrics),
        "worker_id": WORKER_ID,
    }
    if should_recycle():
        body["refresh_worker"] = True
        LOG.warn(
            "recycling_after_success",
            consecutive_failures=consecutive_failures(),
            note="engine.should_recycle() or the failure budget asked for a fresh worker",
        )
    LOG.info(
        "job_completed",
        transport=body["video"].get("transport"),
        size_bytes=body["video"].get("size_bytes"),
        seed=body["seed"],
        metrics=dict(metrics),
    )
    return body


async def handler(job: Mapping[str, Any]) -> dict[str, Any]:
    """Async so the SDK event loop keeps running during a multi-minute generation.

    ``rp_job.run_job`` awaits the handler inline on the loop
    (``runpod/serverless/modules/rp_job.py:257-262``). A synchronous handler
    starves ``JobScaler.monitor_stop_signals`` for the whole job, so a client
    ``/cancel`` is never observed and SIGTERM on scale-down never drains.
    ``asyncio.to_thread`` fixes that, and carries the contextvars our logger
    binds into the worker thread. (The heartbeat is safe either way -- it runs
    in its own process.)

    Being awaitable is only half of it. ``JobScaler.stop_job``
    (``rp_scale.py:321-338``) reacts to the platform's stop channel with
    ``task.cancel()``, and ``handle_job``'s ``finally``
    (``rp_scale.py:341-368``) frees the concurrency slot on ``CancelledError``
    -- but ``asyncio.to_thread`` cannot cancel a thread that has already
    started. Without the ``threading.Event`` below the cancelled generation runs
    to the full budget on a billed GPU, its result is never sent (so no
    ``refresh_worker`` ever reaches the platform), and the next jobs pulled into
    the freed slot fail with ``worker_busy`` against ``_JOB_LOCK``.

    So: shield the thread's task from the cancellation, signal the engine
    cooperatively, and give it the same grace window the engine uses before it
    declares the process poisoned. The ``CancelledError`` is re-raised either
    way -- the platform asked for it.
    """
    cancelled = threading.Event()
    task = asyncio.ensure_future(asyncio.to_thread(run_job, job, cancelled))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cancelled.set()
        LOG.warn("job_cancel_requested", note="signalled the engine; draining the thread")
        grace = float(getattr(C.CONFIG, "cancel_grace_s", 150)) + 30.0
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(task), grace)
        raise


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

register_fitness_checks()

if _autoboot_enabled():
    # Boot at import so the ~30-60 s `import wgp` lands in worker startup rather
    # than in the first request's executionTime. Weight LOADING stays lazy: it is
    # minutes long and must not push worker start past RunPod's 7-minute
    # unhealthy threshold.
    bootstrap()
else:
    LOG.debug("eager_boot_skipped", reason=os.environ.get("WANGP_EAGER_BOOT", "auto"))


def main() -> None:
    """Start the serverless worker. Called only from ``__main__``."""
    if runpod is None:
        raise RuntimeError(
            "the runpod SDK is not importable, so this worker cannot serve: "
            f"{RUNPOD_IMPORT_ERROR}. Install it with "
            "`pip install -r runpod_worker/requirements-worker.txt`."
        )
    bootstrap()
    if BOOT.error is not None and not _FITNESS_REGISTERED:
        # Without fitness checks nothing else will mark this worker unhealthy,
        # and a worker that answers every job with the same error is worse than
        # one that never starts.
        LOG.error(
            "refusing_to_serve",
            reason="boot failed and no fitness check is registered to report it",
            error=f"{type(BOOT.error).__name__}: {BOOT.error}",
        )
        raise SystemExit(1)

    LOG.info(
        "serving",
        worker_id=WORKER_ID,
        model_type=C.CONFIG.model_type,
        boot_ok=BOOT.ok,
        boot_ms=BOOT.boot_ms,
        uptime_s=round(time.monotonic() - _PROCESS_START, 2),
        fitness_checks=list(_FITNESS_REGISTERED),
    )
    # No concurrency_modifier: one generation per process, ever. WanGP's
    # _GENERATION_LOCK (shared/api.py:27) and the process-global
    # redirect_stdout in shared/api_cli.py:48 make anything else unsafe.
    #
    # DO NOT touch sys.argv here: runpod.serverless.start() -> _set_config_args()
    # -> parser.parse_known_args() (runpod/serverless/__init__.py:87-92) still
    # needs --rp_serve_api / --test_input / --rp_log_level.
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
