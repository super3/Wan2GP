"""Structured JSON logging for the RunPod worker.

Standard library only: no torch, no wgp, no CUDA, no third-party imports.

WHY THIS MODULE EXISTS AT ALL
-----------------------------
``print()`` does not work in this worker while a generation is running.

``shared/api_cli.py:48`` installs a process-global
``contextlib.redirect_stdout(stdout_capture)`` (and the matching
``redirect_stderr``) for the entire duration of ``_run_tasks_worker``.
``contextlib.redirect_stdout`` rebinds ``sys.stdout`` for the *whole process*,
not for the thread that entered it — so every ``print()`` anywhere in the
worker, on any thread, is swallowed into WanGP's ``_OutputCapture`` and re-emitted
as a ``stream`` event on the job's event queue. For a 5-25 minute generation that
means the container log is blank exactly when you need it, and the worker's own
diagnostics end up interleaved into the payload we hand back to the caller.

``sys.__stdout__`` is the interpreter's original stdout and is *never* touched by
``redirect_stdout``. We capture it at import time (before any job can start) into
a module-level handle and always write there. That is failure mode 12 in
``docs/RUNPOD_SERVERLESS.md``.

Note the same file at ``shared/api_cli.py:38,44`` passes
``console=sys.__stdout__ if session._console_output else None`` — i.e. WanGP's own
lines reach the real stdout only when the session was built with
``console_output=True``. Keep ``WANGP_CONSOLE=1``.

USAGE
-----
    from .obs import LOG

    LOG.info("wgp_imported", boot_ms=41002, ckpts=[...])
    with LOG.bind(job_id="60902e6c-u1"):
        LOG.warn("budget_exceeded_cancelling", budget_s=1400)

One JSON object per line, on stdout, forever. Nothing else.
"""

from __future__ import annotations

import atexit
import collections
import contextlib
import contextvars
import datetime
import json
import os
import queue
import sys
import threading
import time
import traceback
import urllib.request
from typing import Any, Iterator, Mapping, TextIO

__all__ = ["LOG", "Logger", "stdout_handle", "LEVELS", "ring_tail", "reset_ring"]


# ---------------------------------------------------------------------------
# The captured handle. THIS LINE IS THE POINT OF THE MODULE — do not move it
# into a function and do not re-read sys.stdout later.
#
# sys.__stdout__ can legitimately be None (embedded interpreters, pythonw,
# a closed fd), so fall back to whatever sys.stdout is at import time, which is
# still the real one because nothing has started a job yet.
# ---------------------------------------------------------------------------
_STDOUT: TextIO | None = sys.__stdout__ or sys.stdout

_WRITE_LOCK = threading.Lock()

LEVELS: dict[str, int] = {"debug": 10, "info": 20, "warn": 30, "warning": 30, "error": 40}

_DEFAULT_LEVEL = "info"
_MAX_FIELD_CHARS = 8192


def stdout_handle() -> TextIO | None:
    """The real stdout, captured at import time. ``None`` if there is none."""
    return _STDOUT


def _level_threshold() -> int:
    name = str(os.environ.get("WORKER_LOG_LEVEL", _DEFAULT_LEVEL)).strip().lower()
    return LEVELS.get(name, LEVELS[_DEFAULT_LEVEL])


def _max_field_chars() -> int:
    try:
        return max(256, int(os.environ.get("WORKER_LOG_MAX_FIELD", _MAX_FIELD_CHARS)))
    except (TypeError, ValueError):
        return _MAX_FIELD_CHARS


def _now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _truncate(value: Any, limit: int) -> Any:
    """Keep one runaway field (a traceback, a log tail) from eating the log."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"...<truncated {len(value) - limit} chars>"
    if isinstance(value, (list, tuple)):
        return [_truncate(item, limit) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _truncate(item, limit) for key, item in value.items()}
    return value


# Bound context (job_id, request-scoped fields). ContextVar rather than
# threading.local because the handler runs its body via asyncio.to_thread, which
# copies the *context* into the worker thread — a threading.local would not
# survive that hop.
_BOUND: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("runpod_worker_log_bound")


def _bound_fields() -> dict[str, Any]:
    try:
        return _BOUND.get()
    except LookupError:
        return {}


def _static_fields() -> dict[str, Any]:
    """Process-identity fields, cheap enough to re-read every line."""
    fields: dict[str, Any] = {}
    for key, env in (
        ("worker_id", "RUNPOD_POD_ID"),
        ("endpoint_id", "RUNPOD_ENDPOINT_ID"),
    ):
        value = os.environ.get(env)
        if value:
            fields[key] = value
    return fields


# ---------------------------------------------------------------------------
# Ring buffer.
#
# The console worker-log view has no read API (the /v2/{endpoint}/logs route is
# a worker-key INGEST endpoint), so once a worker terminates its stdout is
# gone. Every record _emit writes is therefore also kept in a bounded in-memory
# ring; handler attaches its tail to failed jobs (and to any job that asks with
# ``runtime.debug``), which makes worker history readable through the job
# status API -- the only log channel that survives the worker.
# ---------------------------------------------------------------------------
_RING_DEFAULT = 200
_RING: collections.deque[str] | None = None
_RING_LOCK = threading.Lock()


def _ring() -> collections.deque[str] | None:
    """The ring, sized once from ``WORKER_LOG_RING`` (0 disables)."""
    global _RING
    with _RING_LOCK:
        if _RING is None:
            try:
                size = int(os.environ.get("WORKER_LOG_RING", _RING_DEFAULT))
            except (TypeError, ValueError):
                size = _RING_DEFAULT
            _RING = collections.deque(maxlen=max(0, size))
        return _RING


def ring_tail(limit: int | None = None) -> list[str]:
    """The most recent ``limit`` serialized log records (all if ``None``)."""
    ring = _ring()
    if not ring:
        return []
    with _RING_LOCK:
        records = list(ring)
    if limit is not None and limit >= 0:
        records = records[-limit:] if limit else []
    return records


def reset_ring() -> None:
    """Drop the ring and re-read its size. Tests only."""
    global _RING
    with _RING_LOCK:
        _RING = None


# ---------------------------------------------------------------------------
# Log shipper.
#
# The ring cannot outlive a worker that dies DURING boot -- the exact failure
# mode (mmgp pinning killed by the container's locked-memory limit) that made
# profiles 1/2 undiagnosable from the API. When ``LOG_SHIP_URL`` is set, every
# emitted record is also queued to a daemon thread that POSTs small JSON
# batches there, so boot logs land somewhere durable within
# ``LOG_SHIP_INTERVAL_S`` (default 2 s) of being written. Best effort by
# design: a SIGKILL can still eat the last partial batch, and any HTTP failure
# is silently dropped after one direct stderr note -- shipping must never slow
# down or take down the worker.
# ---------------------------------------------------------------------------
_SHIP_QUEUE: "queue.Queue[str | None]" = queue.Queue(maxsize=10000)
_SHIP_THREAD: threading.Thread | None = None
_SHIP_LOCK = threading.Lock()
_SHIP_BATCH_MAX = 50
_SHIP_WARNED = False


def _ship_url() -> str:
    return str(os.environ.get("LOG_SHIP_URL", "")).strip()


def _ship_interval() -> float:
    try:
        return max(0.2, float(os.environ.get("LOG_SHIP_INTERVAL_S", "2")))
    except (TypeError, ValueError):
        return 2.0


def _ship_note_failure(exc: BaseException) -> None:
    """One direct line on the real stderr, once per process. Never via LOG --
    a failing shipper logging through _emit would enqueue its own failure."""
    global _SHIP_WARNED
    if _SHIP_WARNED:
        return
    _SHIP_WARNED = True
    try:
        handle = sys.__stderr__ or sys.stderr
        if handle is not None:
            handle.write(f"log shipping to LOG_SHIP_URL failed; dropping batches: {exc!r}\n")
            handle.flush()
    except Exception:  # noqa: BLE001 - last-resort path
        pass


def _ship_post(url: str, batch: list[str]) -> None:
    body = json.dumps(
        {"records": [json.loads(line) for line in batch], **_static_fields()},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=5):  # noqa: S310 - operator-set URL
        pass


def _ship_loop() -> None:  # pragma: no cover - exercised via the queue in tests
    while True:
        batch: list[str] = []
        try:
            item = _SHIP_QUEUE.get(timeout=_ship_interval())
        except queue.Empty:
            continue
        if item is None:
            return
        batch.append(item)
        deadline = time.monotonic() + _ship_interval()
        while len(batch) < _SHIP_BATCH_MAX and time.monotonic() < deadline:
            try:
                item = _SHIP_QUEUE.get_nowait()
            except queue.Empty:
                break
            if item is None:
                break
            batch.append(item)
        url = _ship_url()
        if not url:
            continue
        try:
            _ship_post(url, batch)
        except Exception as exc:  # noqa: BLE001 - shipping is best effort
            _ship_note_failure(exc)
        if item is None:
            return


def _ship_flush_at_exit() -> None:
    """Drain what is already queued, synchronously, within a small budget."""
    url = _ship_url()
    if not url:
        return
    batch: list[str] = []
    while len(batch) < _SHIP_BATCH_MAX:
        try:
            item = _SHIP_QUEUE.get_nowait()
        except queue.Empty:
            break
        if item is not None:
            batch.append(item)
    if batch:
        try:
            _ship_post(url, batch)
        except Exception as exc:  # noqa: BLE001
            _ship_note_failure(exc)


def _ship_enqueue(line: str) -> None:
    if not _ship_url():
        return
    global _SHIP_THREAD
    with _SHIP_LOCK:
        if _SHIP_THREAD is None or not _SHIP_THREAD.is_alive():
            _SHIP_THREAD = threading.Thread(
                target=_ship_loop, name="log-shipper", daemon=True
            )
            _SHIP_THREAD.start()
            atexit.register(_ship_flush_at_exit)
    try:
        _SHIP_QUEUE.put_nowait(line)
    except queue.Full:
        pass



class Logger:
    """Emits exactly one JSON object per line to the captured stdout."""

    __slots__ = ("_name",)

    def __init__(self, name: str = "runpod_worker") -> None:
        self._name = name

    # -- context binding ----------------------------------------------------

    @contextlib.contextmanager
    def bind(self, **fields: Any) -> Iterator[None]:
        """Attach ``fields`` to every record emitted inside the block."""
        merged = dict(_bound_fields())
        merged.update(fields)
        token = _BOUND.set(merged)
        try:
            yield
        finally:
            _BOUND.reset(token)

    def bind_job(self, job_id: str, **fields: Any):
        """``with LOG.bind_job(job["id"]):`` — the common case."""
        return self.bind(job_id=str(job_id), **fields)

    def set_context(self, **fields: Any) -> None:
        """Bind without a scope. For code that cannot use ``with`` (e.g. boot)."""
        merged = dict(_bound_fields())
        merged.update(fields)
        _BOUND.set(merged)

    def clear_context(self) -> None:
        _BOUND.set({})

    # -- emission -----------------------------------------------------------

    def _emit(self, level: str, event: str, fields: Mapping[str, Any]) -> None:
        if LEVELS.get(level, LEVELS[_DEFAULT_LEVEL]) < _level_threshold():
            return
        handle = _STDOUT
        limit = _max_field_chars()
        record: dict[str, Any] = {
            "ts": _now_iso(),
            "level": level,
            "logger": self._name,
            "event": str(event),
        }
        record.update(_static_fields())
        record.update(_bound_fields())
        for key, value in fields.items():
            record[str(key)] = _truncate(value, limit)
        try:
            line = json.dumps(record, default=str, ensure_ascii=False)
        except (TypeError, ValueError):  # pragma: no cover - default=str covers ~all
            line = json.dumps(
                {"ts": record["ts"], "level": "error", "logger": self._name,
                 "event": "log_serialization_failed", "original_event": str(event)}
            )
        ring = _ring()
        if ring is not None and ring.maxlen:
            with _RING_LOCK:
                ring.append(line)
        _ship_enqueue(line)
        if handle is None:
            return
        try:
            with _WRITE_LOCK:
                handle.write(line + "\n")
                handle.flush()
        except Exception:  # noqa: BLE001 - logging must never take down a job
            pass

    def debug(self, event: str, **fields: Any) -> None:
        self._emit("debug", event, fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit("info", event, fields)

    def warn(self, event: str, **fields: Any) -> None:
        self._emit("warn", event, fields)

    #: ``warning`` is an alias so stdlib-logging muscle memory does not misfire.
    warning = warn

    def error(self, event: str, **fields: Any) -> None:
        self._emit("error", event, fields)

    def exception(self, event: str, exc: BaseException | None = None, **fields: Any) -> None:
        """``error`` plus the formatted traceback of the exception in flight."""
        if exc is not None:
            fields.setdefault("exc_type", type(exc).__name__)
            fields.setdefault("exc", str(exc))
            fields.setdefault(
                "traceback",
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            )
        else:
            fields.setdefault("traceback", traceback.format_exc())
        self._emit("error", event, fields)

    # -- timing -------------------------------------------------------------

    @contextlib.contextmanager
    def span(self, event: str, **fields: Any) -> Iterator[dict[str, Any]]:
        """Time a block and emit one record when it ends, success or failure.

        The yielded dict is merged into the closing record, so a block can report
        what it actually did::

            with LOG.span("deliver") as span:
                span["transport"] = "rp_bucket"
        """
        extra: dict[str, Any] = {}
        started = time.monotonic()
        try:
            yield extra
        except BaseException as exc:
            merged = dict(fields)
            merged.update(extra)
            merged["duration_ms"] = int((time.monotonic() - started) * 1000)
            merged["outcome"] = "error"
            merged.setdefault("exc_type", type(exc).__name__)
            merged.setdefault("exc", str(exc))
            self._emit("error", event, merged)
            raise
        merged = dict(fields)
        merged.update(extra)
        merged["duration_ms"] = int((time.monotonic() - started) * 1000)
        merged["outcome"] = "ok"
        self._emit("info", event, merged)


#: The one logger the worker uses.
LOG = Logger()
