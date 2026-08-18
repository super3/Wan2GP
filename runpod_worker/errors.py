"""Stable, machine-readable error taxonomy for the RunPod worker.

Standard library only: no torch, no wgp, no CUDA, no third-party imports. This
module must stay importable on a plain CPU runner — every CPU test in
``runpod_worker/tests`` depends on that.

Clients branch on ``error_code``, never on message text. The codes below are a
published contract: rename one and you break every caller. Add new ones freely;
retire one only with a deprecation window.

See "Failure modes" in ``docs/RUNPOD_SERVERLESS.md`` for the table that maps each
code to its detection point and its recycle policy.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

__all__ = [
    "WorkerError",
    "ALL_CODES",
    "CODE_DESCRIPTIONS",
    "RETRYABLE_CODES",
    "RECYCLE_CODES",
    "is_known_code",
    "default_retryable",
    "default_recycle",
    # codes
    "BAD_REQUEST",
    "UNKNOWN_SETTING",
    "INVALID_SETTING",
    "MEDIA_TOO_LARGE",
    "MEDIA_FETCH_FAILED",
    "MEDIA_UNSUPPORTED",
    "SSRF_BLOCKED",
    "WEIGHTS_MISSING",
    "WANGP_VALIDATION",
    "GENERATION_FAILED",
    "GENERATION_TIMEOUT",
    "GENERATION_CANCELLED",
    "NO_OUTPUT",
    "OUTPUT_TOO_LARGE",
    "UPLOAD_FAILED",
    "WORKER_BUSY",
    "BACKEND_FATAL",
    "OOM",
    "INTERNAL_ERROR",
]

# ---------------------------------------------------------------------------
# The codes. The constant name is for us; the *string value* is the contract.
#
# Three of the names deliberately do not match their value, because the wire
# value was fixed by the response-schema section of the spec before the constant
# was named:
#   GENERATION_TIMEOUT   -> "timeout"
#   GENERATION_CANCELLED -> "cancelled"
#   WANGP_VALIDATION     -> "wangp_validation"  (name matches, listed for symmetry)
# ---------------------------------------------------------------------------

#: Payload is not a JSON object, a forbidden key was supplied, a required field
#: is missing, or the endpoint is pinned to a different ``model_type``.
BAD_REQUEST = "bad_request"

#: ``input.settings`` carried a key that is not in the model's settings universe
#: (``models/_settings.json`` ∪ ``get_default_settings(model_type)``).
UNKNOWN_SETTING = "unknown_setting"

#: A known settings key carried an out-of-range or cross-field-invalid value,
#: rejected by our pre-flight before the model loads.
INVALID_SETTING = "invalid_setting"

#: A single input attachment, or the sum of all of them, exceeded the byte cap.
MEDIA_TOO_LARGE = "media_too_large"

#: An input attachment could not be materialized (bad base64, missing
#: ``volume://`` path, URL fetch failure).
MEDIA_FETCH_FAILED = "media_fetch_failed"

#: An input attachment's *sniffed* content type is not one WanGP accepts for
#: that slot (``shared/utils/utils.py:36-49`` extension whitelists).
MEDIA_UNSUPPORTED = "media_unsupported"

#: A URL input resolved to a private/link-local/loopback address, or a redirect
#: hop did. Only reachable when ``ALLOW_URL_INPUTS=1``.
SSRF_BLOCKED = "ssrf_blocked"

#: The weight set for this ``model_type`` is incomplete on the volume. Raised by
#: the fitness gate *before* any generation so a bad volume never bills a GPU.
WEIGHTS_MISSING = "weights_missing"

#: WanGP's own ``validate_generative_settings`` rejected the request
#: (``GenerationError.stage == "validation"``).
WANGP_VALIDATION = "wangp_validation"

#: The generation ran and failed for a non-validation reason.
GENERATION_FAILED = "generation_failed"

#: The generation exceeded the request's wall-clock budget and was cancelled.
GENERATION_TIMEOUT = "timeout"

#: The generation was cancelled (client ``/cancel``, or worker shutdown) rather
#: than timing out.
GENERATION_CANCELLED = "cancelled"

#: WanGP reported ``success=True`` and produced no video file. This is a silent
#: *configuration* refusal, not a poisoned process — see failure mode 10.
NO_OUTPUT = "no_output"

#: The produced file exceeds every configured transport (no presigned PUT, no
#: bucket, over the base64 cap). Never truncate; always say which env var fixes it.
OUTPUT_TOO_LARGE = "output_too_large"

#: An upload was attempted and did not yield a real remote URL. Includes the
#: ``rp_upload`` local-path fallback, which returns a path instead of raising.
UPLOAD_FAILED = "upload_failed"

#: A second generation was submitted while one is already in flight. Should be
#: unreachable at concurrency 1.
WORKER_BUSY = "worker_busy"

#: The process is permanently unusable: a cancel did not land within the grace
#: window, so a daemon thread still holds WanGP's process-wide
#: ``_GENERATION_LOCK`` (``shared/api.py:27``). Always recycles.
BACKEND_FATAL = "backend_fatal"

#: CUDA OOM (or an equivalent poisoned-device condition). Always recycles.
OOM = "oom"

#: Anything unhandled. If a client sees this, we have a bug.
INTERNAL_ERROR = "internal_error"


ALL_CODES: tuple[str, ...] = (
    BAD_REQUEST,
    UNKNOWN_SETTING,
    INVALID_SETTING,
    MEDIA_TOO_LARGE,
    MEDIA_FETCH_FAILED,
    MEDIA_UNSUPPORTED,
    SSRF_BLOCKED,
    WEIGHTS_MISSING,
    WANGP_VALIDATION,
    GENERATION_FAILED,
    GENERATION_TIMEOUT,
    GENERATION_CANCELLED,
    NO_OUTPUT,
    OUTPUT_TOO_LARGE,
    UPLOAD_FAILED,
    WORKER_BUSY,
    BACKEND_FATAL,
    OOM,
    INTERNAL_ERROR,
)

CODE_DESCRIPTIONS: dict[str, str] = {
    BAD_REQUEST: "malformed or forbidden request payload",
    UNKNOWN_SETTING: "settings key is not part of this model's settings universe",
    INVALID_SETTING: "settings value is out of range or violates a cross-field rule",
    MEDIA_TOO_LARGE: "an input attachment exceeded the byte cap",
    MEDIA_FETCH_FAILED: "an input attachment could not be materialized",
    MEDIA_UNSUPPORTED: "input attachment content type is not accepted for that slot",
    SSRF_BLOCKED: "input URL resolved to a blocked network destination",
    WEIGHTS_MISSING: "model weights are incomplete on this worker",
    WANGP_VALIDATION: "WanGP rejected the settings during task validation",
    GENERATION_FAILED: "the generation ran and failed",
    GENERATION_TIMEOUT: "the generation exceeded the wall-clock budget",
    GENERATION_CANCELLED: "the generation was cancelled",
    NO_OUTPUT: "WanGP reported success but produced no output file",
    OUTPUT_TOO_LARGE: "no configured transport can carry an output this large",
    UPLOAD_FAILED: "the output upload did not produce a remote URL",
    WORKER_BUSY: "a generation is already in flight on this worker",
    BACKEND_FATAL: "the WanGP backend is unusable; the worker must be recycled",
    OOM: "the device ran out of memory",
    INTERNAL_ERROR: "unhandled worker error",
}

#: Codes a client may usefully retry, unchanged, against the same endpoint.
#: Everything else is the caller's fault and will fail identically forever.
RETRYABLE_CODES: frozenset[str] = frozenset(
    {
        MEDIA_FETCH_FAILED,  # transient network / volume hiccup
        GENERATION_TIMEOUT,
        GENERATION_CANCELLED,
        UPLOAD_FAILED,
        WORKER_BUSY,
        BACKEND_FATAL,  # retryable *by the client*: a fresh worker will serve it
        OOM,
        INTERNAL_ERROR,
    }
)

#: Codes that mean this *process* is poisoned and must not serve another job.
#: The handler turns this into ``refresh_worker: True``, which the RunPod SDK
#: pops off the return value and converts to ``stopPod: True``
#: (``runpod/serverless/modules/rp_job.py:266-281``).
RECYCLE_CODES: frozenset[str] = frozenset({BACKEND_FATAL, OOM})


def is_known_code(code: str) -> bool:
    """True when ``code`` is part of the published taxonomy."""
    return code in CODE_DESCRIPTIONS


def default_retryable(code: str) -> bool:
    """Whether ``code`` is retryable when the raiser did not say."""
    return code in RETRYABLE_CODES


def default_recycle(code: str) -> bool:
    """Whether ``code`` poisons the process when the raiser did not say."""
    return code in RECYCLE_CODES


def _normalize_details(details: Any) -> list[str]:
    """Coerce whatever the caller passed into a list of short strings."""
    if details is None:
        return []
    if isinstance(details, Mapping):
        return [f"{key}={value}" for key, value in details.items()]
    if isinstance(details, (str, bytes)):
        text = details.decode("utf-8", "replace") if isinstance(details, bytes) else details
        return [text]
    if isinstance(details, Iterable):
        return [str(item) for item in details]
    return [str(details)]


class WorkerError(Exception):
    """A failure with a stable ``code``, a human ``message`` and optional detail.

    ``code`` is one of the module constants above and is what clients branch on.
    ``message`` is for humans and is free to change between releases.

    ``details`` accepts a list of strings (the common case: validation messages,
    a log tail) *or* a mapping (machine-readable context). Either way both views
    are available afterwards: ``.details`` is always a ``list[str]`` and
    ``.detail`` is always a ``dict`` (empty unless a mapping was supplied).

    ``retryable`` and ``recycle`` default to the taxonomy's policy for ``code``;
    pass them explicitly to override. ``recycle=True`` means *this process* is
    poisoned, which is a strictly stronger statement than ``retryable=True``.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Any = None,
        detail: Mapping[str, Any] | None = None,
        retryable: bool | None = None,
        recycle: bool | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.details: list[str] = _normalize_details(details)
        self.detail: dict[str, Any] = {}
        if isinstance(details, Mapping):
            self.detail.update(details)
        if detail:
            self.detail.update(detail)
            if not self.details:
                self.details = _normalize_details(detail)
        self.retryable = default_retryable(self.code) if retryable is None else bool(retryable)
        self.recycle = default_recycle(self.code) if recycle is None else bool(recycle)
        if cause is not None:
            self.__cause__ = cause

    # -- serialization ------------------------------------------------------

    def to_dict(self, *, include_refresh_worker: bool = True) -> dict[str, Any]:
        """The response envelope fragment for this error.

        Matches the shape ``handler._fail`` builds, so it can be merged straight
        into a handler return value. ``refresh_worker`` is included only when
        ``recycle`` is set, because the RunPod SDK pops that key and turns any
        truthy value into ``stopPod: True``.
        """
        body: dict[str, Any] = {
            "error": self.message,
            "error_code": self.code,
            "retryable": self.retryable,
            "details": list(self.details),
        }
        if self.detail:
            body["detail"] = dict(self.detail)
        if include_refresh_worker and self.recycle:
            body["refresh_worker"] = True
        return body

    # -- construction helpers ----------------------------------------------

    @classmethod
    def wrap(
        cls,
        exc: BaseException,
        *,
        code: str = INTERNAL_ERROR,
        message: str | None = None,
        **kwargs: Any,
    ) -> "WorkerError":
        """Turn an arbitrary exception into a typed one, preserving the cause."""
        if isinstance(exc, cls) and message is None and code == INTERNAL_ERROR:
            return exc
        text = message if message is not None else f"{type(exc).__name__}: {exc}"
        kwargs.setdefault("cause", exc)
        return cls(code, text, **kwargs)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:
        return (
            f"WorkerError(code={self.code!r}, message={self.message!r}, "
            f"retryable={self.retryable!r}, recycle={self.recycle!r})"
        )
