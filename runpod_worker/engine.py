"""The only module in this package that touches WanGP.

Everything here is deliberately import-light at module scope: ``torch``, ``wgp``
and ``shared.api`` are imported *inside functions* so that ``config``, ``errors``,
``obs``, ``schema``, ``media_in`` and ``media_out`` stay CPU-importable and the
whole request-shaping half of the worker can be unit-tested on a laptop.

Three invariants this module exists to enforce:

1. **One WanGP runtime per process.** ``shared/api.py:1061-1097`` (``_ensure_runtime``)
   is the only code that correctly swaps ``sys.argv``, ``chdir``s to the repo
   root for the import, enforces the module-identity check and calls
   ``download_ffmpeg()``. We never ``import wgp`` ourselves. A second ``init()``
   with different root/config/cli args raises (``shared/api.py:1064-1066``), so
   the session is a singleton built once, under ``_BOOT_LOCK``.

2. **One generation at a time, ever.** See ``_JOB_LOCK`` below.

3. **The output is the muxed file on disk**, never the ``_api``/``return_media``
   tensor path: ``shared/api.py:161-178`` hands back an *unmuxed*
   ``torch.uint8 [C,F,H,W]`` tensor that we would have to re-encode ourselves,
   while WanGP writes the muxed container unconditionally anyway. ``run()``
   therefore returns ``result.generated_files`` paths and nothing else.
"""

from __future__ import annotations

import collections
import gc
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from . import config as C
from .errors import (
    BACKEND_FATAL,
    BAD_REQUEST,
    GENERATION_FAILED,
    GENERATION_TIMEOUT,
    NO_OUTPUT,
    OOM,
    WEIGHTS_MISSING,
    WORKER_BUSY,
    WorkerError,
)
from .obs import LOG

__all__ = [
    "RunOutcome",
    "STATS",
    "boot",
    "is_booted",
    "session",
    "warm",
    "maybe_warm",
    "is_warm",
    "assert_weights_complete",
    "weights_report",
    "expected_core_files",
    "run",
    "timeout_error",
    "classify_failure",
    "is_poison",
    "poison_markers",
    "note_failure",
    "note_success",
    "mark_poisoned",
    "should_recycle",
    "recycle_reason",
    "reset_failure_budget",
    "gpu_snapshot",
    "stats",
    "release_model",
    "shutdown",
]


# ---------------------------------------------------------------------------
# Process-wide state
# ---------------------------------------------------------------------------

_SESSION: Any = None
_BOOT_LOCK = threading.Lock()

# THE GENERATION LOCK.
#
# A plain Lock, not an RLock, and held with blocking=False: a second generation
# must fail fast with `worker_busy`, never queue and never re-enter.
#
# Three independent reasons one process may only ever run one generation:
#
#   * ``shared/api_cli.py:29`` takes WanGP's module-level ``_GENERATION_LOCK``
#     (``shared/api.py:27``) and holds it for the entire job. A second job would
#     not fail -- it would block a thread inside WanGP for the whole first
#     generation, invisible to us and to RunPod.
#   * ``shared/api_cli.py:48`` installs ``contextlib.redirect_stdout`` /
#     ``redirect_stderr``, which are *process-global*. Two overlapping jobs would
#     interleave their stdout into each other's event queues, and whichever
#     finished first would restore the wrong stream on the way out.
#   * ``WanGPSession._submit_tasks`` (``shared/api.py:648``) raises a bare
#     ``RuntimeError`` when the previous job is not done. Our lock turns that
#     race into a typed, retryable ``worker_busy`` before anything is submitted.
#
# At RunPod concurrency 1 (no ``concurrency_modifier``) this should be
# unreachable; it is here because "should be unreachable" is not "is".
_JOB_LOCK = threading.Lock()

_WARM_LOCK = threading.Lock()
_WARMED_TYPE: str | None = None
_WARM_ATTEMPTED = False

_FAILURE_LOCK = threading.Lock()
_CONSECUTIVE_FAILURES = 0
_RECYCLE_REASON: str | None = None

#: Counters the handler folds into its ``metrics`` block.
STATS: dict[str, Any] = {
    "jobs_served": 0,
    "boot_ms": 0,
    "warm_ms": 0,
    "consecutive_failures": 0,
    "vram_floor_mb": 0.0,
    "vram_peak_mb": 0.0,
}

#: Set once the first job's post-cleanup VRAM floor is known (leak baseline).
_VRAM_BASELINE_MB: float | None = None


# ---------------------------------------------------------------------------
# Poison detection
# ---------------------------------------------------------------------------

#: Substrings that mean the *allocator* is the problem. Always recycle: mmgp
#: leaves a fragmented, partially-offloaded model behind and the next job on
#: this worker will fail the same way, only slower.
OOM_MARKERS: tuple[str, ...] = (
    "out of memory",
    "outofmemoryerror",
    "cuda_error_out_of_memory",
    "cublas_status_alloc_failed",
    "cannot allocate memory",
    "hip out of memory",
)

#: Substrings that mean the CUDA context itself is unusable. Recycle, but they
#: are not OOM and should not be reported as such.
POISON_MARKERS_FALLBACK: tuple[str, ...] = (
    "cuda error",
    "cublas_status",
    "cudnn_status_",
    "device-side assert",
    "illegal memory access",
    "an illegal instruction was encountered",
    "misaligned address",
    "nccl",
    "unspecified launch failure",
    "no kernel image is available",
)


def poison_markers() -> tuple[str, ...]:
    """The poison-substring set, preferring ``schema.POISON_MARKERS``.

    schema owns the published list (the handler scans WanGP's error strings with
    it too); this module keeps a superset fallback so ``engine`` still classifies
    correctly if schema is ever trimmed.
    """
    markers: list[str] = list(POISON_MARKERS_FALLBACK)
    try:
        from .schema import POISON_MARKERS as _SCHEMA_MARKERS
    except Exception:  # noqa: BLE001 - classification must never raise
        _SCHEMA_MARKERS = ()
    for marker in _SCHEMA_MARKERS:
        text = str(marker).lower()
        if text and text not in markers:
            markers.append(text)
    return tuple(markers)


def classify_failure(text: Any) -> tuple[str, bool]:
    """``(error_code, recycle)`` for an error message / log tail.

    ``recycle=True`` means *this process* is poisoned, which the handler turns
    into ``refresh_worker: True`` (the RunPod SDK pops that key off the return
    value and converts it to ``stopPod: True``).
    """
    lowered = str(text or "").lower()
    if not lowered:
        return GENERATION_FAILED, False
    for marker in OOM_MARKERS:
        if marker in lowered:
            return OOM, True
    for marker in poison_markers():
        if marker in lowered:
            return GENERATION_FAILED, True
    return GENERATION_FAILED, False


def is_poison(text: Any) -> bool:
    """Whether ``text`` indicates a poisoned process rather than a bad request."""
    return classify_failure(text)[1]


# ---------------------------------------------------------------------------
# Small env helpers (re-read every call so tests can monkeypatch os.environ)
# ---------------------------------------------------------------------------


def _flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in ("1", "true", "yes", "on")


def _int_env(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, "")).strip() or default)
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, "")).strip() or default)
    except (TypeError, ValueError):
        return default


def _tail_size() -> int:
    return max(20, _int_env("WANGP_LOG_TAIL", 400))


# ---------------------------------------------------------------------------
# The result of one generation
# ---------------------------------------------------------------------------


@dataclass
class RunOutcome:
    """Everything ``run()`` learned about one generation.

    Iterable for backwards compatibility with the plan's 5-tuple contract::

        result, timed_out, logs, phase_marks, gen_s = engine.run(...)

    Prefer the named fields; the tuple order is frozen only so an older handler
    keeps working.
    """

    result: Any
    timed_out: bool = False
    logs: list[str] = field(default_factory=list)
    phase_marks: dict[str, float] = field(default_factory=dict)
    gen_s: float = 0.0
    files: list[str] = field(default_factory=list)
    video_path: str | None = None
    errors: list[str] = field(default_factory=list)
    stages: tuple[str, ...] = ()
    poisoned: bool = False
    cancelled: bool = False
    progress_events: int = 0
    stream_events: int = 0
    last_progress: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return bool(getattr(self.result, "success", False))

    def __iter__(self) -> Iterator[Any]:
        yield self.result
        yield self.timed_out
        yield self.logs
        yield self.phase_marks
        yield self.gen_s

    def __len__(self) -> int:
        return 5

    def summary(self) -> dict[str, Any]:
        """A log-safe digest (no tensors, no PIL images, no full log tail)."""
        return {
            "success": self.success,
            "cancelled": self.cancelled,
            "timed_out": self.timed_out,
            "gen_s": self.gen_s,
            "files": len(self.files),
            "video_path": self.video_path,
            "errors": self.errors[:5],
            "poisoned": self.poisoned,
            "phase_marks_s": dict(self.phase_marks),
            "progress_events": self.progress_events,
            "stream_events": self.stream_events,
        }


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------


def is_booted() -> bool:
    """True once ``import wgp`` has happened in this process."""
    return _SESSION is not None


def session() -> Any:
    """The live :class:`shared.api.WanGPSession`, booting it if needed."""
    return boot()


def _boot_locked() -> Any:
    """Create the session singleton. Never warms, never loads weights."""
    global _SESSION
    with _BOOT_LOCK:
        if _SESSION is not None:
            return _SESSION

        # cli_args are passed so ensure_wgp_config can reconcile --attention with
        # WANGP_ATTENTION: wgp.py:3304-3305 lets the CLI flag overwrite
        # server_config["attention_mode"], so the file must agree with the flag
        # or the config is a lie.
        cfg_path = C.ensure_wgp_config(C.CONFIG.cli_args)
        C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        root = str(C.WANGP_ROOT)
        if root not in sys.path:
            # shared/api.py:1078 does this too; doing it first means an import
            # failure names our root rather than a stale sys.path[0].
            sys.path.insert(0, root)

        t0 = time.monotonic()
        try:
            from shared.api import init  # shared/api.py:1265-1287 (keyword-only)

            built = init(
                root=C.WANGP_ROOT,
                config_path=cfg_path,          # MUST be named wgp_config.json
                output_dir=C.OUTPUT_DIR,       # shared/api.py:816-831 rewrites save paths
                cli_args=C.CONFIG.cli_args,    # frozen for the process lifetime
                console_output=C.CONFIG.console_output,
                console_isatty=False,          # nothing here is a terminal
            )
        except KeyError as exc:
            # THE CONFIG FILE TRAP. wgp.py:3301 does a bare
            # server_config["attention_mode"] at module scope, and a config file
            # that exists REPLACES wgp's defaults wholesale (wgp.py:2623).
            raise WorkerError(
                BACKEND_FATAL,
                f"import wgp raised KeyError({exc}); if that is a server_config key, "
                f"the rendered {cfg_path} does not carry every key wgp reads unguarded "
                f"at module scope",
                details=[f"required keys: {list(C.REQUIRED_WGP_KEYS)}"],
                cause=exc,
            ) from exc
        except (KeyboardInterrupt, SystemExit):
            # wgp.py:4098 calls exit() on the --check-loras path; never swallow it.
            raise
        except WorkerError:
            raise
        except BaseException as exc:  # noqa: BLE001 - boot failure must be typed
            code, recycle = classify_failure(f"{type(exc).__name__}: {exc}")
            raise WorkerError(
                BACKEND_FATAL if not recycle else code,
                f"WanGP failed to initialize: {type(exc).__name__}: {exc}",
                details=[f"root={C.WANGP_ROOT}", f"config={cfg_path}",
                         f"cli_args={list(C.CONFIG.cli_args)}"],
                recycle=True,
                cause=exc,
            ) from exc

        STATS["boot_ms"] = int((time.monotonic() - t0) * 1000)
        # Assert BEFORE publishing the singleton: a stale ATTACHMENT_KEYS list is
        # a security hole, and leaving _SESSION unset means a retry re-checks
        # instead of silently returning the session the check just rejected.
        _assert_attachment_keys(built)
        _SESSION = built
        LOG.info(
            "wgp_imported",
            boot_ms=STATS["boot_ms"],
            wangp_version=_module_attr(_SESSION, "WanGP_version", "?"),
            model_type=C.CONFIG.model_type,
            config_path=str(cfg_path),
            ckpts=C.checkpoint_paths(),
            loras=C.lora_root(),
            output_dir=str(C.OUTPUT_DIR),
            cli_args=list(C.CONFIG.cli_args),
        )
        return _SESSION


def boot() -> Any:
    """Import wgp and return the session. Does NOT load weights.

    ``preload_model_policy`` is written empty by ``config.authoritative_keys()``,
    so ``wgp.py:4085`` (``if not "P" in preload_model_policy``) leaves
    ``wan_model = None`` and ``reload_needed = True``. Weight loading is minutes
    long and must not push worker start past RunPod's unhealthy threshold --
    unless ``WANGP_WARM=1``, which moves it here on purpose (see :func:`warm`).
    """
    built = _boot_locked()
    _maybe_warm_after_boot()
    return built


def _module_attr(sess: Any, name: str, default: Any = None) -> Any:
    """Read an attribute off the imported ``wgp`` module, defensively."""
    try:
        return getattr(sess._ensure_runtime().module, name, default)
    except Exception:  # noqa: BLE001 - diagnostics must never raise
        return default


def _runtime(sess: Any = None) -> Any:
    return (sess or _boot_locked())._ensure_runtime()


def _assert_attachment_keys(sess: Any) -> None:
    """Fail the worker if upstream grew an attachment key schema does not know.

    ``ATTACHMENT_KEYS`` (``wgp.py:167-168``) is the list
    ``WanGPSession._absolutize_task_paths`` (``shared/api.py:1010-1019``) walks to
    turn settings values into absolute paths. schema.FORBIDDEN_KEYS is derived
    from the same list: a key we do not know about is a key a caller could use to
    name an arbitrary local file, so this is a security check, not hygiene.
    """
    try:
        from .schema import ATTACHMENT_KEYS
    except Exception as exc:  # noqa: BLE001
        LOG.warn("attachment_keys_check_skipped", error=str(exc))
        return
    live = tuple(str(key) for key in (_module_attr(sess, "ATTACHMENT_KEYS", ()) or ()))
    if not live:
        LOG.warn("attachment_keys_missing_upstream", note="wgp.ATTACHMENT_KEYS is empty")
        return
    added = sorted(set(live) - set(ATTACHMENT_KEYS))
    removed = sorted(set(ATTACHMENT_KEYS) - set(live))
    if removed:
        # Harmless (we simply forbid a key that no longer exists) but worth saying.
        LOG.info("attachment_keys_removed_upstream", keys=removed)
    if not added:
        return
    message = (
        f"schema.ATTACHMENT_KEYS is stale; upstream wgp.ATTACHMENT_KEYS added {added}. "
        f"Those keys are NOT rejected in input.settings, which is an arbitrary "
        f"local-file read (failure mode 5)."
    )
    if _flag("WANGP_STRICT_ATTACHMENT_KEYS", "1"):
        raise RuntimeError(message)
    LOG.error("attachment_keys_stale", keys=added, strict=False)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
#
# Two different questions, two different answers, and the plan conflates them:
#
#   get_missing_core_file_entries_for_status (shared/model_dropdowns.py:342)
#       enumerates the CORE set -- transformer (+ "URLs2" submodel), any declared
#       modules, and the text encoder at deps.text_encoder_quantization. This is
#       the set whose absence means "this worker cannot generate anything".
#
#   get_model_download_status (shared/model_dropdowns.py:442)
#       returns EXPECTED only when the core set is present AND
#       has_secondary_model_files_for_status (:391) finds every preload_URL,
#       VAE_URL and model-declared LoRA. So `available` is STRICTLY STRONGER
#       than "core complete" -- the plan's claim that it "can report available
#       while the text encoder is absent" is not what the source does (the text
#       encoder is checked at :396-401 and downgrades the status to PARTIAL).
#
# We gate hard on the core enumeration (it names the missing files, which is what
# an operator needs) and treat a non-EXPECTED status as a warning: a missing VAE
# is a few hundred MB fetched on the first request, not a dead worker. Set
# WANGP_REQUIRE_FULL_WEIGHTS=1 to make that fatal too.


def _entry_name(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("filename") or entry.get("path") or entry)
    return str(entry)


def expected_core_files(model_type: str | None = None) -> list[str]:
    """Every core weight file this worker expects, resolved for the live config.

    Useful for ``scripts/verify_weights.py``: it prints the exact transformer and
    text-encoder filenames the current ``transformer_quantization`` /
    ``text_encoder_quantization`` resolve to.
    """
    model_type = str(model_type or C.CONFIG.model_type)
    sess = _boot_locked()
    runtime = _runtime(sess)
    from shared.api import _pushd  # shared/api.py:1302-1309

    with _pushd(runtime.root):
        deps = runtime.module._get_dropdown_deps()  # wgp.py:13229
        entries = runtime.module.model_dropdowns.get_expected_core_file_entries_for_status(
            deps, model_type
        )
    return [_entry_name(entry) for entry in entries or []]


def weights_report(model_type: str | None = None) -> dict[str, Any]:
    """Non-raising weight inventory for ``model_type``.

    Returns ``{model_type, expected_core, missing_core, status, status_code,
    available, transformer_quantization, text_encoder_quantization,
    checkpoints_paths, loras_root}``.
    """
    model_type = str(model_type or C.CONFIG.model_type)
    sess = _boot_locked()

    if sess.get_model_def(model_type) is None:
        raise WorkerError(
            BAD_REQUEST,
            f"unknown model_type '{model_type}'; this worker is pinned to "
            f"WANGP_MODEL_TYPE and that value is not a model WanGP defines",
        )

    runtime = _runtime(sess)
    from shared.api import _pushd

    # _pushd is process-wide chdir. It is only ever called with runtime.root,
    # which is also the CWD a generation runs under (shared/api_cli.py:29), so a
    # concurrent generation cannot observe a different directory.
    with _pushd(runtime.root):
        deps = runtime.module._get_dropdown_deps()
        module_dropdowns = runtime.module.model_dropdowns
        expected = module_dropdowns.get_expected_core_file_entries_for_status(deps, model_type)
        missing = module_dropdowns.get_missing_core_file_entries_for_status(deps, model_type)

    availability = sess.get_model_availability(model_type)
    return {
        "model_type": model_type,
        "expected_core": [_entry_name(entry) for entry in expected or []],
        "missing_core": [_entry_name(entry) for entry in missing or []],
        "status": availability.get("status"),
        "status_code": availability.get("status_code"),
        "available": bool(availability.get("available")),
        "transformer_quantization": _module_attr(sess, "transformer_quantization"),
        "text_encoder_quantization": _module_attr(sess, "text_encoder_quantization"),
        "checkpoints_paths": C.checkpoint_paths(),
        "loras_root": C.lora_root(),
    }


def assert_weights_complete(model_type: str | None = None) -> dict[str, Any]:
    """Raise ``weights_missing`` unless every core weight file is on disk.

    Run this BEFORE any generation -- as a RunPod fitness check, so a worker with
    a bad volume never serves a request and never silently downloads 27 GB on
    the clock (failure mode 3). Returns the :func:`weights_report` dict on
    success so callers can log what they verified.
    """
    report = weights_report(model_type)
    missing = report["missing_core"]
    if missing:
        raise WorkerError(
            WEIGHTS_MISSING,
            f"weights incomplete for {report['model_type']}: "
            f"{len(missing)} core file(s) missing",
            details=missing,
            detail={
                "model_type": report["model_type"],
                "status": report["status"],
                "transformer_quantization": report["transformer_quantization"],
                "text_encoder_quantization": report["text_encoder_quantization"],
                "checkpoints_paths": report["checkpoints_paths"],
                "hint": (
                    "run scripts/prefetch_weights.py against the volume with the SAME "
                    "WANGP_TRANSFORMER_QUANT / WANGP_TEXT_ENCODER_QUANT this worker runs"
                ),
            },
        )

    if not report["available"]:
        # Core is complete but a secondary file (VAE / preload / model LoRA) is
        # absent: WanGP will fetch it on the first request.
        message = (
            f"{report['model_type']}: core weights present but download status is "
            f"'{report['status']}'; a secondary file (VAE / preload / model LoRA) will "
            f"be fetched on the first request"
        )
        if _flag("WANGP_REQUIRE_FULL_WEIGHTS", "0"):
            raise WorkerError(WEIGHTS_MISSING, message, detail=report)
        LOG.warn("weights_partial", **{k: report[k] for k in ("model_type", "status")})
    else:
        LOG.info(
            "weights_complete",
            model_type=report["model_type"],
            files=len(report["expected_core"]),
            status=report["status"],
        )
    return report


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------


def _normalize_config_selection(selection: Any) -> str:
    """Mirror ``serialize_config_selection(split_config_selection(x))``.

    ``shared/config_groups.py:14-20``: four comma-joined ids, trailing commas
    stripped. The rstripped form is the one that must be stored, because
    ``load_models`` records ``config_id or ""`` verbatim (``wgp.py:4082``) and the
    reload gate at ``wgp.py:6773`` compares those strings.
    """
    values = str(selection or "").split(",")
    values = (values + ["", "", "", ""])[:4]
    return ",".join(str(value or "") for value in values).rstrip(",")


def is_warm(model_type: str | None = None) -> bool:
    """Whether the requested model is already resident in this process."""
    if _SESSION is None:
        return False
    model_type = str(model_type or C.CONFIG.model_type)
    return (
        _module_attr(_SESSION, "wan_model") is not None
        and str(_module_attr(_SESSION, "transformer_type", "")) == model_type
        and not bool(_module_attr(_SESSION, "reload_needed", True))
    )


def warm(model_type: str | None = None, *, config_id: str | None = None) -> dict[str, Any]:
    """Load the model's weights now instead of on the first request.

    Opt-in via ``WANGP_WARM=1``. The trade-off is explicit: warming moves the
    150-250 s volume read out of the first request's ``executionTime`` and into
    worker start, which is billed either way but must not push start past
    RunPod's unhealthy threshold. Leave it off for endpoints that scale from
    zero often; turn it on for endpoints with active workers.

    This calls ``wgp.load_models`` directly and then reproduces the two
    assignments ``generate_media`` would have made, because ``load_models``
    itself only owns ``transformer_type`` / ``loaded_profile`` / ``loaded_config``
    (``wgp.py:3952``, ``:4081-4082``) -- ``wan_model``/``offloadobj`` are locals it
    *returns* (``wgp.py:4055``, ``:4083``) and ``reload_needed`` stays True from
    ``wgp.py:4087``. Forget either and the first job hits the reload gate at
    ``wgp.py:6773``, calls ``release_model()`` and reloads everything, so warming
    would cost time instead of saving it.

    The gate also compares ``profile`` and ``config``, so both are computed the
    same way ``generate_media`` computes them: ``output_type`` from
    ``get_profile_type_for_model(model_type, image_mode=0)`` and
    ``override_profile`` from the model's own default settings.
    """
    global _WARMED_TYPE
    sess = _boot_locked()
    model_type = str(model_type or C.CONFIG.model_type)

    with _WARM_LOCK:
        if is_warm(model_type):
            return {"warmed": False, "reason": "already_loaded", "model_type": model_type}

        # Never download 48 GB on a boot that was only supposed to be warm.
        assert_weights_complete(model_type)

        # get_default_settings json.dump()s settings/<model_type>_settings.json on
        # first call (wgp.py:3175). Doing it here means a read-only or slow
        # filesystem is a boot-time failure, not a request-time one.
        defaults = sess.get_default_settings(model_type)
        try:
            override_profile = int(defaults.get("override_profile", -1))
        except (TypeError, ValueError):
            override_profile = -1
        selection = config_id if config_id is not None else (
            C.CONFIG.model_config or defaults.get("config") or ""
        )
        selection = _normalize_config_selection(selection)

        runtime = _runtime(sess)
        module = runtime.module
        from shared.api import _GENERATION_LOCK, _pushd

        t0 = time.monotonic()
        with _JOB_LOCK:  # never swap weights under a running generation
            with _GENERATION_LOCK, _pushd(runtime.root):
                if module.wan_model is not None:
                    module.release_model()  # wgp.py:233-245
                output_type = module.get_profile_type_for_model(model_type, 0)  # wgp.py:3827
                module.wan_model, module.offloadobj = module.load_models(
                    model_type,
                    override_profile,
                    output_type=output_type,
                    config_id=selection,
                )
                # The two globals load_models does not own. Without this the
                # first request reloads from scratch (wgp.py:6773).
                module.reload_needed = False
        warm_ms = int((time.monotonic() - t0) * 1000)

        STATS["warm_ms"] = warm_ms
        _WARMED_TYPE = model_type
        LOG.info(
            "model_warmed",
            model_type=model_type,
            warm_ms=warm_ms,
            profile=_module_attr(sess, "loaded_profile"),
            config=_module_attr(sess, "loaded_config"),
            **gpu_snapshot(),
        )
        return {
            "warmed": True,
            "model_type": model_type,
            "warm_ms": warm_ms,
            "config": selection,
            "override_profile": override_profile,
        }


def maybe_warm() -> dict[str, Any]:
    """Warm iff ``WANGP_WARM=1``. Never raises unless ``WANGP_WARM_STRICT=1``."""
    global _WARM_ATTEMPTED
    if not _flag("WANGP_WARM", "0"):
        return {"warmed": False, "reason": "disabled"}
    if _WARM_ATTEMPTED:
        return {"warmed": False, "reason": "already_attempted"}
    _WARM_ATTEMPTED = True
    try:
        return warm()
    except BaseException as exc:  # noqa: BLE001 - a failed warm is not a dead worker
        LOG.error("warm_failed", error=f"{type(exc).__name__}: {exc}")
        if _flag("WANGP_WARM_STRICT", "0"):
            raise
        return {"warmed": False, "reason": "failed", "error": f"{type(exc).__name__}: {exc}"}


def _maybe_warm_after_boot() -> None:
    """Called from :func:`boot` OUTSIDE ``_BOOT_LOCK`` (warm re-enters boot)."""
    if _WARM_ATTEMPTED or not _flag("WANGP_WARM", "0"):
        return
    maybe_warm()


# ---------------------------------------------------------------------------
# Failure budget / recycle policy
# ---------------------------------------------------------------------------


def _failure_budget() -> int:
    return max(1, int(getattr(C.CONFIG, "failure_budget", 3) or 3))


def note_failure(code: str | None = None, *, recycle: bool = False, reason: str = "") -> int:
    """Record a failed job. Returns the consecutive-failure count.

    Only ``recycle=True`` latches ``_RECYCLE_REASON``. The consecutive-failure
    BUDGET is deliberately not latched here: it is a soft counter that a single
    success clears, and burning it into a permanent verdict meant every later
    response — including successful ones — carried ``refresh_worker: True`` and
    paid a 150-250 s weight reload. :func:`should_recycle` computes the budget
    half live instead.
    """
    global _CONSECUTIVE_FAILURES, _RECYCLE_REASON
    with _FAILURE_LOCK:
        _CONSECUTIVE_FAILURES += 1
        STATS["consecutive_failures"] = _CONSECUTIVE_FAILURES
        if recycle and _RECYCLE_REASON is None:
            _RECYCLE_REASON = reason or f"poisoned by {code or 'unknown failure'}"
        count = _CONSECUTIVE_FAILURES
    LOG.warn("failure_recorded", error_code=code, consecutive=count, recycle=should_recycle())
    return count


def note_success() -> None:
    """Record a job that produced output. Clears the consecutive-failure run.

    Does NOT clear a poison flag: a process that has seen a CUDA fault does not
    become healthy because the next job happened to survive. It DOES clear the
    soft budget, which is the whole point of a *consecutive*-failure counter.
    """
    global _CONSECUTIVE_FAILURES
    with _FAILURE_LOCK:
        _CONSECUTIVE_FAILURES = 0
        STATS["consecutive_failures"] = 0


def mark_poisoned(reason: str) -> None:
    """Latch "this worker must not serve another job"."""
    global _RECYCLE_REASON
    with _FAILURE_LOCK:
        if _RECYCLE_REASON is None:
            _RECYCLE_REASON = str(reason)
    LOG.error("worker_poisoned", reason=str(reason))


def should_recycle() -> bool:
    """Whether the handler should return ``refresh_worker: True``.

    Two sources: the latched poison flag (permanent, by design) and the
    consecutive-failure budget (live, so a success clears it).
    """
    with _FAILURE_LOCK:
        if _RECYCLE_REASON is not None:
            return True
        return _CONSECUTIVE_FAILURES >= _failure_budget()


def recycle_reason() -> str | None:
    with _FAILURE_LOCK:
        if _RECYCLE_REASON is not None:
            return _RECYCLE_REASON
        budget = _failure_budget()
        if _CONSECUTIVE_FAILURES >= budget:
            return (
                f"{_CONSECUTIVE_FAILURES} consecutive failures "
                f"(WORKER_FAILURE_BUDGET={budget})"
            )
        return None


def reset_failure_budget() -> None:
    """Clear both counters. For tests and for scripted single-shot runs."""
    global _CONSECUTIVE_FAILURES, _RECYCLE_REASON
    with _FAILURE_LOCK:
        _CONSECUTIVE_FAILURES = 0
        _RECYCLE_REASON = None
        STATS["consecutive_failures"] = 0


def timeout_error(budget_s: float, logs: Sequence[str] = ()) -> WorkerError:
    """The canonical budget-exceeded error, for handlers that prefer to raise."""
    return WorkerError(
        GENERATION_TIMEOUT,
        f"generation exceeded the {int(budget_s)}s budget and was cancelled",
        details=list(logs)[-20:],
        recycle=should_recycle(),
    )


# ---------------------------------------------------------------------------
# The generation itself
# ---------------------------------------------------------------------------


def _emit(callback: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]) -> None:
    """Progress emission must never be able to fail a generation."""
    if callback is None:
        return
    try:
        callback(payload)
    except Exception as exc:  # noqa: BLE001
        LOG.warn("progress_emit_failed", error=f"{type(exc).__name__}: {exc}")


def _video_from(files: Sequence[str]) -> str | None:
    try:
        from .schema import VIDEO_EXTS
    except Exception:  # noqa: BLE001
        VIDEO_EXTS = frozenset({".mp4", ".mkv", ".avi", ".mov"})
    for path in files:
        if Path(str(path)).suffix.lower() in VIDEO_EXTS:
            return str(path)
    return None


def run(
    settings: dict[str, Any],
    *,
    budget_s: float | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    emit_progress: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    sess: Any = None,
) -> RunOutcome:
    """Run exactly one generation and return a :class:`RunOutcome`.

    ``settings`` must already be schema-validated and media-materialized: every
    attachment value is an absolute path on this filesystem.

    ``budget_s`` is wall clock. On overrun the job is cancelled cooperatively
    (``job.cancel()`` -> ``gen["abort"] = True`` + ``wan_model._interrupt = True``,
    ``shared/api.py:895-899``), which lands at the model's next interrupt check --
    one denoising step. If it has not landed within ``WANGP_CANCEL_GRACE_S`` the
    process is declared poisoned: the WanGP worker thread is a daemon that cannot
    be killed and it still holds ``_GENERATION_LOCK``.

    ``on_progress`` (alias ``emit_progress``) receives throttled dicts:
    ``{phase, status, pct, step, total_steps, elapsed_s[, eta_s]}``.
    """
    callback = on_progress if on_progress is not None else emit_progress
    budget = float(budget_s if budget_s is not None else C.CONFIG.default_budget_s)
    live = sess if sess is not None else boot()

    # Fail fast rather than queue: at concurrency 1 a second in-flight job means
    # the platform double-dispatched, and blocking would hide that.
    if not _JOB_LOCK.acquire(blocking=False):
        raise WorkerError(
            WORKER_BUSY,
            "a generation is already in flight on this worker",
            details=["this endpoint runs one generation per process; scale with max_workers"],
        )

    tail: collections.deque[str] = collections.deque(maxlen=_tail_size())
    phase_marks: dict[str, float] = {}
    step_anchors: dict[str, tuple[float, int]] = {}
    errors: list[str] = []
    stages: set[str] = set()
    progress_events = 0
    stream_events = 0
    last_progress: dict[str, Any] = {}
    submitted = False
    finished = False
    t0 = time.monotonic()

    try:
        try:
            job = live.submit_task(settings)  # shared/api.py:562-565, non-blocking
        except RuntimeError as exc:
            # shared/api.py:648 -- the session still has an unfinished job.
            raise WorkerError(
                WORKER_BUSY, f"WanGP refused the submission: {exc}", cause=exc
            ) from exc
        submitted = True

        deadline = t0 + budget
        cancelled_at: float | None = None
        cancel_grace = float(getattr(C.CONFIG, "cancel_grace_s", 150))
        interval = float(getattr(C.CONFIG, "progress_interval_s", 5))
        last_emit = float("-inf")

        # DO NOT use job.events.iter(): SessionStream.iter (shared/api.py:263-271)
        # `continue`s on a queue timeout without yielding, so the loop body -- and
        # therefore any wall-clock check inside it -- never runs during a silent
        # stretch. A silent stretch is exactly when a hung job needs the check.
        while True:
            event = job.events.get(timeout=0.5)  # None on timeout AND on close
            now = time.monotonic()

            if event is not None:
                kind = getattr(event, "kind", "")
                data = getattr(event, "data", None)

                if kind == "stream":
                    # Every stdout/stderr LINE is an event, and _OutputCapture
                    # splits on "\r" as well as "\n" (shared/api.py:316-330), so a
                    # tqdm bar is one event per refresh. Ring-buffer it.
                    stream_events += 1
                    tail.append(f"{getattr(data, 'stream', '?')}: {getattr(data, 'text', data)}"[:512])

                elif kind == "progress":
                    progress_events += 1
                    phase = str(getattr(data, "phase", "") or "unknown")
                    phase_marks.setdefault(phase, round(now - t0, 1))
                    payload: dict[str, Any] = {
                        "phase": phase,
                        "status": str(getattr(data, "status", "") or "")[:300],
                        "pct": getattr(data, "progress", None),
                        "step": getattr(data, "current_step", None),
                        "total_steps": getattr(data, "total_steps", None),
                        "elapsed_s": round(now - t0, 1),
                    }
                    eta = _estimate_eta(step_anchors, phase, payload["step"],
                                        payload["total_steps"], now)
                    if eta is not None:
                        payload["eta_s"] = eta
                    last_progress = payload
                    if now - last_emit >= interval:
                        last_emit = now
                        _emit(callback, payload)

                elif kind in ("status", "info"):
                    text = str(data)
                    tail.append(f"{kind}: {text}"[:512])
                    LOG.info("wangp_" + kind, text=text[:300])

                elif kind == "error":
                    message = str(getattr(data, "message", data))
                    stage = str(getattr(data, "stage", "") or "")
                    if stage:
                        stages.add(stage)
                    errors.append(f"[{stage or 'error'}] {message}" if stage else message)
                    tail.append(f"error: {message}"[:512])
                    LOG.error("wangp_error", stage=stage, message=message[:500])

                elif kind == "started":
                    LOG.info(
                        "wangp_started",
                        tasks=data.get("tasks") if isinstance(data, dict) else None,
                    )

                elif kind == "output":
                    tail.append(f"output: {str(data)[:200]}")

            # Termination. job._set_result (shared/api_cli.py:94) fires before
            # job.events.close() (:112), so require BOTH plus a drained queue --
            # otherwise the last few events, including the error text, are lost.
            if event is None and job.done and job.events.closed:
                break

            if cancelled_at is None and (
                now > deadline or (cancel_check is not None and _safe_cancel_check(cancel_check))
            ):
                cancelled_at = now
                reason = "budget_exceeded" if now > deadline else "cancel_requested"
                LOG.warn(reason + "_cancelling", budget_s=budget,
                         elapsed_s=round(now - t0, 1))
                job.cancel()  # cooperative: shared/api_cli.py:64-65 -> shared/api.py:895-899

            if cancelled_at is not None and now - cancelled_at > cancel_grace:
                break

        gen_s = round(time.monotonic() - t0, 2)

        if not job.done:
            # The daemon worker thread cannot be killed and still holds the
            # process-wide _GENERATION_LOCK. This worker is permanently poisoned.
            mark_poisoned(f"cancel did not land within {cancel_grace}s")
            note_failure(BACKEND_FATAL, recycle=True)
            raise WorkerError(
                BACKEND_FATAL,
                f"generation did not stop within {cancel_grace}s of cancel; "
                f"worker will be recycled",
                details=list(tail)[-20:],
                retryable=True,
                recycle=True,
            )

        try:
            result = job.result(timeout=_float_env("WANGP_RESULT_TIMEOUT_S", 10.0))
        except TimeoutError as exc:
            mark_poisoned("job.done was set but result never materialized")
            note_failure(BACKEND_FATAL, recycle=True)
            raise WorkerError(
                BACKEND_FATAL,
                "WanGP signalled completion without producing a result object",
                details=list(tail)[-20:],
                recycle=True,
                cause=exc,
            ) from exc

        finished = True
        files = [str(path) for path in (getattr(result, "generated_files", None) or [])]
        for err in getattr(result, "errors", None) or []:
            stage = str(getattr(err, "stage", "") or "")
            message = str(getattr(err, "message", err))
            entry = f"[{stage or 'error'}] {message}" if stage else message
            if entry not in errors:
                errors.append(entry)
            if stage:
                stages.add(stage)

        poison_text = " ".join(errors) + " " + " ".join(list(tail)[-60:])
        poisoned = bool(errors) and is_poison(poison_text)

        outcome = RunOutcome(
            result=result,
            timed_out=cancelled_at is not None,
            logs=list(tail),
            phase_marks=phase_marks,
            gen_s=gen_s,
            files=files,
            video_path=_video_from(files),
            errors=errors,
            stages=tuple(sorted(stages)),
            poisoned=poisoned,
            cancelled=bool(getattr(result, "cancelled", False)),
            progress_events=progress_events,
            stream_events=stream_events,
            last_progress=last_progress,
        )

        # Free the queued events and the session's reference to the result. We
        # already hold `result`; SessionJob.release_output_payload (shared/api.py:
        # 375-377) only drops the session-side copy and clears the event queue.
        try:
            job.release_output_payload()
            job.release_input_payload()
        except Exception as exc:  # noqa: BLE001
            LOG.warn("release_payload_failed", error=f"{type(exc).__name__}: {exc}")

        # Keep the engine-side failure budget honest without the handler having
        # to remember. NOTE for handler.py: count generation outcomes HERE or
        # there, not both -- doubling the increment halves WORKER_FAILURE_BUDGET.
        if outcome.success and outcome.files:
            note_success()
        elif "validation" in outcome.stages and not poisoned:
            # WanGP's own validate_settings / validate_generative_settings said
            # no. That is the caller's input, not this worker's health -- and it
            # is exactly the class schema.parse cannot pre-empt (reference-video
            # duration, audio duration, control-video soundtrack presence:
            # minimax_h3_handler.py:389-445 all need ffprobe/librosa on the real
            # file). Counting it meant three bad client uploads in a row killed a
            # healthy worker.
            LOG.info("validation_rejected", stages=list(outcome.stages),
                     note="not counted against WORKER_FAILURE_BUDGET")
        else:
            if outcome.timed_out or outcome.cancelled:
                code = GENERATION_TIMEOUT
            elif errors:
                code = classify_failure(poison_text)[0]
            else:
                # success=True with no file is a *configuration* refusal, not a
                # poisoned process (failure mode 10): wgp.py:6816-6819 does
                # send_cmd("info", ...); send_cmd("exit"); return True, and "exit"
                # is unhandled by _handle_command (shared/api_cli.py:193-226).
                code = NO_OUTPUT
            note_failure(code, recycle=poisoned)

        if poisoned:
            mark_poisoned(f"{classify_failure(poison_text)[0]} detected in generation output")

        LOG.info("generation_finished", **outcome.summary())
        return outcome

    finally:
        _JOB_LOCK.release()
        if submitted:
            STATS["jobs_served"] = int(STATS.get("jobs_served", 0)) + 1
            if finished:
                _reset_between_jobs(live)
            else:
                # The WanGP worker thread is still alive and still writing into
                # gen[...] and into CUDA. Truncating its lists or calling
                # empty_cache() underneath it would turn a clean recycle into a
                # crash. The process is being replaced; leave it alone.
                LOG.warn("cleanup_skipped", reason="generation still running")


def _safe_cancel_check(check: Callable[[], bool]) -> bool:
    try:
        return bool(check())
    except Exception as exc:  # noqa: BLE001
        LOG.warn("cancel_check_failed", error=f"{type(exc).__name__}: {exc}")
        return False


def _estimate_eta(
    anchors: dict[str, tuple[float, int]],
    phase: str,
    step: Any,
    total_steps: Any,
    now: float,
) -> float | None:
    """Seconds remaining in this phase, from the observed per-step rate.

    Anchored per phase because WanGP restarts the step counter for each denoising
    pass and for each sliding window (``shared/api.py:743-760`` rebuilds
    ``current_step``/``total_steps`` from ``gen["progress_phase"]``).
    """
    try:
        step_i = int(step)
        total_i = int(total_steps)
    except (TypeError, ValueError):
        return None
    if total_i <= 0 or step_i < 0:
        return None
    anchor = anchors.get(phase)
    if anchor is None or step_i < anchor[1]:
        anchors[phase] = (now, step_i)
        return None
    anchor_t, anchor_step = anchor
    done = step_i - anchor_step
    elapsed = now - anchor_t
    if done <= 0 or elapsed <= 0:
        return None
    return round(max(0, total_i - step_i) * (elapsed / done), 1)


# ---------------------------------------------------------------------------
# Between-jobs cleanup
# ---------------------------------------------------------------------------


def _reset_between_jobs(sess: Any) -> None:
    """Truncate the state WanGP appends to forever, then reclaim VRAM.

    Nothing here may raise: it runs in ``run()``'s ``finally`` and must not be
    able to convert a successful generation into an internal error.
    """
    try:
        gen = sess._state["gen"]
    except Exception as exc:  # noqa: BLE001
        LOG.warn("reset_state_unavailable", error=f"{type(exc).__name__}: {exc}")
        gen = None

    if isinstance(gen, dict):
        # These lists are appended forever and never truncated upstream.
        # _collect_outputs (shared/api.py:862-866) slices from a per-job baseline
        # captured in run_cli_job (shared/api_cli.py:16-17), so clearing them
        # between jobs is safe -- the next job's baseline is simply 0.
        for key in ("file_list", "file_settings_list",
                    "audio_file_list", "audio_file_settings_list"):
            value = gen.get(key)
            if isinstance(value, list):
                value.clear()
        artifacts = gen.get("api_output_artifacts")
        if isinstance(artifacts, dict):
            artifacts.clear()  # setdefault'd (shared/api.py:224), never cleared
        # A preview is a PIL image; holding it pins a frame buffer per job.
        if gen.get("preview") is not None:
            gen["preview"] = None

    gc.collect()
    _cuda_cleanup()


def _cuda_cleanup() -> None:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - CPU dev box
        LOG.debug("torch_unavailable", error=str(exc))
        return
    try:
        if not torch.cuda.is_available():
            return
        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        torch.cuda.empty_cache()
        # max_memory_allocated() is a LIFETIME high-water mark: without this
        # reset it can never distinguish "one big job" from "a leak".
        torch.cuda.reset_peak_memory_stats()
        floor = torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception as exc:  # noqa: BLE001
        LOG.warn("cuda_cleanup_failed", error=f"{type(exc).__name__}: {exc}")
        return

    STATS["vram_peak_mb"] = round(peak, 1)
    STATS["vram_floor_mb"] = round(floor, 1)
    _check_vram_creep(floor)


def _check_vram_creep(floor_mb: float) -> None:
    """Recycle when the post-``empty_cache`` floor drifts away from its baseline.

    The floor is live tensors only -- with mmgp that is the resident model, which
    is constant across jobs of any size. Growth is therefore a leak signal, which
    is the only usable one: ``max_memory_allocated`` is a lifetime mark and
    ``memory_reserved`` is caching-allocator noise. ``WORKER_VRAM_LEAK_MB=0``
    disables the check.
    """
    global _VRAM_BASELINE_MB
    slack = _float_env("WORKER_VRAM_LEAK_MB", 4096.0)
    served = int(STATS.get("jobs_served", 0))
    if _VRAM_BASELINE_MB is None or floor_mb < _VRAM_BASELINE_MB:
        _VRAM_BASELINE_MB = floor_mb
        return
    if slack <= 0 or served < 3:
        return
    drift = floor_mb - _VRAM_BASELINE_MB
    if drift > slack:
        mark_poisoned(
            f"VRAM floor grew {drift:.0f} MB above baseline {_VRAM_BASELINE_MB:.0f} MB "
            f"after {served} jobs (WORKER_VRAM_LEAK_MB={slack:.0f})"
        )


def gpu_snapshot() -> dict[str, Any]:
    """Best-effort VRAM numbers for the metrics block. Never raises."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False}
        return {
            "cuda": True,
            "device": torch.cuda.get_device_name(0),
            "vram_alloc_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 1),
            "vram_reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 1),
            "vram_peak_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1),
            "vram_total_mb": round(
                torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1
            ),
        }
    except Exception:  # noqa: BLE001
        return {"cuda": False}


def stats() -> dict[str, Any]:
    """A copy of :data:`STATS` plus the live recycle verdict."""
    snapshot = dict(STATS)
    snapshot["should_recycle"] = should_recycle()
    reason = recycle_reason()
    if reason:
        snapshot["recycle_reason"] = reason
    snapshot["warm"] = is_warm()
    if _WARMED_TYPE:
        snapshot["warmed_type"] = _WARMED_TYPE
    return snapshot


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


def release_model() -> None:
    """Drop the resident model but keep the imported runtime.

    ``wgp.release_model()`` (``wgp.py:233-245``) frees ``wan_model`` and the mmgp
    offload object and sets ``reload_needed = True``, so the next generation
    reloads from disk. Useful for scripts, never for the hot path.
    """
    if _SESSION is None:
        return
    global _WARMED_TYPE
    runtime = _runtime(_SESSION)
    from shared.api import _GENERATION_LOCK, _pushd

    with _JOB_LOCK, _GENERATION_LOCK, _pushd(runtime.root):
        runtime.module.release_model()
    _WARMED_TYPE = None
    _cuda_cleanup()
    LOG.info("model_released")


def shutdown() -> None:
    """Release the model and forget the session singleton.

    The wgp *module* stays imported -- Python cannot unimport it, and
    ``shared.api._RUNTIME`` is module-global, so a subsequent :func:`boot` would
    reuse the same runtime anyway (and must use identical root/config/cli args,
    ``shared/api.py:1064-1066``).
    """
    global _SESSION
    try:
        release_model()
    except Exception as exc:  # noqa: BLE001
        LOG.warn("shutdown_release_failed", error=f"{type(exc).__name__}: {exc}")
    _SESSION = None
