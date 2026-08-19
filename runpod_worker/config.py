"""Env config + ``wgp_config.json`` rendering. No torch, no wgp, no CUDA.

Standard library only. This module must stay importable on a plain CPU runner —
``tests/test_wgp_config_drift.py`` imports it and text-scans ``wgp.py`` beside it.

Everything here is env-overridable. The defaults encode the deployment shape
described in ``docs/RUNPOD_SERVERLESS.md``: one ``model_type`` per endpoint,
weights on a network volume, concurrency 1.
"""

from __future__ import annotations

import json
import os
import shlex
import string
from dataclasses import dataclass, field
from pathlib import Path

from .obs import LOG

__all__ = [
    "WANGP_ROOT",
    "CONFIG_DIR",
    "VOLUME_ROOT",
    "OUTPUT_DIR",
    "JOB_ROOT",
    "MODEL_TYPE",
    "MODEL_CONFIG",
    "TEMPLATE_PATH",
    "REQUIRED_WGP_KEYS",
    "ATTENTION_MODES",
    "checkpoint_paths",
    "lora_root",
    "attention_mode",
    "render_wgp_config",
    "authoritative_keys",
    "ensure_wgp_config",
    "ensure_hf_transfer_sane",
    "WorkerConfig",
    "CONFIG",
    "reload_config",
]


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name) or default).expanduser()


WANGP_ROOT = _env_path("WANGP_ROOT", "/opt/wangp")
CONFIG_DIR = _env_path("WANGP_CONFIG_DIR", "/opt/wangp/config")
VOLUME_ROOT = _env_path("WANGP_VOLUME_ROOT", "/runpod-volume")
OUTPUT_DIR = _env_path("WANGP_OUTPUT_DIR", "/tmp/wangp-out")
JOB_ROOT = _env_path("WANGP_JOB_ROOT", "/tmp/wangp-jobs")

MODEL_TYPE = os.environ.get("WANGP_MODEL_TYPE", "minimax_h3_fl2va_pruned")
# WanGP `config` string: system_configs,system_configs2,system_configs3,configs
# (shared/config_groups.py:1-3). serialize_config_selection() rstrips trailing
# commas (shared/config_groups.py:20) and normalize_config_selection()
# (shared/config_groups.py:22-28) returns its output, so ALWAYS store the
# rstripped form or the reload test at wgp.py:6773 will never match what
# load_models() recorded at wgp.py:4082 (`loaded_config = config_id or ""`,
# stored UNnormalized).
MODEL_CONFIG = os.environ.get("WANGP_MODEL_CONFIG", "").rstrip(",")

#: The baked template this module renders. Ships beside this file.
TEMPLATE_PATH = Path(__file__).resolve().parent / "wgp_config.json.tmpl"

#: The attention modes wgp.py:3303 accepts on the command line. An unlisted
#: value makes `import wgp` raise (wgp.py:3308); an unlisted *config* value
#: instead lands on the silent-refusal path (wgp.py:6815-6818 does
#: send_cmd("info", ...); send_cmd("exit"); return True), which surfaces as a
#: successful generation with zero output files. Reject it here instead.
ATTENTION_MODES = ("auto", "sdpa", "sage", "sage2", "flash", "xformers")

# --------------------------------------------------------------------------
# THE CONFIG FILE TRAP.
#
# wgp.py:2575  ->  if the config file is ABSENT, wgp builds its full default
#                  dict (wgp.py:2576-2616) and writes it (wgp.py:2618-2619).
# wgp.py:2620-2623 -> if the file EXISTS, wgp does `server_config = json.loads(text)`
#                  and REPLACES the defaults wholesale. Only two keys are
#                  setdefault'ed afterwards (wgp.py:2625, 2631).
# wgp.py:3301  ->  attention_mode = server_config["attention_mode"]   # BARE READ
#
# I grepped every bare subscript of server_config
# (`grep -nP 'server_config\["[^"]+"\](?!\s*=)' wgp.py`). It reports lines
# 2630, 3056, 3058, 3060, 3301, 3310, 3311, 3312 and 11379. Filtering to
# statements that execute at module scope (zero indentation) and that no
# earlier module-scope line assigned, setdefault'ed or `if not "x" in`-guarded
# leaves FOUR:
#   3301  attention_mode  <- genuinely unguarded. This is the one that bites.
#   3310  video_profile   \
#   3311  image_profile    > covered TODAY, but only via a function call
#   3312  audio_profile   /
# The three profile keys are setdefault'ed inside _normalize_profile_defaults
# (wgp.py:2661-2668), which runs at wgp.py:2678 — a call a text-scan cannot see
# through. They are additionally short-circuited whenever --profile is on the
# command line, because `force_profile_no >= 0` wins the conditional. Neither
# protection is something to lean on: drop --profile from WANGP_CLI_ARGS and the
# bare reads execute, and one upstream refactor of that helper turns them fatal.
# So REQUIRED_WGP_KEYS lists all four. Our config supplies them unconditionally,
# so requiring them costs nothing and lets tests/test_wgp_config_drift.py assert
# against the naive scan with no hand-maintained exception list.
#   - multi_prompts_gen_type (2630) is assigned two lines earlier (2626-2629).
#   - 3056/3058/3060 and 11379 are inside function bodies, so they run long
#     after import, on a dict wgp has already normalized.
# A hand-written config that omits attention_mode makes `import wgp` die with
# KeyError before shared/api.py:1082 returns — a stack trace that looks nothing
# like a config problem. tests/test_wgp_config_drift.py re-derives this set from
# source on every CI run so an upstream bump cannot reintroduce it quietly.
#
# Two second-order traps in the same neighbourhood, both handled below:
#   * shared/api.py:1071-1072 raises ValueError unless config_path is literally
#     named "wgp_config.json"; a non-default directory is passed through as
#     `--config <parent dir>` (shared/api.py:1073-1077).
#   * a config file that EXISTS but is stale still replaces wgp's defaults, so
#     ensure_wgp_config() merges rather than blindly overwrites: wgp's own
#     migrations (migrate_extension_defaults, wgp.py:2659) write keys back into
#     that file and must survive our next boot.
# --------------------------------------------------------------------------
REQUIRED_WGP_KEYS = ("attention_mode", "video_profile", "image_profile", "audio_profile")


def _split_env_list(value: str) -> list[str]:
    """Split a ``:``- or ``,``-separated env list, preserving order."""
    parts: list[str] = []
    for chunk in value.replace(os.pathsep, ",").split(","):
        chunk = chunk.strip()
        if chunk and chunk not in parts:
            parts.append(chunk)
    return parts


def checkpoint_paths() -> list[str]:
    """Absolute checkpoint roots, most-preferred first.

    ``fl.set_checkpoints_paths`` (shared/utils/files_locator.py:119-123) strips
    each entry and drops empties, then ``_absolute_normalized_path``
    (:16-17) abspaths them, so a relative entry resolves against the process
    CWD — which shared/api.py pushes to the repo root before importing wgp.
    """
    override = os.environ.get("WANGP_CHECKPOINTS_PATHS", "").strip()
    if override:
        return _split_env_list(override)
    paths: list[str] = []
    if VOLUME_ROOT.is_dir():
        paths.append(str(VOLUME_ROOT / "ckpts"))
    paths.append(str(WANGP_ROOT / "ckpts"))
    paths.append(".")  # keep the entry from
    return paths       # shared/utils/files_locator.py:7


def lora_root() -> str:
    """Absolute LoRA root.

    get_lora_root() reads server_config.get("loras_root", DEFAULT_LORA_ROOT)
    (wgp.py:2469-2477, the read is at :2474) and get_lora_dir() deliberately
    returns a RELATIVE path (the os.path.abspath is commented out at
    wgp.py:2498, the bare `return lora_dir` is :2499). An absolute loras_root is
    the only way volume-staged LoRAs are ever found. Note also that a CLI
    ``--loras`` value beats the config value at wgp.py:2470-2475, so do not put
    ``--loras`` in WANGP_CLI_ARGS unless you mean to override this.
    """
    override = os.environ.get("WANGP_LORA_ROOT", "").strip()
    if override:
        return str(Path(override).expanduser())
    if VOLUME_ROOT.is_dir():
        return str(VOLUME_ROOT / "loras")
    return str(WANGP_ROOT / "loras")


def _attention_from_cli_args(cli_args) -> str | None:
    """The ``--attention`` value inside ``cli_args``, if any.

    wgp.py:3302-3308 lets the CLI flag overwrite ``server_config["attention_mode"]``,
    so if the two disagree the CLI silently wins. Read it here so the rendered
    config agrees with what will actually run.
    """
    args = list(cli_args)
    for index, token in enumerate(args):
        if token == "--attention" and index + 1 < len(args):
            return args[index + 1].strip()
        if token.startswith("--attention="):
            return token.split("=", 1)[1].strip()
    return None


def attention_mode(cli_args=()) -> str:
    """The attention mode to bake into the config, validated against wgp's list."""
    explicit = os.environ.get("WANGP_ATTENTION", "").strip()
    from_cli = _attention_from_cli_args(cli_args)
    mode = explicit or from_cli or "sdpa"
    if mode not in ATTENTION_MODES:
        raise RuntimeError(
            f"WANGP_ATTENTION={mode!r} is not one of {list(ATTENTION_MODES)} "
            f"(the whitelist wgp.py:3303 enforces)"
        )
    if explicit and from_cli and explicit != from_cli:
        # Not fatal: wgp.py:3304-3305 makes the CLI value win, and that is the
        # value that ends up in server_config anyway. Say so loudly.
        LOG.warn(
            "attention_mode_conflict",
            env=explicit,
            cli=from_cli,
            effective=from_cli,
            note="wgp.py:3304-3305 lets --attention overwrite server_config",
        )
        mode = from_cli
    return mode


def _template_substitutions(cli_args=()) -> dict[str, str]:
    """Placeholder -> JSON text. Every value is json.dumps()'d, so the template
    holds each placeholder UNQUOTED and escaping is uniform."""
    try:
        profile = float(os.environ.get("WANGP_PROFILE", "4"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"WANGP_PROFILE must be a number: {exc}") from exc
    return {
        "ATTENTION_MODE": json.dumps(attention_mode(cli_args)),
        "PROFILE": json.dumps(profile),
        "TRANSFORMER_QUANT": json.dumps(os.environ.get("WANGP_TRANSFORMER_QUANT", "int8")),
        "TEXT_ENCODER_QUANT": json.dumps(os.environ.get("WANGP_TEXT_ENCODER_QUANT", "int8")),
        "CHECKPOINTS_PATHS": json.dumps(checkpoint_paths()),
        "LORA_ROOT": json.dumps(lora_root()),
        "MODEL_TYPE": json.dumps(MODEL_TYPE),
        "OUTPUT_DIR": json.dumps(str(OUTPUT_DIR)),
    }


def render_wgp_config(cli_args=()) -> dict:
    """Render ``wgp_config.json.tmpl`` into a dict. No filesystem writes."""
    if not TEMPLATE_PATH.is_file():
        raise RuntimeError(
            f"missing baked config template {TEMPLATE_PATH}; the image did not "
            f"COPY runpod_worker/wgp_config.json.tmpl"
        )
    raw = TEMPLATE_PATH.read_text(encoding="utf-8")
    try:
        rendered = string.Template(raw).substitute(_template_substitutions(cli_args))
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"{TEMPLATE_PATH} has an unresolvable placeholder: {exc}") from exc
    try:
        cfg = json.loads(rendered)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{TEMPLATE_PATH} did not render to valid JSON: {exc}") from exc
    return {key: value for key, value in cfg.items() if not key.startswith("__")}


def authoritative_keys(cli_args=()) -> dict:
    """The keys this worker owns outright and rewrites on every boot.

    Anything not listed here is left to the template on a fresh container and to
    wgp's own migrations thereafter.
    """
    try:
        profile = float(os.environ.get("WANGP_PROFILE", "4"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"WANGP_PROFILE must be a number: {exc}") from exc
    return {
        "attention_mode": attention_mode(cli_args),
        # _normalize_profile_defaults (wgp.py:2661-2668, called at :2678)
        # setdefault()s video/image_profile from `profile`, so pinning all three
        # keeps them consistent when --profile is absent from WANGP_CLI_ARGS.
        # audio_profile is left at the template's 3.5, wgp's own default.
        "profile": profile,
        "video_profile": profile,
        "image_profile": profile,
        "last_model_type": MODEL_TYPE,
        "checkpoints_paths": checkpoint_paths(),
        "loras_root": lora_root(),
        "transformer_quantization": os.environ.get("WANGP_TRANSFORMER_QUANT", "int8"),
        "text_encoder_quantization": os.environ.get("WANGP_TEXT_ENCODER_QUANT", "int8"),
        # Never load a model during import: wgp.py:4085 branches on
        # `if not "P" in preload_model_policy` and, with an empty policy, leaves
        # wan_model None and sets reload_needed = True. Weight loading is minutes
        # long and must not push worker start past RunPod's unhealthy threshold.
        "preload_model_policy": [],
        "notification_sound_enabled": 0,   # also forced by shared/api.py:817
        "save_queue_if_crash": 0,
        "video_container": "mp4",          # wgp.py:3333 default
        "video_output_codec": "libx264_8", # wgp.py:3331 default
        "audio_output_codec": "aac_128",   # wgp.py:3339 default
        # shared/api.py:816-831 rewrites these three from `output_dir` on every
        # run anyway; writing them keeps the file honest if output_dir is None.
        "save_path": str(OUTPUT_DIR),
        "image_save_path": str(OUTPUT_DIR),
        "audio_save_path": str(OUTPUT_DIR),
    }



def ensure_hf_transfer_sane() -> str:
    """Reconcile ``HF_HUB_ENABLE_HF_TRANSFER`` with whether ``hf_transfer`` exists.

    Observed on a real RunPod pod (2026-08-19): the base image exports
    ``HF_HUB_ENABLE_HF_TRANSFER=1`` but does NOT ship the ``hf_transfer``
    package. ``huggingface_hub`` does not degrade gracefully there -- every
    download raises

        ValueError: Fast download using 'hf_transfer' is enabled
        (HF_HUB_ENABLE_HF_TRANSFER=1) but 'hf_transfer' package is not
        available in your environment.

    which surfaced as a partial 28 GB warm and four confusing failures rather
    than one clear error. The variable can come from the base image, the RunPod
    endpoint's env, or a template -- none of which this repo controls -- so the
    check belongs at every entry point, not in the Dockerfile alone.

    Returns one of ``"fast"`` (hf_transfer importable, flag left on),
    ``"disabled"`` (flag was on and unusable, forced to "0"), or ``"off"``
    (flag was not set; nothing to do).
    """
    flag = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "")
    if flag.strip().lower() not in ("1", "true", "yes", "on"):
        return "off"
    try:
        import hf_transfer  # noqa: F401
    except Exception:
        # Force the string "0": huggingface_hub reads the env var, and an unset
        # variable is not the same as a falsy one in every version.
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        return "disabled"
    return "fast"

def ensure_wgp_config(cli_args=()) -> Path:
    """Write ``<CONFIG_DIR>/wgp_config.json`` and return its path.

    Layering, in order: baked template -> whatever is already on disk (so wgp's
    own migrations survive a restart) -> the keys this worker owns. Then assert
    REQUIRED_WGP_KEYS, or `import wgp` dies at wgp.py:3301 with a KeyError that
    looks nothing like a config problem.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / "wgp_config.json"  # name is mandatory: shared/api.py:1071-1072

    cfg: dict = render_wgp_config(cli_args)

    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a corrupt config must not wedge boot
            LOG.warn("wgp_config_unreadable", path=str(path), error=str(exc))
            existing = {}
        if isinstance(existing, dict):
            cfg.update(existing)
        else:
            LOG.warn("wgp_config_not_an_object", path=str(path), type=type(existing).__name__)

    cfg.update(authoritative_keys(cli_args))

    missing = [key for key in REQUIRED_WGP_KEYS if key not in cfg]
    if missing:
        raise RuntimeError(f"wgp_config.json is missing required keys {missing}")

    payload = json.dumps(cfg, indent=2, sort_keys=True) + "\n"
    if not path.is_file() or path.read_text(encoding="utf-8") != payload:
        path.write_text(payload, encoding="utf-8")
        LOG.info(
            "wgp_config_written",
            path=str(path),
            attention_mode=cfg["attention_mode"],
            checkpoints_paths=cfg.get("checkpoints_paths"),
            loras_root=cfg.get("loras_root"),
            keys=len(cfg),
        )
    return path


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    """An int env var that cannot make this module unimportable.

    Every numeric knob below is read at import (``CONFIG = WorkerConfig()``), so
    a typo in one env var would otherwise turn a misconfigured endpoint into an
    ``ImportError`` in ``handler.py`` -- no structured log, no fitness check, no
    way to tell what happened. Fall back and say so instead.

    ``minimum`` also rejects the "0 means unlimited" reading where 0 is not a
    legal value: ``WANGP_MAX_FRAMES=0`` used to collapse the frame cap onto
    ``frames_minimum`` (107) rather than lift it.
    """
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        LOG.warn("env_int_invalid", var=name, value=str(raw)[:64], using=default)
        return default
    if minimum is not None and value < minimum:
        LOG.warn("env_int_too_small", var=name, value=value, minimum=minimum,
                 using=default)
        return default
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        LOG.warn("env_float_invalid", var=name, value=str(raw)[:64], using=default)
        return default


@dataclass(frozen=True)
class WorkerConfig:
    """Runtime knobs. Every field re-reads its env var on construction, so tests
    can monkeypatch the environment and build a fresh ``WorkerConfig()``."""

    cli_args: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            shlex.split(
                os.environ.get("WANGP_CLI_ARGS", "--attention sdpa --profile 4 --verbose 1")
            )
        )
    )
    console_output: bool = field(
        default_factory=lambda: os.environ.get("WANGP_CONSOLE", "1") == "1"
    )
    model_type: str = field(
        default_factory=lambda: os.environ.get("WANGP_MODEL_TYPE", "minimax_h3_fl2va_pruned")
    )
    model_config: str = field(
        default_factory=lambda: os.environ.get("WANGP_MODEL_CONFIG", "").rstrip(",")
    )
    default_budget_s: int = field(
        default_factory=lambda: _env_int("WANGP_DEFAULT_BUDGET_S", 1400, minimum=1)
    )
    max_budget_s: int = field(
        default_factory=lambda: _env_int("WANGP_MAX_BUDGET_S", 2600, minimum=1)
    )
    # minimum=0: "no grace at all" is a legitimate setting (and what the tests
    # use); only a negative value is nonsense.
    cancel_grace_s: int = field(
        default_factory=lambda: _env_int("WANGP_CANCEL_GRACE_S", 150, minimum=0)
    )
    progress_interval_s: float = field(
        default_factory=lambda: _env_float("WANGP_PROGRESS_INTERVAL_S", 5.0)
    )
    # 0 is NOT "unlimited": FL2VA declares no frames_maximum, so an uncapped
    # endpoint lets one request schedule hundreds of sliding windows on a billed
    # GPU. minimum=1 sends 0/negative back to the documented default.
    max_frames: int = field(
        default_factory=lambda: _env_int("WANGP_MAX_FRAMES", 362, minimum=1)
    )
    #: Hard ceiling on the number of attachments one request may carry. The byte
    #: budget counts bytes, not entries; thousands of tiny valid images fit inside
    #: it and still cost thousands of inodes plus a list that long handed to
    #: WanGP's validator.
    max_media_items: int = field(
        default_factory=lambda: _env_int("WANGP_MAX_MEDIA_ITEMS", 16, minimum=1)
    )
    # minimum=0 on the byte caps: 0 is a legitimate "never inline anything".
    b64_out_max: int = field(
        default_factory=lambda: _env_int("WANGP_B64_OUT_MAX", 6 * 1024 * 1024, minimum=0)
    )
    b64_in_max: int = field(
        default_factory=lambda: _env_int("WANGP_B64_IN_MAX", 6 * 1024 * 1024, minimum=0)
    )
    media_total_max: int = field(
        default_factory=lambda: _env_int("WANGP_MEDIA_TOTAL_MAX", 7 * 1024 * 1024, minimum=0)
    )
    allow_url_inputs: bool = field(
        default_factory=lambda: os.environ.get("ALLOW_URL_INPUTS", "0") == "1"
    )
    allow_model_switch: bool = field(
        default_factory=lambda: os.environ.get("ALLOW_MODEL_SWITCH", "0") == "1"
    )
    failure_budget: int = field(
        default_factory=lambda: _env_int("WORKER_FAILURE_BUDGET", 3, minimum=1)
    )
    allowed_loras: tuple[str, ...] = field(
        default_factory=lambda: tuple(_split_env_list(os.environ.get("WANGP_ALLOWED_LORAS", "")))
    )

    @property
    def bucket_configured(self) -> bool:
        """Whether ``rp_upload`` has real credentials.

        ``upload_file_to_bucket`` does NOT raise when it cannot build a boto
        client — it returns a ``local_upload/<name>`` path
        (``rp_upload.py:282-301``, fallback at ``:44-61``). Checking the env up
        front is the cheap half of that guard; media_out still has to verify the
        returned string starts with ``http``.
        """
        return all(
            os.environ.get(key)
            for key in ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY")
        )

    def budget_for(self, requested: float | int | None) -> int:
        """Clamp a caller-supplied wall-clock budget into the endpoint's range."""
        if requested is None:
            return self.default_budget_s
        try:
            value = int(float(requested))
        except (TypeError, ValueError):
            return self.default_budget_s
        return max(1, min(value, self.max_budget_s))


#: The process-wide instance. Rebuild with ``reload_config()`` in tests.
CONFIG = WorkerConfig()


def reload_config() -> WorkerConfig:
    """Rebuild ``CONFIG`` from the current environment and return it."""
    global CONFIG
    CONFIG = WorkerConfig()
    return CONFIG
