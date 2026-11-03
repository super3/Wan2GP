# RunPod Serverless Support for `minimax_h3` — Implementation Plan

**Base commit:** `6e35b37` · **Target:** a self-contained RunPod Serverless worker that serves MiniMax H3 through WanGP's in-process API.

Every WanGP call, settings key and line citation below was checked against the working tree. Anything that could not be verified locally is marked **UNVERIFIED** and collected in the appendix.

> Status: **plan only** — no worker code is added by this document. See "Build order" for the implementation checklist.

---

## Summary

Add one new directory, `runpod_worker/`, containing a RunPod Serverless handler that drives WanGP through `shared/api.py`. **No existing repo file is modified** except one link line in the root `README.md`. In particular `requirements.txt`, `Dockerfile` and `entrypoint.sh` are left alone — mutating them in place is what made PR #317 unmergeable.

Five decisions everything else follows from:

| # | Decision | Why (verified) |
|---|---|---|
| 1 | Drive WanGP via `shared.api.init(...)` → `WanGPSession`, never `import wgp` directly and never the sampler classes. | `_ensure_runtime` (`shared/api.py:1061-1097`) is the only code that correctly swaps `sys.argv`, `chdir`s to the repo root for the import, enforces the module-identity check, and calls `download_ffmpeg()`. |
| 2 | **One generation per process, ever.** No `concurrency_modifier`. Scale with `max_workers`. | `shared/api_cli.py:29` holds the module-level `_GENERATION_LOCK` (`shared/api.py:27`) for the whole job, and `shared/api_cli.py:48` installs `contextlib.redirect_stdout/stderr`, which is process-global. |
| 3 | Output is the **already-muxed `.mp4`** from `result.generated_files`. Never `_api={"return_media": True}`. | `_api` returns an *unmuxed* `torch.uint8 [C,F,H,W]` tensor (`shared/api.py:161-178`) that we would have to re-encode; WanGP writes the muxed file to disk unconditionally anyway (`wgp.py:8184-8205`). |
| 4 | **One `model_type` per endpoint**, env-pinned. | `wgp.py:6773` reloads the entire model when `model_type`, `profile` or `config` differs from the loaded one. |
| 5 | Weights on a **network volume** at `/runpod-volume/ckpts` (phase 1); a baked-weights image is a documented phase-2 variant. | Absolute `checkpoints_paths` in `wgp_config.json` makes the same config degrade gracefully to download-on-first-run. See "Model weights strategy" for the full cost comparison — this is the one genuinely close call in the plan. |

The single hardest-won fact in this document: **`import wgp` raises `KeyError: 'attention_mode'` if you hand it a hand-written `wgp_config.json`.** See "Docker image → the config file trap".

---

## Background: what PR #317 did and why this is different

[PR #317](https://github.com/deepbeepmeep/Wan2GP/pull/317) ("Prepare hyvideo for RunPod serverless deployment", opened June 2025, still open) is the only prior upstream attempt. It was **not rejected** — the maintainer replied *"Thanks I am sure lots of users will be happy with this support. Would you mind updating the PR so that I can merge it with WanGP v6?"* and the author never responded. The door is open.

What it got wrong, and how this plan differs:

| PR #317 | This plan |
|---|---|
| No `runpod.serverless.start({"handler": handler})` anywhere; `runpod` not in requirements. `CMD ["python", "handler.py"]` ran a *mock* and exited. | `runpod.serverless.start(...)` is the last statement of `handler.py`; `runpod>=1.12.0` in an additive `requirements-worker.txt`. |
| Returned `{"video_path": "/app/output/…mp4"}` — a path inside an ephemeral container. | Uploads to object storage and returns a URL; base64 only under a hard size cap; never a container-local path. |
| Imported `hyvideo` directly and **deleted `mmgp`, `peft`, `rembg`, `gradio`, … from the shared root `requirements.txt`**. | Zero edits to `requirements.txt`. mmgp offloading is the entire reason to build on WanGP — it is what lets a 20–33B model run on a 48 GB card instead of demanding an A100 80 GB like every `diffusers`-based Wan worker. |
| `COPY ./hyvideo` only — cross-package imports would have failed. | `COPY . /workspace` (with a `.dockerignore`). |
| Dockerfile downloaded `hunyuan_video_720_bf16.safetensors`; the handler loaded `hunyuan_video_avatar_720_bf16.safetensors`. Guaranteed cold-start crash. | A GPU-side verification gate asserts `get_missing_core_file_entries_for_status(...) == []` before the image tag is ever pointed at an endpoint. |
| `runpod.toml` with an invented schema, read by nothing. | No `runpod.toml`. It is absent from RunPod's current docs and its doc URLs 404; the serverless artefacts are `test_input.json` and optionally `.runpod/hub.json` + `.runpod/tests.json`. |
| `runpod/pytorch:2.2.1-cuda12.1`. | Matches the repo's own stack: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`, `torch==2.10.0+cu128` (`Dockerfile:1,40`). |

Also relevant: issue #2001 ("runpod question") is open and unanswered — demand exists. Issue #586 shows that proxying the Gradio UI through RunPod's port proxy breaks (`unkown api call pattern` from `gradio/route_utils.py`), which is independent confirmation that a real handler beats port-forwarding the web UI.

> **Licensing (verified, `docs/API.md:9`):** *"Any product that integrates WanGP should clearly disclose that it uses WanGP in both its user interface and its documentation."* Mirrored in `LICENSE.txt:316`. The worker README and any product built on this endpoint must carry that disclosure.

---

## Architecture

```
POST https://api.runpod.ai/v2/{ENDPOINT}/run          ← never /runsync (see timeouts)
  { "input": {...}, "webhook": "https://…",
    "policy": { "executionTimeout": 3600000 } }
        │
        ▼  RunPod queue (10 MB request cap, 30 min result retention)
┌──────────────────────────────────────────────────────────────────────────────┐
│ WORKER CONTAINER — 1 GPU, concurrency 1, warm across jobs                     │
│                                                                              │
│  ── module import, once per worker lifetime ─────────────────────────────    │
│  worker_config.ensure_wgp_config()   → /opt/wangp/config/wgp_config.json     │
│        checkpoints_paths = ["/runpod-volume/ckpts","/opt/wangp/ckpts","."]   │
│        loras_root        = "/runpod-volume/loras"      (ABSOLUTE — required) │
│        attention_mode    = "sdpa"                      (see config trap)     │
│  engine.boot():                                                             │
│    shared.api.init(root=/opt/wangp, config_path=…/wgp_config.json,          │
│                    output_dir=/tmp/wangp-out,                                │
│                    cli_args=("--attention","sdpa","--profile","4",           │
│                              "--verbose","1"),                               │
│                    console_output=True, console_isatty=False)                │
│      → sys.argv swapped + chdir(root) + import wgp + download_ffmpeg()       │
│      → NO weights loaded (preload_model_policy = [])                         │
│    weights_complete() assertion  ← BEFORE any load, so a bad volume fails    │
│                                     fast instead of downloading 48 GB        │
│  runpod.serverless.register_fitness_check(…)                                 │
│  runpod.serverless.start({"handler": handler})   ← argv untouched by us      │
│                                                                              │
│  ── per job: async handler → asyncio.to_thread(_run) ────────────────────    │
│   1  schema.parse()          allow-list = models/_settings.json (112 keys)   │
│                              ∪ get_default_settings() ∪ ATTACHMENT_KEYS      │
│                              media/`mode`/`_api`/`client_id` stripped        │
│                              frames floored, overlap rounded, seed resolved  │
│   2  idempotency: HEAD the derived object key → early return, 0 GPU seconds  │
│   3  media_in: b64 | volume:// → /tmp/wangp-jobs/{job_id}/in/<key><ext>      │
│                magic-byte sniff → extension (WanGP validates by extension)   │
│   4  session.submit_task(settings)          [non-blocking]                   │
│        ├ drain thread: while not job.done: job.events.get(timeout=0.5)      │
│        │     progress → runpod.serverless.progress_update (throttled 5 s)    │
│        │     stream   → deque(maxlen=400) tail for error payloads            │
│        └ main:  job.result(timeout=budget)                                   │
│              TimeoutError → job.cancel() → result(timeout=grace)             │
│              still alive → refresh_worker: True   (process is poisoned)      │
│   5  probe: ffprobe → w/h/fps/duration/codecs/has_audio; sha256              │
│   6  deliver: presigned PUT → rp_upload → volume → base64 → structured error │
│   7  finally: rmtree(job dir); unlink generated_files;                        │
│               job.release_output_payload(); gen["file_list"].clear();        │
│               torch.cuda.reset_peak_memory_stats(); empty_cache()            │
└──────────────────────────────────────────────────────────────────────────────┘
        │
        ▼  GET /v2/{ENDPOINT}/status/{id}   (or the webhook)
```

---

## File manifest

| Path | New / changed | Purpose |
|---|---|---|
| `.dockerignore` | **new** (repo root) | Keep `.git`, `ckpts/`, `outputs/`, `loras/`, `settings/` out of the image. Only new root file. |
| `README.md` | **changed** — 1 line | Link to `runpod_worker/README.md`. Optional; drop it if you want a truly zero-change PR. |
| `runpod_worker/__init__.py` | new | Package marker. **Directory must not be named `runpod/`** — `shared/api.py:1078` inserts the repo root at `sys.path[0]`, which would shadow the `runpod` pip package. |
| `runpod_worker/handler.py` | new | RunPod entrypoint: `async def handler` + fitness checks + `runpod.serverless.start`. |
| `runpod_worker/config.py` | new | Env-driven `WorkerConfig`; writes/repairs `wgp_config.json`. No heavy imports. |
| `runpod_worker/errors.py` | new | Stable error-code taxonomy. |
| `runpod_worker/schema.py` | new | Request validation + settings assembly. **No torch / wgp import** → CPU-testable. |
| `runpod_worker/media_in.py` | new | Materialize `b64` / `volume://` (and optionally URL) inputs to absolute temp paths. |
| `runpod_worker/media_out.py` | new | Output transport chain + `ffprobe` metadata. |
| `runpod_worker/engine.py` | new | The only module that imports WanGP. Session singleton, event drain, cancel/recycle. |
| `runpod_worker/obs.py` | new | JSON logging to `sys.__stdout__` (captured at import — see failure mode 12). |
| `runpod_worker/wgp_config.json.tmpl` | new | Baked template; `config.py` renders it with absolute paths. |
| `runpod_worker/requirements-worker.txt` | new | `runpod>=1.12.0,<2`. **Additive only.** |
| `runpod_worker/constraints.txt` | new | `pydantic==2.10.6`, `gradio==5.29.0`, `mcp==1.10.1` — pin-guard for the runpod install. |
| `runpod_worker/Dockerfile` | new | Two-stage; consumes `requirements.txt` unmodified. |
| `runpod_worker/test_input.json` | new | Local one-shot job. Read from **process CWD** by the SDK. |
| `runpod_worker/scripts/prefetch_weights.py` | new | GPU-side volume warmer. |
| `runpod_worker/scripts/verify_weights.py` | new | Pre-deploy gate: asserts weight completeness + warms the settings cache. |
| `runpod_worker/scripts/calibrate.py` | new | Timing matrix → measured timeout/cost numbers. |
| `runpod_worker/tests/test_schema.py` | new | CPU-only. |
| `runpod_worker/tests/test_media.py` | new | CPU-only. |
| `runpod_worker/tests/test_wgp_config_drift.py` | new | **Text-scans `wgp.py` for unguarded `server_config["…"]` reads** and asserts our config covers them. Catches the `attention_mode` class of breakage on every upstream bump. |
| `runpod_worker/README.md` | new | Ops runbook + WanGP disclosure. |
| `.github/workflows/worker-ci.yml` | new | CPU tests + hadolint. No GPU, no torch, no weights. |

---

## The handler

### `runpod_worker/config.py` — the config file, and the trap

```python
"""Env config + wgp_config.json rendering. No torch, no wgp."""
from __future__ import annotations
import json, os, shlex
from dataclasses import dataclass, field
from pathlib import Path

WANGP_ROOT   = Path(os.environ.get("WANGP_ROOT", "/opt/wangp"))
CONFIG_DIR   = Path(os.environ.get("WANGP_CONFIG_DIR", "/opt/wangp/config"))
VOLUME_ROOT  = Path(os.environ.get("WANGP_VOLUME_ROOT", "/runpod-volume"))
OUTPUT_DIR   = Path(os.environ.get("WANGP_OUTPUT_DIR", "/tmp/wangp-out"))
JOB_ROOT     = Path(os.environ.get("WANGP_JOB_ROOT", "/tmp/wangp-jobs"))

MODEL_TYPE   = os.environ.get("WANGP_MODEL_TYPE", "minimax_h3_fl2va_pruned")
# WanGP `config` string: system_configs,system_configs2,system_configs3,configs
# (shared/config_groups.py:1-3). serialize_config_selection() (:18-20) rstrips trailing
# commas, so ALWAYS store the rstripped form or the reload test at wgp.py:6773
# will never match what load_models() recorded at wgp.py:4082.
MODEL_CONFIG = os.environ.get("WANGP_MODEL_CONFIG", "").rstrip(",")

# --------------------------------------------------------------------------
# THE CONFIG FILE TRAP.
#
# wgp.py:2575  ->  if the config file is ABSENT, wgp builds its full default
#                  dict (wgp.py:2576-2617) and writes it.
# wgp.py:2623    -> if the file EXISTS, wgp does `server_config = json.loads(text)`
#                  and REPLACES the defaults wholesale. Only two keys are
#                  setdefault'ed afterwards (wgp.py:2625, 2631).
# wgp.py:3301  ->  attention_mode = server_config["attention_mode"]   # BARE READ
#
# I grepped every module-scope bare subscript of server_config
# (`grep -nP 'server_config\["[^"]+"\](?!\s*=)' wgp.py`). That scan returns
# FOUR module-scope reads: attention_mode (3301) and video/image/audio_profile
# (3310-3312).
#   - Only attention_mode can actually KeyError: the three profile keys are
#     setdefault'ed by _normalize_profile_defaults(server_config), CALLED at
#     module scope (wgp.py:2678), which runs before the reads at 3310-3312.
#   - But a text scan cannot see through that call, so the drift test flags all
#     four. Rather than carry a hand-maintained exception list, REQUIRED_WGP_KEYS
#     lists all four and the config supplies them unconditionally. Costs nothing.
#   - multi_prompts_gen_type (2630) is assigned two lines earlier (2626).
#   - The 3310-3312 reads are additionally short-circuited when force_profile_no
#     >= 0, i.e. whenever --profile is in WANGP_CLI_ARGS. Drop that flag and they
#     execute.
# So a hand-written config MUST carry attention_mode or `import wgp` dies with
# KeyError before shared/api.py:1082 returns. tests/test_wgp_config_drift.py
# re-derives this set from source on every CI run.
# --------------------------------------------------------------------------
REQUIRED_WGP_KEYS = ("attention_mode", "video_profile", "image_profile", "audio_profile")


def checkpoint_paths() -> list[str]:
    paths: list[str] = []
    if VOLUME_ROOT.is_dir():
        paths.append(str(VOLUME_ROOT / "ckpts"))
    paths.append(str(WANGP_ROOT / "ckpts"))
    paths.append(".")                     # keep the entry from
    return paths                          # shared/utils/files_locator.py:7


def lora_root() -> str:
    # get_lora_root() reads server_config["loras_root"] (wgp.py:2469-2477) and
    # get_lora_dir() deliberately returns a RELATIVE path (the os.path.abspath
    # is commented out at wgp.py:2498). An absolute loras_root is the only way
    # volume-staged LoRAs are ever found.
    return str(VOLUME_ROOT / "loras") if VOLUME_ROOT.is_dir() else str(WANGP_ROOT / "loras")


def ensure_wgp_config() -> Path:
    """Merge our keys over any pre-existing config so wgp's own migrations survive."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / "wgp_config.json"          # name is mandatory: shared/api.py:1071
    cfg: dict = {}
    if path.is_file():
        try:
            cfg = json.loads(path.read_text())
        except Exception:
            cfg = {}
    cfg.update({
        "attention_mode": os.environ.get("WANGP_ATTENTION", "sdpa"),
        "profile": float(os.environ.get("WANGP_PROFILE", "4")),
        "checkpoints_paths": checkpoint_paths(),
        "loras_root": lora_root(),
        "transformer_quantization":  os.environ.get("WANGP_TRANSFORMER_QUANT", "int8"),
        "text_encoder_quantization": os.environ.get("WANGP_TEXT_ENCODER_QUANT", "int8"),
        "preload_model_policy": [],            # never load a model during import (wgp.py:4085)
        "notification_sound_enabled": 0,
        "save_queue_if_crash": 0,
        "video_container": "mp4",
        "video_output_codec": "libx264_8",
        "audio_output_codec": "aac_128",
        "save_path": str(OUTPUT_DIR),
        "image_save_path": str(OUTPUT_DIR),
        "audio_save_path": str(OUTPUT_DIR),
    })
    missing = [k for k in REQUIRED_WGP_KEYS if k not in cfg]
    if missing:
        raise RuntimeError(f"wgp_config.json is missing required keys {missing}")
    path.write_text(json.dumps(cfg, indent=2))
    return path


@dataclass(frozen=True)
class WorkerConfig:
    cli_args: tuple[str, ...] = field(default_factory=lambda: tuple(
        shlex.split(os.environ.get("WANGP_CLI_ARGS",
                                   "--attention sdpa --profile 4 --verbose 1"))))
    console_output: bool = os.environ.get("WANGP_CONSOLE", "1") == "1"
    default_budget_s: int = int(os.environ.get("WANGP_DEFAULT_BUDGET_S", "1400"))
    max_budget_s: int     = int(os.environ.get("WANGP_MAX_BUDGET_S", "2600"))
    cancel_grace_s: int   = int(os.environ.get("WANGP_CANCEL_GRACE_S", "150"))
    progress_interval_s: float = float(os.environ.get("WANGP_PROGRESS_INTERVAL_S", "5"))
    max_frames: int  = int(os.environ.get("WANGP_MAX_FRAMES", "362"))
    b64_out_max: int = int(os.environ.get("WANGP_B64_OUT_MAX", str(6 * 1024 * 1024)))
    b64_in_max: int  = int(os.environ.get("WANGP_B64_IN_MAX",  str(6 * 1024 * 1024)))
    media_total_max: int = int(os.environ.get("WANGP_MEDIA_TOTAL_MAX", str(7 * 1024 * 1024)))
    allow_url_inputs: bool = os.environ.get("ALLOW_URL_INPUTS", "0") == "1"
    failure_budget: int = int(os.environ.get("WORKER_FAILURE_BUDGET", "3"))

    @property
    def bucket_configured(self) -> bool:
        return all(os.environ.get(k) for k in ("BUCKET_ENDPOINT_URL",
                                               "BUCKET_ACCESS_KEY_ID",
                                               "BUCKET_SECRET_ACCESS_KEY"))

CONFIG = WorkerConfig()
```

### `runpod_worker/schema.py` — request validation

```python
"""Pure validation. No torch, no wgp, no CUDA — the whole point of this split."""
from __future__ import annotations
import copy, json, os, random
from pathlib import Path
from .errors import WorkerError, BAD_REQUEST, INVALID_SETTING, UNKNOWN_SETTING

# Verified against wgp.py:167-168 (15 keys, same order). CI re-derives this by
# text-scanning wgp.py; see tests/test_schema.py::test_attachment_keys_match.
ATTACHMENT_KEYS = ("image_start", "image_end", "image_refs", "image_guide", "image_mask",
                   "video_guide", "video_guide2", "video_mask", "video_source",
                   "audio_guide", "audio_guide2", "audio_source",
                   "replace_voice_sample", "replace_voice_sample2", "custom_guide")

# Extension whitelists WanGP itself enforces (shared/utils/utils.py:36-49).
# NOTE: .webm is NOT accepted; .avi IS.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".jfif", ".pjpeg"}
AUDIO_EXTS = {".wav", ".mp3", ".aac"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}

MEDIA_KIND = {k: "image" for k in ("image_start", "image_end", "image_refs",
                                   "image_guide", "image_mask")}
MEDIA_KIND.update({k: "video" for k in ("video_guide", "video_guide2",
                                        "video_mask", "video_source")})
MEDIA_KIND.update({k: "audio" for k in ("audio_guide", "audio_guide2", "audio_source",
                                        "replace_voice_sample", "replace_voice_sample2")})
LIST_KEYS = {"image_refs"}

MINIMAX_H3_TYPES = {"minimax_h3_fl2va", "minimax_h3_fl2va_pruned",
                    "minimax_h3_ref2va", "minimax_h3_ref2va_pruned"}
FL2VA_TYPES      = {"minimax_h3_fl2va", "minimax_h3_fl2va_pruned"}

# Keys a caller may never set: they steer WanGP away from the generation path or
# let a caller name an arbitrary local file. `mode` in particular flips
# validate_task into the edit branch (wgp.py:1871-1872, wgp.py:8567) which reads
# video_source straight off disk.
FORBIDDEN_KEYS = set(ATTACHMENT_KEYS) | {"mode", "_api", "client_id", "state",
                                         "type", "base_model_type", "priority"}

POISON_MARKERS = ("cuda error", "out of memory", "cublas_status",
                  "device-side assert", "illegal memory access", "nccl")


def floor_frames(n: int, minimum: int, step: int, offset: int) -> int:
    """Mirror shared/utils/frame_scheduler.py:22-29 (floor_frame_count)."""
    n = max(minimum, int(n))
    if step <= 1:
        return n
    lower = ((n - offset) // step) * step + offset
    if lower >= minimum:
        return lower
    import math
    return math.ceil(max(0, minimum - offset) / step) * step + offset


def round_overlap(n: int, step: int, offset: int) -> int:
    """Mirror shared/utils/frame_scheduler.py:41-49 (normalize_overlap).
    This ROUNDS TO NEAREST, it does not floor: 30 -> 35, not 18."""
    n = int(n)
    if n <= 0:
        return 0
    step = max(1, step)
    offset = max(0, offset)
    overlap = ((n - offset + step // 2) // step) * step + offset
    return max(step if offset == 0 else offset, overlap)


class Request:
    __slots__ = ("model_type", "settings", "media", "output", "budget_s",
                 "idempotency_key", "warnings", "resolved")


def parse(payload: dict, *, session, cfg) -> Request:
    if not isinstance(payload, dict):
        raise WorkerError(BAD_REQUEST, "input must be a JSON object")

    pinned = os.environ.get("WANGP_MODEL_TYPE", "minimax_h3_fl2va_pruned")
    mt = str(payload.get("model_type") or pinned).strip()
    if mt not in MINIMAX_H3_TYPES:
        raise WorkerError(BAD_REQUEST,
                          f"model_type must be one of {sorted(MINIMAX_H3_TYPES)}")
    if mt != pinned and os.environ.get("ALLOW_MODEL_SWITCH") != "1":
        raise WorkerError(BAD_REQUEST,
                          f"this endpoint is pinned to '{pinned}'; a switch costs a "
                          f"full release_model()+reload (wgp.py:6773). "
                          f"Set ALLOW_MODEL_SWITCH=1 to permit it.")

    schema = session.get_model_schema(mt)             # shared/api.py:543-556
    if schema is None:
        raise WorkerError(BAD_REQUEST, f"unknown model_type '{mt}'")
    defaults = schema["default_settings"]             # ~20 keys ONLY
    mdef     = schema["model_def"]

    # ---- allow-list -------------------------------------------------------
    # CRITICAL: get_default_settings() is NOT the settings universe. For
    # minimax_h3 it returns exactly the 13-15 keys written by
    # update_default_settings (minimax_h3_handler.py:511-533) plus
    # settings_version/prompt/resolution/flow_shift (wgp.py:3155-3164).
    # `seed`, `activated_loras`, `frames_positions`, `masking_strength`,
    # `skip_steps_cache_type`, `negative_prompt`, `config`, `override_attention`
    # are all ABSENT from it. The real universe is models/_settings.json
    # (112 keys), merged in by clean_settings (wgp.py:1747-1760).
    universe = set(PRIMARY_SETTINGS) | set(defaults)
    user = payload.get("settings") or {}
    if not isinstance(user, dict):
        raise WorkerError(BAD_REQUEST, "input.settings must be an object")

    bad = sorted(set(user) & FORBIDDEN_KEYS)
    if bad:
        raise WorkerError(BAD_REQUEST,
                          f"settings may not contain {bad}; media goes in input.media")
    unknown = sorted(set(user) - universe)
    if unknown:
        raise WorkerError(UNKNOWN_SETTING, f"unknown settings for '{mt}': {unknown}")

    settings = copy.deepcopy(defaults)

    profile_name = payload.get("profile")
    warnings: list[str] = []
    if profile_name:
        settings.update(_load_accel_profile(mdef, str(profile_name)))
    settings.update(user)
    if payload.get("prompt"):
        settings.setdefault("prompt", payload["prompt"])
    settings["model_type"] = mt

    if not str(settings.get("prompt", "")).strip():
        raise WorkerError(BAD_REQUEST, "prompt is required")

    # ---- frame arithmetic, driven by model_def (minimax_h3_handler.py:186-190)
    step   = int(mdef.get("frames_steps", 1) or 1)      # 17
    offset = int(mdef.get("frames_offset", 0) or 0)     # 5
    minimum = int(mdef.get("frames_minimum", 1) or 1)   # 107
    # frames_maximum exists ONLY for ref2va (=737, minimax_h3_handler.py:251).
    # FL2VA has no upper bound anywhere in the headless path — validate_settings
    # (wgp.py:983) never caps it — so the worker MUST impose one or a single
    # request can schedule hundreds of sliding windows on a billed GPU.
    hard_max = int(mdef.get("frames_maximum", cfg.max_frames))
    cap = min(hard_max, cfg.max_frames)

    vl_in = int(settings.get("video_length", 124))
    if vl_in > cap:
        raise WorkerError(INVALID_SETTING,
                          f"video_length={vl_in} exceeds this endpoint's cap of {cap} "
                          f"frames ({cap/24:.1f}s at 24 fps)")
    vl = floor_frames(vl_in, minimum, step, offset)
    if vl != vl_in:
        warnings.append(f"video_length {vl_in} -> {vl} (must be >= {minimum} and "
                        f"= {offset} mod {step}; WanGP floors at wgp.py:6929)")
    settings["video_length"] = vl

    # sliding_window_size shares the frame quantum but has its own bounds
    # (minimax_h3_handler.py:305-306 / :253-254): window_min 124, window_max 481.
    swd = mdef.get("sliding_window_defaults") or {}
    if "sliding_window_size" in settings and swd:
        w = floor_frames(int(settings["sliding_window_size"]), minimum, step, offset)
        w = max(int(swd.get("window_min", minimum)), min(w, int(swd.get("window_max", cap))))
        if w != settings["sliding_window_size"]:
            warnings.append(f"sliding_window_size -> {w}")
        settings["sliding_window_size"] = w
    if "sliding_window_overlap" in settings and swd:
        o = round_overlap(int(settings["sliding_window_overlap"]),
                          int(swd.get("overlap_step", 17)),
                          int(swd.get("overlap_offset", 1)))
        o = min(o, int(swd.get("overlap_max", 120)))
        if o != settings["sliding_window_overlap"]:
            warnings.append(f"sliding_window_overlap "
                            f"{settings['sliding_window_overlap']} -> {o}")
        settings["sliding_window_overlap"] = o

    # ---- resolution must be a multiple of block_size (=32) ----------------
    block = int(mdef.get("block_size", 0) or 0)
    res = str(settings.get("resolution", ""))
    if block and "x" in res:
        w, h = (int(p) for p in res.lower().split("x", 1))
        nw, nh = (w // block) * block, (h // block) * block
        if (nw, nh) != (w, h):
            raise WorkerError(INVALID_SETTING,
                              f"resolution '{res}' is not a multiple of block_size={block}",
                              details=[f"nearest valid: {nw}x{nh}"])

    # ---- informational no-ops, derived from model_def, not hard-coded -----
    if int(mdef.get("guidance_max_phases", 1) or 0) == 0 and "guidance_scale" in user:
        warnings.append("guidance_scale is ignored: guidance_max_phases=0 "
                        "(MiniMaxH3Pipeline.generate takes no CFG argument)")
    if mdef.get("no_negative_prompt") and settings.get("negative_prompt"):
        warnings.append("negative_prompt is ignored: model declares no_negative_prompt")
    if mdef.get("keep_frames_video_guide_not_supported") and settings.get("keep_frames_video_guide"):
        warnings.append("keep_frames_video_guide is not supported by this model")

    # ---- first_block cache thresholds are enforced at wgp.py:1215 ---------
    if settings.get("skip_steps_cache_type") == "first_block":
        allowed = tuple(mdef.get("first_block_cache_thresholds", ()))
        if allowed and float(settings.get("skip_steps_multiplier", 0)) not in allowed:
            raise WorkerError(INVALID_SETTING,
                              f"skip_steps_multiplier must be one of {list(allowed)} "
                              f"for first_block cache")

    # ---- LoRAs: only basenames that exist locally --------------------------
    # get_lora_local_path (wgp.py:3670-3677) returns `lora` verbatim when
    # os.path.isabs(lora), and maps an https URL to
    # os.path.join(lora_dir, basename(url)). So both an absolute path and an
    # arbitrary URL are dangerous; allow-list by basename instead.
    allowed_loras = set(filter(None, os.environ.get("WANGP_ALLOWED_LORAS", "").split(",")))
    for entry in settings.get("activated_loras") or []:
        name = os.path.basename(str(entry).split("|")[0])
        if os.path.isabs(str(entry)):
            raise WorkerError(BAD_REQUEST, "absolute LoRA paths are not allowed")
        if allowed_loras and name not in allowed_loras:
            raise WorkerError(BAD_REQUEST,
                              f"LoRA '{name}' is not staged on this endpoint",
                              details=[f"allowed: {sorted(allowed_loras)}"])

    # ---- determinism -------------------------------------------------------
    seed = int(settings.get("seed", -1))
    if seed < 0:
        seed = random.SystemRandom().randrange(2 ** 31 - 1)
        warnings.append(f"seed was -1; resolved to {seed}")
    settings["seed"] = seed
    settings["batch_size"] = 1
    settings["repeat_generation"] = 1

    media = payload.get("media") or {}
    if not isinstance(media, dict):
        raise WorkerError(BAD_REQUEST, "input.media must be an object")
    for key in media:
        if key not in MEDIA_KIND:
            raise WorkerError(BAD_REQUEST, f"'{key}' is not a WanGP attachment key",
                              details=[f"valid: {sorted(MEDIA_KIND)}"])
    _check_cross_variant(mt, settings, media)

    req = Request()
    req.model_type, req.settings, req.media = mt, settings, media
    req.output = payload.get("output") or {}
    rt = payload.get("runtime") or {}
    req.budget_s = max(60, min(int(rt.get("timeout_s", cfg.default_budget_s)),
                               cfg.max_budget_s))
    req.idempotency_key = rt.get("idempotency_key")
    req.warnings = warnings
    req.resolved = {}
    return req


def _check_cross_variant(mt, settings, media) -> None:
    """Reject combinations WanGP would reject anyway (minimax_h3_handler.py:345-445)
    and MiniMaxH3Pipeline (pipeline.py:387) — but reject them BEFORE the GPU
    spends 2-5 minutes loading the model."""
    vpt = str(settings.get("video_prompt_type", ""))
    apt = str(settings.get("audio_prompt_type", ""))
    if mt in FL2VA_TYPES:
        if media.get("video_guide2") or media.get("audio_guide2"):
            raise WorkerError(BAD_REQUEST, "video_guide2/audio_guide2 are Ref2VA-only")
        for letter, legal in (("image_prompt_type", set("TSEVL")),
                              ("video_prompt_type", set("GVKFI")),
                              ("audio_prompt_type", set("AK2"))):
            value = set(str(settings.get(letter, "")))
            if value - legal:
                raise WorkerError(INVALID_SETTING,
                                  f"{letter} uses letters {sorted(value - legal)} "
                                  f"not supported by FL2VA (allowed: {sorted(legal)})")
        if "F" in vpt:
            n_pos = len(str(settings.get("frames_positions", "")).replace(",", " ").split())
            n_img = len(media.get("image_refs") or [])
            if n_pos != n_img:
                raise WorkerError(INVALID_SETTING,
                                  f"frame injection requires one frames_positions entry "
                                  f"per image_refs entry ({n_pos} positions, {n_img} images)")
        if "2" in apt and ("A" in apt or "K" in apt):
            raise WorkerError(INVALID_SETTING,
                              "audio_prompt_type '2' cannot combine with 'A' or 'K'")
        if ("2" in apt or "K" in apt) and not ("G" in vpt and "V" in vpt
                                               and media.get("video_guide")):
            raise WorkerError(INVALID_SETTING,
                              "audio_prompt_type '2'/'K' require video_prompt_type 'GV' "
                              "and a video_guide file")
    else:
        for letter, legal in (("image_prompt_type", set("TSEVL")),
                              ("video_prompt_type", set("KIPDEV+-")),
                              ("audio_prompt_type", set("ABK"))):
            value = set(str(settings.get(letter, "")))
            if value - legal:
                raise WorkerError(INVALID_SETTING,
                                  f"{letter} uses letters {sorted(value - legal)} "
                                  f"not supported by Ref2VA (allowed: {sorted(legal)})")
        if len(media.get("image_refs") or []) > 9:
            raise WorkerError(INVALID_SETTING, "Ref2VA accepts at most 9 reference images")
        n_vid = (1 if "V" in vpt and media.get("video_guide") else 0) + \
                (1 if "+" in vpt and media.get("video_guide2") else 0)
        n_aud = (1 if "A" in apt else 0) + (1 if "B" in apt else 0)
        n_img = len(media.get("image_refs") or [])
        if n_aud > n_img + n_vid:
            raise WorkerError(INVALID_SETTING,
                              f"Ref2VA needs at least as many reference images+videos as "
                              f"audio references ({n_img + n_vid} visual, {n_aud} audio)")
        if n_img + n_vid + (0 if "K" in apt else n_aud) > 12:
            raise WorkerError(INVALID_SETTING, "Ref2VA accepts at most 12 reference files")
        # Duration checks (2-15 s per clip, <=15 s total) are NOT replicated here:
        # they need ffprobe/librosa on the real files. WanGP enforces them in
        # validate_task and they surface as wangp_validation in seconds.


def _load_accel_profile(mdef: dict, name: str) -> dict:
    """profiles_dir is ["minimax_h3"] (minimax_h3_handler.py:220); the six shipped
    profiles are plain settings fragments."""
    root = Path(os.environ.get("WANGP_ROOT", "/opt/wangp")) / "profiles"
    for folder in mdef.get("profiles_dir", []) or []:
        p = root / str(folder) / f"{name}.json"
        if p.is_file():
            return json.loads(p.read_text())
    available = sorted(q.stem for folder in (mdef.get("profiles_dir") or [])
                       for q in (root / str(folder)).glob("*.json"))
    raise WorkerError(BAD_REQUEST, f"unknown accelerator profile '{name}'",
                      details=[f"available: {available}"])


PRIMARY_SETTINGS: frozenset = frozenset(
    json.loads((Path(os.environ.get("WANGP_ROOT", "/opt/wangp"))
                / "models" / "_settings.json").read_text()))
```

### `runpod_worker/engine.py` — session and job driver

```python
"""The only module that imports WanGP."""
from __future__ import annotations
import collections, gc, sys, threading, time
from pathlib import Path
from . import config as C
from .errors import WorkerError, GENERATION_TIMEOUT, BACKEND_FATAL
from .obs import LOG

_SESSION = None
_BOOT_LOCK = threading.Lock()
_JOB_LOCK = threading.Lock()
STATS = {"jobs_served": 0, "boot_ms": 0}


def boot():
    """Import wgp. Does NOT load weights (preload_model_policy = [])."""
    global _SESSION
    with _BOOT_LOCK:
        if _SESSION is not None:
            return _SESSION
        cfg_path = C.ensure_wgp_config()
        t0 = time.monotonic()
        if str(C.WANGP_ROOT) not in sys.path:
            sys.path.insert(0, str(C.WANGP_ROOT))
        from shared.api import init                      # shared/api.py:1265
        _SESSION = init(
            root=C.WANGP_ROOT,
            config_path=cfg_path,                        # MUST be named wgp_config.json
            output_dir=C.OUTPUT_DIR,
            cli_args=C.CONFIG.cli_args,                  # frozen for process lifetime
            console_output=C.CONFIG.console_output,      # True => lines still reach
            console_isatty=False,                        #   sys.__stdout__
        )
        STATS["boot_ms"] = int((time.monotonic() - t0) * 1000)
        _assert_attachment_keys(_SESSION)
        LOG.info("wgp_imported", boot_ms=STATS["boot_ms"],
                 ckpts=C.checkpoint_paths(), loras=C.lora_root())
        return _SESSION


def _assert_attachment_keys(session) -> None:
    from .schema import ATTACHMENT_KEYS
    live = tuple(getattr(session._ensure_runtime().module, "ATTACHMENT_KEYS", ()))
    if set(live) - set(ATTACHMENT_KEYS):
        raise RuntimeError(f"schema.ATTACHMENT_KEYS is stale; upstream added "
                           f"{sorted(set(live) - set(ATTACHMENT_KEYS))}")


def assert_weights_complete(model_type: str) -> None:
    """Run BEFORE any generation. get_model_availability() alone is not enough:
    get_model_download_status (shared/model_dropdowns.py:442) can report
    'available' while the text encoder is absent, so gate on the explicit
    missing-file enumeration instead (shared/model_dropdowns.py:342)."""
    from shared.api import _pushd
    session = boot()
    runtime = session._ensure_runtime()
    with _pushd(runtime.root):
        deps = runtime.module._get_dropdown_deps()               # wgp.py:13229
        missing = runtime.module.model_dropdowns.get_missing_core_file_entries_for_status(
            deps, model_type)
    if missing:
        raise RuntimeError(f"weights incomplete for {model_type}: {missing}")


def run(settings: dict, *, budget_s: float, emit_progress) -> tuple:
    session = boot()
    if not _JOB_LOCK.acquire(blocking=False):
        raise WorkerError("worker_busy", "a generation is already in flight",
                          retryable=True)
    tail = collections.deque(maxlen=400)
    phase_marks: dict[str, float] = {}
    t0 = time.monotonic()
    try:
        job = session.submit_task(settings)               # shared/api.py:562, non-blocking
        deadline = t0 + budget_s
        cancelled_at = None
        last_emit = 0.0

        # DO NOT use job.events.iter(): SessionStream.iter (shared/api.py:263-271)
        # `continue`s on a queue timeout without yielding, so the loop body — and
        # therefore any wall-clock check inside it — never runs during a silent
        # stretch. A silent stretch is exactly when a hung job needs the check.
        while True:
            event = job.events.get(timeout=0.5)           # returns None on timeout
            now = time.monotonic()
            if event is not None:
                if event.kind == "stream":               # StreamMessage(stream, text)
                    tail.append(f"{event.data.stream}: {event.data.text}"[:512])
                elif event.kind == "progress":           # ProgressUpdate
                    p = event.data
                    phase_marks.setdefault(p.phase, round(now - t0, 1))
                    if now - last_emit >= C.CONFIG.progress_interval_s:
                        last_emit = now
                        emit_progress({"phase": p.phase, "status": p.status,
                                       "pct": p.progress, "step": p.current_step,
                                       "total_steps": p.total_steps,
                                       "elapsed_s": round(now - t0, 1)})
                elif event.kind in ("status", "info"):
                    tail.append(f"{event.kind}: {event.data}")
                    LOG.info("wangp_" + event.kind, text=str(event.data)[:300])
                elif event.kind == "error":
                    tail.append(f"error: {getattr(event.data, 'message', event.data)}")
            if job.done and event is None and job.events.closed:
                break
            if cancelled_at is None and now > deadline:
                cancelled_at = now
                LOG.warn("budget_exceeded_cancelling", budget_s=budget_s)
                job.cancel()                             # cooperative: shared/api.py:895-900
            if cancelled_at and now - cancelled_at > C.CONFIG.cancel_grace_s:
                break

        if not job.done:
            # The daemon worker thread cannot be killed and still holds the
            # process-wide _GENERATION_LOCK. This worker is permanently poisoned.
            raise WorkerError(BACKEND_FATAL,
                              f"generation did not stop within {C.CONFIG.cancel_grace_s}s "
                              f"of cancel; worker will be recycled",
                              details=list(tail)[-20:], retryable=True, recycle=True)

        result = job.result(timeout=10)                  # already done; cheap
        timed_out = cancelled_at is not None
        job.release_output_payload()
        return result, timed_out, list(tail), phase_marks, round(time.monotonic() - t0, 2)
    finally:
        _JOB_LOCK.release()
        STATS["jobs_served"] += 1
        _reset_between_jobs(session)


def _reset_between_jobs(session) -> None:
    import torch
    gen = session._state["gen"]
    # These lists are appended forever and never truncated upstream.
    # _collect_outputs (shared/api.py:862-866) slices from a per-job baseline
    # captured in run_cli_job, so clearing them between jobs is safe.
    for key in ("file_list", "file_settings_list",
                "audio_file_list", "audio_file_settings_list"):
        value = gen.get(key)
        if isinstance(value, list):
            value.clear()
    artifacts = gen.get("api_output_artifacts")
    if isinstance(artifacts, dict):
        artifacts.clear()                                # setdefault'd, never cleared
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()   # max_memory_allocated() is a lifetime
                                           # high-water mark; without this reset it
                                           # cannot detect a per-job leak.
```

### `runpod_worker/handler.py`

```python
#!/usr/bin/env python3
"""RunPod Serverless entrypoint for WanGP / MiniMax H3."""
from __future__ import annotations
import asyncio, os, shutil, sys, time, traceback
from pathlib import Path

sys.path.insert(0, os.environ.get("WANGP_ROOT", "/opt/wangp"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import runpod                                                    # noqa: E402
from runpod_worker import config as C, engine, media_in, media_out, schema  # noqa: E402
from runpod_worker.errors import WorkerError                     # noqa: E402
from runpod_worker.obs import LOG                                # noqa: E402

_consecutive_failures = 0


def _fail(job_id, code, message, *, retryable, details=None, logs=None, recycle=False):
    body = {"error": message, "error_code": code, "retryable": bool(retryable),
            "details": list(details or []), "logs_tail": list(logs or [])[-30:],
            "worker_id": os.environ.get("RUNPOD_POD_ID", "local")}
    if recycle:
        body["refresh_worker"] = True        # popped by the SDK -> stopPod: True
    LOG.error("job_failed", job_id=job_id, error_code=code, recycle=recycle)
    return body


def _run(job) -> dict:
    """Synchronous body. Runs on a worker thread, off the SDK event loop."""
    global _consecutive_failures
    job_id = str(job.get("id") or "local_test")
    job_dir = C.JOB_ROOT / job_id
    t0 = time.monotonic()
    marks: dict = {}
    logs: list = []
    outputs: list[str] = []
    try:
        session = engine.boot()
        req = schema.parse(job.get("input") or {}, session=session, cfg=C.CONFIG)

        key = media_out.output_key(job_id, req)
        cached = media_out.probe_existing(key, req)
        if cached is not None:
            return {"status": "completed", "video": cached, "model_type": req.model_type,
                    "warnings": req.warnings, "metrics": {"idempotent_hit": True}}

        job_dir.mkdir(parents=True, exist_ok=True)
        req.settings.update(media_in.materialize(req.media, job_dir / "in", C.CONFIG))
        marks["inputs_ms"] = int((time.monotonic() - t0) * 1000)

        def emit(p):
            runpod.serverless.progress_update(job, p)    # fire-and-forget thread

        result, timed_out, logs, phase_marks, gen_s = engine.run(
            req.settings, budget_s=req.budget_s, emit_progress=emit)
        marks["generate_s"] = gen_s
        marks["phase_marks_s"] = phase_marks

        if timed_out or result.cancelled:
            _consecutive_failures += 1
            return _fail(job_id, "timeout",
                         f"generation exceeded the {req.budget_s}s budget",
                         retryable=True, logs=logs,
                         recycle=_consecutive_failures >= C.CONFIG.failure_budget)

        if not result.success:
            msgs = [f"[{e.stage}] {e.message}" for e in result.errors]
            stages = {e.stage for e in result.errors}
            code = "wangp_validation" if "validation" in stages else "generation_failed"
            poisoned = any(m in " ".join(msgs).lower() for m in schema.POISON_MARKERS)
            _consecutive_failures += 1
            return _fail(job_id, code, "; ".join(msgs), retryable=poisoned,
                         details=msgs, logs=logs, recycle=poisoned)

        outputs = list(result.generated_files)
        videos = [f for f in outputs if Path(f).suffix.lower() in schema.VIDEO_EXTS]
        if not videos:
            # NOT a poisoned process. generate_media returns True with no file on
            # several *configuration* paths, e.g. an unsupported attention mode
            # (wgp.py:6815-6818 -> send_cmd("info", ...); send_cmd("exit"); return True).
            # `exit` is not handled by _handle_command (shared/api_cli.py:194-226),
            # so it is silently dropped and the task counts as successful.
            return _fail(job_id, "no_output",
                         "WanGP reported success but produced no video file; "
                         "this usually means a configuration was silently refused",
                         retryable=False, details=[f"generated_files={outputs}"],
                         logs=logs)

        up = time.monotonic()
        video = media_out.deliver(Path(videos[0]), key=key, req=req, cfg=C.CONFIG)
        marks["upload_s"] = round(time.monotonic() - up, 2)
        marks["total_s"] = round(time.monotonic() - t0, 2)
        marks.update(engine.STATS)

        _consecutive_failures = 0
        req.resolved = {k: req.settings[k] for k in (
            "model_type", "seed", "video_length", "num_inference_steps", "resolution",
            "flow_shift", "sample_solver", "sliding_window_size",
            "sliding_window_overlap", "config") if k in req.settings}
        return {"status": "completed", "video": video, "model_type": req.model_type,
                "resolved": req.resolved, "warnings": req.warnings, "metrics": marks}

    except WorkerError as exc:
        return _fail(job_id, exc.code, exc.message, retryable=exc.retryable,
                     details=exc.details, logs=logs, recycle=exc.recycle)
    except Exception as exc:                                       # noqa: BLE001
        _consecutive_failures += 1
        LOG.error("unhandled", job_id=job_id, tb=traceback.format_exc())
        return _fail(job_id, "internal_error", f"{type(exc).__name__}: {exc}",
                     retryable=True, logs=logs,
                     recycle=_consecutive_failures >= C.CONFIG.failure_budget)
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
        for f in outputs:
            try:
                os.unlink(f)
            except OSError:
                pass


async def handler(job):
    """Async so the SDK event loop keeps running during a multi-minute generation.

    runpod/serverless/modules/rp_job.py:257 awaits the handler inline on the loop.
    A synchronous handler starves JobScaler.monitor_stop_signals
    (rp_scale.py:144, :275) for the whole job, so a client /cancel is never
    observed and SIGTERM on scale-down never drains. to_thread fixes both.
    (The heartbeat is safe either way -- rp_ping.py:84 forks a real Process.)
    """
    return await asyncio.to_thread(_run, job)


@runpod.serverless.register_fitness_check
def _fitness_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible to the worker")


@runpod.serverless.register_fitness_check
def _fitness_weights():
    engine.assert_weights_complete(os.environ["WANGP_MODEL_TYPE"])


@runpod.serverless.register_fitness_check
def _fitness_transport():
    if os.environ.get("REQUIRE_BUCKET", "0") == "1" and not C.CONFIG.bucket_configured:
        raise RuntimeError("REQUIRE_BUCKET=1 but BUCKET_* env vars are unset")


if __name__ == "__main__":
    # Import wgp eagerly so the ~30-60 s cost lands in worker startup rather than
    # in the first request's executionTime. Weight loading stays lazy: it is
    # minutes long and must not push worker start past RunPod's 7-minute
    # unhealthy threshold.
    if os.environ.get("WANGP_EAGER_BOOT", "1") == "1":
        engine.boot()
    # DO NOT touch sys.argv here: runpod.serverless.start() ->
    # _set_config_args() -> parser.parse_known_args()
    # (runpod/serverless/__init__.py:87-92) still needs --rp_serve_api /
    # --test_input / --rp_log_level.
    runpod.serverless.start({"handler": handler})
```

### Output delivery — the one guard you must not omit

```python
# runpod_worker/media_out.py (excerpt)
def deliver(path: Path, *, key: str, req, cfg) -> dict:
    size = path.stat().st_size
    meta = {"filename": path.name, "size_bytes": size,
            "sha256": sha256_file(path), **ffprobe(path)}
    mode = str(req.output.get("mode", "auto")).lower()
    order = ["presigned", "rp_bucket", "volume", "base64"] if mode == "auto" else [mode]
    tried: list[str] = []

    for transport in order:
        if transport == "presigned":
            url = req.output.get("presigned_url")
            if not url:
                tried.append("presigned: no output.presigned_url given"); continue
            http_put(url, path, req.output.get("content_type", "video/mp4"))
            return {**meta, "transport": "presigned", "url": str(url).split("?")[0]}

        if transport == "rp_bucket":
            if not cfg.bucket_configured:
                tried.append("rp_bucket: BUCKET_* unset"); continue
            from runpod.serverless.utils import rp_upload
            out = rp_upload.upload_file_to_bucket(
                file_name=path.name, file_location=str(path),
                bucket_name=os.environ.get("BUCKET_NAME"),
                prefix=os.path.dirname(key),
                extra_args={"ContentType": "video/mp4"})
            # ---------------------------------------------------------------
            # THE SINGLE MOST LIKELY SILENT-DATA-LOSS BUG IN THIS DESIGN.
            # runpod/serverless/utils/rp_upload.py:300-301 --
            #     if boto_client is None:
            #         return _save_to_local_fallback(file_name, source_path=...)
            # It does NOT raise. It writes to ./local_upload/ and returns a
            # filesystem path that dies with the worker. Never trust the return
            # value without this check.
            # ---------------------------------------------------------------
            if not isinstance(out, str) or not out.startswith("http"):
                tried.append("rp_bucket: rp_upload fell back to local disk "
                             "(boto3 or credentials missing)")
                if mode != "auto":
                    raise WorkerError("upload_failed", tried[-1])
                continue
            return {**meta, "transport": "rp_bucket", "url": out,
                    "expires_in_s": 604800}      # hardcoded ExpiresIn, rp_upload.py:321

        if transport == "volume":
            root = Path(os.environ.get("WANGP_VOLUME_ROOT", "/runpod-volume"))
            if not (root.is_dir() and os.access(root, os.W_OK)):
                tried.append("volume: not mounted or not writable"); continue
            dest = root / "outputs" / key        # namespaced by job id: RunPod warns
            dest.parent.mkdir(parents=True, exist_ok=True)   # that concurrent writes
            shutil.copy2(path, dest)                         # to one volume can corrupt
            return {**meta, "transport": "volume", "volume_path": f"outputs/{key}"}

        if transport == "base64":
            if size > cfg.b64_out_max:
                tried.append(f"base64: {size} B over the {cfg.b64_out_max} B cap")
                if mode != "auto":
                    raise WorkerError("output_too_large", tried[-1], retryable=False)
                continue
            return {**meta, "transport": "base64",
                    "data": base64.b64encode(path.read_bytes()).decode("ascii")}

    raise WorkerError("output_too_large",
                      f"no transport succeeded for a {size}-byte file",
                      details=tried + ["set output.presigned_url, or the BUCKET_* "
                                       "env vars, or attach a network volume"],
                      retryable=False)
```

---

## Request schema

```jsonc
{
  "input": {
    "model_type": "minimax_h3_fl2va_pruned",   // must match WANGP_MODEL_TYPE
    "prompt": "…",                              // convenience alias for settings.prompt
    "profile": "Turbo Lightx2v FL2V 4 Steps v1.0 768p",   // optional accelerator profile
    "settings": { /* any subset of models/_settings.json ∪ get_default_settings() */ },
    "media":    { /* attachment key -> {"b64"|"volume"} (list for image_refs) */ },
    "output":   { "mode": "auto|presigned|rp_bucket|volume|base64",
                  "presigned_url": null, "content_type": "video/mp4" },
    "runtime":  { "timeout_s": 1400, "idempotency_key": null }
  },
  "webhook": "https://your.app/wangp-done",
  "policy":  { "executionTimeout": 3600000 }
}
```

**Merge order:** `get_default_settings(model_type)` → accelerator-profile fragment → `input.settings` → materialized `media` paths → worker overrides (`seed` resolved, `batch_size=1`, `repeat_generation=1`, `model_type` pinned).

`media` values become **absolute** temp paths. This is not optional: `shared/api.py:1001` resolves relative attachment paths against `Path.cwd()` at submit time, and `_pushd(runtime.root)` (`shared/api_cli.py:29`) `chdir`s the whole process during a job.

### Worked examples — verified field names, one per `model_type`

**(a) `minimax_h3_fl2va_pruned` — text-only, 4-step turbo (the cost-optimal default)**

```json
{"input": {
  "model_type": "minimax_h3_fl2va_pruned",
  "profile": "Turbo Lightx2v FL2V 4 Steps v1.0 768p",
  "settings": {
    "prompt": "integrated_multimodal_description: [Shot 1] A five-second cinematic single take inside a rain-lashed glass observatory at midnight. A radio astronomer leans toward a brass receiver and says clearly (S1) <d>[English] If you can hear me, follow this signal.</d>\noverall_soundscape: Rain on the dome, a low electrical hum, three clean receiver tones, and her synchronized voice.\nnon_diegetic_music: One quiet bowed-glass chord rising and fading.",
    "resolution": "832x480",
    "video_length": 124,
    "sample_solver": "euler",
    "image_prompt_type": "",
    "video_prompt_type": "",
    "audio_prompt_type": "",
    "sliding_window_size": 362,
    "sliding_window_overlap": 18,
    "seed": 918273645
  },
  "runtime": {"timeout_s": 900}
}}
```

The profile file (`profiles/minimax_h3/Turbo Lightx2v FL2V 4 Steps v1.0 768p.json`, read verbatim) supplies `activated_loras` = the `…lightx2v_fl2v_turbo_4step_alpha128_v1.0_768p_bf16.safetensors` URL, `loras_multipliers: "1.0"`, `num_inference_steps: 4`, `guidance_scale: 1`, `flow_shift: 6`. `124 = 5 + 17·7` ✓.

**(b) `minimax_h3_fl2va` (full 33B) — first + last frame, 20 steps, First Block Cache**

```json
{"input": {
  "model_type": "minimax_h3_fl2va",
  "settings": {
    "prompt": "integrated_multimodal_description: …\noverall_soundscape: …\nnon_diegetic_music: …",
    "resolution": "832x480",
    "video_length": 209,
    "num_inference_steps": 20,
    "flow_shift": 12.0,
    "sample_solver": "euler",
    "skip_steps_cache_type": "first_block",
    "skip_steps_multiplier": 0.08,
    "skip_steps_start_step_perc": 25,
    "image_prompt_type": "SE",
    "seed": 4242
  },
  "media": {"image_start": {"b64": "iVBORw0KGgo…"},
            "image_end":   {"b64": "iVBORw0KGgo…"}},
  "runtime": {"timeout_s": 2400}
}}
```

`image_prompt_type` letters come from `"TSEVL"`; `E` is legal without `S` because `end_frames_always_enabled: True`. `skip_steps_multiplier` must be one of `0.06 / 0.08 / 0.10 / 0.12 / 0.14` (`first_block_cache_thresholds`), enforced at `wgp.py:1215`. `209 = 5 + 17·12` ✓.

**(c) `minimax_h3_fl2va_pruned` — control video, generate a new soundtrack only**

```json
{"input": {
  "model_type": "minimax_h3_fl2va_pruned",
  "profile": "Turbo Lightx2v FL2V 4 Steps v1.0 768p",
  "settings": {
    "prompt": "integrated_multimodal_description: …\noverall_soundscape: Footsteps on gravel, distant traffic, a door latch.\nnon_diegetic_music: none",
    "video_prompt_type": "GV",
    "audio_prompt_type": "2",
    "denoising_strength": 1.0,
    "video_length": 124,
    "resolution": "832x480",
    "seed": 77
  },
  "media": {"video_guide": {"volume": "clips/plate.mp4",
                            "range": {"start_frame": 0, "end_frame": 240}}}
}}
```

`"2"` requires `G`+`V` in `video_prompt_type` **and** a real `video_guide`, and cannot combine with `A` or `K` — the worker rejects the bad combination pre-flight instead of burning a model load. The `path|start_frame=…,end_frame=…[,audio_track_no=…]` virtual-media suffix is documented at `docs/API.md:456-477` and survives `_absolutize_setting_path` (`shared/api.py:1028-1043`).

**(d) `minimax_h3_ref2va` — two reference images + one audio reference, 20 steps**

```json
{"input": {
  "model_type": "minimax_h3_ref2va",
  "settings": {
    "prompt": "subject_definitions:\n<Subject 1> is the person in <Picture 1>, preserving their exact identity…\nsummary:\n[reference generation] Place <Subject 1> on a deserted midnight railway platform…\nretention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved…\ndetailed_description:\n[Shot 1] … saying clearly (S1) <d>[English] Some journeys begin when the map runs out.</d>\noverall_soundscape: Station ambience, faint wind, one heavy clockwork click.\nnon_diegetic_music: A minimal celesta figure.",
    "video_prompt_type": "KI",
    "audio_prompt_type": "A",
    "image_refs_relative_size": 100,
    "remove_background_images_ref": 0,
    "resolution": "832x480",
    "video_length": 226,
    "num_inference_steps": 20,
    "flow_shift": 12.0,
    "sliding_window_size": 362,
    "sliding_window_overlap": 18,
    "seed": 1234
  },
  "media": {"image_refs": [{"b64": "…"}, {"b64": "…"}],
            "audio_guide": {"volume": "refs/voice.wav"}},
  "runtime": {"timeout_s": 2600}
}}
```

`KI` = first reference image is the main subject and **defines output dimensions**, so the `resolution` you send may not be the resolution you get — read it back from the response's ffprobe block. `226 = 5 + 17·13` ✓, under Ref2VA's `frames_maximum: 737`.

**(e) `minimax_h3_ref2va_pruned` — two reference videos + two audio references, 20 steps**

```json
{"input": {
  "model_type": "minimax_h3_ref2va_pruned",
  "settings": {
    "prompt": "subject_definitions:\n<Subject 1> is the person in <Picture 1>.\nsummary:\n…\nretention_analysis:\n…\ndetailed_description:\n…\noverall_soundscape: …\nnon_diegetic_music: …",
    "video_prompt_type": "IV+-",
    "audio_prompt_type": "AB",
    "image_refs_relative_size": 120,
    "resolution": "832x480",
    "video_length": 124,
    "num_inference_steps": 20,
    "flow_shift": 12.0,
    "seed": 99
  },
  "media": {"image_refs":   [{"b64": "…"}],
            "video_guide":  {"volume": "refs/motion_a.mp4"},
            "video_guide2": {"volume": "refs/motion_b.mp4"},
            "audio_guide":  {"volume": "refs/voice_a.wav"},
            "audio_guide2": {"volume": "refs/voice_b.wav"}}
}}
```

> **There is no Ref2VA turbo/accelerator LoRA in this repo.** `profiles/minimax_h3/` contains exactly six files, and all six reference `fl2v`/generic LoRAs. `grep -rn "ref2v" ` over the tree returns nothing. **Any plan that hands you a `…lightx2v_ref2v_turbo…` filename invented it.** A Ref2VA endpoint runs at 20 steps and costs roughly 4–5× a 4-step FL2VA job unless the owner supplies a LoRA of their own.

Ref2VA limits the worker pre-checks (from `validate_generative_settings`): ≤9 image refs, ≤2 videos, ≤2 audio refs, `#audio ≤ #images + #videos`, ≤12 total files. The duration rules (each video ≥2 s and truncated to 15 s, total ≤15 s; each audio 2–15 s, total ≤15 s) need ffprobe/librosa on the real files, so those stay with WanGP and surface as `wangp_validation` within seconds.

---

## Response schema

RunPod wraps whatever the handler returns. `rp_job.run_job` (`rp_job.py:266-274`) **pops** `error` and `refresh_worker` from your dict, puts the remainder under `output`, sets `status: FAILED` when `error` is truthy, and sets `stopPod: True` when `refresh_worker` is truthy. So the client sees:

**Success**

```json
{"delayTime": 1842, "executionTime": 131940, "id": "60902e6c-…-u1", "status": "COMPLETED",
 "output": {
   "status": "completed",
   "model_type": "minimax_h3_fl2va_pruned",
   "video": {
     "transport": "rp_bucket",
     "url": "https://bucket.s3.…/wangp/minimax_h3/60902e6c-…-u1.mp4?X-Amz-…",
     "expires_in_s": 604800,
     "filename": "2026-08-18-14h22m01s_seed918273645_….mp4",
     "size_bytes": 8412663, "sha256": "9f2c…",
     "duration_s": 5.167, "fps": 24, "width": 832, "height": 480,
     "container": "mp4", "video_codec": "h264",
     "has_audio": true, "audio_codec": "aac", "audio_sample_rate": 32000,
     "audio_channels": 2
   },
   "resolved": {"model_type": "minimax_h3_fl2va_pruned", "seed": 918273645,
                "video_length": 124, "num_inference_steps": 4, "resolution": "832x480",
                "flow_shift": 6, "sample_solver": "euler",
                "sliding_window_size": 362, "sliding_window_overlap": 18},
   "warnings": ["guidance_scale is ignored: guidance_max_phases=0"],
   "metrics": {"inputs_ms": 61, "generate_s": 128.4, "upload_s": 2.14, "total_s": 131.9,
               "phase_marks_s": {"loading_model": 0.4, "encoding_text": 61.2,
                                 "inference_stage_1": 78.5, "decoding": 118.9},
               "jobs_served": 7, "boot_ms": 41002}
 }}
```

Every field in `video` after `transport`/`url` comes from `ffprobe` on the produced file, not from the request. That matters for Ref2VA `KI`, where the first reference image defines output dimensions.

**Progress**, readable mid-flight via `/status`. Each `progress_update` **overwrites** the previous one — this is a status field, not an append-only log:

```json
{"status": "IN_PROGRESS",
 "output": {"phase": "inference_stage_1", "status": "Prompt 1/1 | Denoising | 7.2s",
            "pct": 44, "step": 2, "total_steps": 4, "elapsed_s": 63.4}}
```

`phase` is one of the normalized values from `_normalize_phase` (`shared/api.py:1100-1118`): `inference_stage_1/2/3`, `loading_model`, `encoding_text`, `decoding`, `downloading_output`, `cancelled`, or the fallback `inference`. `pct` is an **estimate** from `_estimate_progress` (`shared/api.py:1125-1161`), banded per phase and never reaching 100 — completion is signalled by the job status, not by `pct`.

**Failure**

```json
{"status": "FAILED", "id": "…",
 "error": "[validation] MiniMax H3 frame injection requires one position per Reference Image (found 0 positions and 2 images)",
 "output": {"error_code": "wangp_validation", "retryable": false,
            "details": ["[validation] MiniMax H3 frame injection requires …"],
            "logs_tail": ["status: Loading model MiniMax H3 FL2VA Pruned 20B…"],
            "worker_id": "abc123"}}
```

Stable `error_code` values (branch on these, never on message text): `bad_request` · `unknown_setting` · `invalid_setting` · `media_too_large` · `media_fetch_failed` · `ssrf_blocked` · `wangp_validation` · `generation_failed` · `timeout` · `no_output` · `output_too_large` · `upload_failed` · `worker_busy` · `backend_fatal` · `internal_error`.

### The large-output transport decision

RunPod documents payload limits of **10 MB for `/run`** and 20 MB for `/runsync`. Two things make 10 MB the operative number and base64 a debug affordance rather than a transport:

1. A video generation takes minutes. `/runsync` caps its wait at 300 000 ms (5 minutes) and retains results for 1–5 minutes, so callers **must** use `/run` + `/status` or a webhook. The 20 MB figure is unreachable in practice.
2. Base64 inflates by 4/3 plus JSON escaping, so a 10 MB envelope holds roughly 7.5 MB of binary at best. A 5 s 832×480 H.264 clip with an AAC-128 track is plausibly 2–8 MB; 15 s or 720p is not.

Hence the chain: **caller-supplied presigned PUT** (no secrets on the worker, best for multi-tenant) → **`rp_upload` to your own bucket** (7-day presigned URL, hardcoded `ExpiresIn=604800`) → **network volume** (cheap, but RunPod's volume S3 API cannot presign, so you cannot hand a client a link) → **base64 under 6 MB** → **structured `output_too_large` error with the exact env vars to set**. Never a truncated payload, never a dead local path.

---

## Model weights strategy

### Footprint

Sizes below are from the research dossier (read off the Hugging Face tree API) — I did **not** re-verify them from this container. Treat them as ±. The file *names* and *which one gets picked* are verified from `defaults/minimax_h3_*.json` and `models/minimax_h3/minimax_h3_handler.py:15-25`.

| Artifact | v1: pruned 20B, int8 + INT8 TE | Full 33B, int8 + INT8 TE |
|---|---:|---:|
| Transformer `MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors` | 21.06 GB | 34.04 GB (`…FL2VA_int8_convrot`) |
| `Qwen3-VL-32B-Instruct-layer50_quanto_bf16_int8.safetensors` | 26.72 GB | 26.72 GB |
| `MiniMax-H3-video_vae_fp16.safetensors` | 5.21 GB | 5.21 GB |
| `MiniMax-H3-audio_vae_fp32.safetensors` | 0.61 GB | 0.61 GB |
| Tokenizer/config JSON | ~10 MB | ~10 MB |
| **WanGP core shared assets** ¹ | ~5 GB **(UNVERIFIED size)** | ~5 GB |
| Turbo LoRA (FL2VA only) | 1.38 GB | 1.38 GB |
| **Total** | **≈60 GB** | **≈73 GB** |

¹ `download_models(file_type=0)` unconditionally runs `process_files_def(**query_core_shared_model_files())` **and** MatAnyone (`wgp.py:3585-3587`). Verified contents (`wgp.py:3545-3557`): DWPose (`dw-ll_ucoco_384.onnx`, `yolox_l.onnx`), scribble, RAFT, Depth-Anything-V2-vitl, wav2vec ×2, BS-RoFormer, pyannote, det_align. These are **not** optional and **will** download on first request if you do not pre-stage them. This is the most commonly missed multi-GB download in a WanGP container.

Swapping the text encoder to `gguf_q4_k_m` (14.58 GB) drops v1 to ≈48 GB; `gguf_q2_k` (8.49 GB) to ≈42 GB. Selected via the `config` string slot 1 (`system_configs`), e.g. `"gguf_q4_k_m"`.

### Decision: network volume for phase 1

| | Network volume (phase 1) | Baked into image (phase 2) |
|---|---|---|
| Image size | ~20–30 GB | ~70–90 GB — at or over the documented 80 GB cap |
| Weight read | 200–400 MB/s, **billed** (~150–250 s first job) | local NVMe, unbilled if image pull is unbilled |
| Datacenter | **pinned to the volume's DC** → smaller GPU pool, worse failover | any |
| Iterate on quantization / add a model | copy files | full rebuild + repush of ~80 GB |
| GitHub image builder | viable | not viable (30-min `docker build` step limit) |

Phase 1 picks the volume because iteration speed dominates while the numbers are still unmeasured, and because `checkpoints_paths[0] = "/runpod-volume/ckpts"` means downloads land there and lookups read from there — **the same config degrades gracefully to download-on-first-run if the volume is empty**, with no branching.

Two honest caveats:

- **Whether the container's "start time" is billed is contradictory in RunPod's own docs.** One page lists `Initializing` as an unbilled worker state; the pricing page says charges cover "Start time: initializing the container and loading models into GPU memory." I have not resolved this. If start time *is* billed, baking wins by ~$0.08/cold worker on an L40S and phase 2 becomes more urgent.
- **Concurrent cold workers on an empty volume can corrupt it.** RunPod warns that simultaneous writes from multiple workers to one volume may cause data corruption. The download-on-first-run fallback is therefore a **single-worker-only degraded mode**: keep `max_workers=1` until the prefetch has verified completeness.

### Commands

```bash
# 1) Create the volume (200 GB) in a datacenter that carries your GPU tier.
curl -X POST https://rest.runpod.io/v1/networkvolumes \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"name":"wangp-h3-us-ca-2","size":200,"dataCenterId":"US-CA-2"}'
# ~$14/mo at $0.07/GB/mo. Size can be increased, never decreased.

# 2) Launch a temporary GPU **Pod** with the volume attached and the worker image.
#    A GPU is mandatory: `import wgp` calls torch.cuda.get_device_capability at
#    module scope (wgp.py:2508) and again at shared/attention.py:14.
#
#    *** MOUNT PATH ASYMMETRY -- the least obvious thing in this deployment ***
#    Serverless workers mount a network volume at /runpod-volume.
#    Pods mount it at /workspace (it replaces the pod's default volume disk).
#    So on the Pod you must either bind-mount or point the env var:
export WANGP_VOLUME_ROOT=/workspace          # or:  mount --bind /workspace /runpod-volume
#    Clone the repo OUTSIDE the volume so 5 GB of source does not ride along.

# 3) Prefetch, with the SAME transformer_quantization the workers will run.
python3 -m runpod_worker.scripts.prefetch_weights \
    --root /opt/wangp --config /opt/wangp/config/wgp_config.json \
    minimax_h3_fl2va_pruned

# 4) Stage the accelerator LoRA by BASENAME. get_lora_local_path (wgp.py:3670-3677)
#    maps an https:// entry in activated_loras to
#    os.path.join(lora_dir, basename(url)), so a baked file resolves with zero network.
mkdir -p "$WANGP_VOLUME_ROOT/loras/minimax_h3"
wget -P "$WANGP_VOLUME_ROOT/loras/minimax_h3" \
  https://huggingface.co/DeepBeepMeep/MiniMax-H3/resolve/main/loras/minimax_h3_lightx2v_fl2v_turbo_4step_alpha128_v1.0_768p_bf16.safetensors

# 5) Verify with the exact enumeration the worker's fitness check uses.
python3 -m runpod_worker.scripts.verify_weights minimax_h3_fl2va_pruned
#   -> asserts get_missing_core_file_entries_for_status(deps, mt) == []
#   -> asserts get_model_availability(mt)["available"]
#   -> calls get_default_settings(mt) once so settings/<mt>_settings.json exists
#      (wgp.py:3174 json.dump()s it on first call -- do not let that happen
#      at request time on a read-only or slow filesystem)
#   -> prints the resolved transformer + text-encoder filenames for eyeballing
du -sh "$WANGP_VOLUME_ROOT/ckpts"

# 6) Terminate the Pod. Attach the volume to the Serverless endpoint:
#    Serverless -> endpoint -> Manage -> Edit Endpoint -> Advanced -> Network Volumes
```

`prefetch_weights.py` mirrors the file-list construction in `load_models` (`wgp.py:3960-4043`) — all signatures verified in the tree:

```python
model_def = wgp.get_model_def(mt)                               # wgp.py:2799
if config_id:                                                   # wgp.py:3959-3962
    groups = wgp.get_model_config_groups(mt, model_def)         # wgp.py:2918
    model_def = model_def.copy()
    for _, _, cfg in wgp.model_config_groups.selected_model_configs(groups, config_id):
        model_def.update(cfg)
main = wgp.get_model_filename(model_type=mt,                    # wgp.py:2922
                              quantization=wgp.transformer_quantization,
                              dtype_policy=wgp.transformer_dtype_policy,
                              model_def=model_def)
wgp.download_models(main, mt, 0, 1, model_def=model_def)        # wgp.py:3576
#   file_type=0 also pulls query_core_shared_model_files() + MatAnyone (3585-3587)
#   and the handler's query_model_files() manifest (VAEs + tokenizer JSON).
te_urls = wgp.get_model_recursive_prop(mt, "text_encoder_URLs", # wgp.py:2891
                                       return_list=True, model_def=model_def)
if te_urls:
    te = wgp.get_model_filename(model_type=mt,
                                quantization=wgp.text_encoder_quantization,
                                dtype_policy=wgp.transformer_dtype_policy, URLs=te_urls)
    wgp.download_models(te, mt, 2, -1,                          # mirrors wgp.py:4043
                        force_path=model_def.get("text_encoder_folder"),
                        model_def=model_def)
```

**Warm with the same `transformer_quantization` you run with.** `get_model_filename` (`wgp.py:2922-2984`) picks between the two entries in the model's `URLs` list by matching quantization tokens in the basename. Warm as `bf16` and run as `int8` and every cold start re-downloads 21 GB, billed.

---

## Docker image

Two stages: a `devel` builder for SageAttention, and a runtime that still needs a C toolchain (see below).

```dockerfile
# ---------- stage 1: compile SageAttention (needs nvcc; no GPU at build time) ----
ARG CUDA_ARCHITECTURES="8.0;8.6;8.9;9.0"
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS builder
ARG CUDA_ARCHITECTURES
ENV DEBIAN_FRONTEND=noninteractive TORCH_CUDA_ARCH_LIST=${CUDA_ARCHITECTURES} \
    FORCE_CUDA=1 MAX_JOBS=8
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev git cmake ninja-build && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.10.0+cu128 \
      --index-url https://download.pytorch.org/whl/cu128
COPY runpod_worker/scripts/patch_sage_setup.py /tmp/patch_setup.py
RUN git clone --depth 1 https://github.com/thu-ml/SageAttention.git /tmp/sage && \
    cd /tmp/sage && python3 /tmp/patch_setup.py && \
    pip wheel --no-build-isolation --no-deps -w /wheels .

# ---------- stage 2: runtime ----------------------------------------------------
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    HOME=/home/user HF_HOME=/home/user/.cache/huggingface \
    SDL_AUDIODRIVER=dummy PULSE_RUNTIME_PATH=/tmp/pulse-runtime \
    TORCH_ALLOW_TF32_CUBLAS=1 TORCH_ALLOW_TF32_CUDNN=1 \
    WANGP_ROOT=/opt/wangp WANGP_CONFIG_DIR=/opt/wangp/config \
    WANGP_OUTPUT_DIR=/tmp/wangp-out WANGP_JOB_ROOT=/tmp/wangp-jobs \
    WANGP_VOLUME_ROOT=/runpod-volume \
    WANGP_MODEL_TYPE=minimax_h3_fl2va_pruned \
    WANGP_ATTENTION=sdpa WANGP_PROFILE=4 \
    WANGP_CLI_ARGS="--attention sdpa --profile 4 --verbose 1" \
    PYTHONPATH=/opt/wangp
# `devel`, not `runtime`: requirements.txt:64 pins `insightface==0.7.3 ; sys_platform=="linux"`,
# which ships sdist-only on PyPI and compiles mesh_core_cython. pycocotools is
# the same. The repo's own Dockerfile gets away with this only because it builds
# in the devel base. Swapping to cudnn-runtime saves ~3 GB and breaks pip install.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential git wget curl \
      libgl1 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/wangp
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
      torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt /opt/wangp/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# runpod 1.12.0 requires fastapi[all]>=0.141.1, boto3>=1.43.66, cryptography>=50,
# paramiko>=5, colorama<0.4.7. fastapi[all] drags in pydantic-settings /
# pydantic-extra-types, which are free to bump pydantic and break the
# pydantic==2.10.6 / gradio==5.29.0 / mcp==1.10.1 pins in requirements.txt.
# Install under constraints and gate the layer on `pip check`.
COPY runpod_worker/requirements-worker.txt runpod_worker/constraints.txt /opt/wangp/runpod_worker/
RUN pip install --no-cache-dir -c runpod_worker/constraints.txt \
      -r runpod_worker/requirements-worker.txt && pip check

# The application. The repo's own Dockerfile deliberately ships NO app code --
# it bind-mounts $(pwd):/workspace at run time. A serverless image must carry it.
COPY . /opt/wangp

# ffmpeg_bins: shared/api.py:1090 calls module.download_ffmpeg() on every runtime
# init. Baking it removes a network dependency from the cold path.
# (This RUN needs no GPU: shared/ffmpeg_setup is importable standalone.)
RUN python3 -c "import sys; sys.path.insert(0,'/opt/wangp'); \
from shared.ffmpeg_setup import download_ffmpeg; download_ffmpeg()"

# The repo root MUST be writable: `import wgp` does os.mkdir('settings')
# (wgp.py:2549), writes wgp_config.json (2618), and get_default_settings
# json.dump()s settings/<model_type>_settings.json (wgp.py:3174).
# loras_url_cache_v2.json is also a bare relative path. Read-only rootfs breaks this.
RUN useradd -u 1000 -ms /bin/bash user && \
    mkdir -p /opt/wangp/config /opt/wangp/settings /opt/wangp/ckpts /opt/wangp/loras \
             /tmp/wangp-out /tmp/wangp-jobs /home/user/.cache && \
    chown -R user:user /opt/wangp /tmp/wangp-out /tmp/wangp-jobs /home/user

USER user
# No ENTRYPOINT. The repo's entrypoint.sh burns ~95 lines of nvidia-smi
# diagnostics per cold start, does an `su -p user -c` hop, and ends in
# `python3 wgp.py --listen` -- which enters wgp.py's __main__ (line 13623),
# launches Gradio, and can block on select.select([sys.stdin],...) for 2 s
# when startup.lock exists (wgp.py:13787-13812).
CMD ["python3", "-u", "/opt/wangp/runpod_worker/handler.py"]
```

Notes:

- **`--platform linux/amd64` is mandatory on the build command.** ARM images are rejected by RunPod.
- **`CUDA_ARCHITECTURES="8.0;8.6;8.9;9.0"`.** The repo default `"8.0;8.6"` (`Dockerfile:20`) excludes L4/L40S/4090 (8.9) and H100 (9.0) — exactly the fleet a serverless endpoint schedules on. Include 8.0 too if A100 is in your GPU priority list: `is_sage2_supported()` (`shared/sage2_core.py`) checks the *device* capability, not which arches the wheel was built for, so a wheel missing SM80 fails at the first attention kernel launch, minutes into a billed job.
- **Default `--attention sdpa`, not `sage`.** `resolve_attention_mode` (`shared/attention.py:294-302`) *raises* on an unsupported mode, and `wgp.py:6815-6818` turns an unsupported override into `send_cmd("exit"); return True` — a silent, file-less "success". Flip to `sage2` via `WANGP_CLI_ARGS` only after you have confirmed the endpoint's GPU tier and measured that it is actually faster.
- **Never tag `:latest`.** RunPod caches images per host and a mutable tag silently serves stale code. Use `YYYY.MM.DD-N`.
- **`--attention` is validated against a whitelist at `wgp.py:3303`**: `auto`, `sdpa`, `sage`, `sage2`, `flash`, `xformers`. `sage3` and `radial` are rejected at the CLI even when installed; they are reachable only through `wgp_config.json`'s `attention_mode`. (`sol` is an *override* mode, set per-task via `override_attention`, not via `--attention`.)

### `.dockerignore`

```
.git
ckpts
loras
outputs
settings
wgp_config.json
**/__pycache__
*.pyc
```

---

## Cold start, timeouts and GPU sizing

### GPU tier

| Priority | Tier | VRAM | $/s | Rationale |
|---|---|---|---:|---|
| 1 | **L40S / L40 / RTX 6000 Ada** | 48 GB | 0.00053 | SM89 → Sol-Attn eligible (`sol_attention: True`; requires BF16 + Triton ≥3.6 + SM89/90/100/120). |
| 2 | **A6000 / A40** | 48 GB | 0.00034 | 0.64× the price, SM86 so no Sol-Attn. Roughly cost-neutral per job, worse p90 latency. |
| 3 | **A100 80 GB** | 80 GB | 0.00076 | Availability fallback. SM80 — include 8.0 in `CUDA_ARCHITECTURES` or exclude this tier. |

VRAM is not the binding constraint: the README quotes **5–6 GB for 5 s (124 frames) and 8–9 GB for 15 s at 832×480** with mmgp block-swapping. **System RAM is**, because mmgp streams ~48–60 GB of weights. RunPod does not publish per-tier host RAM for serverless workers — **log `psutil.virtual_memory().total` from the first staging worker** and fall back from `--profile 4` to `--profile 5` if it is under ~64 GB. Profile 4 (`LowRAM_LowVRAM`, ≥32 GB RAM / ≥12 GB VRAM) is WanGP's own default and the right starting point.

Do not list 24 GB tiers: a 21 GB int8 transformer plus activations plus a 26.7 GB text encoder forces continuous PCIe block-swapping. Cheap per second, expensive per generation.

### Cold-start budget (L40S, weights on a volume, pruned 20B int8)

| Phase | Estimate | Billed? | Confidence |
|---|---|---|---|
| Image pull, ~20–30 GB compressed, uncached host | 90–240 s | See caveat below | Medium |
| Image pull, cached host / FlashBoot revival | 0–15 s | — | High |
| Container start + `import wgp` (28 handlers imported twice, 217 model defs, torch, gradio patches, mmgp) | **25–60 s** | Yes | Medium |
| Fitness checks (`torch.cuda`, weight enumeration) | 2–5 s | Yes | High |
| **First job: 48–60 GB read from volume @ 200–400 MB/s** | **150–250 s** | **Yes** | Medium |
| Same, from baked local NVMe | 45–90 s | Probably no | Low |
| Generation, 124 frames @ 832×480, **4-step turbo** | **UNMEASURED — est. 90–260 s** | Yes | **Low** |
| Generation, 124 frames @ 832×480, **20 steps stock** | **UNMEASURED — est. 330–900 s** | Yes | **Low** |
| ffmpeg mux + ffprobe + upload of 5–20 MB | 5–20 s | Yes | High |

**Nothing in this repo or its README states an H3 generation wall-clock.** The README gives VRAM only. Every second-level number below "first job" is extrapolation from 14B-Wan-class timings scaled by parameter count and frame count. `scripts/calibrate.py` exists to replace them; **do not sign an SLA before running it.**

> **Billing caveat, unresolved:** RunPod's worker-state table lists `Initializing` as not billed, while the pricing page says charges cover "Start time: initializing the container and loading models into GPU memory." Assume `import wgp` and the weight load **are** billed (the conservative reading) and treat the image pull as unbilled.

### Timeout configuration, derived

| Knob | Value | Reason |
|---|---|---|
| Endpoint **Execution timeout** | **3600 s** | Default is 600 s and would kill the first request outright. Must exceed the handler budget so our cooperative cancel always wins the race against RunPod's hard kill. |
| `WANGP_DEFAULT_BUDGET_S` | **1400** | Handler-side. |
| `WANGP_MAX_BUDGET_S` | **2600** | Ceiling on `runtime.timeout_s`. |
| `WANGP_CANCEL_GRACE_S` | **150** | `job.cancel()` sets `gen["abort"]` and `wan_model._interrupt` (`shared/api.py:895-900`); it lands at the model's next interrupt check — one denoising step. 150 s is longer than any single step at any supported config. |
| Budget arithmetic | 2600 + 150 + ~60 (probe/upload/cleanup) ≈ 2810 < 3600 ✓ | Leaves ~13 min of headroom. Do **not** set the handler budget equal to the endpoint timeout. |
| Endpoint **Job TTL** | leave at the 24 h default | The timer starts at *submission*, so it covers queue wait. Lowering it below the default is a way to drop jobs whose caller has given up — worth doing only once you have measured queue depth. |
| Endpoint **Idle timeout** | **180 s** (default is 5 s) | 180 s idle on an L40S = $0.095. Reloading weights = ~200 s = $0.106, plus 3+ minutes of latency. Keeping the worker warm wins for any inter-arrival gap under ~3 minutes. |
| **Max workers** | ≥3 once the volume is warm; **1** while it is not | Concurrent writes to one volume can corrupt it. |
| **Active workers** | 0 to start | 1 always-on L40S = $1.91/h = **$1,374/mo**, but eliminates the 150–250 s weight load entirely. Sizing formula: `active = (req/min × duration_s) / 60`. |
| **Autoscaling** | Request count, scaler 1 | Queue-delay scaling with a 4 s threshold reacts far too late for multi-minute jobs. |
| `concurrency_modifier` | **unset** | The SDK default already returns 1 (`rp_scale.py`). |
| FlashBoot | on (default) | Free; helps most under steady traffic. |

**Callers must use `/run` + `/status` or a webhook.** `/runsync` waits 90 s by default, 300 s maximum (`?wait=` in ms, 1000–300000). Put this in the first paragraph of the README.

**Silent killer:** an endpoint with **no requests for 3 days** has max workers auto-reduced to 2, and to **0 after 7 days**, and stays reduced until raised manually. Schedule a weekly synthetic job on any low-traffic endpoint.

### Cost per generation (L40S @ $0.00053/s), using the estimate midpoints

| Scenario | Billed seconds | Cost |
|---|---:|---:|
| 4-step turbo, warm worker | ~180 | **$0.095** |
| 20-step stock, warm worker | ~620 | **$0.329** |
| Cold-start adder (import + first-job weight read from volume) | ~240 | **+$0.127** |
| Idle-timeout tail per warm window | 180 | **+$0.095** |
| 4-step turbo on A6000/A40 @ $0.00034 | ~230 | **$0.078** |

The two dominant controllable costs are the idle tail on bursty traffic and cold starts — which is exactly why a 180 s idle timeout and a phase-2 baked image are the levers worth pulling once you have real traffic.

---

## Failure modes

| # | Failure | Detection | Handling | Recycle? |
|---|---|---|---|---|
| 1 | Malformed input, unknown settings key, illegal flag letter | `schema.parse` | `bad_request` / `unknown_setting` / `invalid_setting` in <50 ms, 0 GPU seconds | No |
| 2 | **`import wgp` raises `KeyError: 'attention_mode'`** from a hand-written config | `wgp.py:3301` bare subscript; the config file *replaces* wgp's defaults (`wgp.py:2623`) | `ensure_wgp_config()` always writes `attention_mode`; `REQUIRED_WGP_KEYS` asserts it; `tests/test_wgp_config_drift.py` re-derives the unguarded-read set from `wgp.py` source on every CI run | n/a (boot) |
| 3 | Weights missing / wrong quantization on the volume | `get_missing_core_file_entries_for_status(deps, mt) != []` **before** any load | Fitness check fails → worker exits 1 → marked unhealthy → replaced. Never serves a request, never silently downloads 27 GB on the clock. | n/a |
| 4 | Volume-staged LoRAs invisible | `get_lora_dir` returns a **relative** path (`wgp.py:2498`, the `abspath` is commented out) | `loras_root` is set to an **absolute** path in `wgp_config.json` (verified: `get_lora_root` reads it at `wgp.py:2472`) | No |
| 5 | Arbitrary local-file read via `settings` | `settings.image_start = "/etc/hostname"` would pass straight through `_absolutize_setting_path` to WanGP | `FORBIDDEN_KEYS` rejects all 15 `ATTACHMENT_KEYS` plus `mode` in `input.settings`. Media may come **only** from `input.media`. `mode` matters because `_is_edit_task_params` (`wgp.py:1871`) flips `validate_task` into the edit branch, which reads `video_source` directly. | No |
| 6 | Arbitrary weight download via `activated_loras` | `get_lora_local_path` returns the value verbatim when `os.path.isabs`, and maps any https URL to `lora_dir/basename` | Absolute paths rejected; basenames allow-listed against `WANGP_ALLOWED_LORAS` | No |
| 7 | `video_length: 100000` burns a GPU for the full execution timeout | `frames_maximum` exists **only for Ref2VA** (737); FL2VA has no cap anywhere in the headless path | Worker-imposed `WANGP_MAX_FRAMES` (default 362 = 15.1 s) with a hard `invalid_setting` above it | No |
| 8 | `video_length` / overlap off the quantum | Model-def driven | `video_length` floored (matching `floor_frame_count`, `wgp.py:6929`); `sliding_window_overlap` **rounded to nearest** (matching `normalize_overlap`, `frame_scheduler.py:41-49` — it rounds, it does not floor: 30 → 35) | No |
| 9 | Cross-field violations (`frames_positions` count, `audio_prompt_type "2"` without `GV`, Ref2VA ref-count rules) | Replicated pre-flight from `validate_generative_settings` | `invalid_setting` before the model loads. Duration rules stay with WanGP and surface as `wangp_validation`. | No |
| 10 | `success=True` with **zero output files** | `result.generated_files` empty | `no_output`, `retryable: false`, **no recycle**. This is a *configuration* refusal, not a poisoned process: `wgp.py:6815-6818` and `:6820-6823` do `send_cmd("info", …); send_cmd("exit"); return True`, and `exit` is unhandled by `_handle_command` (`shared/api_cli.py:194-226`), so the task counts as successful. The `info` text is in `logs_tail`. | **No** |
| 11 | `job.events` grows without bound | Every stdout/stderr line becomes a `stream` event (`shared/api.py:800-803`); `_OutputCapture._drain` splits on `\r` **and** `\n`, so every tqdm refresh is one event | Dedicated drain loop calling `job.events.get(timeout=0.5)`, 400-line ring buffer, `job.release_output_payload()` after | No |
| 12 | **Blank container logs for the whole generation** | `shared/api_cli.py:38,44` pass `console=sys.__stdout__ if session._console_output else None`, and `:48` installs a process-global `redirect_stdout` | `console_output=True` (default `WANGP_CONSOLE=1`), **and** `obs.py` writes to `sys.__stdout__` captured at import — otherwise even the worker's own logs are swallowed into the event queue for 5–25 minutes | No |
| 13 | Wall-clock budget never fires during a silent stretch | `SessionStream.iter` (`shared/api.py:263-271`) `continue`s on timeout without yielding, so a deadline check in the loop body never runs while the job is quiet | Drive with `job.events.get(timeout=0.5)` in a `while` loop; check the clock every iteration regardless of whether an event arrived | No |
| 14 | Generation exceeds budget | `TimeoutError` from `job.result` / deadline in the drain loop | `job.cancel()` → wait `cancel_grace_s` → `timeout`, `retryable: true` | After 3 consecutive |
| 15 | **Cancel does not land within the grace window** | Second timeout | `backend_fatal` + `refresh_worker: True`. The daemon thread cannot be killed and still holds `_GENERATION_LOCK`; the process is permanently unusable. | **Yes** |
| 16 | CUDA OOM / illegal access / NCCL | `POISON_MARKERS` scan over error messages | `generation_failed` + `refresh_worker: True` | **Yes** |
| 17 | `rp_upload` silently returns a local path | `not out.startswith("http")` | Falls through the chain in `auto`, hard `upload_failed` otherwise. **The single most likely silent-data-loss bug in this design** (`rp_upload.py:300-301`). | No |
| 18 | Output too large with no transport configured | size check | `output_too_large` with the exact env vars to set. Never a truncated payload. | No |
| 19 | `gen["file_list"]` etc. grow forever | Upstream never truncates them; `_collect_outputs` slices from a per-job baseline | Cleared in `_reset_between_jobs`, along with the `api_output_artifacts` dict (which is `setdefault`ed, never cleared) | No |
| 20 | VRAM creep across jobs | `torch.cuda.max_memory_allocated()` is a **lifetime** high-water mark and cannot detect a leak on its own | `reset_peak_memory_stats()` each job; report `memory_allocated()` *after* `empty_cache()` — the steady-state floor is the leak signal | Threshold |
| 21 | `/cancel` from the API is never observed | The SDK awaits the handler inline on the loop (`rp_job.py:257`), starving `monitor_stop_signals` | `async def handler` + `asyncio.to_thread` | No |
| 22 | Two jobs in one process | `engine._JOB_LOCK` plus `_submit_tasks` raising `RuntimeError` (`shared/api.py:648`) | `worker_busy`, `retryable: true`. Should be unreachable at concurrency 1. | No |
| 23 | Duplicate delivery (`/retry`, client retry) | Derived object key + HEAD probe | Early return, 0 GPU seconds. Seed is pre-resolved so a genuine re-run is reproducible. | No |
| 24 | Read-only rootfs | `import wgp` does `os.mkdir("settings")` (`wgp.py:2549`) and writes `wgp_config.json`; `get_default_settings` writes `settings/<mt>_settings.json` (`wgp.py:3174`); `loras_url_cache_v2.json` is a bare relative path | Repo root is chowned writable in the image; documented as a hard requirement | n/a |
| 25 | Model switch between jobs | `model_type != wgp.transformer_type` → `release_model()` + full reload (`wgp.py:6773`) | Rejected unless `ALLOW_MODEL_SWITCH=1`; one model per endpoint is the documented shape | No |
| 26 | Endpoint silently scaled to 0 | 3 idle days → max 2; 7 idle days → 0 | Weekly synthetic job + alert; documented in the runbook | n/a |

---

## Testing

### Tier 1 — CPU only, no GPU, no torch, no weights (the majority of the code)

`schema.py`, `media_in.py`, `media_out.py`, `errors.py`, `config.py` import nothing from WanGP or torch. That is the whole reason for the split, and it is what makes CI runnable on a plain GitHub runner.

```bash
pip install pytest runpod requests boto3 moto
pytest runpod_worker/tests -v
```

Tests that must exist:

- **`test_wgp_config_drift`** — text-scan `wgp.py` with `re.finditer(r'server_config\["([^"]+)"\](?!\s*=)')`, filter to lines above `def create_ui`, subtract keys that are `setdefault`ed / `if not "x" in server_config` guarded / assigned earlier, and assert the remainder ⊆ our config's keys. **This is the regression test for the one blocker that stops the worker from booting at all.**
- **`test_attachment_keys_match`** — parse the `ATTACHMENT_KEYS = [...]` literal out of `wgp.py` with `ast.literal_eval` (no import, no torch) and compare to `schema.ATTACHMENT_KEYS`.
- **`test_frame_math`** — `floor_frames(124,107,17,5)==124`, `(130,…)==124`, `(50,…)==107`, `(209,…)==209`; `round_overlap(18,17,1)==18`, `(20,…)==18`, **`(30,…)==35`**, `(27,…)==35`. The 27/30 cases are the ones that catch a floor-instead-of-round implementation.
- **`test_forbidden_keys`** — `settings: {"image_start": "/etc/hostname"}` and `{"mode": "edit_postprocessing"}` both raise `bad_request`.
- **`test_lora_guards`** — absolute path rejected; unlisted basename rejected.
- **`test_cross_variant`** — FL2VA rejects `audio_prompt_type: "AB"` and `video_guide2`; Ref2VA rejects 10 `image_refs` and `#audio > #visual`; `"2"` without `GV` rejected; `"F"` with mismatched `frames_positions` rejected.
- **`test_video_length_cap`** — 100000 raises; 362 passes; 363 floors to 362.
- **`test_media_magic_bytes`** — a PNG named `.wav` is rejected for `audio_guide`; extension is taken from content, never from a caller-supplied name. Include real MP3/AAC sync-word variants (`0xFFFA`, `0xFFF3`, `0xFFF9`) — a naive two-byte table rejects ordinary files.
- **`test_rp_upload_local_fallback_is_caught`** — monkeypatch `upload_file_to_bucket` to return `"local_upload/out.mp4"` with all three `BUCKET_*` set, assert the chain falls through to base64 in `auto` mode and raises `upload_failed` in `s3` mode. **Highest-value single test in the suite.**
- **`test_base64_boundary`** — exactly at and one byte over `b64_out_max`.

### Tier 2 — GPU, no RunPod

```bash
# Local one-shot. NOTE: the SDK reads test_input.json from the PROCESS CWD
# (runpod/serverless/modules/rp_local.py:26-33, hardcoded, sys.exit(1) if absent).
cd /opt/wangp/runpod_worker && python3 handler.py

# Or explicitly, from anywhere:
python3 /opt/wangp/runpod_worker/handler.py \
  --test_input "$(cat /opt/wangp/runpod_worker/test_input.json)"

# Local FastAPI. /run does NOT execute the handler locally
# (rp_fastapi.py returns a fake id); only /runsync does.
python3 /opt/wangp/runpod_worker/handler.py --rp_serve_api --rp_api_host 0.0.0.0
curl -X POST localhost:8000/runsync -H 'Content-Type: application/json' \
     -d @runpod_worker/test_input.json | jq '.output.metrics'
```

Assert: `output.video.has_audio == true`, `audio_sample_rate == 32000` (`AUDIO_SAMPLE_RATE = 32000` in `models/minimax_h3/pipeline.py`), `video_codec == "h264"`, `fps == 24`, and the decoded file plays.

### Tier 3 — container, before pushing

```bash
docker build --platform linux/amd64 \
  --build-arg CUDA_ARCHITECTURES="8.0;8.6;8.9;9.0" \
  -f runpod_worker/Dockerfile -t you/wangp-h3:2026.08.18-1 .

# weight gate -- this is the step that would have caught PR #317's
# avatar-vs-non-avatar filename mismatch
docker run --rm --gpus all -v /path/to/ckpts:/runpod-volume/ckpts \
  you/wangp-h3:2026.08.18-1 \
  python3 -u -m runpod_worker.scripts.verify_weights minimax_h3_fl2va_pruned

# one real generation on the same image
docker run --rm --gpus all --env-file .env.staging \
  -v /path/to/ckpts:/runpod-volume/ckpts \
  -v $PWD/runpod_worker/test_input.json:/opt/wangp/runpod_worker/test_input.json \
  -w /opt/wangp/runpod_worker you/wangp-h3:2026.08.18-1 python3 -u handler.py
```

### Tier 4 — on RunPod

1. Staging endpoint: `max_workers=1`, idle 180 s, execution timeout 3600 s, volume attached, GPU priority `[L40S, A6000, A100-80]`.
2. `POST /run` with `test_input.json`; poll `/status/{id}`; record `delayTime` and `executionTime`. These are the real numbers that replace every estimate in the cold-start section.
3. Immediately fire an identical second request. `executionTime` should drop by the weight-load time. If it does not, the model is reloading — check that `model_type`, `profile` and `config` are identical between jobs (`wgp.py:6773`).
4. `scripts/calibrate.py --endpoint $ID --matrix steps=4,8,20 frames=124,209,362 --repeat 3` → p50/p90 per cell, cold-start distribution, $/generation, and a recommended `WANGP_DEFAULT_BUDGET_S` = measured p99 × 1.3.
5. Chaos: `timeout_s: 60` on a 20-step job (forces the cancel path — confirm the response is `timeout`, not a platform kill); a `.png` in `audio_guide`; `video_length: 700`; `model_type: "t2v"`; `audio_prompt_type: "2"` with no `video_guide`; unset `BUCKET_*` with `REQUIRE_BUCKET=1` (confirm the fitness check kills the worker rather than returning a dead path).
6. Watch the console Metrics tab: delay-time P70/P90/P98, cold-start count, throttled workers. If P90 delay exceeds ~400 s, the volume read is the bottleneck → active workers or the phase-2 baked image.

---

## Build order

Effort estimates are for one engineer already familiar with the repo.

| # | Step | Effort |
|---|---|---|
| 1 | Create `runpod_worker/` (**not** `runpod/`), `requirements-worker.txt`, `constraints.txt`, `errors.py`, `obs.py` (log to `sys.__stdout__`), `config.py` with `ensure_wgp_config()`. | 0.5 d |
| 2 | Write `tests/test_wgp_config_drift.py` **first** and make it green. It is the guard against the boot-time `KeyError`. | 0.25 d |
| 3 | Write `schema.py` + `tests/test_schema.py`. All four model types, all cross-field rules, the frame/overlap math, the forbidden-key and LoRA guards. Green on a laptop, no GPU. | 1.5 d |
| 4 | Write `media_in.py` + `media_out.py` + `tests/test_media.py` (moto/MinIO). Make the `rp_upload` local-path test fail first, then fix. | 1 d |
| 5 | Write `engine.py` + `handler.py`. `async def handler` + `to_thread`; the `get(timeout=)` drain loop; conditional `refresh_worker`; `_reset_between_jobs`. | 1 d |
| 6 | Write `Dockerfile`, `.dockerignore`, `test_input.json`, `scripts/patch_sage_setup.py`. Build. Expect 40–90 min on the first build. Verify `docker run … ls /opt/wangp/wgp.py` and that `pip check` passed. | 0.5 d |
| 7 | Create the 200 GB network volume in a datacenter with 48 GB GPU availability. Record ID + DC. | 0.25 d |
| 8 | Temp GPU **Pod** with the volume (remember: `/workspace` on Pods, `/runpod-volume` on Serverless — set `WANGP_VOLUME_ROOT=/workspace`). Run `prefetch_weights.py`, stage the FL2VA turbo LoRA by basename, run `verify_weights.py` until it exits 0. `du -sh` ≈ 55–60 GB. Terminate the Pod. | 0.5 d (+ download wall time) |
| 9 | Tier-3 container tests on a GPU: weight gate, then one real generation. Record `boot_ms` and `generate_s`; replace the estimates in this document with measurements. | 0.5 d |
| 10 | Push the tag (never `:latest`). Create the staging endpoint: import from Docker Registry, attach the volume, GPU priority `[L40S, A6000, A100-80]`, max workers 1, active 0, idle 180 s, execution timeout 3600 s, FlashBoot on. Env: `WANGP_MODEL_TYPE`, `WANGP_ALLOWED_LORAS`, `WANGP_CLI_ARGS`, and either `BUCKET_*` or nothing. | 0.5 d |
| 11 | Tier-4 acceptance + `calibrate.py`. Set `WANGP_DEFAULT_BUDGET_S` and the endpoint timeouts from measured p99. Log `psutil.virtual_memory().total` and confirm profile 4 is viable on the chosen tier. | 1 d |
| 12 | Chaos pass. Confirm the cancel path, the recycle path, the failure-budget breaker, and the `no_output` non-recycle path all behave. | 0.5 d |
| 13 | Try `WANGP_CLI_ARGS="--attention sage2 --profile 4"` and re-measure. Keep only if measurably faster and it does not produce a file-less "success". | 0.25 d |
| 14 | Load test at `max_workers=3` (volume already warm). Watch queue depth, delay p90, and `memory_allocated()` after `empty_cache()` across `jobs_served`. | 0.5 d |
| 15 | `runpod_worker/README.md`: `/run` + `/status` in the first paragraph, env table, endpoint settings, the volume warm procedure, the 3-day/7-day scale-down, rollback-by-tag, and the **WanGP disclosure required by `docs/API.md:9`**. | 0.5 d |
| 16 | Production endpoint + alerts: FAILED rate >2 %, cold-start count spike, delay p90 >60 s, sustained throttled workers, weekly synthetic job failure. | 0.5 d |
| 17 | Open the upstream PR: one new directory, one optional README line, zero shared-file edits, built on `shared/api.py` rather than bypassing mmgp. Reference PR #317 and deepbeepmeep's June 2025 comment. | 0.25 d |

**≈10–11 engineer-days**, plus download and build wall time. Steps 1–5 need no GPU at all.

---

## Open questions for the repo owner

**Blocking, decide before step 6:**

1. **Which `model_type` is the product?** This plan assumes `minimax_h3_fl2va_pruned` (cheapest, and it covers text-only, start frame, end frame and control video). If Ref2VA character consistency is the actual product, note it costs the same footprint but **has no shipped accelerator LoRA** — it runs at 20 steps, roughly 3.5× the cost per clip.
2. **Turbo LoRA by default?** 4 steps vs 20 is ~3.5× cheaper. `README.md:141` warns the 0.5 multiplier default exists because 1.0 can be too strong, and suggests the 8-step profile if quality suffers — though the profile this plan defaults to (`Turbo Lightx2v FL2V 4 Steps v1.0 768p`) ships `1.0` and `flow_shift: 6`. **A quality call I cannot make.**
3. **Text encoder.** INT8 (26.72 GB, default) vs GGUF Q4_K_M (14.58 GB) vs Q2_K (8.49 GB). Only matters for phase 2 (baked image), where Q4_K_M is what makes the image fit. The handler warns more aggressive quantization "can slightly affect prompt interpretation."
4. **Object storage.** Own bucket + `rp_upload` 7-day presigned URLs, or caller-supplied presigned PUT (no secrets on the worker, better for multi-tenant)? The chain supports both — which do you document first?
5. **URL media inputs.** `ALLOW_URL_INPUTS=0` by default. Turning it on means owning an SSRF guard forever (per-hop redirect revalidation, IP pinning against DNS rebinding, metadata-endpoint blocking, streaming byte cap, magic-byte typing). Multi-tenant: leave it off.
6. **Datacenter pinning.** One volume (cheap, smaller GPU pool) vs 2–3 volumes in different DCs (3× storage, manual sync, better availability)?

**Deferrable:**

7. **Phase 2 baked image?** ~70–90 GB, removes the 150–250 s billed volume read and the DC pinning, at the cost of an 80 GB-cap risk and painful iteration. Revisit after step 11 with real numbers.
8. **Should `refresh_worker` fire on plain OOM, or only after the failure budget?** Currently: on any `POISON_MARKERS` hit and on a failed cancel. Unconditional recycling forfeits every warm start — with a 150–250 s weight load that is brutal.
9. **Upstream directory or separate repo?** In-repo `runpod_worker/` is upstreamable (the maintainer asked for it) but couples worker releases to WanGP releases.
10. **Two small upstream additions worth proposing in the same PR:** (a) a `_api` flag to suppress preview decoding — `_build_preview_update` (`shared/api.py:784-798`) VAE-decodes a latent to a PIL image on the driver thread on every preview refresh, and nothing consumes previews in a headless worker; (b) a public `session.preload(model_type, config)` so warming the model does not require reaching into `wgp.load_models`.

---

## Appendix: verified facts vs unverified assumptions

### Verified in the working tree at `6e35b37`

**`shared/api.py`**
- `init(*, root, config_path, output_dir, callbacks, cli_args, console_output, console_isatty, webui_state) -> WanGPSession`, all keyword-only, at `:1265-1287`.
- `config_path` **must** be named `wgp_config.json` or `_ensure_runtime` raises `ValueError` (`:1071-1072`); a non-default location is translated into `--config <parent dir>` (`:1073-1076`).
- `_ensure_runtime` (`:1061-1097`): singleton under `_RUNTIME_LOCK`; raises if re-entered with different root/config/cli_args; `sys.path.insert(0, root)`; `with _pushd(root), _temporary_argv(["wgp.py", *cli_args]): importlib.import_module("wgp")`; module-identity check; `module.download_ffmpeg()`.
- `_GENERATION_LOCK` is a module-level `RLock` at `:27`, held for the whole job at `shared/api_cli.py:29`.
- `submit_task(settings, callbacks=None) -> SessionJob` (`:562-565`), non-blocking; `_submit_tasks` raises `RuntimeError("WanGP session already has a generation in progress")` at `:648`.
- `SessionJob`: `.events`, `.result(timeout)` (raises `TimeoutError`, `:392-403`), `.join`, `.done`, `.cancel()`, `.cancel_requested`, `.release_output_payload()` (`:375-377`).
- `SessionStream.get(timeout)` returns `None` on timeout **and** on close (`:254-261`); `.iter(timeout)` `continue`s on `None` without yielding (`:263-271`) — the reason the drain loop uses `get`, not `iter`.
- Event kinds: `started`, `stream`, `progress`, `preview`, `status`, `info`, `output`, `refresh_models`, `error`, `completed` (`api_cli.py:32,92,106,108,196,202,207,212,216,220,224`). `stream` carries `StreamMessage(stream, text)` (`api.py:800-802`).
- `ProgressUpdate(phase, status, progress, current_step, total_steps, raw_phase, unit)` at `:63-72`.
- `GenerationResult(success, generated_files, errors, total_tasks, successful_tasks, failed_tasks, artifacts)` + `.cancelled` at `:122-133`; `GenerationError(message, task_index, task_id, stage)` at `:137`.
- `_collect_outputs` slices `gen["file_list"]`/`["audio_file_list"]` from a per-job baseline and returns resolved absolute paths (`:862-866`).
- `_absolutize_task_paths` walks the live module's `ATTACHMENT_KEYS` (`:1010-1026`) and resolves relatives against `Path.cwd()` at submit time (`:1001`); virtual-media suffixes survive (`:1028-1043`).
- `get_model_def` / `get_model_schema` / `get_default_settings` / `get_model_availability` / `list_model_metadata` at `:490`, `:543`, `:511`, `:520`, `:479`; all wrap work in `_pushd(runtime.root)`.
- `_configure_runtime` forces `notification_sound_enabled = 0` and rewrites `save_path`/`image_save_path`/`audio_save_path` from `output_dir` (`:816-831`).
- `_api` output options are **opt-in** via `get_api_output_options` (`:154-158`); `_coerce_api_video_tensor_uint8` returns a CPU `torch.uint8` tensor (`:161-178`).
- `docs/API.md:9` carries the WanGP disclosure requirement (mirrored at `LICENSE.txt:316`).

**`shared/api_cli.py`**
- `run_cli_job` holds `_GENERATION_LOCK` + `_pushd(root)` for the whole job (`:29`); `console=sys.__stdout__ if session._console_output else None` (`:38,44`); process-global `redirect_stdout/stderr` at `:48`.
- Cancellation is cooperative: the driver polls `job.cancel_requested` (`:64-65`) → `_request_cancel_unlocked` sets `gen["abort"]` and `wan_model._interrupt` (`api.py:895-900`).
- `validate_task` runs **inside** the worker thread (`:141`); failures become `stage="validation"`.
- `_handle_command` (`:194-226`) handles `progress/preview/status/info/output/refresh_models/error` and **nothing else** — `send_cmd("exit")` is silently dropped.
- `generate_media` is called with `**filtered_params` where the filter is `inspect.signature(wgp.generate_media).parameters` (`:124`, `:157`).

**`wgp.py`**
- `ATTACHMENT_KEYS` = 15 keys at `:167-168`.
- Config: absent → wgp writes its full default dict (`:2575-2618`); present → `server_config = json.loads(text)` replaces the defaults (`:2623`) with only two `setdefault`s (`:2625`, `:2631`). `attention_mode = server_config["attention_mode"]` at **`:3301`** is the only read that can actually `KeyError`; `video/image/audio_profile` (`:3310-3312`) are `setdefault`ed first by `_normalize_profile_defaults`, *called* at module scope (`:2678`). A text scan cannot see through that call, so the drift test reports all four — `REQUIRED_WGP_KEYS` therefore lists all four rather than carrying an exception list.
- `--attention` whitelist `["auto","sdpa","sage","sage2","flash","xformers"]`, raises otherwise (`:3303-3308`).
- `torch.cuda.get_device_capability` at module scope (`:2508`) and `shared/attention.py:14` → **`import wgp` requires a GPU**.
- cwd must be the repo root: `open("models/_settings.json")` (`:2530`), `os.mkdir("settings")` (`:2549`).
- `get_lora_root()` (`:2469-2477`) gives a CLI `--loras` value **precedence over** `server_config["loras_root"]` (`lora_root = cli_lora_root or config_lora_root or DEFAULT_LORA_ROOT`). Putting `--loras` in `WANGP_CLI_ARGS` therefore silently defeats the absolute `loras_root` that volume-staged LoRAs depend on. The worker documents `--config` and `--loras` as forbidden in `WANGP_CLI_ARGS`.
- `get_lora_root()` reads `server_config.get("loras_root", DEFAULT_LORA_ROOT)` (`:2469-2477`); `get_lora_dir()` returns a **relative** path (the `abspath` is commented out at `:2498`); `get_lora_local_path` maps an https URL to `lora_dir/basename` and returns absolute paths verbatim (`:3670-3677`).
- Model reload gate: `if model_type != transformer_type or reload_needed or profile != loaded_profile or config != loaded_config` (`:6773`). `config` is normalized at `:6717` but `load_models` stores `loaded_config = config_id or ""` **unnormalized** at `:4082` — so any manual `load_models` call must pass an already-normalized string (the rstrip lives in `serialize_config_selection`, `shared/config_groups.py:18-20`, which `normalize_config_selection` (`:22-28`) returns).
- `floor_frame_count(video_length, …)` at `:6929`; `sliding_window_size` floored at `:6931`.
- `get_default_settings` **writes** `settings/<model_type>_settings.json` when absent (`:3155-3176`) and returns only ~20 keys.
- `clean_settings` merges over `primary_settings` = `models/_settings.json` = **112 keys** (`:1747-1760`, `:2530-2531`).
- `_is_edit_task_params(params)` = `params["mode"].startswith("edit_")` (`:1871-1872`); `validate_task` branches on it (`:8567`).
- Unsupported attention → `send_cmd("info", …); send_cmd("exit"); return True` (`:6815-6818`); sol on a model without `sol_attention` → same (`:6820-6823`).
- `download_models(model_filename=None, model_type=None, file_type=0, submodel_no=1, force_path=None, model_def=None)` (`:3576`); `file_type=0` also pulls `query_core_shared_model_files()` (`:3545-3557`) + MatAnyone (`:3585-3587`).
- `get_model_filename(model_type, quantization, dtype_policy, module_type, submodel_no, URLs, stack, model_def)` (`:2922`); `get_model_recursive_prop` (`:2891`); `get_model_config_groups` (`:2918`); `_get_dropdown_deps()` (`:13229`).
- Audio mux via `combine_and_concatenate_video_with_audio_tracks(..., audio_codec_key=server_config.get("audio_output_codec","aac_128"))` (`:8184-8195`); container from `server_config.get("video_container","mp4")` (`:8084`); `video_output_codec` default `"libx264_8"` (`:3331`).
- `__main__` guard at `:13623`; `startup.lock` + `select.select([sys.stdin],…)` at `:13787-13812`.

**`models/minimax_h3/minimax_h3_handler.py`**
- Four types at `:26-29`, family `"minimax_h3"`, family map collapses all onto `minimax_h3_fl2va`.
- `fps: 24`, `frames_minimum: 107`, `frames_steps: 17`, `frames_offset: 5`, `block_size: 32`, `vae_block_size: 32`, `guidance_max_phases: 0`, `lora_multiplier_phases: 1`, `no_negative_prompt: True`, `returns_audio: True`, `sol_attention: True`, `keep_frames_video_guide_not_supported: True`, `profiles_dir: ["minimax_h3"]` (`:185-220`).
- `first_block_cache_thresholds = (0.06, 0.08, 0.10, 0.12, 0.14)` (`:30`), enforced at `wgp.py:1215`.
- `sample_solvers = [("Euler","euler"), ("RES Multistep","res_multistep")]`.
- `sliding_window_defaults` — **identical for both variants**: `window_min 124, window_max 481, window_step 17, window_default 362, overlap_min 1, overlap_max 120, overlap_step 17, overlap_offset 1, overlap_default 18`.
- `frames_maximum: 737` — **Ref2VA only**. FL2VA has no upper bound.
- FL2VA letters: `image_prompt_types_allowed "TSEVL"`, `guide_custom_choices.letters_filter "GVKFI"`, `audio_prompt_type_sources.letters_filter "AK2"`, `mask_preprocessing {"", "A", "NA"}`, `one_image_ref_only: True`, `no_background_removal: True`, `output_audio_is_input_audio: True`, `end_frames_always_enabled: True`.
- Ref2VA letters: `image_ref_choices.letters_filter "KI"`, `guide_custom_choices.letters_filter "PDEV+-"`, `audio_prompt_type_sources.letters_filter "ABK"`, `image_refs_relative_size {min 50, max 400}`, `reference_video_max_frames 15*24`, `reference_video_max_size (768,1344)`.
- `update_default_settings` writes exactly: `video_length 124, sliding_window_size 362, sliding_window_overlap 18, num_inference_steps 20, guidance_scale 1.0, flow_shift 12.0, sample_solver "euler", skip_steps_start_step_perc 25, skip_steps_multiplier 0.08, denoising_strength 1.0, audio_prompt_type "", video_prompt_type "", image_mode 0` (+ `image_refs_relative_size 100, remove_background_images_ref 0` for Ref2VA) at `:511-533`.
- `validate_generative_settings` (`:345-445`): overlap via `normalize_overlap(v, 17, 1)`; FL2VA `frames_positions` count == `image_refs` count; `"2"` excludes `A`/`K` and requires `G`+`V`+`video_guide`; `"K"` requires the same plus a real audio track; Ref2VA ≤9 images, ≤2 videos (each ≥2 s, ≤15 s total), ≤2 audio (each 2–15 s, ≤15 s total), `#audio ≤ #images+#videos`, ≤12 files.
- Text-encoder variants: `bf16`, `int8`, `nvfp4_awq`, `gguf_q4_k_m`, `gguf_q2_k` (`:16-25`, `:226-233`); Video VAE `""`/`fp8mix`; DiT priority `""`/`lower_ram`.
- `query_model_files` returns the VAEs + `config.json, tokenizer.json, tokenizer_config.json, preprocessor_config.json, vocab.json` under `Qwen3-VL-32B-Instruct` (`:447-466`).

**`shared/utils/frame_scheduler.py`** — `floor_frame_count` (`:22-29`) **floors**; `normalize_overlap` (`:41-49`) **rounds to nearest** via `+ step // 2`.

**`shared/utils/utils.py:36-49`** — video `.mp4 .mkv .avi .mov`; image `.png .jpg .jpeg .bmp .gif .webp .tif .tiff .jfif .pjpeg`; audio `.wav .mp3 .aac`. **No `.webm`.**

**`shared/utils/files_locator.py:7`** — `default_checkpoints_paths = ["ckpts", "."]`; `_absolute_normalized_path` uses `os.path.abspath` (`:16-17`).

**`defaults/minimax_h3_*.json`** — each contains **only** `model.{name, architecture, description, URLs}` and a `prompt`. All settings come from the handler. Each `URLs` list has exactly two entries (`_bf16` and `_int8_convrot`); `get_model_filename` picks by `transformer_quantization` (default `"int8"`).

**`profiles/minimax_h3/`** — exactly six files, **all FL2V/generic**, read verbatim. No Ref2VA accelerator exists in this repo.

**`Dockerfile` / `entrypoint.sh`** — base `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`; `ARG CUDA_ARCHITECTURES="8.0;8.6"`; `torch==2.10.0+cu128`; SageAttention compiled with a patched `setup.py`; **no `COPY . /workspace`**; `ENTRYPOINT ["/workspace/entrypoint.sh"]` → ~95 lines of `nvidia-smi` diagnostics → `exec su -p user -c "python3 wgp.py --listen $*"`.

**`requirements.txt`** — `mmgp==3.7.12`, `gradio==5.29.0`, `pydantic==2.10.6`, `mcp==1.10.1`, `insightface==0.7.3 ; sys_platform=="linux"` (sdist, needs a C++ toolchain), `pycocotools` (same).

**`runpod==1.12.0`** (wheel downloaded and unpacked)
- `run_job` pops `error` and `refresh_worker` from a dict return, sets `stopPod` on the latter, and drops an empty `output` (`rp_job.py:266-281`).
- `run_local` reads `test_input.json` from the **process CWD**, `sys.exit(1)` if absent (`rp_local.py:26-33`).
- `upload_file_to_bucket(file_name, file_location, bucket_creds, bucket_name, prefix, extra_args)` returns `_save_to_local_fallback(...)` — a `local_upload/<name>` path, **no exception** — when `boto_client is None` (`rp_upload.py:282-301`, fallback at `:44-61`); presigned `ExpiresIn=604800` (`:321`).
- `register_fitness_check` and `progress_update` are real exports (`serverless/__init__.py:__all__`).
- `start()` → `_set_config_args()` → `parser.parse_known_args()` (`serverless/__init__.py:87-92`) — do not clobber `sys.argv` before it.
- The job loop runs `get_jobs` / `run_jobs` / `monitor_stop_signals` as tasks on one event loop (`rp_scale.py:142-144`); the heartbeat is a separate `multiprocessing.Process` (`rp_ping.py:84`).
- Requires Python ≥3.10; pulls `fastapi[all]>=0.141.1`, `boto3>=1.43.66`, `cryptography>=50.0.0`, `paramiko>=5.0.0`, `colorama<0.4.7`.

### UNVERIFIED — needs checking before you rely on it

1. **All weight file sizes** (21.06 GB, 26.72 GB, 5.21 GB, 0.61 GB, 14.58 GB, …) and the ~5 GB core-shared-assets figure. Sourced from the research dossier's Hugging Face tree query; I did not re-fetch them. The file *names* and the selection logic are verified.
2. **All generation wall-clock numbers.** Nothing in the repo or README states an H3 throughput figure. Every second-level estimate is extrapolation. `scripts/calibrate.py` replaces them.
3. **Whether RunPod bills container start / model load.** The docs contradict themselves (worker-state table vs pricing page). This plan assumes they are billed.
4. **Whether the 80 GB image cap applies to the Docker Hub / registry-import path**, or only to RunPod's GitHub builder where it is documented. Material to the phase-2 bake decision.
5. **Whether the 10 MB / 20 MB payload caps bound the request body, the response body, or both.** The docs say "maximum payload size" per operation without qualification. This plan treats 10 MB as the operative ceiling for output.
6. **`RUNPOD_INIT_TIMEOUT`.** `grep -r RUNPOD_INIT` over the 1.12.0 wheel returns nothing — it is not an SDK knob. It is documented as a platform setting for raising the 7-minute unhealthy threshold, but do not build a cold-start safety argument on a container `ENV` alone.
7. **Per-tier host RAM on RunPod serverless.** Not published. mmgp streams 48–60 GB; if a 48 GB GPU tier ships with <64 GB RAM, `--profile 4` must fall back to 5. Log `psutil.virtual_memory().total` from the first staging worker.
8. **Exact `gpuIds` tokens** for `.runpod/hub.json` (the official template shows `"ADA_24"`, so tokens exist; the set is not enumerated in anything I read). Read them off the console at endpoint-creation time.
9. **The complete set of platform-injected `RUNPOD_*` environment variables.** The list in circulation is reverse-engineered from SDK source. Log `os.environ` from one deployed worker if you need certainty.
10. **Whether `.runpod/hub.json` and `.runpod/tests.json` are read from a subdirectory.** RunPod Hub reads them from the **repository root**, which collides with the "zero new root files" constraint. This plan therefore does not depend on them; `test_input.json` plus the Tier-3 container gate cover the same ground.
11. **That the six shipped accelerator profiles produce acceptable quality at their stated multipliers.** `README.md:141` warns that 1.0 can be too strong; two of the six ship `0.5`, four ship `1.0`. A human has to look at the output.