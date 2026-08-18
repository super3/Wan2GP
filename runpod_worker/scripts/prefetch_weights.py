#!/usr/bin/env python3
"""Warm a RunPod network volume with everything one ``model_type`` needs.

Run this ONCE, on a temporary GPU Pod that has the volume attached, before the
image tag is ever pointed at a Serverless endpoint. See "Model weights strategy
-> Commands" in ``docs/RUNPOD_SERVERLESS.md``.

    # On a Pod the volume is at /workspace, on Serverless it is /runpod-volume.
    export WANGP_VOLUME_ROOT=/workspace
    python3 -m runpod_worker.scripts.prefetch_weights \
        --root /opt/wangp --config /opt/wangp/config/wgp_config.json \
        minimax_h3_fl2va_pruned --profile "Turbo Lightx2v FL2V 4 Steps v1.0 768p"

A GPU is mandatory even though nothing is generated: ``import wgp`` calls
``torch.cuda.get_device_capability`` at module scope (``wgp.py:2508``) and again
at ``shared/attention.py:14``.

WHY THIS IS NOT ``huggingface-cli download``
--------------------------------------------
Which files are needed is a *function of the runtime configuration*, not a fixed
list:

* ``get_model_filename`` (``wgp.py:2922-2984``) picks between the two entries of
  the model's ``URLs`` list by matching quantization tokens in the basename. Warm
  as ``bf16``, run as ``int8``, and every cold start re-downloads 21 GB — billed.
* the text encoder is chosen by ``text_encoder_quantization`` and, if the
  ``config`` string selects a ``system_configs`` entry, by that override
  (``minimax_h3_handler.py:226-233``).
* ``download_models(file_type=0)`` additionally pulls ``query_core_shared_model_files()``
  and MatAnyone (``wgp.py:3585-3587``) — DWPose, RAFT, Depth-Anything, wav2vec x2,
  BS-RoFormer, pyannote, det_align — plus the family handler's own
  ``query_model_files()`` manifest (the video/audio VAEs and the tokenizer JSON).
  That is several GB that no naive weight list mentions and that would otherwise
  download on the first paid request.

So this script drives WanGP's own downloader with the worker's own config, and
mirrors the file-list construction in ``load_models`` (``wgp.py:3958-4043``)
exactly. ``--dry-run`` reports what is missing without fetching anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Every heavyweight import (runpod_worker.config, engine, wgp, torch) happens
# INSIDE main(), after the CLI has written its overrides into os.environ:
# runpod_worker.config snapshots WANGP_ROOT / WANGP_CONFIG_DIR / WANGP_VOLUME_ROOT
# into module-level constants at import time, so importing it earlier would pin
# the wrong paths and --root would silently do nothing.

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(REPO_ROOT) not in sys.path:
    # Allow `python3 runpod_worker/scripts/prefetch_weights.py` as well as
    # `python3 -m runpod_worker.scripts.prefetch_weights`.
    sys.path.insert(0, str(REPO_ROOT))

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INTERRUPTED = 130


class PlanError(Exception):
    """This model's file list cannot be built (unknown type, malformed module).

    Deliberately not ``SystemExit``: wgp itself calls ``exit()`` on some paths
    (``wgp.py:4098``), and a per-model ``except SystemExit`` would swallow it.
    """


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prefetch_weights",
        description="Download every weight file a WanGP model_type needs onto this filesystem.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes: 0 = every expected file is present, 1 = something is still "
            "missing or a download failed.\n"
            "Run scripts/verify_weights.py afterwards; it is the gate the worker's "
            "fitness check uses."
        ),
    )
    parser.add_argument(
        "model_types",
        nargs="*",
        metavar="MODEL_TYPE",
        help="model type(s) to prefetch (default: --model-type, else $WANGP_MODEL_TYPE)",
    )
    parser.add_argument(
        "--model-type",
        action="append",
        default=[],
        dest="model_type_flags",
        help="model type to prefetch; repeatable, may be combined with positionals",
    )
    parser.add_argument("--root", help="WanGP repo root (default $WANGP_ROOT or /opt/wangp)")
    parser.add_argument(
        "--config",
        help="path to wgp_config.json, or the directory holding it "
        "(shared/api.py:1071-1072 requires that exact filename)",
    )
    parser.add_argument(
        "--volume-root",
        help="network volume mount point (default $WANGP_VOLUME_ROOT or /runpod-volume). "
        "On a Pod the volume is mounted at /workspace, not /runpod-volume.",
    )
    parser.add_argument(
        "--transformer-quant",
        help="transformer quantization to warm (default $WANGP_TRANSFORMER_QUANT or int8). "
        "MUST match what the workers run or every cold start re-downloads the transformer.",
    )
    parser.add_argument(
        "--text-encoder-quant",
        help="text-encoder quantization to warm (default $WANGP_TEXT_ENCODER_QUANT or int8)",
    )
    parser.add_argument(
        "--config-id",
        help="WanGP `config` selection string, e.g. 'gguf_q4_k_m' to warm the GGUF text "
        "encoder instead of the INT8 one (shared/config_groups.py:1-3)",
    )
    parser.add_argument(
        "--lora",
        action="append",
        default=[],
        metavar="URL_OR_NAME",
        help="stage one LoRA. An https URL is placed where get_lora_local_path "
        "(wgp.py:3670-3677) will look for it; a bare name is only checked for presence. "
        "Repeatable.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        metavar="NAME",
        help="stage the LoRAs of this accelerator profile "
        "(e.g. 'Turbo Lightx2v FL2V 4 Steps v1.0 768p'). Repeatable.",
    )
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="stage the LoRAs of every accelerator profile shipped for the model",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="print the accelerator profiles available for the model and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what is missing and download nothing",
    )
    parser.add_argument(
        "--no-du",
        action="store_true",
        help="skip the closing directory-size walk",
    )
    parser.add_argument("--json", dest="json_out", metavar="PATH", help="write the report as JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="only print the summary")
    return parser


def _apply_env(args: argparse.Namespace) -> None:
    """Translate CLI overrides into the env vars runpod_worker.config reads.

    Done before the first ``import runpod_worker.config`` on purpose — see the
    note at the top of the file.
    """
    if args.root:
        os.environ["WANGP_ROOT"] = str(Path(args.root).expanduser())
    if args.config:
        given = Path(args.config).expanduser()
        # Accept either the file or its directory; config.py owns the filename.
        directory = given.parent if given.suffix == ".json" else given
        os.environ["WANGP_CONFIG_DIR"] = str(directory)
    if args.volume_root:
        os.environ["WANGP_VOLUME_ROOT"] = str(Path(args.volume_root).expanduser())
    if args.transformer_quant:
        os.environ["WANGP_TRANSFORMER_QUANT"] = args.transformer_quant
    if args.text_encoder_quant:
        os.environ["WANGP_TEXT_ENCODER_QUANT"] = args.text_encoder_quant
    if args.config_id is not None:
        os.environ["WANGP_MODEL_CONFIG"] = args.config_id.rstrip(",")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def human_bytes(size: float | int | None) -> str:
    if size is None:
        return "?"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:,.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:,.1f} TB"


def dir_size(path: str | os.PathLike[str]) -> int:
    """Total bytes under ``path``. Follows no symlinks, never raises."""
    total = 0
    root = Path(path)
    if not root.is_dir():
        return 0
    for current, _dirs, files in os.walk(root, followlinks=False):
        for name in files:
            try:
                total += os.lstat(os.path.join(current, name)).st_size
            except OSError:
                continue
    return total


def entry_name(entry: Any) -> str:
    """``get_*_file_entries_for_status`` yields dicts, not strings."""
    if isinstance(entry, dict):
        return str(entry.get("filename") or entry.get("path") or entry)
    return str(entry)


def _basename(value: str) -> str:
    return os.path.basename(str(value).split("|", 1)[0])


# ---------------------------------------------------------------------------
# The plan: mirror wgp.load_models' file-list construction
# ---------------------------------------------------------------------------


def resolve_model_def(module: Any, model_type: str, config_id: str) -> dict[str, Any]:
    """``model_def`` with the ``config`` selection applied — ``wgp.py:3958-3963``.

    ``model_def.copy()`` is not defensive style, it is required: ``get_model_def``
    (``wgp.py:2799``) hands back the *live* entry of ``models_def`` and
    ``update()``ing it in place would permanently rewrite the process's idea of
    the model.
    """
    model_def = module.get_model_def(model_type)
    if model_def is None:
        raise PlanError(f"unknown model_type '{model_type}'")
    if config_id:
        config_groups = module.get_model_config_groups(model_type, model_def)
        model_def = model_def.copy()
        for _group, _cid, current_config in module.model_config_groups.selected_model_configs(
            config_groups, config_id
        ):
            model_def.update(current_config)
    return model_def


def build_plan(module: Any, model_type: str, config_id: str) -> dict[str, Any]:
    """Every file ``load_models`` would fetch, without loading anything.

    Mirrors ``wgp.py:3964-4043``: the main transformer, an optional ``URLs2``
    submodel, every declared module, then the text encoder.
    """
    torch = module.torch
    model_def = resolve_model_def(module, model_type, config_id)

    transformer_quantization = module.transformer_quantization
    dtype_policy = module.transformer_dtype_policy

    model_filename = module.get_model_filename(
        model_type=model_type,
        quantization=transformer_quantization,
        dtype_policy=dtype_policy,
        model_def=model_def,
    )
    model_filename2 = None
    if "URLs2" in model_def:
        model_filename2 = module.get_model_filename(
            model_type=model_type,
            quantization=transformer_quantization,
            dtype_policy=dtype_policy,
            submodel_no=2,
            model_def=model_def,
        )

    modules = module.get_model_recursive_prop(
        model_type, "modules", return_list=True, model_def=model_def
    )
    modules = [
        module.get_model_recursive_prop(item, "modules", sub_prop_name="_list", return_list=True)
        if isinstance(item, str)
        else item
        for item in modules
    ]

    # wgp.py:3979-3985 — the dtype the module files are matched against depends
    # on whether the main checkpoint is already quantized.
    quantize_transformer = (
        transformer_quantization in ("int8", "fp8")
        and bool(model_def.get("auto_quantize", False))
        and "quanto" not in model_filename
        and len(modules) == 0
    )
    transformer_dtype = module.get_transformer_dtype(model_type, dtype_policy)
    if quantize_transformer or "quanto" in model_filename:
        lowered = model_filename
        if "bf16" in lowered or "BF16" in lowered:
            transformer_dtype = torch.bfloat16
        if "fp16" in lowered or "FP16" in lowered:
            transformer_dtype = torch.float16

    # (filename, file_type, submodel_no, kind) -- file_type is download_models'
    # third positional: 0 main model (also pulls the shared assets), 1 module,
    # 2 text encoder.
    entries: list[tuple[str, int, int, str]] = [(model_filename, 0, 1, "transformer")]
    if model_filename2:
        entries.append((model_filename2, 0, 2, "transformer2"))
    for module_type in modules:
        if isinstance(module_type, dict):
            urls1 = module_type.get("URLs")
            urls2 = module_type.get("URLs2")
            if urls1 is None or urls2 is None:
                raise PlanError(f"module of '{model_type}' declares no URLs/URLs2: {module_type}")
            entries.append(
                (
                    module.get_model_filename(
                        model_type, transformer_quantization, transformer_dtype, URLs=urls1
                    ),
                    1,
                    1,
                    "module",
                )
            )
            entries.append(
                (
                    module.get_model_filename(
                        model_type, transformer_quantization, transformer_dtype, URLs=urls2
                    ),
                    1,
                    2,
                    "module",
                )
            )
        else:
            entries.append(
                (
                    module.get_model_filename(
                        model_type, transformer_quantization, transformer_dtype,
                        module_type=module_type,
                    ),
                    1,
                    0,
                    "module",
                )
            )

    text_encoder_filename = ""
    text_encoder_folder = model_def.get("text_encoder_folder")
    text_encoder_urls = module.get_model_recursive_prop(
        model_type, "text_encoder_URLs", return_list=True, model_def=model_def
    )
    # get_model_recursive_prop returns [] (not None) for an absent property,
    # wgp.py:2896 -- so test truthiness, not `is not None`.
    if text_encoder_urls:
        text_encoder_filename = module.get_model_filename(
            model_type=model_type,
            quantization=module.text_encoder_quantization,
            dtype_policy=dtype_policy,
            URLs=text_encoder_urls,
        )

    return {
        "model_type": model_type,
        "model_def": model_def,
        "config_id": config_id,
        "entries": [entry for entry in entries if entry[0]],
        "text_encoder_filename": text_encoder_filename,
        "text_encoder_folder": text_encoder_folder,
        "transformer_quantization": transformer_quantization,
        "text_encoder_quantization": module.text_encoder_quantization,
        "transformer_dtype_policy": str(dtype_policy),
    }


def shared_download_defs(module: Any, model_type: str, model_def: dict[str, Any]) -> list[dict]:
    """The repo-manifest downloads ``download_models(file_type=0)`` triggers.

    ``wgp.py:3585-3587`` (core shared assets + MatAnyone) and ``wgp.py:3648-3651``
    (the family handler's own ``query_model_files`` — for MiniMax H3 the video and
    audio VAEs plus the tokenizer JSON, ``minimax_h3_handler.py:448-466``).

    These are checked by neither ``get_missing_core_file_entries_for_status`` nor
    ``has_secondary_model_files_for_status``, so nothing else in this worker
    notices when they are absent. They are also several GB.
    """
    defs: list[dict[str, Any]] = [
        module.query_core_shared_model_files(),
        module.query_matanyone_download_def(module.server_config),
    ]

    def compute_list(filename: Any) -> list[str]:
        # download_models' own local helper, wgp.py:3577-3582.
        if filename is None:
            return []
        text = str(filename)
        return [text[text.rfind("/") + 1:]]

    base_model_type = module.get_base_model_type(model_type)
    handler = module.model_types_handlers[base_model_type]
    model_files = handler.query_model_files(compute_list, base_model_type, model_def)
    if not isinstance(model_files, list):
        model_files = [model_files]
    defs.extend(one for one in model_files if isinstance(one, dict))
    return defs


# ---------------------------------------------------------------------------
# LoRA staging
# ---------------------------------------------------------------------------


def profile_paths(model_def: dict[str, Any], root: Path) -> list[Path]:
    """Every accelerator-profile JSON shipped for this model.

    Mirrors ``_get_builtin_lset_groups`` (``wgp.py:8891-8907``): the roots come
    from ``model_def["_profile_roots"]`` (``wgp.py:3205``, default ``["profiles"]``)
    and the sub-directories from ``profiles_dir`` (``["minimax_h3"]`` for H3,
    ``minimax_h3_handler.py:220``).
    """
    roots = model_def.get("_profile_roots") or ["profiles"]
    if isinstance(roots, str):
        roots = [roots]
    dirs = model_def.get("profiles_dir") or []
    if isinstance(dirs, str):
        dirs = [dirs]
    found: list[Path] = []
    for profile_root in roots:
        base = Path(str(profile_root))
        if not base.is_absolute():
            base = root / base
        for folder in dirs:
            directory = base / str(folder)
            if directory.is_dir():
                found.extend(sorted(directory.glob("*.json")))
    # Same file reachable through two roots: keep the first.
    unique: dict[str, Path] = {}
    for path in found:
        unique.setdefault(path.name, path)
    return list(unique.values())


def loras_from_profile(path: Path) -> list[str]:
    try:
        fragment = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"accelerator profile {path} is unreadable: {exc}") from exc
    if not isinstance(fragment, dict):
        raise SystemExit(f"accelerator profile {path} is not a JSON object")
    entries = fragment.get("activated_loras") or []
    if isinstance(entries, str):
        entries = [entries]
    return [str(entry) for entry in entries if str(entry).strip()]


def stage_lora(module: Any, lora_dir: str, entry: str, *, dry_run: bool) -> dict[str, Any]:
    """Put one LoRA where ``get_lora_local_path`` will find it.

    ``wgp.py:3670-3677``, verified verbatim::

        def get_lora_local_path(lora_dir, lora):
            if os.path.isabs(lora): return lora
            if (lora.startswith("http:") or lora.startswith("https:")):
                parts = lora.split("|")
                lora_path = os.path.join(fl.clean_relative_path(parts[1]),
                                         os.path.basename(parts[0])) if len(parts) > 1 \
                            else os.path.basename(lora)
            else:
                lora_path = lora
            return lora_path if lora_dir is None else os.path.join(lora_dir, lora_path)

    So the plan's "maps an https entry to ``os.path.join(lora_dir, basename(url))``"
    is right for a plain URL but INCOMPLETE: WanGP also honours a ``url|subfolder``
    form that lands the file in ``lora_dir/<subfolder>/<basename>``. Staging that
    one by basename alone would leave WanGP to re-download it. We therefore ask
    WanGP itself where the file belongs instead of reimplementing the rule.
    """
    destination = module.get_lora_local_path(lora_dir, entry)
    record: dict[str, Any] = {"lora": entry, "path": destination}
    if os.path.isfile(destination):
        record.update(status="present", size_bytes=os.path.getsize(destination))
        return record
    if not (entry.startswith("http:") or entry.startswith("https:")):
        # A bare name or a relative path: there is no URL to fetch it from.
        # download_models raises the same way at wgp.py:3639-3640.
        record.update(status="missing", error="not a URL; copy the file onto the volume yourself")
        return record
    if dry_run:
        record.update(status="would_download")
        return record
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    started = time.monotonic()
    try:
        module.download_file(entry, destination)
    except Exception as exc:  # noqa: BLE001 - report, do not traceback
        if os.path.isfile(destination):
            os.remove(destination)
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        return record
    record.update(
        status="downloaded",
        size_bytes=os.path.getsize(destination) if os.path.isfile(destination) else 0,
        seconds=round(time.monotonic() - started, 1),
    )
    return record


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _apply_env(args)

    from runpod_worker import config as C  # noqa: PLC0415 - must follow _apply_env
    from runpod_worker import engine, schema
    from runpod_worker.errors import WorkerError

    model_types = list(dict.fromkeys([*args.model_types, *args.model_type_flags]))
    if not model_types:
        model_types = [C.CONFIG.model_type]
    config_id = args.config_id.rstrip(",") if args.config_id is not None else C.CONFIG.model_config

    def say(*parts: Any) -> None:
        if not args.quiet:
            print(*parts, flush=True)

    say("=" * 78)
    say(f"WanGP weight prefetch  ->  {', '.join(model_types)}")
    say("=" * 78)
    say(f"  repo root          : {C.WANGP_ROOT}")
    say(f"  config             : {C.CONFIG_DIR / 'wgp_config.json'}")
    say(f"  volume root        : {C.VOLUME_ROOT}"
        f"{'' if C.VOLUME_ROOT.is_dir() else '   (NOT MOUNTED)'}")
    say(f"  checkpoints_paths  : {C.checkpoint_paths()}")
    say(f"  loras_root         : {C.lora_root()}")
    say(f"  transformer quant  : {os.environ.get('WANGP_TRANSFORMER_QUANT', 'int8')}")
    say(f"  text encoder quant : {os.environ.get('WANGP_TEXT_ENCODER_QUANT', 'int8')}")
    say(f"  config selection   : {config_id or '(default)'}")
    say(f"  mode               : {'DRY RUN (no downloads)' if args.dry_run else 'download'}")
    say("")

    if not C.VOLUME_ROOT.is_dir():
        say(
            f"!! {C.VOLUME_ROOT} is not a directory, so nothing will land on the volume.\n"
            f"!! On a Pod the network volume mounts at /workspace, not /runpod-volume:\n"
            f"!!     export WANGP_VOLUME_ROOT=/workspace   (or pass --volume-root)\n"
        )

    started = time.monotonic()
    try:
        session = engine.boot()
    except WorkerError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        for detail in exc.details:
            print(f"        {detail}", file=sys.stderr)
        return EXIT_FAILED
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED to import WanGP: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED

    runtime = session._ensure_runtime()  # shared/api.py:1061-1097
    module = runtime.module
    from shared.api import _pushd  # shared/api.py:1301-1309
    from shared.utils import files_locator as fl
    from shared.utils.download import download_def_missing_files

    say(f"wgp imported in {time.monotonic() - started:.1f}s "
        f"(version {getattr(module, 'WanGP_version', '?')})")
    say("")

    report: dict[str, Any] = {
        "root": str(C.WANGP_ROOT),
        "checkpoints_paths": C.checkpoint_paths(),
        "loras_root": C.lora_root(),
        "config_id": config_id,
        "dry_run": bool(args.dry_run),
        "models": [],
    }
    failures: list[str] = []

    for model_type in model_types:
        say("-" * 78)
        say(f"model_type: {model_type}")
        say("-" * 78)
        model_report: dict[str, Any] = {"model_type": model_type, "files": [], "loras": []}
        report["models"].append(model_report)

        with _pushd(runtime.root):
            try:
                plan = build_plan(module, model_type, config_id)
            except PlanError as exc:
                failures.append(f"{model_type}: {exc}")
                say(f"  !! {exc}")
                continue

            model_def = plan["model_def"]
            model_report.update(
                {
                    "name": model_def.get("name", model_type),
                    "transformer_quantization": plan["transformer_quantization"],
                    "text_encoder_quantization": plan["text_encoder_quantization"],
                }
            )
            say(f"  name               : {model_def.get('name', model_type)}")

            if args.list_profiles:
                for path in profile_paths(model_def, Path(runtime.root)):
                    say(f"  profile            : {path.stem}")
                    for lora in loras_from_profile(path):
                        say(f"      lora           : {_basename(lora)}")
                continue

            # ---- the weight files -----------------------------------------
            for filename, file_type, submodel_no, kind in plan["entries"]:
                local = fl.get_local_model_filename(filename)
                record = {
                    "kind": kind,
                    "filename": _basename(filename),
                    "url": filename,
                    "file_type": file_type,
                    "submodel_no": submodel_no,
                    "local_path": local,
                }
                if local is not None:
                    record["status"] = "present"
                    record["size_bytes"] = os.path.getsize(local) if os.path.isfile(local) else None
                    say(f"  [have] {kind:<12} {_basename(filename)}")
                elif args.dry_run:
                    record["status"] = "would_download"
                    say(f"  [WANT] {kind:<12} {_basename(filename)}")
                else:
                    say(f"  [GET ] {kind:<12} {_basename(filename)}")
                    t0 = time.monotonic()
                    try:
                        # wgp.py:3576 download_models(model_filename, model_type,
                        # file_type, submodel_no, force_path=None, model_def=None)
                        module.download_models(
                            filename, model_type, file_type, submodel_no, model_def=model_def
                        )
                    except Exception as exc:  # noqa: BLE001
                        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                        failures.append(f"{model_type}: {_basename(filename)}: {exc}")
                        say(f"         !! {type(exc).__name__}: {exc}")
                    else:
                        local = fl.get_local_model_filename(filename)
                        record.update(
                            status="downloaded" if local else "failed",
                            local_path=local,
                            seconds=round(time.monotonic() - t0, 1),
                            size_bytes=os.path.getsize(local)
                            if local and os.path.isfile(local)
                            else None,
                        )
                        if not local:
                            failures.append(
                                f"{model_type}: {_basename(filename)} still not found after download"
                            )
                model_report["files"].append(record)

            # ---- the text encoder ------------------------------------------
            te_filename = plan["text_encoder_filename"]
            if te_filename:
                folder = plan["text_encoder_folder"]
                local = fl.get_local_model_filename(te_filename, extra_paths=folder)
                record = {
                    "kind": "text_encoder",
                    "filename": _basename(te_filename),
                    "url": te_filename,
                    "folder": folder,
                    "local_path": local,
                }
                if local is not None:
                    record["status"] = "present"
                    record["size_bytes"] = os.path.getsize(local) if os.path.isfile(local) else None
                    say(f"  [have] text_encoder {_basename(te_filename)}")
                elif args.dry_run:
                    record["status"] = "would_download"
                    say(f"  [WANT] text_encoder {_basename(te_filename)}")
                else:
                    say(f"  [GET ] text_encoder {_basename(te_filename)}")
                    t0 = time.monotonic()
                    try:
                        # Mirrors wgp.py:4043 -- file_type 2, submodel_no -1, and
                        # force_path so it lands in the model's text-encoder folder.
                        module.download_models(
                            te_filename, model_type, 2, -1,
                            force_path=folder, model_def=model_def,
                        )
                    except Exception as exc:  # noqa: BLE001
                        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                        failures.append(f"{model_type}: text encoder: {exc}")
                        say(f"         !! {type(exc).__name__}: {exc}")
                    else:
                        local = fl.get_local_model_filename(te_filename, extra_paths=folder)
                        record.update(
                            status="downloaded" if local else "failed",
                            local_path=local,
                            seconds=round(time.monotonic() - t0, 1),
                            size_bytes=os.path.getsize(local)
                            if local and os.path.isfile(local)
                            else None,
                        )
                        if not local:
                            failures.append(f"{model_type}: text encoder still not found")
                model_report["files"].append(record)

            # ---- the shared assets nothing else checks ----------------------
            #
            # These ride along with ANY download_models(file_type=0) call, so on a
            # cold volume the transformer fetch above already pulled them. But on a
            # volume where the transformer happens to be present and the shared
            # assets are not -- a resumed prefetch, a hand-copied checkpoint, a
            # depth_anything_v2_variant change -- no file_type=0 call happens at
            # all, and the first paid request would download several GB. So ask
            # for them explicitly, the same way wgp.py:4021-4022 does when a model
            # declares no URLs of its own.
            defs = shared_download_defs(module, model_type, model_def)
            missing_shared = download_def_missing_files(defs)
            if missing_shared and not args.dry_run:
                say(f"  [GET ] shared assets  {len(missing_shared)} file(s) "
                    f"(VAEs / tokenizer / DWPose / wav2vec / RAFT / ...)")
                t0 = time.monotonic()
                try:
                    module.download_models("", model_type, 0, -1, model_def=model_def)
                except Exception as exc:  # noqa: BLE001
                    say(f"         !! {type(exc).__name__}: {exc}")
                    failures.append(f"{model_type}: shared assets: {exc}")
                else:
                    say(f"         done in {time.monotonic() - t0:.1f}s")
                missing_shared = download_def_missing_files(defs)

            model_report["missing_shared"] = list(missing_shared)
            if missing_shared:
                if args.dry_run:
                    say(f"  [WANT] shared assets: {len(missing_shared)} file(s) "
                        f"(VAEs / tokenizer / DWPose / wav2vec / ...)")
                    for name in missing_shared[:12]:
                        say(f"         {name}")
                    if len(missing_shared) > 12:
                        say(f"         ... and {len(missing_shared) - 12} more")
                else:
                    say(f"  !! {len(missing_shared)} shared asset(s) still missing after download")
                    for name in missing_shared[:12]:
                        say(f"     {name}")
                    failures.append(
                        f"{model_type}: {len(missing_shared)} shared asset(s) missing "
                        f"(first: {missing_shared[0]})"
                    )
            else:
                say("  [have] shared assets (VAEs, tokenizer, DWPose, wav2vec, ...)")

            # ---- LoRAs ------------------------------------------------------
            wanted: list[str] = list(args.lora)
            profile_names = list(args.profile)
            available_profiles = profile_paths(model_def, Path(runtime.root))
            if args.all_profiles:
                for path in available_profiles:
                    wanted.extend(loras_from_profile(path))
            for name in profile_names:
                try:
                    fragment = schema.load_profile_fragment(
                        name, model_def=model_def, root=runtime.root
                    )
                except WorkerError as exc:
                    failures.append(f"{model_type}: profile '{name}': {exc.message}")
                    say(f"  !! profile '{name}': {exc.message}")
                    for detail in exc.details:
                        say(f"     {detail}")
                    continue
                entries = fragment.get("activated_loras") or []
                if isinstance(entries, str):
                    entries = [entries]
                wanted.extend(str(item) for item in entries if str(item).strip())

            wanted = list(dict.fromkeys(wanted))
            if wanted:
                lora_dir = module.get_lora_dir(model_type)  # wgp.py:2479, creates it
                model_report["lora_dir"] = lora_dir
                say(f"  lora dir           : {os.path.abspath(lora_dir)}")
                for entry in wanted:
                    record = stage_lora(module, lora_dir, entry, dry_run=args.dry_run)
                    model_report["loras"].append(record)
                    tag = {
                        "present": "[have]",
                        "downloaded": "[GET ]",
                        "would_download": "[WANT]",
                    }.get(record["status"], "[FAIL]")
                    say(f"  {tag} lora         {_basename(entry)}")
                    if record["status"] in ("failed", "missing"):
                        say(f"         !! {record.get('error')}")
                        failures.append(f"{model_type}: lora {_basename(entry)}: "
                                        f"{record.get('error')}")

            # ---- verdict ----------------------------------------------------
            deps = module._get_dropdown_deps()  # wgp.py:13229
            missing_core = [
                entry_name(entry)
                for entry in module.model_dropdowns.get_missing_core_file_entries_for_status(
                    deps, model_type
                )
            ]
            status = module.model_dropdowns.get_model_download_status(deps, model_type)
            expected = module.model_dropdowns.MODEL_FILE_STATUS_EXPECTED  # == 2

        model_report["missing_core"] = missing_core
        model_report["status_code"] = status
        # shared/api.py:1250-1251 turns EXPECTED into availability "available".
        model_report["available"] = status == expected
        say("")
        if missing_core:
            say(f"  MISSING CORE FILES ({len(missing_core)}):")
            for name in missing_core:
                say(f"    - {name}")
            if not args.dry_run:
                failures.append(f"{model_type}: {len(missing_core)} core file(s) still missing")
        else:
            say("  core weight set: COMPLETE")
        say(f"  download status : {status} "
            f"({'available' if status == expected else 'partial/missing'})")
        say("")

    if args.list_profiles:
        return EXIT_OK

    if not args.no_du:
        sizes = {}
        for path in [*C.checkpoint_paths(), C.lora_root()]:
            if path == ".":
                continue
            sizes[path] = dir_size(path)
        report["sizes"] = sizes
        say("disk usage")
        for path, size in sizes.items():
            say(f"  {human_bytes(size):>12}  {path}")
        say("")

    report["elapsed_s"] = round(time.monotonic() - started, 1)
    report["failures"] = failures

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        say(f"report written to {args.json_out}")

    if failures:
        print(f"\nPREFETCH INCOMPLETE ({len(failures)} problem(s)):", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        print(
            "\nRe-run to resume (every download is skipped when the file is already "
            "present), then gate the deploy with:\n"
            f"  python3 -m runpod_worker.scripts.verify_weights {' '.join(model_types)}",
            file=sys.stderr,
        )
        return EXIT_FAILED

    say(f"done in {report['elapsed_s']}s")
    if args.dry_run:
        say("dry run: nothing was downloaded")
    else:
        say("next: python3 -m runpod_worker.scripts.verify_weights " + " ".join(model_types))
    return EXIT_OK


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(EXIT_INTERRUPTED) from None
