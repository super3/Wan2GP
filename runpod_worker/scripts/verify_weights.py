#!/usr/bin/env python3
"""Pre-deploy weight gate. Exit 0 means this image + volume can serve requests.

    python3 -m runpod_worker.scripts.verify_weights minimax_h3_fl2va_pruned

Run it on the prefetch Pod after ``prefetch_weights.py``, and again inside the
built container before the tag is pointed at an endpoint::

    docker run --rm --gpus all -v /path/to/ckpts:/runpod-volume/ckpts \\
      you/wangp-h3:2026.08.18-1 \\
      python3 -u -m runpod_worker.scripts.verify_weights minimax_h3_fl2va_pruned

WHAT IT ASSERTS, AND WHY EACH ONE
---------------------------------
1. ``get_missing_core_file_entries_for_status(deps, model_type) == []``
   (``shared/model_dropdowns.py:342``) — via :func:`engine.assert_weights_complete`,
   which is *literally* the function the worker registers as a RunPod fitness
   check (``handler._fitness_weights``). Running the same call here means a green
   verify and a red worker cannot disagree.

   **This is the check that would have caught PR #317.** That PR's Dockerfile
   downloaded ``hunyuan_video_720_bf16.safetensors`` while its handler loaded
   ``hunyuan_video_avatar_720_bf16.safetensors``. The enumeration is derived from
   the *same* ``get_model_filename`` call the loader makes, with the *same*
   ``transformer_quantization`` — so a filename that is one token off is reported
   here, on a Pod, instead of at 3 a.m. on a billed cold start.

2. ``get_model_availability(model_type)["available"]``. Strictly stronger than
   (1): ``get_model_download_status`` (``shared/model_dropdowns.py:442``) returns
   EXPECTED only when ``has_secondary_model_files_for_status`` (``:391``) also
   finds the text encoder (``:394-401``), every ``preload_URL``, every ``VAE_URL``
   and every model-declared LoRA.

3. ``get_default_settings(model_type)`` once, so ``settings/<model_type>_settings.json``
   exists. ``wgp.py:3174-3175`` ``json.dump()``s that file on the FIRST call —
   letting that happen inside a request means a write to the repo root on the
   clock, and an outright failure on a read-only rootfs (failure mode 24).

4. The assets nothing else checks: the video/audio VAEs and tokenizer JSON from
   the family handler's ``query_model_files`` (``minimax_h3_handler.py:448-466``)
   and the ~5 GB of shared preprocessing weights ``download_models(file_type=0)``
   pulls unconditionally (``wgp.py:3585-3587``). Neither enumeration in (1) or (2)
   looks at them, so they are reported as a WARNING here (they download
   automatically, but they download *on the clock*). ``--strict`` makes them fatal.

It also prints the resolved transformer and text-encoder filenames with the paths
they resolved to, which is the eyeball check no assertion replaces.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify_weights",
        description="Assert that this worker can serve requests without downloading weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit 0: ready to deploy. Exit 1: do not point an endpoint at this image.",
    )
    parser.add_argument(
        "model_types",
        nargs="*",
        metavar="MODEL_TYPE",
        help="model type(s) to verify (default: $WANGP_MODEL_TYPE)",
    )
    parser.add_argument("--model-type", action="append", default=[], dest="model_type_flags")
    parser.add_argument("--root", help="WanGP repo root (default $WANGP_ROOT or /opt/wangp)")
    parser.add_argument("--config", help="path to wgp_config.json or the directory holding it")
    parser.add_argument("--volume-root", help="network volume mount (default $WANGP_VOLUME_ROOT)")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="do not fail when the core set is complete but a secondary file "
        "(VAE / preload / model LoRA) is absent",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail on missing shared assets (VAEs, tokenizer, DWPose, wav2vec, ...)",
    )
    parser.add_argument(
        "--skip-shared",
        action="store_true",
        help="do not enumerate the shared assets at all",
    )
    parser.add_argument(
        "--lora",
        action="append",
        default=[],
        metavar="URL_OR_NAME",
        help="also require this LoRA to be staged where WanGP will look for it. Repeatable.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        metavar="NAME",
        help="also require the LoRAs of this accelerator profile. Repeatable.",
    )
    parser.add_argument("--json", dest="json_out", metavar="PATH", help="write the report as JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="print only failures")
    return parser


def _apply_env(args: argparse.Namespace) -> None:
    """CLI -> env, before runpod_worker.config is imported and freezes its paths."""
    if args.root:
        os.environ["WANGP_ROOT"] = str(Path(args.root).expanduser())
    if args.config:
        given = Path(args.config).expanduser()
        os.environ["WANGP_CONFIG_DIR"] = str(given.parent if given.suffix == ".json" else given)
    if args.volume_root:
        os.environ["WANGP_VOLUME_ROOT"] = str(Path(args.volume_root).expanduser())


def human_bytes(size: Any) -> str:
    if size is None:
        return "?"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{int(value)} B" if unit == "B" else f"{value:,.2f} {unit}"
        value /= 1024.0
    return f"{value:,.2f} TB"


def _basename(value: Any) -> str:
    return os.path.basename(str(value).split("|", 1)[0])


def _file_size(path: Any) -> int | None:
    try:
        return os.path.getsize(str(path))
    except (OSError, TypeError):
        return None


def verify_one(
    *,
    module: Any,
    session: Any,
    engine: Any,
    model_type: str,
    args: argparse.Namespace,
    say,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Verify one ``model_type``. Returns ``(report, failures, warnings)``."""
    from shared.api import _pushd  # shared/api.py:1301-1309
    from shared.utils import files_locator as fl
    from shared.utils.download import download_def_missing_files
    from runpod_worker.errors import WorkerError

    failures: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {"model_type": model_type}

    say("=" * 78)
    say(f"verifying {model_type}")
    say("=" * 78)

    # -- 1 + 2: the exact gate the worker's fitness check runs ---------------
    try:
        weights = engine.weights_report(model_type)
    except WorkerError as exc:
        failures.append(f"{model_type}: {exc.message}")
        say(f"  FAIL  {exc.message}")
        return report, failures, warnings

    report.update(weights)
    model_def_public = session.get_model_def(model_type) or {}
    report["name"] = model_def_public.get("name", model_type)

    say(f"  model              : {report['name']}")
    say(f"  transformer quant  : {weights['transformer_quantization']}")
    say(f"  text encoder quant : {weights['text_encoder_quantization']}")
    say(f"  checkpoints_paths  : {weights['checkpoints_paths']}")
    say(f"  loras_root         : {weights['loras_root']}")
    say("")

    # -- resolved filenames, with the paths they resolved to ------------------
    with _pushd(module_root(module)):
        deps = module._get_dropdown_deps()  # wgp.py:13229
        entries = module.model_dropdowns.get_expected_core_file_entries_for_status(
            deps, model_type
        )

        # Label each entry by asking the same two questions the loader asks, so
        # "transformer" and "text_encoder" are the files load_models would open
        # rather than a guess based on position. shared/model_dropdowns.py:283-287
        # is where the status enumeration decides which quantization applies:
        # server_config wins over the module global.
        quantization = module.server_config.get(
            "transformer_quantization", module.transformer_quantization
        )
        dtype_policy = module.server_config.get(
            "transformer_dtype_policy", module.transformer_dtype_policy
        )
        transformer_filename = module.get_model_filename(
            model_type, quantization=quantization, dtype_policy=dtype_policy
        )
        text_encoder_urls = module.get_model_recursive_prop(
            model_type, "text_encoder_URLs", return_list=True
        )
        text_encoder_filename = (
            module.get_model_filename(
                model_type=model_type,
                quantization=module.text_encoder_quantization,
                dtype_policy=dtype_policy,
                URLs=text_encoder_urls,
            )
            if text_encoder_urls
            else ""
        )

        resolved: list[dict[str, Any]] = []
        for entry in entries or []:
            filename = entry.get("filename", "") if isinstance(entry, dict) else str(entry)
            extra_paths = entry.get("extra_paths") if isinstance(entry, dict) else None
            local = fl.get_local_model_filename(filename, extra_paths=extra_paths)
            if filename == transformer_filename:
                role = "transformer"
            elif filename and filename == text_encoder_filename:
                role = "text_encoder"
            else:
                role = "core"
            resolved.append(
                {
                    "role": role,
                    "filename": _basename(filename),
                    "url": filename,
                    "extra_paths": extra_paths,
                    "local_path": local,
                    "size_bytes": _file_size(local) if local else None,
                }
            )

    report["core_files"] = resolved
    report["resolved_transformer"] = _basename(transformer_filename)
    report["resolved_text_encoder"] = _basename(text_encoder_filename)
    say("  core files")
    for item in resolved:
        mark = "ok  " if item["local_path"] else "MISS"
        say(f"    [{mark}] {item['role']:<13} {item['filename']}")
        if item["local_path"]:
            say(f"             {item['local_path']}  ({human_bytes(item['size_bytes'])})")
        else:
            say(f"             searched: {weights['checkpoints_paths']}"
                f"{'' if not item['extra_paths'] else ' + ' + str(item['extra_paths'])}")
    say("")

    total = sum(item["size_bytes"] or 0 for item in resolved)
    report["core_bytes"] = total
    say(f"  core bytes on disk : {human_bytes(total)}")
    # The eyeball check no assertion replaces: is this the checkpoint you meant?
    say(f"  resolved transformer  : {_basename(transformer_filename) or '(none)'}")
    say(f"  resolved text encoder : {_basename(text_encoder_filename) or '(none)'}")
    say("")

    # -- the assertion itself -------------------------------------------------
    try:
        engine.assert_weights_complete(model_type)
    except WorkerError as exc:
        failures.append(f"{model_type}: {exc.message}")
        say(f"  FAIL  {exc.message}")
        for detail in exc.details:
            say(f"        missing: {detail}")
        hint = exc.detail.get("hint") if isinstance(exc.detail, dict) else None
        if hint:
            say(f"        hint: {hint}")
    else:
        say("  PASS  get_missing_core_file_entries_for_status(...) == []")

    if weights["available"]:
        say(f"  PASS  get_model_availability(...)['available'] "
            f"(status={weights['status']!r})")
    else:
        message = (
            f"{model_type}: download status is {weights['status']!r}, not 'available'; a "
            f"secondary file (text encoder / VAE_URL / preload_URL / model LoRA) is absent "
            f"and would be fetched on the first request"
        )
        if args.allow_partial:
            warnings.append(message)
            say(f"  WARN  {message}")
        else:
            failures.append(message)
            say(f"  FAIL  {message}")
    say("")

    # -- 3: warm settings/<model_type>_settings.json --------------------------
    started = time.monotonic()
    try:
        defaults = session.get_default_settings(model_type)  # shared/api.py:511
    except Exception as exc:  # noqa: BLE001
        failures.append(f"{model_type}: get_default_settings failed: {type(exc).__name__}: {exc}")
        say(f"  FAIL  get_default_settings: {type(exc).__name__}: {exc}")
        defaults = {}
    else:
        with _pushd(module_root(module)):
            settings_path = Path(module.get_settings_file_name(model_type))
            if not settings_path.is_absolute():
                settings_path = Path(module_root(module)) / settings_path
        report["settings_file"] = str(settings_path)
        report["settings_file_exists"] = settings_path.is_file()
        if settings_path.is_file():
            say(f"  PASS  settings cache warmed in {time.monotonic() - started:.2f}s")
            say(f"        {settings_path}")
        else:
            # get_default_settings only writes on the miss path (wgp.py:3157-3176);
            # if it is still absent the repo root is not writable.
            message = (
                f"{model_type}: {settings_path} was not created; the repo root is not "
                f"writable by this user and every worker will retry the write per job "
                f"(failure mode 24)"
            )
            failures.append(message)
            say(f"  FAIL  {message}")
        report["defaults"] = {
            key: defaults.get(key)
            for key in (
                "resolution",
                "video_length",
                "num_inference_steps",
                "flow_shift",
                "guidance_scale",
                "sample_solver",
                "sliding_window_size",
                "sliding_window_overlap",
            )
            if key in defaults
        }
        say("        defaults: "
            + ", ".join(f"{key}={value}" for key, value in report["defaults"].items()))
    say("")

    # -- 4: the assets neither enumeration covers -----------------------------
    if not args.skip_shared:
        with _pushd(module_root(module)):
            model_def = module.get_model_def(model_type)
            defs: list[dict[str, Any]] = [
                module.query_core_shared_model_files(),          # wgp.py:3547
                module.query_matanyone_download_def(module.server_config),
            ]

            def compute_list(filename: Any) -> list[str]:
                if filename is None:
                    return []
                text = str(filename)
                return [text[text.rfind("/") + 1:]]

            base_model_type = module.get_base_model_type(model_type)
            handler = module.model_types_handlers[base_model_type]
            handler_files = handler.query_model_files(compute_list, base_model_type, model_def)
            if not isinstance(handler_files, list):
                handler_files = [handler_files]
            defs.extend(one for one in handler_files if isinstance(one, dict))
            missing_shared = download_def_missing_files(defs)

        report["missing_shared"] = list(missing_shared)
        if missing_shared:
            message = (
                f"{model_type}: {len(missing_shared)} shared asset(s) absent "
                f"(first: {missing_shared[0]}); download_models(file_type=0) will fetch "
                f"them during the first request, on the clock"
            )
            if args.strict:
                failures.append(message)
                say(f"  FAIL  {message}")
            else:
                warnings.append(message)
                say(f"  WARN  {message}")
            for name in missing_shared[:15]:
                say(f"        {name}")
            if len(missing_shared) > 15:
                say(f"        ... and {len(missing_shared) - 15} more")
        else:
            say("  PASS  shared assets present (VAEs, tokenizer, DWPose, wav2vec, ...)")
        say("")

    # -- optional: LoRAs the endpoint is expected to serve --------------------
    wanted: list[str] = list(args.lora)
    if args.profile:
        from runpod_worker import schema

        for name in args.profile:
            try:
                fragment = schema.load_profile_fragment(
                    name, model_def=model_def_public, root=module_root(module)
                )
            except WorkerError as exc:
                failures.append(f"{model_type}: profile '{name}': {exc.message}")
                say(f"  FAIL  profile '{name}': {exc.message}")
                continue
            entries = fragment.get("activated_loras") or []
            if isinstance(entries, str):
                entries = [entries]
            wanted.extend(str(item) for item in entries if str(item).strip())

    if wanted:
        with _pushd(module_root(module)):
            lora_dir = module.get_lora_dir(model_type)  # wgp.py:2479
            checked = []
            for entry in dict.fromkeys(wanted):
                # wgp.py:3670-3677 -- the only place that decides where a LoRA lives.
                path = module.get_lora_local_path(lora_dir, entry)
                present = os.path.isfile(path)
                checked.append(
                    {
                        "lora": entry,
                        "path": path,
                        "present": present,
                        "size_bytes": _file_size(path) if present else None,
                    }
                )
                if present:
                    say(f"  PASS  lora {_basename(entry)}  ({human_bytes(_file_size(path))})")
                else:
                    message = f"{model_type}: lora not staged: {path}"
                    failures.append(message)
                    say(f"  FAIL  {message}")
        report["loras"] = checked
        report["lora_dir"] = lora_dir
        say("")

    return report, failures, warnings


def module_root(module: Any) -> str:
    """The repo root wgp was imported from (``shared/api.py:1082-1084`` pins it)."""
    return str(Path(module.__file__).resolve().parent)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _apply_env(args)

    from runpod_worker import config as _C
    _hf = _C.ensure_hf_transfer_sane()
    if _hf == "disabled":
        print("  hf_transfer        : HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer "
              "is not installed -- forced to 0 (downloads would have failed)")

    from runpod_worker import config as C  # noqa: PLC0415 - must follow _apply_env
    from runpod_worker import engine
    from runpod_worker.errors import WorkerError

    model_types = list(dict.fromkeys([*args.model_types, *args.model_type_flags]))
    if not model_types:
        model_types = [C.CONFIG.model_type]

    def say(*parts: Any) -> None:
        if not args.quiet:
            print(*parts, flush=True)

    started = time.monotonic()
    try:
        session = engine.boot()
    except WorkerError as exc:
        print(f"FAIL  worker cannot boot: {exc}", file=sys.stderr)
        for detail in exc.details:
            print(f"      {detail}", file=sys.stderr)
        return EXIT_FAILED
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  import wgp: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED

    module = session._ensure_runtime().module
    say(f"wgp {getattr(module, 'WanGP_version', '?')} imported in "
        f"{time.monotonic() - started:.1f}s from {module_root(module)}")
    say(f"config: {C.CONFIG_DIR / 'wgp_config.json'}")
    say("")

    report: dict[str, Any] = {
        "root": str(C.WANGP_ROOT),
        "config_path": str(C.CONFIG_DIR / "wgp_config.json"),
        "attention_mode": getattr(module, "attention_mode", None),
        "boot_s": round(time.monotonic() - started, 1),
        "models": [],
    }
    failures: list[str] = []
    warnings: list[str] = []

    for model_type in model_types:
        one, model_failures, model_warnings = verify_one(
            module=module,
            session=session,
            engine=engine,
            model_type=model_type,
            args=args,
            say=say,
        )
        report["models"].append(one)
        failures.extend(model_failures)
        warnings.extend(model_warnings)

    report["failures"] = failures
    report["warnings"] = warnings
    report["ok"] = not failures

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        say(f"report written to {args.json_out}")

    say("=" * 78)
    if warnings:
        for line in warnings:
            print(f"WARN  {line}")
    if failures:
        print(f"FAILED: {len(failures)} problem(s)", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        print(
            "\nFix on the prefetch Pod, with the SAME quantization the workers run:\n"
            f"  python3 -m runpod_worker.scripts.prefetch_weights {' '.join(model_types)}",
            file=sys.stderr,
        )
        return EXIT_FAILED

    print(f"OK  {', '.join(model_types)} verified in {time.monotonic() - started:.1f}s"
          f"{' (with warnings)' if warnings else ''}")
    return EXIT_OK


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(EXIT_INTERRUPTED) from None
