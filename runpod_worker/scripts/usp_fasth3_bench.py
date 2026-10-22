#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FastH3 Omni NVFP4 benchmark: one GPU vs two-GPU Ulysses sequence parallelism (models/minimax_h3/usp.py) with VSA.

One fixed job, run 1 cold + 2 warm times per mode: ``minimax_h3_fasth3_omni_nvfp4``, 832x480, video_length 124 (5 s at 24 fps),
seed 42, one generated reference image (``video_prompt_type="I"``, ``image_refs=[png]``), the model's default prompt,
``config=",lightvae_int8"`` (slot 2, the Video VAE: LightVAE Lite INT8 decoder), WanGP defaults otherwise (FastH3 itself forces
VSA attention in the DiT, Euler and its 8 PDD steps). Drives WanGP through the worker's engine (``engine.boot`` /
``engine.run`` -> ``shared.api``) rather than ``handler.run_job``: the worker's request schema admits only the four standard
H3 model types, not FastH3.

On the 2x RTX 5090 box, inside the worker image (or any WanGP checkout with its requirements) with the weights present:

    cd /opt/wangp          # the repo root; or set WANGP_ROOT
    # A -- single process, single GPU. Run it FIRST: it downloads anything missing ONCE (two ranks must never download
    #      concurrently), writes settings/<model>_settings.json, and is the reference B is compared against.
    CUDA_VISIBLE_DEVICES=0 python3 runpod_worker/scripts/usp_fasth3_bench.py --out-dir /workspace/fasth3_bench

    # B -- 2-GPU USP over NCCL (torchrun gives RANK / WORLD_SIZE / LOCAL_RANK / MASTER_*), then the A/B comparison
    torchrun --standalone --nproc_per_node=2 runpod_worker/scripts/usp_fasth3_bench.py --out-dir /workspace/fasth3_bench

    # either order, no GPU work: re-print the comparison from the saved outputs
    python3 runpod_worker/scripts/usp_fasth3_bench.py --out-dir /workspace/fasth3_bench --compare

Each rank pins CUDA_VISIBLE_DEVICES to its LOCAL_RANK before torch loads (wgp / mmgp expect ``cuda:0``), gets its own WanGP
config and output directories (both ranks run the whole job and save a video; sharing them races), and initializes NCCL
here (``usp.activate`` then only patches) with a 30 min timeout so a slow cold load on one rank does not trip the watchdog at
the first collective. Only rank 0 writes bench outputs and prints results.

Output, rank 0, one line per run the moment it completes (also appended to <out-dir>/<mode>-runs.jsonl), then a summary:

    BENCH_FASTH3_RUN {"mode": "gpu1"|"usp2", "run": 0, "kind": "cold"|"warm", "status": "ok", "total_s": ..., "load_s": ...,
                      "text_encode_s": ..., "pre_denoise_s": ..., "denoise_s": ..., "steps": 8, "s_per_step": ...,
                      "vae_decode_s": ..., "audio_decode_s": ..., "save_s": ..., "generate_s": ..., "vsa_calls": ..., ...}
    BENCH_FASTH3_SUMMARY {...}
    BENCH_FASTH3_COMPARE {"run": i, "latent_max_abs": ..., "latent_rel_l2": ..., "video_psnr_db": ..., ...}  # once both modes exist

Phases are timed in-process (class-level wrappers installed before the model loads, CUDA-synchronized where GPU work ends):
text_encode = MiniMaxH3Pipeline._encode_prompt (includes TextEncoderCache hits: the warm runs reuse the prompt, so
``text_encoder_runs`` says whether the encoder actually ran); pre_denoise = generate() start -> first DiT forward (text encode
plus the reference image encode); denoise = first DiT forward start -> last DiT forward end (the 8 Euler steps); vae_decode /
audio_decode = the video / audio VAE decodes; save = wgp's save_video + wav write + audio mux; load = wgp.load_models (cold only);
total = the whole engine.run. Saved per run: <mode>-run<i>.mp4 and <mode>-run<i>-latents.pt (the video and audio latents the
VAEs decode), compared across modes as latent max-abs / relative-L2 difference and video PSNR (all frames).

Env: WANGP_ROOT (default: this checkout), WANGP_CHECKPOINTS_PATHS (default <WANGP_VOLUME_ROOT>/ckpts, then <root>/ckpts),
WANGP_CONFIG_DIR / WANGP_OUTPUT_DIR (default under --out-dir, suffixed per rank). Useful when diagnosing B:
NCCL_DEBUG=INFO (prints the transport: RTX 5090 has no NVLink and GeForce P2P is normally off, so expect SHM over PCIe),
TORCH_NCCL_ASYNC_ERROR_HANDLING=1.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
import os
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

# Per-rank GPU pinning must precede any torch import (as usp_bench.py): each rank's device is then 'cuda:0'.
_LOCAL_RANK = os.environ.get("LOCAL_RANK")
if _LOCAL_RANK is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _LOCAL_RANK

_ROOT = os.environ.setdefault("WANGP_ROOT", str(Path(__file__).resolve().parents[2]))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

MODEL_TYPE = "minimax_h3_fasth3_omni_nvfp4"
RESOLUTION = "832x480"
VIDEO_LENGTH = 124
SEED = 42
CONFIG = ",lightvae_int8"
#: Local copy of runpod_worker.config.CLI_ATTENTION_MODES (it may not be imported before the env is pinned, usp_bench.py).
BENCH_CLI_ATTENTION_MODES = ("auto", "sdpa", "sage", "sage2", "flash", "xformers")
SAVE_HOOKS = ("save_video", "write_wav_file", "combine_and_concatenate_video_with_audio_tracks")


class _Phases:
    """Per-run timings of the wrapped calls: total seconds, first start, last end and call count per key."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total, self.first, self.last, self.calls = {}, {}, {}, {}
        self.latents = {}

    def span(self, key):
        return self.last[key] - self.first[key] if key in self.first and key in self.last else None


PH = _Phases()


def _wrap(owner, name, key, sync=False, before=None):
    """owner.name (a class's function or a module's) timed into PH under key; before(args) runs untimed first."""
    import torch

    original = getattr(owner, name)

    @functools.wraps(original)
    def timed(*args, **kwargs):
        if before is not None:
            before(args)
        start = time.perf_counter()
        PH.first.setdefault(key, start)
        try:
            return original(*args, **kwargs)
        finally:
            if sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            PH.total[key] = PH.total.get(key, 0.0) + end - start
            PH.last[key] = end
            PH.calls[key] = PH.calls.get(key, 0) + 1

    setattr(owner, name, timed)


def _install_hooks(capture):
    """Class-level timers on the H3 pipeline, DiT, VAEs and VSA. Call after usp.activate() (the DiT forward is then USP's)
    and before the model loads."""
    from models.minimax_h3 import audio_vae, pipeline, text_encoder, transformer, video_vae
    from models.minimax_h3.vsa import attention as vsa

    def keep(name):
        def grab(args):
            if capture:
                PH.latents[name] = args[1].detach().float().cpu()
        return grab

    _wrap(pipeline.MiniMaxH3Pipeline, "generate", "generate", sync=True)
    _wrap(pipeline.MiniMaxH3Pipeline, "_encode_prompt", "text_encode", sync=True)
    _wrap(text_encoder.MiniMaxH3TextEncoder, "encode", "text_encoder_run")
    _wrap(transformer.MiniMaxH3Model, "forward", "forward", sync=True)
    _wrap(video_vae.MiniMaxH3VideoVAE, "decode", "vae_decode", sync=True, before=keep("video"))
    _wrap(audio_vae.MiniMaxH3AudioVAE, "decode", "audio_decode", sync=True, before=keep("audio"))
    _wrap(vsa.VSAAttention, "__call__", "vsa")


def _install_wgp_hooks(module):
    _wrap(module, "load_models", "load", sync=True)
    for name in SAVE_HOOKS:
        if hasattr(module, name):
            _wrap(module, name, "save")


def _reference_image(path):
    """A plain character for <Picture 1>: a figure in a teal jacket on a warm gradient."""
    from PIL import Image, ImageDraw

    width, height = 512, 640
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        draw.line([(0, y), (width, y)], fill=(250 - y // 8, 200 - y // 10, 150 + y // 12))
    draw.ellipse((196, 90, 316, 220), fill=(236, 196, 160), outline=(60, 40, 30), width=3)  # head
    draw.pieslice((186, 70, 326, 190), 180, 360, fill=(70, 45, 30))  # hair
    draw.ellipse((228, 140, 242, 154), fill=(30, 30, 30))
    draw.ellipse((270, 140, 284, 154), fill=(30, 30, 30))
    draw.arc((232, 160, 280, 196), 20, 160, fill=(150, 60, 60), width=4)
    draw.rounded_rectangle((166, 230, 346, 470), radius=30, fill=(20, 140, 140), outline=(10, 60, 60), width=3)  # jacket
    draw.rectangle((196, 470, 240, 600), fill=(40, 50, 90))  # legs
    draw.rectangle((272, 470, 316, 600), fill=(40, 50, 90))
    draw.rectangle((120, 240, 166, 420), fill=(20, 140, 140))  # arms
    draw.rectangle((346, 240, 392, 420), fill=(20, 140, 140))
    image.save(path)
    return path


def _video_frames(path):
    import imageio

    reader = imageio.get_reader(str(path), "ffmpeg")
    try:
        for frame in reader:
            yield frame
    finally:
        reader.close()


def _psnr(path_a, path_b):
    import numpy as np

    total, count, frames = 0.0, 0, 0
    for a, b in itertools.zip_longest(_video_frames(path_a), _video_frames(path_b)):
        if a is None or b is None:
            return {"error": f"frame counts differ (equal for the first {frames})"}
        if a.shape != b.shape:
            return {"error": f"frame shape {a.shape} != {b.shape}"}
        diff = a.astype(np.float64) - b.astype(np.float64)
        total += float((diff * diff).sum())
        count += diff.size
        frames += 1
    if not count:
        return {"error": "no frames"}
    mse = total / count
    return {"video_psnr_db": round(10 * math.log10(255.0 ** 2 / mse), 2) if mse > 0 else "inf",
            "video_mse": mse, "frames_compared": frames}


def _latent_diff(path_a, path_b):
    import torch

    a, b = torch.load(path_a), torch.load(path_b)
    out = {}
    for name in sorted(set(a) & set(b)):
        x, y = a[name].float(), b[name].float()
        if x.shape != y.shape:
            out[name] = {"error": f"shape {tuple(x.shape)} != {tuple(y.shape)}"}
            continue
        out[name] = {"max_abs": (x - y).abs().max().item(),
                     "rel_l2": ((x - y).norm() / x.norm().clamp_min(1e-12)).item()}
    return out


def _compare(out_dir, mode_a="gpu1", mode_b="usp2"):
    """Every run index present in both modes, plus each mode's warm runs against its own run 0 (determinism)."""
    results = []
    for index in range(16):
        files = {mode: (out_dir / f"{mode}-run{index}.mp4", out_dir / f"{mode}-run{index}-latents.pt") for mode in (mode_a, mode_b)}
        if not all(path.is_file() for pair in files.values() for path in pair):
            continue
        row = {"run": index, "a": mode_a, "b": mode_b}
        row.update({f"latent_{name}_{k}": v for name, stats in _latent_diff(files[mode_a][1], files[mode_b][1]).items()
                    for k, v in stats.items()})
        row.update(_psnr(files[mode_a][0], files[mode_b][0]))
        results.append(row)
        print("BENCH_FASTH3_COMPARE " + json.dumps(row), flush=True)
    for mode in (mode_a, mode_b):
        base = out_dir / f"{mode}-run0-latents.pt"
        for index in range(1, 16):
            other = out_dir / f"{mode}-run{index}-latents.pt"
            if base.is_file() and other.is_file():
                row = {"determinism": mode, "run": index, "vs_run": 0}
                row.update({f"latent_{name}_{k}": v for name, stats in _latent_diff(base, other).items() for k, v in stats.items()})
                print("BENCH_FASTH3_COMPARE " + json.dumps(row), flush=True)
                results.append(row)
    if not results:
        print(f"BENCH_FASTH3_COMPARE {{\"skipped\": \"no run present for both {mode_a} and {mode_b} in {out_dir}\"}}", flush=True)
    return results


def _round(value, digits=2):
    return None if value is None else round(value, digits)


def _report(out_dir, mode, world, boot_s, runs):
    """Rank 0: the summary line and results file, then the A/B comparison once both modes have run."""
    def mean(kind, key):
        values = [r[key] for r in runs if r["kind"] == kind and r["status"] == "ok" and r[key] is not None]
        return _round(sum(values) / len(values)) if values else None

    keys = ("total_s", "text_encode_s", "pre_denoise_s", "denoise_s", "s_per_step", "vae_decode_s", "save_s")
    summary = {"mode": mode, "world": world, "boot_s": boot_s, "runs": len(runs),
               "failed": sum(r["status"] != "ok" for r in runs),
               "cold": {key: mean("cold", key) for key in keys + ("load_s",)},
               "warm_mean": {key: mean("warm", key) for key in keys}}
    print("BENCH_FASTH3_SUMMARY " + json.dumps(summary), flush=True)
    (out_dir / f"{mode}-results.json").write_text(json.dumps({"summary": summary, "runs": runs}, indent=1), encoding="utf-8")
    other = "usp2" if mode == "gpu1" else "gpu1"
    if any((out_dir / f"{other}-run{index}.mp4").is_file() for index in range(16)):
        try:
            _compare(out_dir, "gpu1", mode if mode != "gpu1" else other)
        except Exception as exc:  # noqa: BLE001 - the timings above are the result; --compare can redo this
            print(f"bench: comparison failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=os.environ.get("BENCH_OUT_DIR", "/workspace/fasth3_bench"))
    parser.add_argument("--runs", type=int, default=3, help="1 cold + (runs - 1) warm")
    parser.add_argument("--attention", default="sdpa",
                        help="global attention mode for everything but the DiT blocks, which FastH3 pins to VSA")
    parser.add_argument("--profile", default=os.environ.get("WANGP_PROFILE", "4"), help="mmgp memory profile (WanGP default 4)")
    parser.add_argument("--budget-s", type=float, default=3600.0)
    parser.add_argument("--compare", action="store_true", help="only compare the saved outputs of both modes, then exit")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if args.compare:
        _compare(out_dir)
        return 0

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    mode = f"usp{world}" if world > 1 else "gpu1"
    lead = rank == 0
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pin every knob before runpod_worker.config is imported (it freezes them at import; usp_bench.py explains the trap).
    if args.attention not in BENCH_CLI_ATTENTION_MODES:
        parser.error(f"--attention must be one of {BENCH_CLI_ATTENTION_MODES}")
    suffix = f"-rank{rank}" if world > 1 else ""
    os.environ["WANGP_CONFIG_DIR"] = (os.environ.get("WANGP_CONFIG_DIR") or str(out_dir / "wgp-config")) + f"/{mode}{suffix}"
    os.environ["WANGP_OUTPUT_DIR"] = (os.environ.get("WANGP_OUTPUT_DIR") or str(out_dir / "wangp-out")) + f"/{mode}{suffix}"
    os.environ["WANGP_MODEL_TYPE"] = MODEL_TYPE
    os.environ["WANGP_MODEL_CONFIG"] = CONFIG
    os.environ["WANGP_ATTENTION"] = args.attention
    os.environ["WANGP_PROFILE"] = str(args.profile)
    cli = f"--attention {args.attention} --profile {args.profile} --verbose 1"
    os.environ["WANGP_CLI_ARGS"] = cli
    os.environ.setdefault("WANGP_EAGER_BOOT", "0")
    os.environ["WANGP_WARM"] = "0"  # the cold run must include the model load

    from runpod_worker import config as C  # noqa: PLC0415

    assert C.CLI_ATTENTION_MODES == BENCH_CLI_ATTENTION_MODES, C.CLI_ATTENTION_MODES
    live = C.reload_config()
    assert list(live.cli_args) == cli.split(), f"CONFIG.cli_args={list(live.cli_args)} != {cli.split()}"
    assert live.model_type == MODEL_TYPE and live.model_config == CONFIG, (live.model_type, live.model_config)

    import torch
    import torch.distributed as dist

    if world > 1:
        # usp.activate() would init NCCL itself with the default timeout; a cold load can differ by minutes between ranks.
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)) % max(torch.cuda.device_count(), 1))
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    from models.minimax_h3 import usp  # noqa: PLC0415

    usp.activate()  # before the model loads: class-level patches; a passthrough when WORLD_SIZE is 1
    assert usp.is_active() == (world > 1)
    _install_hooks(capture=lead)

    from runpod_worker import engine  # noqa: PLC0415

    boot_started = time.monotonic()
    session = engine.boot()
    boot_s = round(time.monotonic() - boot_started, 2)
    module = engine._runtime(session).module
    _install_wgp_hooks(module)

    if world > 1:
        report = engine.weights_report(MODEL_TYPE)
        if report["missing_core"]:
            if lead:
                print(f"bench: {len(report['missing_core'])} core weight file(s) missing -- run mode A (one process) first so "
                      f"they download once: {report['missing_core'][:4]}", file=sys.stderr, flush=True)
            dist.destroy_process_group()
            return 2

    ref_path = out_dir / "reference.png"
    if lead and not ref_path.is_file():
        _reference_image(ref_path)
    if world > 1:
        dist.barrier()
    settings = session.merge_settings_with_defaults({
        "model_type": MODEL_TYPE, "resolution": RESOLUTION, "video_length": VIDEO_LENGTH, "seed": SEED,
        "video_prompt_type": "I", "image_refs": [str(ref_path)], "config": CONFIG,
    })
    if lead:
        header = {"mode": mode, "world": world, "torch": torch.__version__, "boot_s": boot_s, "profile": str(args.profile),
                  "attention": args.attention, "model_type": MODEL_TYPE, "config": settings.get("config"),
                  "steps": settings.get("num_inference_steps"), "solver": settings.get("sample_solver"),
                  "prompt": str(settings.get("prompt", ""))[:80]}
        try:  # diagnostics only: rank 0 must not die here while the other ranks head into the collectives
            header["device"] = torch.cuda.get_device_name(0)
            header["nccl"] = str(torch.cuda.nccl.version()) if world > 1 else None
        except Exception as exc:  # noqa: BLE001
            header["device"] = f"unknown ({type(exc).__name__})"
        print("BENCH_FASTH3_START " + json.dumps(header), flush=True)

    runs = []
    for index in range(args.runs):
        PH.reset()
        started = time.perf_counter()
        error = None
        try:
            outcome = engine.run(dict(settings), budget_s=args.budget_s)
        except Exception as exc:  # noqa: BLE001 - a failed leg is a result, not a crash
            outcome, error = None, f"{type(exc).__name__}: {exc}"
        total_s = time.perf_counter() - started
        ok = outcome is not None and outcome.success and outcome.video_path is not None
        if error is None and not ok:
            error = "; ".join((outcome.errors if outcome is not None else []) or ["no video produced"])
        steps = PH.calls.get("forward", 0)
        denoise_s = PH.span("forward")
        pre_denoise_s = PH.first["forward"] - PH.first["generate"] if "forward" in PH.first and "generate" in PH.first else None
        run = {"mode": mode, "world": world, "run": index, "kind": "cold" if index == 0 else "warm",
               "status": "ok" if ok and error is None else "error", "error": error,
               "total_s": _round(total_s), "load_s": _round(PH.total.get("load")), "generate_s": _round(PH.total.get("generate")),
               "text_encode_s": _round(PH.total.get("text_encode")), "text_encoder_runs": PH.calls.get("text_encoder_run", 0),
               "pre_denoise_s": _round(pre_denoise_s), "denoise_s": _round(denoise_s), "steps": steps,
               "s_per_step": _round(denoise_s / steps, 3) if denoise_s and steps else None,
               "vae_decode_s": _round(PH.total.get("vae_decode")), "audio_decode_s": _round(PH.total.get("audio_decode")),
               "save_s": _round(PH.total.get("save")), "vsa_calls": PH.calls.get("vsa", 0),
               "vram_peak_gb": _round((engine.STATS.get("vram_peak_mb") or 0) / 1024),
               "engine_gen_s": outcome.gen_s if outcome is not None else None}
        if lead and ok:
            try:  # never raise here: the other ranks are already waiting in the next run's collectives
                video = out_dir / f"{mode}-run{index}.mp4"
                shutil.copyfile(outcome.video_path, video)
                latents = out_dir / f"{mode}-run{index}-latents.pt"
                torch.save(PH.latents, latents)
                run.update(video=str(video), latents=str(latents))
            except Exception as exc:  # noqa: BLE001
                run["save_error"] = f"{type(exc).__name__}: {exc}"
        if ok and not run["vsa_calls"]:
            run.update(status="error", error="VSA never ran: the DiT did not use video sparse attention")
        runs.append(run)
        if lead:
            print("BENCH_FASTH3_RUN " + json.dumps(run), flush=True)
            with open(out_dir / f"{mode}-runs.jsonl", "a", encoding="utf-8") as log:
                log.write(json.dumps(run) + "\n")
        del outcome

    try:
        if lead:
            _report(out_dir, mode, world, boot_s, runs)
    finally:  # every rank reaches the barrier, even when rank 0's reporting fails
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
    return 1 if any(r["status"] != "ok" for r in runs) else 0


if __name__ == "__main__":
    sys.exit(main())
