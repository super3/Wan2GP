#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark harness: single-GPU attention modes and multi-GPU USP denoise.

Runs the production job path (``handler.run_job``) end to end with the fixed
production clip (832x480, 124 frames, 4-step turbo LoRA) and reports per-run
metrics. Two launch shapes, both inside the worker image with weights present:

    # single GPU, any attention mode
    python3 runpod_worker/scripts/usp_bench.py --attention sdpa --profile 1

    # 2-GPU Ulysses sequence parallelism (models/minimax_h3/usp.py)
    torchrun --nproc-per-node=2 runpod_worker/scripts/usp_bench.py \
        --attention sdpa --profile 1 --tag usp2

Each seed runs once; the first run is the cold (model-load) number, later
seeds are warm. Rank 0 prints one JSON line per run and a final summary, and
POSTs the summary to $LOG_SHIP_URL if set. Ranks > 0 run the identical jobs
(USP needs lockstep) but stay silent and discard outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

# Per-rank GPU pinning must precede any torch import: torchrun gives
# LOCAL_RANK; mapping it to CUDA_VISIBLE_DEVICES makes each rank's device
# 'cuda:0' -> a distinct physical GPU, which is what wgp/mmgp expect.
_LOCAL_RANK = os.environ.get("LOCAL_RANK")
if _LOCAL_RANK is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _LOCAL_RANK

_ROOT = os.environ.get("WANGP_ROOT") or "/opt/wangp"
for _p in (_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: The production clip, verbatim (same payload the endpoint serves).
PROMPT = (
    "integrated_multimodal_description: [Shot 1] A five-second cinematic close-up of a "
    "brass desk lamp switching on in a dark study, dust drifting through the beam. The "
    "camera pushes in slowly.\n"
    "overall_soundscape: A single click, a low electrical hum, faint room tone.\n"
    "non_diegetic_music: One soft sustained cello note."
)
ACCEL_PROFILE = "Turbo Lightx2v FL2V 4 Steps v1.0 768p"
RESOLUTION = "832x480"
VIDEO_LENGTH = 124


def _ship(payload: dict) -> None:
    url = os.environ.get("LOG_SHIP_URL", "").strip()
    if not url:
        return
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps({"records": [payload]}, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "wangp-usp-bench/1"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception as exc:  # noqa: BLE001 - shipping is best effort
        print(f"bench: result shipping failed: {exc!r}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention", default="sdpa", help="sdpa | sage2 | sol | ...")
    parser.add_argument("--profile", default=os.environ.get("WANGP_PROFILE", "1"))
    parser.add_argument("--seeds", default="12345,5555")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    tag = args.tag or (f"usp{world}" if world > 1 else f"single-{args.attention}")

    # Config must be pinned before runpod_worker.config / wgp import.
    os.environ["WANGP_ATTENTION"] = args.attention
    os.environ["WANGP_PROFILE"] = str(args.profile)
    # "sol" is config-only: wgp.py:3303 rejects it as a CLI value, so pass it
    # through WANGP_ATTENTION (-> wgp_config.json) with no --attention flag.
    # NOTE: runpod_worker.config must NOT be imported before this point --
    # it builds its CONFIG singleton at import time, so an early import
    # silently freezes the DEFAULT profile/attention (a bench run reported
    # "profile 1" while actually running profile 4: vram_peak 5.6 GB, not 25).
    from runpod_worker.config import CLI_ATTENTION_MODES  # noqa: PLC0415

    cli = f"--profile {args.profile} --verbose 1"
    if args.attention in CLI_ATTENTION_MODES:
        cli = f"--attention {args.attention} " + cli
    os.environ["WANGP_CLI_ARGS"] = cli
    os.environ.setdefault("WANGP_EAGER_BOOT", "0")

    if world > 1:
        from models.minimax_h3 import usp
        usp.activate()
        if args.attention == "sol":
            print("bench: sol attention is unsupported under USP", file=sys.stderr)
            return 2

    from runpod_worker import handler as H

    boot_started = time.monotonic()
    H.bootstrap()
    boot_s = round(time.monotonic() - boot_started, 2)

    # WanGP degrades to SDPA SILENTLY when the requested kernel is missing or
    # unsupported (a bench run reported sdpa/sol/sage2 within 0.1 s of each
    # other -- all three were SDPA). Ask the runtime what it actually holds
    # instead of trusting the request.
    def probe_attention():
        """shared_state['_attention'] is only populated once a generation has
        configured the model -- probing at bootstrap returns None."""
        try:
            from mmgp import offload  # noqa: PLC0415
            return offload.shared_state.get("_attention")
        except Exception as exc:  # noqa: BLE001
            return f"probe_failed: {exc!r}"

    effective = installed = supported = None
    try:
        from mmgp import offload  # noqa: PLC0415
        from shared.attention import (  # noqa: PLC0415
            get_attention_modes,
            get_supported_attention_modes,
        )
        installed = list(get_attention_modes())
        supported = list(get_supported_attention_modes())
    except Exception as exc:  # noqa: BLE001 - probe must never fail the bench
        effective = f"probe_failed: {exc!r}"
    if rank == 0:
        print(f"BENCH_ATTENTION requested={args.attention} installed={installed}", flush=True)

    runs = []
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        job = {
            "id": f"bench-{tag}-{seed}-r{rank}",
            "input": {
                "model_type": os.environ.get("WANGP_MODEL_TYPE", "minimax_h3_fl2va_pruned"),
                "profile": ACCEL_PROFILE,
                "settings": {
                    "prompt": PROMPT,
                    "resolution": RESOLUTION,
                    "video_length": VIDEO_LENGTH,
                    "seed": seed,
                },
                "output": {"mode": "b64"},
                "runtime": {"timeout_s": 2400},
            },
        }
        started = time.monotonic()
        response = H.run_job(job)
        wall_s = round(time.monotonic() - started, 2)
        metrics = dict(response.get("metrics") or {})
        run = {
            "tag": tag, "attention": args.attention, "profile": str(args.profile),
            "world": world, "rank": rank, "seed": seed,
            "status": response.get("status"),
            "error": response.get("error_message"),
            "wall_s": wall_s,
            "generate_s": metrics.get("generate_s"),
            "vram_peak_mb": metrics.get("vram_peak_mb"),
            "phase_marks_s": metrics.get("phase_marks_s"),
        }
        # Do NOT infer the profile from VRAM: measured on a 96 GB PRO 6000,
        # profile 1 pins every component to reserved RAM and streams from
        # there, so its VRAM peak (5.6 GB) is indistinguishable from profile
        # 4's. The profile is proven by mmgp's own "Pinning data of ..." lines
        # in the shipped stream log (profile 1 pins transformer + text encoder
        # + VAEs, ~51 GB; profile 4 pins only the transformer, ~20 GB).
        run["effective_attention"] = probe_attention()
        runs.append(run)
        if rank == 0:
            print("BENCH_RUN " + json.dumps(run), flush=True)
        # Every response references container-shared paths; drop it promptly.
        del response

    effective = runs[-1].get("effective_attention") if runs else None
    attention_ok = effective == args.attention
    summary = {"event": "usp_bench_summary", "tag": tag, "attention": args.attention,
               "effective_attention": effective, "attention_ok": attention_ok,
               "installed_attention": installed, "supported_attention": supported,
               "profile": str(args.profile), "world": world, "boot_s": boot_s, "runs": runs}
    if rank == 0:
        print("BENCH_SUMMARY " + json.dumps(summary), flush=True)
        _ship(summary)
    failed = any(r["status"] != "completed" for r in runs)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
