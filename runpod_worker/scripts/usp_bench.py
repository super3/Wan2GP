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
#: Attention modes WanGP honours ONLY as a per-generation override, never as a
#: global mode (shared/attention.py:28-33 keeps them out of get_attention_modes).
OVERRIDE_ONLY_MODES = ("sol",)
#: Local copy of runpod_worker.config.CLI_ATTENTION_MODES -- needed BEFORE
#: that module may be imported (see the freeze note in main()). A drift
#: assert below keeps the two in sync.
BENCH_CLI_ATTENTION_MODES = ("auto", "sdpa", "sage", "sage2", "flash", "xformers")

ACCEL_PROFILE = "Turbo Lightx2v FL2V 4 Steps v1.0 768p"
RESOLUTION = "832x480"
VIDEO_LENGTH = 124


def _ship(payload: dict) -> None:
    url = os.environ.get("LOG_SHIP_URL", "").strip()
    if not url:
        return
    # Several pods may share one sink, and the bench posts its summary directly
    # rather than through obs.py, so nothing else stamps an identity on it. Two
    # concurrent pods produced one interleaved stream of same-shaped summaries
    # that could not be attributed after the fact.
    payload = dict(payload)
    payload.setdefault("run_id", os.environ.get("BENCH_RUN_ID", ""))
    payload.setdefault("host_gpu", os.environ.get("BENCH_HOST_GPU", ""))
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
    #: The frame lattice is video_length >= 107 and == 5 (mod 17), so only
    #: 107 + 17k is legal (124 = 5.2 s, 243 = 10.1 s at 24 fps).
    parser.add_argument("--frames", type=int, default=VIDEO_LENGTH)
    #: torch.compile the transformer (wgp.py:3316 --compile -> compile="transformer").
    parser.add_argument("--compile", action="store_true")
    #: WanGP "config selection" string, e.g. "fp8mix" for the FP8 video VAE
    #: (minimax_h3_handler.py:237). Comma-separated for several.
    parser.add_argument("--config", default="")
    #: server_config["video_output_codec"]: libx264_8 (default) | h264_nvenc | ...
    parser.add_argument("--codec", default="")
    #: server_config["vae_config"]: 0 = "auto", which for MiniMax H3 is a hard
    #: 256 px tile regardless of VRAM (video_vae.py:52-55). 1 = no tiling.
    parser.add_argument("--vae-config", dest="vae_config", default="")
    args = parser.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    tag = args.tag or (f"usp{world}" if world > 1 else f"single-{args.attention}")

    # EVERY knob must be pinned into os.environ before runpod_worker.config is
    # imported: `CONFIG = WorkerConfig()` runs at import time and freezes
    # cli_args / model_config for the process lifetime. Importing it even once
    # -- e.g. just to read CLI_ATTENTION_MODES -- locks in whatever the image
    # baked (ENV WANGP_CLI_ARGS="--attention sdpa --profile 4 --verbose 1"),
    # and every later os.environ write is silently ignored. That bug voided a
    # whole matrix: --attention sage2, --compile and --config fp8mix legs all
    # ran plain sdpa/no-compile/fp16 while reporting their own tag. Hence the
    # local copy of the whitelist below and the reload_config() assertions.
    os.environ["WANGP_ATTENTION"] = args.attention
    os.environ["WANGP_PROFILE"] = str(args.profile)

    # Measured: a "sol" leg ran at SDPA speed and never printed
    # "[MiniMax H3] Sol-Attn enabled" -- WanGP ignores sol as a global mode and
    # takes it only from the per-job override_attention setting.
    override_attention = None
    if args.attention in OVERRIDE_ONLY_MODES:
        override_attention = args.attention
        os.environ["WANGP_ATTENTION"] = "sdpa"

    if args.config:
        os.environ["WANGP_MODEL_CONFIG"] = args.config
    if args.codec:
        os.environ["WANGP_VIDEO_CODEC"] = args.codec
    if args.vae_config:
        os.environ["WANGP_VAE_CONFIG"] = args.vae_config

    cli = f"--profile {args.profile} --verbose 1"
    if args.compile:
        cli += " --compile"
    # "sol" is config-only: wgp.py:3303 rejects it as a CLI value, so it must
    # reach wgp through WANGP_ATTENTION -> wgp_config.json with no --attention.
    if os.environ["WANGP_ATTENTION"] in BENCH_CLI_ATTENTION_MODES:
        cli = f"--attention {os.environ['WANGP_ATTENTION']} " + cli
    os.environ["WANGP_CLI_ARGS"] = cli
    os.environ.setdefault("WANGP_EAGER_BOOT", "0")

    # Only NOW may config be imported. reload_config() makes the singleton
    # match the env even if some earlier import already built it.
    from runpod_worker import config as _cfg  # noqa: PLC0415

    assert _cfg.CLI_ATTENTION_MODES == BENCH_CLI_ATTENTION_MODES, (
        "runpod_worker.config.CLI_ATTENTION_MODES drifted from the bench copy: "
        f"{_cfg.CLI_ATTENTION_MODES} != {BENCH_CLI_ATTENTION_MODES}"
    )
    live = _cfg.reload_config()
    # Hard-fail rather than silently benchmark the wrong thing.
    assert list(live.cli_args) == cli.split(), (
        f"CONFIG.cli_args={list(live.cli_args)} does not match the requested "
        f"{cli.split()} -- config was frozen before the env was pinned"
    )
    assert live.model_config == args.config.rstrip(","), (
        f"CONFIG.model_config={live.model_config!r} != requested {args.config!r}"
    )

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
    # GROUND TRUTH: shared_state["_attention"] is set by a CONTEXT MANAGER
    # (shared/attention.py attention_config_shared_state) that RESTORES the old
    # value when the generation ends -- so reading it after a job reports the
    # restored default, not what ran. A leg once looked like an sdpa fallback
    # purely because of that. Wrap the DiT's own pay_attention reference (bound
    # at import in models/minimax_h3/transformer.py, so patching the module it
    # came from would not take) and record the mode on the first real call.
    _kernel = {}

    def install_kernel_probe():
        """Wrap EVERY module-level pay_attention reference the DiT might use.

        transformer.py binds one at import, but MiniMax H3's DiT blocks never
        call it: ``self.sol_attention`` is always constructed (transformer.py
        :473), so ``Attention.forward`` takes the
        ``self.sol_attention(qkv_list, use_sol)`` branch, and that lands on the
        SEPARATE reference bound in sol_attention.py. Wrapping only the first
        one probed the TokenRefiner instead of the 50 DiT blocks -- which is
        how a sage2 leg came back reporting "sdpa"."""
        from mmgp import offload  # noqa: PLC0415

        def wrap(module, label):
            original = getattr(module, "pay_attention", None)
            if original is None:
                return

            def probing(qkv_list, *a, **kw):
                # Record PER MODULE, not first-call-wins: the TokenRefiner
                # (transformer.py) fires before the first DiT block, so a
                # setdefault here reports the refiner's kernel and hides the
                # 50 blocks that actually dominate the step time.
                _kernel.setdefault("by_module", {}).setdefault(
                    label, offload.shared_state.get("_attention")
                )
                return original(qkv_list, *a, **kw)

            module.pay_attention = probing

        try:
            from models.minimax_h3 import sol_attention as S  # noqa: PLC0415
            from models.minimax_h3 import transformer as T  # noqa: PLC0415

            wrap(S, "sol_attention")   # the real DiT path for this model
            wrap(T, "transformer")     # TokenRefiner / any non-sol block
        except Exception as exc:  # noqa: BLE001 - never fail a bench over a probe
            _kernel["mode"] = f"probe_failed: {exc!r}"

    def probe_attention():
        """The DiT's kernel, not the TokenRefiner's: sol_attention.py is the
        module MiniMax H3's 50 blocks route through."""
        if "mode" in _kernel:          # probe_failed sentinel
            return _kernel["mode"]
        by_module = _kernel.get("by_module") or {}
        return by_module.get("sol_attention") or by_module.get("transformer")

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
    install_kernel_probe()
    if rank == 0:
        print(f"BENCH_ATTENTION requested={args.attention} installed={installed}", flush=True)

    runs = []
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        job = {
            "id": f"bench-{tag}-{seed}-r{rank}",
            "input": {
                "model_type": os.environ.get("WANGP_MODEL_TYPE", "minimax_h3_fl2va_pruned"),
                "profile": ACCEL_PROFILE,
                "settings": dict(
                    {
                        "prompt": PROMPT,
                        "resolution": RESOLUTION,
                        "video_length": args.frames,
                        "seed": seed,
                    },
                    **({"override_attention": override_attention} if override_attention else {}),
                ),
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
        run["attention_via"] = dict(_kernel.get("by_module") or {})
        # The tail (VAE decode + mux) is the metric for codec changes: denoise
        # variance on a shared host swamped a 29.6s-vs-32.9s control pair, but
        # decoding -> video_saved isolates exactly what an encoder swap moves.
        marks = run.get("phase_marks_s") or {}
        if marks.get("decoding") and marks.get("video_saved"):
            run["tail_s"] = round(marks["video_saved"] - marks["decoding"], 2)
        runs.append(run)
        if rank == 0:
            print("BENCH_RUN " + json.dumps(run), flush=True)
        # Every response references container-shared paths; drop it promptly.
        del response

    effective = runs[-1].get("effective_attention") if runs else None
    attention_ok = effective == args.attention
    summary = {"event": "usp_bench_summary", "tag": tag, "frames": args.frames,
               "compile": bool(args.compile), "config": args.config, "codec": args.codec,
               "vae_config": args.vae_config,
               "attention": args.attention,
               "effective_attention": effective, "attention_ok": attention_ok,
               "installed_attention": installed, "supported_attention": supported,
               # The knobs as wgp ACTUALLY received them. Without this the
               # config-freeze bug (see main()) silently mislabels a whole
               # matrix -- the tag says sage2/compile/fp8mix, the run is sdpa.
               "cli_args": list(live.cli_args), "model_config": live.model_config,
               "profile": str(args.profile), "world": world, "boot_s": boot_s, "runs": runs}
    if rank == 0:
        print("BENCH_SUMMARY " + json.dumps(summary), flush=True)
        _ship(summary)
    failed = any(r["status"] != "completed" for r in runs)
    if not attention_ok:
        # A leg whose kernel silently fell back is not a result, it is noise.
        print(f"BENCH_ATTENTION_MISMATCH requested={args.attention} "
              f"effective={effective}", file=sys.stderr, flush=True)
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
