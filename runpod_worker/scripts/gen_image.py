#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a still with a WanGP image model (Krea 2 Turbo by default).

The RunPod worker cannot do this: schema.py:1656-1662 pins ``image_mode`` to 0
because that endpoint exists to produce video, and a non-zero image_mode there
would silently bill a video GPU for a still. This script therefore drives
``shared.api`` directly, outside the worker's request path.

    python3 runpod_worker/scripts/gen_image.py \
        --prompt "..." --resolution 1664x960 --out /out/board.png

Written for the storyboard-grid pipeline: one image holding a 2x2 grid of
keyframes, cropped by grid_to_film.py into four independent clip start frames.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = os.environ.get("WANGP_ROOT") or "/opt/wangp"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model-type", default="krea2_turbo")
    ap.add_argument("--resolution", default="1664x960")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--out", default="/out/board.png")
    args = ap.parse_args()

    # Pin the model BEFORE runpod_worker.config is imported: CONFIG freezes at
    # import (see the note in usp_bench.py).
    os.environ["WANGP_MODEL_TYPE"] = args.model_type
    os.environ.setdefault("WANGP_ATTENTION", "sdpa")
    os.environ["WANGP_CLI_ARGS"] = "--attention sdpa --profile 4 --verbose 1"
    outdir = Path(args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)
    os.environ["WANGP_OUTPUT_DIR"] = str(outdir)

    from runpod_worker import config as C  # noqa: PLC0415
    C.reload_config()
    cfg_path = C.ensure_wgp_config(C.CONFIG.cli_args)
    print(f"wgp config: {cfg_path}", flush=True)

    from shared import api  # noqa: PLC0415
    t0 = time.time()
    session = api.init(root=ROOT, config_path=str(cfg_path), output_dir=str(outdir),
                       cli_args=list(C.CONFIG.cli_args), console_output=True)
    print(f"session ready in {time.time()-t0:.1f}s", flush=True)

    settings = {
        "model_type": args.model_type,
        "prompt": args.prompt,
        # image_mode 1 is what makes this a still rather than a 1-frame video.
        "image_mode": 1,
        "resolution": args.resolution,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "batch_size": 1,
    }
    if args.seed >= 0:
        settings["seed"] = args.seed

    t1 = time.time()
    result = session.run_task(settings)
    print(f"generated in {time.time()-t1:.1f}s", flush=True)

    files = [Path(a.path) for a in (getattr(result, "artifacts", None) or [])
             if getattr(a, "path", None)]
    if not files:  # fall back to whatever landed in the output dir
        files = sorted(outdir.glob("*.png")) + sorted(outdir.glob("*.jpg"))
    if not files:
        print("no image produced", file=sys.stderr)
        return 1
    src = files[-1]
    if src.resolve() != Path(args.out).resolve():
        shutil.copy(src, args.out)
    print(f"IMAGE {args.out} {Path(args.out).stat().st_size} bytes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
