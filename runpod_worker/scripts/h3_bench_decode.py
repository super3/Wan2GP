#!/usr/bin/env python3
"""Decode-only multi-GPU test: WanGP's INT8 H3 video VAE, seeded latents, decoded on GPU 0 alone and split across GPU 0 and a decode
worker process on GPU 1. No DiT, no USP, no model download but the VAE: only what multi-GPU decode touches.

Needs plugins/chunk_streaming from branch feature/multi-gpu-decode (h3_bench_boot.sh with H3_BENCH=decode builds the checkout).
Run it with WanGP's flags, e.g.
    python runpod_worker/scripts/h3_bench_decode.py --attention sage2 --profile 1 --verbose 1
Prints RESULT lines: kernel setup, load time, per size the single-GPU and split decode (full decode time, first chunk, time to first
frame with streaming), and whether the split output is bit-identical to the single-GPU one.

Environment: WANGP_ROOT (default: the checkout this script is in), H3_BENCH_OUT (default <WanGP>/outputs/h3_bench).
"""

import json
import os
import sys
import time
import traceback

ROOT = os.path.abspath(os.environ.get("WANGP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.abspath(os.environ.get("H3_BENCH_OUT") or os.path.join(ROOT, "outputs", "h3_bench"))
sys.path[:0] = [ROOT, os.path.join(ROOT, "plugins")]
os.chdir(ROOT)
CLI = sys.argv[1:]
FPS = 24.0
CELLS = [("480p, 5 s", 37, 30, 52), ("480p, 10 s", 72, 30, 52), ("720p, 5 s", 37, 44, 80), ("720p, 10 s", 72, 44, 80)]


def say(tag, payload):
    print("RESULT " + tag + " " + json.dumps(payload), flush=True)


def main():
    from shared.api import init

    session = init(root=ROOT, output_dir=os.path.join(OUT, "out_vae"), cli_args=CLI, console_output=False)
    wgp = session._ensure_runtime().module
    import torch
    from chunk_streaming import streaming

    from shared.kernels import int8_backend

    streaming.install_recipe_hook()
    from mmgp import offload
    from shared.attention import resolve_attention_mode

    # what a generation sets before the pipeline runs (wgp.generate_video): the VAE's attention reads it
    offload.shared_state["_attention"] = resolve_attention_mode(wgp.attention_mode)
    offload.shared_state["_attention_fallback"] = wgp.attention_mode
    offload.shared_state["_sage_v_scaling"] = False
    from models.minimax_h3 import minimax_h3_main as h3

    say("env", {"gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "torch": torch.__version__,
                "int8_kernels": getattr(wgp, "int8_kernels", None), "int8_backend_wgp": bool(int8_backend.wgp_enabled()), "cli": CLI,
                "attention": offload.shared_state.get("_attention")})
    started = time.monotonic()
    vae = h3._load_video_vae(h3.VIDEO_VAE_INT8_FILE, h3.VIDEO_VAE_SPLIT_INT8)  # as model_factory loads the INT8 VAE (qkv splitting on)
    vae = vae.eval().to("cuda:0")
    say("load", {"seconds": round(time.monotonic() - started, 1), "recipe": streaming._RECIPE_ATTR in vae.__dict__,
                 "int8_decoder": any(getattr(p, "_qtype", None) is not None for p in vae.decoder.parameters())})

    def latents(t, h, w):
        generator = torch.Generator(device="cuda:0").manual_seed(1000 + t * 7 + h)
        return torch.randn(1, 24, t, h, w, generator=generator, device="cuda:0").to(vae._model_dtype)

    def decode(rec, z):
        rec.reset()
        torch.cuda.synchronize()
        t0 = time.monotonic()
        with torch.inference_mode():
            out = vae.decode(z, "truncate")
        torch.cuda.synchronize()
        wall = time.monotonic() - t0
        ready = rec.chunk_ready_rel(t0)
        durations = [(c["frame_end"] - c["frame_start"]) / FPS for c in rec.chunks]
        ttff = streaming.no_buffer_start(ready, durations) if ready and len(ready) == len(durations) else None
        info = {"wall_s": round(wall, 3), "first_chunk_s": round(ready[0], 3) if ready else None,
                "ttff_s": round(ttff, 3) if ttff is not None else None, "chunks": len(ready), "plan": rec.decode_plan}
        return out, info

    single = {}
    rec = streaming.activate(store_frames=False, decode_devices=None)
    try:
        for label, t, h, w in CELLS:
            z = latents(t, h, w)
            decode(rec, z)  # warm-up: kernel autotuning for these shapes, on GPU 0
            out, info = decode(rec, z)
            single[label] = (z, out)
            say("single", dict(info, cell=label, frames=int(out.shape[2])))
    finally:
        streaming.deactivate()

    rec = streaming.activate(store_frames=False, decode_devices=["cuda:0", "cuda:1"])
    try:
        boot = time.monotonic()
        z, _ = single[CELLS[0][0]]
        _, info = decode(rec, z)  # starts the decode worker on GPU 1 (WanGP session + VAE load)
        say("worker_boot", dict(info, seconds=round(time.monotonic() - boot, 1)))
        for label, t, h, w in CELLS:
            z, reference = single[label]
            decode(rec, z)  # warm-up on the worker (its autotuning for these shapes)
            out, info = decode(rec, z)
            same = bool(torch.equal(out, reference))
            diff = None if same else float((out.float() - reference.float()).abs().max())
            say("split", dict(info, cell=label, bit_identical=same, max_abs_diff=diff))
    finally:
        streaming.deactivate()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        print("RESULT FAILED", flush=True)
        raise
    print("RESULT DONE", flush=True)
    os._exit(0)
