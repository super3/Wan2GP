#!/usr/bin/env python3
"""MiniMax H3 on 2 GPUs with the Multi-GPU (USP) plugin: per model combination, USP with the video decode split across both
GPUs (the chunk-streaming plugin's decode worker on GPU 1), then USP alone, so the two compare on one host.

Needs a WanGP checkout with plugins/usp_multi_gpu (this branch) and plugins/chunk_streaming (branch
feature/chunk-streaming-rebased for streaming; feature/multi-gpu-decode for the "mgd" mode). h3_bench_boot.sh builds one.
Rank 0 is this process: a headless WanGP session driving the plugin's own Start (helper process on GPU 1, NCCL group,
lockstep) and its generate_media hook. Every run streams (instrumented), so each yields the full render time and the earliest
stall-free playback start (first frame). Start it with the WanGP flags the helper must share:

    python runpod_worker/scripts/h3_bench_usp.py --attention sage2 --profile 1 --verbose 1

Environment:
    H3_BENCH_MODES          "mgd,usp" (default), "usp" or "mgd": USP + multi-GPU decode, USP alone
    H3_BENCH_DECODE_DEVICES decode devices of the "mgd" mode (default "cuda:0,cuda:1")
    H3_BENCH_COMBOS         combinations to run, e.g. "3,1" (default: all but VDN, which USP refuses)
    H3_BENCH_OUT            output folder (default <WanGP>/outputs/h3_bench)
    WANGP_ROOT              WanGP folder (default: the checkout this script is in)

Per run it reports the phases (queue, prompt encoding, denoising and seconds per step, audio decode, video decode and its
first clip, save + mux), checks the multi-GPU decode's video against USP alone's (decoded streams, and per finalized chunk
on the raw uint8 frames), and prints one RESULT line; h3_bench_tables.py turns the log into tables.
"""

import json
import os
import shutil
import sys
import time
import traceback

ROOT = os.path.abspath(os.environ.get("WANGP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "plugins")]
os.chdir(ROOT)
CLI = sys.argv[1:]
DECODE = os.environ.get("H3_BENCH_DECODE_DEVICES") or "cuda:0,cuda:1"  # multi-GPU decode (chunk_streaming decode worker)
MODES = os.environ.get("H3_BENCH_MODES", "mgd,usp").split(",")  # mgd: USP + multi-GPU decode; usp: USP alone
OUT = os.path.abspath(os.environ.get("H3_BENCH_OUT") or os.path.join(ROOT, "outputs", "h3_bench"))
VIDEO_DIR = os.path.join(OUT, "videos_usp")
CELLS = [("832x480", 124), ("832x480", 243), ("1280x704", 124), ("1280x704", 243)]
NO_LORA = {"activated_loras": [], "loras_multipliers": "", "guidance_phases": 1, "guidance_scale": 1.0}


def say(tag, payload):
    print("RESULT " + tag + " " + json.dumps(payload), flush=True)


def profile(path, **overrides):
    with open(os.path.join(ROOT, "profiles", path)) as handle:
        settings = json.load(handle)
    settings.pop("profile_priority", None)
    settings.update(overrides)
    return settings


COMBOS = [  # VDN (4) is left out: USP refuses VDN hybrid attention by design
    ("3", "minimax_h3_fl2va_pruned", "Lightx2v 4 Steps v0.1", profile("minimax_h3_fl2va/Lightx2v FL2V 4 Steps v0.1.json")),
    ("6", "minimax_h3_fl2va", "None (28 steps)", dict(NO_LORA, num_inference_steps=28, sample_solver="euler", flow_shift=12.0)),
    ("1", "minimax_h3_fl2va", "Larry v4 ckpt600", profile("minimax_h3/Larry v4 ema ckpt600 4 steps.json", num_inference_steps=6)),
    ("2", "minimax_h3_fl2va", "Lightx2v 8 Steps v1.0", profile("minimax_h3_fl2va/Lightx2v FL2V 8 Steps v1.0.json")),
    ("2b", "minimax_h3_fl2va_pruned", "Lightx2v 8 Steps v1.0", profile("minimax_h3_fl2va/Lightx2v FL2V 8 Steps v1.0.json")),
    ("1b", "minimax_h3_fl2va_pruned", "Larry v4 ckpt600", profile("minimax_h3/Larry v4 ema ckpt600 4 steps.json", num_inference_steps=6)),
    ("5", "minimax_h3_fl2va_hyperflow", "None (built in)", dict(NO_LORA, num_inference_steps=6)),
]


if os.environ.get("H3_BENCH_COMBOS"):
    _keep = os.environ["H3_BENCH_COMBOS"].split(",")
    COMBOS = [c for c in COMBOS if c[0] in _keep]


def stream_md5(path, kind):
    """md5 of the decoded video ("v") or audio ("a") stream: container metadata does not count."""
    import subprocess
    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", f"0:{kind}:0", "-f", "md5", "-"],
                          capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else f"error:{proc.stderr.strip()[:80]}"


def main():
    from shared.api import init

    session = init(root=ROOT, output_dir=os.path.join(OUT, "out_usp"), cli_args=CLI, console_output=False)
    wgp = session._ensure_runtime().module
    from chunk_streaming import streaming
    from usp_multi_gpu import launcher
    from usp_multi_gpu.plugin import UspMultiGpuPlugin

    say("gpus", {"gpus": launcher.gpus(), "cli": CLI})
    plugin = UspMultiGpuPlugin()
    plugin.args = wgp.args
    plugin.generate_media = wgp.generate_media
    plugin.release_model = wgp.release_model
    plugin.is_generation_in_progress = wgp.is_generation_in_progress
    plugin._set_wgp_global_func = lambda name, value: setattr(wgp, name, value)
    started = time.monotonic()
    status = list(plugin._start(2, ""))
    say("usp_start", {"seconds": round(time.monotonic() - started, 1), "status": status[-1][:400]})
    if launcher.running() is None:
        say("usp_not_started", {})
        return

    os.makedirs(VIDEO_DIR, exist_ok=True)
    prompt = json.load(open(os.path.join(ROOT, "defaults", "minimax_h3_fl2va.json")))["prompt"]

    # -- phase timing (rank 0): the pipeline's generate call and every DiT forward, on top of the plugin's own hooks --
    import functools

    import torch

    from models.minimax_h3 import transformer as h3_transformer
    from models.minimax_h3.pipeline import MiniMaxH3Pipeline

    marks = {"generate": [], "forward": []}
    mgd_frames = {}  # (combo, resolution, frames) -> the multi-GPU decode run's uint8 frames per chunk

    def compare_chunks(mine, theirs):
        """Per finalized chunk, the uint8 frames of two decodes of one generation: equal, or how far apart.
        With two decode devices clip i decodes on device i % 2 (odd clips on the GPU 1 worker)."""
        out = {"chunks": [len(mine), len(theirs)], "equal": [], "differ": []}
        for i, (a, b) in enumerate(zip(mine, theirs)):
            if a.shape != b.shape:
                out["differ"].append([i, "shape", list(a.shape), list(b.shape)])
            elif torch.equal(a, b):
                out["equal"].append(i)
            else:
                d = (a.to(torch.int16) - b.to(torch.int16)).abs()
                out["differ"].append([i, int(d.max()), round(float(d.float().mean()), 4), round(float((d > 0).float().mean()), 4)])
        return out

    def timed(cls, name, key, sync):
        original = getattr(cls, name)

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            span = [time.monotonic(), None]
            marks[key].append(span)
            try:
                return original(*args, **kwargs)
            finally:
                if sync and torch.cuda.is_available():
                    torch.cuda.synchronize()  # the step's GPU work, not just its launch
                span[1] = time.monotonic()

        setattr(cls, name, wrapper)

    timed(h3_transformer.MiniMaxH3Model, "forward", "forward", True)  # USP's sharded forward (installed by Start)
    timed(MiniMaxH3Pipeline, "generate", "generate", False)  # outside the plugin's lockstep gate

    def phases(t0, t_end, rec, segments):
        """Seconds per phase of one task, from rank 0's clocks."""
        r = lambda v: None if v is None else round(v, 2)
        gen, fwd, out = marks["generate"], marks["forward"], {}
        if not gen or gen[0][1] is None:
            return out
        g0, g1 = gen[0][0], gen[-1][1]
        out["queue_s"] = r(g0 - t0)  # settings, model load on a cold run, the lockstep task hand-off
        if fwd:
            f0, f1 = fwd[0][0], fwd[-1][1]
            out.update(prompt_s=r(f0 - g0), denoise_s=r(f1 - f0), forwards=len(fwd), step_s=r((f1 - f0) / len(fwd)))
            if rec.audio_start_t:
                out["denoise_to_decode_s"] = r(rec.audio_start_t - f1)
        if rec.audio_start_t and rec.audio_end_t:
            out["audio_decode_s"] = r(rec.audio_end_t - rec.audio_start_t)
        if rec.decode_start_t and rec.decode_end_t:
            out["video_decode_s"] = r(rec.decode_end_t - rec.decode_start_t)
            if rec.chunks:
                out["first_clip_s"] = r(rec.chunks[0]["t"] - rec.decode_start_t)
                out["clips"] = len(rec.chunks)
            out["pipeline_tail_s"] = r(g1 - rec.decode_end_t)
            done = [seg["t_done"] for seg in segments if seg.get("t_done")]
            if done and rec.chunks:
                out["first_segment_mux_s"] = r(done[0] - rec.chunks[0]["t"])  # live fMP4 segment, after its clip
                out["last_segment_after_decode_s"] = r(done[-1] - rec.decode_end_t)
        out["save_mux_s"] = r(t_end - g1)  # WanGP's full .mp4: encode, audio mux, metadata
        return out

    def run(combo, model_type, extra, resolution, frames, tag, mode):
        settings = {"model_type": model_type, "prompt": prompt, "resolution": resolution, "video_length": frames,
                    "seed": 42, "prompt_enhancer": "", "image_prompt_type": "", "video_prompt_type": "",
                    "audio_prompt_type": ""}
        settings.update(extra)
        # streaming stays on across runs, as the plugin's UI toggle keeps it: the decode worker then lives across generations
        # (switching decode devices, here between the two modes, closes it)
        # (feature/chunk-streaming-rebased has no decode_devices: the "usp" mode runs on it too)
        decode = {"decode_devices": DECODE if mode == "mgd" else None} if "mgd" in MODES else {}
        rec = streaming.activate(store_frames=True, live_dir=os.path.join(OUT, "stream_usp", f"{mode}-{combo}-{tag}-{resolution}-{frames}"),
                                 audio_first=True, **decode)
        rec.reset()
        out = {"combo": combo, "mode": mode, "resolution": resolution, "frames": frames, "tag": tag,
               "steps": settings.get("num_inference_steps")}
        try:
            marks["generate"].clear()
            marks["forward"].clear()
            t0 = time.monotonic()
            result = session.submit_task(settings).result()
            t_end = time.monotonic()
            wall = t_end - t0
            segments = rec.live.close() if rec.live is not None else []
            shift = 0.0 if rec.audio_first_effective else rec.audio_s
            ready = [s["t_done"] - t0 + shift for s in segments if s.get("t_done")]
            durations = [s["duration_s"] for s in segments if s.get("t_done")]
            out.update(ok=result.success, wall_s=round(wall, 2), errors=[str(e)[:400] for e in result.errors][:2],
                       audio_first=rec.audio_first_effective, segments=len(segments),
                       segments_with_audio=sum(1 for s in segments if s.get("has_audio")))
            if ready and len(ready) == len(durations):
                out["ttff_s"] = round(streaming.no_buffer_start(ready, durations), 2)
            try:
                out["phases"] = phases(t0, t_end, rec, segments)
            except Exception as exc:  # timing must never fail a run
                out["phases"] = {"error": repr(exc)[:200]}
            files = list(getattr(result, "generated_files", []) or [])
            if files and tag == "warm":
                dest = f"{VIDEO_DIR}/{mode}_{combo}_{resolution}_{frames}f.mp4"
                shutil.copy(files[0], dest)
                out["video"] = os.path.basename(dest)
                key = (combo, resolution, frames)
                if mode == "mgd":
                    mgd_frames[key] = list(rec.frames)
                elif key in mgd_frames:  # [chunk, max abs diff, mean abs diff, share of values that differ]
                    out["chunk_compare"] = compare_chunks(mgd_frames.pop(key), list(rec.frames))
                other = f"{VIDEO_DIR}/{'mgd' if mode == 'usp' else 'usp'}_{combo}_{resolution}_{frames}f.mp4"
                if os.path.exists(other):  # the same generation decoded on one GPU and split across two
                    out["same_frames_as_other_mode"] = stream_md5(other, "v") == stream_md5(dest, "v")
                    out["same_audio_as_other_mode"] = stream_md5(other, "a") == stream_md5(dest, "a")
        except Exception as exc:
            traceback.print_exc()
            out["error"] = repr(exc)[:400]
        plan = getattr(rec, "decode_plan", None)
        if isinstance(plan, dict):
            out["decode_plan"] = {k: (v if len(str(v)) < 300 else str(v)[:300]) for k, v in plan.items()}
        elif plan is not None:
            out["decode_plan"] = str(plan)[:300]
        lead = launcher.leader()
        out["usp_broken"] = getattr(lead, "broken", None) if lead is not None else "stopped"
        return out

    for combo, model_type, label, extra in COMBOS:
        # the cold run loads the model (and, with multi-GPU decode, starts the decode worker on GPU 1)
        cold = run(combo, model_type, extra, "832x480", 124, "cold", MODES[0])
        cold["profile"] = label
        say("cold", cold)
        broken = cold.get("usp_broken")
        if MODES[0] == "mgd" and not broken:
            # untimed: a new decode worker autotunes its INT8 kernels for each new clip shape; the cold run covered 480p
            prewarm = run(combo, model_type, extra, "1280x704", 124, "prewarm", "mgd")
            say("prewarm", {k: prewarm.get(k) for k in ("combo", "ok", "wall_s", "ttff_s", "phases", "decode_plan", "error")})
            broken = prewarm.get("usp_broken")
        for mode in MODES:
            for resolution, frames in CELLS:
                warm = run(combo, model_type, extra, resolution, frames, "warm", mode)
                warm["profile"] = label
                say("warm" if mode == "mgd" else "warm_usp", warm)
                broken = broken or warm.get("usp_broken")
        if broken:
            say("usp_broken_stopping", {"after": combo, "reason": broken})
            break
    streaming.deactivate()
    say("usp_stop", {"status": plugin._stop()[:200]})


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        print("RESULT FAILED", flush=True)
        raise
    print("RESULT DONE", flush=True)
    os._exit(0)
