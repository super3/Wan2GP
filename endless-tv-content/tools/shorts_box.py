#!/usr/bin/env python3
"""Cartoon shorts on MiniMax H3 FL2VA 33B: each shot animates from a START key frame to an END key frame (BF16 weights and text
encoder, sdpa attention, full BF16 video VAE, Larry v4 4-step LoRA, Audio Refinement Extra Phase), 1280x704.
Key frames come from runs on the Endless TV app named pack-<anything>: {"images": {"<key>": "<jpeg base64>", ...}}; every pack seen
is cached to /root/report/img/.  Jobs come from runs CONTROL_ID-1, -2, ... in order:
  {"cmd": "jobs", "jobs": [{"name": "bold_shapes__s01", "start": "bold_shapes/s01_a", "end": "bold_shapes/s01_b", "frames": 362,
                            "seed": 9001, "prompt": "..."}, ...]}
  {"cmd": "stop"}
After the last job it waits IDLE_MIN minutes for another command, then ends.  Clips: /root/report/<name>.mp4 (x264 CRF 14, audio copied).

    python shorts_box.py --attention sdpa --profile 2 --verbose 1"""
import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

ROOT, REPORT = "/root/Wan2GP", "/root/report"
CLI = sys.argv[1:]
CONTROL_URL = os.environ.get("CONTROL_URL", "https://wan2gp-production.up.railway.app/api/runs")
CONTROL_ID = os.environ.get("CONTROL_ID", "shorts")
IDLE_MIN = float(os.environ.get("IDLE_MIN", "15"))
UA = {"User-Agent": "curl/8.5.0"}  # clorecloud and Railway refuse urllib's own agent
OVERRIDES = "attention_mode='sdpa', transformer_quantization='bf16', text_encoder_quantization='bf16'"
LARRY = "minimax_h3/Larry v4 ema ckpt600 4 steps.json"
BASE = {"model_type": "minimax_h3_fl2va", "guidance_scale": 1.0, "guidance_phases": 1, "flow_shift": 12.0, "sample_solver": "euler",
        "custom_sigmas": "", "skip_steps_cache_type": "", "prompt_enhancer": "", "video_prompt_type": "", "audio_prompt_type": "",
        "denoising_strength": 1.0, "sliding_window_size": 481, "sliding_window_overlap": 18, "sliding_window_discard_last_frames": 0,
        "overrides": OVERRIDES, "config": ",bf16", "resolution": "1280x704", "image_prompt_type": "SE",
        "custom_settings": {"audio_refinement": "enabled"}}
RESULTS = {"base": dict(BASE), "clips": {}, "commands": []}
IMG = os.path.join(REPORT, "img")
r3 = lambda v: None if v is None else round(float(v), 3)  # noqa: E731


def anchors(frames):
    return ("For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n"
            f"For the target video, at {frames / 24:.2f} seconds into the target video, <Picture 2> (from [Shot 1]) is fully referenced.\n")


def say(tag, payload):
    print("RESULT " + tag + " " + json.dumps(payload)[:3000], flush=True)


def save():
    json.dump(RESULTS, open(os.path.join(REPORT, "results.json"), "w"), indent=1)


def runs():
    try:
        return json.loads(urllib.request.urlopen(urllib.request.Request(CONTROL_URL, headers=UA), timeout=120).read()).get("runs", [])
    except Exception:  # noqa: BLE001 - try again later
        return None


def img_path(key):
    return os.path.join(IMG, re.sub(r"[^A-Za-z0-9_.-]", "__", key) + ".jpg")


def image(key):
    """The key frame `key` as a local file; fetches and caches every pack it can see when one is missing."""
    path = img_path(key)
    if os.path.exists(path):
        return path
    for _ in range(20):
        for run in runs() or []:
            if str(run.get("id", "")).startswith("pack-"):
                for k, b64 in (run.get("images") or {}).items():
                    p = img_path(k)
                    if not os.path.exists(p):
                        open(p + ".tmp", "wb").write(base64.b64decode(b64))
                        os.replace(p + ".tmp", p)
        if os.path.exists(path):
            return path
        time.sleep(30)
    raise RuntimeError(f"no key frame {key}")


def generate(session, settings):
    """Run one task; (result, wall seconds)."""
    job = session.submit_task(settings)
    t0 = time.monotonic()
    threading.Thread(target=lambda: [None for _ in job.events.iter(timeout=2)], daemon=True).start()
    result = job.result()
    return result, time.monotonic() - t0


def job(session, spec, larry):
    name, frames, seed = spec["name"], int(spec["frames"]), int(spec.get("seed", 9001))
    record = RESULTS["clips"][name] = {"seed": seed, "frames": frames}
    try:
        settings = {**BASE, **larry, "prompt": anchors(frames) + spec["prompt"], "video_length": frames, "seed": seed,
                    "image_start": image(spec["start"]), "image_end": image(spec.get("end") or spec["start"])}
        result, wall = generate(session, settings)
        record.update(ok=result.success, wall_s=r3(wall), errors=[str(e)[:300] for e in result.errors][:2])
        if result.success:
            out = os.path.join(REPORT, f"{name}.mp4")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", result.generated_files[-1], "-c:v", "libx264", "-crf", "14", "-preset",
                            "slow", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", out + ".tmp.mp4"], check=True)
            os.replace(out + ".tmp.mp4", out)
            record["bytes"] = os.path.getsize(out)
    except Exception as error:  # noqa: BLE001 - one job failing leaves the others
        traceback.print_exc()
        record["error"] = repr(error)[:400]
    say(f"clip.{name}", record)
    save()


def main():
    os.makedirs(IMG, exist_ok=True)
    os.chdir(ROOT)
    sys.path[:0] = [ROOT]
    from shared.api import init

    larry = json.load(open(os.path.join(ROOT, "profiles", LARRY)))
    larry.pop("profile_priority", None)
    RESULTS["larry"] = larry
    session = init(root=ROOT, output_dir=os.path.join(ROOT, "outputs", "shorts"), cli_args=CLI, console_output=False)
    session._ensure_runtime().module.server_config["audio_output_codec"] = "aac_320"
    n, waited = 1, time.monotonic()
    while time.monotonic() - waited < IDLE_MIN * 60:
        order = next((r for r in runs() or [] if r.get("id") == f"{CONTROL_ID}-{n}"), None)
        if order is None:
            time.sleep(30)
            continue
        RESULTS["commands"].append({k: v for k, v in order.items() if k != "jobs"} | {"jobs": [j.get("name") for j in order.get("jobs") or []]})
        say(f"command.{n}", {k: v for k, v in order.items() if k != "jobs"} | {"jobs": len(order.get("jobs") or [])})
        if order.get("cmd") == "stop":
            break
        for spec in order.get("jobs") or []:
            job(session, spec, larry)
        n, waited = n + 1, time.monotonic()
        save()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - reported
        traceback.print_exc()
        RESULTS["error"] = repr(error)[:400]
        save()
    print("RESULT DONE " + json.dumps({"error": RESULTS.get("error")}), flush=True)
