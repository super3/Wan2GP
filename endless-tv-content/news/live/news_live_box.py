#!/usr/bin/env python3
"""Riverside News, Wednesday November 25, rendered live in one H3 Live session (WanGP feature/endless-tv, plugins/h3_live).

The real-time settings measured on an RTX PRO 6000 (preload_box.py, v2020_box.py): MiniMax H3 FL2VA Pruned pi-H3 DWD 2-Step, 832x480,
124-frame clips (5.17 s at 24 fps), every clip starting and ending on the anchor still (<Picture 1> at 0.00 s, <Picture 2> at 5.17 s),
image encodes cached, steady VRAM preload, NVENC.

1. Warm-up: one clip whose prompt encoding also encodes every line of the segment with the anchor still, as the text encoder sees it:
   the session starts with all its text embeddings computed (no text encoder work between clips).
2. The session: the airing (the Today opening, the body, the Today sign-off), then the Tonight opening and sign-off. As each clip is
   ready it is copied to /root/report/<name>.mp4 and announced (RESULT clip.<name>), so it can be fetched and put on air at once.
3. Timing: cycle per clip and speed against real time. The box then waits FINAL_WAIT_S for the last clips to be fetched.
Prints RESULT lines; writes /root/report/results.json.

    python news_live_box.py --attention sage2 --profile 1 --verbose 1"""
import json
import os
import shutil
import sys
import threading
import time
import traceback

ROOT, REPORT = "/root/Wan2GP", "/root/report"
CLI = sys.argv[1:]
MODEL = "minimax_h3_fl2va_pruned_pi_dwd_2step"
HERE = os.path.join(ROOT, "endless-tv-content", "news", "live")  # checked out from feature/endless-tv-content (PIH3_FILES_FROM)
IMAGE = os.path.join(HERE, "news_anchor_832x480.png")  # the Dana Reyes anchor still, centre-cropped and resized to 832x480
LINES = json.load(open(os.path.join(HERE, "news_lines.json")))  # split_lines.py's 5-second lines, a few breaks fixed by hand
END_S = "5.17"  # 124 frames at 24 fps
ANCHORS = ("For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n"
           f"For the target video, at {END_S} seconds into the target video, <Picture 2> (from [Shot 1]) is fully referenced.\n")
KEEP = ("Continue directly from <Picture 1>, preserving the news anchor, her shoulder-length natural curls, the emerald green blazer "
        "over a cream blouse, the small pearl earrings, the papers on the anchor desk and the studio behind her with the city skyline "
        "on the video wall.")
SOUND = "overall_soundscape: Quiet television studio room tone and her clear, close-miked voice.\nnon_diegetic_music: N/A"
RESULTS = {"clips": {}}


def prompt(line):
    return (ANCHORS + f"integrated_multimodal_description: [Shot 1] {KEEP} In a static medium shot she looks straight into the camera "
            f"with a calm, composed expression and says (S1) <d>[English] {line}</d> She then settles back into the same composed pose, "
            f"matching <Picture 2> at the end. The camera does not move.\n{SOUND}")


def plan():
    """(name, line) in render order: the airing first (news_001 ...), then the Tonight versions of the opening and sign-off."""
    airing = LINES["opening"]["Today"] + LINES["body"] + LINES["signoff"]["Today"]
    items = [(f"news_{i:03d}", line) for i, line in enumerate(airing, 1)]
    items += [(f"tonight_open_{i}", line) for i, line in enumerate(LINES["opening"]["Tonight"], 1)]
    items += [(f"tonight_close_{i}", line) for i, line in enumerate(LINES["signoff"]["Tonight"], 1)]
    return items


def r3(x):
    return round(x, 3) if isinstance(x, (int, float)) else x


def say(tag, payload):
    print("RESULT " + tag + " " + json.dumps(payload)[:3500], flush=True)


def save():
    json.dump(RESULTS, open(os.path.join(REPORT, "results.json"), "w"), indent=1)


def task(text, seed):
    return {"model_type": MODEL, "prompt": text, "resolution": "832x480", "video_length": 124, "seed": seed, "prompt_enhancer": "",
            "image_prompt_type": "SE", "image_start": IMAGE, "image_end": IMAGE, "video_prompt_type": "", "audio_prompt_type": ""}


def publish(live, names, stop):
    """Copies each finished clip of the running session to the report folder as soon as it is ready."""
    done = set()
    while True:
        session = live.STATE.get("session") or live.STATE.get("last_session")
        for record in list(getattr(session, "records", []) or []):
            index = record.get("clip")
            if index in done or index is None or index >= len(names):
                continue
            done.add(index)
            name = names[index]
            if record.get("error") or not record.get("mp4"):
                say(f"clip.{name}", {"ok": False, "error": str(record.get("error"))[:300]})
                continue
            shutil.copyfile(os.path.join(session.dir, record["mp4"]), os.path.join(REPORT, f"{name}.mp4"))
            RESULTS["clips"][name] = {k: record.get(k) for k in ("clip", "seed", "generate_s", "ready_at_s", "seconds", "encode_s")}
            say(f"clip.{name}", {"ok": True, **RESULTS["clips"][name]})
        if stop.is_set() and (session is None or len(done) >= len(getattr(session, "records", []) or [])):
            return
        time.sleep(0.5)


def main():
    os.makedirs(REPORT, exist_ok=True)
    os.chdir(ROOT)
    sys.path[:0] = [ROOT, os.path.join(ROOT, "plugins")]
    from shared.api import init

    session = init(root=ROOT, output_dir=os.path.join(ROOT, "outputs", "news"), cli_args=CLI, console_output=False)
    session._ensure_runtime()
    from models.minimax_h3.pipeline import MiniMaxH3Pipeline
    from h3_live import live

    RESULTS["install_problems"] = live.install(MiniMaxH3Pipeline)
    items = plan()
    names, prompts = [n for n, _ in items], [prompt(line) for _, line in items]
    RESULTS["plan"] = [{"name": n, "line": line} for n, line in items]
    say("plan", {"clips": len(items), "airing": sum(n.startswith("news_") for n in names)})

    # 1. warm-up: the first encoding shown the anchor still also encodes every other line with it
    original = MiniMaxH3Pipeline._encode_prompt
    pending, encodes, pre = list(prompts), [], {}

    def encode(self, text, presentation):
        started = time.monotonic()
        if presentation and pending:
            self.text_encoder_cache.max_size_bytes = max(self.text_encoder_cache.max_size_bytes, 24 << 30)  # room for all of them
            for other in list(dict.fromkeys(pending)):
                original(self, other, presentation)
            pre.update(prompts=len(set(pending)), s=r3(time.monotonic() - started))
            pending.clear()
            started = time.monotonic()
        try:
            return original(self, text, presentation)
        finally:
            encodes.append(r3(time.monotonic() - started))

    MiniMaxH3Pipeline._encode_prompt = encode
    live.configure(None)
    t0 = time.monotonic()
    result = session.submit_task(task(prompts[0], seed=1)).result()
    RESULTS["warmup"] = {"ok": result.success, "wall_s": r3(time.monotonic() - t0), "pre_encode": dict(pre),
                         "errors": [str(e)[:300] for e in result.errors][:2]}
    say("warmup", RESULTS["warmup"])
    save()

    # 2. the session, each clip published as it is ready
    encodes.clear()
    stop = threading.Event()
    publisher = threading.Thread(target=publish, args=(live, names, stop), daemon=True)
    publisher.start()
    live.configure(live.LiveSettings(enabled=True, prompts=prompts, clips=len(prompts), encoder="auto", pre_encode_prompts=False,
                                     steady_vram_preload=True, cache_image_encodes=True))
    t0 = time.monotonic()
    say("session_start", {"clips": len(prompts), "utc": time.strftime("%H:%M:%S", time.gmtime())})
    result = session.submit_task(task(prompts[0], seed=42)).result()
    wall = time.monotonic() - t0
    live.configure(None)
    stop.set()
    publisher.join(timeout=120)
    summary = live.STATE["last_session"].summary()
    slow_encodes = [e for e in encodes if e > 0.05]
    RESULTS["session"] = {"ok": result.success, "wall_s": r3(wall), "errors": [str(e)[:300] for e in result.errors][:2],
                          "encode_calls": len(encodes), "slow_encode_calls": slow_encodes[:10],
                          **{k: summary.get(k) for k in ("clips", "errors", "cycle_s", "realtime_x", "generate_s", "generation_x",
                                                         "first_clip_ready_s", "encoder", "dir")}}
    say("session", RESULTS["session"])
    save()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - reported
        traceback.print_exc()
        RESULTS["error"] = repr(error)[:400]
        save()
    print("RESULT DONE " + json.dumps({"error": RESULTS.get("error")}), flush=True)
    time.sleep(float(os.environ.get("FINAL_WAIT_S", "300")))  # time to fetch the last clips before the box is destroyed
