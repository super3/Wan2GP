#!/usr/bin/env python3
"""placeholders.py OUT_DIR [--voice] [--short ID] [--long ID] [--noaudio ID]
Stand-in renders for testing assemble.py and for the animatic: <id>.mp4 at exactly `frames` (1280x704, 24 fps) holding the start key,
crossfading to the end key in the middle 30 %, holding the end key.  Audio: silence, or with --voice a few synthetic voice-like bursts
(to test the ducking).  --short / --long / --noaudio make one shot deliberately 5 frames short / 7 frames long / without audio."""
import argparse
import json
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

SR = 48000


def burst_track(seconds, seed):
    n = int(seconds * SR)
    out = np.zeros(n)
    rng = np.random.default_rng(seed)
    t0 = .6
    while t0 < seconds - 1.2:
        dur = rng.uniform(.5, 1.1)
        m = int(dur * SR)
        tt = np.arange(m) / SR
        f0 = rng.uniform(110, 220) * (1 + .1 * np.sin(2 * np.pi * 3 * tt))
        src = np.sign(np.sin(2 * np.pi * np.cumsum(f0) / SR)) * .3
        env = np.clip(tt / .03, 0, 1) * np.clip((dur - tt) / .05, 0, 1) * (.6 + .4 * np.abs(np.sin(2 * np.pi * 4 * tt)))
        out[int(t0 * SR):int(t0 * SR) + m] += src * env * .25
        t0 += dur + rng.uniform(1.0, 3.0)
    return np.stack([out, out], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--deliv", default="/home/user/Wan2GP/outputs/shorts/desert_chase")
    ap.add_argument("--voice", action="store_true")
    ap.add_argument("--short", default=None)
    ap.add_argument("--long", default=None)
    ap.add_argument("--noaudio", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    shots = json.load(open(os.path.join(a.deliv, "shots.json")))["shots"]
    for k, s in enumerate(shots):
        n = s["frames"] + (-5 if s["id"] == a.short else 7 if s["id"] == a.long else 0)
        D = n / 24
        off, xd = .35 * D, .3 * D
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y",
               "-loop", "1", "-t", f"{off + xd + .1:.4f}", "-framerate", "24", "-i", os.path.join(a.deliv, s["start"]),
               "-loop", "1", "-t", f"{D - off + .1:.4f}", "-framerate", "24", "-i", os.path.join(a.deliv, s["end"])]
        with tempfile.TemporaryDirectory() as tmp:
            if s["id"] != a.noaudio:
                wav = os.path.join(tmp, "a.wav")
                audio = burst_track(D, k) if a.voice else np.zeros((int(D * SR), 2), np.float32)
                sf.write(wav, audio, SR, subtype="PCM_16")
                cmd += ["-i", wav]
            fc = f"[0:v][1:v]xfade=transition=fade:duration={xd:.4f}:offset={off:.4f},format=yuv420p[v]"
            cmd += ["-filter_complex", fc, "-map", "[v]"]
            if s["id"] != a.noaudio:
                cmd += ["-map", "2:a", "-c:a", "aac", "-b:a", "128k"]
            cmd += ["-frames:v", str(n), "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-r", "24",
                    os.path.join(a.out, f"{s['id']}.mp4")]
            subprocess.run(cmd, check=True)
        print(s["id"], n, end=" ", flush=True)
    print()


if __name__ == "__main__":
    main()
