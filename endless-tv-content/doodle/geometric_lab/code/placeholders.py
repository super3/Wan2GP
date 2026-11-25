#!/usr/bin/env python3
"""placeholders.py OUT_DIR [--voice]: stand-in renders for testing assemble.py (and for the animatic): for each shot, hold the
start key, crossfade to the end key over the middle 20 %, hold the end key; exact frame count, 1280x704, 24 fps.
Silent by default; with --voice, a synthesised gibberish voice line is added to every other shot (to test the ducking)."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

DELIV = "/home/user/Wan2GP/outputs/shorts/geometric_lab"


def main():
    out = sys.argv[1]
    voice = "--voice" in sys.argv
    os.makedirs(out, exist_ok=True)
    spec = json.load(open(os.path.join(DELIV, "shots.json")))
    v = None
    if voice:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import snd
        v = snd.voice()[int(1.6 * snd.SR):int(3.8 * snd.SR)]
        v = v / (np.sqrt((v ** 2).mean()) + 1e-9) * 10 ** (-20 / 20)
    for k, s in enumerate(spec["shots"]):
        n = s["frames"]
        d = n / 24
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-loop", "1", "-framerate", "24", "-t", f"{d:.4f}", "-i", os.path.join(DELIV, s["start"]),
               "-loop", "1", "-framerate", "24", "-t", f"{d:.4f}", "-i", os.path.join(DELIV, s["end"])]
        fc = f"[0][1]xfade=transition=fade:duration={.2 * d:.4f}:offset={.4 * d:.4f},format=yuv420p[v]"
        with tempfile.TemporaryDirectory() as tmp:
            amap = []
            if v is not None and k % 2 == 0:
                a = np.zeros(int(d * 48000))
                a[48000:48000 + len(v)] = v[:len(a) - 48000]
                wav = os.path.join(tmp, "v.wav")
                sf.write(wav, np.stack([a, a], 1), 48000)
                cmd += ["-i", wav]
                amap = ["-map", "2:a", "-c:a", "aac", "-b:a", "160k"]
            cmd += ["-filter_complex", fc, "-map", "[v]", *amap, "-frames:v", str(n), "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-threads", "2",
                    "-r", "24", os.path.join(out, s["id"] + ".mp4")]
            subprocess.run(cmd, check=True)
        print(s["id"], n)


if __name__ == "__main__":
    main()
