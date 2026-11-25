"""Placeholder renders (start key -> end key crossfades at each shot's exact frame count) and the animatic.

  short_animatic.py placeholders DIR [test]   write DIR/<id>.mp4; with "test", odd shots get a synthetic speech-like track
                                              and every third shot has no audio stream at all (exercises assemble.py's paths)
  short_animatic.py animatic                  write outputs/shorts/<slug>/animatic.mp4 (cards + keys on the real timeline + score)
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import short_shots as SH  # noqa: E402

SR = 48000


def speechlike(dur, seed):
    """Syllable bursts of a buzzy formant-filtered tone at about -18 dBFS RMS, from 0.6 s to 70% of the shot, plus one pop."""
    from scipy.signal import butter, sosfilt
    n = int(dur * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    f0 = 120 * (1 + .1 * np.sin(2 * np.pi * .7 * t))
    src = 2 * ((np.cumsum(f0) / SR) % 1) - 1
    y = sosfilt(butter(2, [500 / (SR / 2), 1600 / (SR / 2)], "band", output="sos"), src)
    syl = (np.sin(2 * np.pi * 4.2 * t + rng.uniform(0, 6)) > -.2).astype(float)
    win = ((t > .6) & (t < .7 * dur)).astype(float)
    env = sosfilt(butter(1, 30 / (SR / 2), output="sos"), syl * win)
    y = y * env
    y = y / (np.sqrt(np.mean(y[env > .5] ** 2)) + 1e-9) * 10 ** (-18 / 20)
    k = int(.82 * dur * SR)
    y[k:k + 2400] += np.sin(2 * np.pi * 600 * t[:2400]) * np.exp(-t[:2400] * 60) * .5
    return np.stack([y, y], 1).astype(np.float32)


def placeholders(out_dir, test=False):
    os.makedirs(out_dir, exist_ok=True)
    js = json.load(open(os.path.join(SH.OUT, "shots.json")))
    for k, s in enumerate(js["shots"]):
        n = s["frames"]
        d = n / 24
        a = os.path.join(SH.OUT, s["start"])
        b = os.path.join(SH.OUT, s["end"])
        out = os.path.join(out_dir, f"{s['id']}.mp4")
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-threads", "2", "-loop", "1", "-framerate", "24", "-t", f"{d:.4f}", "-i", a,
               "-loop", "1", "-framerate", "24", "-t", f"{d:.4f}", "-i", b]
        fc = f"[0:v][1:v]xfade=transition=fade:duration={d * .3:.4f}:offset={d * .35:.4f},format=yuv420p[v]"
        with tempfile.TemporaryDirectory() as tmp:
            amap = []
            if test and k % 3 != 1:
                wav = os.path.join(tmp, "a.wav")
                if k % 2 == 0:
                    sf.write(wav, speechlike(d, k), SR)
                else:
                    sf.write(wav, np.zeros((int(d * SR), 2), np.float32), SR)
                cmd += ["-i", wav]
                amap = ["-map", "2:a", "-c:a", "aac", "-b:a", "160k"]
            cmd += ["-filter_complex", fc, "-map", "[v]"] + amap + ["-frames:v", str(n), "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
                                                                   "-r", "24", "-t", f"{d:.4f}", out]
            subprocess.run(cmd, check=True)
    return out_dir


def animatic():
    with tempfile.TemporaryDirectory() as tmp:
        placeholders(tmp, test=False)
        subprocess.run([sys.executable, os.path.join(HERE, "assemble.py"), tmp, os.path.join(SH.OUT, "animatic.mp4")], check=True)


if __name__ == "__main__":
    if sys.argv[1] == "placeholders":
        placeholders(sys.argv[2], test=len(sys.argv) > 3 and sys.argv[3] == "test")
    elif sys.argv[1] == "animatic":
        animatic()
