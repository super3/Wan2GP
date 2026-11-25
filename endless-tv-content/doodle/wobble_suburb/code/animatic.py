"""animatic.py [OUT.mp4] [--placeholders DIR]: key frames on the real timeline (hold start, crossfade to end in each shot) with the
title/end cards and the music bed.  With --placeholders, instead writes one silent-picture placeholder render per shot
(<id>.mp4, exact frame count, 1280x704, 24 fps) carrying a gibberish test voice so assemble.py's ducking can be tested."""
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import soundfile as sf

A = "/home/user/Wan2GP/outputs/shorts/wobble_suburb"
W, H, FPS, SR = 1280, 704, 24, 48000


def load(p):
    return cv2.imread(os.path.join(A, p))[:, :, ::-1].astype(np.float32)


def shot_frames(s):
    a, b = load(s["start"]), load(s["end"])
    n = s["frames"]
    for i in range(n):
        u = np.clip((i / (n - 1) - .35) / .3, 0, 1)
        u = u * u * (3 - 2 * u)
        yield (a * (1 - u) + b * u + .5).astype(np.uint8)


def decode(path):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True,
                         check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def card_audio(path):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2)


def encoder(out, wav=None, fps=FPS):
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
    if wav:
        cmd += ["-i", wav, "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def animatic(out):
    doc = json.load(open(os.path.join(A, "shots.json")))
    title, end = decode(os.path.join(A, "title_card.mp4")), decode(os.path.join(A, "end_card.mp4"))
    bed, _ = sf.read(os.path.join(A, "music_bed.wav"), dtype="float32", always_2d=True)
    nfr = len(title) + sum(s["frames"] for s in doc["shots"]) + len(end)
    N = int(nfr / FPS * SR)
    mix = np.zeros((N, 2), np.float32)
    mix[:len(bed[:N])] += bed[:N] * .7
    ta = card_audio(os.path.join(A, "title_card.mp4"))
    ea = card_audio(os.path.join(A, "end_card.mp4"))
    mix[:len(ta)] += ta[:N]
    e0 = int((nfr - len(end)) / FPS * SR)
    mix[e0:e0 + len(ea)] += ea[:N - e0]
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        sf.write(wav, mix / max(1.0, np.max(np.abs(mix)) / .95), SR)
        p = encoder(out, wav)
        for f in title:
            p.stdin.write(f.tobytes())
        for s in doc["shots"]:
            for f in shot_frames(s):
                p.stdin.write(f.tobytes())
        for f in end:
            p.stdin.write(f.tobytes())
        p.stdin.close()
        assert p.wait() == 0
    print(out, nfr, "frames", nfr / FPS, "s")


def placeholders(d):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import snd
    os.makedirs(d, exist_ok=True)
    doc = json.load(open(os.path.join(A, "shots.json")))
    for k, s in enumerate(doc["shots"]):
        dur = s["frames"] / FPS
        x = np.zeros(int(dur * SR))
        v = snd.gibber([("b", "a", .16, 1.0), ("d", "e", .16, 1.1), ("", "o", .16, .95), ("s", "a", .2, 1.05)] * 4, 300 + 40 * (k % 3), seed=k)
        t0 = int(1.0 * SR)
        x[t0:t0 + len(v)] += v[:len(x) - t0] / (np.max(np.abs(v)) + 1e-9) * .3
        x += np.random.default_rng(k).normal(0, .003, len(x))  # room tone
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "a.wav")
            sf.write(wav, np.stack([x, x], 1), SR)
            p = encoder(os.path.join(d, s["id"] + ".mp4"), wav)
            for f in shot_frames(s):
                p.stdin.write(f.tobytes())
            p.stdin.close()
            assert p.wait() == 0
        print("placeholder", s["id"], s["frames"])


if __name__ == "__main__":
    if "--placeholders" in sys.argv:
        placeholders(sys.argv[sys.argv.index("--placeholders") + 1])
    else:
        animatic(sys.argv[1] if len(sys.argv) > 1 else os.path.join(A, "animatic.mp4"))
