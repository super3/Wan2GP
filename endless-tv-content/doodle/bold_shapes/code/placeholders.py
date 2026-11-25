"""Placeholder 'renders' for testing assemble.py (and for the animatic): each shot holds its start key, crossfades to its end key
in the middle 30%, and holds the end key, at the exact frame count.

modes: 'silent'  - silent audio track (as the brief suggests)
       'stress'  - babble-like 'speech' bursts in most shots (exercises the ducking), one clip with no audio stream, one clip at
                   1280x720, one 3 frames too long and one 2 frames too short (exercises conforming)
captions=True burns the shot id and its dialogue into the frames (animatic only)."""
import json
import os
import subprocess
import tempfile

import cv2
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

DELIV = "/home/user/Wan2GP/outputs/shorts/bold_shapes"
FPS, SR = 24, 48000
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def babble(dur, seed, t0=.25, t1=.75):
    n = int(dur * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    f0 = 260 + 40 * np.sin(2 * np.pi * 1.3 * t)
    x = np.sign(np.sin(2 * np.pi * np.cumsum(f0) / SR)) * .3 + rng.normal(0, .05, n)
    syl = (np.sin(2 * np.pi * 4.5 * t) > -.2).astype(float)
    win = ((t > t0 * dur) & (t < t1 * dur)).astype(float)
    k = np.ones(480) / 480
    env = np.convolve(syl * win, k, mode="same")
    return (x * env * .25).astype(np.float32)


def caption(img, sid, lines):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f1, f2 = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 26)
    d.rectangle([12, 12, 96, 44], fill=(0, 0, 0))
    d.text((20, 15), sid, font=f1, fill=(255, 255, 255))
    if lines:
        txt = "  /  ".join(f"{who}: {ln}" for who, ln in lines)
        w = d.textlength(txt, font=f2)
        d.rectangle([640 - w / 2 - 12, 650, 640 + w / 2 + 12, 690], fill=(0, 0, 0))
        d.text((640 - w / 2, 655), txt, font=f2, fill=(255, 230, 120))
    return np.asarray(im)


def make(out_dir, mode="silent", captions=False):
    import story
    os.makedirs(out_dir, exist_ok=True)
    shots = json.load(open(os.path.join(DELIV, "shots.json")))["shots"]
    lines = {s["id"]: s["lines"] for s in story.SHOTS}
    for k, s in enumerate(shots):
        a = cv2.imread(os.path.join(DELIV, s["start"]))[:, :, ::-1].astype(np.float32)
        b = cv2.imread(os.path.join(DELIV, s["end"]))[:, :, ::-1].astype(np.float32)
        n = s["frames"]
        size = "1280x704"
        if mode == "stress" and s["id"] == "s03":
            n += 3
        if mode == "stress" and s["id"] == "s06":
            n -= 2
        if mode == "stress" and s["id"] == "s05":
            size = "1280x720"
        path = os.path.join(out_dir, f"{s['id']}.mp4")
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "a.wav")
            dur = n / FPS
            audio = babble(dur, k) if mode == "stress" else np.zeros(int(dur * SR), np.float32)
            sf.write(wav, np.stack([audio, audio], 1), SR)
            no_audio = mode == "stress" and s["id"] == "s07"
            cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x704", "-r", str(FPS), "-i", "-"]
            if not no_audio:
                cmd += ["-i", wav]
            cmd += ["-vf", f"scale={size.replace('x', ':')}", "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
            cmd += [] if no_audio else ["-c:a", "aac", "-b:a", "128k"]
            cmd += ["-frames:v", str(n), path]
            pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            for i in range(n):
                u = min(max((i / max(n - 1, 1) - .35) / .3, 0), 1)
                u = u * u * (3 - 2 * u)
                img = (a * (1 - u) + b * u + .5).astype(np.uint8)
                if captions:
                    img = caption(img, s["id"], lines[s["id"]])
                pr.stdin.write(img.tobytes())
            pr.stdin.close()
            assert pr.wait() == 0
    return out_dir
