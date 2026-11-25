#!/usr/bin/env python3
"""make.py: render the geometric_lab deliverables (key frame PNG/JPG + 5 s code-only mp4 with synthesised sound).
Uses at most 2 worker processes (the machine is shared)."""
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cv2  # noqa: E402
import soundfile as sf  # noqa: E402

import geo  # noqa: E402
import snd  # noqa: E402

OUT = "/home/user/Wan2GP/outputs/style_protos/geometric_lab"
JOBS = 2


def _one(args):
    i, tmp = args
    cv2.imwrite(os.path.join(tmp, f"f{i:04d}.png"), geo.frame(i)[:, :, ::-1])
    return i


def key():
    img = geo.key_frame()                     # frame 0 of the animation, 1280x720
    k = img[8:712]                            # -> exactly 1280x704 (nothing important within 8 px of top/bottom)
    assert k.shape[:2] == (704, 1280)
    cv2.imwrite(os.path.join(OUT, "key_1280x704.png"), k[:, :, ::-1])
    cv2.imwrite(os.path.join(OUT, "key_1280x704.jpg"), k[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 96])


def video():
    t0 = time.time()
    audio = snd.sound()
    with tempfile.TemporaryDirectory(dir=os.path.join(HERE)) as tmp:
        with Pool(JOBS) as pool:
            for _ in pool.imap_unordered(_one, [(i, tmp) for i in range(geo.N)]):
                pass
        wav = os.path.join(tmp, "a.wav")
        sf.write(wav, audio, snd.SR, subtype="PCM_24")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-framerate", str(geo.FPS), "-i", os.path.join(tmp, "f%04d.png"), "-i", wav,
                        "-c:v", "libx264", "-crf", "15", "-preset", "slow", "-tune", "animation", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-b:a", "256k", "-t", str(geo.DUR), "-movflags", "+faststart", os.path.join(OUT, "code_5s.mp4")], check=True)
    print(f"video done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1:] or ["key", "video"]
    if "key" in what:
        key()
    if "video" in what:
        video()
