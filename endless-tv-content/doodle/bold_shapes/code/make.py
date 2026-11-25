#!/usr/bin/env python3
"""make.py: build every bold_shapes deliverable (key frame PNG/JPG, 5 s code-only shot with sound).
Usage: python make.py [key|video|all]   (renders with at most 2 processes)"""
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

import cv2
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import draw  # noqa: E402

OUT = "/home/user/Wan2GP/outputs/style_protos/bold_shapes"


def _one(args):
    i, tmp = args
    cv2.imwrite(os.path.join(tmp, f"f{i:04d}.png"), draw.frame(i)[:, :, ::-1])
    return i


def key():
    k = draw.key_frame()
    assert k.shape == (704, 1280, 3), k.shape
    cv2.imwrite(os.path.join(OUT, "key_1280x704.png"), k[:, :, ::-1])
    cv2.imwrite(os.path.join(OUT, "key_1280x704.jpg"), k[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 96])


def video():
    import sound
    t0 = time.time()
    audio = sound.sound()
    t1 = time.time()
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        with Pool(2) as pool:
            for _ in pool.imap_unordered(_one, [(i, tmp) for i in range(draw.N)]):
                pass
        wav = os.path.join(tmp, "a.wav")
        sf.write(wav, audio, 48000, subtype="PCM_24")
        sf.write(os.path.join(HERE, "audio.wav"), audio, 48000, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-framerate", str(draw.FPS), "-i", os.path.join(tmp, "f%04d.png"), "-i", wav,
                        "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-tune", "animation", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "256k", "-t", "5", "-movflags", "+faststart", os.path.join(OUT, "code_5s.mp4")], check=True)
    print(f"sound {t1 - t0:.1f}s, picture {time.time() - t1:.1f}s")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT, exist_ok=True)
    if what in ("key", "all"):
        key()
    if what in ("video", "all"):
        video()
