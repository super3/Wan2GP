#!/usr/bin/env python3
"""make.py [key|film|all] [OUT_DIR]
key  -> key_1280x704.png / .jpg (q96): the H3 start = end frame (Pip's proud pose, mouth closed, eyes open)
film -> code_5s.mp4: 120 frames, 24 fps, 1280x720, synthesised stereo sound (rendered with 2 processes)."""
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
OUT = "/home/user/Wan2GP/outputs/style_protos/desert_chase"


def one(args):
    i, tmp = args
    import anim
    cv2.imwrite(os.path.join(tmp, f"f{i:04d}.png"), anim.frame(i)[:, :, ::-1])
    return i


def key(out):
    import anim
    im = anim.key()
    assert im.shape == (704, 1280, 3), im.shape
    cv2.imwrite(os.path.join(out, "key_1280x704.png"), im[:, :, ::-1])
    cv2.imwrite(os.path.join(out, "key_1280x704.jpg"), im[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 96])
    print("key ->", out)


def film(out):
    import anim
    import sound
    t0 = time.time()
    audio = sound.sound()
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        with Pool(2) as pool:
            for _ in pool.imap_unordered(one, [(i, tmp) for i in range(anim.N)]):
                pass
        wav = os.path.join(tmp, "a.wav")
        sf.write(wav, audio, sound.SR, subtype="PCM_24")
        dst = os.path.join(out, "code_5s.mp4")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-framerate", str(anim.FPS), "-i", os.path.join(tmp, "f%04d.png"), "-i", wav,
                        "-c:v", "libx264", "-crf", "15", "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-shortest",
                        "-movflags", "+faststart", dst], check=True)
    print("film ->", dst, f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = sys.argv[2] if len(sys.argv) > 2 else OUT
    os.makedirs(out, exist_ok=True)
    if what in ("key", "all"):
        key(out)
    if what in ("film", "all"):
        film(out)
