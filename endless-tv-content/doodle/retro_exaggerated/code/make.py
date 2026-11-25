#!/usr/bin/env python3
"""make.py [key|video|all]: render the key frame (PNG + JPG at 1280x704), the 5 s code-only shot with sound, and a contact sheet.
Uses at most 2 processes."""
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

import cv2
import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bertram as BT  # noqa: E402
import sound as SND  # noqa: E402

OUT = "/home/user/Wan2GP/outputs/style_protos/retro_exaggerated"


def _one(args):
    i, d = args
    cv2.imwrite(os.path.join(d, f"f{i:04d}.png"), BT.frame(i)[:, :, ::-1])
    return i


def key():
    k = BT.key_image(ss=3)
    assert k.shape == (704, 1280, 3)
    cv2.imwrite(os.path.join(OUT, "key_1280x704.png"), k[:, :, ::-1])
    cv2.imwrite(os.path.join(OUT, "key_1280x704.jpg"), k[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 96])
    print("key written")


def video():
    t0 = time.time()
    audio = SND.sound()
    with tempfile.TemporaryDirectory() as tmp:
        with Pool(2) as pool:
            for _ in pool.imap_unordered(_one, [(i, tmp) for i in range(BT.N)], chunksize=4):
                pass
        wav = os.path.join(tmp, "a.wav")
        sf.write(wav, audio, SND.SR, subtype="PCM_24")
        out = os.path.join(OUT, "code_5s.mp4")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-framerate", str(BT.FPS), "-i", os.path.join(tmp, "f%04d.png"), "-i", wav,
                        "-c:v", "libx264", "-crf", "15", "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
                        "-t", str(BT.DUR), "-movflags", "+faststart", out], check=True)
        # contact sheet: every 8th frame
        ims = [cv2.resize(cv2.imread(os.path.join(tmp, f"f{i:04d}.png")), (320, 180), interpolation=cv2.INTER_AREA) for i in range(0, BT.N, 8)]
        while len(ims) % 5:
            ims.append(np.zeros_like(ims[0]))
        sheet = np.vstack([np.hstack(ims[r * 5:(r + 1) * 5]) for r in range(len(ims) // 5)])
        cv2.imwrite(os.path.join(HERE, "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"video written in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT, exist_ok=True)
    if what in ("key", "all"):
        key()
    if what in ("video", "all"):
        video()
