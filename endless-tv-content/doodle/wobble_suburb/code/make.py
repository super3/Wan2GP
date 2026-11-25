#!/usr/bin/env python3
"""make.py [key|code|all]: build the style-study deliverables (key frame PNG/JPG and the 5 s code-only shot with sound)."""
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import cv2
import soundfile as sf

OUT = "/home/user/Wan2GP/outputs/style_protos/wobble_suburb"
HERE = os.path.dirname(os.path.abspath(__file__))


def _one(args):
    mod, i, tmp = args
    import importlib
    m = importlib.import_module(mod)
    cv2.imwrite(os.path.join(tmp, f"f{i:04d}.png"), m.frame(i)[:, :, ::-1])
    return i


def render(mod, n, audio, out, jobs=2, fps=24):
    with tempfile.TemporaryDirectory() as tmp:
        with Pool(jobs) as pool:
            list(pool.imap_unordered(_one, [(mod, i, tmp) for i in range(n)]))
        wav = os.path.join(tmp, "a.wav")
        sf.write(wav, audio, 48000, subtype="PCM_24")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-framerate", str(fps), "-i", os.path.join(tmp, "f%04d.png"), "-i", wav,
                        "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
                        "-shortest", "-movflags", "+faststart", out], check=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT, exist_ok=True)
    if what in ("key", "all"):
        subprocess.run([sys.executable, os.path.join(HERE, "key.py"), os.path.join(OUT, "key_1280x704.png")], check=True)
        im = cv2.imread(os.path.join(OUT, "key_1280x704.png"))
        assert im.shape == (704, 1280, 3)
        cv2.imwrite(os.path.join(OUT, "key_1280x704.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 96])
        os.replace(os.path.join(OUT, "key_1280x704_face.png"), os.path.join(HERE, "key_face.png"))
    if what in ("code", "all"):
        import anim
        import snd
        audio = snd.sound_code_shot(anim.DUR)
        sf.write(os.path.join(HERE, "code_shot.wav"), audio, 48000)
        render("anim", anim.N, audio, os.path.join(OUT, "code_5s.mp4"))
    print("done", what)
