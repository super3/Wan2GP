"""Shared toolkit for the code-only cartoon prototypes: cairo drawing helpers, line boil, finishes, and the render-to-video loop.
Everything is deterministic (fixed seeds); no image files, no fonts, no samples."""
import math
import os
import subprocess
import tempfile
from multiprocessing import Pool

import cairo
import cv2
import numpy as np
import soundfile as sf

W, H, FPS = 1280, 720, 24


def hexc(h, a=1.0):
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, a)


def canvas(ss=2):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * ss, H * ss)
    ctx = cairo.Context(surf)
    ctx.scale(ss, ss)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    return surf, ctx


def to_rgb(surf, ss=2):
    a = np.ndarray((H * ss, W * ss, 4), np.uint8, surf.get_data())[:, :, :3][:, :, ::-1].astype(np.float32) / 255
    return cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)


def to8(img):
    return (np.clip(img, 0, 1) * 255 + .5).astype(np.uint8)


# ------------------------------------------------------------------------------------------------ paths
def cr_path(ctx, pts, closed=False, tension=1.0, move=True):
    p = [np.asarray(q, float) for q in pts]
    n = len(p)
    get = (lambda i: p[i % n]) if closed else (lambda i: p[min(max(i, 0), n - 1)])
    if move:
        ctx.move_to(*p[0])
    else:
        ctx.line_to(*p[0])
    for i in range(n if closed else n - 1):
        p0, p1, p2, p3 = get(i - 1), get(i), get(i + 1), get(i + 2)
        c1, c2 = p1 + (p2 - p0) * tension / 6, p2 - (p3 - p1) * tension / 6
        ctx.curve_to(c1[0], c1[1], c2[0], c2[1], p2[0], p2[1])
    if closed:
        ctx.close_path()


def poly(ctx, pts, closed=True):
    ctx.move_to(*pts[0])
    for q in pts[1:]:
        ctx.line_to(*q)
    if closed:
        ctx.close_path()


def ellipse(ctx, cx, cy, rx, ry, rot=0.0):
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(rot)
    ctx.scale(max(rx, 1e-3), max(ry, 1e-3))
    ctx.new_sub_path()
    ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.restore()


def fill(ctx, c, preserve=False):
    ctx.set_source_rgba(*c)
    ctx.fill_preserve() if preserve else ctx.fill()


def stroke(ctx, c, w, preserve=False):
    ctx.set_source_rgba(*c)
    ctx.set_line_width(w)
    ctx.stroke_preserve() if preserve else ctx.stroke()


def fs(ctx, fc, lc=None, w=0.0):
    if lc is None:
        fill(ctx, fc)
    else:
        fill(ctx, fc, True)
        stroke(ctx, lc, w)


def taper(ctx, pts, w0, w1, color):
    p = np.asarray(pts, float)
    d = np.gradient(p, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    w = np.linspace(w0, w1, len(p))[:, None] / 2
    cr_path(ctx, list(p + nrm * w) + list((p - nrm * w)[::-1]), closed=True, tension=.7)
    fill(ctx, color)


def bez(p0, p1, p2, p3, n=24):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(q, float) for q in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def boil(pts, step, amp, seed):
    """Hand-drawn line boil: the same shape, redrawn with a new small jitter every `step` (deterministic)."""
    p = np.asarray(pts, float)
    rng = np.random.default_rng(seed * 7919 + step)
    return p + rng.normal(0, amp, p.shape)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------------------------------------ finishes
_PAPER = {}


def paper(img, amount=.05, seed=1):
    """Static paper fibre texture (the same every frame, like a painted background card)."""
    key = (img.shape, seed)
    if key not in _PAPER:
        rng = np.random.default_rng(seed)
        n = rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32)
        n = cv2.resize(cv2.GaussianBlur(n, (0, 0), 1.2), (W, H))
        f = rng.normal(0, 1, (H, W)).astype(np.float32)
        f = cv2.GaussianBlur(f, (0, 0), sigmaX=6, sigmaY=.6)
        _PAPER[key] = (n * .6 + f * .9)[..., None]
    return img * (1 + _PAPER[key] * amount)


def grain(img, i, amount=.012):
    rng = np.random.default_rng(10000 + i)
    g = cv2.resize(rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32), (W, H))
    return img + g[..., None] * amount


def vignette(img, k=.3, cx=.5, cy=.5):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = ((xx / W - cx) ** 2 + (yy / H - cy) ** 2) * 2
    return img * (1 - k * np.clip(v, 0, 1))[..., None]


def broadcast(img, i, bleed=1.6, scan=.02):
    """A soft 90s-TV transfer: chroma bleeds sideways, a hint of scanlines and a tiny line wobble."""
    yuv = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_RGB2YCrCb)
    yuv[:, :, 1:] = cv2.GaussianBlur(yuv[:, :, 1:], (0, 0), sigmaX=bleed, sigmaY=.3)
    img = cv2.cvtColor(yuv, cv2.COLOR_YCrCb2RGB)
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=.45, sigmaY=.2)
    rows = (1 - scan * (np.arange(H) % 2))[:, None, None]
    return img * rows


def bloom(img, thresh=.75, amount=.5, tint=(1, .9, .8)):
    lum = img @ np.array([.299, .587, .114], np.float32)
    b = img * np.clip((lum - thresh) / (1 - thresh), 0, 1)[..., None]
    b = cv2.GaussianBlur(b, (0, 0), W / 80) * .6 + cv2.GaussianBlur(b, (0, 0), W / 25) * .4
    return 1 - (1 - img) * (1 - b * amount * np.array(tint, np.float32))


# ------------------------------------------------------------------------------------------------ render
def _one(args):
    module, i, out = args
    import importlib
    m = importlib.import_module(module)
    cv2.imwrite(os.path.join(out, f"f{i:04d}.png"), m.frame(i)[:, :, ::-1])
    return i


def render(module, nframes, audio, sr, out_path, jobs=4):
    with tempfile.TemporaryDirectory() as tmp:
        with Pool(jobs) as pool:
            for _ in pool.imap_unordered(_one, [(module, i, tmp) for i in range(nframes)]):
                pass
        wav = os.path.join(tmp, "a.wav")
        sf.write(wav, audio, sr, subtype="PCM_24")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-framerate", str(FPS), "-i", os.path.join(tmp, "f%04d.png"), "-i", wav,
                        "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-shortest",
                        "-movflags", "+faststart", out_path], check=True)
    return out_path
