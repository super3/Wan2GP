"""cards.py: the title card (3 s) and end card (2 s) for "One Small Passenger", 1280x704 @ 24 fps, each with a synthesised sting.
Mid-century card: a teal field with a turning sunburst, the brass floor dial, hand-set lettering (DejaVu Sans Bold, outlined, with a
flat offset shadow), Mr. Hoist's head and Mrs. Dawdle popping on with overshoot, animated on twos."""
import math
import os
import subprocess
import sys
import tempfile

import cairo
import numpy as np
import soundfile as sf

import geo
import short
import snd
import synth as S
from geo import INK, ell, flat, hx, ink, line, rpoly

W, H, SS, FPS = 1280, 704, 2, 24
OUT = "/home/user/Wan2GP/outputs/shorts/geometric_lab"
FONT = "DejaVu Sans"


def pop(t, t0, d=.25):
    """0 -> overshoot -> 1 scale-in, held (sampled on twos by the caller)."""
    x = (t - t0) / d
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    c1 = 2.2                                              # ease-out-back: overshoots ~12 %, then settles
    return 1 + (c1 + 1) * (x - 1) ** 3 + c1 * (x - 1) ** 2


def snp(s):
    s.flush()
    a = np.ndarray((s.get_height(), s.get_width(), 4), np.uint8, s.get_data())
    return a[:, :, [2, 1, 0, 3]].astype(np.float32) / 255


def ctx_new():
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * SS, H * SS)
    c = cairo.Context(s)
    c.scale(SS, SS)
    c.set_line_join(cairo.LINE_JOIN_ROUND)
    c.set_line_cap(cairo.LINE_CAP_ROUND)
    return s, c


def backdrop(c, t):
    c.rectangle(0, 0, W, H)
    flat(c, hx("#2a6c7c"))
    cx, cy = 640, 380
    rot = t * .12
    for k in range(24):                                   # sunburst
        if k % 2:
            continue
        a0, a1 = rot + k * math.pi / 12, rot + (k + 1) * math.pi / 12
        c.move_to(cx, cy)
        c.line_to(cx + 1600 * math.cos(a0), cy + 1600 * math.sin(a0))
        c.line_to(cx + 1600 * math.cos(a1), cy + 1600 * math.sin(a1))
        c.close_path()
    flat(c, hx("#2f7888"))
    for k in range(3):                                    # off-register sparkles
        for j in range(5):
            x, y = 120 + j * 260 + (k % 2) * 130, 90 + k * 250
            pts = [(x, y - 16), (x + 4, y - 4), (x + 16, y), (x + 4, y + 4), (x, y + 16), (x - 4, y + 4), (x - 16, y), (x - 4, y - 4)]
            c.save()
            c.translate(3, 2)
            rpoly(c, pts, 1)
            flat(c, hx("#4fa6a6"))
            c.restore()
            rpoly(c, pts, 1)
            c.set_source_rgba(*hx("#1e5560"))
            c.set_line_width(1.8)
            c.stroke()
    c.rectangle(0, H - 70, W, 70)                          # checkerboard floor strip
    flat(c, hx("#efe0c0"))
    for k in range(0, 40):
        if k % 2 == 0:
            c.rectangle(k * 35, H - 70, 35, 35)
            c.rectangle(k * 35 + 35, H - 35, 35, 35)
    flat(c, hx("#d9916c"))
    line(c, [(0, H - 70), (W, H - 70)], 3.0)


def dial(c, x, y, r, needle):
    c.move_to(x - r, y)
    c.arc(x, y, r, math.pi, 2 * math.pi)
    c.close_path()
    ink(c, geo.C["brass"], 4.0)
    c.move_to(x - r + 14, y)
    c.arc(x, y, r - 14, math.pi, 2 * math.pi)
    c.close_path()
    ink(c, geo.C["dial_face"], 3.0)
    for k in range(9):
        a = math.pi + k * math.pi / 8
        r0 = r - 36 if k % 2 == 0 else r - 28
        line(c, [(x + r0 * math.cos(a), y + r0 * math.sin(a)), (x + (r - 20) * math.cos(a), y + (r - 20) * math.sin(a))], 3.0)
    a = math.pi + needle
    line(c, [(x, y), (x + (r - 26) * math.cos(a), y + (r - 26) * math.sin(a))], 5.0, hx("#3a2a12"))
    ell(c, x, y - 2, 9, 9)
    ink(c, geo.C["brass2"], 3.0)


def lettering(c, text, x, y, size, fill, shadow, s=1.0, tilt=-.04):
    if s <= 0.01:
        return
    c.save()
    c.translate(x, y)
    c.rotate(tilt)
    c.scale(s, s)
    c.select_font_face(FONT, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    c.set_font_size(size)
    e = c.text_extents(text)
    ox, oy = -e.x_advance / 2, e.height / 2
    for dx, dy, col, wid in ((size * .07, size * .07, shadow, size * .12), (0, 0, fill, size * .12)):
        c.move_to(ox + dx, oy + dy)
        c.text_path(text)
        c.set_source_rgba(*INK)
        c.set_line_width(wid)
        c.stroke_preserve()
        c.set_source_rgba(*col)
        c.fill()
    c.restore()


def hoist_head(c, x, y, s, p):
    c.save()
    c.translate(x, y)
    c.scale(s, s)
    geo.head(c, p)
    c.restore()


def title_frame(i):
    t = (i // 2 * 2) / FPS
    surf, c = ctx_new()
    backdrop(c, t)
    dial(c, 640, 168, 104, min(1.0, max(0.0, (t - .2) / 1.6)) ** .7 * math.pi * .999)
    lettering(c, "MR. HOIST", 640, 318, 128, hx("#fbf1d6"), geo.C["red"], pop(t, .25))
    lettering(c, "in", 640, 410, 40, hx("#fbf1d6"), hx("#1e5560"), pop(t, .7), tilt=0)
    lettering(c, "One Small Passenger", 640, 482, 70, geo.C["gold"], hx("#1e5560"), pop(t, .9), tilt=-.02)
    sh = pop(t, 1.35, .3)
    p = dict(geo.NEUTRAL, brow=.9, mstyle="smile", eye=1.0, look=.6, hrot=.12)
    if sh > 0:
        hoist_head(c, 128, H - 66 + (1 - min(sh, 1)) * 260, .94, p)
    ss = pop(t, 1.75, .3)
    if ss > 0:
        short.snail(c, 1158, H - 84, 1.3 * ss, -1, dict(look=.6, stalk=(-.2, .3)))
    import cv2
    img = cv2.resize(snp(surf)[..., :3], (W, H), interpolation=cv2.INTER_AREA)
    return geo.to8(np.clip(img, 0, 1))


def end_frame(i):
    t = (i // 2 * 2) / FPS
    surf, c = ctx_new()
    backdrop(c, t + 3)
    dial(c, 640, 168, 104, math.pi / 8 * pop(t, .15, .3))       # needle settles on floor 2, of course
    lettering(c, "THE END", 640, 360, 140, hx("#fbf1d6"), geo.C["red"], pop(t, .1))
    short.snail(c, 300 + 30 * t, H - 92, 1.5, 1, dict(look=.5, stalk=(.0, .35)))
    import cv2
    img = cv2.resize(snp(surf)[..., :3], (W, H), interpolation=cv2.INTER_AREA)
    return geo.to8(np.clip(img, 0, 1))


def title_sting(dur=3.0):
    a, b, v, tp, fx = (S.buf(dur) for _ in range(5))
    for k, m in enumerate((65, 69, 72, 77, 81, 84)):           # vibes run up as the dial sweeps
        S.put(v, snd.vibe(m, .8, .8), .2 + k * .09)
    S.put(tp, snd.trumpet(72, .1, 1), .9)                     # ta-ta-TAAA on the subtitle
    S.put(tp, snd.trumpet(72, .1, 2), 1.03)
    S.put(tp, snd.trumpet(77, .7, 3), 1.16)
    S.put(tp, snd.trumpet(72, .7, 4) * .6, 1.16)
    S.put(tp, snd.trumpet(69, .7, 5) * .5, 1.16)
    for tb, m in ((.25, 41), (.9, 48), (1.16, 41), (1.75, 41)):
        S.put(a, snd.bass_note(m, .5, int(tb * 10)), tb)
    S.put(b, snd.ride(9, 1.6, 1.0), 1.16)
    S.put(b, snd.brush(8, .4, 1.0), 1.16)
    S.put(fx, snd.bell(1046.5, 1.4) * .6, 1.75)               # ding as the snail pops on
    S.put(fx, S.slide("sine", 900, 1500, .12) * np.hanning(int(.12 * S.SR)) * .3, 1.78)
    return S.master([(a, .3, -.1, .05), (v, .22, .2, .3), (tp, .3, 0, .25), (b, .12, .3, .2), (fx, .22, 0, .2)], wet=.18, fade_out=.4)


def end_sting(dur=2.0):
    a, v, fx = (S.buf(dur) for _ in range(3))
    S.put(fx, snd.bell(1046.5, 1.5) * .7, .12)
    for k, m in enumerate((65, 69, 74, 79, 84)):
        S.put(v, snd.vibe(m, 1.4, .8), .3 + k * .02)
    S.put(a, snd.bass_note(41, .7, 1), .3)
    S.put(a, snd.bass_note(29, .7, 2) * .8, .3)
    return S.master([(a, .32, -.1, .05), (v, .25, .2, .3), (fx, .25, 0, .2)], wet=.2, fade_out=.5)


def write(frames_fn, n, audio, path):
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        sf.write(wav, audio, S.SR, subtype="PCM_24")
        p = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                              "-i", "-", "-i", wav, "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "aac",
                              "-b:a", "256k", "-ar", "48000", "-ac", "2", "-t", f"{n / FPS:.4f}", "-movflags", "+faststart", path],
                             stdin=subprocess.PIPE)
        for i in range(n):
            p.stdin.write(np.ascontiguousarray(frames_fn(i)).tobytes())
        p.stdin.close()
        assert p.wait() == 0


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1:] or ["title", "end"]
    if "title" in what:
        write(title_frame, 72, title_sting(), os.path.join(OUT, "title_card.mp4"))
    if "end" in what:
        write(end_frame, 48, end_sting(), os.path.join(OUT, "end_card.mp4"))
