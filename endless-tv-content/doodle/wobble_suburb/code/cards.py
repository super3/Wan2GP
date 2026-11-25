"""Title and end cards for the short: boiling hand-inked lettering (DejaVu Sans Bold outlines, re-traced with jitter every
two frames) over a painted card, with synthesised stings.  cards.py OUTDIR  -> title_card.mp4, end_card.mp4"""
import math
import os
import subprocess
import sys
import tempfile

import cairo
import cv2
import numpy as np
import soundfile as sf

import cast as CA
import chars as CH
import ink as K
import paint
import snd
import tsound as S

W, H, FPS = 1280, 704, 24
TITLE_N, END_N = 72, 48


def text_polys(text, size, x, y, center=True):
    """Outline polylines of text (DejaVu Sans Bold) at (x, y) baseline-centre."""
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 8, 8)
    ctx = cairo.Context(surf)
    ctx.select_font_face("DejaVu Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(size)
    xb, yb, tw, th, xa, ya = ctx.text_extents(text)
    ctx.move_to(x - (tw / 2 + xb if center else 0), y)
    ctx.text_path(text)
    polys, cur = [], []
    for kind, pts in ctx.copy_path_flat():
        if kind == cairo.PATH_MOVE_TO:
            if len(cur) > 2:
                polys.append(np.array(cur))
            cur = [pts]
        elif kind == cairo.PATH_LINE_TO:
            cur.append(pts)
        elif kind == cairo.PATH_CLOSE_PATH:
            if len(cur) > 2:
                polys.append(np.array(cur))
            cur = []
    if len(cur) > 2:
        polys.append(np.array(cur))
    return polys


def resample(p, step=4.0):
    p = np.vstack([p, p[:1]])
    s = K.arclen(p)
    n = max(8, int(s[-1] / step))
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def boil_word(ctx, polys, fill, tag, i, amp=1.3, lw=5.0, shadow=(6, 7), scale=1.0, cx=640, cy=350, rot=0.0):
    r = np.random.default_rng(abs(hash((tag, i))) % (2 ** 32))
    jit = []
    for k, p in enumerate(polys):
        q = resample(p)
        # smooth low-frequency wobble per tracing
        nz = r.normal(0, amp, q.shape)
        nz = cv2.GaussianBlur(nz.astype(np.float32).reshape(-1, 1, 2), (1, 0), sigmaX=0, sigmaY=2.5).reshape(-1, 2) if len(q) > 6 else nz
        q = q + nz * 1.6
        q = K.rot(q - (cx, cy), rot) * scale + (cx, cy)
        jit.append(q)
    ctx.save()
    ctx.translate(*shadow)
    for q in jit:
        K.poly_path(ctx, q)
    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    ctx.set_source_rgba(*K.INK)
    ctx.fill()
    ctx.restore()
    for q in jit:
        K.poly_path(ctx, q)
    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    ctx.set_source_rgba(*fill)
    ctx.fill_preserve()
    ctx.set_source_rgba(*K.INK)
    ctx.set_line_width(lw)
    ctx.stroke()
    ctx.set_fill_rule(cairo.FILL_RULE_WINDING)


_CARD = {}


def card_bg(kind):
    if kind in _CARD:
        return _CARD[kind]
    surf, ctx = K.canvas()
    r = np.random.default_rng(3 if kind == "title" else 4)
    if kind == "title":
        cols = (paint.hexc("#f2a03d"), paint.hexc("#f7c95a"))
        cx, cy = 640, 760
        for k in range(24):  # painted sunburst
            a0, a1 = k * math.pi / 12, (k + 1) * math.pi / 12
            K.poly_path(ctx, np.array([(cx, cy), (cx + 2000 * math.cos(a0), cy + 2000 * math.sin(a0)), (cx + 2000 * math.cos(a1), cy + 2000 * math.sin(a1))]))
            ctx.set_source_rgba(*cols[k % 2][:3], 1)
            ctx.fill()
        ctx.rectangle(0, 0, W, H)
        ctx.clip()
        paint.brush(ctx, cols[0], r, 1600, 30, 120, 6, 16, -1.2, angj=1.2, vj=.06, hj=.02, alpha=(.08, .2), curve=.3)
        ctx.reset_clip()
        # a strip of striped lawn along the bottom
        ctx.rectangle(0, 600, W, 104)
        ctx.set_source_rgba(*paint.P["lawn"][:3], 1)
        ctx.fill()
        for k in range(0, W, 120):
            K.poly_path(ctx, np.array([(k, 600), (k + 60, 600), (k + 80, 704), (k + 20, 704)]))
            ctx.set_source_rgba(*paint.P["lawn_l"][:3], .6)
            ctx.fill()
    else:
        g = cairo.LinearGradient(0, 0, 0, H)
        g.add_color_stop_rgb(0, *paint.hexc("#8fb9dd")[:3])
        g.add_color_stop_rgb(1, *paint.hexc("#f3dcb0")[:3])
        ctx.rectangle(0, 0, W, H)
        ctx.set_source(g)
        ctx.fill()
        paint.brush(ctx, paint.P["sky_mid"], r, 900, 60, 200, 8, 18, 0.0, angj=.05, vj=.04, alpha=(.1, .25), curve=.05)
        ctx.rectangle(0, 560, W, 144)
        ctx.set_source_rgba(*paint.P["lawn"][:3], 1)
        ctx.fill()
    img = paint.finish_paint(paint.to_rgba(surf)[..., :3], seed=7)
    _CARD[kind] = img
    return img


def title_frame(i):
    t = (i // 2 * 2) / FPS
    surf, ctx = K.canvas()
    K.set_boil((i // 2) % 3, .9)
    K.set_wmul(1.0)
    # the dandelion, bobbing on the lawn strip
    CH.dandelion(ctx, 1080, 640, s=2.0, sway=.3 * math.sin(t * 3), bob=-6 * abs(math.sin(t * 4)))
    pop1 = 1 - math.exp(-max(0, t - .1) * 9) * math.cos(max(0, t - .1) * 18) if t > .1 else 0
    pop2 = 1 - math.exp(-max(0, t - .55) * 9) * math.cos(max(0, t - .55) * 18) if t > .55 else 0
    pop3 = 1 - math.exp(-max(0, t - .95) * 9) * math.cos(max(0, t - .95) * 18) if t > .95 else 0
    if pop1 > 0:
        boil_word(ctx, WEN_P, paint.hexc("#58aee6"), "wen", (i // 2) % 3, scale=pop1, cx=640, cy=200, rot=-.04)
    if pop2 > 0:
        boil_word(ctx, IN_P, paint.hexc("#fbf8f0"), "in", (i // 2) % 3, scale=pop2, cx=640, cy=290, lw=3.5, shadow=(4, 5))
    if pop3 > 0:
        boil_word(ctx, ONE_P, paint.hexc("#ffdf3f"), "one", (i // 2) % 3, scale=pop3, cx=640, cy=400, rot=.02)
        boil_word(ctx, DAN_P, paint.hexc("#ffdf3f"), "dan", (i // 2) % 3, scale=pop3, cx=640, cy=500, rot=-.02)
    img = paint.over(card_bg("title"), paint.to_rgba(surf))
    return (np.clip(img, 0, 1) * 255 + .5).astype(np.uint8)


def end_frame(i):
    t = (i // 2 * 2) / FPS
    surf, ctx = K.canvas()
    K.set_boil((i // 2) % 3, .9)
    K.set_wmul(1.0)
    CA.seeds(ctx, 30, (60, 40 - 30 * t, 1220, 520 - 30 * t), seed=11, s=1.8)
    sc = 1 - math.exp(-t * 9) * math.cos(t * 18)
    boil_word(ctx, END_P, paint.hexc("#ffdf3f"), "end", (i // 2) % 3, scale=sc, cx=640, cy=340, rot=-.03)
    CH.dandelion(ctx, 200, 640, s=1.6, state="bare")
    img = paint.over(card_bg("end"), paint.to_rgba(surf))
    return (np.clip(img, 0, 1) * 255 + .5).astype(np.uint8)


WEN_P = text_polys("WENDELL", 150, 640, 255)
IN_P = text_polys("in", 64, 640, 312)
ONE_P = text_polys("ONE LOUSY", 104, 640, 437)
DAN_P = text_polys("DANDELION", 104, 640, 540)
END_P = text_polys("THE END", 150, 640, 395)


def title_sting(dur=3.0):
    st = snd.new_stems(dur)
    for k, m in enumerate(range(48, 84, 2)):  # piano glissando up
        S.put(st["piano"], snd.piano(m, .25, .35, 1.2), .05 + k * .022)
    for t0, m in ((.55, 67), (.75, 72)):  # "ta-DA"
        S.put(st["horn"], snd.trumpet(m, .22 if t0 < .7 else 1.3, 0, 1.0) + snd.trumpet(m + 4, .22 if t0 < .7 else 1.3, 0, .7), t0)
    S.put(st["fx"], snd.boing(260, .5), 1.0, .6)
    S.put(st["drums"], snd.crash_hit(1.8), .75, 1.0)
    S.put(st["bass"], snd.ubass(36, .6, 1.0), .75)
    S.put(st["piano"], snd.piano(60, 1.6, .5) + snd.piano(64, 1.6, .4) + snd.piano(67, 1.6, .4) + snd.piano(70, 1.6, .35), .75)
    return snd.mixdown(st)


def end_sting(dur=2.0):
    st = snd.new_stems(dur)
    S.put(st["bass"], snd.ubass(43, .25, 1.0), .1)
    S.put(st["bass"], snd.ubass(36, .9, 1.1), .32)
    S.put(st["piano"], snd.piano(60, 1.2, .5) + snd.piano(64, 1.2, .4) + snd.piano(67, 1.2, .4) + snd.piano(69, 1.2, .35), .32)
    S.put(st["drums"], snd.crash_hit(1.4), .32, .8)
    S.put(st["horn"], snd.trumpet(72, .6, 1, .7, bend=-.04), .34)
    return snd.mixdown(st)


def write_mp4(frame_fn, n, audio, out):
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        sf.write(wav, audio, 48000, subtype="PCM_24")
        p = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                              "-i", "-", "-i", wav, "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac",
                              "-b:a", "192k", "-ar", "48000", "-ac", "2", "-t", f"{n / FPS:.4f}", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
        for i in range(n):
            p.stdin.write(frame_fn(i).tobytes())
        p.stdin.close()
        assert p.wait() == 0


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/home/user/Wan2GP/outputs/shorts/wobble_suburb"
    ta, ea = title_sting(TITLE_N / FPS), end_sting(END_N / FPS)
    sf.write(os.path.join(os.path.dirname(os.path.abspath(__file__)), "title_sting.wav"), ta, 48000)
    sf.write(os.path.join(os.path.dirname(os.path.abspath(__file__)), "end_sting.wav"), ea, 48000)
    write_mp4(title_frame, TITLE_N, ta, os.path.join(out, "title_card.mp4"))
    write_mp4(end_frame, END_N, ea, os.path.join(out, "end_card.mp4"))
    print("cards done")
