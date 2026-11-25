"""Title card (3 s) and end card (2 s) for "Clean Sweep": painted desert, a hand-built plaque that pops in with an overshoot,
DejaVu Sans Bold lettering with an ink outline and an off-register shadow, and Pip in his proud pose.  The sting for each card is
the matching section of the music bed (so the card files work on their own; the assembly uses the bed for those seconds)."""
import math
import os
import subprocess
import sys
import tempfile

import cairo
import cv2
import numpy as np
import soundfile as sf

import cast
import desert as D
import scene
import shotlist as SL
import world

W, H, SS = 1280, 704, 2
CREAM, RED, INKC = D.hexc("#fbf0d6"), D.hexc("#d8453a"), D.INK


def text(ctx, s, x, y, size, fill, outline=INKC, ow=6.0, shadow=D.hexc("#7a3b4a"), anchor="c", slant=0.0):
    ctx.save()
    ctx.select_font_face("DejaVu Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(size)
    ext = ctx.text_extents(s)
    tx = x - (ext.width / 2 + ext.x_bearing if anchor == "c" else 0)
    ty = y
    ctx.translate(tx, ty)
    if slant:
        ctx.transform(cairo.Matrix(1, 0, slant, 1, 0, 0))
    for dx, dy, col, mode in ((6, 6, shadow, "fill"), (0, 0, outline, "stroke"), (0, 0, fill, "fill")):
        ctx.new_path()
        ctx.move_to(dx, dy)
        ctx.text_path(s)
        ctx.set_source_rgba(*col)
        if mode == "stroke":
            ctx.set_line_width(ow)
            ctx.set_line_join(cairo.LINE_JOIN_ROUND)
            ctx.stroke()
        else:
            ctx.fill()
    ctx.restore()


def plaque(ctx, cx, cy, w, h, s, rot, col=CREAM):
    """A flat cream plaque with scalloped ends, an ink outline and a coloured inner rule."""
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(rot)
    ctx.scale(s, s)
    pts = []
    for a in np.linspace(-math.pi / 2, math.pi / 2, 9):
        pts.append((w / 2 + 26 * math.cos(a) + (8 if abs(a) < 1.2 else 0) * math.cos(a), h / 2 * math.sin(a)))
    for a in np.linspace(math.pi / 2, 3 * math.pi / 2, 9):
        pts.append((-w / 2 + 26 * math.cos(a) - (8 if abs(a - math.pi) < 1.2 else 0), h / 2 * math.sin(a)))
    P = D.crs(pts, True, 8, .6)
    D.fillp(ctx, P + [10, 12], D.alpha(D.hexc("#5c2f3c"), .55))
    D.fillp(ctx, P, col)
    D.ink(ctx, P, D.closed_widths(P, 5.0), closed=True)
    Q = D.crs([(x * .9, y * .8) for x, y in pts], True, 8, .6)
    D.ink(ctx, Q, 2.2, D.alpha(RED, .9), closed=True)
    ctx.restore()


def pop(f, f0, dur=8):
    """0 -> overshoot -> 1 scale for a plaque popping in."""
    t = (f - f0) / dur
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    return 1 + .25 * math.sin(t * math.pi * 1.5) * (1 - t) if t > .35 else t / .35 * 1.15


def card_frame(kind, f):
    cam = "B" if kind == "title" else "A"
    bgk = dict() if kind == "title" else dict(time="sunset")

    def chars(ctx):
        if kind == "title":
            p = dict(D.PROUD, x=1080, gy=700, sc=.78, blink=1.0 if f in (40, 41) else 0.0)
            D.pip(ctx, p)
        else:
            D.pip(ctx, dict(D.PROUD, x=300, gy=700, sc=.7, blink=1.0 if f in (30, 31) else 0.0))
            cast.bramble(ctx, dict(cast.BRAMBLE, x=1020, gy=700, sc=.7, eyes="happy", mouth="grin", scarf=True, head_rot=.08))

    img = scene.compose(cam, bg=bgk, layers=[dict(draw=chars)], seed=f)[8:712]
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * SS, H * SS)
    ctx = cairo.Context(surf)
    ctx.scale(SS, SS)
    s = pop(f, 2)
    if kind == "title":
        if s > 0:
            plaque(ctx, 560, 300, 720, 250, s, -.03)
            ctx.save()
            ctx.translate(560, 300)
            ctx.rotate(-.03)
            ctx.scale(s, s)
            text(ctx, SL.SERIES.upper(), 0, -62, 30, D.hexc("#3c6fb0"), ow=4.0, shadow=D.hexc("#a8bfd8"))
            text(ctx, SL.TITLE.upper(), 0, 46, 94, RED, ow=8.0, slant=-.12)
            ctx.restore()
        s2 = pop(f, 14, 6)
        if s2 > 0:   # a little dust puff under the plaque as it lands
            D.dust(ctx, 300, 430, 22 * s2, 3, n=6)
            D.dust(ctx, 820, 430, 22 * s2, 4, n=6)
    else:
        if s > 0:
            plaque(ctx, 640, 290, 460, 190, s, .02)
            ctx.save()
            ctx.translate(640, 290)
            ctx.rotate(.02)
            ctx.scale(s, s)
            text(ctx, "THE END", 0, 34, 96, RED, ow=8.0, slant=-.12)
            ctx.restore()
    a = np.ndarray((H * SS, W * SS, 4), np.uint8, surf.get_data())
    a = cv2.resize(a.astype(np.float32) / 255, (W, H), interpolation=cv2.INTER_AREA)
    rgb, al = a[:, :, 2::-1], a[:, :, 3:4]
    out = rgb + img.astype(np.float32) / 255 * (1 - al)
    return D.to8(out)


def make(kind, out_path, bed_path):
    nfr = SL.TITLE_FRAMES if kind == "title" else SL.END_FRAMES
    bed, sr = sf.read(bed_path, dtype="float32")
    if kind == "title":
        seg = bed[:int(nfr / SL.FPS * sr)]
    else:
        seg = bed[-int(nfr / SL.FPS * sr):]
    seg = seg * np.clip(np.linspace(0, nfr / SL.FPS, len(seg))[::-1][:, None] / .25, 0, 1) if kind == "end" else seg
    with tempfile.TemporaryDirectory() as tmp:
        for f in range(nfr):
            cv2.imwrite(os.path.join(tmp, f"f{f:04d}.png"), card_frame(kind, f)[:, :, ::-1])
        wav = os.path.join(tmp, "a.wav")
        sf.write(wav, seg, sr, subtype="PCM_24")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-framerate", str(SL.FPS), "-i", os.path.join(tmp, "f%04d.png"), "-i", wav,
                        "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                        "-t", f"{nfr / SL.FPS:.4f}", "-movflags", "+faststart", out_path], check=True)


if __name__ == "__main__":
    out_dir, bed_path = sys.argv[1], sys.argv[2]
    which = sys.argv[3:] or ["title", "end"]
    for k in which:
        make(k, os.path.join(out_dir, f"{k}_card.mp4"), bed_path)
        print(k, "card done")
