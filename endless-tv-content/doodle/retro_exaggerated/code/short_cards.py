"""Title card (3 s) and end card (2 s) for "The Littlest Crossing": early-60s TV title style (rotating starburst, a kidney
plaque, bouncy hand-set lettering with an ink outline and offset shadow), with Bertram and Tilly drawn by the shared code.
Their stings are the matching stretches of music_bed.wav, so the assembled short and the cards agree."""
import math
import os
import subprocess
import sys
import tempfile

import cairo
import cv2
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bigsby as G  # noqa: E402
import short_shots as SH  # noqa: E402
from bigsby import INK, C, hexc  # noqa: E402

W, H = 1280, 704
FONT = "DejaVu Sans"
NAVY = hexc("#223a72")
CREAM = hexc("#fff4dc")


def pop(f, f0, dur=6):
    """Overshoot pop-in scale for an element that appears at frame f0."""
    u = (f - f0) / dur
    if u <= 0:
        return 0.0
    if u >= 1:
        return 1.0
    return math.sin(u * math.pi / 2) + .3 * math.sin(u * math.pi)


def lettering(ctx, text, x, y, size, f, f0, step=1.5, fill=CREAM, shadow=hexc("#ff8f3a"), bounce=7, seed=0, center=True):
    """Hand-set letters: each glyph gets its own little tilt and baseline hop and pops in on its own frame."""
    ctx.select_font_face(FONT, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(size)
    rng = np.random.default_rng(seed)
    widths = [ctx.text_extents(ch).x_advance * .96 for ch in text]
    total = sum(widths)
    cx = x - total / 2 if center else x
    for k, ch in enumerate(text):
        s = pop(f, f0 + k * step)
        if ch == " " or s <= 0:
            cx += widths[k]
            continue
        rot = rng.uniform(-.07, .07)
        dy = bounce * math.sin(k * 1.7 + seed)
        ext = ctx.text_extents(ch)
        gx, gy = cx + widths[k] / 2, y + dy - ext.height / 2
        for layer in ("shadow", "main"):
            ctx.save()
            ctx.translate(gx + (6 if layer == "shadow" else 0), gy + (6 if layer == "shadow" else 0))
            ctx.rotate(rot)
            ctx.scale(s, s)
            ctx.move_to(-ext.x_advance / 2 * .96 - ext.x_bearing * 0, ext.height / 2)
            ctx.text_path(ch)
            ctx.restore()
            if layer == "shadow":
                ctx.set_source_rgba(*shadow)
                ctx.fill_preserve()
                ctx.set_source_rgba(*INK)
                ctx.set_line_width(size * .07)
                ctx.stroke()
            else:
                ctx.set_source_rgba(*fill)
                ctx.fill_preserve()
                ctx.set_source_rgba(*INK)
                ctx.set_line_width(size * .075)
                ctx.stroke()
        cx += widths[k]


def starburst(ctx, cx, cy, ang, c0, c1, n=28):
    ctx.rectangle(0, 0, W, H)
    ctx.set_source_rgba(*c0)
    ctx.fill()
    for k in range(n):
        a0 = ang + k * 2 * math.pi / n
        a1 = a0 + math.pi / n
        ctx.move_to(cx, cy)
        ctx.line_to(cx + 2000 * math.cos(a0), cy + 2000 * math.sin(a0))
        ctx.line_to(cx + 2000 * math.cos(a1), cy + 2000 * math.sin(a1))
        ctx.close_path()
    ctx.set_source_rgba(*c1)
    ctx.fill()


def sparkle(ctx, x, y, r, col=CREAM):
    pts = []
    for k in range(8):
        a = k * math.pi / 4
        rr = r if k % 2 == 0 else r * .28
        pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
    G.fs(ctx, np.array(pts), col, 2.4)


def surface(ss=2):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * ss, H * ss)
    ctx = cairo.Context(surf)
    ctx.scale(ss, ss)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    return surf, ctx


def to_img(surf, ss=2):
    surf.flush()
    a = np.ndarray((H * ss, W * ss, 4), np.uint8, surf.get_data())[:, :, :3][:, :, ::-1].astype(np.float32) / 255
    return cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)


def title_frame(f, ss=2):
    surf, ctx = surface(ss)
    starburst(ctx, 640, 300, f * .006, hexc("#8fd3c8"), hexc("#a6ddd2"))
    rng = np.random.default_rng(3)
    for k in range(14):
        x, y = rng.uniform(40, 1240), rng.uniform(30, 680)
        if 250 < x < 1030 and 120 < y < 520:
            continue
        sparkle(ctx, x, y, rng.uniform(10, 20) * (1 + .15 * math.sin(f * .3 + k)))
    # the kidney-shaped plaque
    s = pop(f, 0, 8)
    if s > 0:
        ctx.save()
        ctx.translate(660, 356)
        ctx.scale(s, s)
        G.LS[0] = 1.0
        plaque = G.cr([(-470, -150), (-200, -205), (120, -170), (460, -200), (500, 40), (420, 175), (60, 150), (-300, 190), (-500, 120)], per=16)
        ctx.translate(8, 10)
        G.fs(ctx, plaque, hexc("#c86f55"), 0)
        ctx.translate(-8, -10)
        G.fs(ctx, plaque, hexc("#f3a487"), 5)
        ctx.restore()
    surf2, ctx2 = surface(ss)
    lettering(ctx, "BERTRAM BIGSBY", 660, 245, 54, f, 6, step=.8, fill=hexc("#ffd34d"), shadow=NAVY, bounce=4, seed=2)
    lettering(ctx, "in", 660, 300, 34, f, 16, fill=CREAM, shadow=NAVY, bounce=0, seed=1)
    lettering(ctx, "The Littlest", 660, 378, 92, f, 18, step=1.2, seed=4)
    lettering(ctx, "Crossing!", 660, 484, 104, f, 28, step=1.4, seed=6)
    img = to_img(surf, ss)
    # Bertram's head peeking up from the bottom-left, Tilly perched on his cap
    rise = min(max((f - 4) / 10, 0), 1)
    rise = 1 - (1 - rise) ** 3
    by = 860 - 230 * rise
    P = G.bpose(lid=.5 if 50 <= f < 54 else .3, mouth="grin", hr=-.12, px=2)

    def draw_b(ctx_):
        ctx_.save()
        ctx_.translate(150, by)
        ctx_.scale(1.55, 1.55)
        G.LS[0] = 1.55 ** -.75
        G.draw_head(ctx_, P, np.array([0.0, 0.0]))
        ctx_.restore()
    rgb, al = G.cel(draw_b, (1.0, 640, 360), ss)
    rgb, al = rgb[8:712], al[8:712]
    img = img * (1 - al[..., None]) + rgb
    ts = pop(f, 20, 6)
    if ts > .02:
        T = G.tpose(X=166, Y=by - 158 + 8, S=.62 * ts, face=1, arms="wave" if (f // 6) % 2 == 0 else "raise", mouth="smile", sit=True)
        rgb, al = G.cel(lambda c: G.draw_tilly(c, T, 1.0), (1.0, 640, 360), ss)
        img = img * (1 - al[8:712, :, None]) + rgb[8:712]
    return img


def end_frame(f, ss=2):
    surf, ctx = surface(ss)
    starburst(ctx, 640, 352, -f * .008, hexc("#2b4a8a"), NAVY, n=24)
    r = 330 - 40 * min(f / 10, 1)
    if f > 38:
        r *= max(0.0, 1 - (f - 38) / 9) ** 1.5
    ctx.arc(640, 330, max(r, .1), 0, 2 * math.pi)
    ctx.set_source_rgba(*hexc("#8fd3c8"))
    ctx.fill_preserve()
    ctx.set_source_rgba(*INK)
    ctx.set_line_width(7)
    ctx.stroke()
    ctx.save()
    ctx.arc(640, 330, max(r, .1), 0, 2 * math.pi)
    ctx.clip()
    lettering(ctx, "The End", 640, 300, 110, f, 3, step=1.5, seed=9)
    lettering(ctx, "BERTRAM BIGSBY", 640, 392, 40, f, 12, step=.6, fill=hexc("#ffd34d"), shadow=NAVY, bounce=3, seed=3)
    ctx.restore()
    img = to_img(surf, ss)
    if 15 <= f <= 40:
        T = G.tpose(X=640, Y=556, S=.75 * pop(f, 14, 6), face=1, arms="wave" if (f // 6) % 2 == 0 else "raise", mouth="smile")
        surfm, ctxm = surface(ss)
        rgb, al = G.cel(lambda c: G.draw_tilly(c, T, 1.0), (1.0, 640, 360), ss)
        yy, xx = np.mgrid[0:H, 0:W]
        inside = (((xx - 640) ** 2 + (yy - 330) ** 2) < (r - 4) ** 2).astype(np.float32)
        al = al[8:712] * inside
        img = img * (1 - al[..., None]) + rgb[8:712] * inside[..., None]
    if f > 46:
        img = img * 0
    return img


def write_card(kind, out, ss=2):
    import short_music as SM
    nfr = SH.TITLE_FRAMES if kind == "title" else SH.END_FRAMES
    bed, sr = sf.read(os.path.join(SH.OUT, "music_bed.wav"))
    if kind == "title":
        audio = bed[:int(nfr / 24 * sr)]
    else:
        audio = bed[int(SM.END0 * sr):int(SM.END0 * sr) + int(nfr / 24 * sr)]
    audio = audio / (np.abs(audio).max() + 1e-9) * .89          # the sting alone, brought up to -1 dBFS
    with tempfile.TemporaryDirectory() as tmp:
        for f in range(nfr):
            img = (title_frame if kind == "title" else end_frame)(f, ss)
            cv2.imwrite(os.path.join(tmp, f"f{f:04d}.png"), G.to8(img)[:, :, ::-1])
        wav = os.path.join(tmp, "a.wav")
        sf.write(wav, audio.astype(np.float32), sr, subtype="PCM_24")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-framerate", "24", "-i", os.path.join(tmp, "f%04d.png"), "-i", wav,
                        "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                        "-t", f"{nfr / 24:.4f}", "-movflags", "+faststart", out], check=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("title", "both"):
        write_card("title", os.path.join(SH.OUT, "title_card.mp4"))
    if which in ("end", "both"):
        write_card("end", os.path.join(SH.OUT, "end_card.mp4"))
