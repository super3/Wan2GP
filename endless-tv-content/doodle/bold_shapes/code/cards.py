"""Title card (3 s) and end card (2 s) for 'One Little Smudge', with synthesised stings.

Title: a pane of glass with the smudge; a giant squeegee wipes across and reveals a pop-art title card... but the smudge is
still there, and Squeaky pops up in the corner to glare at it.  End: 'THE END' over Squeaky's face print; Mrs. Fig's lace
hanky wipes the print away."""
import math
import os
import subprocess
import tempfile

import cairo
import cv2
import numpy as np
import soundfile as sf

import scenes as SC
import squeaky as SQK
import tsound as S
from draw import INK, P, WHITE, ellipse, fill, hexc, lerp, poly, rrect, sparkle, stroke, ease

W, H, SS, FPS = 1280, 704, 2, 24
FONT = "DejaVu Sans"


def canvas():
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * SS, H * SS)
    ctx = cairo.Context(surf)
    ctx.set_matrix(SC.cam())
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    return surf, ctx


def to_img(surf):
    surf.flush()
    a = np.ndarray((H * SS, W * SS, 4), np.uint8, surf.get_data()).astype(np.float32) / 255
    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
    return a[:, :, 2::-1] * SC.texture()


def lettering(ctx, text, cx, cy, size, fillc, pop=None, shadow=hexc("#2b1f6e"), outline=12, jaunt=1.0, spacing=1.0, seed=0):
    """Jaunty hand-set lettering: each glyph tilted and bounced a little, thick black outline, offset drop shadow.
    pop(i) -> scale of glyph i (for pop-in animation)."""
    ctx.save()
    ctx.select_font_face(FONT, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(size)
    adv = [ctx.text_extents(ch).x_advance * spacing if ch != " " else size * .38 for ch in text]
    total = sum(adv)
    x = cx - total / 2
    rng = np.random.default_rng(seed)
    glyphs = []
    for i, ch in enumerate(text):
        if ch != " ":
            ext = ctx.text_extents(ch)
            gx = x + adv[i] / 2
            rot = (rng.uniform(-.09, .09)) * jaunt
            dy = (rng.uniform(-.07, .07) * size) * jaunt
            s = 1.0 if pop is None else pop(i)
            glyphs.append((ch, gx, cy + dy, rot, s, ext))
        x += adv[i]
    for layer in ("shadow", "outline", "fill"):
        for ch, gx, gy, rot, s, ext in glyphs:
            if s <= .01:
                continue
            ctx.save()
            off = (7, 8) if layer == "shadow" else (0, 0)
            ctx.translate(gx + off[0], gy + off[1])
            ctx.rotate(rot)
            ctx.scale(s, s)
            ctx.move_to(-ext.x_bearing - ext.width / 2, ext.height / 2 - (ext.height + ext.y_bearing))
            ctx.text_path(ch)
            if layer == "shadow":
                ctx.set_line_width(outline)
                ctx.set_source_rgba(*shadow)
                ctx.fill_preserve()
                ctx.stroke()
            elif layer == "outline":
                ctx.set_line_width(outline)
                ctx.set_source_rgba(*INK)
                ctx.stroke()
            else:
                ctx.set_source_rgba(*fillc)
                ctx.fill()
            ctx.restore()
    ctx.restore()


def glass_bg(ctx):
    ctx.rectangle(-10, -10, W + 20, H + 20)
    fill(ctx, P["glass"])
    poly(ctx, [(-10, H + 10), (W + 10, H * .2), (W + 10, H + 10)])
    fill(ctx, P["glass2"])
    for (a, b) in ((120, 140), (300, 50)):
        poly(ctx, [(a, 520), (a + 380, 120), (a + 380 + b, 120), (a + b, 520)])
        fill(ctx, hexc("#ffffff", .8))
    ctx.rectangle(-10, -10, W + 20, H + 20)  # the window frame around the edges
    ctx.rectangle(26, 26, W - 52, H - 52)
    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    fill(ctx, P["fac"])
    ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
    ctx.rectangle(26, 26, W - 52, H - 52)
    stroke(ctx, P["fac_ln"], 3)


def burst_card(ctx, t):
    ctx.rectangle(-10, -10, W + 20, H + 20)
    fill(ctx, P["burst"])
    cx, cy = 640, 330
    for k in range(32):
        a0 = k * 2 * math.pi / 32 + t * .25
        a1 = a0 + math.pi / 32
        poly(ctx, [(cx, cy), (cx + 2000 * math.cos(a0), cy + 2000 * math.sin(a0)), (cx + 2000 * math.cos(a1), cy + 2000 * math.sin(a1))])
    fill(ctx, P["burst2"])
    ellipse(ctx, cx, cy, 470, 220)  # a cream lozenge behind the title
    fill(ctx, hexc("#2b1f6e"))
    ellipse(ctx, cx - 8, cy - 10, 470, 220)
    fill(ctx, P["cream"], True)
    stroke(ctx, INK, 7)
    for k in range(10):  # atomic dots around the lozenge
        a = k * 2 * math.pi / 10 + .3
        ellipse(ctx, cx + 520 * math.cos(a), cy + 262 * math.sin(a), 9, 9)
    fill(ctx, P["hat"])


def title_frame(i):
    t = i / FPS
    surf, ctx = canvas()
    wipe_x = -160 + (W + 320) * ease((t - .45) / .5)
    # revealed card (left of the squeegee)
    ctx.save()
    ctx.rectangle(-10, -10, wipe_x + 10, H + 20)
    ctx.clip()
    burst_card(ctx, t)
    lettering(ctx, "SQUEAKY DOYLE", 632, 214, 50, P["hat"], seed=3, outline=10, spacing=1.04)
    lettering(ctx, "in", 632, 270, 40, WHITE, seed=5, outline=9, jaunt=.3)

    def pop(j):
        u = (t - 1.0 - j * .045) / .22
        if u <= 0:
            return 0.0
        return 1 + .25 * math.sin(math.pi * min(u, 1)) if u < 1 else 1.0
    lettering(ctx, "ONE LITTLE", 632, 344, 78, WHITE, pop=pop, seed=7, spacing=1.02)
    lettering(ctx, "SMUDGE!", 632, 438, 96, P["coral"], pop=lambda j: pop(j + 10), seed=9)
    ctx.restore()
    # glass not yet wiped (right of the squeegee)
    ctx.save()
    ctx.rectangle(wipe_x, -10, W + 200, H + 20)
    ctx.clip()
    glass_bg(ctx)
    ctx.restore()
    # the smudge survives the wipe, of course
    SC.smudge(ctx, 985, 520, 2.0)
    if .4 < t < 1.0:  # the giant squeegee
        x = wipe_x
        poly(ctx, [(x - 14, -20), (x + 6, -20), (x + 6, H + 20), (x - 14, H + 20)])
        fill(ctx, P["rubber"], True)
        stroke(ctx, INK, 4)
        poly(ctx, [(x - 54, -20), (x - 14, -20), (x - 14, H + 20), (x - 54, H + 20)])
        fill(ctx, P["steel"], True)
        stroke(ctx, INK, 5)
        poly(ctx, [(x - 54, 350), (x - 300, 380)], closed=False)
        stroke(ctx, INK, 30)
        poly(ctx, [(x - 54, 350), (x - 300, 380)], closed=False)
        stroke(ctx, P["pole"], 20)
        for k in range(6):  # speed streaks behind it
            yy = 60 + k * 110
            poly(ctx, [(x - 80 - 40 * (k % 2), yy), (x - 260 - 60 * (k % 3), yy)], closed=False)
            stroke(ctx, WHITE, 5)
    if t > 1.9:  # Squeaky pops up in the corner and glares at the smudge
        u = ease((t - 1.9) / .3)
        oy = lerp(1080, 800, u)
        p = SC.P_(mirror=True, turn=.55, look=(1.0, -.6), lid=.3, slant=13, mouth="set", prop="none", hang_bucket=False, rig=False,
                  grip=(60.0, -60.0), elbowR=(80.0, -90.0))
        SQK.squeaky(ctx, SC.cam(), p, (1150.0, oy), .62, lw=.8)
    if 1.45 < t < 2.3:
        s = math.sin(math.pi * (t - 1.45) / .85)
        sparkle(ctx, 905, 300, s * 1.1, t * 2)
    return to_img(surf)


def end_frame(i):
    t = i / FPS
    surf, ctx = canvas()
    glass_bg(ctx)
    hx = lerp(-260, W + 300, ease((t - .95) / .7))  # Mrs. Fig's lace hanky wipes the print away
    ctx.save()
    ctx.rectangle(hx, -10, W + 400, H + 20)
    ctx.clip()
    SC.face_print(ctx, 640, 330, 1.25)
    ctx.restore()

    def pop(j):
        u = (t - .15 - j * .06) / .2
        if u <= 0:
            return 0.0
        return 1 + .25 * math.sin(math.pi * min(u, 1)) if u < 1 else 1.0
    lettering(ctx, "THE END", 640, 330, 120, P["hat"], pop=pop, seed=12)
    if .9 < t < 1.75:
        cel = SQK.Cel(ctx, SC.cam(), lw=1.2)
        ctx.set_matrix(SC.cam())
        ctx.translate(hx, 470)
        ctx.scale(2.6, 2.6)
        SQK.hanky(cel, 0, 0, 1.0, .3)
        cel.draw()
        ctx.set_matrix(SC.cam())
    if t > 1.55:
        s = math.sin(math.pi * min((t - 1.55) / .45, 1))
        sparkle(ctx, 880, 230, s * 1.2, t * 3)
    return to_img(surf)


# ------------------------------------------------------------------------------------------------ stings
def chime(f, dur=.8, g=1.0):
    n = int(dur * S.SR)
    t = S.tt(n)
    return (np.sin(2 * np.pi * f * t) + .3 * np.sin(2 * np.pi * f * 2.76 * t) + .1 * np.sin(2 * np.pi * f * 5.4 * t)) * np.exp(-t * 6) * g


def brass_chord(notes, dur):
    n = int(dur * S.SR)
    t = S.tt(n)
    out = np.zeros(n)
    for m in notes:
        f = S.hz(m) * (1 + .004 * np.sin(2 * np.pi * 5 * t))
        out += S.lp(S.osc("saw", f * 1.003, n) + S.osc("saw", f * .997, n), 2400)
    return out * S.adsr(n, .015, .15, .6, .3)


def vibes(notes, dur):
    n = int(dur * S.SR)
    t = S.tt(n)
    out = np.zeros(n)
    for m in notes:
        f = S.hz(m)
        out += (np.sin(2 * np.pi * f * t) + .25 * np.sin(2 * np.pi * f * 4 * t) * np.exp(-t * 9)) * np.exp(-t * 1.8)
    return out * (1 + .3 * np.sin(2 * np.pi * 5.5 * t))


def squeak(dur=.4, f0=1500, f1=2600):
    n = int(dur * S.SR)
    t = S.tt(n)
    f = f0 + (f1 - f0) * np.sin(np.pi * t / dur) + 70 * np.sin(2 * np.pi * 40 * t)
    ph = np.cumsum(f) / S.SR
    return (np.sin(2 * np.pi * ph) + .4 * np.sin(4 * np.pi * ph)) * np.clip(t / .01, 0, 1) * np.clip((dur - t) / .05, 0, 1)


def title_sting():
    d = 3.0
    mus, sfx = S.buf(d), S.buf(d)
    S.put(sfx, S.whoosh(.55, 500, 5000, seed=2), .42, .9)
    S.put(sfx, squeak(.5), .48, .5)
    S.put(mus, brass_chord((52, 59, 64, 68, 71), 1.2), .98, .8)  # E major 'ta-da'
    S.put(mus, vibes((76, 80, 83, 88), 1.8), 1.0, .5)
    for k, f in enumerate((200, 240, 280, 330)):
        S.put(mus, S.tom(f, .2), 1.0 + .08 * k, .35)
    S.put(sfx, chime(2637), 1.5, .5)
    S.put(sfx, chime(3520), 1.62, .4)
    nb = int(.4 * S.SR)
    tb = S.tt(nb)
    S.put(sfx, np.sin(2 * np.pi * np.cumsum(200 + 260 * tb + 60 * np.sin(2 * np.pi * 16 * tb) * np.exp(-tb * 6)) / S.SR) * np.exp(-tb * 5), 1.92, .55)
    return S.master([(mus, .55, 0, .3), (sfx, .45, .1, .2)], wet=.2, rev_s=1.2, drive=1.2, fade_out=.3)


def end_sting():
    d = 2.0
    mus, sfx = S.buf(d), S.buf(d)
    S.put(mus, vibes((64, 68, 71, 76), 1.9), .15, .7)
    rim = S.bp(S.noise(int(.06 * S.SR), 77), 1500, 6000) * np.exp(-S.tt(int(.06 * S.SR)) * 70)
    S.put(sfx, rim, .15, .8)
    S.put(mus, S.lp(S.osc("saw", S.hz(40), int(.5 * S.SR)), 600) * np.exp(-S.tt(int(.5 * S.SR)) * 5), .15, .8)
    S.put(sfx, squeak(.5, 1200, 1900), .95, .35)
    S.put(sfx, chime(3136), 1.55, .45)
    S.put(sfx, chime(3951), 1.65, .35)
    return S.master([(mus, .55, 0, .3), (sfx, .45, .1, .2)], wet=.2, rev_s=1.2, drive=1.2, fade_out=.25)


def write_mp4(path, frames_fn, n, audio):
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        sf.write(wav, audio, S.SR, subtype="PCM_16")
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
               "-i", "-", "-i", wav, "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-tune", "animation", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{n / FPS:.4f}", "-movflags", "+faststart", path]
        pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i in range(n):
            img = frames_fn(i)
            pr.stdin.write((np.clip(img, 0, 1) * 255 + .5).astype(np.uint8).tobytes())
        pr.stdin.close()
        assert pr.wait() == 0


def make(out):
    write_mp4(os.path.join(out, "title_card.mp4"), title_frame, 72, title_sting())
    write_mp4(os.path.join(out, "end_card.mp4"), end_frame, 48, end_sting())
