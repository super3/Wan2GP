"""The 5-second code-only shot: Wendell calls out the dandelion, it boings at him, he does an extreme take and gulps back
into his pose.  Animated pose-to-pose on twos with holds; every drawing boils (three tracings cycling, like the triple-inked
cels of the era).  frame(i) -> 1280x720 RGB uint8."""
import math

import cv2
import numpy as np

import chars as CH
import ink as K
import key as KY
import paint

FPS = 24
DUR = 5.0
N = int(DUR * FPS)
OW, OH = 1280, 720

BASE = dict(KY.KEY_POSE)


def P(**kw):
    d = dict(BASE)
    d.update(kw)
    return d


# pose keys: (time, pose).  Between keys we ease; holds are just repeated keys.
POINT = dict(lean=.08, handB=(200, -300), handBs="point", bendB=1, handBa=.32, handF=(-70, -244), handFs="hip", browang=1.0, brow=-.1,
             squint=.45, look=(.9, .3), mouth="open", jaw=.45, wide=.2, tongue=.4)
TAKE = dict(lean=-.18, hipx=-14, hipy=26, sy=1.0, sx=.96, eyesize=1.55, eye=1.0, squint=0, browang=-.6, brow=1.6, look=(.2, .1), hair_up=1.2,
            mouth="yell", jaw=1.0, wide=1.0, tongue=1.0, handF=(-200, -350), handFs="open", bendF=-1, handB=(160, -400), handBs="open",
            bendB=1, footF=(-58, -6), footB=(62, -4))
KEYS = [
    (0.00, P()),
    (0.92, P()),
    (1.08, P(lean=-.14, hipx=-10, squint=.7, browang=1.0, brow=-.2, look=(.9, .3), handB=(60, -300), handBs="fist")),  # wind-up
    (1.25, P(**POINT, jaw_flap=True)),  # snap: finger jab (talks 1.25-2.45)
    (2.45, P(**POINT, jaw_flap=True)),
    (2.60, P(**{**POINT, "mouth": "grit", "jaw": 0, "squint": .55})),
    (2.95, P(**{**POINT, "mouth": "grit", "jaw": 0, "squint": .55})),
    (3.04, P(**{**TAKE, "sy": 1.07, "sx": .86, "hipy": 10})),  # smear/stretch frame
    (3.12, P(**TAKE)),
    (3.62, P(**TAKE)),
    (3.80, P(eye=1.0, squint=0, eyesize=1.15, brow=.8, browang=-.4, look=(.6, .1), mouth="closed", smile=-.6, sy=.94, sx=1.05,
             handF=(-60, -252), handB=(64, -254))),  # gulp squash
    (4.05, P(eye=.95, squint=.1, brow=.4, browang=.2, smile=-.2)),
    (4.30, P()),
    (5.10, P()),
]
DD_T = 2.92  # dandelion boing time


def lerp(a, b, t):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return a + (b - a) * t
    if isinstance(a, tuple) and isinstance(b, tuple):
        return tuple(x + (y - x) * t for x, y in zip(a, b))
    return b if t >= .5 else a


def pose_at(t):
    for k in range(len(KEYS) - 1):
        t0, p0 = KEYS[k]
        t1, p1 = KEYS[k + 1]
        if t0 <= t < t1:
            u = (t - t0) / (t1 - t0)
            u = u * u * (3 - 2 * u)
            out = {}
            for key in set(p0) | set(p1):
                a, b = p0.get(key, CH.DEF.get(key)), p1.get(key, CH.DEF.get(key))
                out[key] = lerp(a, b, u)
            # strings that change mid-ease snap on the drawing nearest the key
            return out, p0.get("jaw_flap") and p1.get("jaw_flap")
    return dict(KEYS[-1][1]), False


_BG = {}


def bg():
    if "bg" not in _BG:
        _BG["bg"] = paint.backyard()
    return _BG["bg"]


def frame(i):
    k = i // 2 * 2  # on twos
    t = k / FPS
    p, talking = pose_at(t)
    p.pop("jaw_flap", None)
    if talking:  # mouth flaps per syllable (the synth voice uses the same syllable clock)
        syl = int((t - 1.25) / .125)
        p["jaw"] = (.15, .55, .35, .7, .2, .6, .4, .75, .25, .5)[syl % 10]
        p["wide"] = .15 + .2 * (syl % 3 == 0)
    if 3.12 <= t < 3.62:  # trembling hold in the take
        p["hipx"] = p.get("hipx", 0) + 3 * math.sin(i * 2.3)
        p["jaw"] = .9 + .1 * math.sin(i * 1.7)
    # dandelion: idle sway, then a springy boing with squash and stretch
    dd = dict(KY.DANDE)
    sway = .25 * math.sin(t * 2.1)
    bob = 0.0
    sc = 1.0
    if t >= DD_T:
        a = t - DD_T
        spring = math.exp(-a * 5) * math.sin(a * 26)
        bob = -26 * spring
        sway = sway - 1.2 * spring
        sc = 1 + .12 * spring
    dd.update(sway=sway, bob=bob)
    K.set_boil((i // 2) % 3, 1.0)
    surf, ctx = K.canvas()
    ctx.save()
    ctx.translate(dd["x"], dd["y"])
    ctx.scale(1 / sc ** .5, sc)
    ctx.translate(-dd["x"], -dd["y"])
    CH.dandelion(ctx, **dd)
    ctx.restore()
    if t >= 3.04 and t < 3.5:  # pop lines around the take
        K.set_boil((i // 2) % 3, 1.2)
        cx, cy = BASE["x"] + 40, BASE["y"] - 420
        for m in range(9):
            a = -math.pi + m * math.pi / 8
            r0, r1 = 250, 290 + 20 * (m % 2)
            K.line(ctx, [(cx + r0 * math.cos(a), cy + r0 * math.sin(a) * .8), (cx + r1 * math.cos(a), cy + r1 * math.sin(a) * .8)],
                   f"pop{m}", lw=4.0)
    CH.wendell(ctx, CH.pose(**p))
    cel = paint.to_rgba(surf)
    img = paint.over(bg(), cel)
    # a soft broadcast-cel finish: tiny chroma bleed and film grain, nothing that fights the line
    yuv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2YCrCb)
    yuv[:, :, 1:] = cv2.GaussianBlur(yuv[:, :, 1:], (0, 0), sigmaX=1.2, sigmaY=.3)
    img = cv2.cvtColor(yuv, cv2.COLOR_YCrCb2RGB)
    g = cv2.resize(np.random.default_rng(900 + i).normal(0, 1, (K.H // 2, K.W // 2)).astype(np.float32), (K.W, K.H))
    img = img + g[..., None] * .008
    img = cv2.resize(np.clip(img, 0, 1), (OW, OH), interpolation=cv2.INTER_CUBIC)
    return (np.clip(img, 0, 1) * 255 + .5).astype(np.uint8)
