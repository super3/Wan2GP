"""Cast for "Clean Sweep": Bramble the javelina (an original supporting character in the same 1950s theatrical cel style),
Pip's rolled-up ball pose, and the dusty / muddy treatments applied to a character's cel layer."""
import math

import cairo
import cv2
import numpy as np

import desert as D
from desert import (INK, Rm, Sm, Tm, ap, ap1, circ, crs, fillp, hexc, ink, mix, shape, tline, tube, clip_shape, closed_widths, alpha)

BC = dict(
    body=hexc("#8b7f96"), body_dk=hexc("#5f5570"), far=hexc("#7a6f86"), collar=hexc("#f2e5ca"), snout=hexc("#eba39c"),
    nostril=hexc("#7c3b45"), tusk=hexc("#fffaf0"), hoof=hexc("#2a2026"), ear_in=hexc("#d197a1"), white=hexc("#fffdf6"),
    mouth=hexc("#5a1e2c"), tongue=hexc("#e2747a"), scarf=hexc("#d7403a"), dot=hexc("#ffe9d0"),
)

BRAMBLE = dict(
    x=900.0, gy=652.0, facing=1.0, sc=1.0, rot=0.0, piv=(20.0, -130.0), dy=0.0,
    head_rot=0.0, head_dx=0.0, head_dy=0.0,
    eyes="open", pup_x=-.3, pup_y=.1, lid=.25, brow=0.0, brow_tilt=0.0,
    mouth="grin", grin=1.0, open_=0.0, ear_rot=0.0,
    feet=((-82, 0), (-52, 0), (86, 0), (114, 0)),   # far-front, near-front, far-rear, near-rear (root space)
    scarf=False, shadow=True,
)

BODY = [(-100, -150), (-110, -112), (-92, -72), (-40, -58), (30, -58), (92, -66), (132, -92), (150, -140), (142, -184), (96, -214),
        (24, -236), (-40, -232), (-84, -210)]
CREST = [(-84, -214), (-40, -236), (24, -240), (96, -218), (142, -186)]
HIPS = ((-72, -82), (-48, -76), (84, -80), (110, -90))
NECK = (-88, -178)


def _root(P):
    return Tm(P["x"], P["gy"]) @ Sm(P["facing"] * P["sc"], P["sc"])


def _body(P):
    if P.get("roll"):
        return _root(P) @ Tm(0, -244 + P["dy"]) @ Sm(1, -1)
    px, py = P["piv"]
    return _root(P) @ Tm(0, P["dy"]) @ Tm(px, py) @ Rm(P["rot"]) @ Tm(-px, -py)


def _leg(ctx, Mr, Mb, hip, foot, col, near, bend):
    H = ap1(Mb, *hip)
    # work in root space: convert the hip into root space
    inv = np.linalg.inv(Mr)
    h = ap1(inv, *H)
    E, F = D.ik(h, foot, 46, 44, bend)
    tube(ctx, [tuple(h), tuple(E), tuple(F)], 19, col, Mr, lw=2.6)
    d = F - E
    ang = math.atan2(d[1], d[0]) - math.pi / 2
    Mh = Mr @ Tm(F[0], F[1]) @ Rm(ang)
    for dx in (-5.5, 5.5):
        shape(ctx, [(dx - 6, -4), (dx + 6, -4), (dx + 7, 6), (dx, 9), (dx - 7, 6)], BC["hoof"], D.LI * .9, Mh, tension=.4)


def bramble(ctx, P):
    Mr, Mb = _root(P), _body(P)
    sc = P["sc"]
    if P["shadow"]:
        fillp(ctx, circ(P["x"] + 18 * sc * np.sign(P["facing"]), P["gy"] + 2, 150 * sc, 12 * sc, 48), D.PC["shadow"])
    ff, nf, fr, nr = P["feet"]
    # far legs behind the body
    bs = -1 if P.get("roll") else 1
    _leg(ctx, Mr, Mb, HIPS[0], ff, BC["far"], False, -bs)
    _leg(ctx, Mr, Mb, HIPS[2], fr, BC["far"], False, bs)
    # bristly crest along the spine (spikes poke above the back line)
    Cp = crs(CREST, False, 8)
    d = np.gradient(Cp, axis=0)
    nrm = np.stack([d[:, 1], -d[:, 0]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    spikes = []
    for k in range(len(Cp)):
        spikes.append(Cp[k] + nrm[k] * (16 if k % 2 == 0 else 2))
    spikes = np.vstack([spikes, Cp[::-1] - nrm[::-1] * 10])
    S = ap(Mb, spikes)
    fillp(ctx, S, BC["body_dk"])
    ink(ctx, S, closed_widths(S, D.LW * .9), closed=True)
    # body
    B = shape(ctx, BODY, BC["body"], D.LW, Mb)
    ctx.save()
    clip_shape(ctx, B)
    # pale collar stripe from the shoulder hump down round the chest
    shape(ctx, [(-58, -250), (-22, -250), (-62, -170), (-82, -112), (-130, -110), (-104, -170)], BC["collar"], 0, Mb, line=None, tension=.6)
    tline(ctx, [(-22, -246), (-62, -170), (-82, -112)], D.LI * .8, Mb, col=mix(BC["collar"], BC["body_dk"], .5), t0=.3, t1=.3)
    # bristle ticks
    rng = np.random.default_rng(5)
    for k in range(26):
        x, y = rng.uniform(-60, 140), rng.uniform(-220, -80)
        tline(ctx, [(x, y), (x + 6, y - 9)], D.LI * .6, Mb, col=BC["body_dk"], t0=.3, t1=.1)
    ctx.restore()
    ink(ctx, B, closed_widths(B, D.LW), closed=True)
    tline(ctx, [(140, -150), (154, -158), (160, -148), (152, -140)], D.LI * 1.1, Mb, t0=.6, t1=.2)   # little tail curl
    # near legs in front
    _leg(ctx, Mr, Mb, HIPS[1], nf, BC["body"], True, -bs)
    _leg(ctx, Mr, Mb, HIPS[3], nr, BC["body"], True, bs)
    if P["scarf"]:
        shape(ctx, [(-70, -200), (-104, -168), (-112, -118), (-86, -146), (-66, -170)], BC["scarf"], D.LI * 1.2, Mb, tension=.6)
        for d2 in ((-92, -160), (-100, -140), (-82, -176)):
            fillp(ctx, ap(Mb, circ(d2[0], d2[1], 2.6, 2.2, 10)), BC["dot"])
    if P.get("roll"):
        Mh = Mr @ Tm(NECK[0] + 6 + P["head_dx"], -58 + P["dy"] + P["head_dy"]) @ Rm(P["head_rot"]) @ Sm(1.22)
    else:
        Mh = Mb @ Tm(NECK[0] + P["head_dx"], NECK[1] + P["head_dy"]) @ Rm(P["head_rot"]) @ Sm(P.get("head_flip", 1.0) * 1.22, 1.22)
    head(ctx, Mh, P)


def head(ctx, M, P):
    # ears (behind the head)
    for base, ang, col in (((-6, -62), -.35 + P["ear_rot"], BC["far"]), ((14, -58), .25 + P["ear_rot"], BC["body"])):
        Me = M @ Tm(*base) @ Rm(ang)
        shape(ctx, [(-12, 4), (-14, -18), (-4, -40), (4, -40), (12, -16), (11, 4)], col, D.LW * .9, Me)
        shape(ctx, [(-6, -2), (-7, -16), (-2, -30), (5, -14), (5, -2)], BC["ear_in"], 0, Me, line=None)
    hd = [(22, -8), (20, -50), (2, -70), (-36, -72), (-72, -54), (-104, -36), (-126, -22), (-134, -6), (-130, 12), (-112, 24), (-80, 32),
          (-44, 34), (-10, 28), (16, 14)]
    Hd = shape(ctx, hd, BC["body"], D.LW, M)
    ctx.save()
    clip_shape(ctx, Hd)
    shape(ctx, [(30, -40), (10, -30), (-10, 0), (0, 40), (40, 40)], BC["collar"], 0, M, line=None, tension=.6)   # collar reaches the jaw
    ctx.restore()
    ink(ctx, Hd, closed_widths(Hd, D.LW), closed=True)
    for k, (x, y) in enumerate(((-30, -70), (-14, -72), (2, -66))):   # head tuft
        shape(ctx, [(x - 7, y + 6), (x - 2 + 4 * k, y - 16 - 3 * k), (x + 7, y + 6)], BC["body_dk"], D.LI, M, tension=.3)
    # mouth
    mo = P["mouth"]
    if mo == "open":
        o = max(P["open_"], .4)
        Mo = shape(ctx, [(-120, 18), (-90, 22), (-56, 18), (-50, 12), (-58, 22 + 26 * o), (-92, 30 + 28 * o), (-116, 26 + 10 * o)], BC["mouth"],
                   D.LI * 1.1, M, tension=.8)
        fillp(ctx, ap(M, circ(-84, 26 + 20 * o, 16, 7 * o + 2, 20)), BC["tongue"])
        ink(ctx, Mo, D.LI * 1.1, closed=True)
    elif mo == "sheepish":
        tline(ctx, [(-118, 18), (-100, 22), (-86, 18), (-72, 22), (-58, 18)], D.LI * 1.1, M, t0=.3, t1=.3)
    elif mo == "flat":
        tline(ctx, [(-118, 20), (-90, 22), (-58, 20)], D.LI * 1.1, M, t0=.3, t1=.3)
    else:
        g = P["grin"]
        tline(ctx, [(-120, 16), (-96, 22 + 4 * g), (-66, 20 + 2 * g), (-48, 10 - 6 * g)], D.LI * 1.15, M, t0=.3, t1=.6)
        tline(ctx, [(-44, 2 - 4 * g), (-42, 14 - 4 * g), (-48, 22)], D.LI * .8, M, t0=.2, t1=.2)   # cheek crease
    # tusk poking up from the lower lip
    shape(ctx, [(-112, 22), (-104, 22), (-110, 6)], BC["tusk"], D.LI * .9, M, tension=.3)
    # snout disc
    Sd = ap(M, circ(-134, 4, 11, 20, 32, rot=.12))
    fillp(ctx, Sd, BC["snout"])
    ink(ctx, Sd, D.LI * 1.2, closed=True)
    for ny in (-4, 12):
        fillp(ctx, ap(M, circ(-136, ny, 3.2, 4.4, 12)), BC["nostril"])
    # eyes (Jones-style: two touching tall ovals), brows
    for (cx, cy, rx, ry), near in (((-64, -46, 10, 14.5), False), ((-45, -51, 12.5, 17.5), True)):
        if P["eyes"] in ("happy", "shut"):
            curve = [(cx - rx, cy + 2), (cx, cy - (8 if P["eyes"] == "happy" else -2)), (cx + rx, cy + 2)]
            tline(ctx, curve, D.LI * 1.4, M, t0=.4, t1=.4)
            continue
        E = ap(M, circ(cx, cy, rx, ry, 40))
        fillp(ctx, E, BC["white"])
        ctx.save()
        clip_shape(ctx, E)
        fillp(ctx, ap(M, circ(cx + P["pup_x"] * rx * .55, cy + P["pup_y"] * ry * .5 + 2, rx * .48, ry * .5, 24)), INK)
        lid = P["lid"]
        if lid > .02:
            top = cy - ry - 2
            yl = top + (2 * ry + 4) * lid
            L = ap(M, [(cx - rx * 2, top - 30), (cx + rx * 2, top - 30), (cx + rx * 2, yl), (cx - rx * 2, yl)])
            fillp(ctx, L, BC["body"])
            ink(ctx, ap(M, [(cx - rx * 1.3, yl), (cx + rx * 1.3, yl)]), D.LI * 1.1)
        ctx.restore()
        ink(ctx, E, closed_widths(E, D.LI, .5), closed=True)
    b, bt = P["brow"], P["brow_tilt"]
    tline(ctx, [(-74, -64 - 8 * b - 4 * bt), (-64, -68 - 9 * b), (-55, -66 - 8 * b + 6 * bt)], D.LI * 1.4, M, t0=.2, t1=.5)
    tline(ctx, [(-56, -70 - 9 * b + 6 * bt), (-44, -74 - 10 * b), (-31, -70 - 9 * b - 5 * bt)], D.LI * 1.4, M, t0=.5, t1=.2)


def bramble_fluff(ctx, x, gy, sc=1.0, facing=1.0, mood="beam", seed=7):
    """Bramble after the storm: a giant fluffy dust ball with his snout, eyes, ears and hooves poking out."""
    M = Tm(x, gy) @ Sm(facing * sc, sc)
    rng = np.random.default_rng(seed)
    fillp(ctx, ap(M, circ(10, 2, 190, 13, 48)), D.PC["shadow"])
    # hooves under the ball
    for hx in (-90, -50, 70, 110):
        shape(ctx, [(hx - 12, -14), (hx + 12, -14), (hx + 13, 2), (hx - 13, 2)], BC["hoof"], D.LI, M, tension=.4)
    puffs = []
    for k in range(22):
        a = k / 22 * 2 * np.pi
        r = rng.uniform(46, 70)
        puffs.append((10 + 150 * math.cos(a) * rng.uniform(.85, 1.0), -150 + 128 * math.sin(a) * rng.uniform(.85, 1.0), r))
    puffs.append((10, -150, 140))
    P = [(ap1(M, px, py), r * sc) for px, py, r in puffs]
    for (cx, cy), r in P:
        fillp(ctx, circ(cx, cy, r + 3.2, r * .9 + 3.2, 28), INK)
    for (cx, cy), r in P:
        fillp(ctx, circ(cx, cy, r, r * .9, 28), hexc("#ecd6ad"))
    for (cx, cy), r in P[:-1]:
        fillp(ctx, circ(cx + r * .2, cy + r * .32, r * .7, r * .4, 20), hexc("#dbbd8d"))
    for (cx, cy), r in P[:10]:
        ink(ctx, circ(cx - r * .1, cy, r * .5, r * .36, 20)[3:11], 1.6, alpha(hexc("#a98457"), 1))
    # ears on top, snout + eyes peeking from the front
    for bx, ang in ((-52, -.45), (-14, .2)):
        Me = M @ Tm(bx, -270) @ Rm(ang) @ Sm(1.2)
        shape(ctx, [(-12, 4), (-15, -16), (-6, -34), (5, -34), (13, -14), (11, 4)], BC["body"], D.LW * .9, Me)
        shape(ctx, [(-6, -2), (-8, -14), (-2, -26), (6, -12), (5, -2)], BC["ear_in"], 0, Me, line=None)
    Mh = M @ Tm(-62, -140)
    Sd = ap(Mh, circ(-96, 6, 17, 28, 32, rot=.12))
    fillp(ctx, Sd, BC["snout"])
    ink(ctx, Sd, D.LI * 1.2, closed=True)
    for ny in (-4, 15):
        fillp(ctx, ap(Mh, circ(-98, ny, 3.8, 5.4, 12)), BC["nostril"])
    tline(ctx, [(-84, 24), (-64, 28), (-44, 18)], D.LI * 1.1, Mh, t0=.3, t1=.6)
    for (cx, cy, rx, ry) in ((-34, -52, 12, 17), (-10, -56, 14.5, 20.5)):
        E = ap(Mh, circ(cx, cy, rx, ry, 40))
        fillp(ctx, E, BC["white"])
        ctx.save()
        clip_shape(ctx, E)
        fillp(ctx, ap(Mh, circ(cx - 3, cy + 2, rx * .48, ry * .5, 24)), INK)
        if mood != "beam":
            fillp(ctx, ap(Mh, [(cx - 30, cy - 40), (cx + 30, cy - 40), (cx + 30, cy - ry * .3), (cx - 30, cy - ry * .3)]), BC["body"])
        ctx.restore()
        ink(ctx, E, closed_widths(E, D.LI, .5), closed=True)


def pip_ball(ctx, x, gy, sc=1.0, facing=1.0, peek=0.0):
    """Pip rolled up into a tight banded ball: head plate, ear tips and tail tip tucked in; peek>0 lets one eye show."""
    M = Tm(x, gy) @ Sm(facing * sc, sc)
    fillp(ctx, ap(M, circ(10, 2, 96, 11, 40)), D.PC["shadow"])
    R = 92
    ball = ap(M, circ(0, -R, R, R * .96, 64))
    fillp(ctx, ball, D.PC["shell"])
    ctx.save()
    clip_shape(ctx, ball)
    for k, a in enumerate(np.linspace(-.9, .9, 7)):   # bands wrap round the ball
        xs = R * math.sin(a)
        pts = [(xs - 10 * a, -R - R * .97), (xs + 8 * math.sin(a), -R), (xs - 10 * a, -R + R * .97)]
        tline(ctx, pts, D.LI * 1.05, M, t0=.6, t1=.6)
        for yy in np.linspace(-R * .8, R * .8, 6):
            tline(ctx, [(xs + 8 + 4 * a, -R + yy), (xs + 20 + 4 * a, -R + yy + 2)], D.LI * .6, M, col=D.PC["shell_dk"], t0=.4, t1=.4)
    ctx.restore()
    ink(ctx, ball, closed_widths(ball, D.LW), closed=True)
    # tucked head: the armour plate shows as a lighter cap low on the front, the pink nose tip under it, ear tips behind
    for bx, by, ang in ((-58, -150, -.9), (-40, -160, -.55)):
        Me = M @ Tm(bx, by) @ Rm(ang) @ Sm(.55)
        shape(ctx, [(-12, 4), (-16, -24), (-6, -52), (4, -54), (12, -30), (11, 4)], D.PC["skin"], D.LW * .9, Me)
    cap = [(-96, -70), (-88, -104), (-66, -126), (-46, -122), (-56, -96), (-72, -66)]
    shape(ctx, cap, D.PC["shell_lt"], D.LI * 1.2, M, tension=.9)
    for k in range(3):
        tline(ctx, [(-90 + 8 * k, -78 - 14 * k), (-74 + 8 * k, -92 - 14 * k)], D.LI * .7, M, col=D.PC["shell_dk"], t0=.3, t1=.3)
    shape(ctx, [(-98, -66), (-100, -52), (-88, -46), (-80, -58)], D.PC["skin"], D.LI * 1.1, M, tension=.8)
    fillp(ctx, ap(M, circ(-96, -52, 6, 5, 14)), D.PC["nose"])
    # tail tip curling out along the ground at the back
    tail = ap(M, crs([(70, -20), (104, -8), (128, -12), (136, -26)], False, 10))
    shape(ctx, None, D.PC["shell"], D.LW * .9, dense=D.ribbon(tail, 16, 5))
    if peek > 0:
        E = ap(M, circ(-70, -60, 10, 13 * peek + 1, 30))
        fillp(ctx, E, D.PC["white"])
        fillp(ctx, ap(M, circ(-73, -58, 4.5, 6 * peek + .5, 20)), INK)
        ink(ctx, E, D.LI, closed=True)


# ------------------------------------------------------------------------------------------------ layer treatments
def treat(rgb, a, kind, amount=1.0, seed=0):
    """Post-process a character's premultiplied cel layer: 'dust' greys every flat colour toward dust (ink kept) and adds specks;
    'mud' darkens toward wet mud brown."""
    if kind is None or amount <= 0:
        return rgb, a
    eps = 1e-6
    col = rgb / np.maximum(a, eps)
    lum = col @ np.array([.299, .587, .114], np.float32)
    keep = np.clip((lum - .2) / .15, 0, 1)[..., None]   # protect the ink lines
    if kind == "dust":
        target = np.array([.86, .76, .62], np.float32)
        k = .5 * amount
    else:
        target = np.array([.45, .31, .22], np.float32)
        k = .62 * amount
    col = col * (1 - k * keep) + target * (k * keep)
    rng = np.random.default_rng(seed)
    if kind == "dust":
        sp = rng.uniform(0, 1, a.shape[:2]).astype(np.float32)
        sp = (sp > .985).astype(np.float32)
        sp = cv2.GaussianBlur(sp, (0, 0), .7)[..., None] * 3
        col = col * (1 - .25 * np.clip(sp, 0, 1) * keep)
    return np.clip(col, 0, 1) * a, a


def splats(ctx, pts, seed=1, col=hexc("#5b3d29")):
    """Mud splats: blobs with little satellite drops and one drip each.  Draw with cairo.OPERATOR_ATOP onto a character layer."""
    rng = np.random.default_rng(seed)
    for (x, y, r) in pts:
        n = 9
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rad = r * rng.uniform(.65, 1.25, n)
        P = np.stack([x + rad * np.cos(a), y + rad * np.sin(a)], 1)
        fillp(ctx, crs(P, True, 8), col)
        for k in range(4):
            aa = rng.uniform(0, 2 * np.pi)
            d = r * rng.uniform(1.4, 2.2)
            fillp(ctx, circ(x + d * math.cos(aa), y + d * math.sin(aa), r * rng.uniform(.15, .3), None, 12), col)
        fillp(ctx, np.array([(x - r * .25, y), (x + r * .25, y), (x + r * .18, y + r * 1.6), (x, y + r * 1.9), (x - r * .18, y + r * 1.6)]), col)
