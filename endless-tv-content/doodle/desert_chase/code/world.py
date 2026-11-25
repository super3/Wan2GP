"""The extended painted world for the short "Clean Sweep": Pip's home ledge (the proto background) continued to the right past a
neat row of pebbles (the property line) to Bramble's dusty wallow.  Cameras pan along it with Noble-style flat parallax layers
(sky, far mesas, canyon floor, buttes, ledge) or zoom in.  Variants: sunset, the dust storm rising, engulfed, and the aftermath
(dust drifts on everything except the clean wedge in Bramble's lee)."""
import math

import cairo
import cv2
import numpy as np

import desert as D
from desert import (BG, H, SS, W, alpha, circ, crs, drybrush, fillp, hexc, ink, mix, tline, clip_shape, path, offreg)

PAR = dict(sky=.08, far=.3, floor=.5, butte=.75, ledge=1.0)
CAMS = dict(
    A=dict(x0=0.0, zoom=1.0, cx=640.0, cy=360.0),     # Pip's home (the proto framing)
    C=dict(x0=0.0, zoom=1.85, cx=560.0, cy=330.0),    # close on Pip at home
    B=dict(x0=1300.0, zoom=1.0, cx=640.0, cy=360.0),  # Bramble's wallow
    D=dict(x0=1000.0, zoom=1.0, cx=640.0, cy=360.0),  # two-shot across the property line
)
HOLLOW = (1800.0, 642.0)     # world position of Bramble's wallow (ledge layer)
PROPERTY_X = 1460.0
SUNSET = dict(sky_top=hexc("#7a73b8"), sky_mid=hexc("#ee9a86"), sky_low=hexc("#ffd28e"), cloud=hexc("#ffe2c6"), cloud_under=hexc("#c88ba8"))
DUSTC = np.array([.93, .82, .64], np.float32)
STORMC = np.array([.78, .55, .34], np.float32)


def apply_cam(ctx, cam, par):
    c = CAMS[cam] if isinstance(cam, str) else cam
    ctx.translate(c["cx"], c["cy"])
    ctx.scale(c["zoom"], c["zoom"])
    ctx.translate(-c["cx"], -c["cy"])
    ctx.translate(-c["x0"] * par, 0)


def screen_x(cam, wx, par=1.0):
    c = CAMS[cam]
    return c["cx"] + (wx - c["x0"] * par - c["cx"]) * c["zoom"]


def to_screen(cam, x, y, par=1.0):
    c = CAMS[cam]
    return c["cx"] + (x - c["x0"] * par - c["cx"]) * c["zoom"], c["cy"] + (y - c["cy"]) * c["zoom"]


# ------------------------------------------------------------------------------------------------ extensions (world x > 1280)
def sky_fill(ctx, pal):
    g = cairo.LinearGradient(0, 0, 0, 400)
    g.add_color_stop_rgba(0, *pal["sky_top"])
    g.add_color_stop_rgba(.55, *pal["sky_mid"])
    g.add_color_stop_rgba(1, *pal["sky_low"])
    ctx.rectangle(-400, -400, 4000, 820)
    ctx.set_source(g)
    ctx.fill()


def clouds(ctx, pal, groups, seed):
    rng = np.random.default_rng(seed)
    for gx, gy, L, h in groups:
        for j in range(4):
            ll = L * rng.uniform(.45, 1.0) * (1 - .12 * j)
            ox = gx + rng.uniform(-.3, .3) * L
            oy = gy + j * h * 1.7 + rng.uniform(-2, 2)
            hh = h * rng.uniform(.55, 1.0) * (1 - .15 * j)
            s = np.linspace(0, 1, 40)
            prof = np.sin(np.pi * s) ** .8 * (1 + .25 * np.sin(2 * np.pi * s * rng.uniform(1, 2.5) + rng.uniform(0, 6)))
            xs = ox - ll / 2 + ll * s
            top = np.stack([xs, oy - hh * prof], 1)
            bot = np.stack([xs[::-1], (oy + hh * .25 * prof)[::-1]], 1)
            fillp(ctx, np.vstack([top, bot]) + [3, 3], alpha(pal["cloud_under"], .9))
            fillp(ctx, np.vstack([top, bot]), pal["cloud"])
            drybrush(ctx, xs[4], oy - hh * .3, xs[-5], oy - hh * .25, hh * .5, alpha(hexc("#ffffff"), .5), rng, broken=.6, wob=.5)


def sky(ctx, pal):
    """Sky with the proto's cloud groups plus more to the right (sky layer barely moves)."""
    clouds(ctx, pal, ((250, 120, 380, 10), (690, 196, 220, 7), (1120, 66, 420, 11), (1210, 170, 200, 7)), 11)
    clouds(ctx, pal, ((1560, 120, 300, 9),), 12)


def far_ext(ctx):
    prof = [(1280, D.HORIZON), (1280, 374), (1330, 372), (1350, 350), (1460, 346), (1480, 368), (1560, 370), (1580, 356), (1640, 354),
            (1660, 376), (1760, 378), (1790, 360), (1900, 358), (1920, 380), (2000, 382), (2000, D.HORIZON)]
    fillp(ctx, np.array(prof, float), BG["far"])
    for a, b in (((1350, 350), (1376, 372)), ((1580, 356), (1606, 376)), ((1790, 360), (1816, 380))):
        fillp(ctx, np.array([a, (a[0] + 26, a[1]), (b[0] + 20, D.HORIZON), (b[0] - 8, D.HORIZON)], float), BG["far_sh"])
    ctx.rectangle(1270, D.HORIZON - 2, 800, 6)
    ctx.set_source_rgba(*BG["far2"])
    ctx.fill()


def floor_fill(ctx):
    g = cairo.LinearGradient(0, D.HORIZON, 0, 600)
    g.add_color_stop_rgba(0, *BG["floor_far"])
    g.add_color_stop_rgba(1, *BG["floor_near"])
    ctx.rectangle(-400, D.HORIZON + 1, 4000, 260)
    ctx.set_source(g)
    ctx.fill()


def floor_ext(ctx):
    rng = np.random.default_rng(19)
    for k in range(50):
        y = D.HORIZON + 4 + (rng.uniform() ** 1.6) * 190
        depth = (y - D.HORIZON) / 190
        x = rng.uniform(1260, 2100)
        L = rng.uniform(60, 260) * (.4 + depth)
        drybrush(ctx, x, y, x + L, y + rng.normal(0, 1), .8 + 2.2 * depth, alpha(mix(BG["floor_near"], BG["floor_dot"], .35), .35), rng)
    for x0, x1, top, base in ((1350, 1460, 428, 450), (1620, 1700, 418, 436), (1800, 1950, 432, 452)):
        body = np.array([(x0, base), (x0 + 6, top + 4), (x0 + 14, top), (x1 - 12, top), (x1 - 4, top + 5), (x1, base)], float)
        fillp(ctx, body, mix(BG["far2"], BG["floor_far"], .35))
        shade = np.array([(x1 - (x1 - x0) * .3, top), (x1 - 12, top), (x1 - 4, top + 5), (x1, base), (x1 - (x1 - x0) * .32, base)], float)
        fillp(ctx, shade, mix(BG["far"], BG["floor_far"], .3))
    for k in range(30):
        y = D.HORIZON + 8 + rng.uniform() ** 1.2 * 160
        x = rng.uniform(1290, 2100)
        r = .8 + 2.4 * (y - D.HORIZON) / 170
        for c in range(3):
            fillp(ctx, circ(x + (c - 1) * r * 1.3, y - (c == 1) * r * .5, r * (1.1 if c == 1 else .8), r * .7, 12), alpha(BG["floor_dot"], .75))
    for x0, top, wd in ((1520, 480, 26), (1690, 500, 18)):
        pts = [(x0 - wd / 2, 560), (x0 - wd / 2 + 3, top + 14), (x0 - wd * .7, top + 4), (x0 - wd * .2, top - 6), (x0 + wd * .6, top + 2),
               (x0 + wd / 2 - 2, top + 14), (x0 + wd / 2, 560)]
        fillp(ctx, np.array(pts, float), BG["hoodoo"])
    g = cairo.LinearGradient(0, 470, 0, 580)
    g.add_color_stop_rgba(0, *alpha(BG["sky_low"], 0))
    g.add_color_stop_rgba(1, *alpha(BG["sky_low"], .55))
    ctx.rectangle(1290, 470, 900, 120)
    ctx.set_source(g)
    ctx.fill()


def buttes_ext(ctx):
    rng = np.random.default_rng(23)
    # twin columns joined at the base
    lit = [(1480, 600), (1486, 420), (1478, 330), (1490, 230), (1486, 170), (1500, 150), (1548, 146), (1556, 170), (1552, 260), (1566, 300),
           (1580, 214), (1594, 200), (1630, 198), (1640, 226), (1636, 330), (1648, 440), (1656, 600)]
    sh = [(1530, 150), (1548, 146), (1556, 170), (1552, 260), (1566, 300), (1572, 600), (1540, 600), (1546, 420), (1536, 300), (1542, 200),
          ]
    sh2 = [(1614, 200), (1630, 198), (1640, 226), (1636, 330), (1648, 440), (1656, 600), (1620, 600), (1626, 420), (1618, 300)]
    D.butte(ctx, lit, sh, [(1490, 160), (1500, 150), (1548, 146), (1556, 166), (1520, 168)], BG["spire"], BG["spire_sh"], BG["spire_cap"],
            BG["spire_line"], rng, strata=(190, 240, 300, 360, 430, 500), vert=True,
            edge=([(1500, 150), (1548, 146), (1556, 170), (1552, 260)], [(1594, 200), (1630, 198), (1640, 226), (1636, 330), (1648, 440)]))
    fillp(ctx, np.array(sh2, float), BG["spire_sh"])
    # a balanced rock: a boulder perched on a thin pillar (Noble-ish whimsy)
    fillp(ctx, np.array([(1736, 600), (1740, 430), (1748, 380), (1764, 380), (1770, 430), (1776, 600)], float), mix(BG["mesa"], BG["far2"], .3))
    fillp(ctx, np.array([(1758, 380), (1764, 380), (1770, 430), (1776, 600), (1760, 600), (1762, 430)], float), mix(BG["mesa_sh"], BG["far"], .3))
    boulder = circ(1756, 352, 40, 30, 24)
    fillp(ctx, boulder, mix(BG["mesa"], BG["far2"], .25))
    ctx.save()
    clip_shape(ctx, boulder)
    fillp(ctx, circ(1782, 362, 36, 30, 24), mix(BG["mesa_sh"], BG["far"], .25))
    ctx.restore()
    offreg(ctx, list(boulder[2:16]), alpha(BG["mesa_line"], .8), w=1.4, rng=rng, n=2, dx=3, dy=-2)
    # a broad flat-topped mesa further right
    lit = [(1850, 600), (1856, 400), (1848, 330), (1866, 290), (1890, 272), (2240, 268), (2262, 290), (2270, 600)]
    sh = [(2120, 272), (2240, 268), (2262, 290), (2270, 600), (2130, 600), (2140, 460), (2126, 350)]
    D.butte(ctx, lit, sh, [(1880, 284), (1890, 270), (2240, 266), (2244, 286), (1900, 290)], BG["mesa"], BG["mesa_sh"],
            mix(BG["mesa"], BG["spire_cap"], .5), BG["mesa_line"], rng, strata=(318, 366, 420, 480, 540), vert=False,
            edge=([(1890, 272), (2240, 268), (2262, 290), (2270, 560)], [(2126, 350), (2140, 460), (2130, 580)]))


LEDGE_EXT = [(1200, 576), (1290, 572), (1400, 568), (1520, 576), (1640, 570), (1760, 575), (1880, 569), (2000, 577), (2140, 570),
             (2280, 574), (2420, 568), (2600, 573), (2700, 572)]


def ledge_ext(ctx):
    edge = crs(LEDGE_EXT, False, 10, .8)
    P = np.vstack([edge, [(2700, 730), (1200, 730)]])
    g = cairo.LinearGradient(0, 560, 0, 720)
    g.add_color_stop_rgba(0, *BG["ledge"])
    g.add_color_stop_rgba(1, *BG["ledge2"])
    path(ctx, P)
    ctx.set_source(g)
    ctx.fill()
    rng = np.random.default_rng(34)
    ctx.save()
    clip_shape(ctx, P)
    ctx.rectangle(1286, 0, 2000, 800)  # only paint texture right of the proto ledge (it already has its own)
    ctx.clip()
    for k in range(140):
        y = 572 + rng.uniform() ** .8 * 150
        x = rng.uniform(1240, 2700)
        L = rng.uniform(30, 160)
        light = rng.uniform() < .4
        drybrush(ctx, x, y, x + L, y + rng.normal(0, 1), rng.uniform(1, 3.5),
                 alpha(BG["lip"] if light else mix(BG["ledge2"], BG["crack"], .5), .32 if light else .3), rng)
    for k in range(300):
        x, y = rng.uniform(1286, 2700), rng.uniform(572, 720)
        ctx.new_sub_path()
        ctx.arc(x, y, rng.uniform(.6, 1.8), 0, 2 * math.pi)
        ctx.set_source_rgba(*alpha(BG["crack"], rng.uniform(.15, .4)))
        ctx.fill()
    ctx.restore()
    e2 = edge[edge[:, 0] > 1270]
    ink(ctx, e2 + [0, 2.5], 4.0, alpha(BG["lip"], .95))
    ink(ctx, e2 + [2, -1], 1.6, alpha(BG["crack"], .9))
    for pts in (((1330, 650), (1370, 660), (1400, 654)), ((2140, 690), (2190, 676), (2230, 684)), ((2400, 620), (2440, 632), (2470, 626))):
        tline(ctx, pts, 2.0, col=alpha(BG["crack"], .85), t0=.1, t1=.1)
    for x, y, r in ((1330, 606, 6), (2260, 640, 8), (2282, 648, 4), (2500, 690, 9), (2380, 700, 6)):
        fillp(ctx, circ(x + r * .5, y + r * .45, r * 1.25, r * .45), alpha(BG["crack"], .45))
        fillp(ctx, circ(x, y, r, r * .72), BG["peb"])
        fillp(ctx, circ(x - r * .25, y - r * .25, r * .55, r * .35), BG["peb_lt"])


def props_ext(ctx):
    rng = np.random.default_rng(57)
    # Pip's side: neat, evenly spaced broom-stroke arcs (he sweeps right up to his line)
    for k, x in enumerate(np.arange(860, 1440, 46)):
        for row, y in enumerate((610, 652, 694)):
            xx = x + (row % 2) * 23
            if xx > PROPERTY_X - 20:
                continue
            a = np.linspace(-.5, .5, 12)
            pts = np.stack([xx + 26 * np.sin(a), y + 6 * np.cos(a) - 6], 1)
            ink(ctx, pts, np.linspace(.3, 1.4, 12), alpha(mix(BG["ledge"], BG["lip"], .7), .9))
    # the property line: a neat row of little white pebbles, receding to the ledge edge
    for k, t in enumerate(np.linspace(0, 1, 13)):
        y = 584 + 128 * t ** 1.1
        x = PROPERTY_X - 10 * (1 - t)
        r = 3.2 + 4.2 * t
        fillp(ctx, circ(x + r * .4, y + r * .45, r * 1.2, r * .4), alpha(BG["crack"], .5))
        fillp(ctx, circ(x, y, r, r * .7, 14), hexc("#fbf3e2"))
        ink(ctx, circ(x, y, r, r * .7, 14)[3:11], .9, alpha(BG["crack"], .8))
    # Bramble's side: scuffed dust tufts and hoof prints
    for k in range(26):
        x, y = rng.uniform(1500, 2560), rng.uniform(600, 715)
        if abs(x - HOLLOW[0]) < 190 and abs(y - HOLLOW[1]) < 50:
            continue
        s = .6 + (y - 600) / 115
        fillp(ctx, circ(x, y, 9 * s, 3.4 * s, 16), alpha(BG["crack"], .5))
        fillp(ctx, circ(x + 6 * s, y + 1, 7 * s, 2.8 * s, 16), alpha(BG["crack"], .5))
    for k in range(10):
        x, y = rng.uniform(1520, 2500), rng.uniform(600, 712)
        s = .7 + (y - 600) / 140
        for dx in (-3.5, 3.5):
            fillp(ctx, circ(x + dx * s, y, 2.6 * s, 3.6 * s, 10, rot=.2 * np.sign(dx)), alpha(BG["rock_line"], .45))
    # the wallow: a shallow oval hollow with a raised lip and a heap of powder-soft dust
    hx, hy = HOLLOW
    fillp(ctx, circ(hx, hy, 170, 38, 60), mix(BG["ledge"], BG["lip"], .55))
    fillp(ctx, circ(hx + 4, hy + 3, 150, 30, 60), mix(BG["ledge2"], BG["crack"], .35))
    fillp(ctx, circ(hx + 10, hy + 6, 120, 20, 60), mix(BG["ledge2"], BG["crack"], .55))
    ctx.save()
    clip_shape(ctx, circ(hx + 4, hy + 3, 150, 30, 60))
    for k in range(40):
        x, y = rng.uniform(hx - 150, hx + 150), rng.uniform(hy - 26, hy + 30)
        drybrush(ctx, x, y, x + rng.uniform(15, 50), y + rng.normal(0, .8), rng.uniform(1, 2.5), alpha(BG["lip"], .35), rng)
    ctx.restore()
    for k, (x, y, r) in enumerate(((hx - 110, hy - 6, 22), (hx - 80, hy - 14, 26), (hx - 52, hy - 8, 18), (hx + 96, hy - 10, 20),
                                    (hx + 122, hy - 2, 15))):
        fillp(ctx, circ(x, y, r * 1.3, r * .55, 24), hexc("#f2dcb2"))
        fillp(ctx, circ(x + r * .3, y + r * .25, r * .9, r * .3, 20), hexc("#e2c38f"))
    # his scratching post: a gnarled dead branch stuck upright, worn smooth on one side
    post = [(2028, 640), (2022, 560), (2030, 500), (2016, 452), (2008, 420)]
    branch = [(2024, 520), (2048, 488), (2066, 470), (2072, 452)]
    for pts, w0, w1 in ((post, 20, 9), (branch, 9, 4)):
        P = crs(pts, False, 10)
        R = D.ribbon(P, w0, w1)
        fillp(ctx, R, hexc("#a88a70"))
        ink(ctx, R, 1.6, alpha(hexc("#5e4636"), .9), closed=True)
        tline(ctx, list(np.array(pts) + [3, 0]), 1.6, col=alpha(hexc("#7c604c"), .8), t0=.2, t1=.2)
    fillp(ctx, circ(2034, 642, 26, 6, 20), alpha(BG["crack"], .5))
    # a couple of tufts on his side
    for tx, ty, s in ((2380, 664, 1.0), (1590, 690, .9)):
        for k in range(11):
            a = -math.pi / 2 + (k - 5) * .17 + rng.normal(0, .05)
            L = s * (24 + 14 * math.cos((k - 5) * .3)) * rng.uniform(.8, 1.1)
            tline(ctx, [(tx + (k - 5) * 1.6, ty), (tx + (k - 5) * 1.6 + L * .55 * math.cos(a), ty + L * .55 * math.sin(a)),
                        (tx + (k - 5) * 1.6 + L * math.cos(a) * 1.05, ty + L * math.sin(a))], 2.0 * s,
                  col=mix(BG["cactus_sh"], BG["crack"], .5), t0=.9, t1=.1)


# ------------------------------------------------------------------------------------------------ variants
def storm_wall(ctx, level, seed=71):
    """A haboob: a towering wall of billowing dust on the horizon (screen space, behind the buttes).  level 0..1 = height.
    Three layers of cauliflower billows, darker and taller at the back, each shaded darker toward its base, with streaks."""
    if level <= 0:
        return
    rng = np.random.default_rng(seed)
    base = D.HORIZON + 8
    layers = ((hexc("#a87a58"), hexc("#8a5f45"), 0.0, 1.0), (hexc("#c4936a"), hexc("#a27352"), .25, .82), (hexc("#dcae80"), hexc("#bb8a60"), .5, .6))
    for li, (top_c, low_c, off, hk) in enumerate(layers):
        top = base - (40 + 330 * level * hk)
        xs = np.linspace(-60 + 40 * li, W + 60, 9 + 2 * li)
        circles = []
        for j, x in enumerate(xs):
            r = rng.uniform(60, 95) * (.5 + .5 * level) * (1 - .15 * li)
            y = top + r * .8 + rng.uniform(-25, 25) * level
            circles.append((x + rng.uniform(-20, 20), y, r))
        ctx.save()
        ctx.new_path()
        for x, y, r in circles:
            ctx.new_sub_path()
            ctx.arc(x, y, r, 0, 2 * math.pi)
        ctx.rectangle(-100, min(c[1] for c in circles), W + 200, base - min(c[1] for c in circles) + 10)
        g = cairo.LinearGradient(0, top, 0, base)
        g.add_color_stop_rgba(0, *top_c)
        g.add_color_stop_rgba(1, *low_c)
        ctx.set_source(g)
        ctx.fill_preserve()
        ctx.clip()
        for x, y, r in circles:   # a soft darker crescent under each billow, so the domes read as rounded
            fillp(ctx, circ(x + r * .15, y + r * .75, r * 1.05, r * .4, 30), alpha(low_c, .55))
        for j in range(int(40 * level) + 4):
            y = rng.uniform(top, base)
            x = rng.uniform(-60, W)
            drybrush(ctx, x, y, x + rng.uniform(40, 180), y + rng.normal(0, 2), rng.uniform(1.5, 4), alpha(low_c, .35), rng, broken=.5)
        ctx.restore()
        for x, y, r in circles[1:-1]:   # Noble-style off-register contour on the top billows
            a0 = np.linspace(math.pi * 1.1, math.pi * 1.9, 14)
            P = np.stack([x + r * np.cos(a0) + 3, y + r * np.sin(a0) - 2], 1)
            ink(ctx, P, 1.3, alpha(low_c, .8))


def drifts(ctx, cam, wedge):
    """Aftermath: an even, pale blanket of dust over the ledge (lumpy top edge, ripples) except inside `wedge` (screen polygon)."""
    rng = np.random.default_rng(81)
    ctx.save()
    if wedge is not None:  # clip to everything outside the wedge (even-odd with a big rectangle)
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        ctx.rectangle(-100, -100, W + 200, H + 200)
        path(ctx, np.asarray(wedge, float))
        ctx.clip()
        ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
    top = [(-20, 584)] + [(x, 580 + rng.uniform(-6, 6)) for x in np.linspace(40, W - 40, 18)] + [(W + 20, 584)]
    blanket = np.vstack([crs(top, False, 8), [(W + 20, 730), (-20, 730)]])
    fillp(ctx, blanket, hexc("#f1dfbb"))
    for k in range(26):  # ripples
        x, y = rng.uniform(-40, W), rng.uniform(600, 715)
        L = rng.uniform(60, 160)
        a = np.linspace(0, 1, 12)
        P = np.stack([x + L * a, y - 5 * np.sin(np.pi * a)], 1)
        ink(ctx, P, np.linspace(.4, 1.6, 12), alpha(hexc("#cdb084"), .8))
    for k in range(30):
        x, y = rng.uniform(0, W), rng.uniform(590, 715)
        drybrush(ctx, x, y, x + rng.uniform(30, 90), y + rng.normal(0, 1), rng.uniform(1, 3), alpha(hexc("#fff6e0"), .7), rng)
    ctx.restore()
    if wedge is not None:   # a crisp drift edge along the wedge, so the clean patch reads at a glance
        Wp = np.asarray(wedge, float)
        ink(ctx, Wp, 2.2, alpha(hexc("#b8946a"), .9), closed=True)
        ink(ctx, Wp + [2, -2], 1.4, alpha(hexc("#fff6e0"), .9), closed=True)


def caps(ctx):
    """Dust heaped on the tops of buttes and ledges (drawn per world layer so it follows them)."""
    for x0, x1, y in ((958, 1046, 92), (1146, 1300, 236), (1500, 1548, 148), (1594, 1630, 200), (1890, 2240, 270), (314, 408, 318)):
        n = max(3, int((x1 - x0) / 24))
        for k in range(n):
            x = x0 + (x1 - x0) * (k + .5) / n
            fillp(ctx, circ(x, y - 2, (x1 - x0) / n * .8, 7, 16), hexc("#f0dcb4"))
        tline(ctx, [(x0, y - 4), ((x0 + x1) / 2, y - 9), (x1, y - 4)], 1.2, col=alpha(hexc("#c9a676"), .8))


# ------------------------------------------------------------------------------------------------ assemble a background
_CACHE = {}


def background(cam="A", time="day", storm=0.0, after=None, engulf=0.0):
    """Painted background for a camera, float32 RGB (H, W).  after: None or a screen polygon of the clean wedge."""
    key = (cam, time, round(storm, 3), None if after is None else tuple(map(tuple, np.round(after))), round(engulf, 3))
    if key in _CACHE:
        return _CACHE[key]
    pal = dict(BG)
    if time == "sunset":
        pal.update(SUNSET)
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * SS, H * SS)
    ctx = cairo.Context(surf)
    ctx.scale(SS, SS)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)

    def layer(par, *fns):
        ctx.save()
        apply_cam(ctx, cam, par)
        for f in fns:
            f()
        ctx.restore()

    saved = dict(BG)
    BG.update(pal)  # the proto sky reads the palette from BG
    try:
        layer(PAR["sky"], lambda: sky_fill(ctx, pal), lambda: sky(ctx, pal))
        if storm > 0:
            ctx.save()
            storm_wall(ctx, storm)
            ctx.restore()
        layer(PAR["far"], lambda: D.bg_far(ctx), lambda: far_ext(ctx))
        layer(PAR["floor"], lambda: floor_fill(ctx), lambda: D.bg_floor(ctx), lambda: floor_ext(ctx))
        layer(PAR["butte"], lambda: D.bg_buttes(ctx), lambda: buttes_ext(ctx), *( [lambda: caps(ctx)] if after is not None else []))
        layer(PAR["ledge"], lambda: ledge_ext(ctx), lambda: D.bg_ledge(ctx), lambda: D.bg_rock(ctx), lambda: D.bg_props(ctx), lambda: props_ext(ctx))
        if after is not None:
            drifts(ctx, cam, after)
    finally:
        BG.clear()
        BG.update(saved)
    a = np.ndarray((H * SS, W * SS, 4), np.uint8, surf.get_data())[:, :, :3][:, :, ::-1].astype(np.float32) / 255
    img = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
    if time == "sunset":   # warm the whole card, push shadows toward violet
        lum = img.mean(2, keepdims=True)
        img = img * np.array([1.04, .9, .86], np.float32) * (.92 + .08 * lum) + (1 - lum) * np.array([.02, 0, .05], np.float32)
    if storm > 0:          # the light goes amber as the wall rises
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        img = img * (1 - .3 * storm) + STORMC * .3 * storm * (1 - .3 * yy)
    if after is not None:  # everything a little dustier
        img = img * .78 + DUSTC * .22
    rng = np.random.default_rng(3)
    low = cv2.resize(rng.normal(0, 1, (H // 24, W // 24)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    mid = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 2.2)
    fib = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigmaX=5, sigmaY=.7)
    img = img * (1 + .018 * low + .05 * mid + .03 * fib)[..., None]
    if engulf > 0:
        img = engulfed(img, engulf)
    out = np.clip(img, 0, 1).astype(np.float32)
    _CACHE[key] = out
    return out


def engulfed(img, k):
    """Inside the dust storm: everything sinks into a swirling amber-brown haze with horizontal wind streaks."""
    rng = np.random.default_rng(91)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    swirl = .5 + .5 * np.sin(xx / 90 + 2.5 * np.sin(yy / 70)) * np.cos(yy / 55)
    haze = STORMC * (.85 + .25 * swirl[..., None])
    out = img * (1 - .72 * k) + haze * .72 * k
    streak = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigmaX=40, sigmaY=1.5)
    return out * (1 + .25 * k * streak[..., None])
