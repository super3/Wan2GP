"""desert_chase: an original 1950s-theatrical-style desert gag, drawn entirely in code (pycairo vectors + numpy paint finish).

Pip, a fussy little armadillo with a slate-blue banded shell, a red neckerchief and a straw broom, keeps his patch of a high
desert ledge spotless.  Background: a Noble-style painted desert (flat graphic buttes, violet shadow planes, dry-brush strata,
off-register line work, huge pale sky, a long road winding along the canyon floor far below).  Character: a flat cel with clean
near-black ink of varied weight, composited over the painted card.

Design space is 1280 x 720 (y down).  frame(i) -> uint8 RGB 1280x720.  key() -> uint8 RGB 1280x704 (frame 0, 8 px trimmed top/bottom).
"""
import math

import cairo
import cv2
import numpy as np

W, H, FPS = 1280, 720, 24
SS = 3  # supersampling


def hexc(h, a=1.0):
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, a)


def mix(c1, c2, t):
    return tuple(a + (b - a) * t for a, b in zip(c1, c2))


def alpha(c, a):
    return (c[0], c[1], c[2], a)


# ------------------------------------------------------------------------------------------------ palette
INK = hexc("#1d1418")
BG = dict(
    sky_top=hexc("#7cc3c6"), sky_mid=hexc("#c3e2c6"), sky_low=hexc("#f7e4b2"),
    cloud=hexc("#fff5de"), cloud_under=hexc("#f4cdb6"),
    far=hexc("#c9a2bf"), far_sh=hexc("#ab88ad"), far2=hexc("#d8b3bf"),
    floor_far=hexc("#f0d2bb"), floor_near=hexc("#e5ae8e"), floor_dot=hexc("#b88070"),
    road=hexc("#f8e3b7"), road_edge=hexc("#c78a63"),
    hoodoo=hexc("#dc9c84"), hoodoo_sh=hexc("#a8728e"),
    spire=hexc("#ec926a"), spire_sh=hexc("#9b5379"), spire_cap=hexc("#c86a50"), spire_line=hexc("#b4584f"),
    mesa=hexc("#e8a37d"), mesa_sh=hexc("#a66a88"), mesa_line=hexc("#b96a62"),
    rock=hexc("#eea657"), rock2=hexc("#df8c48"), rock_sh=hexc("#a64f4c"), rock_deep=hexc("#71344a"), rock_line=hexc("#8c3b3e"),
    hole=hexc("#3c1d31"), hole_in=hexc("#26101f"),
    ledge=hexc("#ecb86c"), ledge2=hexc("#e2a25a"), lip=hexc("#f7d495"), crack=hexc("#b8743f"),
    peb=hexc("#c98547"), peb_lt=hexc("#f4cb8b"),
    cactus=hexc("#86a35c"), cactus_sh=hexc("#5d7a46"), cactus_line=hexc("#3f5a34"), bloom=hexc("#ec5d78"),
    mat=hexc("#c7503c"), mat2=hexc("#efc06a"),
)
PC = dict(
    shell=hexc("#5f7ea8"), shell_dk=hexc("#45618c"), shell_lt=hexc("#7f9bc0"),
    skin=hexc("#f1c4a2"), skin_dk=hexc("#d99a86"), belly=hexc("#fbe7cb"),
    nose=hexc("#3a2230"), white=hexc("#fffdf6"), mouth=hexc("#5a1e2c"), tongue=hexc("#e2747a"),
    scarf=hexc("#d7403a"), scarf_dk=hexc("#a92d2e"), dot=hexc("#ffe9d0"),
    handle=hexc("#9b6434"), straw=hexc("#ecc85c"), straw_dk=hexc("#bf9330"), band=hexc("#c8423a"),
    shadow=hexc("#8a4a2c", .32),
    dust=hexc("#f4dfb4"), dust_sh=hexc("#e2c08a"), dust_line=hexc("#b58757"),
    weed=hexc("#c7a066"), weed_dk=hexc("#7a5532"),
)


# ------------------------------------------------------------------------------------------------ geometry
def crs(pts, closed=False, n=12, tension=1.0):
    """Dense Catmull-Rom samples through pts (numpy Nx2)."""
    p = np.asarray(pts, float)
    N = len(p)
    if N < 2:
        return p
    out = []
    segs = N if closed else N - 1
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    for i in range(segs):
        if closed:
            p0, p1, p2, p3 = p[(i - 1) % N], p[i % N], p[(i + 1) % N], p[(i + 2) % N]
        else:
            p0, p1, p2, p3 = p[max(i - 1, 0)], p[i], p[min(i + 1, N - 1)], p[min(i + 2, N - 1)]
        c1, c2 = p1 + (p2 - p0) * tension / 6, p2 - (p3 - p1) * tension / 6
        out.append((1 - t) ** 3 * p1 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * p2)
    if not closed:
        out.append(p[-1:])
    return np.concatenate(out)


def Tm(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], float)


def Rm(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def Sm(sx, sy=None):
    return np.array([[sx, 0, 0], [0, sy if sy is not None else sx, 0], [0, 0, 1]], float)


I3 = np.eye(3)


def ap(M, pts):
    p = np.asarray(pts, float).reshape(-1, 2)
    return p @ M[:2, :2].T + M[:2, 2]


def ap1(M, x, y):
    return ap(M, [(x, y)])[0]


def path(ctx, P, closed=True):
    ctx.move_to(*P[0])
    for q in P[1:]:
        ctx.line_to(*q)
    if closed:
        ctx.close_path()


def fillp(ctx, P, col):
    path(ctx, P)
    ctx.set_source_rgba(*col)
    ctx.fill()


SHADOW_DIR = np.array([.55, .83])


def closed_widths(P, wb, k=.6):
    q = np.vstack([P, P[:1]])
    area = .5 * np.sum(q[:-1, 0] * q[1:, 1] - q[1:, 0] * q[:-1, 1])
    t = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([t[:, 1], -t[:, 0]], 1) * (1 if area > 0 else -1)
    d = np.clip(nrm @ SHADOW_DIR, 0, 1)
    return wb * (1 - k / 2 + k * d)


def ink(ctx, P, w, col=INK, closed=False):
    """Variable-width ink line: consecutive segments of equal (quantised) width are stroked together with round caps."""
    P = np.asarray(P, float)
    w = np.broadcast_to(np.asarray(w, float), (len(P),)).copy()
    if closed:
        P = np.vstack([P, P[:1]])
        w = np.append(w, w[0])
    q = np.round(w * 6) / 6
    ctx.set_source_rgba(*col)
    s = 0
    n = len(P)
    for i in range(1, n):
        if i == n - 1 or q[i] != q[s]:
            if q[s] > .05:
                ctx.move_to(*P[s])
                for j in range(s + 1, i + 1):
                    ctx.line_to(*P[j])
                ctx.set_line_width(q[s])
                ctx.stroke()
            s = i


def shape(ctx, pts, fc, wb=3.2, M=I3, n=12, tension=1.0, k=.6, line=INK, dense=None):
    P = ap(M, crs(pts, True, n, tension)) if dense is None else dense
    if fc is not None:
        fillp(ctx, P, fc)
    if line is not None and wb > 0:
        ink(ctx, P, closed_widths(P, wb, k), line, closed=True)
    return P


def tline(ctx, pts, w, M=I3, n=12, tension=1.0, col=INK, t0=.25, t1=.25, dense=None):
    """Open line, tapered at both ends (t0/t1: fraction of width left at the start/end)."""
    P = ap(M, crs(pts, False, n, tension)) if dense is None else dense
    s = np.linspace(0, 1, len(P))
    prof = np.where(s < .5, t0 + (1 - t0) * np.sin(np.pi * s) ** .6, t1 + (1 - t1) * np.sin(np.pi * s) ** .6)
    ink(ctx, P, w * prof, col)
    return P


def tube(ctx, pts, width, fc, M=I3, lw=2.6, n=10):
    """A noodle limb: an ink tube with a colour tube inside (round ends)."""
    P = ap(M, crs(pts, False, n))
    path(ctx, P, closed=False)
    ctx.set_source_rgba(*INK)
    ctx.set_line_width(width + 2 * lw)
    ctx.stroke()
    path(ctx, P, closed=False)
    ctx.set_source_rgba(*fc)
    ctx.set_line_width(width)
    ctx.stroke()
    return P


def ribbon(P, w0, w1):
    P = np.asarray(P, float)
    d = np.gradient(P, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    w = np.linspace(w0, w1, len(P))[:, None] / 2
    return np.vstack([P + nrm * w, (P - nrm * w)[::-1]])


def circ(cx, cy, rx, ry=None, n=40, rot=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ry = rx if ry is None else ry
    x, y = rx * np.cos(a), ry * np.sin(a)
    c, s = math.cos(rot), math.sin(rot)
    return np.stack([cx + c * x - s * y, cy + s * x + c * y], 1)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------------------------------------ painted background
def drybrush(ctx, x0, y0, x1, y1, w, col, rng, broken=.35, wob=1.2):
    """A dry-brush stroke: a line broken into dabs of slightly varying width with gaps (paint skipping on rough paper)."""
    L = math.hypot(x1 - x0, y1 - y0)
    n = max(2, int(L / 7))
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    ys = ys + np.cumsum(rng.normal(0, wob * .3, n))
    ctx.set_source_rgba(*col)
    on = True
    for i in range(n - 1):
        if rng.uniform() < broken * .5:
            on = not on
        if on:
            ctx.set_line_width(max(.4, w * rng.uniform(.55, 1.1)))
            ctx.move_to(xs[i], ys[i])
            ctx.line_to(xs[i + 1], ys[i + 1])
            ctx.stroke()


def offreg(ctx, pts, col, w=1.6, dx=3.0, dy=-2.0, rng=None, skip=.15, n=8, closed=False):
    """Noble-style off-register line work: a thin coloured contour that sits a few px off the colour shape, with breaks."""
    P = crs(pts, closed, n) + np.array([dx, dy])
    if rng is None:
        rng = np.random.default_rng(1)
    ctx.set_source_rgba(*col)
    ctx.set_line_width(w)
    seg = 6
    for s in range(0, len(P) - 1, seg):
        if rng.uniform() < skip:
            continue
        q = P[s:s + seg + 1]
        ctx.move_to(*q[0])
        for p in q[1:]:
            ctx.line_to(*p)
        ctx.stroke()


def clip_shape(ctx, P):
    path(ctx, P)
    ctx.clip()


def bg_sky(ctx):
    g = cairo.LinearGradient(0, 0, 0, 400)
    g.add_color_stop_rgba(0, *BG["sky_top"])
    g.add_color_stop_rgba(.55, *BG["sky_mid"])
    g.add_color_stop_rgba(1, *BG["sky_low"])
    ctx.rectangle(0, 0, W, 420)
    ctx.set_source(g)
    ctx.fill()
    # stylised streak clouds: long thin spindles with pointed ends, loosely grouped (painted with one stroke each)
    rng = np.random.default_rng(11)
    groups = ((250, 120, 380, 10), (690, 196, 220, 7), (1120, 66, 420, 11), (1210, 170, 200, 7))
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
            fillp(ctx, np.vstack([top, bot]) + [3, 3], alpha(BG["cloud_under"], .9))
            fillp(ctx, np.vstack([top, bot]), BG["cloud"])
            drybrush(ctx, xs[4], oy - hh * .3, xs[-5], oy - hh * .25, hh * .5, alpha(hexc("#ffffff"), .5), rng, broken=.6, wob=.5)


HORIZON = 394


def bg_far(ctx):
    rng = np.random.default_rng(5)
    # far mesa band: flat tops, sheer sides
    prof = [(-10, HORIZON), (-10, 368), (60, 366), (72, 352), (180, 350), (196, 372), (300, 374), (318, 360), (370, 358), (384, 384),
            (590, 386), (606, 352), (626, 340), (720, 338), (740, 360), (790, 366), (840, 380), (1060, 382), (1090, 360), (1180, 356),
            (1200, 372), (1290, 374), (1290, HORIZON)]
    fillp(ctx, np.array(prof, float), BG["far"])
    for a, b in (((72, 352), (100, 372)), ((626, 340), (660, 378)), ((1090, 360), (1120, 382))):  # small shadow wedges
        fillp(ctx, np.array([a, (a[0] + 26, a[1]), (b[0] + 20, HORIZON), (b[0] - 8, HORIZON)], float), BG["far_sh"])
    ctx.rectangle(-10, HORIZON - 2, W + 20, 6)
    ctx.set_source_rgba(*BG["far2"])
    ctx.fill()


LEDGE_EDGE = [(-10, 566), (120, 562), (260, 570), (420, 566), (520, 574), (640, 568), (780, 576), (900, 570), (1010, 578), (1120, 571),
              (1200, 576), (1290, 572)]

ROAD = [(1060, 600), (980, 566), (860, 546), (760, 532), (712, 516), (760, 498), (880, 484), (930, 470), (900, 456), (812, 444), (770, 430),
        (790, 418), (845, 408), (870, 400), (858, 396)]
ROAD_W = [30, 26, 21, 17, 14, 11.5, 9.5, 8, 6.5, 5.2, 4.2, 3.4, 2.6, 2.0, 1.4]


def road_dense():
    P = crs(ROAD, False, 16, 1.0)
    w = np.interp(np.linspace(0, len(ROAD) - 1, len(P)), np.arange(len(ROAD)), ROAD_W)
    return P, w


def bg_floor(ctx):
    g = cairo.LinearGradient(0, HORIZON, 0, 600)
    g.add_color_stop_rgba(0, *BG["floor_far"])
    g.add_color_stop_rgba(1, *BG["floor_near"])
    ctx.rectangle(-10, HORIZON + 1, W + 20, 230)
    ctx.set_source(g)
    ctx.fill()
    rng = np.random.default_rng(9)
    # canyon floor: horizontal dry-brush bands that get finer with distance
    for k in range(70):
        y = HORIZON + 4 + (rng.uniform() ** 1.6) * 190
        depth = (y - HORIZON) / 190
        x = rng.uniform(-40, W)
        L = rng.uniform(60, 260) * (.4 + depth)
        drybrush(ctx, x, y, x + L, y + rng.normal(0, 1), .8 + 2.2 * depth, alpha(mix(BG["floor_near"], BG["floor_dot"], .35), .35), rng)
    # low, pale floor mesas at mid distance (aerial perspective: paler and cooler than anything on the ledge)
    for x0, x1, top, base in ((540, 690, 432, 452), (1072, 1140, 438, 456), (150, 300, 420, 444), (700, 760, 410, 424)):
        body = np.array([(x0, base), (x0 + 6, top + 4), (x0 + 14, top), (x1 - 12, top), (x1 - 4, top + 5), (x1, base)], float)
        fillp(ctx, body, mix(BG["far2"], BG["floor_far"], .35))
        shade = np.array([(x1 - (x1 - x0) * .3, top), (x1 - 12, top), (x1 - 4, top + 5), (x1, base), (x1 - (x1 - x0) * .32, base)], float)
        fillp(ctx, shade, mix(BG["far"], BG["floor_far"], .3))
        fillp(ctx, np.array([(x0 - 30, base), (x1 + 40, base), (x1 + 30, base + 3), (x0 - 20, base + 3)], float), alpha(BG["far_sh"], .35))
    # the road, winding back and forth to the horizon
    P, w = road_dense()
    R = []
    d = np.gradient(P, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    L, Rr = P + nrm * w[:, None] / 2, P - nrm * w[:, None] / 2
    fillp(ctx, np.vstack([L, Rr[::-1]]), BG["road"])
    for side in (L, Rr):
        ink(ctx, side, np.clip(w * .07, .5, 2.0), alpha(BG["road_edge"], .9))
    # tiny shrubs (dots with a shadow tick) scattered on the canyon floor, bigger when nearer
    Pr, wr = road_dense()
    for k in range(44):
        y = HORIZON + 8 + rng.uniform() ** 1.2 * 160
        x = rng.uniform(0, W)
        if np.min(np.hypot(Pr[:, 0] - x, (Pr[:, 1] - y) * 2.5)) < 26:
            continue
        depth = (y - HORIZON) / 170
        r = .8 + 2.4 * depth
        for c in range(3):
            fillp(ctx, circ(x + (c - 1) * r * 1.3, y - (c == 1) * r * .5, r * (1.1 if c == 1 else .8), r * .7, 12), alpha(BG["floor_dot"], .75))
        fillp(ctx, circ(x + r, y + r * .7, r * 2.2, r * .35, 12), alpha(BG["floor_dot"], .3))
    g = cairo.LinearGradient(0, 470, 0, 580)
    g.add_color_stop_rgba(0, *alpha(BG["sky_low"], 0))
    g.add_color_stop_rgba(1, *alpha(BG["sky_low"], .55))
    ctx.rectangle(-10, 470, W + 20, 120)
    ctx.set_source(g)
    ctx.fill()
    # two small hoodoos on the canyon floor, left of centre, for scale
    for x0, top, wd in ((392, 470, 30), (448, 492, 20)):
        pts = [(x0 - wd / 2, 560), (x0 - wd / 2 + 3, top + 14), (x0 - wd * .7, top + 4), (x0 - wd * .2, top - 6), (x0 + wd * .6, top + 2),
               (x0 + wd / 2 - 2, top + 14), (x0 + wd / 2, 560)]
        P = np.array(pts, float)
        fillp(ctx, P, BG["hoodoo"])
        fillp(ctx, np.array([(x0 + 2, top + 6), (x0 + wd * .6, top + 2), (x0 + wd / 2 - 2, top + 14), (x0 + wd / 2, 560), (x0 + 4, 560)]),
              BG["hoodoo_sh"])


def angular(pts, rng=None, j=0.0):
    p = np.asarray(pts, float)
    if rng is not None and j > 0:
        p = p + rng.normal(0, j, p.shape)
    return p


def butte(ctx, lit_pts, sh_pts, cap_pts, lit, sh, cap, line, rng, strata=(), vert=False, edge=None):
    """A Noble butte: a flat lit plane, a hard-edged violet shadow plane, a cap rock, strata and dry-brush, off-register line."""
    P = angular(lit_pts)
    fillp(ctx, P, lit)
    ctx.save()
    clip_shape(ctx, P)
    x0, y0 = P.min(0)
    x1, y1 = P.max(0)
    for y in strata:  # long thin strata lines, slightly broken
        drybrush(ctx, x0 - 10, y, x1 + 10, y + rng.normal(0, 1.5), rng.uniform(1.6, 3.2), alpha(mix(lit, line, .6), .6), rng, broken=.4, wob=.6)
    for k in range(int((y1 - y0) / 6)):
        y = rng.uniform(y0, y1)
        x = rng.uniform(x0 - 30, x1)
        drybrush(ctx, x, y, x + rng.uniform(14, 60), y + rng.normal(0, 1.0), rng.uniform(.8, 2.0),
                 alpha(mix(lit, (1, 1, 1, 1), .3) if rng.uniform() < .5 else mix(lit, line, .35), .45), rng)
    if sh_pts is not None:
        S = angular(sh_pts)
        fillp(ctx, S, sh)
        ctx.save()
        clip_shape(ctx, S)
        for y in strata:
            drybrush(ctx, x0 - 10, y + 2, x1 + 10, y + 2 + rng.normal(0, 1.5), rng.uniform(1.4, 2.8), alpha(mix(sh, (0, 0, 0, 1), .22), .6), rng,
                     broken=.4, wob=.6)
        for k in range(int((y1 - y0) / 9)):
            y = rng.uniform(y0, y1)
            x = rng.uniform(x0, x1)
            if vert:
                drybrush(ctx, x, y, x + rng.normal(0, 2), y + rng.uniform(16, 60), rng.uniform(1, 2.2), alpha(mix(sh, (0, 0, 0, 1), .18), .45), rng)
            else:
                drybrush(ctx, x - 20, y, x + rng.uniform(10, 50), y + rng.normal(0, 1), rng.uniform(1, 2.2), alpha(mix(sh, (0, 0, 0, 1), .15), .45), rng)
        ctx.restore()
    if cap_pts is not None:
        fillp(ctx, angular(cap_pts), cap)
    ctx.restore()
    if edge is not None:
        for e in edge:
            offreg(ctx, e, alpha(line, .9), w=1.6, rng=rng, n=2, dx=3.5, dy=-2.5, skip=.12)


def bg_buttes(ctx):
    rng = np.random.default_rng(21)
    # right: a needle spire rising from the canyon floor far above the horizon (vertical drama)
    lit = [(944, 600), (948, 470), (938, 400), (950, 330), (944, 262), (956, 200), (950, 150), (962, 112), (976, 96), (1036, 90), (1048, 104),
           (1046, 150), (1058, 214), (1054, 290), (1066, 360), (1062, 430), (1076, 520), (1082, 600)]
    sh = [(1012, 98), (1036, 90), (1048, 104), (1046, 150), (1058, 214), (1054, 290), (1066, 360), (1062, 430), (1076, 520), (1082, 600),
          (1028, 600), (1034, 520), (1022, 430), (1030, 350), (1018, 262), (1026, 182), (1012, 130)]
    cap = [(962, 112), (976, 96), (1036, 90), (1048, 104), (1046, 120), (1000, 116), (958, 126)]
    butte(ctx, lit, sh, cap, BG["spire"], BG["spire_sh"], BG["spire_cap"], BG["spire_line"], rng,
          strata=(140, 176, 238, 298, 352, 418, 484), vert=True,
          edge=([(976, 96), (1036, 90), (1048, 104), (1046, 150), (1058, 214), (1054, 290), (1066, 360), (1062, 430), (1076, 520), (1082, 580)],
                [(1012, 130), (1026, 182), (1018, 262), (1030, 350), (1022, 430)],
                [(962, 112), (950, 150), (956, 200), (944, 262)]))
    # far right: a broad flat-topped mesa, a little paler (further away), cut by the frame edge
    lit = [(1108, 600), (1114, 420), (1110, 330), (1124, 272), (1146, 240), (1300, 236), (1300, 600)]
    sh = [(1196, 244), (1300, 240), (1300, 600), (1204, 600), (1214, 470), (1200, 360), (1208, 290)]
    cap = [(1138, 254), (1146, 238), (1300, 234), (1300, 256), (1150, 260)]
    butte(ctx, lit, sh, cap, BG["mesa"], BG["mesa_sh"], mix(BG["mesa"], BG["spire_cap"], .5), BG["mesa_line"], rng,
          strata=(282, 330, 388, 452, 520), vert=False,
          edge=([(1146, 240), (1300, 236)], [(1208, 290), (1200, 360), (1214, 470), (1204, 580)], [(1124, 272), (1110, 330), (1114, 420)]))
    # left of centre: a low, pale mid-distance mesa partly behind the burrow rock
    lit = [(298, 560), (302, 350), (314, 322), (408, 316), (420, 334), (424, 560)]
    sh = [(384, 318), (408, 316), (420, 334), (424, 560), (390, 560), (396, 420)]
    butte(ctx, lit, sh, None, mix(BG["mesa"], BG["far2"], .5), mix(BG["mesa_sh"], BG["far"], .45), None, mix(BG["mesa_line"], BG["far"], .3),
          rng, strata=(340, 380, 430, 490), vert=False, edge=([(314, 322), (408, 316), (420, 334), (424, 540)],))


def bg_ledge(ctx):
    edge = crs(LEDGE_EDGE, False, 10, .8)
    P = np.vstack([edge, [(1290, 730), (-10, 730)]])
    g = cairo.LinearGradient(0, 560, 0, 720)
    g.add_color_stop_rgba(0, *BG["ledge"])
    g.add_color_stop_rgba(1, *BG["ledge2"])
    path(ctx, P)
    ctx.set_source(g)
    ctx.fill()
    rng = np.random.default_rng(33)
    ctx.save()
    clip_shape(ctx, P)
    for k in range(120):  # sandy dry-brush texture, horizontal, sparser near the edge
        y = 572 + rng.uniform() ** .8 * 150
        x = rng.uniform(-40, W)
        L = rng.uniform(30, 160)
        light = rng.uniform() < .4
        drybrush(ctx, x, y, x + L, y + rng.normal(0, 1), rng.uniform(1, 3.5),
                 alpha(BG["lip"] if light else mix(BG["ledge2"], BG["crack"], .5), .32 if light else .3), rng)
    for k in range(260):  # stipple
        x, y = rng.uniform(0, W), rng.uniform(572, 720)
        ctx.new_sub_path()
        ctx.arc(x, y, rng.uniform(.6, 1.8), 0, 2 * math.pi)
        ctx.set_source_rgba(*alpha(BG["crack"], rng.uniform(.15, .4)))
        ctx.fill()
    ctx.restore()
    # the lit lip of the ledge edge and a thin darker drop line just below it
    ink(ctx, edge + [0, 2.5], 4.0, alpha(BG["lip"], .95))
    ink(ctx, edge + [2, -1], 1.6, alpha(BG["crack"], .9))
    # cracks
    for pts in (((860, 640), (900, 652), (934, 648), (968, 668)), ((1090, 600), (1120, 612), (1150, 608)),
                ((300, 690), (340, 682), (372, 692), (420, 688)), ((1180, 680), (1220, 668), (1262, 676))):
        tline(ctx, pts, 2.0, col=alpha(BG["crack"], .85), t0=.1, t1=.1)
    # pebbles with flat shadows
    for x, y, r in ((420, 642, 7), (436, 650, 4), (900, 612, 6), (1000, 690, 9), (1018, 698, 5), (760, 700, 6), (1210, 630, 7),
                    (1236, 640, 4), (40, 700, 8), (330, 615, 5)):
        fillp(ctx, circ(x + r * .5, y + r * .45, r * 1.25, r * .45), alpha(BG["crack"], .45))
        fillp(ctx, circ(x, y, r, r * .72), BG["peb"])
        fillp(ctx, circ(x - r * .25, y - r * .25, r * .55, r * .35), BG["peb_lt"])


def bg_rock(ctx):
    """The tall foreground rock at left with Pip's burrow at its foot."""
    rng = np.random.default_rng(44)
    out = [(-20, 700), (-20, 140), (20, 110), (70, 96), (118, 112), (150, 104), (196, 128), (210, 180), (236, 214), (240, 300), (266, 330),
           (276, 420), (302, 460), (318, 540), (330, 610), (346, 664), (330, 690)]
    P = crs(out, True, 8, .3)
    fillp(ctx, P, BG["rock"])
    ctx.save()
    clip_shape(ctx, P)
    # facets: a lit top plane, a mid plane, a violet shadow plane on the right, a deep crevice
    fillp(ctx, crs([(-20, 140), (20, 110), (70, 96), (118, 112), (150, 104), (196, 128), (210, 180), (120, 168), (40, 176), (-20, 190)], True,
                   8, .2), mix(BG["rock"], BG["lip"], .35))
    fillp(ctx, crs([(140, 240), (236, 214), (240, 300), (266, 330), (276, 420), (302, 460), (318, 540), (330, 610), (346, 664), (330, 700),
                    (230, 700), (250, 560), (226, 430), (196, 330)], True, 8, .2), BG["rock_sh"])
    fillp(ctx, crs([(-20, 300), (60, 292), (130, 310), (150, 330), (60, 336), (-20, 344)], True, 8, .2), BG["rock2"])
    fillp(ctx, crs([(-20, 450), (40, 438), (120, 446), (200, 470), (90, 478), (-20, 488)], True, 8, .2), BG["rock2"])
    for y in (190, 236, 288, 352, 404, 470, 520):  # strata
        drybrush(ctx, -30, y, 360, y + rng.normal(0, 4), rng.uniform(2, 4.5), alpha(BG["rock_line"], .45), rng, broken=.55)
    for k in range(110):
        y = rng.uniform(100, 700)
        x = rng.uniform(-40, 320)
        lightc = rng.uniform() < .5
        drybrush(ctx, x, y, x + rng.uniform(16, 70), y + rng.normal(0, 1.2), rng.uniform(.8, 2.4),
                 alpha(mix(BG["rock"], (1, 1, 1, 1), .35) if lightc else BG["rock_line"], .4 if lightc else .3), rng)
    fillp(ctx, crs([(196, 330), (214, 400), (204, 470), (222, 540), (212, 600), (200, 540), (186, 470), (190, 400)], True, 8, .4), BG["rock_deep"])
    for x, y, r in ((52, 228, 9), (78, 240, 5), (128, 392, 7), (40, 520, 10), (64, 534, 5), (150, 268, 4), (104, 600, 6)):  # pockmarks
        fillp(ctx, circ(x, y, r * 1.3, r * .8, 20), BG["rock_sh"])
        fillp(ctx, circ(x - r * .25, y - r * .2, r * .9, r * .5, 20), BG["rock_deep"])
        ink(ctx, circ(x + 2, y + 1.5, r * 1.3, r * .8, 20)[8:18], 1.2, alpha(mix(BG["rock"], BG["lip"], .6), .9))
    for pts in (((96, 410), (92, 450), (100, 486), (94, 520)), ((26, 250), (30, 290), (22, 330))):  # crevices
        tline(ctx, pts, 3.0, col=alpha(BG["rock_deep"], .85), t0=.1, t1=.1)
    ctx.restore()
    offreg(ctx, [(20, 110), (70, 96), (118, 112), (150, 104), (196, 128), (210, 180), (236, 214), (240, 300), (266, 330), (276, 420),
                 (302, 460), (318, 540), (330, 610), (346, 664)], alpha(BG["rock_line"], .9), w=1.8, rng=rng, n=6, dx=3.5, dy=-2.5, skip=.2)
    offreg(ctx, [(20, 180), (120, 168), (210, 182)], alpha(BG["rock_line"], .8), w=1.4, rng=rng, n=8, dx=2.5, dy=-2)
    # burrow: a dark arch with a deeper inside, a lip of light stones, and a little round welcome mat (no lettering)
    arch = [(150, 668), (152, 610), (176, 578), (214, 572), (246, 590), (258, 630), (260, 668)]
    A = crs(arch, False, 12, 1.0)
    fillp(ctx, A, BG["hole"])
    fillp(ctx, crs([(168, 668), (172, 620), (196, 596), (226, 598), (242, 630), (244, 668)], False, 12, 1.0), BG["hole_in"])
    ink(ctx, A, closed_widths(A, 3.0), BG["rock_deep"], closed=False)
    for k, (x, y) in enumerate(((150, 664), (148, 640), (154, 614), (168, 592), (190, 578), (216, 574), (238, 582), (254, 602), (260, 630),
                                (262, 656))):
        fillp(ctx, circ(x, y, 9 - (k % 3), 6 + (k % 2), 16, rot=.4 * k), mix(BG["rock"], BG["lip"], .45))
        ink(ctx, circ(x + 1.5, y - 1, 9 - (k % 3), 6 + (k % 2), 16, rot=.4 * k), 1.1, alpha(BG["rock_line"], .7), closed=True)


def bg_props(ctx):
    rng = np.random.default_rng(55)
    # a little striped rug in front of the burrow (no lettering)
    rug = np.array([(156, 674), (258, 674), (272, 698), (142, 698)], float)
    fillp(ctx, rug + [4, 4], alpha(BG["crack"], .45))
    fillp(ctx, rug, BG["mat"])
    for k, f in enumerate((.22, .5, .78)):
        y0, y1 = 674 + 24 * (f - .07), 674 + 24 * (f + .07)
        xl0, xl1 = 156 - 14 * (f - .07), 156 - 14 * (f + .07)
        xr0, xr1 = 258 + 14 * (f - .07), 258 + 14 * (f + .07)
        fillp(ctx, np.array([(xl0, y0), (xr0, y0), (xr1, y1), (xl1, y1)]), BG["mat2"])
    ink(ctx, rug, 1.3, alpha(BG["rock_deep"], .8), closed=True)
    for x in np.linspace(146, 268, 14):  # fringe on the near edge
        ink(ctx, np.array([(x, 699), (x - 1, 705)]), 1.1, alpha(BG["rock_deep"], .7))
    # a small barrel cactus with one pink bloom, right of the burrow
    cx, cy = 372, 646
    fillp(ctx, circ(cx + 14, cy + 4, 30, 7), alpha(BG["crack"], .45))
    body = [(cx - 24, cy), (cx - 26, cy - 30), (cx - 16, cy - 50), (cx, cy - 56), (cx + 16, cy - 50), (cx + 26, cy - 30), (cx + 24, cy)]
    P = crs(body, True, 10, 1.0)
    fillp(ctx, P, BG["cactus"])
    ctx.save()
    clip_shape(ctx, P)
    fillp(ctx, crs([(cx + 4, cy - 56), (cx + 16, cy - 50), (cx + 26, cy - 30), (cx + 24, cy), (cx + 8, cy)], True, 8, .6), BG["cactus_sh"])
    ctx.restore()
    for dx in (-14, 0, 14):
        tline(ctx, [(cx + dx * .9, cy - 52 + abs(dx) * .3), (cx + dx * 1.15, cy - 28), (cx + dx * .95, cy - 2)], 1.4, col=BG["cactus_line"])
    ink(ctx, P, 1.6, BG["cactus_line"], closed=True)
    for a in np.linspace(-2.6, -.5, 7):  # spines
        x, y = cx + 25 * math.cos(a), cy - 28 + 28 * math.sin(a)
        ink(ctx, np.array([(x, y), (x + 5 * math.cos(a), y + 5 * math.sin(a))]), .9, BG["cactus_line"])
    fl = circ(cx + 2, cy - 58, 7, 5, 12)
    fillp(ctx, fl, BG["bloom"])
    ink(ctx, fl, 1.0, alpha(hexc("#8d2a44"), .9), closed=True)
    # a spiky dry tuft on the right of the ledge and one by the rock
    for tx, ty, s in ((1150, 668, 1.0), (60, 640, .8)):
        for k in range(11):
            a = -math.pi / 2 + (k - 5) * .17 + rng.normal(0, .05)
            L = s * (24 + 14 * math.cos((k - 5) * .3)) * rng.uniform(.8, 1.1)
            tline(ctx, [(tx + (k - 5) * 1.6, ty), (tx + (k - 5) * 1.6 + L * .55 * math.cos(a), ty + L * .55 * math.sin(a)),
                        (tx + (k - 5) * 1.6 + L * math.cos(a) * 1.05, ty + L * math.sin(a))], 2.0 * s,
                  col=mix(BG["cactus_sh"], BG["crack"], .5), t0=.9, t1=.1)


_BG = None


def background():
    """The static painted card (cached per process), float32 RGB 1280x720 with the gouache finish applied."""
    global _BG
    if _BG is not None:
        return _BG
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * SS, H * SS)
    ctx = cairo.Context(surf)
    ctx.scale(SS, SS)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    bg_sky(ctx)
    bg_far(ctx)
    bg_floor(ctx)
    bg_buttes(ctx)
    bg_ledge(ctx)
    bg_rock(ctx)
    bg_props(ctx)
    a = np.ndarray((H * SS, W * SS, 4), np.uint8, surf.get_data())[:, :, :3][:, :, ::-1].astype(np.float32) / 255
    img = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
    # gouache finish: low-frequency mottling + paper tooth (static, like a painted card)
    rng = np.random.default_rng(3)
    low = cv2.resize(rng.normal(0, 1, (H // 24, W // 24)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    mid = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 2.2)
    fib = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigmaX=5, sigmaY=.7)
    tex = 1 + .018 * low + .05 * mid + .03 * fib
    img = img * tex[..., None]
    _BG = np.clip(img, 0, 1).astype(np.float32)
    return _BG


# ------------------------------------------------------------------------------------------------ Pip, the armadillo (flat cel)
LW = 3.4   # outer ink weight
LI = 2.1   # inner ink weight

PROUD = dict(
    x=612.0, gy=652.0, facing=1.0, sx=1.0, sy=1.0, lean=.035, hop=0.0,
    head_rot=.15, head_dx=0.0, head_dy=0.0,
    eye_s=1.0, pup_x=-.30, pup_y=.30, lid=.27, brow=0.0, brow_tilt=.0, blink=0.0,
    ear_far=-.26, ear_near=.30, ear_len=.92,
    mouth=0.0, smile=1.0, frown=0.0,
    hn_x=126.0, hn_y=-178.0, broom_rot=0.0,   # near hand position (body space) and broom angle about the grip
    two_hand=0.0,                             # 0 = far hand on hip, 1 = far hand on the broom handle too
    crouch=0.0, tail_curl=0.0,
)


def eye(ctx, M, cx, cy, rx, ry, P, near):
    E = ap(M, circ(cx, cy, rx * P["eye_s"], ry * P["eye_s"], 48))
    fillp(ctx, E, PC["white"])
    ctx.save()
    clip_shape(ctx, E)
    if P["blink"] < .99:
        pr = (5.2 if not near else 6.2) * (1 - .35 * max(0, P["eye_s"] - 1))
        px = cx + P["pup_x"] * rx * P["eye_s"] * .62
        py = cy + P["pup_y"] * ry * P["eye_s"] * .55 + 2
        fillp(ctx, ap(M, circ(px, py, pr, pr * 1.6, 32)), INK)
    # upper lid (skin) coming down over the eye: lid 0 = wide open, 1 = shut
    lid = max(P["lid"], P["blink"])
    if lid > .02:
        top = cy - ry * P["eye_s"] - 2
        yl = top + (2 * ry * P["eye_s"] + 4) * lid
        tilt = (P["brow_tilt"] * 6) * (1 if near else -1)
        L = ap(M, [(cx - rx * 2, top - 30), (cx + rx * 2, top - 30), (cx + rx * 2, yl - tilt), (cx, yl + 2.5), (cx - rx * 2, yl + tilt)])
        fillp(ctx, crs(L, True, 6, .4), PC["skin"])
        ink(ctx, ap(M, crs([(cx - rx * 1.3, yl + tilt * .6), (cx, yl + 2.5), (cx + rx * 1.3, yl - tilt * .6)], False, 10)), LI * 1.15)
    ctx.restore()
    ink(ctx, E, closed_widths(E, LI * 1.05, .5), closed=True)


def head(ctx, M, P):
    # ears first (they sit behind the head)
    for base, ang, far in (((8, -100), P["ear_far"], True), ((32, -92), P["ear_near"], False)):
        Me = M @ Tm(*base) @ Rm(ang) @ Sm(1.08, P["ear_len"])
        ear = [(-15, 4), (-21, -26), (-18, -56), (-8, -80), (3, -94), (12, -74), (19, -40), (14, 4)]
        shape(ctx, ear, PC["skin"] if not far else mix(PC["skin"], PC["skin_dk"], .25), LW * .95, Me)
        inner = [(-8, -6), (-12, -30), (-9, -58), (0, -80), (6, -64), (9, -36), (6, -6)]
        shape(ctx, inner, PC["skin_dk"], 0, Me, line=None)
    # skull + long snout as one silhouette
    hd = [(30, -6), (50, -36), (50, -72), (32, -100), (2, -112), (-26, -104), (-46, -88), (-72, -70), (-100, -54), (-120, -42),
          (-128, -28), (-120, -14), (-98, -9), (-70, -5), (-48, 0), (-22, 6), (6, 8)]
    Hd = shape(ctx, hd, PC["skin"], LW, M, tension=1.0)
    ctx.save()
    clip_shape(ctx, Hd)
    # head shield: one armour plate from the crown down the top of the snout, with transverse scute rows
    cap = [(70, -60), (60, -120), (10, -150), (-60, -120), (-110, -80), (-140, -60), (-128, -40), (-114, -38), (-96, -45), (-72, -58),
           (-50, -74), (-26, -90), (-4, -92), (20, -88), (42, -78), (56, -64)]
    C = ap(M, crs(cap, True, 10, .8))
    fillp(ctx, C, PC["shell"])
    seam = [(-114, -38), (-96, -45), (-72, -58), (-50, -74), (-26, -90), (-4, -92), (20, -88), (42, -78), (56, -64)]
    tline(ctx, seam, LI * 1.1, M, t0=.5, t1=.5)
    for k, (sx, sy) in enumerate(((-108, -42), (-96, -47), (-84, -53), (-72, -60), (-60, -67), (-48, -76), (-36, -86))):
        L = 10 + k * 2.2
        tline(ctx, [(sx, sy), (sx + L * .45, sy - L)], LI * .75, M, col=PC["shell_dk"], t0=.6, t1=.2)
    for row in ((-8, -102, 14, -104, 34, -96, 50, -78), (0, -116, 22, -116, 42, -100)):
        tline(ctx, np.array(row, float).reshape(-1, 2), LI * .75, M, col=PC["shell_dk"], t0=.3, t1=.3)
    ctx.restore()
    ink(ctx, Hd, closed_widths(Hd, LW), closed=True)
    # cheek fold and the closed, self-satisfied mouth under the snout
    sm, fr, mo = P["smile"], P["frown"], P["mouth"]
    if mo < .05:
        lip = [(-104, -17), (-82, -14), (-60, -13 + 2 * fr), (-46, -16 + 6 * fr - 6 * sm)]
        tline(ctx, lip, LI * 1.05, M, t0=.2, t1=.5)
        tline(ctx, [(-40, -30 + 4 * fr), (-36, -22 + 4 * fr), (-40, -14)], LI * .9, M, t0=.2, t1=.2)
    else:
        # an open "O" mouth for the take / a grunt
        mo_pts = [(-92, -16), (-72, -14), (-56, -12), (-58, -12 + 18 * mo), (-74, -8 + 22 * mo), (-90, -10 + 12 * mo)]
        Mo = shape(ctx, mo_pts, PC["mouth"], LI * 1.1, M, tension=.9)
        fillp(ctx, ap(M, circ(-72, -8 + 14 * mo, 9, 5 * mo + 1, 20)), PC["tongue"])
        ink(ctx, Mo, LI * 1.1, closed=True)
    # nose button
    N = ap(M, circ(-127, -28, 12, 10, 32, rot=-.3))
    fillp(ctx, N, PC["nose"])
    fillp(ctx, ap(M, circ(-131, -33, 3.4, 2.2, 16, rot=-.3)), mix(PC["nose"], PC["white"], .45))
    ink(ctx, N, LI, closed=True)
    # eyes: two tall ovals that touch, high on the face; the far one narrower
    eye(ctx, M, -40, -84, 11.5, 19, P, False)
    eye(ctx, M, -16, -88, 14.5, 21.5, P, True)
    # brows: floating tapered strokes
    b, bt = P["brow"], P["brow_tilt"]
    tline(ctx, [(-50, -110 - 10 * b - 5 * bt), (-42, -114 - 12 * b + 1 * bt), (-32, -113 - 11 * b + 8 * bt)], LI * 1.5, M, t0=.15, t1=.5)
    tline(ctx, [(-24, -118 - 12 * b + 8 * bt), (-12, -121 - 13 * b + 1 * bt), (0, -117 - 11 * b - 6 * bt)], LI * 1.5, M, t0=.5, t1=.15)


def fist(ctx, M, x, y, rot=0.0, s=1.0):
    Mf = M @ Tm(x, y) @ Rm(rot) @ Sm(s)
    shape(ctx, [(-10, -9), (2, -12), (11, -6), (12, 5), (4, 11), (-9, 9), (-13, 0)], PC["skin"], LI * 1.2, Mf)
    for yy in (-3, 3):
        tline(ctx, [(2, yy - 1), (9, yy)], LI * .7, Mf, t0=.3, t1=.3)
    shape(ctx, [(-12, -6), (-6, -13), (2, -11), (-2, -4)], PC["skin"], LI, Mf)  # thumb over the top


def foot(ctx, M, ax, ay, dirx, rot=0.0):
    Mf = M @ Tm(ax, ay) @ Sm(dirx, 1) @ Rm(rot)
    pts = [(-12, -10), (-16, 0), (-10, 8), (20, 9), (44, 8), (55, 2), (53, -8), (41, -14), (20, -14), (2, -16)]
    shape(ctx, pts, PC["skin"], LW * .95, Mf)
    for a, b2 in (((44, -12), (41, -1)), ((32, -13), (30, -3))):
        tline(ctx, [a, b2], LI * .8, Mf, t0=.2, t1=.6)
    for tx, ty in ((54, -2), (47, 5), (36, 7)):
        fillp(ctx, ap(Mf, circ(tx, ty, 2.6, 1.8, 10)), INK)


def broom_matrix(M0, P):
    return M0 @ Tm(P["hn_x"], P["hn_y"]) @ Rm(P["broom_rot"])


def broom(ctx, Mb):
    """Straw broom in grip space: the near hand holds it at the origin, the bristles point down +y."""
    tube(ctx, [(-1, 124), (0, -12), (1, -140)], 8.5, PC["handle"], Mb, lw=2.4)
    br = [(-15, 130), (14, 130), (30, 162), (35, 180), (22, 182), (6, 179), (-10, 182), (-26, 180), (-34, 181), (-30, 162)]
    B = shape(ctx, br, PC["straw"], LW * .9, Mb, tension=.6)
    ctx.save()
    clip_shape(ctx, B)
    for k in range(9):
        u = (k - 4) / 4
        tline(ctx, [(u * 12, 132), (u * 22, 158), (u * 31, 182)], 1.5, Mb, col=PC["straw_dk"], t0=.3, t1=.8)
    ctx.restore()
    shape(ctx, [(-15, 118), (15, 118), (16, 132), (-16, 132)], PC["band"], LI * 1.1, Mb, tension=.2)
    tline(ctx, [(-15, 125), (15, 125)], LI * .7, Mb, col=PC["scarf_dk"])


def canteen(ctx, M, spill=0.0):
    """A round felt-covered canteen with a cap and a strap; spill>0 tips water out of the spout."""
    body = circ(0, 0, 30, 30, 40)
    shape(ctx, body, hexc("#6f8f7a"), LW * .9, M)
    shape(ctx, circ(0, 0, 21, 21, 32), hexc("#86a690"), LI * .7, M, line=hexc("#4b6656"))
    shape(ctx, [(-8, -28), (8, -28), (9, -44), (-9, -44)], hexc("#c9ccd2"), LI, M, tension=.2)
    shape(ctx, [(-11, -44), (11, -44), (11, -52), (-11, -52)], hexc("#9aa0a8"), LI, M, tension=.2)
    tline(ctx, [(-28, -10), (-44, -40), (-30, -66), (0, -74)], 3.0, M, col=hexc("#7a5a3c"), t0=.8, t1=.8)
    if spill > 0:
        for k in range(5):
            s = (k + 1) / 5
            fillp(ctx, ap(M, circ(-2 + 6 * (k % 2), -60 - 26 * s * spill, 5 - .6 * k, 7 - .8 * k, 12)), hexc("#9fd6ee"))


def ik(S, Hh, L1, L2, bend=1.0):
    S, Hh = np.asarray(S, float), np.asarray(Hh, float)
    d = Hh - S
    dist = float(np.linalg.norm(d)) + 1e-9
    u = d / dist
    dist = min(max(dist, abs(L1 - L2) + 1), L1 + L2 - .5)
    a = (L1 * L1 - L2 * L2 + dist * dist) / (2 * dist)
    h = math.sqrt(max(L1 * L1 - a * a, 0))
    E = S + u * a + np.array([-u[1], u[0]]) * h * bend
    return E, S + u * dist


def arm(ctx, M0, S, Hh, L1, L2, bend, fist_rot):
    E, Hr = ik(S, Hh, L1, L2, bend)
    d = Hr - E
    d = d / (np.linalg.norm(d) + 1e-9)
    tube(ctx, [tuple(S), tuple(E), tuple(Hr - d * 6)], 11, PC["skin"], M0)
    fist(ctx, M0, Hr[0], Hr[1], rot=fist_rot, s=1.05)


def body_matrix(P):
    sc = P.get("sc", 1.0)
    return Tm(P["x"], P["gy"] - P["hop"]) @ Sm(P["facing"] * sc, sc) @ Rm(P["lean"]) @ Sm(P["sx"], P["sy"])


def pip(ctx, P):
    hop = P["hop"]
    M0 = body_matrix(P)
    cr = P["crouch"]
    # cast shadow on the ground (flat, darker ochre), shrinks when he is in the air
    sc = P.get("sc", 1.0)
    sw = (150 + 40 * cr) * (1 / (1 + hop / 120)) * sc
    if P.get("shadow", True):
        fillp(ctx, circ(P["x"] + 24 * sc * np.sign(P["facing"]), P["gy"] + 2, sw * .62 * P["sx"], 11 * sc, 48), PC["shadow"])
    # tail: banded, tapering, curling at the tip (behind everything)
    tc = P["tail_curl"]
    tail = [(78, -86), (112, -54), (146, -26), (182, -12 - 8 * tc), (212, -16 - 26 * tc), (228, -34 - 34 * tc), (222, -52 - 40 * tc)]
    TP = ap(M0, crs(tail, False, 10))
    R = ribbon(TP, 26, 6)
    shape(ctx, None, PC["shell"], LW, dense=R)
    for k in range(3, len(TP) - 3, 5):
        d = TP[min(k + 1, len(TP) - 1)] - TP[k - 1]
        d /= np.linalg.norm(d) + 1e-9
        nrm = np.array([-d[1], d[0]])
        w = (26 + (6 - 26) * k / len(TP)) / 2 * .92
        ink(ctx, np.array([TP[k] + nrm * w, TP[k] - nrm * w]), LI * .8)
    # legs + feet
    kx = 12 * cr
    tube(ctx, [(-26, -96), (-36 - kx, -54 + 8 * cr), (-38, -14)], 12, PC["skin"], M0)
    tube(ctx, [(14, -94), (24 + kx, -54 + 8 * cr), (30, -14)], 12, PC["skin"], M0)
    foot(ctx, M0, -38, -9, -1)
    foot(ctx, M0, 30, -9, 1)
    # shell (the banded back), then the soft belly in front of it
    shell = [(-28, -244), (0, -260), (40, -256), (78, -230), (102, -188), (110, -140), (102, -100), (84, -74), (52, -62), (20, -66), (-4, -80),
             (-16, -140), (-24, -200)]
    S = shape(ctx, shell, PC["shell"], LW, M0)
    ctx.save()
    clip_shape(ctx, S)
    for j, y in enumerate((-204, -184, -164, -144, -124)):
        band = [(-10, y - 4), (40, y + 6), (80, y + 5), (116, y - 4)]
        tline(ctx, band, LI * 1.0, M0, t0=.6, t1=.6)
        for tx in np.linspace(14, 104, 7):  # scute ticks between the bands
            ty = y + 6 - 7 * ((tx - 50) / 60) ** 2
            tline(ctx, [(tx + (j % 2) * 6, ty + 3), (tx + (j % 2) * 6 + 1, ty + 13)], LI * .6, M0, col=PC["shell_dk"], t0=.4, t1=.4)
    for (a, b2, c) in (((20, -236), (50, -238), (78, -222)), ((24, -94), (56, -86), (84, -94))):  # scapular / pelvic shield rows
        tline(ctx, [a, b2, c], LI * .7, M0, col=PC["shell_dk"], t0=.3, t1=.3)
    ctx.restore()
    ink(ctx, S, closed_widths(S, LW), closed=True)
    belly = [(-24, -240), (-44, -216), (-56, -180), (-62, -142), (-56, -108), (-38, -88), (-10, -82), (6, -92), (10, -132), (8, -182),
             (0, -222), (-8, -242)]
    shape(ctx, belly, PC["belly"], LW, M0)
    prop = P.get("prop", "broom")
    two = P["two_hand"] > .5 or prop != "broom"
    FS, NS = (-40, -212), (8, -214)
    if not two:  # far arm: hand on hip, elbow out
        arm(ctx, M0, FS, (-58, -132), 57, 52, 1.0, .6)
    # neckerchief: a red triangle over the chest, knot and flaps on the near side
    sc = [(-36, -238), (-20, -244), (8, -246), (18, -238), (4, -224), (-14, -202), (-26, -220)]
    shape(ctx, sc, PC["scarf"], LI * 1.3, M0, tension=.7)
    for d in ((-18, -230), (-4, -236), (-10, -218), (6, -240), (-24, -236)):
        fillp(ctx, ap(M0, circ(d[0], d[1], 2.4, 2.0, 10)), PC["dot"])
    shape(ctx, [(14, -242), (34, -256), (40, -246), (24, -236)], PC["scarf"], LI * 1.2, M0)
    shape(ctx, [(14, -236), (34, -226), (28, -218), (14, -228)], PC["scarf"], LI * 1.2, M0)
    shape(ctx, circ(16, -238, 6.5, 6, 16), PC["scarf_dk"], LI * 1.1, M0)
    # broom + near arm, then (two-handed) the far hand on the handle; or a canteen held in both hands; or free hands
    if prop == "broom":
        Mb = broom_matrix(M0, P)
        broom(ctx, Mb)
        arm(ctx, M0, NS, (P["hn_x"], P["hn_y"]), 62, 68, P.get("bend_n", 1.0), P["broom_rot"] - .2)
        if two:
            g2 = ap1(Tm(P["hn_x"], P["hn_y"]) @ Rm(P["broom_rot"]), 0, -46)
            arm(ctx, M0, FS, g2, 57, 60, P.get("bend_f", 1.0), P["broom_rot"] - .2)
    else:
        hn, hf = (P["hn_x"], P["hn_y"]), (P.get("hf_x", P["hn_x"] - 40), P.get("hf_y", P["hn_y"]))
        arm(ctx, M0, FS, hf, 57, 60, P.get("bend_f", 1.0), P.get("hand_rot", 0.0))
        if prop == "canteen":
            canteen(ctx, M0 @ Tm((hn[0] + hf[0]) / 2, (hn[1] + hf[1]) / 2 - 6) @ Rm(P.get("canteen_rot", 0.0)), P.get("spill", 0.0))
        arm(ctx, M0, NS, hn, 62, 68, P.get("bend_n", 1.0), P.get("hand_rot", 0.0))
    # head
    neck = (-6, -236)
    Mh = M0 @ Tm(neck[0] + P["head_dx"], neck[1] + P["head_dy"]) @ Rm(P["head_rot"])
    head(ctx, Mh, P)


# ------------------------------------------------------------------------------------------------ effects (also flat cels)
def tumbleweed(ctx, x, y, r, rot, seed=4):
    rng = np.random.default_rng(seed)
    M = Tm(x, y) @ Rm(rot) @ Sm(r / 32)
    fillp(ctx, ap(M, circ(0, 0, 29, 27, 32)), alpha(PC["weed"], .42))
    lw = max(.7, 1.7 * r / 32)
    for k in range(20):  # tangled loops
        n = rng.integers(4, 7)
        ang = rng.uniform(0, 2 * np.pi) + np.linspace(0, rng.uniform(2.5, 5), n)
        rad = rng.uniform(10, 31, n)
        pts = np.stack([np.cos(ang) * rad, np.sin(ang) * rad * .95], 1)
        tline(ctx, pts, lw * rng.uniform(.7, 1.1), M, col=PC["weed_dk"] if k % 3 else INK, t0=.3, t1=.3)
    ring = circ(0, 0, 31, 29, 18) + rng.normal(0, 2.2, (18, 2))
    P = ap(M, crs(ring, True, 6))
    ink(ctx, P, closed_widths(P, max(1.0, 2.6 * r / 32)), closed=True)
    for k in range(9):  # twigs poking out
        a = rng.uniform(0, 2 * np.pi)
        tline(ctx, [(29 * math.cos(a), 29 * math.sin(a)), (38 * math.cos(a + .15), 38 * math.sin(a + .15))], lw, M, t0=.8, t1=.2)


def dust(ctx, cx, cy, size, seed, grow=1.0, drift=(0.0, 0.0), fade=1.0, n=9, spread=(1.0, .45)):
    """A cartoon dust cloud: outlined puffs (outer silhouette only), a shadow tone underneath and a few inner curls."""
    if fade <= .01 or size * grow < 1:
        return
    rng = np.random.default_rng(seed)
    puffs = []
    for k in range(n):
        ox, oy = rng.normal(0, spread[0]), rng.normal(0, spread[1])
        rr = rng.uniform(.45, .8) * (1 - .3 * abs(ox) / 2)
        puffs.append((cx + (ox * size * .7 + drift[0] * (k / n + .5)) * grow ** .5, cy + oy * size * .5 + drift[1] * (k / n + .5) - .2 * size * (grow - 1),
                      size * rr * grow))
    col_line = alpha(PC["dust_line"], fade)
    for x, y, r in puffs:
        fillp(ctx, circ(x, y, r + 2.0, r * .86 + 2.0, 24), col_line)
    for x, y, r in puffs:
        fillp(ctx, circ(x, y, r, r * .86, 24), alpha(PC["dust"], fade))
    for x, y, r in puffs:
        fillp(ctx, circ(x + r * .18, y + r * .32, r * .7, r * .42, 20), alpha(PC["dust_sh"], .8 * fade))
    for x, y, r in puffs[:4]:
        if r > 10:
            P = circ(x - r * .1, y - r * .05, r * .5, r * .38, 20)[3:11]
            ink(ctx, P, 1.4, col_line)


def smear_arc(ctx, M0, P, r0, r1, n=10, span=1.7):
    """Dry-brush smear for a fast broom swing: thick broken streaks where the bristles travelled, fading toward the start."""
    rng = np.random.default_rng(int(abs(r0 * 100 + r1 * 37)) + 5)
    if abs(r1 - r0) > span:
        r0 = r1 - math.copysign(span, r1 - r0)
    G = M0 @ Tm(P["hn_x"], P["hn_y"])
    for k in range(n):
        rad = 128 + 56 * k / (n - 1)
        a = np.linspace(r0, r1, 30)
        pts = np.stack([-np.sin(a) * rad, np.cos(a) * rad], 1)
        Q = ap(G, pts)
        col = PC["straw"] if k % 3 else PC["straw_dk"]
        s = np.linspace(0, 1, len(Q))
        w = rng.uniform(3.5, 7.0) * s ** 1.2
        ctx.set_source_rgba(*col)
        start = rng.uniform(0, .35)
        for j in range(len(Q) - 1):
            if s[j] < start or rng.uniform() < .14:
                continue
            ctx.set_line_width(max(.3, w[j]))
            ctx.move_to(*Q[j])
            ctx.line_to(*Q[j + 1])
            ctx.stroke()
    for rad in (122, 188):  # two thin ink speed lines bounding the smear
        a = np.linspace(r0 + (r1 - r0) * .45, r1, 20)
        Q = ap(G, np.stack([-np.sin(a) * rad, np.cos(a) * rad], 1))
        ink(ctx, Q, np.linspace(.2, 1.8, len(Q)))


def speed_lines(ctx, x0, x1, ys, col, w=2.0, seed=0):
    rng = np.random.default_rng(seed)
    for y in ys:
        a = x0 + rng.uniform(0, 30)
        b = x1 - rng.uniform(0, 60)
        drybrush(ctx, a, y, b, y + rng.normal(0, 1), w * rng.uniform(.6, 1.2), col, rng, broken=.3, wob=.3)


# ------------------------------------------------------------------------------------------------ compositing
def cel_layer(draw_fn):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * SS, H * SS)
    ctx = cairo.Context(surf)
    ctx.scale(SS, SS)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    draw_fn(ctx)
    a = np.ndarray((H * SS, W * SS, 4), np.uint8, surf.get_data())
    a = cv2.resize(a.astype(np.float32) / 255, (W, H), interpolation=cv2.INTER_AREA)
    return a[:, :, 2::-1], a[:, :, 3:4]   # premultiplied RGB, alpha


def finish(img, i=0, grain=.010):
    img = cv2.GaussianBlur(img, (0, 0), .45)
    rng = np.random.default_rng(777 + i)
    g = cv2.resize(rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32), (W, H))
    img = img + g[..., None] * grain
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = ((xx / W - .5) ** 2 * 1.2 + (yy / H - .5) ** 2) * 2
    img = img * (1 - .10 * np.clip(v, 0, 1))[..., None]
    return np.clip(img, 0, 1)


def to8(img):
    return (np.clip(img, 0, 1) * 255 + .5).astype(np.uint8)


def render(P, fx=None, i=0, pre=None):
    bg = background()
    rgb, a = cel_layer(lambda ctx: (pre(ctx) if pre else None, pip(ctx, P) if P is not None else None, fx(ctx) if fx else None))
    img = rgb + bg * (1 - a)
    return to8(finish(img, i))


def key():
    return render(dict(PROUD), None, 0)[8:712]
