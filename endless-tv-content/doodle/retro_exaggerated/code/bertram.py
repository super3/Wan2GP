"""Bertram Bigsby, crossing guard: a top-heavy bison in a late-90s TV-cartoon style (thick even outer contour, flat cel colour,
small head on a giant rounded mass, pin legs) over a Hanna-Barbera-flavoured painted background (flat shapes, sponged texture,
a thin loose line laid slightly off-register).  Everything is drawn in code; deterministic."""
import math

import cairo
import cv2
import numpy as np

W, H, FPS = 1280, 720, 24
DUR = 5.0
N = int(DUR * FPS)
X0, Y0 = 520, 655          # the character's ground point (between the feet), screen px


def hexc(h, a=1.0):
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, a)


INK = hexc("#17111a")
C = dict(
    wool=hexc("#a65a37"), face=hexc("#d39663"), muzzle=hexc("#f4d0a2"), nose=hexc("#6a3a30"), horn=hexc("#f7eed6"),
    cap=hexc("#2f4f93"), cap2=hexc("#223a72"), badge=hexc("#ffd34d"), sash=hexc("#ff8f3a"), trouser=hexc("#2f4f93"),
    shoe=hexc("#211a24"), hand=hexc("#6b3d2a"), red=hexc("#e8413f"), white=hexc("#fdf8ec"), pole=hexc("#e8e3d6"),
    eye=hexc("#fffdf6"), mouth=hexc("#5e1d27"), tongue=hexc("#ec7a80"), shine=hexc("#ffffff"),
)
LW = 3.6          # interior ink width (1x px)
OUTLINE = 2.6     # extra silhouette contour added around the whole cel (1x px)


# =============================================================================================== geometry helpers
def cr(pts, closed=True, per=20, tension=1.0):
    """Dense Catmull-Rom polyline through pts."""
    p = np.asarray(pts, float)
    n = len(p)
    out = []
    segs = n if closed else n - 1
    for i in range(segs):
        if closed:
            p0, p1, p2, p3 = p[(i - 1) % n], p[i], p[(i + 1) % n], p[(i + 2) % n]
        else:
            p0, p1, p2, p3 = p[max(i - 1, 0)], p[i], p[min(i + 1, n - 1)], p[min(i + 2, n - 1)]
        c1, c2 = p1 + (p2 - p0) * tension / 6, p2 - (p3 - p1) * tension / 6
        t = np.linspace(0, 1, per, endpoint=False)[:, None]
        out.append((1 - t) ** 3 * p1 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * p2)
    if not closed:
        out.append(p[-1:])
    return np.concatenate(out)


def resample(P, step=2.0, closed=True):
    Q = np.vstack([P, P[:1]]) if closed else P
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    m = max(int(L / step), 8)
    t = np.linspace(0, L, m, endpoint=not closed)
    return np.stack([np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])], 1), L


def normals(P, closed=True):
    if closed:
        d = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    else:
        d = np.gradient(P, axis=0)
    n = np.stack([d[:, 1], -d[:, 0]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    return n * (1 if area > 0 else -1)   # outward for either winding (y down)


def scallop(P, lam=44.0, amp=10.0, power=.55, ranges=None, seed=0, amp_fn=None, skew=1.0):
    """Wool / cloud edge: round bumps with sharp notches between them, along the closed outline P.
    ranges: list of (f0, f1, amp_scale) over the perimeter fraction; default all."""
    Q, L = resample(P, 1.5)
    nrm = normals(Q)
    s = np.linspace(0, 1, len(Q), endpoint=False)
    rng = np.random.default_rng(seed)
    nb = max(int(L / lam), 3)
    jit = rng.uniform(.8, 1.2, nb + 2)
    edges = np.concatenate([[0], np.cumsum(jit)])
    edges = edges / edges[nb] * L
    pos = s * L
    k = np.clip(np.searchsorted(edges, pos, side="right") - 1, 0, nb - 1)
    u = (pos - edges[k]) / (edges[k + 1] - edges[k])
    u = u ** skew
    bump = np.sqrt(np.clip(1 - (2 * u - 1) ** 2, 0, 1)) ** power
    a = np.zeros(len(Q))
    for f0, f1, g in (ranges or [(0, 1, 1)]):
        if f0 <= f1:
            w = (s >= f0) & (s <= f1)
            fade = np.clip(np.minimum(s - f0, f1 - s) / .03, 0, 1)
        else:
            w = (s >= f0) | (s <= f1)
            fade = np.clip(np.minimum((s - f0) % 1, (f1 - s) % 1) / .03, 0, 1)
        a = np.where(w, np.maximum(a, g * fade), a)
    amp_k = amp * rng.uniform(.75, 1.15, nb)[k]
    if amp_fn is not None:
        a = a * amp_fn(Q)
    return Q + nrm * (bump * amp_k * a)[:, None]


def tube(pts, w0, w1, per=14):
    """Closed outline of a limb along a centreline, width w0 -> w1, round caps."""
    C_ = cr(pts, closed=False, per=per)
    C_, _ = resample(C_, 2.0, closed=False)
    d = np.gradient(C_, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    w = np.linspace(w0, w1, len(C_))[:, None] / 2
    left, right = C_ + n * w, C_ - n * w
    a_end = math.atan2(n[-1, 1], n[-1, 0])
    cap_e = [C_[-1] + w[-1, 0] * np.array([math.cos(a_end - t), math.sin(a_end - t)]) for t in np.linspace(0, math.pi, 12)]
    a_st = math.atan2(-n[0, 1], -n[0, 0])
    cap_s = [C_[0] + w[0, 0] * np.array([math.cos(a_st - t), math.sin(a_st - t)]) for t in np.linspace(0, math.pi, 12)]
    return np.vstack([left, cap_e, right[::-1], cap_s])


def ell(cx, cy, rx, ry, rot=0.0, n=72):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    x, y = rx * np.cos(t), ry * np.sin(t)
    c, s = math.cos(rot), math.sin(rot)
    return np.stack([cx + x * c - y * s, cy + x * s + y * c], 1)


def path(ctx, P, closed=True):
    ctx.move_to(*P[0])
    for q in P[1:]:
        ctx.line_to(*q)
    if closed:
        ctx.close_path()


def fs(ctx, P, col, lw=LW, closed=True):
    path(ctx, P, closed)
    ctx.set_source_rgba(*col)
    if lw:
        ctx.fill_preserve()
        ctx.set_source_rgba(*INK)
        ctx.set_line_width(lw)
        ctx.stroke()
    else:
        ctx.fill()


def line(ctx, P, lw=LW, col=INK):
    path(ctx, P, False)
    ctx.set_source_rgba(*col)
    ctx.set_line_width(lw)
    ctx.stroke()


def taper_line(ctx, pts, w0, w1, col=INK, per=14):
    """A brush line that thins toward its ends (interior detail lines)."""
    C_ = cr(pts, closed=False, per=per)
    C_, _ = resample(C_, 1.5, closed=False)
    d = np.gradient(C_, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    t = np.linspace(0, 1, len(C_))
    w = (w0 + (w1 - w0) * t) * np.sin(np.pi * np.clip(t * .85 + .075, 0, 1)) ** .5
    w = w[:, None] / 2
    path(ctx, np.vstack([C_ + n * w, (C_ - n * w)[::-1]]))
    ctx.set_source_rgba(*col)
    ctx.fill()


def lerp(a, b, t):
    return a + (b - a) * t


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# =============================================================================================== background
BG_LINE = hexc("#3b3448")
B = dict(
    sky0=hexc("#9fd6cd"), sky1=hexc("#b2dfd2"), sky2=hexc("#c5e7d6"), cloud=hexc("#f8f5e9"), cloud_sh=hexc("#dcefe5"),
    hill0=hexc("#93cbb0"), hill1=hexc("#7fbe9f"),
    house=hexc("#f5d47e"), house_sh=hexc("#e8c06a"), trim=hexc("#fff7e4"), roof=hexc("#5c6386"), roof2=hexc("#4c5375"),
    door=hexc("#ee7c61"), win=hexc("#86b8d2"), win2=hexc("#a8cfe0"), stone=hexc("#d9bc90"), stone2=hexc("#c6a77a"),
    school=hexc("#f3a487"), school_sh=hexc("#e48f73"), slab=hexc("#f7f1e3"), slab2=hexc("#ddd1b9"), swin=hexc("#6f9fc4"),
    sframe=hexc("#47608a"), breeze=hexc("#fbf3e2"), sdoor=hexc("#7cc6b9"),
    tree=hexc("#5aaa89"), tree_d=hexc("#3f8f73"), tree_l=hexc("#84c8a5"), trunk=hexc("#6e4f3b"),
    lawn=hexc("#a8d276"), lawn_d=hexc("#8fc061"), hedge=hexc("#64b083"), hedge_d=hexc("#4c9a6d"),
    walk_far=hexc("#ecdcc3"), curb_far=hexc("#ccbaa1"), road=hexc("#8f96b8"), road_d=hexc("#848bae"), zebra=hexc("#f8f4e9"),
    curb_top=hexc("#f2e4ca"), curb=hexc("#c8b397"), walk=hexc("#eedab6"), walk_j=hexc("#d5be97"), shadow=hexc("#cdb48f"),
)


class BGPainter:
    def __init__(self, ctx, seed=11):
        self.ctx = ctx
        self.lines = []   # (polyline, closed) for the loose overlay pass
        self.rng = np.random.default_rng(seed)

    def shape(self, P, col, outline=True, closed=True):
        ctx = self.ctx
        path(ctx, P, closed)
        ctx.set_source_rgba(*col)
        ctx.fill()
        if outline:
            self.lines.append((np.asarray(P, float), closed))

    def sponge(self, P, col, n, rmin, rmax, alpha=1.0, seed=0, region=None):
        """Sponge-daubed texture inside P (clipped): little irregular blobs."""
        ctx = self.ctx
        rng = np.random.default_rng(seed)
        x0, y0 = P.min(0)
        x1, y1 = P.max(0)
        if region is not None:
            x0, y0, x1, y1 = region
        ctx.save()
        path(ctx, P)
        ctx.clip()
        for _ in range(n):
            cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
            r = rng.uniform(rmin, rmax)
            blob = ell(cx, cy, r, r * rng.uniform(.5, .9), rng.uniform(0, math.pi), n=10)
            blob += rng.normal(0, r * .18, blob.shape)
            path(ctx, cr(blob, per=4))
            ctx.set_source_rgba(col[0], col[1], col[2], alpha)
            ctx.fill()
        ctx.restore()

    def flush(self):
        self.overlay_lines()
        self.lines = []

    def overlay_lines(self, offset=(2.2, -1.6), lw=2.0):
        """The Hanna-Barbera habit: a thin, slightly loose line drawn over the colour shapes, a little off-register,
        with an occasional break."""
        ctx = self.ctx
        rng = self.rng
        for P, closed in self.lines:
            Q, L = resample(P, 2.0, closed)
            Q = Q + np.asarray(offset) + rng.normal(0, .25, Q.shape)
            n = len(Q)
            keep = np.ones(n, bool)
            for _ in range(int(L // 260) + (1 if L > 120 else 0)):
                a = rng.integers(0, n)
                keep[a:a + rng.integers(4, 12)] = False
            segs, cur = [], []
            for k in range(n + (1 if closed else 0)):
                kk = k % n
                if keep[kk]:
                    cur.append(Q[kk])
                elif cur:
                    segs.append(cur)
                    cur = []
            if cur:
                segs.append(cur)
            for sgm in segs:
                if len(sgm) < 2:
                    continue
                path(ctx, np.asarray(sgm), False)
                ctx.set_source_rgba(*BG_LINE)
                ctx.set_line_width(lw * rng.uniform(.85, 1.1))
                ctx.stroke()


def lollipop_tree(bg, x, ground, h, r, seed):
    ctx = bg.ctx
    rng = np.random.default_rng(seed)
    # a thin trunk, slightly bowed, with a fork
    tr = np.array([(x, ground), (x + 3, ground - h * .45), (x - 2, ground - h * .8)])
    T = tube(tr, 9, 5)
    bg.shape(T, B["trunk"])
    cy = ground - h
    crown = scallop(cr(ell(x, cy, r, r * .82, 0, n=9) + rng.normal(0, r * .05, (9, 2))), lam=r * .55, amp=r * .1, seed=seed)
    bg.shape(crown, B["tree"])
    bg.sponge(crown, B["tree_d"], 70, r * .06, r * .16, .85, seed=seed + 1,
              region=(x - r * 1.1, cy - r * .1, x + r * 1.1, cy + r))
    bg.sponge(crown, B["tree_l"], 30, r * .05, r * .12, .9, seed=seed + 2,
              region=(x - r * .9, cy - r * .9, x + r * .3, cy - r * .2))
    # fan-shaped bloom strokes (the stylised leaf clusters of late-50s TV backgrounds)
    for k in range(5):
        a = -2.4 + k * .45 + rng.normal(0, .05)
        px, py = x + math.cos(a) * r * .55, cy + math.sin(a) * r * .45 + r * .1
        for j in range(3):
            b = a + (j - 1) * .28
            ctx.move_to(px, py)
            ctx.line_to(px + math.cos(b) * r * .22, py + math.sin(b) * r * .22)
        ctx.set_source_rgba(*B["tree_d"])
        ctx.set_line_width(2.2)
        ctx.stroke()


def cloud(bg, cx, cy, w, h, seed):
    rng = np.random.default_rng(seed)
    base = cr([(cx - w * .5, cy + h * .25), (cx - w * .38, cy - h * .2), (cx - w * .12, cy - h * .5), (cx + w * .15, cy - h * .42),
               (cx + w * .4, cy - h * .15), (cx + w * .52, cy + h * .25)])
    P = scallop(base, lam=h * .55, amp=h * .16, power=.7, ranges=[(0.0, .66, 1)], seed=seed)
    bg.shape(P, B["cloud"])
    ctx = bg.ctx
    ctx.save()
    path(ctx, P)
    ctx.clip()
    xs = np.linspace(cx - w * .6, cx + w * .6, 40)
    wav = cy + h * .1 + h * .07 * np.sin((xs - cx) / (h * .45) + seed)
    band = np.vstack([np.stack([xs, wav], 1), [(cx + w * .6, cy + h), (cx - w * .6, cy + h)]])
    path(ctx, band)
    ctx.set_source_rgba(*B["cloud_sh"])
    ctx.fill()
    ctx.restore()


def background():
    ss = 2
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * ss, H * ss)
    ctx = cairo.Context(surf)
    ctx.scale(ss, ss)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    bg = BGPainter(ctx)

    # sky: a soft airbrushed card, deeper at the top
    g = cairo.LinearGradient(0, 0, 0, 470)
    g.add_color_stop_rgba(0, *B["sky0"])
    g.add_color_stop_rgba(.55, *B["sky1"])
    g.add_color_stop_rgba(1, *B["sky2"])
    ctx.rectangle(0, 0, W, 480)
    ctx.set_source(g)
    ctx.fill()
    cloud(bg, 210, 120, 250, 70, 3)
    cloud(bg, 1040, 82, 200, 56, 5)
    cloud(bg, 1010, 196, 140, 36, 8)
    bg.flush()

    # far rolling hills
    hills = cr([(-40, 470), (-40, 400), (120, 372), (330, 392), (520, 360), (760, 388), (980, 356), (1180, 380), (1330, 366), (1330, 470)])
    bg.shape(hills, B["hill0"], outline=False)
    bg.sponge(hills, B["hill1"], 160, 3, 9, .7, seed=21, region=(0, 360, W, 440))
    hill_line = cr([(-40, 400), (120, 372), (330, 392), (520, 360), (760, 388), (980, 356), (1180, 380), (1330, 366)], closed=False)
    bg.lines.append((hill_line, False))
    bg.flush()

    # ---- mid-century ranch house (left) -------------------------------------------------------------------
    hx0, hx1, hy_top, hy_bot = 20, 330, 330, 462
    bg.shape(np.array([(hx0, hy_bot), (hx0, hy_top), (hx1, hy_top), (hx1, hy_bot)]), B["house"])
    ctx.rectangle(hx0, hy_top, 40, hy_bot - hy_top)           # a side shadow plane
    ctx.set_source_rgba(*B["house_sh"])
    ctx.fill()
    roof = np.array([(hx0 - 30, hy_top + 6), (150, 268), (hx1 + 36, hy_top + 4), (hx1 + 36, hy_top + 16), (hx0 - 30, hy_top + 18)])
    bg.shape(roof, B["roof"])
    bg.shape(np.array([(hx0 - 30, hy_top + 12), (hx1 + 36, hy_top + 10), (hx1 + 36, hy_top + 18), (hx0 - 30, hy_top + 20)]), B["roof2"], outline=False)
    stone = np.array([(hx0, hy_bot), (hx0, hy_bot - 34), (130, hy_bot - 34), (130, hy_bot)])
    bg.shape(stone, B["stone"])
    rng = np.random.default_rng(4)
    for k in range(22):                                       # flat stone shapes
        sx, sy = rng.uniform(hx0 + 4, 124), rng.uniform(hy_bot - 30, hy_bot - 6)
        bg.shape(ell(sx, sy, rng.uniform(6, 12), rng.uniform(4, 6), 0, n=10), B["stone2"], outline=False)
    bg.shape(np.array([(150, 365), (296, 365), (296, 425), (150, 425)]), B["trim"])     # picture window
    bg.shape(np.array([(158, 372), (288, 372), (288, 418), (158, 418)]), B["win"])
    bg.shape(np.array([(166, 378), (200, 378), (178, 412), (166, 412)]), B["win2"], outline=False)
    for xx in (201, 245):
        bg.lines.append((np.array([(xx, 372), (xx, 418)]), False))
    bg.shape(np.array([(62, 370), (112, 370), (112, 462), (62, 462)]), B["door"])       # door
    bg.shape(ell(102, 418, 3.5, 3.5), B["trim"], outline=False)
    for k in range(3):                                        # starburst ornament on the gable
        a = k * math.pi / 3
        bg.lines.append((np.array([(150 - 14 * math.cos(a), 300 - 14 * math.sin(a)), (150 + 14 * math.cos(a), 300 + 14 * math.sin(a))]), False))

    bg.flush()
    # ---- trees --------------------------------------------------------------------------------------------
    lollipop_tree(bg, 375, 466, 150, 62, 31)
    bg.flush()

    # ---- schoolhouse (right) -------------------------------------------------------------------------------
    sx0, sx1, sy_top, sy_bot = 860, 1200, 318, 462
    bg.shape(np.array([(sx0, sy_bot), (sx0, sy_top), (sx1, sy_top), (sx1, sy_bot)]), B["school"])
    bg.shape(np.array([(sx0 - 34, sy_top - 2), (sx0 - 22, sy_top - 22), (sx1 + 30, sy_top - 22), (sx1 + 40, sy_top - 2)]), B["slab"])
    bg.shape(np.array([(sx0 - 34, sy_top - 2), (sx1 + 40, sy_top - 2), (sx1 + 40, sy_top + 5), (sx0 - 34, sy_top + 5)]), B["slab2"], outline=False)
    ctx.rectangle(sx0, sy_top + 5, sx1 - sx0, 12)
    ctx.set_source_rgba(*B["school_sh"])
    ctx.fill()
    for k in range(5):                                        # ribbon windows
        wx = sx0 + 118 + k * 46
        bg.shape(np.array([(wx, 345), (wx + 38, 345), (wx + 38, 402), (wx, 402)]), B["swin"])
        bg.shape(np.array([(wx + 5, 350), (wx + 16, 350), (wx + 7, 372), (wx + 5, 372)]), B["win2"], outline=False)
    bg.shape(np.array([(sx0 + 110, 404), (sx0 + 348, 404), (sx0 + 348, 410), (sx0 + 110, 410)]), B["slab"])
    # breeze-block screen wall: the retro-modern pattern
    bx0, bx1, by0, by1 = sx0 + 14, sx0 + 96, 336, 462
    bg.shape(np.array([(bx0, by1), (bx0, by0), (bx1, by0), (bx1, by1)]), B["breeze"])
    for yy in np.arange(by0 + 10, by1 - 4, 20):
        for xx in np.arange(bx0 + 10, bx1 - 4, 20):
            d = np.array([(xx, yy - 7), (xx + 7, yy), (xx, yy + 7), (xx - 7, yy)])
            path(ctx, d)
            ctx.set_source_rgba(*B["school_sh"])
            ctx.fill()
    bg.shape(np.array([(sx1 - 92, 420), (sx1 - 40, 420), (sx1 - 40, 462), (sx1 - 92, 462)]), B["sdoor"])
    bg.lines.append((np.array([(sx1 - 66, 420), (sx1 - 66, 462)]), False))
    bg.shape(np.array([(sx1 - 104, 412), (sx1 - 28, 412), (sx1 - 36, 420), (sx1 - 96, 420)]), B["slab"])
    # a tiny bell cupola on the roof
    bg.shape(np.array([(1010, 296), (1010, 268), (1046, 268), (1046, 296)]), B["slab"])
    bg.shape(np.array([(1004, 270), (1028, 246), (1052, 270)]), B["roof"])
    bg.shape(ell(1028, 284, 7, 8), B["sframe"], outline=False)
    bg.flush()
    lollipop_tree(bg, 1225, 466, 175, 70, 37)
    bg.flush()

    # ---- lawn, hedges, far sidewalk --------------------------------------------------------------------------
    lawn = np.array([(-10, 482), (-10, 460), (W + 10, 460), (W + 10, 482)])
    bg.shape(lawn, B["lawn"], outline=False)
    bg.sponge(lawn, B["lawn_d"], 260, 2, 5, .8, seed=41)
    bg.lines.append((np.array([(-10, 461), (W + 10, 461)]), False))
    for hx, hw in ((470, 120), (620, 90), (760, 70)):
        hed = scallop(cr([(hx - hw / 2, 472), (hx - hw / 2, 452), (hx, 440), (hx + hw / 2, 452), (hx + hw / 2, 472)]), lam=16, amp=4,
                      ranges=[(0.08, .92, 1)], seed=hx)
        bg.shape(hed, B["hedge"])
        bg.sponge(hed, B["hedge_d"], 40, 2, 5, .9, seed=hx + 1)
    bg.shape(np.array([(-10, 498), (-10, 482), (W + 10, 482), (W + 10, 498)]), B["walk_far"])
    bg.shape(np.array([(-10, 505), (-10, 498), (W + 10, 498), (W + 10, 505)]), B["curb_far"], outline=False)
    bg.flush()

    # ---- the road with the zebra crossing ---------------------------------------------------------------------
    road = np.array([(-10, 590), (-10, 505), (W + 10, 505), (W + 10, 590)])
    bg.shape(road, B["road"], outline=False)
    bg.sponge(road, B["road_d"], 500, 2, 6, .55, seed=51)
    zx0, zx1 = 880, 1130
    ys = np.linspace(510, 586, 6)
    for k in range(5):                                        # stripes: wider and thicker toward us
        ya, yb = ys[k] + 3, ys[k] + (ys[k + 1] - ys[k]) * .55 + 3
        spread = (ya - 505) * .45
        bg.shape(np.array([(zx0 - spread, yb), (zx0 - spread + 14, ya), (zx1 + spread, ya), (zx1 + spread - 14, yb)]), B["zebra"])
    # near curb + sidewalk
    bg.shape(np.array([(-10, 600), (-10, 590), (W + 10, 590), (W + 10, 600)]), B["curb_top"])
    bg.shape(np.array([(-10, 612), (-10, 600), (W + 10, 600), (W + 10, 612)]), B["curb"], outline=False)
    walk = np.array([(-10, H + 10), (-10, 612), (W + 10, 612), (W + 10, H + 10)])
    bg.shape(walk, B["walk"], outline=False)
    for xx in (-120, 190, 860, 1170):                           # expansion joints, splayed a little for depth
        bg.lines.append((np.array([(xx + 60, 614), (xx + (xx - 520) * .18 + 60, H + 10)]), False))
    bg.lines.append((np.array([(-10, 612), (W + 10, 612)]), False))

    bg.flush()
    a = np.ndarray((H * ss, W * ss, 4), np.uint8, surf.get_data())[:, :, :3][:, :, ::-1].astype(np.float32) / 255
    img = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
    return gouache(img)


def gouache(img, seed=5):
    """Painted-card texture: slow mottling of the flat paint, fine paper tooth and a faint horizontal dry-brush drag."""
    rng = np.random.default_rng(seed)
    m = cv2.resize(rng.normal(0, 1, (H // 24, W // 24)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    f = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), .8)
    d = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigmaX=14, sigmaY=.7)
    m2 = cv2.resize(rng.normal(0, 1, (H // 6, W // 6)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    k = 1 + .012 * m + .008 * m2 + .016 * f + .022 * d
    return np.clip(img * k[..., None], 0, 1)


_BG = None


def get_bg():
    global _BG
    if _BG is None:
        _BG = background()
    return _BG


# =============================================================================================== the character
MASS = [(0, -124), (36, -130), (68, -150), (106, -194), (150, -258), (186, -328), (196, -398), (176, -462), (120, -518), (40, -560),
        (-44, -580), (-126, -574), (-198, -546), (-254, -494), (-284, -426), (-276, -352), (-230, -282), (-160, -212), (-94, -160),
        (-42, -131)]
ANCHOR = (0.0, -125.0)
SASH = [(-320, -610), (-30, -372), (250, -150)]
BADGE = (-92, -424)
HEAD0 = (152.0, -334.0)
HEAD_TILT = -.045
POSE0 = dict(x=0.0, y=0.0, sx=1.0, sy=1.0, lean=0.0, hx=0.0, hy=0.0, hr=0.0, cap=0.0, capr=0.0, capx=0.0, sign=0.0, smear=0.0,
             jaw=0.0, mouth="smile", lid=.24, px=-2.5, py=0.0, brow=0.0, blink=0.0)


def body_matrix(P):
    m = cairo.Matrix()
    m = m.multiply(cairo.Matrix(x0=-ANCHOR[0], y0=-ANCHOR[1]))
    sc = cairo.Matrix(xx=P["sx"], yy=P["sy"])
    rot = cairo.Matrix.init_rotate(P["lean"])
    m = m.multiply(sc).multiply(rot).multiply(cairo.Matrix(x0=ANCHOR[0], y0=ANCHOR[1]))
    return m


def xf(m, p):
    return np.array(m.transform_point(float(p[0]), float(p[1])))


_MASS_OUTLINE = None


def hem(x):
    """Where the shaggy wool mantle ends and the short-haired waist begins."""
    return -226 + (np.asarray(x) + 270) * .11


def mass_outline():
    global _MASS_OUTLINE
    if _MASS_OUTLINE is None:
        base = cr(MASS, per=24)
        _MASS_OUTLINE = scallop(base, lam=42, amp=11, power=.5, seed=7, skew=1.25,
                                amp_fn=lambda Q: np.clip((hem(Q[:, 0]) - Q[:, 1]) / 26, 0, 1))
        xs = np.linspace(-330, 270, 60)
        hemP = np.vstack([np.stack([xs, hem(xs)], 1), [(270, -720), (-330, -720)]])
        global _MANTLE
        _MANTLE = scallop(hemP, lam=30, amp=13, power=.5, seed=12, skew=1.3, amp_fn=lambda Q: (Q[:, 1] > -650).astype(float))
    return _MASS_OUTLINE


_MANTLE = None


def sign_geom(P):
    """Hand grip, pole tilt and disc position for sign raise s (0 planted, 1 raised; extrapolates for overshoot)."""
    s = P["sign"]
    hand = np.array([lerp(292, 338, s), lerp(-250, -318, s)])
    tilt = lerp(0.0, .07, s)
    up = np.array([math.sin(tilt), -math.cos(tilt)])
    disc = hand + up * 262
    bottom = hand - up * lerp(252, 252, s)
    return hand, disc, bottom, tilt


def draw_sign(ctx, P):
    hand, disc, bottom, tilt = sign_geom(P)
    up = (disc - hand) / np.linalg.norm(disc - hand)
    side = np.array([-up[1], up[0]])
    sm = P["smear"]
    pole = np.array([bottom + side * 6.5, disc + side * 6.5, disc - side * 6.5, bottom - side * 6.5])
    fs(ctx, pole, C["pole"])
    if sm > 0:                                   # smear: the disc stretched back along its path
        trail = disc - up * 150 * sm
        top = ell(disc[0], disc[1], 60, 60, 0, 48)[24:]          # upper half of the disc (angles pi..2pi)
        bot = ell(trail[0], trail[1], 40, 40, 0, 48)[:25]         # lower half of the trail end (0..pi)
        S = np.vstack([top, bot])
        fs(ctx, S, C["red"])
    fs(ctx, ell(disc[0], disc[1], 60, 60, tilt), C["white"])
    fs(ctx, ell(disc[0], disc[1], 48, 48, tilt), C["red"], 2.4)
    # a raised-palm "halt" pictogram, no letters
    ctx.save()
    ctx.translate(*disc)
    ctx.rotate(tilt)
    palm = cr([(-15, 22), (-17, 0), (-25, -8), (-21, -13), (-12, -6), (-13, -26), (-7, -30), (-3, -10), (-2, -33), (4, -34), (5, -10),
               (9, -30), (15, -28), (13, -6), (18, -22), (23, -19), (19, 2), (14, 22)], per=6, tension=.6)
    fs(ctx, palm, C["white"], 0)
    ctx.restore()
    return hand


def fist(ctx, c, ang, scale=1.0, col=None):
    """A chunky cartoon fist seen from the knuckle side: mitten with three knuckle bumps."""
    ctx.save()
    ctx.translate(*c)
    ctx.rotate(ang)
    ctx.scale(scale, scale)
    F = cr([(-20, -16), (-4, -22), (14, -20), (24, -10), (25, 8), (16, 20), (-4, 22), (-20, 14), (-25, 0)], per=10)
    fs(ctx, F, col or C["hand"])
    for y in (-8, 2, 12):                               # finger creases
        taper_line(ctx, [(8, y - 2), (18, y), (23, y + 1)], 2.6, 2.0)
    taper_line(ctx, [(-14, -10), (-4, -4), (2, 6)], 2.8, 2.0)   # thumb line
    ctx.restore()


def hip_fist(ctx, c, scale=1.0):
    ctx.save()
    ctx.translate(*c)
    ctx.scale(scale, scale)
    F = cr([(-22, -14), (-6, -24), (14, -22), (24, -8), (20, 10), (4, 20), (-16, 18), (-26, 4)], per=10)
    fs(ctx, F, C["hand"])
    for x in (-10, 0, 10):
        taper_line(ctx, [(x - 3, 18), (x, 8), (x + 2, 2)], 2.6, 1.8)
    ctx.restore()


def draw_legs(ctx, P):
    # pelvis / trousers
    for sx_, ax, ay in ((-1, -36, -16), (1, 32, -14)):
        leg = tube([(sx_ * 24, -104), (sx_ * 25, -62), (ax, ay)], 27, 17)
        fs(ctx, leg, C["trouser"])
    pel = cr([(-46, -150), (46, -152), (58, -112), (44, -88), (0, -84), (-44, -86), (-58, -110)], per=12)
    fs(ctx, pel, C["trouser"])
    taper_line(ctx, [(0, -112), (1, -98), (2, -88)], 2.6, 1.8)
    # tiny shoes
    for cx, rot in ((-42, -.12), (42, .14)):
        sh = cr(ell(cx, -9, 24, 11, rot, 10) + np.array([[0, 0]] * 10), per=6)
        fs(ctx, sh, C["shoe"])
        fs(ctx, ell(cx - 6 * np.sign(cx), -14, 8, 3, rot, 16), hexc("#4a4352"), 0)


def wool_curl(ctx, x, y, s=1.0, flip=1):
    taper_line(ctx, [(x - 10 * s * flip, y - 8 * s), (x - 13 * s * flip, y + 2 * s), (x - 4 * s * flip, y + 9 * s), (x + 7 * s * flip, y + 6 * s)], 3.0, 2.0)


def draw_head(ctx, P, c):
    ctx.save()
    ctx.translate(*c)
    ctx.rotate(P["hr"] + HEAD_TILT)
    jaw = P["jaw"]
    # beard (behind the face)
    beard = scallop(cr([(-50, 34), (-14, 52), (24, 58), (58, 46), (60, 80), (30, 106 + jaw * .6), (-14, 108 + jaw * .5), (-48, 84)]),
                    lam=20, amp=6, ranges=[(.25, .98, 1)], seed=3)
    fs(ctx, beard, C["wool"])
    # horns and ears
    for pts, w in (([(46, -50), (72, -60), (84, -78), (83, -98)], 22), ([(-52, -44), (-82, -50), (-100, -68), (-102, -92)], 24)):
        fs(ctx, tube(pts, w, 6), C["horn"])
    fs(ctx, ell(64, -30, 15, 8, -.55, 32), C["face"])
    fs(ctx, ell(-70, -20, 19, 9, .35, 32), C["face"])
    taper_line(ctx, [(-80, -22), (-70, -20), (-60, -18)], 2.2, 1.6)
    # head
    headP = cr([(-66, -28), (-58, -56), (-20, -72), (30, -70), (62, -50), (72, -12), (70, 22), (50, 48 + jaw * .4), (0, 56 + jaw * .5),
                (-44, 44), (-66, 12)], per=14)
    fs(ctx, headP, C["face"])
    # eyes: two touching ovals (far one first)
    lid = max(P["lid"], P["blink"])
    for ex, ey, rx, ry, pdx in ((16, -14, 15, 21, 5.5), (-18, -12, 17, 22, 5.5)):
        E = ell(ex, ey, rx, ry, 0, 64)
        fs(ctx, E, C["eye"], 3.0)
        ctx.save()
        path(ctx, E)
        ctx.clip()
        if lid < .95:
            ell_p = ell(ex + pdx + P["px"], ey + 3 + P["py"], rx * .42, ry * .42, 0, 32)
            fs(ctx, ell_p, INK, 0)
        top = ey - ry
        ly = top + 2 * ry * lid
        ctx.rectangle(ex - rx - 4, top - 6, 2 * rx + 8, ly - top + 6)
        ctx.set_source_rgba(*C["face"])
        ctx.fill()
        if lid > .05:
            line(ctx, np.array([(ex - rx - 2, ly + 1.5), (ex, ly - .5), (ex + rx + 2, ly + 1.5)]), 3.0)
        ctx.restore()
        path(ctx, E)
        ctx.set_source_rgba(*INK)
        ctx.set_line_width(3.0)
        ctx.stroke()
    # fringe of wool over the brow
    b = P["brow"]
    fr = cr([(-72, -26 + b), (-70, -60), (-34, -82), (16, -86), (58, -70), (70, -40 + b), (44, -36 + b), (14, -34 + b), (-14, -30 + b),
             (-44, -26 + b)], per=14)
    fr = scallop(fr, lam=18, amp=5.5, power=.6, ranges=[(.0, 1, 1)], seed=5)
    fs(ctx, fr, C["wool"])
    # muzzle, nose, mouth
    mz = cr([(-14, 18), (0, 2), (30, -4), (66, 2), (84, 18), (82, 40 + jaw * .5), (60, 56 + jaw * .7), (24, 60 + jaw * .7), (-6, 50 + jaw * .5)], per=8)
    fs(ctx, mz, C["muzzle"])
    nose = cr([(14, 10), (26, 0), (46, -2), (68, 2), (78, 12), (70, 22), (46, 20), (24, 22)], per=8)
    fs(ctx, nose, C["nose"], 3.0)
    fs(ctx, ell(24, 13, 5.2, 4.2, .5, 20), INK, 0)
    fs(ctx, ell(70, 11, 4.4, 3.6, -.5, 20), INK, 0)
    fs(ctx, ell(44, 5, 9, 2.4, -.04, 20), hexc("#a5675a"), 0)
    m = P["mouth"]
    if m in ("smile", "mm"):
        if m == "smile":
            taper_line(ctx, [(4, 40), (28, 47), (52, 45), (72, 34)], 3.4, 2.6)
            taper_line(ctx, [(68, 29), (74, 34), (74, 41)], 2.6, 1.8)
        else:
            taper_line(ctx, [(6, 43), (30, 46), (52, 45), (70, 40)], 3.2, 2.6)
    else:
        o = max(jaw, 4)
        if m == "oh":
            M = ell(38, 44 + o * .45, 12 + o * .1, 6 + o * .5, 0, 40)
        elif m == "ee":
            M = cr([(6, 38), (38, 42), (70, 34), (64, 42 + o * .6), (36, 48 + o * .7), (12, 44 + o * .5)], per=8)
        else:  # "ah"
            M = cr([(8, 38), (38, 42), (70, 33), (62, 44 + o * .8), (38, 52 + o), (16, 46 + o * .8)], per=8)
        fs(ctx, M, C["mouth"], 3.2)
        ctx.save()
        path(ctx, M)
        ctx.clip()
        fs(ctx, ell(38, 56 + o * 1.0, 24, 10 + o * .2, 0, 32), C["tongue"], 0)
        if m == "ee":
            ctx.rectangle(0, 30, 70, 8 + o * .15)
            ctx.set_source_rgba(*C["eye"])
            ctx.fill()
        ctx.restore()
        path(ctx, M)
        ctx.set_source_rgba(*INK)
        ctx.set_line_width(3.2)
        ctx.stroke()
    # the too-small cap, perched between the horns
    ctx.save()
    ctx.translate(P["capx"], P["cap"])
    ctx.rotate(P["capr"])
    crown = cr([(-34, -74), (-32, -94), (-12, -106), (16, -106), (34, -94), (38, -74)], per=10)
    fs(ctx, crown, C["cap"])
    band = np.array([(-35, -86), (38, -86), (39, -72), (-35, -72)])
    fs(ctx, cr(band, per=3, tension=.3), C["cap2"])
    visor = cr([(26, -78), (54, -80), (66, -72), (48, -68), (24, -71)], per=8)
    fs(ctx, visor, C["cap2"])
    star = np.array([(4 + (8 if k % 2 == 0 else 3.6) * math.cos(-math.pi / 2 + k * math.pi / 5),
                      -94 + (8 if k % 2 == 0 else 3.6) * math.sin(-math.pi / 2 + k * math.pi / 5)) for k in range(10)])
    fs(ctx, star, C["badge"], 2.2)
    ctx.restore()
    ctx.restore()


def draw_character(ctx, P):
    ctx.save()
    ctx.translate(X0 + P["x"], Y0 + P["y"])
    M = body_matrix(P)
    # far arm + sign (behind the body)
    hand, disc, bottom, tilt = sign_geom(P)
    sh_far = xf(M, (150, -300))
    elbow_far = (sh_far + hand) / 2 + np.array([10, 40 - 20 * P["sign"]])
    fs(ctx, tube([sh_far, elbow_far, hand + np.array([-12, 4])], 50, 40), C["wool"])
    draw_sign(ctx, P)
    fist(ctx, hand + np.array([2, 0]), .05 + tilt)
    # near arm upper part (behind the mass)
    sh_near = xf(M, (-224, -350))
    elbow = xf(M, (-312, -270)) + np.array([P.get("ex", 0.0), P.get("ey", 0.0)])
    fs(ctx, tube([sh_near, elbow], 66, 54), C["wool"])
    # legs + trousers
    draw_legs(ctx, P)
    # the mass
    ctx.save()
    ctx.transform(M)
    O = mass_outline()
    fs(ctx, O, C["face"])
    ctx.save()
    path(ctx, O)
    ctx.clip()
    fs(ctx, _MANTLE, C["wool"])
    fs(ctx, tube(SASH, 66, 66), C["sash"])
    for (x, y, s, f) in ((-196, -410, 1.1, 1), (-214, -330, 1.0, 1), (-20, -505, 1.0, -1), (70, -470, .9, -1), (-120, -500, .9, 1)):
        wool_curl(ctx, x, y, s, f)
    ctx.restore()
    # badge on the sash
    bx, by = BADGE
    bd = cr([(bx - 16, by - 14), (bx + 14, by - 20), (bx + 16, by + 10), (bx, by + 26), (bx - 16, by + 10)], per=6, tension=.6)
    fs(ctx, bd, C["badge"], 3.0)
    fs(ctx, ell(bx, by, 6, 6), hexc("#e8a92a"), 0)
    ctx.restore()
    # belt under the waist
    belt = cr([(-50, -128), (0, -122), (50, -130), (51, -116), (0, -108), (-51, -114)], per=8, tension=.6)
    fs(ctx, belt, C["cap2"], 3.0)
    fs(ctx, cr([(-4, -126), (12, -127), (13, -110), (-3, -109)], per=3, tension=.3), C["badge"], 2.4)
    # near forearm + fist planted on his "waist"
    hipc = xf(M, (-160, -196))
    fs(ctx, tube([elbow, (elbow + hipc) / 2 + np.array([-4, 6]), hipc + np.array([-10, -6])], 52, 40), C["wool"])
    taper_line(ctx, [elbow + np.array([10, -16]), elbow + np.array([20, -4]), elbow + np.array([16, 10])], 3.0, 2.0)
    hip_fist(ctx, hipc)
    # head
    hc = xf(M, HEAD0) + np.array([P["hx"], P["hy"]])
    draw_head(ctx, P, hc)
    ctx.restore()


# =============================================================================================== compositing
def cel_layer(P, ss=2, f=None):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * ss, H * ss)
    ctx = cairo.Context(surf)
    ctx.scale(ss, ss)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    draw_character(ctx, P)
    surf.flush()
    mask = (np.ndarray((H * ss, W * ss, 4), np.uint8, surf.get_data())[:, :, 3] > 127).astype(np.uint8)
    if f is not None:
        accents(ctx, f, P)
        surf.flush()
    a = np.ndarray((H * ss, W * ss, 4), np.uint8, surf.get_data()).astype(np.float32) / 255
    rgb, al = a[:, :, 2::-1], a[:, :, 3]
    # the thick, even outer contour: grow the cel's silhouette and fill the ring with ink
    d = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    ring = np.clip(OUTLINE * ss + .5 - d, 0, 1) * (1 - al)
    rgb = rgb + np.array(INK[:3], np.float32) * ring[..., None]
    al = al + ring
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    al = cv2.resize(al, (W, H), interpolation=cv2.INTER_AREA)
    return rgb, al


def ground_shadow(img, P):
    yy, xx = np.ogrid[0:H, 0:W]
    cx, cy = X0 + P["x"] + 2, Y0 + 1
    lift = max(0.0, -P["y"]) / 80
    rx, ry = 112 * (1 - .3 * lift) * (.85 + .15 * P["sx"]), 13 * (1 - .3 * lift)
    e = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    m = np.clip((1 - e) * 6, 0, 1)[..., None]
    tint = np.array(B["shadow"][:3], np.float32) / np.array(B["walk"][:3], np.float32)
    return img * (1 - m + m * tint)


def compose(P, ss=2):
    bg = ground_shadow(get_bg().copy(), P)
    rgb, al = cel_layer(P, ss)
    return np.clip(bg * (1 - al[..., None]) + rgb, 0, 1)


def finish(img, i=0, grain=.008):
    """A light film/broadcast transfer: tiny softening and a whisper of grain, nothing that muddies the cel."""
    img = cv2.GaussianBlur(img, (0, 0), .35)
    rng = np.random.default_rng(1000 + i)
    g = cv2.resize(rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32), (W, H))
    return np.clip(img + g[..., None] * grain, 0, 1)


def to8(img):
    return (np.clip(img, 0, 1) * 255 + .5).astype(np.uint8)


def key_image(ss=3):
    img = finish(compose(dict(POSE0), ss), 0)
    return to8(img[8:712])


# =============================================================================================== animation
# syllables of the gibberish "line": (start frame, length, mouth, jaw, pitch in semitones) -- shared with sound.py
SYL = [(28, 6, "ah", 15, 0), (36, 4, "ee", 8, 2), (40, 4, "ah", 11, -1), (46, 4, "oh", 10, 1), (50, 4, "ah", 13, 3),
       (54, 4, "ee", 8, 1), (58, 4, "oh", 10, 0), (62, 8, "ah", 17, 5)]
EV = dict(antic=16, raise_=22, pop=72, land=82, lower=88, thunk=94)   # event frames, shared with sound.py

KEYS = [  # (frame, overrides); numeric params interpolate with smoothstep between keys
    (0, {}), (16, {}),
    (20, dict(sy=.91, sx=1.05, hy=10, hr=.06, lid=.62, sign=0.0, brow=5, ex=8, ey=8)),
    (22, dict(sy=1.1, sx=.94, sign=.78, smear=1, hy=-12, hr=-.05, lid=0, brow=-8, y=-12, ex=-6, ey=-10)),
    (24, dict(sy=1.06, sx=1.04, sign=1.12, hy=-8, hr=-.07, lid=.05, y=-3, brow=-6)),
    (27, dict(sy=1.03, sx=1.05, sign=1.0, hy=-4, hr=-.04, lid=.12, px=-3, brow=-3)),
    (62, dict(sy=1.06, sx=1.09, sign=1.0, hy=-6, hr=-.05, lid=.12, px=-3, brow=-4)),
    (70, dict(sy=1.075, sx=1.16, sign=1.02, hy=-10, hr=-.09, lid=.55, px=-3, brow=-6)),
    (72, dict(sy=1.08, sx=1.17, sign=1.02, hy=-11, hr=-.09, lid=.55, cap=-30, capr=-.22, capx=-4, brow=-8)),
    (76, dict(sy=1.06, sx=1.13, sign=1.0, hy=-9, hr=-.07, lid=.4, cap=-78, capr=-.5, capx=-10, py=-5, brow=-10)),
    (80, dict(sy=1.04, sx=1.07, sign=1.0, hy=-6, hr=-.05, lid=.3, cap=-34, capr=-.2, capx=-4, py=-6, brow=-6)),
    (82, dict(sy=1.0, sx=1.06, sign=1.0, hy=4, hr=.02, lid=.85, cap=0, capr=0, capx=0, brow=4)),
    (86, dict(sy=1.03, sx=1.05, sign=1.0, hy=-2, hr=-.03, lid=.3, brow=-2)),
    (88, dict(sy=1.03, sx=1.04, sign=1.0, hy=-2, hr=-.03, lid=.3)),
    (92, dict(sy=1.0, sx=1.0, sign=.05, hy=0, hr=-.01, lid=.25)),
    (94, dict(sy=.95, sx=1.04, sign=0, hy=5, hr=.02, lid=.3)),
    (98, dict(sy=1.015, sx=.995, sign=0, hy=-2, lid=.24)),
    (102, {}), (N - 1, {}),
]
NUM = [k for k, v in POSE0.items() if isinstance(v, float)] + ["ex", "ey"]


def _full(d):
    p = dict(POSE0, ex=0.0, ey=0.0)
    p.update(d)
    return p


def pose_at(f):
    """Pose for frame f.  The shot is timed on twos except the snap of the raise (frames 18-27), which is on ones."""
    if not 18 <= f <= 27:
        f = f - f % 2
    for (fa, a), (fb, b) in zip(KEYS, KEYS[1:]):
        if fa <= f <= fb:
            break
    A, Bp = _full(a), _full(b)
    u = (f - fa) / max(fb - fa, 1)
    u = smooth(u)
    P = dict(A)
    for k in NUM:
        P[k] = lerp(A[k], Bp[k], u)
    P["smear"] = 1.0 if f in (21, 22) else 0.0
    if f == 21:
        P["sign"] = .45
    # mouth
    P["mouth"], P["jaw"] = "smile", 0.0
    if 18 <= f < 22:
        P["mouth"] = "mm"
    elif 22 <= f < 28:
        P["mouth"], P["jaw"] = "oh", 8.0
    for (s0, n, m, jaw, _) in SYL:
        if s0 <= f < s0 + n:
            P["mouth"], P["jaw"] = m, float(jaw)
            k = (f - s0) / n
            P["jaw"] *= 1.0 if k < .7 else .6
            P["hr"] += -.025 * math.sin(math.pi * k)
            P["hy"] += -3 * math.sin(math.pi * k)
            P["sy"] += .012 * math.sin(math.pi * k)
            P["sx"] += .008 * math.sin(math.pi * k)
            break
        if s0 + n <= f < s0 + n + 2 and f < 70:
            P["mouth"] = "mm"
    if 28 <= f < 70 and P["mouth"] == "smile":
        P["mouth"] = "mm"
    if 76 <= f < 82:
        P["mouth"] = "oh"
        P["jaw"] = 6.0
    # blinks
    if f in (10, 11, 108, 109):
        P["blink"] = 1.0
    if f in (12, 13, 110, 111):
        P["blink"] = .6
    return P


def accents(ctx, f, P):
    """Emanata: little accent lines for the snap, the cap pop and the thunk (cel ink, no text)."""
    def burst(cx, cy, r0, r1, angs, lw=3.2):
        for a in angs:
            line(ctx, np.array([(cx + r0 * math.cos(a), cy + r0 * math.sin(a)), (cx + r1 * math.cos(a), cy + r1 * math.sin(a))]), lw)
    ctx.save()
    ctx.translate(X0 + P["x"], Y0 + P["y"])
    if P["smear"] > 0:
        hand, disc, _, _ = sign_geom(P)
        for dx, y0, L in ((-80, 30, 150), (80, 30, 150), (-102, 70, 90)):
            line(ctx, np.array([(disc[0] + dx, disc[1] + y0), (disc[0] + dx, disc[1] + y0 + L)]), 2.8)
    if 23 <= f <= 27:
        hand, disc, _, _ = sign_geom(P)
        k = (f - 23) / 4
        burst(disc[0], disc[1], 72 + 14 * k, 92 + 18 * k, [-2.6, -1.9, -1.2, -.5, .2])
    if 72 <= f <= 75:
        M = body_matrix(P)
        hc = xf(M, HEAD0) + np.array([P["hx"], P["hy"]])
        burst(hc[0] + 2, hc[1] - 96, 26, 46, [-2.5, -2.0, -1.1, -.6])
    if 82 <= f <= 85:
        M = body_matrix(P)
        hc = xf(M, HEAD0) + np.array([P["hx"], P["hy"]])
        burst(hc[0] + 2, hc[1] - 96, 50, 66, [-2.8, -2.4, -.7, -.3], 2.8)
    if 94 <= f <= 97:
        _, _, bottom, _ = sign_geom(P)
        burst(bottom[0], bottom[1] - 2, 18, 34, [-2.9, -2.5, -.6, -.2], 2.8)
    ctx.restore()


def cel_frame(f, ss=2):
    P = pose_at(f)
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * ss, H * ss)
    ctx = cairo.Context(surf)
    ctx.scale(ss, ss)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    draw_character(ctx, P)
    return surf, ctx, P


def frame(i):
    P = pose_at(i)
    bg = ground_shadow(get_bg().copy(), P)
    rgb, al = cel_layer(P, 2, f=i)
    img = np.clip(bg * (1 - al[..., None]) + rgb, 0, 1)
    return to8(finish(img, i))
