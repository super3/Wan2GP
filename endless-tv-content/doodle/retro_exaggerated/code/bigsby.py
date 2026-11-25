"""Shared drawing code for the short "The Littlest Crossing" (copied from the bertram.py prototype and generalised).
Bertram Bigsby, crossing guard: a top-heavy bison in a late-90s TV-cartoon style (thick even outer contour, flat cel colour,
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
LS = [1.0]        # line-width scale for the current camera/character scale (keeps screen line weight steady)


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
        ctx.set_line_width(lw * LS[0])
        ctx.stroke()
    else:
        ctx.fill()


def line(ctx, P, lw=LW, col=INK):
    path(ctx, P, False)
    ctx.set_source_rgba(*col)
    ctx.set_line_width(lw * LS[0])
    ctx.stroke()


def taper_line(ctx, pts, w0, w1, col=INK, per=14):
    """A brush line that thins toward its ends (interior detail lines)."""
    C_ = cr(pts, closed=False, per=per)
    C_, _ = resample(C_, 1.5, closed=False)
    d = np.gradient(C_, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    t = np.linspace(0, 1, len(C_))
    w = LS[0] * (w0 + (w1 - w0) * t) * np.sin(np.pi * np.clip(t * .85 + .075, 0, 1)) ** .5
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
            Q = Q + np.asarray(offset) * LS[0] + rng.normal(0, .25 * LS[0], Q.shape)
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
                ctx.set_line_width(lw * LS[0] * rng.uniform(.85, 1.1))
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
        ctx.set_line_width(2.2 * LS[0])
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


def bg_corner(bg):
    ctx = bg.ctx

    # sky: a soft airbrushed card, deeper at the top
    g = cairo.LinearGradient(0, 0, 0, 470)
    g.add_color_stop_rgba(0, *B["sky0"])
    g.add_color_stop_rgba(.55, *B["sky1"])
    g.add_color_stop_rgba(1, *B["sky2"])
    ctx.rectangle(-400, -300, W + 800, 780)
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
    walk = np.array([(-10, H + 200), (-10, 612), (W + 10, 612), (W + 10, H + 200)])
    bg.shape(walk, B["walk"], outline=False)
    for xx in (-120, 190, 860, 1170):                           # expansion joints, splayed a little for depth
        bg.lines.append((np.array([(xx + 60, 614), (xx + (xx - 520) * .18 + 60, H + 10)]), False))
    bg.lines.append((np.array([(-10, 612), (W + 10, 612)]), False))

    bg.flush()


def gouache(img, seed=5):
    """Painted-card texture: slow mottling of the flat paint, fine paper tooth and a faint horizontal dry-brush drag."""
    rng = np.random.default_rng(seed)
    m = cv2.resize(rng.normal(0, 1, (H // 24, W // 24)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    f = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), .8)
    d = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigmaX=14, sigmaY=.7)
    m2 = cv2.resize(rng.normal(0, 1, (H // 6, W // 6)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    k = 1 + .012 * m + .008 * m2 + .016 * f + .022 * d
    return np.clip(img * k[..., None], 0, 1)






# =============================================================================================== perspective street (the crossing)
VPX, HZ = 640.0, 368.0
L_CURB, R_CURB = 450.0, 830.0     # curb lines where they cross the bottom of the frame
WALK_Y = 665.0


def on(x0, y):
    """x on the line through the vanishing point that passes x0 at screen y=720."""
    return VPX + (x0 - VPX) * (y - HZ) / (720 - HZ)


def depth(y):
    return (y - HZ) / (720 - HZ)


def facade(x0b, h0, y_near, y_far):
    """A vertical wall parallel to the street, as a screen quad: near-base, near-top, far-top, far-base."""
    xn, xf = on(x0b, y_near), on(x0b, y_far)
    return np.array([(xn, y_near), (xn, y_near - h0 * depth(y_near)), (xf, y_far - h0 * depth(y_far)), (xf, y_far)])


def quad_pt(Q, u, v):
    """Bilinear point in a facade quad: u along the street (0 near, 1 far), v up (0 base, 1 top)."""
    base = Q[0] + (Q[3] - Q[0]) * u
    top = Q[1] + (Q[2] - Q[1]) * u
    return base + (top - base) * v


def quad_sub(Q, u0, u1, v0, v1):
    return np.array([quad_pt(Q, u0, v0), quad_pt(Q, u0, v1), quad_pt(Q, u1, v1), quad_pt(Q, u1, v0)])


def bg_cross(bg):
    ctx = bg.ctx
    g = cairo.LinearGradient(0, -300, 0, HZ + 20)
    g.add_color_stop_rgba(0, *B["sky0"])
    g.add_color_stop_rgba(.6, *B["sky1"])
    g.add_color_stop_rgba(1, *B["sky2"])
    ctx.rectangle(-1200, -600, 3700, HZ + 640)
    ctx.set_source(g)
    ctx.fill()
    cloud(bg, 250, 110, 240, 66, 13)
    cloud(bg, 1010, 70, 210, 56, 15)
    cloud(bg, 620, 190, 120, 32, 17)
    bg.flush()
    # far hills and a little water tower at the end of the street
    hills = cr([(-1200, HZ + 30), (-1200, HZ - 20), (-300, HZ - 46), (200, HZ - 24), (520, HZ - 52), (860, HZ - 30), (1300, HZ - 58),
                (2500, HZ - 30), (2500, HZ + 30)])
    bg.shape(hills, B["hill0"])
    bg.sponge(hills, B["hill1"], 260, 2, 6, .6, seed=61, region=(-300, HZ - 60, 1600, HZ + 10))
    bg.flush()
    tw = 420, HZ - 50
    for dx in (-10, 10):
        bg.shape(np.array([(tw[0] + dx * 1.6, HZ - 2), (tw[0] + dx * .6, tw[1]), (tw[0] + dx * .6 + 2, tw[1]), (tw[0] + dx * 1.6 + 2, HZ - 2)]), B["slab2"])
    bg.shape(ell(tw[0] + 1, tw[1] - 10, 20, 15), B["win2"])
    bg.shape(np.array([(tw[0] - 20, tw[1] - 22), (tw[0] + 1, tw[1] - 40), (tw[0] + 22, tw[1] - 22)]), B["roof"])
    bg.flush()
    # ground: lawns either side
    bg.shape(np.array([(-1200, HZ), (2500, HZ), (2500, 1300), (-1200, 1300)]), B["lawn"], outline=False)
    bg.sponge(np.array([(-1200, HZ), (2500, HZ), (2500, 900), (-1200, 900)]), B["lawn_d"], 900, 2, 7, .7, seed=62)
    # distant little houses near the vanishing point
    for x0b, h, yn, yf, col in ((-260, 260, 430, 395, B["house"]), (1560, 300, 432, 398, B["door"]), (-700, 300, 400, 382, B["win"]),
                                (2000, 280, 402, 384, B["house"])):
        Q = facade(x0b, h, yn, yf)
        bg.shape(Q, col)
        rf = np.array([Q[1] + (0, 2), Q[2] + (0, 2), (Q[2][0], Q[2][1] - 10 * depth(yf)), (Q[1][0], Q[1][1] - 26 * depth(yn))])
        bg.shape(rf, B["roof"])
    bg.flush()
    # sidewalks, curbs and the road
    Y1 = 1300
    road = np.array([(on(L_CURB, Y1), Y1), (VPX - 2, HZ), (VPX + 2, HZ), (on(R_CURB, Y1), Y1)])
    for xa, xb in ((L_CURB - 420, L_CURB), (R_CURB, R_CURB + 420)):
        side = np.array([(on(xa, Y1), Y1), (VPX, HZ), (on(xb, Y1), Y1)])
        bg.shape(side, B["walk"])
        for k in range(1, 14):                             # paving joints, spaced by depth
            y = HZ + (720 - HZ) * (k / 9) ** 2
            if y < HZ + 4:
                continue
            bg.lines.append((np.array([(on(xa, y), y), (on(xb, y), y)]), False))
    bg.shape(road, B["road"])
    bg.sponge(road, B["road_d"], 900, 2, 6, .5, seed=63, region=(200, HZ, 1100, 760))
    for xa, xb in ((L_CURB - 6, L_CURB + 6), (R_CURB - 6, R_CURB + 6)):     # curb faces
        bg.shape(np.array([(on(xa, Y1), Y1), (VPX, HZ), (on(xb, Y1), Y1)]), B["curb"], outline=False)
    # centre line dashes receding
    for k in range(9):
        ya, yb = HZ + (720 - HZ) * (k / 8.5) ** 1.8, HZ + (720 - HZ) * ((k + .45) / 8.5) ** 1.8
        if ya > 600:
            break
        w = 4 * depth(ya)
        bg.shape(np.array([(on(640 - w, ya), ya), (on(640 + w, ya), ya), (on(640 + w * 1.6, yb), yb), (on(640 - w * 1.6, yb), yb)]), B["zebra"],
                 outline=False)
    # the zebra crossing in the foreground: stripes parallel to the traffic, converging on the vanishing point
    ya, yb = 616, 716
    xs = np.linspace(L_CURB + 14, R_CURB - 8, 7)
    for k in range(6):
        a, b = xs[k] + 8, xs[k] + (xs[1] - xs[0]) * .6
        bg.shape(np.array([(on(a, ya), ya), (on(b, ya), ya), (on(b, yb), yb), (on(a, yb), yb)]), B["zebra"])
    bg.flush()
    # right side: the schoolhouse, side wall along the street + its end wall facing us
    Q = facade(1560, 640, 560, 404)
    bg.shape(Q, B["school"])
    endw = np.array([Q[0], Q[1], (1400, Q[1][1]), (1400, Q[0][1])])
    bg.shape(endw, B["school_sh"])
    slab = np.array([quad_pt(Q, -.02, 1.0) + (0, -2), quad_pt(Q, 1.03, 1.0) + (0, -2), quad_pt(Q, 1.03, 1.0) + (0, -16 * depth(404)),
                     quad_pt(Q, -.02, 1.0) + (0, -24 * depth(560))])
    bg.shape(slab, B["slab"])
    bg.shape(np.array([Q[1] + (0, -24 * depth(560)), (1400, Q[1][1] - 24 * depth(560)), (1400, Q[1][1]), Q[1]]), B["slab2"])
    for k in range(5):
        u0 = .08 + k * .17
        bg.shape(quad_sub(Q, u0, u0 + .11, .42, .8), B["swin"])
    bg.shape(quad_sub(Q, .02, .98, .36, .4), B["slab"])
    # breeze-block panel on the end wall
    for yy in np.arange(Q[1][1] + 30, Q[0][1] - 20, 26):
        for xx in np.arange(Q[0][0] + 28, 1390, 26):
            path(ctx, np.array([(xx, yy - 8), (xx + 8, yy), (xx, yy + 8), (xx - 8, yy)]))
            ctx.set_source_rgba(*B["school"])
            ctx.fill()
    bg.flush()
    # left side: a mint ranch house
    QL = facade(-520, 420, 600, 450)
    bg.shape(QL, B["hedge"])
    rfL = np.array([QL[1], QL[2], quad_pt(QL, 1.05, 1.0) + (0, -40 * depth(450)), quad_pt(QL, -.05, 1.0) + (0, -70 * depth(600))])
    bg.shape(rfL, B["roof"])
    bg.shape(quad_sub(QL, .2, .55, .3, .75), B["trim"])
    bg.shape(quad_sub(QL, .23, .52, .36, .7), B["win"])
    bg.shape(quad_sub(QL, .66, .8, .0, .7), B["door"])
    bg.flush()
    # lollipop trees along both sidewalks, smaller with distance
    for x0b, y in ((-60, 470), (-60, 410), (1340, 470), (1340, 412)):
        d = depth(y)
        lollipop_tree(bg, on(x0b, y), y, 340 * d, 140 * d, int(x0b + y))
        bg.flush()
    lollipop_tree(bg, on(-40, 560), 560, 330 * depth(560), 120 * depth(560), 77)
    bg.flush()


def bg_title(bg):
    pass


BGS = {"corner": bg_corner, "cross": bg_cross}


# =============================================================================================== camera
def cam_matrix(cam):
    z, cx, cy = cam
    return cairo.Matrix(xx=z, yy=z, x0=640 - cx * z, y0=360 - cy * z)


def new_surface(ss):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * ss, H * ss)
    ctx = cairo.Context(surf)
    ctx.scale(ss, ss)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    return surf, ctx


_BGC = {}


def render_bg(name, cam, ss=2):
    key = (name, tuple(cam), ss)
    if key not in _BGC:
        surf, ctx = new_surface(ss)
        ctx.transform(cam_matrix(cam))
        LS[0] = cam[0] ** -.75
        BGS[name](BGPainter(ctx))
        LS[0] = 1.0
        surf.flush()
        a = np.ndarray((H * ss, W * ss, 4), np.uint8, surf.get_data())[:, :, :3][:, :, ::-1].astype(np.float32) / 255
        _BGC[key] = gouache(cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA))
    return _BGC[key]


def finish(img, i=0, grain=.008):
    """A light film/broadcast transfer: tiny softening and a whisper of grain, nothing that muddies the cel."""
    img = cv2.GaussianBlur(img, (0, 0), .35)
    rng = np.random.default_rng(1000 + i)
    g = cv2.resize(rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32), (W, H))
    return np.clip(img + g[..., None] * grain, 0, 1)


def to8(img):
    return (np.clip(img, 0, 1) * 255 + .5).astype(np.uint8)


# =============================================================================================== Bertram (generalised rig)
C.update(wool_wet=hexc("#7b4027"), water=hexc("#9fdcf5"), green=hexc("#8fcf6b"), shell=hexc("#e29a37"), shell2=hexc("#f2bd5c"),
         plastron=hexc("#f6e3a3"), hat=hexc("#e8413f"), hat2=hexc("#c22f31"), lens=hexc("#f4fbff"), satchel=hexc("#ffd34d"),
         hydrant=hexc("#e5403c"), hydrant2=hexc("#b92c2c"), metal=hexc("#d9dde6"), duck=hexc("#ffd84a"), duck2=hexc("#f6b93b"),
         beak=hexc("#ff9a2e"), duckcap=hexc("#4f8fd6"))
MASS = [(0, -124), (36, -130), (68, -150), (106, -194), (150, -258), (186, -328), (196, -398), (176, -462), (120, -518), (40, -560),
        (-44, -580), (-126, -574), (-198, -546), (-254, -494), (-284, -426), (-276, -352), (-230, -282), (-160, -212), (-94, -160),
        (-42, -131)]
ANCHOR = (0.0, -125.0)
SASH = [(-320, -610), (-30, -372), (250, -150)]
BADGE = (-92, -424)
HEAD0 = (152.0, -334.0)
HEAD_TILT = -.045
BPOSE = dict(X=520.0, Y=655.0, S=1.0, x=0.0, y=0.0, sx=1.0, sy=1.0, lean=0.0, hx=0.0, hy=0.0, hr=0.0, cap=0.0, capr=0.0, capx=0.0,
             sign=0.0, jaw=0.0, mouth="smile", lid=.24, px=-2.5, py=0.0, brow=0.0, blink=0.0, eyes=1.0, squeeze=False,
             far="sign", near="akimbo", signx=None, wet=False, grope=(270.0, -80.0))


def bpose(**kw):
    p = dict(BPOSE)
    p.update(kw)
    return p


def body_matrix(P):
    m = cairo.Matrix(x0=-ANCHOR[0], y0=-ANCHOR[1])
    m = m.multiply(cairo.Matrix(xx=P["sx"], yy=P["sy"])).multiply(cairo.Matrix.init_rotate(P["lean"]))
    return m.multiply(cairo.Matrix(x0=ANCHOR[0], y0=ANCHOR[1]))


def xf(m, p):
    return np.array(m.transform_point(float(p[0]), float(p[1])))


def hem(x):
    return -226 + (np.asarray(x) + 270) * .11


_MO = {}


def mass_outline(wet=False):
    if wet not in _MO:
        base = cr(MASS, per=24)
        O = scallop(base, lam=42 if not wet else 30, amp=11 if not wet else 4, power=.5, seed=7, skew=1.25 if not wet else 2.2,
                    amp_fn=lambda Q: np.clip((hem(Q[:, 0]) - Q[:, 1]) / 26, 0, 1))
        xs = np.linspace(-330, 270, 60)
        hemP = np.vstack([np.stack([xs, hem(xs) + (12 if wet else 0)], 1), [(270, -720), (-330, -720)]])
        M_ = scallop(hemP, lam=30 if not wet else 22, amp=13 if not wet else 16, power=.5 if not wet else 1.6, seed=12,
                     skew=1.3 if not wet else 1.0, amp_fn=lambda Q: (Q[:, 1] > -650).astype(float))
        _MO[wet] = (O, M_)
    return _MO[wet]


def sign_geom(P):
    s = P["sign"]
    hand = np.array([lerp(292, 338, s), lerp(-250, -318, s)])
    tilt = lerp(0.0, .07, s)
    up = np.array([math.sin(tilt), -math.cos(tilt)])
    return hand, hand + up * 262, hand - up * 252, tilt


def planted_sign_geom(x):
    return None, np.array([x, -514.0]), np.array([x, 0.0]), 0.0


def draw_sign(ctx, disc, bottom, tilt):
    up = (disc - bottom) / np.linalg.norm(disc - bottom)
    side = np.array([-up[1], up[0]])
    fs(ctx, np.array([bottom + side * 6.5, disc + side * 6.5, disc - side * 6.5, bottom - side * 6.5]), C["pole"])
    fs(ctx, ell(disc[0], disc[1], 60, 60, tilt), C["white"])
    fs(ctx, ell(disc[0], disc[1], 48, 48, tilt), C["red"], 2.4)
    ctx.save()
    ctx.translate(*disc)
    ctx.rotate(tilt)
    palm = cr([(-15, 22), (-17, 0), (-25, -8), (-21, -13), (-12, -6), (-13, -26), (-7, -30), (-3, -10), (-2, -33), (4, -34), (5, -10),
               (9, -30), (15, -28), (13, -6), (18, -22), (23, -19), (19, 2), (14, 22)], per=6, tension=.6)
    fs(ctx, palm, C["white"], 0)
    ctx.restore()


def fist(ctx, c, ang, scale=1.0, col=None):
    ctx.save()
    ctx.translate(*c)
    ctx.rotate(ang)
    ctx.scale(scale, scale)
    F = cr([(-20, -16), (-4, -22), (14, -20), (24, -10), (25, 8), (16, 20), (-4, 22), (-20, 14), (-25, 0)], per=10)
    fs(ctx, F, col or C["hand"])
    for y in (-8, 2, 12):
        taper_line(ctx, [(8, y - 2), (18, y), (23, y + 1)], 2.6, 2.0)
    taper_line(ctx, [(-14, -10), (-4, -4), (2, 6)], 2.8, 2.0)
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


def open_hand(ctx, c, ang, scale=1.0, spread=1.0, col=None):
    """A cartoon four-finger hand pointing along +x after rotation: three fat fingers and a thumb."""
    ctx.save()
    ctx.translate(*c)
    ctx.rotate(ang)
    ctx.scale(scale, scale)
    col = col or C["hand"]
    for k, a in enumerate((-.32, 0.0, .32)):
        a *= spread
        p0 = np.array([8.0, k * 9 - 9.0])
        p1 = p0 + 24 * np.array([math.cos(a), math.sin(a)])
        fs(ctx, tube([p0, p1], 13, 12), col, 3.0)
    fs(ctx, tube([(-2, -12), (8, -26 - 4 * spread)], 13, 11), col, 3.0)
    fs(ctx, cr(ell(0, 0, 18, 16, 0, 10), per=6), col, 3.0)
    ctx.restore()


def draw_legs(ctx, P):
    for sx_, ax, ay in ((-1, -36, -16), (1, 32, -14)):
        fs(ctx, tube([(sx_ * 24, -104), (sx_ * 25, -62), (ax, ay)], 27, 17), C["trouser"])
    pel = cr([(-46, -150), (46, -152), (58, -112), (44, -88), (0, -84), (-44, -86), (-58, -110)], per=12)
    fs(ctx, pel, C["trouser"])
    taper_line(ctx, [(0, -112), (1, -98), (2, -88)], 2.6, 1.8)
    for cx, rot in ((-42, -.12), (42, .14)):
        fs(ctx, cr(ell(cx, -9, 24, 11, rot, 10), per=6), C["shoe"])
        fs(ctx, ell(cx - 6 * np.sign(cx), -14, 8, 3, rot, 16), hexc("#4a4352"), 0)


def wool_curl(ctx, x, y, s=1.0, flip=1):
    taper_line(ctx, [(x - 10 * s * flip, y - 8 * s), (x - 13 * s * flip, y + 2 * s), (x - 4 * s * flip, y + 9 * s), (x + 7 * s * flip, y + 6 * s)], 3.0, 2.0)


def drip(ctx, x, y, s=1.0):
    D = cr([(x, y - 8 * s), (x + 5 * s, y + 2 * s), (x + 4 * s, y + 7 * s), (x, y + 9 * s), (x - 4 * s, y + 7 * s), (x - 5 * s, y + 2 * s)], per=6)
    fs(ctx, D, C["water"], 2.4)


def draw_head(ctx, P, c):
    wool = C["wool_wet"] if P["wet"] else C["wool"]
    ctx.save()
    ctx.translate(*c)
    ctx.rotate(P["hr"] + HEAD_TILT)
    jaw = P["jaw"]
    wet = P["wet"]
    beard = scallop(cr([(-50, 34), (-14, 52), (24, 58), (58, 46), (60, 80 + 10 * wet), (30, 106 + jaw * .6 + 14 * wet), (-14, 108 + jaw * .5 + 14 * wet),
                        (-48, 84 + 8 * wet)]), lam=20 if not wet else 14, amp=6, ranges=[(.25, .98, 1)], seed=3, skew=1.0 if not wet else 2.0)
    fs(ctx, beard, wool)
    for pts, w in (([(46, -50), (72, -60), (84, -78), (83, -98)], 22), ([(-52, -44), (-82, -50), (-100, -68), (-102, -92)], 24)):
        fs(ctx, tube(pts, w, 6), C["horn"])
    fs(ctx, ell(64, -30, 15, 8, -.55, 32), C["face"])
    fs(ctx, ell(-70, -20 + 6 * wet, 19, 9, .35 + .5 * wet, 32), C["face"])
    headP = cr([(-66, -28), (-58, -56), (-20, -72), (30, -70), (62, -50), (72, -12), (70, 22), (50, 48 + jaw * .4), (0, 56 + jaw * .5),
                (-44, 44), (-66, 12)], per=14)
    fs(ctx, headP, C["face"])
    lid = max(P["lid"], P["blink"])
    es = P["eyes"]
    for ex, ey, rx, ry, pdx in ((16, -14, 15, 21, 5.5), (-18, -12, 17, 22, 5.5)):
        rx, ry = rx * es, ry * es
        ey = ey - (es - 1) * 14
        if P["squeeze"]:
            line(ctx, np.array([(ex - rx, ey - 6), (ex + rx * .2, ey + 2), (ex - rx * .1, ey + 4), (ex + rx, ey + 10)]) if ex < 0 else
                 np.array([(ex + rx, ey - 6), (ex - rx * .2, ey + 2), (ex + rx * .1, ey + 4), (ex - rx, ey + 10)]), 3.2)
            continue
        E = ell(ex, ey, rx, ry, 0, 64)
        fs(ctx, E, C["eye"], 3.0)
        ctx.save()
        path(ctx, E)
        ctx.clip()
        if lid < .95:
            fs(ctx, ell(ex + pdx + P["px"], ey + 3 + P["py"], 17 * .42 * min(es, 1.1), 22 * .42 * min(es, 1.1), 0, 32), INK, 0)
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
        ctx.set_line_width(3.0 * LS[0])
        ctx.stroke()
    b = P["brow"] + 6 * wet
    fr = cr([(-72, -26 + b), (-70, -60), (-34, -82), (16, -86), (58, -70), (70, -40 + b), (44, -36 + b), (14, -34 + b), (-14, -30 + b),
             (-44, -26 + b)], per=14)
    fr = scallop(fr, lam=18 if not wet else 12, amp=5.5 if not wet else 4, power=.6, ranges=[(.0, 1, 1)], seed=5, skew=1.0 if not wet else 2.4)
    fs(ctx, fr, wool)
    mz = cr([(-14, 18), (0, 2), (30, -4), (66, 2), (84, 18), (82, 40 + jaw * .5), (60, 56 + jaw * .7), (24, 60 + jaw * .7), (-6, 50 + jaw * .5)], per=8)
    fs(ctx, mz, C["muzzle"])
    fs(ctx, cr([(14, 10), (26, 0), (46, -2), (68, 2), (78, 12), (70, 22), (46, 20), (24, 22)], per=8), C["nose"], 3.0)
    fs(ctx, ell(24, 13, 5.2, 4.2, .5, 20), INK, 0)
    fs(ctx, ell(70, 11, 4.4, 3.6, -.5, 20), INK, 0)
    fs(ctx, ell(44, 5, 9, 2.4, -.04, 20), hexc("#a5675a"), 0)
    m = P["mouth"]
    if m == "smile":
        taper_line(ctx, [(4, 40), (28, 47), (52, 45), (72, 34)], 3.4, 2.6)
        taper_line(ctx, [(68, 29), (74, 34), (74, 41)], 2.6, 1.8)
    elif m == "grin":           # a huge closed, self-satisfied grin
        taper_line(ctx, [(-6, 30), (20, 48), (50, 48), (78, 28)], 3.6, 2.8)
        taper_line(ctx, [(74, 22), (81, 28), (80, 36)], 2.6, 1.8)
        taper_line(ctx, [(-2, 24), (-8, 30), (-6, 37)], 2.6, 1.8)
    elif m == "mm":
        taper_line(ctx, [(6, 43), (30, 46), (52, 45), (70, 40)], 3.2, 2.6)
    elif m == "frown":
        taper_line(ctx, [(6, 50), (28, 43), (52, 43), (72, 50)], 3.2, 2.6)
    elif m == "wavy":           # flustered squiggle
        taper_line(ctx, [(6, 46), (18, 41), (30, 47), (42, 41), (54, 47), (68, 42)], 3.0, 2.4)
    else:
        o = max(jaw, 4)
        if m == "oh":
            M = ell(38, 44 + o * .45, 12 + o * .1, 6 + o * .5, 0, 40)
        else:
            M = cr([(8, 38), (38, 42), (70, 33), (62, 44 + o * .8), (38, 52 + o), (16, 46 + o * .8)], per=8)
        fs(ctx, M, C["mouth"], 3.2)
        ctx.save()
        path(ctx, M)
        ctx.clip()
        fs(ctx, ell(38, 56 + o * 1.0, 24, 10 + o * .2, 0, 32), C["tongue"], 0)
        ctx.restore()
        path(ctx, M)
        ctx.set_source_rgba(*INK)
        ctx.set_line_width(3.2 * LS[0])
        ctx.stroke()
    ctx.save()
    ctx.translate(P["capx"], P["cap"] + 4 * wet)
    ctx.rotate(P["capr"])
    fs(ctx, cr([(-34, -74), (-32, -94), (-12, -106), (16, -106), (34, -94), (38, -74)], per=10), C["cap"])
    fs(ctx, cr(np.array([(-35, -86), (38, -86), (39, -72), (-35, -72)]), per=3, tension=.3), C["cap2"])
    fs(ctx, cr([(26, -78), (54, -80), (66, -72 + 8 * wet), (48, -66 + 8 * wet), (24, -71)], per=8), C["cap2"])
    star = np.array([(4 + (8 if k % 2 == 0 else 3.6) * math.cos(-math.pi / 2 + k * math.pi / 5),
                      -94 + (8 if k % 2 == 0 else 3.6) * math.sin(-math.pi / 2 + k * math.pi / 5)) for k in range(10)])
    fs(ctx, star, C["badge"], 2.2)
    ctx.restore()
    if wet:
        for x, y, s in ((60, -60, 1.0), (-30, -26, .9), (80, 30, 1.0), (40, 72, 1.1), (-30, 126, 1.2), (20, 132, 1.0), (60, 110, .9)):
            drip(ctx, x, y, s)
    ctx.restore()


def draw_hydrant(ctx, Hd, z=1.0):
    """A stubby red fire hydrant.  Hd: X, Y (base), S, rot, capoff."""
    ctx.save()
    ctx.translate(Hd["X"], Hd["Y"])
    ctx.rotate(Hd.get("rot", 0.0))
    s = Hd.get("S", 1.0)
    ctx.scale(s, s)
    old = LS[0]
    LS[0] = (z * s * Hd.get("parent", 1.0)) ** -.75
    fs(ctx, cr([(-30, 0), (-30, -12), (30, -12), (30, 0)], per=3, tension=.3), C["hydrant2"])
    fs(ctx, cr([(-22, -10), (-22, -74), (22, -74), (22, -10)], per=3, tension=.2), C["hydrant"])
    for sx_ in (-1, 1):                                   # side nozzles
        if Hd.get("capoff") and sx_ == -1:
            fs(ctx, ell(sx_ * 30, -48, 7, 10), INK, 0)
            continue
        fs(ctx, cr([(sx_ * 20, -58), (sx_ * 36, -58), (sx_ * 36, -38), (sx_ * 20, -38)], per=3, tension=.3), C["metal"])
    fs(ctx, cr([(-28, -70), (-26, -92), (0, -104), (26, -92), (28, -70)], per=8), C["hydrant"])
    fs(ctx, cr([(-30, -72), (30, -72), (30, -64), (-30, -64)], per=3, tension=.3), C["hydrant2"])
    fs(ctx, cr([(-8, -100), (-8, -112), (8, -112), (8, -100)], per=3, tension=.3), C["metal"])
    fs(ctx, ell(-10, -44, 4, 14), hexc("#f27a6c"), 0)    # a painted shine
    LS[0] = old
    ctx.restore()


def draw_bertram(ctx, P, z=1.0):
    S = P["S"]
    LS[0] = (z * S) ** -.75
    wool = C["wool_wet"] if P["wet"] else C["wool"]
    ctx.save()
    ctx.translate(P["X"] + P["x"], P["Y"] + P["y"])
    ctx.scale(S, S)
    M = body_matrix(P)
    hc = xf(M, HEAD0) + np.array([P["hx"], P["hy"]])
    sh_far = xf(M, (150, -300))
    far = P["far"]
    # ---- far arm and the sign, behind the body
    if P["signx"] is not None:
        _, disc, bottom, tilt = planted_sign_geom(P["signx"])
        draw_sign(ctx, disc, bottom, tilt)
    if far == "sign":
        hand, disc, bottom, tilt = sign_geom(P)
        elbow_far = (sh_far + hand) / 2 + np.array([10, 40 - 20 * P["sign"]])
        fs(ctx, tube([sh_far, elbow_far, hand + np.array([-12, 4])], 50, 40), wool)
        draw_sign(ctx, disc, bottom, tilt)
        fist(ctx, hand + np.array([2, 0]), .05 + tilt)
    elif far == "ear":
        hand_ear = hc + np.array([96, -30])
        fs(ctx, tube([sh_far, sh_far + np.array([120, 40]), hand_ear + np.array([4, 22])], 50, 40), wool)
    elif far == "grope":
        g = np.array(P["grope"])
        el = xf(M, (250, -230))
        fs(ctx, tube([sh_far, el, g + np.array([-6, -14])], 50, 40), wool)
    elif far == "cradle":
        el = xf(M, (262, -190))
        fs(ctx, tube([sh_far, el], 50, 46), wool)
    elif far == "wave":
        el = xf(M, (300, -360))
        hw = xf(M, (330, -500))
        fs(ctx, tube([sh_far, el, hw], 50, 40), wool)
    # ---- near upper arm (behind the mass)
    sh_near = xf(M, (-224, -350))
    near = P["near"]
    if near == "akimbo":
        elbow = xf(M, (-312, -270))
    elif near == "raise":
        elbow = xf(M, (-330, -420))
    elif near == "sign":
        elbow = xf(M, (-306, -292))
    else:  # hang
        elbow = xf(M, (-300, -250))
    fs(ctx, tube([sh_near, elbow], 66, 54), wool)
    draw_legs(ctx, P)
    # ---- the mass
    O, MANT = mass_outline(P["wet"])
    ctx.save()
    ctx.transform(M)
    fs(ctx, O, C["face"])
    ctx.save()
    path(ctx, O)
    ctx.clip()
    fs(ctx, MANT, wool)
    fs(ctx, tube(SASH, 66, 66), C["sash"])
    if not P["wet"]:
        for (x, y, s, f) in ((-196, -410, 1.1, 1), (-214, -330, 1.0, 1), (-20, -505, 1.0, -1), (70, -470, .9, -1), (-120, -500, .9, 1)):
            wool_curl(ctx, x, y, s, f)
    ctx.restore()
    bx, by = BADGE
    fs(ctx, cr([(bx - 16, by - 14), (bx + 14, by - 20), (bx + 16, by + 10), (bx, by + 26), (bx - 16, by + 10)], per=6, tension=.6), C["badge"], 3.0)
    fs(ctx, ell(bx, by, 6, 6), hexc("#e8a92a"), 0)
    if P["wet"]:
        for x in np.linspace(-150, 120, 7):
            drip(ctx, x + 8, hem(x) + 30, 1.3)
    ctx.restore()
    fs(ctx, cr([(-50, -128), (0, -122), (50, -130), (51, -116), (0, -108), (-51, -114)], per=8, tension=.6), C["cap2"], 3.0)
    fs(ctx, cr([(-4, -126), (12, -127), (13, -110), (-3, -109)], per=3, tension=.3), C["badge"], 2.4)
    # ---- near forearm + hand
    if near == "akimbo":
        hipc = xf(M, (-160, -196))
        fs(ctx, tube([elbow, (elbow + hipc) / 2 + np.array([-4, 6]), hipc + np.array([-10, -6])], 52, 40), wool)
        hip_fist(ctx, hipc)
    elif near == "sign":
        nh = np.array([-344.0, -262.0])
        draw_sign(ctx, np.array([-346.0, -516.0]), np.array([-342.0, 0.0]), -.008)
        fs(ctx, tube([elbow, nh + np.array([12, -4])], 52, 40), wool)
        fist(ctx, nh, 3.1)
    elif near == "raise":
        fp = xf(M, (-300, -580))
        fs(ctx, tube([elbow, fp + np.array([0, 18])], 52, 42), wool)
        fist(ctx, fp, -1.5)
    else:
        hp_ = xf(M, (-292, -120))
        fs(ctx, tube([elbow, hp_ + np.array([0, -16])], 50, 40), wool)
        open_hand(ctx, hp_, 1.45, 1.0, .5)
    # ---- far arm in front (grope hand / cradle)
    if far == "grope":
        open_hand(ctx, np.array(P["grope"]), 1.25, 1.05, 1.3)
    if far == "cradle":
        el = xf(M, (262, -190))
        hnd = xf(M, (292, -150))
        fs(ctx, tube([el, hnd + np.array([-10, -4])], 46, 38), wool)
        open_hand(ctx, hnd, -.35, 1.0, .5)
        hb = xf(M, CRADLE[:2])
        draw_hydrant(ctx, dict(X=hb[0], Y=hb[1], S=CRADLE[3], rot=CRADLE[2], capoff=P.get("capoff", False), parent=S), z)
        LS[0] = (z * S) ** -.75
    # ---- head
    draw_head(ctx, P, hc)
    if far == "ear":
        open_hand(ctx, hc + np.array([98, -34]), -1.9, 1.0, .5)
    if far == "wave":
        open_hand(ctx, xf(M, (330, -520)), -1.45, 1.05, 1.2)
    ctx.restore()


CRADLE = (206.0, -168.0, 1.25, .95)     # hydrant base (local), rotation, scale when cradled


def hydrant_nozzle(P):
    """World position of the open (left) nozzle of the cradled hydrant."""
    M = body_matrix(P)
    hb = xf(M, CRADLE[:2])
    r, s = CRADLE[2], CRADLE[3]
    lx, ly = -38 * s, -48 * s
    p = hb + np.array([lx * math.cos(r) - ly * math.sin(r), lx * math.sin(r) + ly * math.cos(r)])
    return np.array([P["X"] + P["x"] + p[0] * P["S"], P["Y"] + P["y"] + p[1] * P["S"]])


def bertram_head(P):
    M = body_matrix(P)
    hc = xf(M, HEAD0) + np.array([P["hx"], P["hy"]])
    return np.array([P["X"] + P["x"] + hc[0] * P["S"], P["Y"] + P["y"] + hc[1] * P["S"]])


def bertram_disc(P):
    """World position of the sign disc centre (for seating Tilly on it)."""
    M = body_matrix(P)
    if P["far"] == "sign":
        _, disc, _, _ = sign_geom(P)
    else:
        _, disc, _, _ = planted_sign_geom(P["signx"])
    return np.array([P["X"] + P["x"] + disc[0] * P["S"], P["Y"] + P["y"] + disc[1] * P["S"]])


# =============================================================================================== Tilly the tortoise
TPOSE = dict(X=700.0, Y=655.0, S=.62, face=1, arms="down", sit=False, lid=.0, px=0.0, py=0.0, mouth="flat", climb=False)


def tpose(**kw):
    p = dict(TPOSE)
    p.update(kw)
    return p


def draw_tilly(ctx, T, z=1.0):
    """Tilly Tuckett: a tiny, unflappable tortoise schoolgirl.  Head-heavy where Bertram is chest-heavy."""
    S = T["S"]
    LS[0] = (z * S) ** -.75 * .9
    ctx.save()
    ctx.translate(T["X"], T["Y"])
    ctx.scale(S * T["face"], S)
    arms = T["arms"]
    # legs
    if T["sit"]:
        for y0, dx in ((-34, 0), (-30, 8)):
            fs(ctx, tube([(4 + dx, y0), (34 + dx, y0 + 4)], 20, 18), C["green"])
            fs(ctx, ell(40 + dx, y0 + 4, 10, 12, 0, 24), C["green"])
    elif T["climb"]:
        fs(ctx, tube([(-8, -30), (14, -18), (26, -24)], 20, 17), C["green"])
        fs(ctx, tube([(10, -30), (20, -4), (28, -8)], 20, 17), C["green"])
    else:
        for x0, x1 in ((-10, -12), (12, 14)):
            fs(ctx, tube([(x0, -34), (x1, -6)], 21, 19), C["green"])
            fs(ctx, cr(ell(x1 + 5, -5, 14, 7, 0, 10), per=6), C["green"])
    # back arm (behind the shell edge)
    if arms in ("climb",):
        fs(ctx, tube([(-4, -86), (22, -112)], 17, 15), C["green"])
        fs(ctx, ell(26, -116, 9, 9, 0, 24), C["green"])
    # shell on her back
    shell = cr([(-48, -46), (-52, -78), (-40, -108), (-14, -118), (8, -100), (10, -40), (-20, -30)], per=12)
    fs(ctx, shell, C["shell"])
    ctx.save()
    path(ctx, shell)
    ctx.clip()
    for (hx_, hy_) in ((-26, -76), (-44, -104), (-46, -50), (-8, -112), (-10, -44)):
        hexp = np.array([(hx_ + 13 * math.cos(a), hy_ + 13 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 7)[:-1] + math.pi / 6])
        fs(ctx, hexp, C["shell2"], 2.6)
    ctx.restore()
    path(ctx, shell)
    ctx.set_source_rgba(*INK)
    ctx.set_line_width(3.6 * LS[0])
    ctx.stroke()
    # body / plastron
    fs(ctx, cr(ell(6, -66, 25, 34, .05, 12), per=6), C["plastron"])
    for y in (-78, -60):
        taper_line(ctx, [(-12, y), (6, y + 2), (26, y)], 2.4, 1.8)
    # satchel strap + satchel
    line(ctx, np.array([(-6, -94), (28, -44)]), 4.0, C["satchel"])
    fs(ctx, cr([(16, -50), (42, -52), (44, -26), (18, -24)], per=3, tension=.3), C["satchel"], 3.0)
    line(ctx, np.array([(16, -42), (43, -43)]), 2.4)
    # head
    hcx, hcy = 12, -126
    fs(ctx, cr(ell(hcx, hcy, 31, 28, 0, 14), per=6), C["green"])
    # glasses + eyes
    for ex in (hcx - 4, hcx + 20):
        fs(ctx, ell(ex, hcy - 2, 11, 11, 0, 40), C["lens"], 3.2)
        if T["lid"] < .9:
            fs(ctx, ell(ex + 2 + T["px"], hcy - 1 + T["py"], 3.6, 4.2, 0, 20), INK, 0)
        if T["lid"] > .05:
            ctx.save()
            path(ctx, ell(ex, hcy - 2, 11, 11, 0, 40))
            ctx.clip()
            ly = hcy - 13 + 22 * T["lid"]
            ctx.rectangle(ex - 12, hcy - 14, 24, ly - (hcy - 14))
            ctx.set_source_rgba(*C["green"])
            ctx.fill()
            line(ctx, np.array([(ex - 12, ly), (ex + 12, ly)]), 2.6)
            ctx.restore()
            path(ctx, ell(ex, hcy - 2, 11, 11, 0, 40))
            ctx.set_source_rgba(*INK)
            ctx.set_line_width(3.2 * LS[0])
            ctx.stroke()
    line(ctx, np.array([(hcx + 7, hcy - 4), (hcx + 9, hcy - 5)]), 2.6)
    m = T["mouth"]
    if m == "flat":
        taper_line(ctx, [(hcx + 4, hcy + 15), (hcx + 14, hcy + 16), (hcx + 24, hcy + 14)], 2.6, 2.0)
    elif m == "smile":
        taper_line(ctx, [(hcx + 2, hcy + 12), (hcx + 13, hcy + 18), (hcx + 26, hcy + 11)], 2.8, 2.0)
    elif m == "frown":
        taper_line(ctx, [(hcx + 4, hcy + 18), (hcx + 14, hcy + 14), (hcx + 24, hcy + 18)], 2.6, 2.0)
    # bobble hat
    fs(ctx, cr([(hcx - 30, hcy - 12), (hcx - 26, hcy - 34), (hcx, hcy - 46), (hcx + 26, hcy - 36), (hcx + 32, hcy - 14)], per=8), C["hat"])
    fs(ctx, cr([(hcx - 32, hcy - 18), (hcx + 34, hcy - 20), (hcx + 34, hcy - 8), (hcx - 32, hcy - 6)], per=3, tension=.4), C["hat2"])
    fs(ctx, scallop(cr(ell(hcx - 2, hcy - 52, 11, 10, 0, 8)), lam=7, amp=2, seed=2), C["white"])
    # front arm
    if arms == "down":
        fs(ctx, tube([(18, -88), (30, -58)], 17, 15), C["green"])
        fs(ctx, ell(31, -54, 9, 9, 0, 24), C["green"])
    elif arms in ("raise", "wave"):
        tip = (40, -168) if arms == "raise" else (48, -160)
        fs(ctx, tube([(18, -88), (32, -120), tip], 17, 15), C["green"])
        fs(ctx, ell(tip[0] + 2, tip[1] - 6, 10, 11, 0, 24), C["green"])
        for a in (-.5, 0, .5):
            line(ctx, np.array([(tip[0] + 2 + 8 * math.sin(a), tip[1] - 12 - 6 * math.cos(a)), (tip[0] + 2 + 13 * math.sin(a), tip[1] - 16 - 10 * math.cos(a))]), 0.1)
    elif arms == "tap":
        fs(ctx, tube([(18, -88), (40, -76), (62, -84)], 17, 15), C["green"])
        fs(ctx, ell(66, -86, 9, 9, 0, 24), C["green"])
    elif arms == "facepalm":
        fs(ctx, tube([(18, -88), (40, -104), (hcx + 14, hcy + 2)], 17, 15), C["green"])
        fs(ctx, cr(ell(hcx + 12, hcy - 2, 14, 12, .3, 10), per=6), C["green"])
    elif arms == "climb":
        fs(ctx, tube([(18, -88), (24, -126), (26, -140)], 17, 15), C["green"])
        fs(ctx, ell(27, -144, 9, 9, 0, 24), C["green"])
    elif arms == "hips":
        fs(ctx, tube([(18, -88), (36, -72), (24, -52)], 17, 15), C["green"])
        fs(ctx, ell(22, -50, 8, 8, 0, 24), C["green"])
    ctx.restore()


# =============================================================================================== ducklings (the button gag)
def draw_ducklings(ctx, D, z=1.0):
    """D: list of dicts X, Y, S, wing (0/1), look"""
    for d in D:
        S = d["S"]
        LS[0] = (z * S) ** -.75 * .9
        ctx.save()
        ctx.translate(d["X"], d["Y"])
        ctx.scale(S, S)
        for fx in (-8, 8):
            fs(ctx, cr([(fx - 4, -6), (fx + 10, -4), (fx + 12, 0), (fx - 6, 0)], per=4, tension=.4), C["beak"], 2.6)
        fs(ctx, tube([(-6, -14), (-6, -4)], 6, 6), C["beak"], 2.2)
        fs(ctx, tube([(8, -14), (8, -4)], 6, 6), C["beak"], 2.2)
        fs(ctx, cr(ell(0, -30, 24, 19, 0, 12), per=6), C["duck"])
        if d.get("wing"):
            fs(ctx, cr([(-8, -36), (-22, -62), (-12, -64), (4, -40)], per=6), C["duck2"], 3.0)
        else:
            fs(ctx, cr(ell(-6, -30, 12, 8, -.2, 10), per=6), C["duck2"], 3.0)
        fs(ctx, cr(ell(12, -60, 16, 15, 0, 12), per=6), C["duck"])
        fs(ctx, cr([(24, -60), (40, -56), (26, -51)], per=4, tension=.4), C["beak"], 2.8)
        fs(ctx, ell(16 + d.get("look", 0), -64, 3, 3.6, 0, 16), INK, 0)
        fs(ctx, cr([(-4, -70), (6, -80), (24, -76), (28, -68), (0, -66)], per=6), C["duckcap"], 2.8)
        ctx.restore()


# =============================================================================================== scene rendering
def cel(draw_fn, cam, ss=2):
    surf, ctx = new_surface(ss)
    ctx.transform(cam_matrix(cam))
    draw_fn(ctx)
    LS[0] = 1.0
    surf.flush()
    a = np.ndarray((H * ss, W * ss, 4), np.uint8, surf.get_data()).astype(np.float32) / 255
    rgb, al = a[:, :, 2::-1], a[:, :, 3]
    mask = (al > .5).astype(np.uint8)
    d = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    ring = np.clip(OUTLINE * ss + .5 - d, 0, 1) * (1 - al)
    rgb = rgb + np.array(INK[:3], np.float32) * ring[..., None]
    al = al + ring
    return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA), cv2.resize(al, (W, H), interpolation=cv2.INTER_AREA)


def shadows(img, ells, cam, ss=2):
    if not ells:
        return img
    surf, ctx = new_surface(ss)
    ctx.transform(cam_matrix(cam))
    for (cx, cy, rx, ry) in ells:
        path(ctx, ell(cx, cy, rx, ry, 0, 64))
        ctx.set_source_rgba(0, 0, 0, 1)
        ctx.fill()
    surf.flush()
    a = np.ndarray((H * ss, W * ss, 4), np.uint8, surf.get_data())[:, :, 3].astype(np.float32) / 255
    a = cv2.GaussianBlur(cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA), (0, 0), 1.2)[..., None]
    tint = np.array([.84, .8, .86], np.float32)
    return img * (1 - a + a * tint)


def item_shadow(kind, p):
    if kind == "bertram":
        if p.get("noshadow"):
            return None
        return (p["X"] + p["x"] + 2 * p["S"], p["Y"] + 1, 110 * p["S"] * (.85 + .15 * p["sx"]), 13 * p["S"])
    if kind == "tilly":
        if p.get("sit") or p.get("climb") or p.get("noshadow"):
            return None
        return (p["X"] + 4 * p["S"], p["Y"] + 1, 42 * p["S"], 8 * p["S"])
    if kind == "hydrant":
        return (p["X"], p["Y"] + 1, 38 * p.get("S", 1), 8 * p.get("S", 1))
    return None


DRAW = {"bertram": draw_bertram, "tilly": draw_tilly, "hydrant": draw_hydrant, "ducks": draw_ducklings}


def render_scene(sc, ss=2, i=0, grain=True):
    cam = sc["cam"]
    img = render_bg(sc["bg"], cam, ss).copy()
    sh = []
    for kind, p in sc["items"]:
        if kind == "ducks":
            sh += [(d["X"], d["Y"] + 1, 22 * d["S"], 5 * d["S"]) for d in p]
        else:
            e = item_shadow(kind, p)
            if e:
                sh.append(e)
    img = shadows(img, sh, cam, ss)
    for kind, p in sc["items"]:
        rgb, al = cel(lambda ctx, k=kind, q=p: DRAW[k](ctx, q, cam[0]), cam, ss)
        img = img * (1 - al[..., None]) + rgb
        if sc.get("fx"):
            pass
    for fx in sc.get("fx", []):
        rgb, al = cel(lambda ctx, f=fx: f(ctx, cam[0]), cam, ss)
        img = img * (1 - al[..., None]) + rgb
    img = np.clip(img, 0, 1)
    return finish(img, i) if grain else img


def key_png(sc, ss=3):
    return to8(render_scene(sc, ss))[8:712]


def water_fx(spout, to, z=1.0, s=1.0):
    """A cel-painted water dribble from the hydrant nozzle plus a few drops (ink-outlined, flat)."""
    def f(ctx, zz):
        LS[0] = (zz * s) ** -.75
        sp, tp = np.array(spout, float), np.array(to, float)
        mid = (sp + tp) / 2 + np.array([18 * s, -10 * s])
        fs(ctx, tube([sp, mid, tp], 16 * s, 22 * s), C["water"], 2.8)
        for k, (dx, dy) in enumerate(((-26, -18), (30, -26), (-34, 10), (36, 6), (-10, -36))):   # splash
            drip(ctx, tp[0] + dx * s, tp[1] + dy * s, s * 1.4)
        LS[0] = 1.0
    return f
