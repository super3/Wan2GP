"""Painted backgrounds for wobble_suburb: soft gouache/watercolour suburbia.

Shapes are laid in flat with cairo, then each is scrubbed with many semi-transparent brush strokes (value and hue jittered,
directional per surface), foliage is dabbed, and the whole card gets a hand-wobbled edge warp, pigment pooling at edges,
granulation and paper tooth.  Backgrounds carry no black ink: only thin darker-local-colour brush lines, like a BG painter's."""
import math

import cairo
import cv2
import numpy as np

import ink as K
from ink import hexc

W, H, SS = K.W, K.H, K.SS


# ------------------------------------------------------------------------------------------------ helpers
def to_rgba(surf, w=W, h=H, ss=SS):
    a = np.ndarray((h * ss, w * ss, 4), np.uint8, surf.get_data()).astype(np.float32) / 255
    a = a[:, :, [2, 1, 0, 3]]
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)


def over(dst, src):
    """dst RGB float; src premultiplied RGBA float."""
    return src[..., :3] + dst * (1 - src[..., 3:4])


def jcol(c, r, vj=.05, hj=.02):
    v = 1 + r.normal(0, vj)
    return tuple(float(np.clip(c[i] * v + r.normal(0, hj), 0, 1)) for i in range(3))


def mix(a, b, t):
    return tuple(a[i] * (1 - t) + b[i] * t for i in range(3)) + (1.0,)


def brush(ctx, base, r, n, lmin, lmax, wmin, wmax, ang, angj=.15, vj=.05, hj=.015, alpha=(.18, .35), curve=.15):
    """Scrub strokes inside the current clip (call after ctx.clip())."""
    x0, y0, x1, y1 = ctx.clip_extents()
    for _ in range(n):
        x, y = r.uniform(x0 - 10, x1 + 10), r.uniform(y0 - 10, y1 + 10)
        L = r.uniform(lmin, lmax)
        a = ang + r.normal(0, angj)
        dx, dy = math.cos(a) * L / 2, math.sin(a) * L / 2
        bend = r.normal(0, curve) * L
        c = jcol(base, r, vj, hj)
        ctx.set_source_rgba(*c, r.uniform(*alpha))
        ctx.set_line_width(r.uniform(wmin, wmax))
        ctx.move_to(x - dx, y - dy)
        ctx.curve_to(x - dx / 3 - dy / L * bend, y - dy / 3 + dx / L * bend, x + dx / 3 - dy / L * bend, y + dy / 3 + dx / L * bend,
                     x + dx, y + dy)
        ctx.stroke()


def region(ctx, path_fn, base, seed, strokes=(), edge=None):
    """Fill a region flat, then scrub it with the given stroke passes (dicts of brush() kwargs)."""
    r = np.random.default_rng(seed)
    path_fn(ctx)
    ctx.set_source_rgba(*base[:3], 1)
    if edge is not None:
        ctx.fill_preserve()
        ctx.set_source_rgba(*edge)
        ctx.set_line_width(1.6)
        ctx.stroke()
    else:
        ctx.fill()
    if strokes:
        ctx.save()
        path_fn(ctx)
        ctx.clip()
        for s in strokes:
            s = dict(s)
            b = s.pop("base", base)
            brush(ctx, b, r, **s)
        ctx.restore()


def bline(ctx, pts, c, w, seed, wob=1.2, alpha=.8):
    """A thin hand-painted line (darker local colour, no black)."""
    r = np.random.default_rng(seed)
    p = np.asarray(pts, float)
    d = K.crs(p + r.normal(0, wob, p.shape), False, 8)
    s = K.arclen(d)
    ww = w * (0.65 + .35 * np.sin(s / 23 + r.uniform(0, 6)))
    ctx.set_source_rgba(*c[:3], alpha)
    for a in range(0, len(d) - 1, 3):
        ctx.set_line_width(float(ww[a]))
        K.poly_path(ctx, d[a:a + 4], False)
        ctx.stroke()


def dabs(ctx, cx, cy, rx, ry, cols, n, seed, size=(5, 12), light=(.6, -.7), alpha=(.55, .9), clip_circle=True):
    """Foliage: dab round brush marks, darker on the shadow side, lighter toward the light."""
    r = np.random.default_rng(seed)
    for _ in range(n):
        a, rr = r.uniform(0, 2 * np.pi), math.sqrt(r.uniform(0, 1))
        x, y = cx + rx * rr * math.cos(a), cy + ry * rr * math.sin(a)
        side = ((x - cx) / rx * light[0] + (y - cy) / ry * light[1]) / 1.2  # -1 shadow .. 1 light
        t = np.clip(.5 + side * .55 + r.normal(0, .18), 0, .999)
        c = jcol(cols[int(t * len(cols))], r, .04, .01)
        s = r.uniform(*size)
        ctx.set_source_rgba(*c, r.uniform(*alpha))
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(r.uniform(0, 3.14))
        ctx.scale(s, s * r.uniform(.6, 1))
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.restore()
        ctx.fill()


# ------------------------------------------------------------------------------------------------ finishing
_NZ = {}


def _noise(h, w, sigma, seed):
    key = (h, w, sigma, seed)
    if key not in _NZ:
        r = np.random.default_rng(seed)
        n = r.normal(0, 1, (h, w)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigma)
        _NZ[key] = n / (n.std() + 1e-6)
    return _NZ[key]


def finish_paint(img, seed=5, warp=1.3, pool=.55, gran=.022, tooth=.035):
    h, w = img.shape[:2]
    # hand-wobbled edges
    dx = _noise(h, w, 6, seed) * warp
    dy = _noise(h, w, 6, seed + 1) * warp
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    img = cv2.remap(img, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    img = cv2.GaussianBlur(img, (0, 0), .55)
    # pigment pooling: darken the edges of washes a touch, like gouache/watercolour drying at the rim
    lo = cv2.GaussianBlur(img, (0, 0), 2.5)
    img = img + (img - lo) * pool
    # granulation and paper tooth
    g = _noise(h, w, .7, seed + 2)[..., None] * gran
    t1 = _noise(h, w, 2.2, seed + 3)
    fib = cv2.GaussianBlur(np.random.default_rng(seed + 4).normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=5, sigmaY=.6)
    fib /= fib.std() + 1e-6
    img = img * (1 + g + (t1 * .6 + fib * .5)[..., None] * tooth)
    return np.clip(img, 0, 1)


# ------------------------------------------------------------------------------------------------ palette
P = dict(
    sky_top=hexc("#7fb8de"), sky_mid=hexc("#a9d2e6"), sky_low=hexc("#e8ecd6"), cloud=hexc("#fbf7ee"), cloud_sh=hexc("#c9d3e3"),
    far_tree=hexc("#86a99a"), far_tree2=hexc("#9dbbaa"),
    roof1=hexc("#a8584a"), roof1_d=hexc("#7c3b36"), wall1=hexc("#ecbca6"), wall1_d=hexc("#c98f7f"), trim=hexc("#f6efe0"),
    roof2=hexc("#5f7e93"), roof2_d=hexc("#45606f"), wall2=hexc("#ece0a4"), wall2_d=hexc("#c5b77a"),
    win=hexc("#7c9fb8"), win_l=hexc("#c7dde6"), curtain=hexc("#f2d36a"),
    tree_cols=[hexc("#2f5a3a"), hexc("#3f7444"), hexc("#5a8f48"), hexc("#7fae55"), hexc("#a9c86a")],
    trunk=hexc("#6e4b3a"),
    fence=hexc("#d9a46a"), fence_l=hexc("#e9bf86"), fence_d=hexc("#a86e45"), fence_gap=hexc("#7a4c33"),
    lawn=hexc("#9ccb52"), lawn_l=hexc("#b2d863"), lawn_d=hexc("#86b542"), lawn_sh=hexc("#6f9a5c"),
    soil=hexc("#8a5a3e"), soil_d=hexc("#5e3a2a"), soil_l=hexc("#a87552"),
    bush_cols=[hexc("#2a4f36"), hexc("#386a40"), hexc("#4f8746"), hexc("#73a552")],
    flower=hexc("#f08a4b"), flower2=hexc("#f6d04d"), pole=hexc("#8a6a55"), wire=hexc("#4d4a55"),
)


# ------------------------------------------------------------------------------------------------ the backyard world
def backyard(cam=(640, 352, 1.0), holes=(), mounds=(), extra=None, seed=11, lawn_state="perfect", morning=False):
    """Paint the backyard card.  cam = (world cx, world cy, zoom).  holes/mounds: world (x, y, r) of dug lawn.
    extra(ctx) may paint additional set dressing in world coordinates."""
    cx, cy, z = cam
    surf, ctx = K.canvas()
    ctx.translate(W / 2, H / 2)
    ctx.scale(z, z)
    ctx.translate(-cx, -cy)
    X0, X1 = cx - W / 2 / z - 40, cx + W / 2 / z + 40

    # sky: graded wash with horizontal scrubbing
    g = cairo.LinearGradient(0, -200, 0, 360)
    if morning:
        g.add_color_stop_rgb(0, *hexc("#8fb9dd")[:3])
        g.add_color_stop_rgb(.55, *hexc("#e6c7c4")[:3])
        g.add_color_stop_rgb(1, *hexc("#f9d9a6")[:3])
    else:
        g.add_color_stop_rgb(0, *P["sky_top"][:3])
        g.add_color_stop_rgb(.55, *P["sky_mid"][:3])
        g.add_color_stop_rgb(1, *P["sky_low"][:3])
    ctx.rectangle(X0, -400, X1 - X0, 900)
    ctx.set_source(g)
    ctx.fill()
    r = np.random.default_rng(seed)
    ctx.save()
    ctx.rectangle(X0, -400, X1 - X0, 760)
    ctx.clip()
    for k in range(260):
        y = r.uniform(-300, 360)
        tcol = mix(hexc("#8fb9dd") if morning else P["sky_top"], hexc("#f9d9a6") if morning else P["sky_low"], np.clip((y + 200) / 560, 0, 1))
        ctx.set_source_rgba(*jcol(tcol, r, .035, .01), r.uniform(.15, .3))
        ctx.set_line_width(r.uniform(6, 16))
        x = r.uniform(X0, X1)
        L = r.uniform(60, 220)
        ctx.move_to(x, y)
        ctx.curve_to(x + L / 3, y + r.normal(0, 3), x + 2 * L / 3, y + r.normal(0, 3), x + L, y + r.normal(0, 2))
        ctx.stroke()
    ctx.restore()
    # clouds: soft stacked puffs with lilac-grey bellies
    for (ccx, ccy, s, sd) in ((170, 70, 1.0, 1), (760, 40, 1.25, 2), (1160, 95, .8, 3)):
        rr = np.random.default_rng(100 + sd)
        for layer, col, dy in ((0, P["cloud_sh"], 10), (1, P["cloud"], 0)):
            for q in range(9):
                px = ccx + s * rr.uniform(-120, 120)
                py = ccy + s * rr.uniform(-18, 14) + dy - s * 30 * math.exp(-((px - ccx) / (90 * s)) ** 2)
                rad = s * rr.uniform(26, 48)
                ctx.save()
                ctx.translate(px, py)
                ctx.scale(1.0, .62)
                ctx.arc(0, 0, rad, 0, 2 * math.pi)
                ctx.restore()
                ctx.set_source_rgba(*col[:3], .72 if layer else .55)
                ctx.fill()
    # telephone wires sagging across the sky
    for k, (y0, y1, sag) in enumerate(()):
        pts = [(x, y0 + (y1 - y0) * (x - X0) / (X1 - X0) + sag * math.sin(math.pi * (x - X0) / (X1 - X0))) for x in np.linspace(X0, X1, 12)]
        bline(ctx, pts, P["wire"], 1.5, 40 + k, wob=.3, alpha=.7)

    # far trees, cool and soft (atmospheric)
    for k, (tx, ty, rx, ry) in enumerate(((520, 260, 120, 70), (650, 250, 110, 80), (780, 268, 120, 60), (330, 270, 90, 50),
                                          (1180, 250, 110, 70), (-20, 255, 120, 70))):
        dabs(ctx, tx, ty, rx, ry, [P["far_tree"], P["far_tree2"], hexc("#b4cbb8")], 260, 200 + k, size=(7, 15), light=(.5, -.8),
             alpha=(.5, .8))

    # left neighbour's house: pink siding, gabled roof, a window with yellow curtains (a little wonky)
    def wall1(c):
        K.poly_path(c, np.array([(-40, 330), (-40, 175), (120, 120), (290, 168), (300, 330)]))
    region(ctx, wall1, P["wall1"], 21, [dict(n=500, lmin=30, lmax=90, wmin=4, wmax=9, ang=0.02, vj=.04)])
    for k, y in enumerate(range(190, 330, 16)):
        bline(ctx, [(-40, y + 2), (120, y - 1), (298, y + 3)], P["wall1_d"], 1.3, 300 + k, alpha=.45)

    def roof1(c):
        K.poly_path(c, np.array([(-70, 192), (118, 92), (322, 178), (312, 196), (118, 118), (-62, 212)]))
    region(ctx, roof1, P["roof1"], 22, [dict(n=320, lmin=20, lmax=60, wmin=3, wmax=7, ang=-.5, vj=.07)])
    bline(ctx, [(-62, 212), (118, 118), (312, 196)], P["roof1_d"], 2.2, 23, alpha=.6)

    def win1(c):
        K.poly_path(c, np.array([(78, 200), (152, 196), (156, 282), (80, 286)]))
    region(ctx, win1, P["win"], 24, [dict(n=120, lmin=10, lmax=40, wmin=3, wmax=8, ang=1.0, vj=.08, base=P["win_l"], alpha=(.15, .35))])
    for side in (0, 1):
        def cur(c, side=side):
            if side == 0:
                K.poly_path(c, np.array([(80, 200), (104, 199), (96, 240), (86, 284), (80, 286)]))
            else:
                K.poly_path(c, np.array([(128, 197), (152, 196), (156, 282), (150, 283), (138, 240)]))
        region(ctx, cur, P["curtain"], 25 + side, [dict(n=40, lmin=10, lmax=30, wmin=2, wmax=5, ang=1.5, vj=.06)])
    bline(ctx, [(78, 200), (152, 196), (156, 282), (80, 286), (78, 200)], P["wall1_d"], 2.4, 27, alpha=.8)
    bline(ctx, [(117, 198), (118, 284)], P["trim"], 3, 28, alpha=.9)

    # right: a neighbour roof behind a big round tree
    def wall2(c):
        K.poly_path(c, np.array([(840, 320), (846, 214), (1360, 206), (1360, 320)]))
    region(ctx, wall2, P["wall2"], 30, [dict(n=500, lmin=30, lmax=90, wmin=4, wmax=9, ang=0.0, vj=.04)])
    for k, y in enumerate(range(226, 320, 15)):
        bline(ctx, [(846, y), (1360, y - 3)], P["wall2_d"], 1.2, 340 + k, alpha=.45)
    def roof2(c):
        K.poly_path(c, np.array([(820, 222), (880, 150), (1380, 140), (1380, 214)]))
    region(ctx, roof2, P["roof2"], 31, [dict(n=300, lmin=20, lmax=70, wmin=3, wmax=8, ang=-0.02, vj=.06)])
    for k, y in enumerate(range(160, 216, 13)):
        bline(ctx, [(880 - (y - 150) * .85, y), (1380, y - 2)], P["roof2_d"], 1.2, 320 + k, alpha=.45)
    bline(ctx, [(820, 222), (1380, 214)], P["roof2_d"], 2.4, 329, alpha=.7)
    region(ctx, lambda c: K.poly_path(c, np.array([(1030, 330), (1036, 250), (1052, 250), (1060, 330)])), P["trunk"], 32,
           [dict(n=40, lmin=10, lmax=30, wmin=2, wmax=5, ang=1.57, vj=.08)])
    dabs(ctx, 1040, 160, 170, 120, P["tree_cols"], 1600, 33, size=(7, 16))
    dabs(ctx, 960, 215, 110, 70, P["tree_cols"], 600, 34, size=(7, 15))
    dabs(ctx, 1150, 210, 120, 80, P["tree_cols"], 700, 35, size=(7, 15))

    # the tall plank fence (the yard's back wall)
    top = lambda x: 300 + (x - 640) * .012  # noqa: E731  a hair off level, like a hand-drawn layout
    bot = lambda x: 446 + (x - 640) * .006  # noqa: E731
    pw = 46
    xs = np.arange(math.floor((X0 + 0) / pw) * pw - pw, X1 + pw, pw)
    for k, x in enumerate(xs):
        rr = np.random.default_rng(int(x) * 7 + 100003)
        lean = rr.normal(0, 1.6)
        t0, t1 = top(x) + rr.normal(0, 4), top(x + pw) + rr.normal(0, 4)
        pts = np.array([(x + lean, t0 + 8), (x + pw / 2 + lean, t0 - 6), (x + pw + lean, t1 + 8), (x + pw, bot(x + pw) + 4), (x, bot(x) + 4)])
        base = jcol(P["fence"], rr, .045, .012)
        region(ctx, lambda c, pts=pts: K.poly_path(c, pts), base + (1,), 400 + k,
               [dict(n=26, lmin=40, lmax=120, wmin=3, wmax=7, ang=1.57, angj=.04, vj=.06, curve=.05),
                dict(n=6, lmin=20, lmax=60, wmin=1, wmax=2, ang=1.57, angj=.08, vj=.15, base=P["fence_d"], alpha=(.25, .45))])
        bline(ctx, [pts[0], pts[4]], P["fence_gap"], 2.2, 500 + k, alpha=.75)
        if rr.uniform() < .35:  # a knot in the wood
            kx, ky = x + rr.uniform(12, pw - 12), rr.uniform(330, 420)
            ctx.save()
            ctx.translate(kx, ky)
            ctx.scale(4, 6)
            ctx.arc(0, 0, 1, 0, 2 * math.pi)
            ctx.restore()
            ctx.set_source_rgba(*P["fence_d"][:3], .55)
            ctx.fill()
    # lawn with mowing stripes fanning from a far vanishing point
    def lawn(c):
        K.poly_path(c, np.array([(X0 - 200, 440), (X1 + 200, 440), (X1 + 200, 1200), (X0 - 200, 1200)]))
    region(ctx, lawn, P["lawn"], 50)
    ctx.save()
    lawn(ctx)
    ctx.clip()
    r2 = np.random.default_rng(51)
    brush(ctx, P["lawn"], r2, 1800, 14, 46, 3, 7, 0.0, angj=.25, vj=.08, hj=.02, alpha=(.12, .28), curve=.3)
    brush(ctx, P["lawn_d"], r2, 500, 6, 16, 1.5, 3, -1.45, angj=.35, vj=.1, alpha=(.25, .5), curve=.4)  # grass flicks
    brush(ctx, P["lawn_l"], r2, 300, 6, 14, 1.5, 3, -1.6, angj=.35, vj=.06, alpha=(.25, .5), curve=.4)
    # dad's pride: crisp mowing stripes fanning from a far vanishing point, glazed over the scrub
    vp = (560, -700)
    for k in range(-16, 17):
        a0, a1 = k * .085, (k + 1) * .085
        pts = [vp] + [(vp[0] + 3000 * math.sin(a), vp[1] + 3000 * math.cos(a)) for a in (a0, a1)]
        K.poly_path(ctx, np.array(pts))
        ctx.set_source_rgba(*(P["lawn_l"] if k % 2 == 0 else P["lawn_d"])[:3], .38)
        ctx.fill()
    # big soft value patches so the lawn doesn't read as one flat sheet
    for k in range(14):
        px, py = r2.uniform(X0, X1), r2.uniform(470, 720)
        ctx.save(); ctx.translate(px, py); ctx.scale(r2.uniform(90, 220), r2.uniform(20, 45)); ctx.arc(0, 0, 1, 0, 2 * math.pi); ctx.restore()
        ctx.set_source_rgba(*(P["lawn_d"] if k % 2 else P["lawn_l"])[:3], .16)
        ctx.fill()
    # foreground darkens toward the bottom of the card
    g3 = cairo.LinearGradient(0, 560, 0, 760)
    g3.add_color_stop_rgba(0, *P["lawn_sh"][:3], 0)
    g3.add_color_stop_rgba(1, *P["lawn_sh"][:3], .35)
    ctx.rectangle(X0 - 200, 560, X1 - X0 + 400, 300)
    ctx.set_source(g3)
    ctx.fill()
    # soft cool shadow along the fence foot
    g2 = cairo.LinearGradient(0, 440, 0, 500)
    g2.add_color_stop_rgba(0, *P["lawn_sh"][:3], .85)
    g2.add_color_stop_rgba(1, *P["lawn_sh"][:3], 0)
    ctx.rectangle(X0 - 200, 436, X1 - X0 + 400, 70)
    ctx.set_source(g2)
    ctx.fill()
    ctx.restore()
    # bushes in the fence corners and a few marigolds
    for k, (bx, by, brx, bry) in enumerate(((-30, 430, 150, 70), (150, 445, 70, 40), (1270, 430, 160, 80), (1120, 448, 70, 36))):
        dabs(ctx, bx, by, brx, bry, P["bush_cols"], int(brx * bry / 14), 60 + k, size=(6, 13), light=(.4, -.9))
    rf = np.random.default_rng(70)
    for k in range(26):
        fx = rf.choice([rf.uniform(-60, 230), rf.uniform(1080, 1340)])
        fy = rf.uniform(436, 470)
        ctx.arc(fx, fy, rf.uniform(3.5, 6), 0, 2 * math.pi)
        ctx.set_source_rgba(*(P["flower"] if k % 3 else P["flower2"])[:3], .9)
        ctx.fill()

    # dug-up lawn: holes and dirt mounds, painted
    for k, (hx, hy, hr) in enumerate(holes):
        region(ctx, lambda c, hx=hx, hy=hy, hr=hr: (c.save(), c.translate(hx, hy), c.scale(hr, hr * .38), c.arc(0, 0, 1, 0, 2 * math.pi),
                                                    c.restore()), P["soil_l"], 80 + k)
        region(ctx, lambda c, hx=hx, hy=hy, hr=hr: (c.save(), c.translate(hx, hy + hr * .05), c.scale(hr * .86, hr * .3),
                                                    c.arc(0, 0, 1, 0, 2 * math.pi), c.restore()), P["soil_d"], 90 + k,
               [dict(n=40, lmin=8, lmax=24, wmin=2, wmax=5, ang=0, vj=.1)])
    for k, (mx, my, mr) in enumerate(mounds):
        def mound(c, mx=mx, my=my, mr=mr):
            c.move_to(mx - mr, my)
            c.curve_to(mx - mr * .6, my - mr * .75, mx + mr * .5, my - mr * .8, mx + mr, my)
            c.close_path()
        region(ctx, mound, P["soil"], 95 + k, [dict(n=int(mr * 2), lmin=6, lmax=20, wmin=2, wmax=6, ang=-.3, angj=.8, vj=.12,
                                                     base=P["soil_l"])])
    if lawn_state == "crater":
        crater(ctx)
    elif lawn_state == "wrecked":
        wrecked(ctx)
    if extra is not None:
        extra(ctx)
    img = to_rgba(surf)[..., :3]
    if morning:  # low warm sun: a peach glaze, cooler shadows
        img = img * np.array([1.04, .97, .88], np.float32) + np.array([.02, .0, -.01], np.float32)
    return finish_paint(img, seed=seed)


CRATER = (840, 606, 252, 90)  # world cx, cy, rx, ry
ISLAND = (905, 642, 36, 12)


def crater(ctx):
    cx, cy, rx, ry = CRATER
    r = np.random.default_rng(300)
    # thrown-up dirt piled round the back rim
    for k in range(9):
        a = math.pi + k * math.pi / 8
        mx, my = cx + (rx + 10) * math.cos(a), cy + (ry + 6) * math.sin(a) + 6
        mr = r.uniform(40, 70)
        def mound(c, mx=mx, my=my, mr=mr):
            c.move_to(mx - mr, my)
            c.curve_to(mx - mr * .6, my - mr * .7, mx + mr * .5, my - mr * .75, mx + mr, my)
            c.close_path()
        region(ctx, mound, P["soil"], 310 + k, [dict(n=int(mr * 2), lmin=6, lmax=20, wmin=2, wmax=6, ang=-.3, angj=.8, vj=.12, base=P["soil_l"])])
    # the pit: far inner wall lit, depths dark
    def pit(c):
        c.save(); c.translate(cx, cy); c.scale(rx, ry); c.arc(0, 0, 1, 0, 2 * math.pi); c.restore()
    region(ctx, pit, P["soil_d"], 320)
    ctx.save()
    pit(ctx)
    ctx.clip()
    g = cairo.LinearGradient(0, cy - ry, 0, cy + ry)
    g.add_color_stop_rgba(0, *P["soil_l"][:3], .95)
    g.add_color_stop_rgba(.45, *P["soil"][:3], .8)
    g.add_color_stop_rgba(1, *hexc("#2e1c14")[:3], .95)
    ctx.rectangle(cx - rx, cy - ry, 2 * rx, 2 * ry)
    ctx.set_source(g)
    ctx.fill()
    brush(ctx, P["soil_d"], r, 160, 8, 30, 2, 5, 1.4, angj=.5, vj=.15, alpha=(.2, .4))
    ctx.restore()
    bline(ctx, [(cx - rx, cy), (cx - rx * .7, cy - ry * .72), (cx, cy - ry), (cx + rx * .7, cy - ry * .72), (cx + rx, cy)], P["soil_d"], 2.4, 321, alpha=.7)
    # the island the dandelion stands on: a grass cap on a dirt column dropping into the dark
    ix, iy, irx, iry = ISLAND
    def col(c):
        K.poly_path(c, np.array([(ix - irx, iy), (ix + irx, iy), (ix + irx - 4, cy + ry + 40), (ix - irx + 4, cy + ry + 40)]))
    region(ctx, col, P["soil"], 330, [dict(n=60, lmin=8, lmax=24, wmin=2, wmax=5, ang=1.57, angj=.3, vj=.12, base=P["soil_l"])])
    def cap(c):
        c.save(); c.translate(ix, iy); c.scale(irx, iry); c.arc(0, 0, 1, 0, 2 * math.pi); c.restore()
    region(ctx, cap, P["lawn"], 331, [dict(n=30, lmin=4, lmax=10, wmin=1.5, wmax=3, ang=-1.5, angj=.4, vj=.1, base=P["lawn_d"])])
    bline(ctx, [(ix - irx, iy), (ix - irx + 3, cy + ry + 30)], P["soil_d"], 2.0, 332, alpha=.7)
    bline(ctx, [(ix + irx, iy), (ix + irx - 3, cy + ry + 30)], P["soil_d"], 2.0, 333, alpha=.7)


def wrecked(ctx):
    """After the vacuum bag blew: the pit slopped full again, lumps and clods everywhere, bald patches."""
    cx, cy, rx, ry = CRATER
    r = np.random.default_rng(400)
    def patch(c, x, y, a, b, rr):
        pts = [(x + a * (1 + .18 * rr.normal()) * math.cos(t), y + b * (1 + .18 * rr.normal()) * math.sin(t)) for t in np.linspace(0, 6.28, 14, endpoint=False)]
        K.poly_path(c, K.crs(np.array(pts), True, 6))
    for k, (x, y, a, b) in enumerate(((cx, cy + 10, rx * .95, ry * .8), (330, 560, 90, 26), (600, 680, 140, 34), (1120, 600, 80, 22),
                                       (420, 650, 60, 18), (1010, 690, 100, 26))):
        rr = np.random.default_rng(410 + k)
        region(ctx, lambda c, x=x, y=y, a=a, b=b, rr=rr: patch(c, x, y, a, b, np.random.default_rng(410 + k)), P["soil"], 420 + k,
               [dict(n=int(a * b / 30), lmin=6, lmax=20, wmin=2, wmax=6, ang=0, angj=.8, vj=.12, base=P["soil_l"])])
    for k in range(10):
        mx, my, mr = r.uniform(250, 1180), r.uniform(520, 700), r.uniform(22, 46)
        def mound(c, mx=mx, my=my, mr=mr):
            c.move_to(mx - mr, my)
            c.curve_to(mx - mr * .6, my - mr * .7, mx + mr * .5, my - mr * .75, mx + mr, my)
            c.close_path()
        region(ctx, mound, P["soil"], 440 + k, [dict(n=int(mr * 2), lmin=6, lmax=18, wmin=2, wmax=5, ang=-.3, angj=.8, vj=.12, base=P["soil_l"])])
    for k in range(260):  # flung clods, even on the fence
        x, y = r.uniform(150, 1280), r.uniform(300, 704)
        ctx.save(); ctx.translate(x, y); ctx.rotate(r.uniform(0, 3)); ctx.scale(r.uniform(2, 6), r.uniform(1.5, 4)); ctx.arc(0, 0, 1, 0, 6.3); ctx.restore()
        ctx.set_source_rgba(*jcol(P["soil_d"], r, .1, .02), r.uniform(.6, .9))
        ctx.fill()
    ix, iy, irx, iry = ISLAND
    def tuft(c):
        c.save(); c.translate(ix, iy); c.scale(irx * 1.2, iry * 1.3); c.arc(0, 0, 1, 0, 2 * math.pi); c.restore()
    region(ctx, tuft, P["lawn"], 460, [dict(n=30, lmin=4, lmax=10, wmin=1.5, wmax=3, ang=-1.5, angj=.4, vj=.1, base=P["lawn_d"])])
