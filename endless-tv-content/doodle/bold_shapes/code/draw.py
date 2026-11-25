"""bold_shapes: a late-90s 'bold shapes' TV-cartoon look (UPA-descended), drawn entirely in code.

Original character: SQUEAKY DOYLE, the smallest, bravest window-washer in Sky-High City, dangling from a bosun's chair
high above a mint-and-teal retro-modern skyline, with a squeegee twice his size.

Two layers per frame:
  * background card: flat colour fields, coloured (not black) thin linework, limited mint/teal/lilac palette, a hint of
    gouache texture; rendered once and cached.
  * cel layer: the rig and the character, drawn as a 'cel' in two passes -- a heavy ink silhouette pass first, then each
    part filled and outlined with a lighter interior line -- so the outer contour is thick and the inner lines lighter.
"""
import math

import cairo
import cv2
import numpy as np

W, H, FPS, SS = 1280, 720, 24, 2
DUR = 5.0
N = int(DUR * FPS)
BASE = cairo.Matrix(SS, 0, 0, SS, 0, 0)


def hexc(h, a=1.0):
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, a)


INK = hexc("#17121c")
WHITE = hexc("#ffffff")
P = {k: hexc(v) for k, v in dict(
    sky="#b3e2d5", sun="#f8f0c6", cloud="#f3f8ea", cloud_ln="#83c2b2",
    far="#9dd3c5", far_win="#8cc6b8", far2="#a9d8cc",
    mid1="#69b7a6", mid2="#549f91", mid_ln="#3a8576", mid_win="#468f81", peri="#8f93d9", peri_ln="#5d61ac", peri_win="#7378c4",
    low="#438c7e", low_ln="#2f6e62", low_win="#3a7f72",
    coral="#f48877", coral_ln="#bd5a50", cream="#fbf0d4",
    fac="#b2aeec", fac_dark="#9692dc", fac_ln="#6b67b6", glass="#d4f1ee", glass2="#c0e6e6",
    skin="#ffe3d0", tee="#fbfbf6", hat="#ffd43c", suit="#ff8a2a", boot="#e8343c", pole="#2f78ea", iris="#2e86f2", steel="#cbd4dd",
    steel_dk="#9aa6b3", rubber="#2a2630", wood="#e3a75e", wood_ln="#a2672f", rope="#f6e8c3", bucket="#e8343c",
    bucket_in="#8f1622", suds="#ffffff", mouth="#7c1631", tongue="#ff8193", burst="#5a49d4", burst2="#6f60ea").items()}

ORIGIN = np.array([560.0, 530.0])  # world position of the plank-seat centre (rest)
PIVOT = np.array([560.0, -320.0])  # the rope hangs from far above the frame: the rig sways around here
RING = np.array([0.0, -452.0])  # local (seat-relative) position of the strap ring above his hat


# ================================================================================================== path helpers
def cr_path(ctx, pts, closed=False, tension=1.0):
    p = [np.asarray(q, float) for q in pts]
    n = len(p)
    get = (lambda i: p[i % n]) if closed else (lambda i: p[min(max(i, 0), n - 1)])
    ctx.move_to(*p[0])
    for i in range(n if closed else n - 1):
        p0, p1, p2, p3 = get(i - 1), get(i), get(i + 1), get(i + 2)
        c1, c2 = p1 + (p2 - p0) * tension / 6, p2 - (p3 - p1) * tension / 6
        ctx.curve_to(c1[0], c1[1], c2[0], c2[1], p2[0], p2[1])
    if closed:
        ctx.close_path()


def poly(ctx, pts, closed=True):
    ctx.move_to(*pts[0])
    for q in pts[1:]:
        ctx.line_to(*q)
    if closed:
        ctx.close_path()


def ellipse(ctx, cx, cy, rx, ry, rot=0.0):
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(rot)
    ctx.scale(max(rx, 1e-3), max(ry, 1e-3))
    ctx.new_sub_path()
    ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.restore()


def superellipse(ctx, cx, cy, rx, ry, n=2.5, squash_bottom=1.0, k=72):
    """A 'bold shape' head: between an ellipse and a rounded box; squash_bottom > 1 broadens the lower half."""
    pts = []
    for j in range(k):
        a = 2 * math.pi * j / k
        c, s_ = math.cos(a), math.sin(a)
        x = rx * math.copysign(abs(c) ** (2 / n), c)
        y = ry * math.copysign(abs(s_) ** (2 / n), s_)
        if y > 0:
            x *= 1 + (squash_bottom - 1) * (y / ry)
        pts.append((cx + x, cy + y))
    poly(ctx, pts)


def rrect(ctx, x0, y0, x1, y1, r):
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    ctx.new_sub_path()
    ctx.arc(x1 - r, y0 + r, r, -math.pi / 2, 0)
    ctx.arc(x1 - r, y1 - r, r, 0, math.pi / 2)
    ctx.arc(x0 + r, y1 - r, r, math.pi / 2, math.pi)
    ctx.arc(x0 + r, y0 + r, r, math.pi, 1.5 * math.pi)
    ctx.close_path()


def fill(ctx, c, preserve=False):
    ctx.set_source_rgba(*c)
    ctx.fill_preserve() if preserve else ctx.fill()


def stroke(ctx, c, w, preserve=False):
    ctx.set_source_rgba(*c)
    ctx.set_line_width(w)
    ctx.stroke_preserve() if preserve else ctx.stroke()


def lerp(a, b, u):
    return a + (b - a) * u


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def rot(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


# ================================================================================================== the cel system
class Cel:
    """Collects the parts of one character cel, then inks it in two passes (heavy silhouette, then fills + lighter lines).
    Each part keeps the transform it was drawn under, but every line is stroked in screen space so weights stay constant."""

    def __init__(self, ctx, outer=6.0, inner=3.6):
        self.ctx, self.outer, self.inner, self.ops = ctx, outer, inner, []

    def shape(self, fn, color, sil=True, line=None):
        self.ops.append(("shape", self.ctx.get_matrix(), fn, color, sil, line))

    def tube(self, pts, w, color, sil=True):
        m = self.ctx.get_matrix()
        dev = [np.array(m.transform_point(*q)) / SS for q in pts]
        sc = math.sqrt(abs(m.xx * m.yy - m.xy * m.yx)) / SS
        self.ops.append(("tube", None, dev, color, sil, w * sc))

    def line(self, fn, w, color=INK):
        self.ops.append(("line", self.ctx.get_matrix(), fn, color, False, w))

    def raw(self, fn):
        """Arbitrary drawing (clips, highlights) in the part's transform, in pass 2 order."""
        self.ops.append(("raw", self.ctx.get_matrix(), fn, None, False, None))

    def _path(self, op):
        ctx = self.ctx
        if op[0] == "tube":
            pts = op[2]
            ctx.set_matrix(BASE)
            cr_path(ctx, pts, tension=.9) if len(pts) > 2 else poly(ctx, pts, closed=False)
        else:
            ctx.set_matrix(op[1])
            op[2](ctx)
            ctx.set_matrix(BASE)

    def _path_keep(self, op):
        """Like _path but keeps the current translation (used for the weighted second ink pass)."""
        ctx = self.ctx
        m0 = ctx.get_matrix()
        if op[0] == "tube":
            cr_path(ctx, op[2], tension=.9) if len(op[2]) > 2 else poly(ctx, op[2], closed=False)
        else:
            m = cairo.Matrix(*op[1]) * cairo.Matrix(1, 0, 0, 1, m0.x0 - BASE.x0, m0.y0 - BASE.y0)
            ctx.set_matrix(m)
            op[2](ctx)
            ctx.set_matrix(m0)

    def draw(self):
        ctx = self.ctx
        ctx.save()
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        for op in self.ops:  # pass 1: the silhouette contour
            if not op[4]:
                continue
            w = self.outer * 2 + (op[5] if op[0] == "tube" else 0)
            self._path(op)
            stroke(ctx, INK, w)
            ctx.save()
            ctx.translate(1.1, 1.6)
            self._path_keep(op)
            stroke(ctx, INK, w)
            ctx.restore()
        for op in self.ops:  # pass 2: fills and interior lines, in order
            kind = op[0]
            if kind == "shape":
                self._path(op)
                fill(ctx, op[3], True)
                stroke(ctx, INK, op[5] or self.inner)
            elif kind == "tube":
                self._path(op)
                stroke(ctx, INK, op[5] + self.inner * 2, True)
                stroke(ctx, op[3], op[5])
            elif kind == "line":
                self._path(op)
                stroke(ctx, op[3], op[5])
            else:
                ctx.save()
                ctx.set_matrix(op[1])
                op[2](ctx)
                ctx.restore()
        ctx.restore()
        ctx.set_matrix(BASE)


# ================================================================================================== background card
def cloud(ctx, cx, cy, w, h, ln):
    """A flat-bottomed UPA cloud: a row of round bumps on a straight base, outlined as one shape."""
    bumps = [(-.34, .42, .16), (-.12, .74, .2), (.13, .95, .21), (.35, .58, .16)]

    def shapes():
        rrect(ctx, cx - w / 2, cy - h * .32, cx + w / 2, cy, h * .16)
        for bx, bh, r in bumps:
            ellipse(ctx, cx + bx * w, cy - h * bh + r * w * .55, r * w, r * w)
    shapes()
    stroke(ctx, ln, 4.4)
    shapes()
    fill(ctx, P["cloud"])


def windows(ctx, x0, x1, y0, y1, ww, wh, gx, gy, col, skip=None, rng=None):
    cols = int((x1 - x0 - gx) // (ww + gx))
    if cols <= 0:
        return
    off = (x1 - x0 - cols * (ww + gx) + gx) / 2
    y = y0
    while y + wh < y1:
        for c in range(cols):
            if rng is not None and skip and rng.uniform() < skip:
                continue
            ctx.rectangle(x0 + off + c * (ww + gx), y, ww, wh)
        y += wh + gy
    fill(ctx, col)


def building(ctx, x0, x1, top, kind, col, ln, win, rng, bottom=730, lw=2.0):
    """One mid-ground building: a flat shape, thin coloured outline, tidy rows of windows, a retro-modern roof."""
    if kind == "dome":
        ctx.move_to(x0, bottom)
        ctx.line_to(x0, top + (x1 - x0) / 2)
        ctx.arc(x0 + (x1 - x0) / 2, top + (x1 - x0) / 2, (x1 - x0) / 2, math.pi, 2 * math.pi)
        ctx.line_to(x1, bottom)
        ctx.close_path()
        fill(ctx, col, True)
        stroke(ctx, ln, lw)
        r = (x1 - x0) / 2
        for k in range(1, 3):  # ribs on the dome
            ctx.new_sub_path()
            ctx.arc(x0 + r, top + r, r * (1 - .32 * k), math.pi, 2 * math.pi)
            stroke(ctx, ln, 1.6)
        windows(ctx, x0 + 8, x1 - 8, top + r + 16, bottom, 7, 26, 9, 14, win)
        return
    if kind == "step":
        w = x1 - x0
        tiers = [(x0, x1, top + 60), (x0 + w * .15, x1 - w * .15, top + 26), (x0 + w * .32, x1 - w * .32, top)]
        for a, b, t in tiers:
            ctx.rectangle(a, t, b - a, bottom - t)
            fill(ctx, col, True)
            stroke(ctx, ln, lw)
        windows(ctx, x0 + 6, x1 - 6, top + 74, bottom, 9, 12, 12, 18, win, .3, rng)
        poly(ctx, [(x0 + w / 2, top), (x0 + w / 2, top - 46)], closed=False)
        stroke(ctx, ln, 2.4)
        ellipse(ctx, x0 + w / 2, top - 48, 4.5, 4.5)
        fill(ctx, P["coral"], True)
        stroke(ctx, ln, 1.4)
        return
    if kind == "ribbon":  # modernist slab with long ribbon windows
        ctx.rectangle(x0, top, x1 - x0, bottom - top)
        fill(ctx, col, True)
        stroke(ctx, ln, lw)
        y = top + 12
        while y < bottom:
            ctx.rectangle(x0 + 6, y, x1 - x0 - 12, 7)
            y += 18
        fill(ctx, win)
        return
    if kind == "tank":  # flat roof with a water tower
        ctx.rectangle(x0, top, x1 - x0, bottom - top)
        fill(ctx, col, True)
        stroke(ctx, ln, lw)
        windows(ctx, x0 + 6, x1 - 6, top + 14, bottom, 10, 13, 12, 18, win, .3, rng)
        tx = x0 + (x1 - x0) * .62
        for lx in (-14, 14):
            poly(ctx, [(tx + lx * .6, top - 22), (tx + lx, top)], closed=False)
            stroke(ctx, ln, 2.2)
        rrect(ctx, tx - 18, top - 52, tx + 18, top - 20, 6)
        fill(ctx, P["coral"], True)
        stroke(ctx, P["coral_ln"], 1.8)
        poly(ctx, [(tx - 22, top - 52), (tx, top - 66), (tx + 22, top - 52)])
        fill(ctx, P["coral"], True)
        stroke(ctx, P["coral_ln"], 1.8)
        return
    if kind == "star":  # tall slab with an atomic starburst on a mast
        ctx.rectangle(x0, top, x1 - x0, bottom - top)
        fill(ctx, col, True)
        stroke(ctx, ln, lw)
        windows(ctx, x0 + 7, x1 - 7, top + 14, bottom, 9, 13, 11, 17, win, .25, rng)
        cx = (x0 + x1) / 2
        poly(ctx, [(cx, top), (cx, top - 60)], closed=False)
        stroke(ctx, ln, 2.4)
        for k in range(8):
            a = k * math.pi / 4 + .2
            L = 22 if k % 2 == 0 else 14
            poly(ctx, [(cx, top - 72), (cx + L * math.cos(a), top - 72 + L * math.sin(a))], closed=False)
            stroke(ctx, P["coral_ln"], 2.2)
            ellipse(ctx, cx + L * math.cos(a), top - 72 + L * math.sin(a), 3.2, 3.2)
            fill(ctx, P["coral"])
        ellipse(ctx, cx, top - 72, 6, 6)
        fill(ctx, P["cream"], True)
        stroke(ctx, P["coral_ln"], 1.8)
        return
    # plain flat-roof slab, slightly skewed the UPA way
    sk = rng.uniform(-5, 5)
    poly(ctx, [(x0, bottom), (x0 + sk, top), (x1 + sk, top), (x1, bottom)])
    fill(ctx, col, True)
    stroke(ctx, ln, lw)
    windows(ctx, x0 + 6 + max(sk, 0), x1 - 6 + min(sk, 0), top + 14, bottom, 9, 12, 12, 18, win, .3, rng)


def googie_tower(ctx, x, top):
    """The skyline's landmark: a needle tower with a flying-saucer top."""
    ln = P["peri_ln"]
    poly(ctx, [(x - 22, 730), (x - 7, top + 40), (x + 7, top + 40), (x + 22, 730)])
    fill(ctx, P["peri"], True)
    stroke(ctx, ln, 2.0)
    for yy in np.arange(top + 90, 720, 38):  # horizontal bands down the shaft
        f = (yy - top - 40) / (730 - top - 40)
        hw = 7 + 15 * f
        poly(ctx, [(x - hw, yy), (x + hw, yy)], closed=False)
        stroke(ctx, ln, 1.6)
    ellipse(ctx, x, top + 46, 30, 7)  # lower ring
    fill(ctx, P["cream"], True)
    stroke(ctx, P["coral_ln"], 1.8)
    ellipse(ctx, x, top + 22, 74, 15)  # the saucer
    fill(ctx, P["coral"], True)
    stroke(ctx, P["coral_ln"], 2.0)
    ctx.new_sub_path()
    ctx.arc(x, top + 18, 34, math.pi, 2 * math.pi)  # glass dome
    ctx.close_path()
    fill(ctx, P["cream"], True)
    stroke(ctx, P["coral_ln"], 2.0)
    for k in range(-3, 4):  # portholes on the saucer rim
        ellipse(ctx, x + k * 17, top + 25, 3.4, 2.6)
    fill(ctx, P["coral_ln"])
    poly(ctx, [(x, top - 16), (x, top - 58)], closed=False)
    stroke(ctx, ln, 2.4)
    ellipse(ctx, x, top - 62, 6, 6)
    fill(ctx, P["coral"], True)
    stroke(ctx, P["coral_ln"], 1.6)


def facade(ctx):
    """The skyscraper he is cleaning: a flat elevation in lilac with shiny glass panes."""
    x0 = 905
    ctx.rectangle(x0, -20, W + 40 - x0, H + 40)
    fill(ctx, P["fac"])
    ctx.rectangle(x0, -20, 32, H + 40)  # the corner pier
    fill(ctx, P["fac_dark"], True)
    stroke(ctx, P["fac_ln"], 2.2)
    cols = [952, 1094, 1236]
    rows = [-70, 118, 306, 494, 682]
    for wy in rows:
        ctx.rectangle(x0 + 32, wy + 156, W + 40, 10)  # sill band under each row
        fill(ctx, P["fac_dark"], True)
        stroke(ctx, P["fac_ln"], 1.8)
        for wx in cols:
            ctx.rectangle(wx, wy, 120, 150)
            fill(ctx, P["glass"], True)
            stroke(ctx, P["fac_ln"], 2.2)
            ctx.save()
            ctx.rectangle(wx, wy, 120, 150)
            ctx.clip()
            poly(ctx, [(wx - 10, wy + 150), (wx + 120, wy + 40), (wx + 130, wy + 160)])  # reflection band, lower right
            fill(ctx, P["glass2"])
            for k, (a, b) in enumerate(((18, 30), (40, 12))):  # two white shine streaks, upper left
                s0 = a + k * 6
                poly(ctx, [(wx + s0, wy + 82), (wx + s0 + 64, wy + 18), (wx + s0 + 64 + b, wy + 18), (wx + s0 + b, wy + 82)])
                fill(ctx, hexc("#ffffff", .85))
            ctx.restore()
            poly(ctx, [(wx + 60, wy), (wx + 60, wy + 150)], closed=False)  # mullion
            stroke(ctx, P["fac_ln"], 2.0)


_BG = None


def background():
    """The static painted background card at 1x as float RGB (cached)."""
    global _BG
    if _BG is not None:
        return _BG
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * SS, H * SS)
    ctx = cairo.Context(surf)
    ctx.set_matrix(BASE)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    rng = np.random.default_rng(4)
    ctx.rectangle(-10, -10, W + 20, H + 20)
    fill(ctx, P["sky"])
    ellipse(ctx, 196, 150, 78, 78)  # the sun, a plain cream disc with one soft ring
    fill(ctx, P["sun"])
    ctx.new_sub_path()
    ctx.arc(196, 150, 96, 0, 2 * math.pi)
    stroke(ctx, hexc("#f8f0c6", .55), 6)
    cloud(ctx, 120, 330, 190, 64, P["cloud_ln"])
    cloud(ctx, 790, 120, 210, 70, P["cloud_ln"])
    cloud(ctx, 360, 82, 120, 40, P["cloud_ln"])
    # far skyline: pale, no lines, near the sky value
    x = -10
    rngf = np.random.default_rng(11)
    while x < 920:
        w = rngf.uniform(36, 80)
        top = rngf.uniform(440, 505)
        kind = rngf.integers(0, 4)
        col = P["far"] if rngf.uniform() < .6 else P["far2"]
        if kind == 0:
            ctx.rectangle(x, top + w / 2, w, 800)
            ctx.new_sub_path()
            ctx.arc(x + w / 2, top + w / 2, w / 2, math.pi, 2 * math.pi)
        elif kind == 1:
            poly(ctx, [(x, 800), (x, top + 20), (x + w * .3, top + 20), (x + w * .3, top), (x + w * .7, top), (x + w * .7, top + 20),
                       (x + w, top + 20), (x + w, 800)])
        else:
            ctx.rectangle(x, top, w, 800)
        fill(ctx, col)
        if kind != 0 and rngf.uniform() < .6:
            windows(ctx, x + 5, x + w - 5, top + 26, 720, 6, 8, 10, 14, P["far_win"], .4, rngf)
        x += w + rngf.uniform(-6, 8)
    googie_tower(ctx, 236, 226)
    mids = [(-12, 92, 392, "tank", "mid2"), (84, 170, 452, "step", "peri"), (268, 350, 372, "dome", "mid1"),
            (346, 432, 456, "flat", "peri"), (430, 520, 420, "ribbon", "mid2"), (512, 602, 392, "flat", "mid1"),
            (596, 690, 448, "step", "peri"), (684, 760, 412, "flat", "mid2"), (752, 832, 352, "star", "peri"),
            (826, 912, 436, "ribbon", "mid1")]
    for x0, x1, top, kind, c in mids:
        if c == "peri":
            building(ctx, x0, x1, top, kind, P["peri"], P["peri_ln"], hexc("#7e83cd"), rng)
        else:
            building(ctx, x0, x1, top, kind, P[c], P["mid_ln"], hexc("#57a493") if c == "mid1" else hexc("#4a9285"), rng)
    lows = [(-12, 120, 600), (112, 210, 640), (204, 300, 612), (292, 400, 652), (392, 520, 628), (512, 620, 660), (612, 720, 618),
            (712, 830, 646), (822, 912, 606)]
    for x0, x1, top in lows:  # the nearest, darkest rooftops along the bottom
        sk = rng.uniform(-4, 4)
        poly(ctx, [(x0, 740), (x0 + sk, top), (x1 + sk, top), (x1, 740)])
        fill(ctx, P["low"], True)
        stroke(ctx, P["low_ln"], 2.0)
        windows(ctx, x0 + 8, x1 - 8, top + 14, 740, 12, 14, 16, 20, P["low_win"], .35, rng)
    facade(ctx)
    a = np.ndarray((H * SS, W * SS, 4), np.uint8, surf.get_data())[:, :, :3][:, :, ::-1].astype(np.float32) / 255
    img = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
    # a whisper of gouache: low-frequency unevenness plus a faint dry-brush streak, fixed like a painted card
    r = np.random.default_rng(7)
    lo = cv2.resize(r.normal(0, 1, (H // 24, W // 24)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    st = cv2.GaussianBlur(r.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigmaX=9, sigmaY=.8)
    fine = cv2.GaussianBlur(r.normal(0, 1, (H, W)).astype(np.float32), (0, 0), .7)
    img = img * (1 + (lo * .008 + st * .022 + fine * .007)[..., None])
    _BG = np.clip(img, 0, 1)
    return _BG


# ================================================================================================== the character
REST = dict(sway=0.0, hop=0.0, sq=1.0, lean=0.0, head_dx=0.0, head_dy=0.0, head_tilt=0.0, turn=.22, look=(0.12, 0.05),
            blink=0.0, lid=0.0, happy=0.0, closed_happy=0.0, mouth="smile", mopen=0.0,
            handL=(-124.0, -100.0), elbowL=(-88.0, -78.0),
            grip=(130.0, -96.0), elbowR=(92.0, -72.0), sq_head=(402.0, -287.0),
            legL=0.0, legR=0.0, smear=None)
_d = np.array(REST["sq_head"]) - np.array(REST["grip"])
REST["blade"] = math.atan2(_d[0], -_d[1])  # at rest the blade sits square to the pole


def pose_mix(a, b, u):
    out = {}
    for k, v in a.items():
        w = b.get(k, v)
        if isinstance(v, tuple):
            out[k] = tuple(lerp(x, y, u) for x, y in zip(v, w))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = lerp(v, w, u)
        else:
            out[k] = w if u >= .5 else v
    return out


def eye(ctx, cx, cy, rx, ry, p, side):
    """A huge cartoon eye: white, a big blue iris, black pupil, two white catch-lights, heavy upper lid."""
    lx, ly = p["look"]
    if p["closed_happy"] > .5:  # happy closed eyes: an upward arc
        cr_path(ctx, [(cx - rx * .8, cy + ry * .15), (cx, cy - ry * .45), (cx + rx * .8, cy + ry * .15)], tension=1.0)
        stroke(ctx, INK, 7)
        return
    if p["blink"] > .7:  # a blink: the lid line
        cr_path(ctx, [(cx - rx * .95, cy - ry * .05), (cx, cy + ry * .28), (cx + rx * .95, cy - ry * .05)], tension=1.0)
        stroke(ctx, INK, 7)
        return
    ellipse(ctx, cx, cy, rx, ry)
    fill(ctx, WHITE)
    ctx.save()
    ellipse(ctx, cx, cy, rx, ry)
    ctx.clip()
    ix, iy = cx + lx * rx * .36, cy + ly * ry * .28 + ry * .14
    ir = rx * .76
    ellipse(ctx, ix, iy, ir, ir * 1.12)  # one flat iris colour, no gradient
    fill(ctx, P["iris"], True)
    stroke(ctx, INK, 3.0)
    ellipse(ctx, ix, iy + ir * .04, ir * .44, ir * .5)
    fill(ctx, INK)
    ellipse(ctx, ix + ir * .34, iy - ir * .34, ir * .3, ir * .34, .35)  # big catch-light
    fill(ctx, WHITE)
    ellipse(ctx, ix - ir * .36, iy + ir * .44, ir * .13, ir * .13)  # small catch-light
    fill(ctx, WHITE)
    lid = max(p["lid"], p["blink"] * 1.2)
    if lid > 0:  # lowered upper lid: skin from the top down to a slightly curved line
        ly0 = cy - ry + 2 * ry * lid * .62
        poly(ctx, [(cx - rx - 4, cy - ry - 4), (cx + rx + 4, cy - ry - 4), (cx + rx + 4, ly0 - 6 * side), (cx - rx - 4, ly0 + 6 * side)])
        fill(ctx, P["skin"])
        poly(ctx, [(cx - rx - 4, ly0 + 6 * side), (cx + rx + 4, ly0 - 6 * side)], closed=False)
        stroke(ctx, INK, 6)
    if p["happy"] > 0:  # happy squint: the cheek pushes up from below
        hy = cy + ry * (1.05 - .75 * p["happy"])
        cr_path(ctx, [(cx - rx * 1.2, cy + ry * 1.4), (cx - rx * 1.2, hy + ry * .2), (cx, hy - ry * .08), (cx + rx * 1.2, hy + ry * .2),
                      (cx + rx * 1.2, cy + ry * 1.4)], closed=True, tension=.8)
        fill(ctx, P["skin"], True)
        stroke(ctx, INK, 4.5)
    ctx.restore()
    ellipse(ctx, cx, cy, rx, ry)
    stroke(ctx, INK, 4.2)
    if lid <= .05:  # the heavy upper-lid line
        ctx.new_sub_path()
        ctx.save()
        ctx.translate(cx, cy)
        ctx.scale(rx, ry)
        ctx.arc(0, 0, 1, math.pi * 1.08, math.pi * 1.92)
        ctx.restore()
        stroke(ctx, INK, 8.5)


def mouth(ctx, mx, my, p):
    kind, o = p["mouth"], p["mopen"]
    if kind == "smile":
        cr_path(ctx, [(mx - 17, my - 3), (mx - 4, my + 4), (mx + 10, my + 2), (mx + 19, my - 6)], tension=1.0)
        stroke(ctx, INK, 4.6)
        poly(ctx, [(mx + 17, my - 9), (mx + 22, my - 3)], closed=False)  # the confident dimple tick
        stroke(ctx, INK, 3.4)
        return
    if kind == "set":
        cr_path(ctx, [(mx - 14, my + 1), (mx, my - 1), (mx + 15, my + 2)], tension=1.0)
        stroke(ctx, INK, 4.6)
        return
    # open mouths: a flat-topped 'D' (talk / grin) or a small 'o'
    wdt = {"talk": 20, "grin": 27, "o": 11}[kind]
    hgt = {"talk": 8 + 14 * o, "grin": 19, "o": 9 + 8 * o}[kind]
    if kind == "o":
        ellipse(ctx, mx, my + 2, wdt, hgt)
    else:
        ctx.move_to(mx - wdt, my - 6)
        ctx.curve_to(mx - wdt * .6, my - 9, mx + wdt * .6, my - 9, mx + wdt, my - 6)
        ctx.curve_to(mx + wdt * .9, my - 6 + hgt, mx - wdt * .9, my - 6 + hgt, mx - wdt, my - 6)
        ctx.close_path()
    fill(ctx, P["mouth"], True)
    ctx.save()
    ctx.clip_preserve()
    ellipse(ctx, mx + 2, my - 6 + hgt * 1.02, wdt * .62, hgt * .45)
    fill(ctx, P["tongue"])
    ctx.restore()
    stroke(ctx, INK, 4.2)


def rig_and_kid(ctx, p):
    """The rope, ring, straps, plank seat, bucket and Squeaky himself, as one cel."""
    cel = Cel(ctx)
    ctx.set_matrix(BASE)
    # sway: the whole rig rotates about the far-off pivot
    ctx.translate(*PIVOT)
    ctx.rotate(p["sway"])
    ctx.translate(*(ORIGIN - PIVOT))
    rig_m = ctx.get_matrix()
    ring = RING
    cel.tube([(0, -900), tuple(ring + (0, -10))], 9, P["rope"])  # the rope up out of frame
    for sx in (-1, 1):  # straps from the ring to the plank ends (behind him)
        cel.tube([tuple(ring), (sx * 160, 6)], 8, P["rope"])
    cel.shape(lambda c: ellipse(c, ring[0], ring[1], 14, 14), P["steel"])
    cel.shape(lambda c: ellipse(c, ring[0], ring[1], 6, 6), P["rubber"], sil=False, line=2.5)
    # bucket hanging off the left end of the plank (handle behind)
    bx, by = -198.0, 68.0
    cel.line(lambda c: cr_path(c, [(bx - 44, by + 2), (bx - 30, by - 36), (-170, 20), (bx + 30, by - 36), (bx + 44, by + 2)], tension=.9), 4.2)
    cel.shape(lambda c: poly(c, [(-176, -12), (176, -12), (184, 2), (-184, 2)]), hexc("#f0c07e"))  # plank top surface
    cel.shape(lambda c: rrect(c, -184, 0, 184, 26, 4), P["wood"])  # the plank seat, front face
    cel.line(lambda c: poly(c, [(-150, 9), (-60, 9)], closed=False), 2.2, P["wood_ln"])
    cel.line(lambda c: poly(c, [(40, 15), (140, 15)], closed=False), 2.2, P["wood_ln"])
    for sx in (-1, 1):
        cel.shape(lambda c, sx=sx: ellipse(c, sx * 160, 6, 7, 7), P["steel"], sil=False, line=2.6)
    cel.shape(lambda c: poly(c, [(bx - 46, by), (bx + 46, by), (bx + 37, by + 74), (bx - 37, by + 74)]), P["bucket"])
    cel.shape(lambda c: ellipse(c, bx, by, 46, 11), P["bucket_in"], sil=False)
    cel.line(lambda c: poly(c, [(bx - 42, by + 18), (bx + 42, by + 18)], closed=False), 3.0)

    def suds(c):
        for (dx, dy, r) in ((-28, -6, 15), (-8, -14, 19), (16, -10, 16), (33, -3, 12), (-38, 2, 9)):
            ellipse(c, bx + dx, by + dy, r, r * .9)
    cel.shape(suds, P["suds"], sil=True, line=3.2)

    # --- Squeaky (seat-relative; hop lifts him, squash scales about the seat, lean rotates about the seat)
    ctx.translate(0, -p["hop"])
    ctx.rotate(p["lean"])
    ctx.scale(1 / math.sqrt(p["sq"]), p["sq"])
    body_m = ctx.get_matrix()
    for side, sw in ((-1, p["legL"]), (1, p["legR"])):  # knees over the plank edge, short shins, tall red rubber boots
        kx = side * 36
        foot = np.array([side * 40 + 24 * math.sin(sw), 82 - 8 * (1 - math.cos(sw))])
        cel.tube([(kx, -6), (kx + side * 2, 12), tuple(foot - (0, 14))], 36, P["suit"])
        fx, fy = foot
        boot = lambda c, fx=fx, fy=fy, side=side: cr_path(c, [(fx - 19, fy - 40), (fx + 19, fy - 40), (fx + 20, fy - 8), (fx + 24 + side * 8, fy - 2),
                                                           (fx + 26 + side * 8, fy + 14), (fx - 4, fy + 18), (fx - 24, fy + 12), (fx - 21, fy - 10)],
                                                          closed=True, tension=.55)
        cel.shape(boot, P["boot"])
        cel.shape(lambda c, fx=fx, fy=fy: rrect(c, fx - 23, fy - 46, fx + 23, fy - 34, 5), hexc("#ff6a6f"), line=3.4)  # boot cuff
        cel.line(lambda c, fx=fx, fy=fy, side=side: poly(c, [(fx - 22, fy + 8), (fx + 25 + side * 8, fy + 8)], closed=False), 3.0)
    shirt = [(-46, -152), (46, -152), (60, -120), (64, -60), (-64, -60), (-60, -120)]
    cel.shape(lambda c: cr_path(c, shirt, closed=True, tension=.6), P["tee"])
    pants = [(-66, -70), (66, -70), (72, -34), (76, -8), (40, -2), (0, -4), (-40, -2), (-76, -8), (-72, -34)]
    cel.shape(lambda c: cr_path(c, pants, closed=True, tension=.6), P["suit"])
    for side in (-1, 1):  # thighs come toward us over the plank: two round knees on its front edge
        kx = side * 38
        cel.shape(lambda c, kx=kx: ellipse(c, kx, 2, 30, 19), P["suit"], sil=False, line=3.6)
    bib = lambda c: (c.move_to(-38, -64), c.line_to(-38, -112), c.curve_to(-38, -122, -32, -124, -24, -124),
                     c.line_to(24, -124), c.curve_to(32, -124, 38, -122, 38, -112), c.line_to(38, -64), c.close_path())
    cel.shape(bib, P["suit"], sil=False)
    for sx in (-1, 1):  # overall straps up over the shoulders, with big buttons
        cel.shape(lambda c, sx=sx: poly(c, [(sx * 24, -120), (sx * 38, -120), (sx * 48, -152), (sx * 32, -152)]), P["suit"], sil=False, line=3.2)
        cel.shape(lambda c, sx=sx: ellipse(c, sx * 28, -112, 7, 7), INK, sil=False, line=1.0)

    def bib_detail(c):
        rrect(c, -20, -106, 20, -74, 4)  # chest pocket
        fill(c, P["suit"], True)
        stroke(c, INK, 3.0)
        ellipse(c, 0, -90, 11, 11)  # badge: a white roundel with a blue droplet
        fill(c, WHITE, True)
        stroke(c, INK, 2.6)
        c.move_to(0, -98)
        c.curve_to(5, -91, 6, -87, 0, -83)
        c.curve_to(-6, -87, -5, -91, 0, -98)
        fill(c, P["pole"])
    cel.raw(bib_detail)
    # screen-left arm: white sleeve, bare arm, mitten hand holding the strap
    sh = (-50, -124)
    shL, eL = np.array((-44.0, -128.0)), np.array(p["elbowL"])
    cel.tube([tuple(shL), p["elbowL"], p["handL"]], 25, P["skin"])
    cel.tube([tuple(shL), tuple(shL + (eL - shL) * .5)], 36, P["tee"], sil=False)
    hl = p["handL"]
    cel.shape(lambda c: ellipse(c, hl[0], hl[1] + 2, 21, 24, -.3), P["skin"])
    # head (head transform: offset and tilt about the neck)
    ctx.translate(p["head_dx"], -150 + p["head_dy"])
    ctx.rotate(p["head_tilt"])
    ctx.translate(0, 150)
    hcx, hcy, hrx, hry = 0.0, -272.0, 152.0, 120.0
    tw = p["turn"]
    cel.shape(lambda c: superellipse(c, hcx, hcy, hrx, hry, 2.35, 1.0), P["skin"])
    for side in (-1, 1):  # hair tufts poking out under the hat
        pts = [(140, -332), (172, -328), (152, -314), (180, -302), (150, -296), (162, -280), (138, -286)]
        cel.shape(lambda c, side=side, pts=pts: poly(c, [(side * x + tw * 6, y) for x, y in pts]), INK, line=2.0)
    fx = tw * 18  # features wrap round the head toward where he looks
    ex, ey, erx, ery = 59.0, -246.0, 52.0, 58.0

    def face(c):
        eye(c, fx - ex, ey, erx * (1 + .05 * tw), ery, p, -1)
        eye(c, fx + ex, ey, erx * (1 - .13 * tw), ery * (1 - .03 * tw), p, 1)
        for sx in (-1, 1):
            for dx, dy in ((0, 0), (11, -4), (8, 8)):
                ellipse(c, fx + sx * (72 + dx) - (8 if sx > 0 else 0) * tw, -178 + dy, 3.2, 3.2)
        fill(c, hexc("#e7976f"))
        mouth(c, fx + 4, -170, p)
    cel.raw(face)
    # hard hat, worn at a rakish tilt
    ctx.translate(0, -340)
    ctx.rotate(-.06)
    ctx.translate(0, 340)
    dome = lambda c: (c.move_to(-150, -324), c.curve_to(-150, -420, 150, -420, 150, -324), c.close_path())
    cel.shape(dome, P["hat"])

    def brim(c):
        c.move_to(-178, -330)
        c.curve_to(-90, -316, 90, -316, 178, -330)
        c.curve_to(184, -324, 182, -314, 176, -312)
        c.curve_to(90, -298, -90, -298, -176, -312)
        c.curve_to(-182, -314, -184, -324, -178, -330)
        c.close_path()
    cel.shape(brim, P["hat"])
    rx_ = tw * 24
    cel.shape(lambda c: (c.move_to(rx_ - 12, -394), c.curve_to(rx_ - 22, -370, rx_ - 20, -345, rx_ - 17, -322),
                         c.line_to(rx_ + 17, -322), c.curve_to(rx_ + 20, -345, rx_ + 22, -370, rx_ + 12, -394), c.close_path()),
              P["hat"], sil=False, line=3.2)

    def shine(c):
        c.new_sub_path()
        c.save()
        c.translate(0, -330)
        c.scale(118, 70)
        c.arc(0, 0, 1, math.pi * 1.16, math.pi * 1.36)
        c.restore()
        stroke(c, WHITE, 7)
    cel.raw(shine)
    # squeegee + the right arm holding it (back to body space)
    ctx.set_matrix(body_m)
    g = np.array(p["grip"])
    hd = np.array(p["sq_head"])
    d = (hd - g) / (np.linalg.norm(hd - g) + 1e-9)
    nrm = np.array([-d[1], d[0]])
    if p["smear"] is not None:
        sm = p["smear"]
        cel.raw(lambda c: smear(c, sm))
    butt = g - d * 78
    ba = p["blade"]  # the squeegee head swivels on the pole
    nr = np.array([math.cos(ba), math.sin(ba)])
    a0, a1 = hd + nr * 98, hd - nr * 98
    dd = np.array([nr[1], -nr[0]])
    cel.tube([tuple(butt), tuple(hd - dd * 14 - d * 16)], 15, P["pole"])
    blade = [a0 + dd * 2, a1 + dd * 2, a1 + dd * 16, a0 + dd * 16]
    chan = [a0 - dd * 14, a1 - dd * 14, a1 + dd * 4, a0 + dd * 4]
    cel.shape(lambda c: poly(c, [tuple(q) for q in blade]), P["rubber"], line=3.0)
    cel.shape(lambda c: poly(c, [tuple(q) for q in chan]), P["steel"])
    cel.line(lambda c: poly(c, [tuple(a0 - dd * 5), tuple(a1 - dd * 5)], closed=False), 2.2, hexc("#7f8b99"))
    hc = hd - dd * 14
    cel.shape(lambda c: poly(c, [tuple(hc - d * 18 + nrm * 9), tuple(hc - d * 18 - nrm * 9), tuple(hc - nrm * 17),
                                 tuple(hc + nrm * 17)]), P["steel_dk"])
    shR, eR = np.array((44.0, -128.0)), np.array(p["elbowR"])
    cel.tube([tuple(shR), p["elbowR"], tuple(g + nrm * 2)], 25, P["skin"])
    cel.tube([tuple(shR), tuple(shR + (eR - shR) * .5)], 36, P["tee"], sil=False)
    cel.shape(lambda c: ellipse(c, g[0], g[1], 22, 24, math.atan2(d[1], d[0])), P["skin"])
    cel.draw()
    ctx.set_matrix(rig_m)
    return rig_m


def smear(c, sm):
    """A smear drawing: the squeegee's sweep as one flat shape with ink edge, plus drybrush speed streaks."""
    pts_a, pts_b = sm
    shape = [tuple(q) for q in pts_a] + [tuple(q) for q in pts_b[::-1]]
    cr_path(c, shape, closed=True, tension=.6)
    fill(c, hexc("#e9eef3"), True)
    stroke(c, INK, 3.6)
    for k in range(1, 4):
        u = k / 4
        line = [tuple(lerp(np.array(a), np.array(b), u)) for a, b in zip(pts_a, pts_b)]
        cr_path(c, line[: max(2, len(line) - k)], tension=.8)
        stroke(c, hexc("#9aa6b3"), 2.4)


# ================================================================================================== effects
def bubbles(ctx, t):
    """Soap bubbles drifting up from the bucket on a 5 s loop (so the shot's first and last frames match)."""
    base = ORIGIN + np.array([-205.0, 50.0])
    for k in range(6):
        tk = (t + k * DUR / 6 + .35) % DUR
        life = 2.7
        if tk > life + .12:
            continue
        rr = np.random.default_rng(k + 40)
        drift = rr.uniform(-130, -40)
        u = min(tk / life, 1.0)
        x = base[0] + drift * u + 14 * math.sin(tk * 3.1 + k)
        y = base[1] - 300 * (1 - (1 - u) ** 1.6)
        r = rr.uniform(9, 16) * (.55 + .45 * min(tk / .4, 1))
        if tk > life:  # pop: a little burst of ticks
            for q in range(6):
                a = q * math.pi / 3 + k
                poly(ctx, [(x + r * .9 * math.cos(a), y + r * .9 * math.sin(a)), (x + r * 1.6 * math.cos(a), y + r * 1.6 * math.sin(a))],
                     closed=False)
            stroke(ctx, INK, 2.6)
            continue
        ellipse(ctx, x, y, r, r)
        fill(ctx, hexc("#e9fbff", .55), True)
        stroke(ctx, INK, 2.8)
        ctx.new_sub_path()
        ctx.arc(x, y, r * .62, math.pi * 1.1, math.pi * 1.45)
        stroke(ctx, WHITE, 2.6)


def sparkle(ctx, x, y, s, a=0.0):
    """A four-point twinkle star, white with an ink edge."""
    if s <= .02:
        return
    pts = []
    for k in range(8):
        r = (34 if k % 2 == 0 else 7) * s
        ang = a + k * math.pi / 4
        pts.append((x + r * math.cos(ang), y + r * math.sin(ang)))
    cr_path(ctx, pts, closed=True, tension=.25)
    fill(ctx, WHITE, True)
    stroke(ctx, INK, 3.0)
    ellipse(ctx, x, y, 5 * s, 5 * s)
    fill(ctx, P["hat"])


def burst_bg(ctx, cx, cy, i):
    """Pop-art action card: flat violet with radiating stripes, swapped in for the swipe's split second."""
    ctx.rectangle(-20, -20, W + 40, H + 40)
    fill(ctx, P["burst"])
    for k in range(28):
        a0 = k * 2 * math.pi / 28 + (i % 2) * .05
        a1 = a0 + math.pi / 28 * .8
        poly(ctx, [(cx, cy), (cx + 2200 * math.cos(a0), cy + 2200 * math.sin(a0)), (cx + 2200 * math.cos(a1), cy + 2200 * math.sin(a1))])
    fill(ctx, P["burst2"])
    rng = np.random.default_rng(i)
    for k in range(16):  # white speed streaks
        a = rng.uniform(0, 2 * math.pi)
        r0, r1 = rng.uniform(260, 420), rng.uniform(520, 900)
        poly(ctx, [(cx + r0 * math.cos(a), cy + r0 * math.sin(a)), (cx + r1 * math.cos(a), cy + r1 * math.sin(a))], closed=False)
        stroke(ctx, hexc("#ffffff", .9), rng.uniform(3, 7))


# ================================================================================================== timeline
T_ANTIC, T_SWIPE, T_IMPACT = 14, 28, 30
# the gibberish line: (start s, length s, vowel, pitch Hz, consonant onset); drives both the mouth flaps and the synth voice
SYL = [(2.18, .12, "a", 330, "w"), (2.32, .10, "i", 370, "p"), (2.44, .17, "a", 410, "d"), (2.68, .11, "o", 340, "k"),
       (2.81, .10, "i", 380, "d"), (2.93, .24, "u", 430, "sh"), (3.24, .07, "e", 470, "h"), (3.34, .07, "e", 480, "h")]


def mouth_for(t):
    for t0, dur, v, f0, c in SYL:
        if t0 - .03 <= t < t0 + dur - .01:
            return {"a": ("talk", .9), "o": ("o", .7), "u": ("o", .9), "i": ("talk", .35), "e": ("grin", 0)}[v]
    return None


def pose_at(i):
    """Key poses with eased in-betweens; the character animates on twos except the swipe (on ones)."""
    p = dict(REST)
    k = i if T_SWIPE - 1 <= i <= T_IMPACT + 2 else i // 2 * 2
    t = k / FPS
    # blink near the start
    if k in (8, 10):
        p["blink"] = .85 if k == 8 else 1.0
    antic = dict(REST, lean=-.07, sq=.93, head_dx=-10, head_dy=6, head_tilt=-.05, turn=.55, look=(1.0, -.6), lid=.28, mouth="set",
                 grip=(150.0, -176.0), elbowR=(108.0, -158.0), sq_head=(410.0, -400.0), blade=-.04, legL=.35, legR=-.25)
    hit = dict(REST, lean=.08, sq=1.04, head_dx=10, head_dy=-2, head_tilt=.06, turn=.6, look=(1.0, .5), lid=.18, mouth="o", mopen=.3,
               grip=(172.0, -84.0), elbowR=(116.0, -72.0), sq_head=(484.0, -112.0), blade=.05, legL=-.3, legR=.45,
               handL=(-122.0, -104.0))
    proud = dict(REST, lean=-.02, sq=1.03, head_dx=-4, head_dy=-6, head_tilt=-.04, turn=0.0, look=(0.0, .05), mouth="talk", mopen=.5,
                 grip=(118.0, -134.0), elbowR=(96.0, -84.0), sq_head=(198.0, -478.0), blade=0.0, legL=.15, legR=-.15)
    if k < T_ANTIC:
        pass
    elif k < T_SWIPE:  # wind-up
        u = ease((k - T_ANTIC) / (T_SWIPE - 2 - T_ANTIC))
        p = pose_mix(REST, antic, u)
    elif k < T_IMPACT:  # smear frames
        u = (k - T_SWIPE + 1) / 3
        p = pose_mix(antic, hit, u)
        p["mouth"], p["lid"], p["look"] = "o", .1, (1.0, .3)
        arc_a, arc_b = [], []
        for sv in np.linspace(0, 1, 9):
            w = sv * u
            hd = np.array(antic["sq_head"]) + (np.array(hit["sq_head"]) - np.array(antic["sq_head"])) * w
            hd = hd + np.array([46, 0]) * math.sin(math.pi * w)
            ba = lerp(antic["blade"], hit["blade"], w)
            nr = np.array([math.cos(ba), math.sin(ba)])
            arc_a.append(hd + nr * 104)
            arc_b.append(hd - nr * 104)
        p["smear"] = (arc_a, arc_b)
        p["sq_head"] = tuple(np.array(p["sq_head"]) + np.array([46, 0]) * math.sin(math.pi * u))
    elif k < 42:  # impact hold with a little overshoot settle
        u = (k - T_IMPACT) / 12
        p = dict(REST)
        p.update(hit)
        p["sq"] = 1.04 + .05 * math.exp(-u * 4) * math.cos(u * 12)
        if k >= 34:
            p["mouth"], p["lid"], p["happy"] = "smile", 0.0, .35
            p["look"] = (.2, .1)
            p["turn"] = .35
    elif k < 52:  # swing round to camera, pole up
        u = ease((k - 42) / 10)
        start = dict(REST)
        start.update(hit)
        start.update(mouth="smile", lid=0.0, happy=.35, look=(.2, .1), turn=.35)
        p = pose_mix(start, proud, u)
        p["mouth"] = "smile"
        p["mopen"] = 0.0
    elif k < 80:  # the line (gibberish voice), mouth flaps on twos
        p = dict(REST)
        p.update(proud)
        j = (k - 52) // 2
        m = mouth_for(t)
        p["mouth"], p["mopen"] = m if m else (("grin", 0) if t > 3.15 else ("smile", 0))
        p["head_dy"] = proud["head_dy"] + (3 if m and m[1] > .8 else 0)  # head bobs on the stressed syllables
        p["head_tilt"] = proud["head_tilt"] + (.03 if 2.6 <= t < 2.95 else 0)
        if t > 3.1:
            p["happy"] = .6
    elif k < 96:  # the hop: squash, stretch up off the seat, land
        u = (k - 80) / 16
        p = dict(REST)
        p.update(proud)
        p.update(mouth="grin", happy=0.0, closed_happy=1.0)
        if u < .2:
            p["sq"], p["hop"] = .9, 0.0
        elif u < .7:
            v = (u - .2) / .5
            p["hop"] = 46 * math.sin(math.pi * v)
            p["sq"] = 1.1 - .1 * v
            p["legL"], p["legR"] = .7 * math.sin(v * 6), -.7 * math.sin(v * 6)
        else:
            v = (u - .7) / .3
            p["sq"] = .9 + .1 * v
        if u > .85:
            p["closed_happy"], p["mouth"], p["happy"] = 0.0, "smile", .5
    elif k < 110:  # settle back to the key pose
        u = ease((k - 96) / 12)
        start = dict(REST)
        start.update(proud)
        start.update(mouth="smile", mopen=0.0, happy=.5)
        p = pose_mix(start, REST, u)
        p["happy"] = .5 * (1 - u)
    # rig sway: kicked by the swipe and by the hop, faded out so the end matches the start exactly
    sw = 0.0
    for t0, amp in ((T_IMPACT / FPS, .035), (88 / FPS, -.03)):
        if t > t0:
            sw += amp * math.exp(-(t - t0) * 1.3) * math.sin((t - t0) * 4.2)
    sw += .006 * math.sin(t * 2 * math.pi / DUR)  # a gentle idle drift that loops on the 5 s cycle
    fade = 1 - ease((t - 3.9) / .7)
    p["sway"] = sw * fade if t > 1.2 else .006 * math.sin(t * 2 * math.pi / DUR)
    p["legL"] = p.get("legL", 0) + .08 * math.sin(t * 2 * math.pi / DUR * 2)
    p["legR"] = p.get("legR", 0) - .08 * math.sin(t * 2 * math.pi / DUR * 2)
    return p, t


def frame(i, finish=True):
    p, t = pose_at(i)
    bg = background()
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * SS, H * SS)
    ctx = cairo.Context(surf)
    ctx.set_matrix(BASE)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    burst = T_SWIPE <= i < T_IMPACT + 4
    if burst:
        burst_bg(ctx, 980, 300, i)
    else:
        # the freshly squeegeed streak on the glass, fading, and the twinkle
        v = (i - T_IMPACT) / FPS
        if 0 <= v < 2.0:
            ctx.save()
            ctx.rectangle(952, 118, 120, 150)
            ctx.rectangle(952, 306, 120, 150)
            ctx.clip()
            poly(ctx, [(930, 150), (1080, 120), (1100, 440), (950, 470)])
            fill(ctx, hexc("#ffffff", .55 * (1 - ease(v / 2.0))))
            ctx.restore()
        if 0.08 <= v < 1.6:
            s = math.sin(math.pi * min((v - .08) / 1.5, 1)) ** .6
            tw_ = 1 + .18 * ((i // 2) % 2)
            sparkle(ctx, 1018, 176, s * 1.35 * tw_, v * 1.2)
            sparkle(ctx, 1058, 400, s * .8 / tw_, -v * 1.6)
            sparkle(ctx, 975, 300, s * .55 * tw_, v)
    bubbles(ctx, (i // 2 * 2) / FPS)
    rig_and_kid(ctx, p)
    if T_SWIPE <= i < T_IMPACT + 2:  # speed lines over the action
        rng = np.random.default_rng(i + 3)
        for k in range(9):
            y0 = rng.uniform(80, 520)
            x0 = rng.uniform(820, 1000)
            poly(ctx, [(x0, y0), (x0 + rng.uniform(120, 260), y0 + rng.uniform(60, 140))], closed=False)
            stroke(ctx, WHITE, rng.uniform(3, 6))
    a = np.ndarray((H * SS, W * SS, 4), np.uint8, surf.get_data()).astype(np.float32) / 255
    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
    rgb, al = a[:, :, 2::-1], a[:, :, 3:4]
    img = rgb + bg * (1 - al)
    if T_IMPACT <= i < T_IMPACT + 4:  # impact shake
        amp = 7 * (1 - (i - T_IMPACT) / 4)
        M = np.float32([[1, 0, amp * (1 if i % 2 else -1)], [0, 1, amp * .5 * (-1 if i % 2 else 1)]])
        img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    if finish:
        img = finish_img(img, i)
    return (np.clip(img, 0, 1) * 255 + .5).astype(np.uint8)


def finish_img(img, i):
    """Light broadcast-cel finish: a touch of chroma bleed and softness, a whisper of film grain."""
    yuv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2YCrCb)
    yuv[:, :, 1:] = cv2.GaussianBlur(yuv[:, :, 1:], (0, 0), sigmaX=.9, sigmaY=.25)
    img = cv2.cvtColor(yuv, cv2.COLOR_YCrCb2RGB)
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=.35, sigmaY=.3)
    rng = np.random.default_rng(900 + i)
    g = cv2.resize(rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32), (W, H))
    return img + g[..., None] * .006


def key_frame():
    """The key frame used as first and last frame for the image-to-video render: frame 0, cropped to 1280x704."""
    return frame(0)[8:712]
