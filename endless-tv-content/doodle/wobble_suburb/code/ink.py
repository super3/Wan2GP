"""Ink primitives for the wobble_suburb style: hand-inked outlines with a living width, flat cel fills, and line boil.

Every shape is resampled into a dense polyline, nudged by a deterministic jitter (the "boil": each drawing is a new tracing of
the same pose) and then filled flat and inked with a stroke whose width breathes along its length like a dip pen.
Nothing here reads image files or fonts."""
import math
import zlib

import cairo
import numpy as np

W, H = 1280, 704
SS = 2


def hexc(h, a=1.0):
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, a)


INK = hexc("#1b1216")
LW = 4.3

# boil state: index of the current tracing and the jitter amplitude (px, in local drawing units)
BOIL = {"i": 0, "a": 0.55, "w": 1.0}


def set_boil(i, a):
    BOIL["i"], BOIL["a"] = int(i), float(a)


def set_wmul(w):
    """Line-width multiplier so a close-up keeps the same on-screen pen as a wide shot."""
    BOIL["w"] = float(w)


def _seed(tag):
    if isinstance(tag, str):
        return zlib.crc32(tag.encode())
    return int(tag) & 0xFFFFFFFF


def rng_for(tag, extra=0):
    return np.random.default_rng((_seed(tag) * 1315423911 + BOIL["i"] * 2654435761 + extra) & 0xFFFFFFFFFFFF)


def canvas(w=W, h=H, ss=SS):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w * ss, h * ss)
    ctx = cairo.Context(surf)
    ctx.scale(ss, ss)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    return surf, ctx


# ------------------------------------------------------------------------------------------------ geometry
def crs(pts, closed=False, n=10, tension=1.0):
    """Dense Catmull-Rom samples through pts."""
    p = np.asarray(pts, float)
    m = len(p)
    if m < 2:
        return p
    if closed:
        idx = lambda i: p[i % m]  # noqa: E731
        segs = m
    else:
        idx = lambda i: p[min(max(i, 0), m - 1)]  # noqa: E731
        segs = m - 1
    out = []
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    for i in range(segs):
        p0, p1, p2, p3 = idx(i - 1), idx(i), idx(i + 1), idx(i + 2)
        c1, c2 = p1 + (p2 - p0) * tension / 6, p2 - (p3 - p1) * tension / 6
        out.append((1 - t) ** 3 * p1 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * p2)
    if not closed:
        out.append(p[-1:])
    return np.concatenate(out)


def jitter(pts, tag, k=1.0):
    p = np.asarray(pts, float)
    a = BOIL["a"] * k
    if a <= 0:
        return p
    r = rng_for(tag)
    # low-frequency jitter: each control point moves a bit, neighbours move together a little
    j = r.normal(0, a, p.shape)
    if len(p) > 3:
        j = (j + np.roll(j, 1, 0) * .4 + np.roll(j, -1, 0) * .4) / 1.4
    return p + j


def rot(p, a, c=(0, 0)):
    p = np.asarray(p, float) - c
    ca, sa = math.cos(a), math.sin(a)
    return np.stack([p[..., 0] * ca - p[..., 1] * sa, p[..., 0] * sa + p[..., 1] * ca], -1) + c


def arclen(p):
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return np.concatenate([[0], np.cumsum(d)])


def ellipse_pts(cx, cy, rx, ry, n=10, a0=0.0, rotn=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + a0
    p = np.stack([rx * np.cos(a), ry * np.sin(a)], 1)
    return rot(p, rotn) + (cx, cy)


# ------------------------------------------------------------------------------------------------ drawing
def poly_path(ctx, p, closed=True):
    ctx.move_to(*p[0])
    for q in p[1:]:
        ctx.line_to(*q)
    if closed:
        ctx.close_path()


def fillc(ctx, c):
    ctx.set_source_rgba(*c)
    ctx.fill()


def ink_poly(ctx, d, tag, lw=LW, closed=True, taper=(0.25, 0.25), ink=INK, var=.22):
    """Stroke a dense polyline with a pen whose width breathes along the line (chunked round-capped strokes)."""
    if len(d) < 2 or lw <= 0:
        return
    lw = lw * BOIL["w"]
    s = arclen(d)
    L = s[-1] + 1e-6
    r = rng_for(tag, 7)
    ph1, ph2 = r.uniform(0, 6.3, 2)
    w = lw * (1 + var * np.sin(s / 31 + ph1) + var * .55 * np.sin(s / 11 + ph2))
    if not closed:
        e0 = np.clip(s / max(14.0, L * .18), 0, 1)
        e1 = np.clip((L - s) / max(14.0, L * .18), 0, 1)
        w = w * (taper[0] + (1 - taper[0]) * np.sqrt(e0)) * (taper[1] + (1 - taper[1]) * np.sqrt(e1))
    if closed:
        d = np.concatenate([d, d[:2]])
        w = np.concatenate([w, w[:2]])
    ctx.set_source_rgba(*ink)
    k = 4
    for a in range(0, len(d) - 1, k):
        seg = d[a:a + k + 1]
        if len(seg) < 2:
            continue
        ctx.set_line_width(float(np.mean(w[a:a + k + 1])))
        poly_path(ctx, seg, closed=False)
        ctx.stroke()


def shape(ctx, pts, fc, tag, lw=LW, closed=True, tension=1.0, k=1.0, n=10, ink=INK, dense=False):
    """Flat fill + hand ink.  pts are control points (or a dense polyline if dense=True)."""
    p = jitter(pts, tag, k)
    d = p if dense else crs(p, closed, n, tension)
    if fc is not None and closed:
        poly_path(ctx, d)
        fillc(ctx, fc)
    if lw > 0:
        ink_poly(ctx, d, tag, lw, closed, ink=ink)
    return d


def line(ctx, pts, tag, lw=LW, tension=1.0, taper=(0.25, 0.25), k=1.0, n=10, ink=INK):
    p = jitter(pts, tag, k)
    d = crs(p, False, n, tension)
    ink_poly(ctx, d, tag, lw, False, taper, ink=ink)
    return d


def tube(ctx, pts, width, fc, tag, lw=LW, k=1.0, n=10, cap=True):
    """A limb: inked outline by stroking wide in ink then narrower in the fill colour."""
    p = jitter(pts, tag, k)
    d = crs(p, False, n, 1.0)
    lw = lw * BOIL["w"]
    ctx.set_line_cap(cairo.LINE_CAP_ROUND if cap else cairo.LINE_CAP_BUTT)
    poly_path(ctx, d, False)
    ctx.set_source_rgba(*INK)
    ctx.set_line_width(width + 2 * lw)
    ctx.stroke()
    poly_path(ctx, d, False)
    ctx.set_source_rgba(*fc)
    ctx.set_line_width(width)
    ctx.stroke()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    return d


def dot(ctx, x, y, r, c, tag=None, k=1.0):
    if tag is not None:
        x, y = jitter([(x, y)], tag, k)[0]
    ctx.new_sub_path()
    ctx.arc(x, y, r, 0, 2 * math.pi)
    fillc(ctx, c)


def ik2(a, t, l1, l2, bend=1.0, stretch=True):
    """Two-bone IK: root a, target t; returns (joint, end).  Rubber-hose: if out of reach, the limb stretches."""
    a, t = np.asarray(a, float), np.asarray(t, float)
    v = t - a
    d = float(np.linalg.norm(v)) + 1e-9
    if d >= l1 + l2 - 1e-3:
        if stretch:
            s = d / (l1 + l2)
            l1, l2 = l1 * s, l2 * s
        else:
            t = a + v / d * (l1 + l2)
            d = l1 + l2
        j = a + v / d * l1
        # keep a hint of bend so it doesn't look like a stick
        nrm = np.array([-v[1], v[0]]) / d
        return j + nrm * bend * 4, t
    cosA = (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d)
    A = math.acos(max(-1, min(1, cosA)))
    base = math.atan2(v[1], v[0])
    ang = base - A * (1 if bend > 0 else -1)
    j = a + l1 * np.array([math.cos(ang), math.sin(ang)])
    return j, t
