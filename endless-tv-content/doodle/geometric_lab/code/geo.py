"""geometric_lab: a late-90s CN / UPA-revival style shot, drawn entirely in code.

Mr. Ferdinand Hoist, a tiny ram-rod-stiff elderly elevator operator, stands at the brass controls of Car No. 7 in a 1950s
department-store lobby.  Character + moving props are a "cel" layer (black ink, flat colour, heavier outer silhouette);
the background is a "painted card" (coloured lines, flat fields, soft gradients, paper tooth)."""
import math

import cairo
import cv2
import numpy as np

W, H, FPS = 1280, 720, 24
SS = 2                      # supersampling
DUR = 5.0
N = int(DUR * FPS)
LW = 4.4                    # cel ink line (1x px)
SIL = 1.9                   # extra outer-silhouette thickness (1x px)


def hx(h, a=1.0):
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, a)


INK = hx("#16121a")
C = dict(
    # lobby
    wall=hx("#4cb5aa"), wall_ln=hx("#2c847c"), cornice=hx("#378f87"), cornice2=hx("#2f7d76"),
    wains=hx("#2a6c7c"), wains2=hx("#25606f"), wains_ln=hx("#1a4a57"),
    floor_a=hx("#efe0c0"), floor_b=hx("#d9916c"), floor_ln=hx("#b56a45"),
    brass=hx("#e6b04a"), brass2=hx("#cf9433"), brass3=hx("#f3cd78"), brass_ln=hx("#8c5d17"),
    # cab
    cab_back=hx("#7c6aa8"), cab_l=hx("#6a5996"), cab_r=hx("#5e4e8a"), cab_ceil=hx("#9483be"), cab_floor=hx("#4a3d6f"),
    cab_floor2=hx("#3f335f"), cab_ln=hx("#4b3d72"), lightpanel=hx("#fff2c6"),
    # props
    leaf=hx("#2f8a55"), leaf2=hx("#57ad6e"), leaf_ln=hx("#1b5636"), pot=hx("#f4ead2"), pot_ln=hx("#9a7d52"),
    dial_face=hx("#fbf1d6"), lamp_off=hx("#a87a33"), lamp_on=hx("#fff07a"),
    # character
    skin=hx("#f8c79b"), nose=hx("#f29a78"), white=hx("#ffffff"), hairw=hx("#f4f4f0"),
    red=hx("#d7302b"), red2=hx("#b3241f"), gold=hx("#f4b938"), navy=hx("#2b3458"), shoe=hx("#1d1a24"),
    mouth=hx("#5e1a26"), tongue=hx("#e56f73"), knob=hx("#1f1b26"),
)

# camera-static layout (1x px)
OPEN = (446, 170, 834, 600)          # inner door opening x0, y0, x1, y1
BACK = (512, 235, 768, 518)          # cab back wall: the opening scaled .66 toward the VP (640, 360)          # cab back wall
DIAL = (640, 124, 70)                # floor-indicator half dial, centre + radius
LEVER_PIVOT = (508, 520)
LEVER_LEN = 74
HERO = (650, 597)                    # feet, between the heels


# ============================================================================================ geometry helpers
def rpoly(ctx, pts, r=6.0, closed=True):
    """Polygon with rounded corners (per-vertex radius) — the core of the geometric construction."""
    P = np.asarray(pts, float)
    n = len(P)
    rr = np.broadcast_to(np.asarray(r, float), (n,))
    rng = range(n) if closed else range(1, n - 1)
    if not closed:
        ctx.move_to(*P[0])
    first = closed
    for i in rng:
        p, a, b = P[i], P[i - 1], P[(i + 1) % n]
        da, db = a - p, b - p
        la, lb = np.hypot(*da) + 1e-9, np.hypot(*db) + 1e-9
        k = min(rr[i], la * .5, lb * .5)
        s, e = p + da / la * k, p + db / lb * k
        if first:
            ctx.move_to(*s)
            first = False
        else:
            ctx.line_to(*s)
        c1, c2 = s + (p - s) * .66, e + (p - e) * .66
        ctx.curve_to(c1[0], c1[1], c2[0], c2[1], e[0], e[1])
    if closed:
        ctx.close_path()
    else:
        ctx.line_to(*P[-1])


def ell(ctx, cx, cy, rx, ry, a0=0.0, a1=2 * math.pi, rot=0.0):
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(rot)
    ctx.scale(max(rx, 1e-3), max(ry, 1e-3))
    ctx.new_sub_path()
    ctx.arc(0, 0, 1, a0, a1)
    ctx.restore()


def ink(ctx, fc, w=LW, lc=INK):
    """Fill + outline (cel style)."""
    if fc is not None:
        ctx.set_source_rgba(*fc)
        ctx.fill_preserve()
    ctx.set_source_rgba(*lc)
    ctx.set_line_width(w)
    ctx.stroke()


def flat(ctx, fc):
    ctx.set_source_rgba(*fc)
    ctx.fill()


def line(ctx, pts, w=LW, c=INK, cap=cairo.LINE_CAP_ROUND):
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.set_line_cap(cap)
    ctx.set_source_rgba(*c)
    ctx.set_line_width(w)
    ctx.stroke()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)


def tube(ctx, pts, width, fc, w=LW, cap=cairo.LINE_CAP_ROUND):
    """An outlined tube (sleeve / leg): ink stroke, then a colour stroke on top."""
    for col, wid in ((INK, width + 2 * w), (fc, width)):
        ctx.move_to(*pts[0])
        for p in pts[1:]:
            ctx.line_to(*p)
        ctx.set_line_cap(cap)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_source_rgba(*col)
        ctx.set_line_width(wid)
        ctx.stroke()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)


def mix(a, b, t):
    return tuple(a[k] + (b[k] - a[k]) * t for k in range(len(a)))


def new_surface():
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * SS, H * SS)
    c = cairo.Context(s)
    c.scale(SS, SS)
    c.set_line_join(cairo.LINE_JOIN_ROUND)
    c.set_line_cap(cairo.LINE_CAP_ROUND)
    return s, c


def surf_np(s):
    """premultiplied float RGBA (H*SS, W*SS, 4) from a cairo surface."""
    s.flush()
    a = np.ndarray((H * SS, W * SS, 4), np.uint8, s.get_data())
    return a[:, :, [2, 1, 0, 3]].astype(np.float32) / 255


# ============================================================================================ background (painted card)
def bg_static(ctx):
    # ---- lobby wall: two-tone harlequin diamonds, very low contrast
    ctx.rectangle(0, 0, W, H)
    flat(ctx, C["wall"])
    dw, dh = 72, 104
    for r in range(-1, 9):
        for q in range(-1, 20):
            cx, cy = q * dw + (dw / 2 if r % 2 else 0), 40 + r * dh / 2
            if r % 2 == 0:
                continue
            ctx.move_to(cx, cy - dh / 2)
            ctx.line_to(cx + dw / 2, cy)
            ctx.line_to(cx, cy + dh / 2)
            ctx.line_to(cx - dw / 2, cy)
            ctx.close_path()
    flat(ctx, hx("#46ada2"))
    # tiny four-point "atomic" sparkles at some diamond joins (off-register: fill shifted from its line, UPA style)
    for r in range(0, 9, 3):
        for q in range(0, 19, 4):
            cx, cy = q * dw + (r % 2) * dw * 2 + 36, 40 + r * dh / 2 + 52
            star = [(cx, cy - 13), (cx + 3.5, cy - 3.5), (cx + 13, cy), (cx + 3.5, cy + 3.5), (cx, cy + 13), (cx - 3.5, cy + 3.5),
                    (cx - 13, cy), (cx - 3.5, cy - 3.5)]
            ctx.save()
            ctx.translate(2.5, 2.0)
            rpoly(ctx, star, 1.0)
            flat(ctx, hx("#6cc8bd"))
            ctx.restore()
            rpoly(ctx, star, 1.0)
            ctx.set_source_rgba(*hx("#3a9a90"))
            ctx.set_line_width(1.6)
            ctx.stroke()
    # ---- cornice band
    ctx.rectangle(0, 0, W, 30)
    flat(ctx, C["cornice"])
    ctx.rectangle(0, 30, W, 8)
    flat(ctx, C["cornice2"])
    line(ctx, [(0, 30), (W, 30)], 2.2, C["wall_ln"])
    line(ctx, [(0, 38), (W, 38)], 2.2, C["wall_ln"])
    # ---- wainscot
    ctx.rectangle(0, 516, W, 90)
    flat(ctx, C["wains"])
    for x0 in list(range(-30, 400, 108)) + list(range(890, W + 40, 108)):
        rpoly(ctx, [(x0 + 12, 532), (x0 + 96, 532), (x0 + 96, 592), (x0 + 12, 592)], 4)
        flat(ctx, C["wains2"])
        rpoly(ctx, [(x0 + 12, 532), (x0 + 96, 532), (x0 + 96, 592), (x0 + 12, 592)], 4)
        ctx.set_source_rgba(*C["wains_ln"])
        ctx.set_line_width(2)
        ctx.stroke()
    ctx.rectangle(0, 506, W, 12)
    flat(ctx, C["brass"])
    line(ctx, [(0, 506), (W, 506)], 2.2, C["brass_ln"])
    line(ctx, [(0, 518), (W, 518)], 2.2, C["brass_ln"])

    # ---- lobby floor: perspective checkerboard
    vx, vy, y0, y1 = 640, 360, 604, H
    ctx.rectangle(0, y0, W, y1 - y0)
    flat(ctx, C["floor_a"])
    rows = [y1]
    j = 1
    while rows[-1] > y0:
        rows.append(vy + (y1 - vy) / (1 + j * .16))
        j += 1
    rows[-1] = y0
    step = 96
    for a in range(len(rows) - 1):
        ya, yb = rows[a], rows[a + 1]
        for k in range(-12, 13):
            if (k + a) % 2:
                continue
            xs = []
            for yy in (ya, yb):
                f = (yy - vy) / (y1 - vy)
                xs.append((vx + (k - .5) * step * f, vx + (k + .5) * step * f))
            ctx.move_to(xs[0][0], ya)
            ctx.line_to(xs[0][1], ya)
            ctx.line_to(xs[1][1], yb)
            ctx.line_to(xs[1][0], yb)
            ctx.close_path()
    flat(ctx, C["floor_b"])
    line(ctx, [(0, y0), (W, y0)], 2.4, C["floor_ln"])

    # ---- elevator surround: stepped brass frame
    ox0, oy0, ox1, oy1 = OPEN
    for (pad, col) in ((44, C["brass2"]), (36, C["brass"]), (14, C["brass3"])):
        rpoly(ctx, [(ox0 - pad, oy0 - pad), (ox1 + pad, oy0 - pad), (ox1 + pad, oy1 + 6), (ox0 - pad, oy1 + 6)], 3)
        flat(ctx, col)
        rpoly(ctx, [(ox0 - pad, oy0 - pad), (ox1 + pad, oy0 - pad), (ox1 + pad, oy1 + 6), (ox0 - pad, oy1 + 6)], 3)
        ctx.set_source_rgba(*C["brass_ln"])
        ctx.set_line_width(2.2)
        ctx.stroke()
    # lintel
    rpoly(ctx, [(ox0 - 62, oy0 - 58), (ox1 + 62, oy0 - 58), (ox1 + 62, oy0 - 38), (ox0 - 62, oy0 - 38)], 3)
    flat(ctx, C["brass"])
    rpoly(ctx, [(ox0 - 62, oy0 - 58), (ox1 + 62, oy0 - 58), (ox1 + 62, oy0 - 38), (ox0 - 62, oy0 - 38)], 3)
    ctx.set_source_rgba(*C["brass_ln"])
    ctx.set_line_width(2.2)
    ctx.stroke()
    # mid-century cone sconces either side of the door, each throwing a painted fan of light up the wall
    for sx in (-1, 1):
        cx, cy = 640 + sx * 312, 286
        ctx.move_to(cx - 18, cy - 8)
        ctx.line_to(cx - 92, cy - 190)
        ctx.line_to(cx + 92, cy - 190)
        ctx.line_to(cx + 18, cy - 8)
        ctx.close_path()
        g = cairo.LinearGradient(0, cy - 8, 0, cy - 190)
        g.add_color_stop_rgba(0, 1, .97, .8, .32)
        g.add_color_stop_rgba(1, 1, .97, .8, 0)
        ctx.set_source(g)
        ctx.fill()
        rpoly(ctx, [(cx - 4, cy + 4), (cx + 4, cy + 4), (cx + 4, cy + 30), (cx - 4, cy + 30)], 2)
        flat(ctx, C["brass2"])
        rpoly(ctx, [(cx - 13, cy + 24), (cx + 13, cy + 24), (cx + 13, cy + 36), (cx - 13, cy + 36)], 3)
        flat(ctx, C["brass"])
        rpoly(ctx, [(cx - 13, cy + 24), (cx + 13, cy + 24), (cx + 13, cy + 36), (cx - 13, cy + 36)], 3)
        ctx.set_source_rgba(*C["brass_ln"])
        ctx.set_line_width(2)
        ctx.stroke()
        cone = [(cx - 26, cy - 22), (cx + 26, cy - 22), (cx + 8, cy + 8), (cx - 8, cy + 8)]
        rpoly(ctx, cone, [3, 3, 4, 4])
        flat(ctx, hx("#fbe9b8"))
        rpoly(ctx, cone, [3, 3, 4, 4])
        ctx.set_source_rgba(*C["brass_ln"])
        ctx.set_line_width(2.2)
        ctx.stroke()
        line(ctx, [(cx - 20, cy - 13), (cx + 20, cy - 13)], 2, hx("#d9b56a"))

    # ---- cab interior (one-point perspective)
    bx0, by0, bx1, by1 = BACK
    poly = lambda pts: (ctx.move_to(*pts[0]), [ctx.line_to(*p) for p in pts[1:]], ctx.close_path())
    poly([(ox0, oy0), (ox1, oy0), (bx1, by0), (bx0, by0)])
    flat(ctx, C["cab_ceil"])
    poly([(ox0, oy0), (bx0, by0), (bx0, by1), (ox0, oy1)])
    flat(ctx, C["cab_l"])
    poly([(ox1, oy0), (bx1, by0), (bx1, by1), (ox1, oy1)])
    flat(ctx, C["cab_r"])
    poly([(ox0, oy1), (bx0, by1), (bx1, by1), (ox1, oy1)])
    flat(ctx, C["cab_floor"])
    ctx.rectangle(bx0, by0, bx1 - bx0, by1 - by0)
    flat(ctx, C["cab_back"])
    # back wall panels
    pw = (bx1 - bx0) / 3
    for k in range(3):
        rpoly(ctx, [(bx0 + k * pw + 12, by0 + 18), (bx0 + (k + 1) * pw - 12, by0 + 18), (bx0 + (k + 1) * pw - 12, by1 - 20),
                    (bx0 + k * pw + 12, by1 - 20)], 6)
        flat(ctx, hx("#8574b2"))
        rpoly(ctx, [(bx0 + k * pw + 12, by0 + 18), (bx0 + (k + 1) * pw - 12, by0 + 18), (bx0 + (k + 1) * pw - 12, by1 - 20),
                    (bx0 + k * pw + 12, by1 - 20)], 6)
        ctx.set_source_rgba(*C["cab_ln"])
        ctx.set_line_width(2)
        ctx.stroke()
    # side wall panel insets (perspective trapezoids)
    for side in (0, 1):
        xa, xb = (ox0, bx0) if side == 0 else (ox1, bx1)
        f = lambda t, y_front, y_back: y_front + (y_back - y_front) * t
        ta, tb = .22, .8
        pts = [(xa + (xb - xa) * ta, f(ta, oy0, by0) + 22), (xa + (xb - xa) * tb, f(tb, oy0, by0) + 20),
               (xa + (xb - xa) * tb, f(tb, oy1, by1) - 22), (xa + (xb - xa) * ta, f(ta, oy1, by1) - 24)]
        poly(pts)
        ctx.set_source_rgba(*C["cab_ln"])
        ctx.set_line_width(2)
        ctx.stroke()
    # carpet border on the floor
    poly([(ox0 + 22, oy1 - 4), (bx0 + 10, by1 + 6), (bx1 - 10, by1 + 6), (ox1 - 22, oy1 - 4)])
    flat(ctx, C["cab_floor2"])
    # airbrushed light falloff: bright under the ceiling panel, dimmer toward the floor and the back corners
    for pts, ya, yb, a0, a1 in (([(ox0, oy0), (bx0, by0), (bx0, by1), (ox0, oy1)], oy0, oy1, .10, -.22),
                                ([(ox1, oy0), (bx1, by0), (bx1, by1), (ox1, oy1)], oy0, oy1, .06, -.26),
                                ([(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)], by0, by1, .12, -.20),
                                ([(ox0, oy1), (bx0, by1), (bx1, by1), (ox1, oy1)], by1, oy1, -.28, 0.0)):
        poly(pts)
        g = cairo.LinearGradient(0, ya, 0, yb)
        for u, a in ((0, a0), (1, a1)):
            if a >= 0:
                g.add_color_stop_rgba(u, 1, .95, .8, a)
            else:
                g.add_color_stop_rgba(u, .12, .06, .22, -a)
        ctx.set_source(g)
        ctx.fill()
    # ceiling light panel (glows)
    poly([(ox0 + 60, oy0 + 8), (ox1 - 60, oy0 + 8), (bx1 - 34, by0 - 8), (bx0 + 34, by0 - 8)])
    flat(ctx, C["lightpanel"])
    for k in range(1, 4):
        t = k / 4
        line(ctx, [(ox0 + 60 + (ox1 - ox0 - 120) * t, oy0 + 8), (bx0 + 34 + (bx1 - bx0 - 68) * t, by0 - 8)], 1.6, hx("#e8d08f"))
    # cab edges
    for pts in ([(ox0, oy0), (bx0, by0)], [(ox1, oy0), (bx1, by0)], [(ox0, oy1), (bx0, by1)], [(ox1, oy1), (bx1, by1)]):
        line(ctx, pts, 2.2, C["cab_ln"])
    ctx.rectangle(bx0, by0, bx1 - bx0, by1 - by0)
    ctx.set_source_rgba(*C["cab_ln"])
    ctx.set_line_width(2.2)
    ctx.stroke()
    # handrail on the back wall
    rpoly(ctx, [(bx0 + 4, 424), (bx1 - 4, 424), (bx1 - 4, 434), (bx0 + 4, 434)], 4)
    flat(ctx, C["brass"])
    rpoly(ctx, [(bx0 + 4, 424), (bx1 - 4, 424), (bx1 - 4, 434), (bx0 + 4, 434)], 4)
    ctx.set_source_rgba(*C["brass_ln"])
    ctx.set_line_width(2)
    ctx.stroke()
    # retracted door edges + threshold plate
    for xa, xb in ((ox0, ox0 + 14), (ox1 - 14, ox1)):
        ctx.rectangle(xa, oy0, xb - xa, oy1 - oy0)
        flat(ctx, C["brass2"])
        ctx.rectangle(xa, oy0, xb - xa, oy1 - oy0)
        ctx.set_source_rgba(*C["brass_ln"])
        ctx.set_line_width(2)
        ctx.stroke()
    rpoly(ctx, [(ox0 - 8, oy1 - 4), (ox1 + 8, oy1 - 4), (ox1 + 22, oy1 + 10), (ox0 - 22, oy1 + 10)], 2)
    flat(ctx, C["brass2"])
    rpoly(ctx, [(ox0 - 8, oy1 - 4), (ox1 + 8, oy1 - 4), (ox1 + 22, oy1 + 10), (ox0 - 22, oy1 + 10)], 2)
    ctx.set_source_rgba(*C["brass_ln"])
    ctx.set_line_width(2)
    ctx.stroke()

    # ---- floor-indicator half dial above the door
    dx, dy, dr = DIAL
    ctx.move_to(dx - dr, dy)
    ctx.arc(dx, dy, dr, math.pi, 2 * math.pi)
    ctx.close_path()
    flat(ctx, C["brass"])
    ctx.move_to(dx - dr, dy)
    ctx.arc(dx, dy, dr, math.pi, 2 * math.pi)
    ctx.close_path()
    ctx.set_source_rgba(*C["brass_ln"])
    ctx.set_line_width(2.4)
    ctx.stroke()
    ctx.move_to(dx - dr + 12, dy)
    ctx.arc(dx, dy, dr - 12, math.pi, 2 * math.pi)
    ctx.close_path()
    flat(ctx, C["dial_face"])
    ctx.move_to(dx - dr + 12, dy)
    ctx.arc(dx, dy, dr - 12, math.pi, 2 * math.pi)
    ctx.close_path()
    ctx.set_source_rgba(*C["brass_ln"])
    ctx.set_line_width(2)
    ctx.stroke()
    for k in range(9):
        a = math.pi + k * math.pi / 8
        r0, r1 = (dr - 30, dr - 17) if k % 2 == 0 else (dr - 24, dr - 17)
        line(ctx, [(dx + r0 * math.cos(a), dy + r0 * math.sin(a)), (dx + r1 * math.cos(a), dy + r1 * math.sin(a))], 2.6 if k % 2 == 0 else 1.8,
             C["brass_ln"])

    # ---- potted plant (left): sansevieria in an atomic-age cone planter
    px, py = 236, 668
    for lx, ang in ((-30, -.38), (0, 0), (28, .32)):  # splayed legs
        line(ctx, [(px + lx * .6, py - 36), (px + lx * .6 + 60 * math.sin(ang), py)], 4, C["pot_ln"])
    leaves = [(-50, -150, -.55), (-24, -236, -.22), (4, -268, .04), (30, -214, .3), (52, -140, .62)]
    for k, (lx, lh, lean) in enumerate(leaves):
        base = (px + lx * .35, py - 86)
        tip = (base[0] + abs(lh) * math.sin(lean), base[1] + lh)
        bw = 17
        pts = [(base[0] - bw, base[1]), (tip[0], tip[1]), (base[0] + bw, base[1])]
        mid = [(base[0] - 5, base[1]), (tip[0] + (base[0] - tip[0]) * .18, tip[1] + (base[1] - tip[1]) * .18), (base[0] + 5, base[1])]
        rpoly(ctx, pts, [3, 4, 3])
        flat(ctx, C["leaf"] if k % 2 == 0 else hx("#2a7c4c"))
        ctx.save()
        ctx.translate(-2, 2)
        rpoly(ctx, mid, [2, 2, 2])
        flat(ctx, C["leaf2"])
        ctx.restore()
        rpoly(ctx, pts, [3, 4, 3])
        ctx.set_source_rgba(*C["leaf_ln"])
        ctx.set_line_width(2.2)
        ctx.stroke()
    pot = [(px - 52, py - 96), (px + 52, py - 96), (px + 30, py - 34), (px - 30, py - 34)]
    rpoly(ctx, pot, [5, 5, 8, 8])
    flat(ctx, C["pot"])
    ctx.rectangle(px - 46, py - 80, 92, 9)
    ctx.save()
    rpoly(ctx, pot, [5, 5, 8, 8])
    ctx.clip()
    ctx.rectangle(px - 60, py - 78, 120, 9)
    flat(ctx, C["floor_b"])
    ctx.restore()
    rpoly(ctx, pot, [5, 5, 8, 8])
    ctx.set_source_rgba(*C["pot_ln"])
    ctx.set_line_width(2.2)
    ctx.stroke()

    # ---- call-button plate (right of the door)
    cx0, cy0 = 914, 352
    rpoly(ctx, [(cx0, cy0), (cx0 + 40, cy0), (cx0 + 40, cy0 + 92), (cx0, cy0 + 92)], 9)
    flat(ctx, C["brass"])
    rpoly(ctx, [(cx0, cy0), (cx0 + 40, cy0), (cx0 + 40, cy0 + 92), (cx0, cy0 + 92)], 9)
    ctx.set_source_rgba(*C["brass_ln"])
    ctx.set_line_width(2.2)
    ctx.stroke()
    for k, up in ((0, True), (1, False)):
        by = cy0 + 26 + k * 40
        ell(ctx, cx0 + 20, by, 12, 12)
        flat(ctx, C["dial_face"])
        ell(ctx, cx0 + 20, by, 12, 12)
        ctx.set_source_rgba(*C["brass_ln"])
        ctx.set_line_width(2)
        ctx.stroke()
        s = -1 if up else 1
        ctx.move_to(cx0 + 20, by + 5 * s)
        ctx.line_to(cx0 + 26, by - 4 * s)
        ctx.line_to(cx0 + 14, by - 4 * s)
        ctx.close_path()
        flat(ctx, C["brass_ln"])

    # ---- starburst wall clock (right) — ticks only, no numerals
    kx, ky = 1118, 250
    for k in range(24):
        a = k * math.pi / 12
        r1 = 92 if k % 2 == 0 else 70
        line(ctx, [(kx + 40 * math.cos(a), ky + 40 * math.sin(a)), (kx + r1 * math.cos(a), ky + r1 * math.sin(a))], 4.5 if k % 2 == 0 else 3,
             C["brass2"])
        ell(ctx, kx + r1 * math.cos(a), ky + r1 * math.sin(a), 5 if k % 2 == 0 else 3.5, 5 if k % 2 == 0 else 3.5)
        flat(ctx, C["brass"])
    ell(ctx, kx, ky, 42, 42)
    flat(ctx, C["brass"])
    ell(ctx, kx, ky, 42, 42)
    ctx.set_source_rgba(*C["brass_ln"])
    ctx.set_line_width(2.2)
    ctx.stroke()
    ell(ctx, kx, ky, 33, 33)
    flat(ctx, hx("#235d6a"))
    for k in range(12):
        a = k * math.pi / 6
        line(ctx, [(kx + 25 * math.cos(a), ky + 25 * math.sin(a)), (kx + 30 * math.cos(a), ky + 30 * math.sin(a))], 2.2, C["brass3"])
    line(ctx, [(kx, ky), (kx + 15 * math.cos(-2.2), ky + 15 * math.sin(-2.2))], 3.4, C["brass3"])   # ~10:10
    line(ctx, [(kx, ky), (kx + 24 * math.cos(-.95), ky + 24 * math.sin(-.95))], 2.4, C["brass3"])
    ell(ctx, kx, ky, 3.5, 3.5)
    flat(ctx, C["brass3"])


def paint_finish(img):
    """Gouache card: soft light falloff + low-frequency paint mottling + paper tooth.  Background only."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx /= w
    yy /= h
    # warm light spilling out of the cab, cooler/darker at the frame edges and ceiling
    glow = np.exp(-(((xx - .5) / .42) ** 2 + ((yy - .55) / .7) ** 2))
    img = img * (0.88 + 0.14 * glow[..., None])
    img = img * (1 - .16 * np.clip((yy - .84) / .16, 0, 1) ** 1.5)[..., None]      # airbrushed floor falloff
    lum = img @ np.array([.299, .587, .114], np.float32)
    img = lum[..., None] + (img - lum[..., None]) * .86                            # BG a notch quieter than the cels
    rng = np.random.default_rng(7)
    m = cv2.resize(rng.normal(0, 1, (18, 32)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    m2 = cv2.resize(rng.normal(0, 1, (90, 160)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    fib = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=3.0, sigmaY=.8)
    tex = m * .018 + m2 * .010 + fib * .05
    return np.clip(img * (1 + tex[..., None]), 0, 1)


_BG = None


def background():
    """Static painted card as a cairo surface (premultiplied, opaque), cached per process."""
    global _BG
    if _BG is None:
        s, ctx = new_surface()
        bg_static(ctx)
        a = surf_np(s)[..., :3]
        a = paint_finish(a)
        out = np.zeros((H * SS, W * SS, 4), np.uint8)
        out[..., :3] = (a[..., ::-1] * 255 + .5).astype(np.uint8)
        out[..., 3] = 255
        bs = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * SS, H * SS)
        np.ndarray((H * SS, W * SS, 4), np.uint8, bs.get_data())[:] = out
        bs.mark_dirty()
        _BG = bs
    return _BG


def bg_dynamic(ctx, st):
    """Painted elements that change: the dial lamps/needle and the cast shadow."""
    dx, dy, dr = DIAL
    for sx, up in ((-1, False), (1, True)):
        lx = dx + sx * (dr + 26)
        on = st.get("lamp", 0.0) if up else 0.0
        if on > 0:
            ell(ctx, lx, dy - 18, 26, 26)
            ctx.set_source_rgba(*hx("#fff6a0", .35 * on))
            ctx.fill()
        s = -1 if up else 1
        pts = [(lx, dy - 20 + 13 * s), (lx + 15, dy - 20 - 10 * s), (lx - 15, dy - 20 - 10 * s)]
        rpoly(ctx, pts, 2.5)
        ctx.set_source_rgba(*mix(C["lamp_off"], C["lamp_on"], on))
        ctx.fill_preserve()
        ctx.set_source_rgba(*C["brass_ln"])
        ctx.set_line_width(2)
        ctx.stroke()
    a = math.pi + .2 + st.get("needle", 0.0)
    line(ctx, [(dx, dy), (dx + (dr - 22) * math.cos(a), dy + (dr - 22) * math.sin(a))], 3.4, hx("#3a2a12"))
    ell(ctx, dx, dy - 1, 7, 7)
    flat(ctx, C["brass2"])
    # flat cast shadow under the hero and the lever stand
    hx_, hy_ = HERO
    ell(ctx, hx_ + 4 + st.get("x", 0) * .5, hy_ + 1, 96 * st.get("shadow", 1.0), 11)
    flat(ctx, hx("#2c2346", .75))
    ell(ctx, LEVER_PIVOT[0] + 2, hy_ - 4, 30, 6)
    flat(ctx, hx("#2c2346", .75))


# ============================================================================================ cel layer: props
def lever_stand(ctx, ang, smear=0.0):
    px, py = LEVER_PIVOT
    # pedestal column
    rpoly(ctx, [(px - 9, py + 6), (px + 9, py + 6), (px + 12, HERO[1] - 4), (px - 12, HERO[1] - 4)], 3)
    ink(ctx, C["brass2"])
    rpoly(ctx, [(px - 22, HERO[1] - 12), (px + 22, HERO[1] - 12), (px + 26, HERO[1] - 2), (px - 26, HERO[1] - 2)], 3)
    ink(ctx, C["brass"])
    # half-moon switch box
    ctx.move_to(px - 40, py + 8)
    ctx.arc(px, py + 8, 40, math.pi, 2 * math.pi)
    ctx.close_path()
    ink(ctx, C["brass"])
    ctx.move_to(px - 29, py + 8)
    ctx.arc(px, py + 8, 29, math.pi, 2 * math.pi)
    ctx.close_path()
    ink(ctx, hx("#33293f"), 2.6)
    for k in range(5):                          # notch marks
        a = math.pi + (k + 1) * math.pi / 6
        line(ctx, [(px + 22 * math.cos(a), py + 8 + 22 * math.sin(a)), (px + 28 * math.cos(a), py + 8 + 28 * math.sin(a))], 2.2, C["brass3"])
    # lever
    tip = (px + LEVER_LEN * math.sin(ang), py - LEVER_LEN * math.cos(ang))
    if smear != 0:                              # speed arcs trailing the knob (tapered: thick at the knob end)
        sg = 1 if smear > 0 else -1
        for k, (rr, frac) in enumerate(((LEVER_LEN + 30, 1.0), (LEVER_LEN + 16, .8), (LEVER_LEN - 24, .9))):
            a_end = -math.pi / 2 + ang - sg * .3
            a_start = a_end - sg * abs(smear) * frac
            n = 16
            for j in range(n):
                u0, u1 = j / n, (j + 1) / n
                aa, ab = a_start + (a_end - a_start) * u0, a_start + (a_end - a_start) * u1
                ctx.new_sub_path()
                (ctx.arc if sg > 0 else ctx.arc_negative)(px, py, rr, aa, ab)
                ctx.set_source_rgba(*INK)
                ctx.set_line_width(.6 + 3.6 * u1)
                ctx.stroke()
    tube(ctx, [(px, py + 2), tip], 7, C["brass3"], w=3.4)
    ell(ctx, px, py + 4, 9, 9)
    ink(ctx, C["brass2"], 3.4)
    ell(ctx, tip[0], tip[1], 12, 12)
    ink(ctx, C["knob"], 3.4)
    ell(ctx, tip[0] - 4, tip[1] - 4, 3.5, 3.5)
    flat(ctx, hx("#8a8098"))
    return tip


# ============================================================================================ cel layer: Mr. Hoist
def ik(s, t, l1, l2, bend=1):
    s, t = np.asarray(s, float), np.asarray(t, float)
    d = t - s
    L = np.hypot(*d)
    L = min(max(L, 1e-3), l1 + l2 - 1e-3)
    u = d / (np.hypot(*d) + 1e-9)
    t = s + u * L
    a = (l1 ** 2 - l2 ** 2 + L ** 2) / (2 * L)
    hgt = math.sqrt(max(l1 ** 2 - a ** 2, 0))
    nrm = np.array([-u[1], u[0]]) * bend
    return s + u * a + nrm * hgt, t


def hand(ctx, x, y, ang, kind, s=1.0):
    """Geometric mitten hands.  Origin at the wrist, +x along the forearm."""
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(ang)
    ctx.scale(s, s)
    if kind == "fist":
        rpoly(ctx, [(-2, -15), (26, -16), (32, -6), (30, 14), (-2, 15)], [5, 9, 6, 9, 5])
        ink(ctx, C["skin"])
        for yy in (-6, 4):
            line(ctx, [(22, yy), (30, yy)], 3)
        rpoly(ctx, [(4, -15), (20, -22), (24, -16), (10, -8)], 4)      # thumb over the fingers
        ink(ctx, C["skin"])
    elif kind == "grip":                         # seen knuckles-on, wrapped round the lever knob
        for yy in (-11, 0, 11):                  # curled finger tips on the far side of the knob
            rpoly(ctx, [(20, yy - 6), (33, yy - 6), (36, yy), (33, yy + 6), (20, yy + 6)], [2, 4, 3, 4, 2])
            ink(ctx, C["skin"], 3.4)
        rpoly(ctx, [(-4, -18), (26, -18), (26, 18), (-4, 18)], [6, 4, 4, 6])
        ink(ctx, C["skin"])
        rpoly(ctx, [(2, 6), (24, 10), (30, 18), (6, 20)], [4, 5, 5, 4])  # thumb across the top (hand is turned by pi)
        ink(ctx, C["skin"], 3.4)
    elif kind == "point":                        # index finger raised: "Attention, please!"
        ctx.rotate(-ang + (ang + math.pi / 2) * .3)          # palm to camera, kept nearly upright
        rpoly(ctx, [(-18, -26), (-18, -68), (-4, -68), (-4, -26)], [2, 7, 7, 2])     # index finger, thumb side
        ink(ctx, C["skin"])
        rpoly(ctx, [(-18, -36), (14, -38), (19, -24), (16, 0), (-15, 2)], [6, 8, 6, 6, 6])
        ink(ctx, C["skin"])
        line(ctx, [(-1, -27), (17, -27)], 3)                                         # curled fingers
        line(ctx, [(-1, -16), (17, -16)], 3)
        rpoly(ctx, [(-17, -13), (5, -21), (9, -13), (-12, -2)], 4)                   # thumb across the front
        ink(ctx, C["skin"], 3.4)
    elif kind == "open":                         # palm out, fingers spread (exasperation / alarm)
        for j, a in enumerate((-.55, -.18, .18, .52)):
            ctx.save()
            ctx.translate(22, 0)
            ctx.rotate(a)
            rpoly(ctx, [(0, -5.5), (26, -5.5), (26, 5.5), (0, 5.5)], [2, 5, 5, 2])
            ink(ctx, C["skin"], 3.4)
            ctx.restore()
        rpoly(ctx, [(-2, -16), (26, -18), (30, 0), (26, 18), (-2, 16)], [5, 8, 6, 8, 5])
        ink(ctx, C["skin"])
        ctx.save()
        ctx.translate(12, 16)
        ctx.rotate(.9)
        rpoly(ctx, [(0, -5), (20, -5), (20, 5), (0, 5)], [2, 5, 5, 2])
        ink(ctx, C["skin"], 3.4)
        ctx.restore()
    elif kind == "flat":                         # salute: fingers together, straight and stiff
        rpoly(ctx, [(-2, -13), (40, -14), (50, -6), (50, 6), (40, 12), (-2, 13)], [4, 6, 8, 8, 6, 4])
        ink(ctx, C["skin"])
        line(ctx, [(30, -2), (46, -2)], 2.6)
        rpoly(ctx, [(2, 10), (16, 20), (22, 14), (12, 6)], 4)
        ink(ctx, C["skin"])
    ctx.restore()


def head(ctx, p):
    """Head origin = bottom of the chin.  Wide boxy bell shape, slab brows, wedge nose, block moustache, pillbox cap."""
    m = p["mouth"]
    eye = p["eye"]
    br = p["brow"]
    face_dx = -8                                 # features turned a touch toward the controls (screen left)
    # ears (behind)
    for sx in (-1, 1):
        ell(ctx, sx * 76, -84, 17, 22)
        ink(ctx, C["skin"])
        line(ctx, [(sx * 74, -94), (sx * 82, -84), (sx * 74, -74)], 2.6)
    # white side tufts: zigzag spikes
    for sx in (-1, 1):
        pts = [(sx * 60, -128), (sx * 98, -126), (sx * 86, -114), (sx * 106, -104), (sx * 88, -96), (sx * 100, -82), (sx * 66, -88)]
        rpoly(ctx, pts, 2.5)
        ink(ctx, C["hairw"])
    # head mass (jaw drops when talking)
    jaw = 16 * m
    hp = [(-54, 0 + jaw), (54, 0 + jaw), (82, -34), (80, -140), (68, -154), (-68, -154), (-80, -140), (-82, -34)]
    rpoly(ctx, hp, [22, 22, 30, 10, 10, 10, 10, 30])
    ink(ctx, C["skin"])
    # forehead creases
    for k, (w0, yy) in enumerate(((22, -134), (16, -126))):
        line(ctx, [(face_dx - w0, yy + 2), (face_dx + w0, yy)], 2.4)
    # eyes: small upright black ovals with a white glint; blink = ovals flatten into a lid line
    big = p.get("eye_big", 1.0)
    for k, ex in enumerate((-34 + face_dx, 22 + face_dx)):
        ey = -88 + 3 * p.get("look_y", 0.0)
        look = p["look"]
        e = p.get("eye_l" if k == 0 else "eye_r", eye)
        if p.get("swirl"):                       # dizzy: square-ish spirals
            pts, r, a = [], 13.0, k * 1.3
            for j in range(26):
                pts.append((ex + r * math.cos(a), ey + r * math.sin(a)))
                a += .62
                r *= .9
            line(ctx, pts, 2.8)
        elif e > .25:
            ell(ctx, ex + look * 3, ey, 9.5 * big, 13.5 * e * big)
            flat(ctx, INK)
            ell(ctx, ex + look * 3 - 2.5 * big, ey - 4 * e * big, 2.6 * big, 3.2 * e * big)
            flat(ctx, C["white"])
        elif p.get("happy_eyes"):                # closed, upturned (pleased)
            line(ctx, [(ex - 10, ey + 3), (ex, ey - 4), (ex + 10, ey + 3)], 3.4)
        else:
            line(ctx, [(ex - 10, ey + 1), (ex, ey + 4), (ex + 10, ey + 1)], 3.4)
    if p.get("stress"):                          # twitchy stress marks at the temple
        for j in range(3):
            a = -.9 + j * .45
            line(ctx, [(60 + 10 * math.cos(a), -112 + 10 * math.sin(a)), (60 + 22 * math.cos(a), -112 + 22 * math.sin(a))], 2.6)
    if p.get("sweat"):
        rpoly(ctx, [(-70, -128), (-62, -108), (-70, -100), (-78, -108)], [1, 6, 6, 6])
        ink(ctx, hx("#bfe6ff"), 2.6)
    # slab brows: white, angular, outer ends kicked up; br raises (+) or knits (-)
    for sx, ex in ((-1, -34 + face_dx), (1, 22 + face_dx)):
        lift = -12 * br + (-3 if sx > 0 else 0)
        tilt = 7 * p["brow_tilt"]              # + knits the inner ends down (cross), - lifts them (worried)
        inner = (ex - sx * 17, -110 + lift + tilt)
        outer = (ex + sx * 20, -112 + lift)
        pts = [(inner[0], inner[1] + 6), (inner[0], inner[1] - 7), (outer[0], outer[1] - 9),
               (outer[0] + sx * 15, outer[1] - 20), (outer[0] + sx * 10, outer[1] - 6), (outer[0] + sx * 16, outer[1] - 1),
               (outer[0], outer[1] + 6)]
        rpoly(ctx, pts, [3, 3, 3, 1.5, 2, 1.5, 3])
        ink(ctx, C["hairw"], 3.8)
    # mouth (only when open), drawn before the moustache
    if m > .04:
        mw = 26 * p["mwide"]
        mp = [(face_dx - mw, -34), (face_dx + mw, -34), (face_dx + mw * .7, -14 + jaw * .9), (face_dx - mw * .7, -14 + jaw * .9)]
        rpoly(ctx, mp, [4, 4, 9, 9])
        ink(ctx, C["mouth"], 3.6)
        ctx.save()
        rpoly(ctx, mp, [4, 4, 9, 9])
        ctx.clip()
        ell(ctx, face_dx + 4, -10 + jaw * .9, mw * .6, 9)
        flat(ctx, C["tongue"])
        ctx.restore()
        rpoly(ctx, mp, [4, 4, 9, 9])
        ctx.set_source_rgba(*INK)
        ctx.set_line_width(3.6)
        ctx.stroke()
    elif p.get("mstyle") == "grit":             # strained, toothy smile
        gp = [(face_dx - 30, -30), (face_dx + 30, -30), (face_dx + 26, -12), (face_dx - 26, -12)]
        rpoly(ctx, gp, [4, 4, 6, 6])
        ink(ctx, C["white"], 3.4)
        line(ctx, [(face_dx - 27, -21), (face_dx + 27, -21)], 2.4)
        for j in (-15, 0, 15):
            line(ctx, [(face_dx + j, -29), (face_dx + j, -13)], 2.0)
    elif p.get("mstyle") == "o":                # little round snore mouth
        ell(ctx, face_dx + 2, -18, 7, 8)
        ink(ctx, C["mouth"], 3.2)
    elif p.get("mstyle") == "smile":
        line(ctx, [(face_dx - 18, -24), (face_dx - 6, -16), (face_dx + 8, -16), (face_dx + 18, -24)], 3.0)
    elif p.get("mstyle") == "wavy":
        line(ctx, [(face_dx - 16, -18), (face_dx - 8, -23), (face_dx, -17), (face_dx + 8, -23), (face_dx + 16, -18)], 3.0)
    else:
        line(ctx, [(face_dx - 12, -20), (face_dx, -17), (face_dx + 12, -20)], 3.0)   # closed lower lip
    # block moustache, wider than the face, with a bristly zigzag hem
    lift = -5 * m - 3 * p.get("bristle", 0)
    mx = face_dx
    half = [(40, -64), (80, -74), (108, -94), (98, -62), (74, -42), (62, -48), (46, -32), (32, -38), (14, -28)]
    dr = p.get("droop", 0.0)
    if dr:                                       # tips sag down past the chin
        sag = [(40, -62), (72, -58), (88, -14), (74, -18), (62, -30), (54, -36), (42, -26), (30, -34), (14, -26)]
        half = [(a[0] + (b[0] - a[0]) * dr, a[1] + (b[1] - a[1]) * dr) for a, b in zip(half, sag)]
    must = [(0, -68)] + half + [(0, -36)] + [(-x, y) for x, y in half[::-1]]
    must = [(x + mx, y + lift) for x, y in must]
    hr = [8, 6, 2, 6, 4, 1.5, 4, 1.5, 4]
    rpoly(ctx, must, [8] + hr + [5] + hr[::-1])
    ink(ctx, C["white"])
    # wedge nose sitting on the moustache
    nose = [(-18 + face_dx, -100), (8 + face_dx, -60 + lift * .5), (-46 + face_dx, -56 + lift * .5)]
    rpoly(ctx, nose, [9, 12, 12])
    ink(ctx, C["nose"])
    # pillbox cap, tilted
    ctx.save()
    ctx.translate(10 + p.get("cap_dx", 0.0), -146 - p.get("cap_dy", 0.0))
    ctx.rotate(-.12 + p.get("cap_rot", 0.0))
    cw, ch, ce = 50, 50, 12
    ctx.move_to(-cw, -ch)
    ctx.line_to(-cw, 0)
    ctx.curve_to(-cw * .6, ce * 1.3, cw * .6, ce * 1.3, cw, 0)
    ctx.line_to(cw, -ch)
    ctx.close_path()
    ink(ctx, C["red"])
    ell(ctx, 0, -ch, cw, ce)                     # top
    ink(ctx, C["red2"])
    ctx.save()                                   # gold band
    ctx.move_to(-cw, -ch)
    ctx.line_to(-cw, 0)
    ctx.curve_to(-cw * .6, ce * 1.3, cw * .6, ce * 1.3, cw, 0)
    ctx.line_to(cw, -ch)
    ctx.close_path()
    ctx.clip()
    ctx.move_to(-cw - 4, -14)
    ctx.curve_to(-cw * .6, -14 + ce * 1.3, cw * .6, -14 + ce * 1.3, cw + 4, -14)
    ctx.line_to(cw + 4, 10)
    ctx.line_to(-cw - 4, 10)
    ctx.close_path()
    flat(ctx, C["gold"])
    ctx.restore()
    ctx.move_to(-cw, -14)
    ctx.curve_to(-cw * .6, -14 + ce * 1.3, cw * .6, -14 + ce * 1.3, cw, -14)
    ctx.set_source_rgba(*INK)
    ctx.set_line_width(3.2)
    ctx.stroke()
    ctx.move_to(-cw, -ch)
    ctx.line_to(-cw, 0)
    ctx.curve_to(-cw * .6, ce * 1.3, cw * .6, ce * 1.3, cw, 0)
    ctx.line_to(cw, -ch)
    ctx.set_source_rgba(*INK)
    ctx.set_line_width(LW)
    ctx.stroke()
    ell(ctx, 0, -ch - 4, 8, 4.5)                 # button on top
    ink(ctx, C["gold"], 3)
    ctx.restore()


ARM_L = (56, 56)   # upper, fore


def hoist(ctx, p):
    """Draw Mr. Hoist.  p = pose dict (see NEUTRAL)."""
    X, Y = HERO
    ctx.save()
    ctx.translate(X + p["x"], Y)
    ctx.rotate(p["lean"])
    ctx.scale(p["sx"], p["sy"])
    # ---- legs + shoes (heels together, toes out: standing at attention)
    for sx in (-1, 1):
        tube(ctx, [(sx * 18, -70), (sx * 17, -12)], 17, C["navy"], cap=cairo.LINE_CAP_BUTT)
        line(ctx, [(sx * 18 + sx * 5, -64), (sx * 17 + sx * 5, -16)], 2.6, C["gold"])     # trouser stripe
        shoe = [(sx * 4, -16), (sx * 30, -14), (sx * 58, -4), (sx * 56, 2), (sx * 2, 2)]
        rpoly(ctx, shoe, [4, 8, 6, 3, 4])
        ink(ctx, C["shoe"])
        line(ctx, [(sx * 20, -9), (sx * 40, -7)], 2.2, hx("#6e6a80"))                     # shine
    # ---- tunic: barrel-chested bell with a pointed front hem
    body = [(-78, -60), (0, -46), (78, -60), (86, -128), (64, -212), (-64, -212), (-86, -128)]
    rpoly(ctx, body, [8, 8, 8, 34, 20, 20, 34])
    ink(ctx, C["red"])
    ctx.save()                                   # gold hem piping
    rpoly(ctx, body, [8, 8, 8, 34, 20, 20, 34])
    ctx.clip()
    rpoly(ctx, [(-90, -68), (0, -54), (90, -68), (90, -30), (-90, -30)], 2)
    flat(ctx, C["gold"])
    ctx.restore()
    line(ctx, [(-84, -69), (0, -55), (84, -69)], 3.0)
    rpoly(ctx, body, [8, 8, 8, 34, 20, 20, 34])
    ctx.set_source_rgba(*INK)
    ctx.set_line_width(LW)
    ctx.stroke()
    for yy in (-176, -140, -104):                # double row of buttons
        for sx in (-1, 1):
            ell(ctx, sx * 24, yy, 6.5, 6.5)
            ink(ctx, C["gold"], 3)
    # collar
    rpoly(ctx, [(-36, -222), (36, -222), (34, -200), (-34, -200)], 5)
    ink(ctx, C["gold"])
    # ---- arms below head level (lever arm + hip arm), then epaulettes
    sh = {-1: np.array([-60.0, -192.0]), 1: np.array([60.0, -192.0])}
    arms = []
    if p.get("rarm_over"):
        sh[1] = np.array([62.0, -202.0])         # the shoulder rides up with a raised arm
    for side, key in ((-1, "larm"), (1, "rarm")):
        tgt, bend, kind, hang = p[key]
        lens = (62, 64) if (side == 1 and p.get("rarm_over")) else ARM_L   # rubbery cartoon reach for the raised arm
        el, wr = ik(sh[side], tgt, *lens, bend)
        arms.append((side, el, wr, kind, hang, p.get(key + "_over", False)))

    def draw_arm(side, el, wr, kind, hang):
        fa = math.atan2(wr[1] - el[1], wr[0] - el[0])
        cuff = wr - np.array([math.cos(fa), math.sin(fa)]) * 9
        tube(ctx, [sh[side], el, cuff], 26, C["red"])
        ctx.save()                               # gold cuff band
        ctx.translate(*cuff)
        ctx.rotate(fa)
        rpoly(ctx, [(-8, -15), (8, -15), (8, 15), (-8, 15)], 3)
        ink(ctx, C["gold"], 3.4)
        ctx.restore()
        hand(ctx, wr[0] + math.cos(fa) * 6, wr[1] + math.sin(fa) * 6, fa if hang is None else hang, kind)

    for side, el, wr, kind, hang, over in arms:
        if not over:
            draw_arm(side, el, wr, kind, hang)
    for sx in (-1, 1):                           # epaulettes with fringe
        ep = [(sx * 30, -222), (sx * 74, -216), (sx * 88, -202), (sx * 80, -194), (sx * 30, -200)]
        for k in range(4):
            fx = sx * (52 + k * 9)
            rpoly(ctx, [(fx - 3.5, -200), (fx + 3.5, -200), (fx + 3.5, -184 + (k % 2) * 3), (fx - 3.5, -184 + (k % 2) * 3)], 2)
            ink(ctx, C["gold"], 2.6)
        rpoly(ctx, ep, [6, 8, 6, 6, 6])
        ink(ctx, C["gold"], 3.6)
    # ---- head
    ctx.save()
    ctx.translate(p["hx"], -206 + p["hy"])
    ctx.rotate(p["hrot"])
    head(ctx, p)
    ctx.restore()
    for side, el, wr, kind, hang, over in arms:
        if over:
            draw_arm(side, el, wr, kind, hang)
    ctx.restore()


def body_to_world(pt, p):
    """Body-local point -> screen (for hands that must land on world props)."""
    X, Y = HERO
    c, s = math.cos(p["lean"]), math.sin(p["lean"])
    x, y = pt[0] * p["sx"], pt[1] * p["sy"]
    return np.array([X + p["x"] + c * x - s * y, Y + s * x + c * y])


def world_to_body(pt, p):
    X, Y = HERO
    c, s = math.cos(-p["lean"]), math.sin(-p["lean"])
    x, y = pt[0] - X - p["x"], pt[1] - Y
    return np.array([(c * x - s * y) / p["sx"], (s * x + c * y) / p["sy"]])


# ============================================================================================ poses + timing
NEUTRAL = dict(x=0.0, lean=0.0, sx=1.0, sy=1.0, hx=0.0, hy=-2.0, hrot=-.04, brow=0.3, brow_tilt=-.3, eye=1.0, look=-.15, mouth=0.0,
               mwide=1.0, bristle=0.0, cap_rot=0.0, lever=0.0, lamp=0.0, needle=0.0,
               rarm=((84.0, -112.0), 1, "fist", None))   # right arm: fist on the hip, elbow out


def render_layers(p, lever_smear=0.0, extra=None, post=None):
    """Return the final 1x RGB float image for pose p."""
    bs, bctx = new_surface()
    bctx.set_source_surface(background(), 0, 0)
    bctx.get_source().set_matrix(cairo.Matrix(xx=SS, yy=SS))
    bctx.paint()
    bg_dynamic(bctx, dict(lamp=p["lamp"], needle=p["needle"], x=p["x"]))
    cs, cctx = new_surface()
    tip = lever_stand(cctx, p["lever"], lever_smear)
    pp = dict(p)
    # left (lever) hand rides the knob
    if p.get("grip", 1.0) > .5:
        tb = world_to_body(tip, p)
        pp["larm"] = ((tb[0] + 15, tb[1] + 3), -1, "grip", math.pi + .05)
    else:
        pp["larm"] = p["larm_free"]
    if extra:
        extra(cctx, pp)
    hoist(cctx, pp)
    if post:
        post(cctx, pp)
    bg = surf_np(bs)[..., :3]
    cel = surf_np(cs)
    a = cel[..., 3]
    k = int(round(SIL * SS))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    d = cv2.dilate(a, ker)
    d = cv2.GaussianBlur(d, (0, 0), .6)
    out = cel[..., :3] + (1 - a)[..., None] * (1 - d)[..., None] * bg + (1 - a)[..., None] * d[..., None] * np.array(INK[:3], np.float32)
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA)


def finish(img, i=0):
    """Broadcast finish: the gentlest chroma softening and a whisper of grain (keeps the line crisp)."""
    yuv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2YCrCb)
    yuv[:, :, 1:] = cv2.GaussianBlur(yuv[:, :, 1:], (0, 0), sigmaX=1.1, sigmaY=.2)
    img = cv2.cvtColor(yuv, cv2.COLOR_YCrCb2RGB)
    rng = np.random.default_rng(500 + i // 2)
    g = cv2.resize(rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32), (W, H))
    return np.clip(img + g[..., None] * .006, 0, 1)


def to8(img):
    return (np.clip(img, 0, 1) * 255 + .5).astype(np.uint8)


def key_frame():
    return frame(0)


# ============================================================================================ animation (on twos, 12 frames = 1 beat)
def _ease(x, kind):
    x = min(max(x, 0.0), 1.0)
    if kind == "io":
        return x * x * (3 - 2 * x)
    if kind == "out":
        return 1 - (1 - x) ** 3
    if kind == "in":
        return x ** 3
    if kind == "step":
        return 1.0 if x >= 1 else 0.0
    return x


def kf(chan, f):
    """chan: [(frame, value[, ease-into-this-key])]; values may be numbers or tuples."""
    if f <= chan[0][0]:
        return chan[0][1]
    for a, b in zip(chan, chan[1:]):
        if f <= b[0]:
            e = b[2] if len(b) > 2 else "io"
            if e == "step":
                return b[1] if f >= b[0] else a[1]
            x = _ease((f - a[0]) / max(b[0] - a[0], 1e-6), e)
            va, vb = a[1], b[1]
            if isinstance(va, tuple):
                return tuple(u + (v - u) * x for u, v in zip(va, vb))
            return va + (vb - va) * x
    return chan[-1][1]


# speech: (start s, end s, openness, width, stressed) — shared with the sound module
SYL = [(1.75, 1.88, .75, .75, 0, "o"), (1.88, 2.00, .45, 1.15, 0, "i"), (2.02, 2.32, 1.0, .85, 1, "u"),
       (2.50, 2.60, .5, 1.1, 0, "e"), (2.60, 2.82, .45, 1.3, 1, "i"), (2.82, 2.91, .4, 1.25, 0, "i"),
       (2.93, 3.02, .4, 1.25, 0, "i"), (3.02, 3.11, .55, 1.0, 0, "e"), (3.11, 3.20, .7, .8, 0, "u"),
       (3.25, 3.58, 1.0, .9, 1, "o")]

T_YANK, T_DING, T_POINT, T_SALUTE, T_BACK, T_BUTTON = 24, 26, 36, 90, 104, 108

LEVER = [(0, 0.0), (14, 0.0), (20, -.2, "io"), (23, -.2), (24, .62, "step"), (26, 1.1, "step"), (28, .97, "out"), (31, .95, "io"),
         (102, .95), (104, -.14, "step"), (106, .08, "step"), (108, 0.0, "io")]
SX = [(0, 1.0), (14, 1.0), (20, 1.035), (23, 1.035), (24, .955, "step"), (26, .975, "step"), (30, 1.0, "io"), (86, 1.0), (88, 1.02, "io"),
      (90, .965, "step"), (94, .985, "out"), (102, .985), (104, 1.02, "step"), (108, 1.0, "io")]
SY = [(0, 1.0), (14, 1.0), (20, .955), (23, .955), (24, 1.065, "step"), (26, 1.04, "step"), (30, 1.0, "io"), (86, 1.0), (88, .975, "io"),
      (90, 1.055, "step"), (94, 1.03, "out"), (102, 1.03), (104, .975, "step"), (108, 1.0, "io")]
LEAN = [(0, 0.0), (14, 0.0), (20, .04), (23, .04), (24, -.035, "step"), (26, -.02, "step"), (30, 0.0, "io"), (102, 0.0), (104, .015, "step"),
        (108, 0.0, "io")]
HY = [(0, -2.0), (14, -2.0), (20, 9.0), (23, 9.0), (24, -12.0, "step"), (26, -14.0, "step"), (30, -7.0, "io"), (86, -7.0), (88, 2.0, "io"),
      (90, -12.0, "step"), (94, -9.0, "out"), (102, -9.0), (104, 2.0, "step"), (108, -2.0, "io")]
HROT = [(0, -.04), (14, -.04), (20, .05), (23, .05), (24, -.07, "step"), (26, -.09, "step"), (30, -.06, "io"), (86, -.06), (88, -.02, "io"),
        (90, .07, "step"), (94, .05, "out"), (102, .05), (104, -.03, "step"), (108, -.04, "io")]
BROW = [(0, .3), (14, .3), (20, -.45), (23, -.45), (24, .9, "step"), (26, 1.0, "step"), (30, .6, "io"), (86, .6), (88, .2, "io"),
        (90, 1.0, "step"), (94, .85, "out"), (102, .85), (104, .3, "step"), (108, .3)]
BTILT = [(0, -.3), (14, -.3), (20, .7), (23, .7), (24, -.4, "step"), (30, -.3, "io"), (108, -.3)]
EYE = [(0, 1.0), (6, 1.0), (8, .1, "step"), (10, .55, "step"), (12, 1.0, "step"), (14, 1.0), (20, .5, "io"), (23, .5), (24, 1.0, "step"),
       (26, 0.0, "step"), (32, 0.0), (34, 1.0, "step"), (76, 1.0), (78, 0.0, "step"), (86, 0.0), (88, 1.0, "step"),
       (90, 0.0, "step"), (100, 0.0), (102, 1.0, "step"), (112, 1.0), (114, .1, "step"), (116, .55, "step"), (118, 1.0, "step")]
HIP = (84.0, -112.0)
POINT = (110.0, -280.0)
SALUTE = (84.0, -306.0)
RARM = [(0, HIP), (32, HIP), (34, (116.0, -210.0), "step"), (36, (114.0, -292.0), "step"), (38, POINT, "io"), (86, POINT),
        (88, (122.0, -262.0), "io"), (90, SALUTE, "step"), (102, SALUTE), (104, (128.0, -176.0), "step"), (106, (92.0, -120.0), "step"),
        (108, HIP, "io")]


def pose_at(f):
    """Pose for frame f (already quantised to twos by the caller)."""
    p = dict(NEUTRAL)
    t = f / FPS
    p["lever"] = kf(LEVER, f)
    p["sx"], p["sy"] = kf(SX, f), kf(SY, f)
    p["lean"] = kf(LEAN, f)
    p["hy"], p["hrot"] = kf(HY, f), kf(HROT, f)
    p["brow"], p["brow_tilt"] = kf(BROW, f), kf(BTILT, f)
    p["eye"] = kf(EYE, f)
    p["lamp"] = 1.0 if T_DING <= f < T_BACK else 0.0
    p["needle"] = .05 * math.sin((f - T_DING) * 1.9) * math.exp(-(f - T_DING) / 6) if f >= T_DING else 0.0
    # lip sync: lead the sound by one frame; stressed syllables bob the head and lift the brows
    ts = t + 1.0 / FPS
    m, wdt = 0.0, 1.0
    for a, b, op, wd, st, _v in SYL:
        if a <= ts < b:
            x = (ts - a) / (b - a)
            m = op * min(1.0, x / .25) * (1 - max(0.0, (x - .7) / .3) * .6)
            wdt = wd
            if st:
                p["hy"] -= 6 * math.sin(math.pi * min(x * 1.5, 1.0))
                p["brow"] += .35
                p["hrot"] += .03 * (1 if a < 2.4 else -1)
    p["mouth"], p["mwide"] = m, wdt
    p["bristle"] = 1.0 if 24 <= f < 30 or 88 <= f < 94 else 0.0
    # right arm
    tgt = kf(RARM, f)
    if 40 <= f < 86:                              # finger wags on the stressed beats
        for a, b, op, wd, st, _v in SYL:
            if st and a <= ts < b + .1:
                tgt = (tgt[0] - 6, tgt[1] - 5)
    if f < 34 or f >= 106:
        p["rarm"], p["rarm_over"] = (tgt, 1, "fist", None), False
    elif f < 88:
        p["rarm"], p["rarm_over"] = (tgt, 1, "point", -math.pi / 2 + .25), True
    elif f < 104:
        p["rarm"], p["rarm_over"] = (tgt, 1, "flat", -2.75), True
    else:
        p["rarm"], p["rarm_over"] = (tgt, 1, "fist", None), True
    return p


def speedlines(f):
    """Smear frames for the finger-up and the salute: tapered speed lines trailing the hand."""
    if f == 34:
        segs = [((150, -196), (138, -236)), ((166, -206), (152, -240)), ((134, -190), (126, -222))]
    elif f == 88:
        segs = [((150, -232), (140, -286)), ((164, -244), (152, -292)), ((136, -226), (130, -270))]
    else:
        return None

    def post(ctx, pp):
        X, Y = HERO
        ctx.save()
        ctx.translate(X + pp["x"], Y)
        ctx.rotate(pp["lean"])
        ctx.scale(pp["sx"], pp["sy"])
        for (a, b) in segs:
            a, b = np.array(a, float), np.array(b, float)
            for j in range(10):
                u0, u1 = j / 10, (j + 1) / 10
                line(ctx, [a + (b - a) * u0, a + (b - a) * u1], .5 + 3.4 * u1)
        ctx.restore()
    return post


def frame(i):
    f = (i // 2) * 2
    p = pose_at(f)
    smear = .95 if f == T_YANK else (-.85 if f == T_BACK else 0.0)
    img = render_layers(p, smear, post=speedlines(f))
    return to8(finish(img, i))
