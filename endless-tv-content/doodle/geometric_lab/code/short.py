"""short.py: key frames for the 2-minute short "One Small Passenger" (Mr. Hoist, Car No. 7).

Reuses the geo.py drawing code (Mr. Hoist, the lever, the lobby) and adds: a camera (wide / medium / close-ups, with
line weights compensated so close-ups are not fat), background variants (day lobby, night lobby, the Garden Department
floor), Mrs. Dawdle the snail and her babies, props (carpet roll + runner, lettuce pole, pocket watch, dizzy stars),
and a night grade lit by the cab, the sconces and the light spill on the floor."""
import math

import cairo
import cv2
import numpy as np

import geo
from geo import C, INK, ell, flat, hx, ink, line, rpoly, tube

W, H, SS = geo.W, geo.H, geo.SS
KH = 704                                  # key frame height (render 720, crop 8 px top and bottom)


# ============================================================================================ camera
class Cam:
    def __init__(self, z=1.0, cx=640.0, cy=360.0):
        self.z, self.cx, self.cy = z, cx, cy

    def apply(self, ctx):
        ctx.translate(W / 2, H / 2)
        ctx.scale(self.z, self.z)
        ctx.translate(-self.cx, -self.cy)

    @property
    def wscale(self):                      # user-space line scale: close-ups get only slightly heavier lines
        return self.z ** -0.62


WIDE = Cam()
MED = Cam(1.45, 650, 392)
MCU = Cam(1.9, 646, 330)
SNAIL_CU = Cam(4.0, 170, 632)


class WCtx:
    """cairo.Context proxy that scales every line width (so all geo.* drawing code works under the camera)."""

    def __init__(self, ctx, k):
        object.__setattr__(self, "_c", ctx)
        object.__setattr__(self, "_k", k)

    def set_line_width(self, w):
        self._c.set_line_width(w * self._k)

    def __getattr__(self, n):
        return getattr(self._c, n)


def surface(cam):
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * SS, H * SS)
    c = cairo.Context(s)
    c.scale(SS, SS)
    c.set_line_join(cairo.LINE_JOIN_ROUND)
    c.set_line_cap(cairo.LINE_CAP_ROUND)
    cam.apply(c)
    return s, WCtx(c, cam.wscale)


# ============================================================================================ background variants
def clock_face(ctx, hh, mm, kx=1118, ky=250):
    ell(ctx, kx, ky, 33, 33)
    flat(ctx, hx("#235d6a"))
    for k in range(12):
        a = k * math.pi / 6
        line(ctx, [(kx + 25 * math.cos(a), ky + 25 * math.sin(a)), (kx + 30 * math.cos(a), ky + 30 * math.sin(a))], 2.2, C["brass3"])
    ah = (hh % 12 + mm / 60) / 12 * 2 * math.pi - math.pi / 2
    am = mm / 60 * 2 * math.pi - math.pi / 2
    line(ctx, [(kx, ky), (kx + 15 * math.cos(ah), ky + 15 * math.sin(ah))], 3.4, C["brass3"])
    line(ctx, [(kx, ky), (kx + 24 * math.cos(am), ky + 24 * math.sin(am))], 2.4, C["brass3"])
    ell(ctx, kx, ky, 3.5, 3.5)
    flat(ctx, C["brass3"])


def sconce(ctx, cx, cy=286):
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
    for pts, col in (([(cx - 4, cy + 4), (cx + 4, cy + 4), (cx + 4, cy + 30), (cx - 4, cy + 30)], C["brass2"]),
                     ([(cx - 13, cy + 24), (cx + 13, cy + 24), (cx + 13, cy + 36), (cx - 13, cy + 36)], C["brass"])):
        rpoly(ctx, pts, 2.5)
        ink(ctx, col, 2.0, C["brass_ln"])
    rpoly(ctx, [(cx - 26, cy - 22), (cx + 26, cy - 22), (cx + 8, cy + 8), (cx - 8, cy + 8)], [3, 3, 4, 4])
    ink(ctx, hx("#fbe9b8"), 2.2, C["brass_ln"])


def garden(ctx):
    """Second floor, the Garden Department: painted over the lobby everywhere except the elevator surround."""
    ctx.save()
    ctx.rectangle(-50, -50, W + 100, H + 100)
    flat(ctx, hx("#a3d18c"))
    for r in range(0, 9):                                # leaf wallpaper: little geometric leaves on a grid
        for q in range(0, 19):
            cx, cy = q * 72 + (36 if r % 2 else 0), 70 + r * 58
            a = .7 if (q + r) % 2 else -.7
            ctx.save()
            ctx.translate(cx, cy)
            ctx.rotate(a)
            rpoly(ctx, [(-14, 0), (0, -6), (14, 0), (0, 6)], [6, 3, 6, 3])
            flat(ctx, hx("#b9dea6"))
            rpoly(ctx, [(-14, 0), (0, -6), (14, 0), (0, 6)], [6, 3, 6, 3])
            ctx.set_source_rgba(*hx("#7bb267"))
            ctx.set_line_width(1.6)
            ctx.stroke()
            ctx.restore()
    ctx.rectangle(0, 0, W, 30)
    flat(ctx, hx("#6fa860"))
    ctx.rectangle(0, 30, W, 8)
    flat(ctx, hx("#5e9551"))
    line(ctx, [(0, 30), (W, 30)], 2.2, hx("#4d7f44"))
    line(ctx, [(0, 38), (W, 38)], 2.2, hx("#4d7f44"))
    # picket-fence wainscot
    ctx.rectangle(0, 584, W, 22)
    flat(ctx, hx("#e9e1c6"))
    for x in range(-10, W + 20, 34):
        pk = [(x, 604), (x, 516), (x + 12, 500), (x + 24, 516), (x + 24, 604)]
        rpoly(ctx, pk, [1, 2, 2, 2, 1])
        flat(ctx, hx("#f6f0dc"))
        rpoly(ctx, pk, [1, 2, 2, 2, 1])
        ctx.set_source_rgba(*hx("#b9a982"))
        ctx.set_line_width(2)
        ctx.stroke()
    for yy in (536, 572):
        ctx.rectangle(0, yy, W, 9)
        flat(ctx, hx("#efe7cf"))
        line(ctx, [(0, yy), (W, yy)], 1.8, hx("#b9a982"))
        line(ctx, [(0, yy + 9), (W, yy + 9)], 1.8, hx("#b9a982"))
    # terracotta floor tiles in the same perspective as the lobby checkerboard
    vx, vy, y0, y1 = 640, 360, 604, H
    ctx.rectangle(0, y0, W, y1 - y0)
    flat(ctx, hx("#e3a477"))
    rows, j = [y1], 1
    while rows[-1] > y0:
        rows.append(vy + (y1 - vy) / (1 + j * .16))
        j += 1
    rows[-1] = y0
    for a in range(len(rows) - 1):
        ya, yb = rows[a], rows[a + 1]
        for k in range(-12, 13):
            if (k + a) % 2:
                continue
            xs = [(vx + (k - .5) * 96 * (yy - vy) / (y1 - vy), vx + (k + .5) * 96 * (yy - vy) / (y1 - vy)) for yy in (ya, yb)]
            ctx.move_to(xs[0][0], ya)
            ctx.line_to(xs[0][1], ya)
            ctx.line_to(xs[1][1], yb)
            ctx.line_to(xs[1][0], yb)
            ctx.close_path()
    flat(ctx, hx("#d68f62"))
    line(ctx, [(0, y0), (W, y0)], 2.4, hx("#a8643f"))
    # sconces again
    for sx in (-1, 1):
        sconce(ctx, 640 + sx * 312)
    # left: a big terracotta pot with three geometric sunflowers
    px, py = 228, 676
    for k, (dx, hgt, rad) in enumerate(((-46, 250, 30), (6, 330, 36), (52, 210, 26))):
        top = (px + dx * 1.3, py - 90 - hgt)
        line(ctx, [(px + dx * .4, py - 92), top], 5, hx("#3f8a4a"))
        for side in (-1, 1):
            lf = [(px + dx * .7, py - 150 - hgt * .3), (px + dx * .7 + side * 34, py - 172 - hgt * .3), (px + dx * .7 + side * 6, py - 160 - hgt * .3)]
            rpoly(ctx, lf, 3)
            ink(ctx, hx("#4f9e55"), 2, hx("#2b6234"))
        for j in range(12):
            a = j * math.pi / 6
            pet = [(top[0] + rad * .55 * math.cos(a - .26), top[1] + rad * .55 * math.sin(a - .26)), (top[0] + rad * 1.2 * math.cos(a), top[1] + rad * 1.2 * math.sin(a)),
                   (top[0] + rad * .55 * math.cos(a + .26), top[1] + rad * .55 * math.sin(a + .26))]
            rpoly(ctx, pet, 1.5)
            ink(ctx, hx("#ffcf3a"), 2, hx("#b37f12"))
        ell(ctx, top[0], top[1], rad * .6, rad * .6)
        ink(ctx, hx("#7a4a26"), 2, hx("#4e2e16"))
    pot = [(px - 62, py - 96), (px + 62, py - 96), (px + 44, py), (px - 44, py)]
    rpoly(ctx, pot, [4, 4, 8, 8])
    ink(ctx, hx("#cf6e43"), 2.2, hx("#7f3d20"))
    rpoly(ctx, [(px - 68, py - 104), (px + 68, py - 104), (px + 66, py - 86), (px - 66, py - 86)], 4)
    ink(ctx, hx("#d9784c"), 2.2, hx("#7f3d20"))
    # call plate again
    rpoly(ctx, [(914, 352), (954, 352), (954, 444), (914, 444)], 9)
    ink(ctx, C["brass"], 2.2, C["brass_ln"])
    for k, up in ((0, True), (1, False)):
        by = 378 + k * 40
        ell(ctx, 934, by, 12, 12)
        ink(ctx, C["dial_face"], 2, C["brass_ln"])
        s = -1 if up else 1
        ctx.move_to(934, by + 5 * s)
        ctx.line_to(940, by - 4 * s)
        ctx.line_to(928, by - 4 * s)
        ctx.close_path()
        flat(ctx, C["brass_ln"])
    # a garden trellis arch on the upper right wall
    for k in range(7):
        x = 1040 + k * 30
        line(ctx, [(x, 120), (x + 60, 400)], 2.4, hx("#f6f0dc"))
        line(ctx, [(x + 60, 120), (x, 400)], 2.4, hx("#f6f0dc"))
    for k in range(9):
        lx, ly = 1060 + (k * 37) % 190, 140 + (k * 61) % 240
        ctx.save()
        ctx.translate(lx, ly)
        ctx.rotate(k)
        rpoly(ctx, [(-12, 0), (0, -6), (12, 0), (0, 6)], [6, 3, 6, 3])
        ink(ctx, hx("#4f9e55"), 2, hx("#2b6234"))
        ctx.restore()
    ctx.restore()


def bg_scene(ctx, variant, clock):
    geo.bg_static(ctx)
    if variant == "garden":                              # paint the floor-2 set as a layer with the elevator surround punched out
        ctx.push_group()
        garden(ctx)
        ctx.set_operator(cairo.OPERATOR_CLEAR)
        for x0, y0, x1, y1 in ((402, 125, 878, 607), (383, 111, 897, 133), (420, 594, 860, 611)):
            ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
            ctx.fill()
        ctx.move_to(640 - 72, 125)
        ctx.arc(640, 124, 72, math.pi, 2 * math.pi)
        ctx.close_path()
        ctx.fill()
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.pop_group_to_source()
        ctx.paint()
    else:
        clock_face(ctx, *clock)


_BG = {}


def background(variant, clock, cam):
    key = (variant, clock, cam.z, cam.cx, cam.cy)
    if key not in _BG:
        s, ctx = surface(cam)
        bg_scene(ctx, variant, clock)
        a = geo.surf_np(s)[..., :3]
        a = geo.paint_finish(a)
        out = np.zeros((H * SS, W * SS, 4), np.uint8)
        out[..., :3] = (a[..., ::-1] * 255 + .5).astype(np.uint8)
        out[..., 3] = 255
        bs = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * SS, H * SS)
        np.ndarray((H * SS, W * SS, 4), np.uint8, bs.get_data())[:] = out
        bs.mark_dirty()
        _BG[key] = bs
    return _BG[key]


def night_mask(cam):
    """Where the night is lit: the cab, its light spilling onto the floor, the sconces and the dial lamps."""
    s, ctx = surface(cam)
    ctx.rectangle(-2000, -2000, 6000, 6000)
    ctx.set_source_rgba(0, 0, 0, 1)
    ctx.fill()
    ox0, oy0, ox1, oy1 = geo.OPEN
    ctx.rectangle(ox0 - 6, oy0 - 6, ox1 - ox0 + 12, oy1 - oy0 + 16)
    ctx.set_source_rgba(1, 1, 1, 1)
    ctx.fill()
    ctx.move_to(ox0 - 10, oy1 + 6)                       # floor spill, widening toward camera
    ctx.line_to(ox1 + 10, oy1 + 6)
    ctx.line_to(ox1 + 150, H + 40)
    ctx.line_to(ox0 - 150, H + 40)
    ctx.close_path()
    g = cairo.LinearGradient(0, oy1, 0, H + 40)
    g.add_color_stop_rgba(0, 1, 1, 1, .85)
    g.add_color_stop_rgba(1, 1, 1, 1, .25)
    ctx.set_source(g)
    ctx.fill()
    for cx in (328, 952):
        g = cairo.RadialGradient(cx, 230, 10, cx, 230, 190)
        g.add_color_stop_rgba(0, 1, 1, 1, .9)
        g.add_color_stop_rgba(1, 1, 1, 1, 0)
        ctx.set_source(g)
        ctx.arc(cx, 230, 190, 0, 2 * math.pi)
        ctx.fill()
    g = cairo.RadialGradient(640, 110, 10, 640, 110, 140)
    g.add_color_stop_rgba(0, 1, 1, 1, .7)
    g.add_color_stop_rgba(1, 1, 1, 1, 0)
    ctx.set_source(g)
    ctx.arc(640, 110, 140, 0, 2 * math.pi)
    ctx.fill()
    m = geo.surf_np(s)[..., 0]
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(m, (0, 0), 18 * cam.z ** .5)


def night_grade(img, cam, variant="lobby"):
    m = night_mask(cam)[..., None]
    tint = [.36, .42, .72] if variant == "lobby" else [.5, .56, .8]
    dark = img * np.array(tint, np.float32) + np.array([.015, .02, .06], np.float32)
    return dark * (1 - m) + img * m * np.array([1.0, .97, .9], np.float32)


# ============================================================================================ Mrs. Dawdle (and babies)
SHELLS = [hx("#8a55a8"), hx("#2f9c8f"), hx("#e0864a"), hx("#d0609b"), hx("#e3b83a"), hx("#5f7fd0")]


def snail(ctx, x, y, s=1.0, face=1, st=None):
    """Snail, side view.  (x, y) = ground point under her middle; s = scale (s=1: ~92 px long).
    st keys: shell (colour), hat (bool), stalk (l, r) angles in rad (0 = up, + = forward), eye (1 open / 0 closed),
    look (-1..1 forward), look_y (+ down), tucked (asleep inside the shell), smile, salute (stalk to hat), blush."""
    st = st or {}
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(s * face, s)
    shell_c = st.get("shell", SHELLS[0])
    dk = tuple(v * .66 for v in shell_c[:3]) + (1,)
    lt = tuple(min(1, v * 1.25 + .08) for v in shell_c[:3]) + (1,)
    body_c, body_dk = hx("#cfe07e"), hx("#a9bd57")
    # shadow
    ell(ctx, -2, 1, 50, 6)
    flat(ctx, hx("#2c2346", .28))
    if st.get("tucked"):
        sc = (-4, -27)
        rpoly(ctx, [(-30, 0), (24, 0), (26, -6), (-30, -6)], 3)
        ink(ctx, body_c)
    else:
        sc = (-12, -34)
        body = [(-56, 0), (36, 0), (48, -6), (50, -34), (42, -46), (28, -46), (22, -20), (-20, -12), (-50, -5)]
        rpoly(ctx, body, [2, 6, 8, 12, 9, 6, 10, 10, 4])
        ink(ctx, body_c)
        line(ctx, [(-40, -4), (20, -4)], 2.2, body_dk)
    # shell: a disc with a squared-off spiral (very UPA)
    R = 28
    ell(ctx, sc[0], sc[1], R, R)
    ink(ctx, shell_c)
    ctx.save()
    ell(ctx, sc[0], sc[1], R, R)
    ctx.clip()
    ctx.new_path()
    ctx.arc(sc[0], sc[1], R - 5, math.pi * 1.05, math.pi * 1.55)
    ctx.set_source_rgba(*lt)
    ctx.set_line_width(5)
    ctx.stroke()
    ctx.restore()
    pts, r, a = [], R - 4, -math.pi / 2
    for j in range(13):
        pts.append((sc[0] + r * math.cos(a), sc[1] + r * math.sin(a)))
        a += math.pi / 2
        r *= .8
    line(ctx, pts, 3.2, dk)
    if not st.get("tucked"):
        # eye stalks (behind one, in front one) with round eyes
        hx0, hy0 = 40, -44
        for k, (dx, ang) in enumerate(((-4, st.get("stalk", (0.1, 0.25))[0]), (6, st.get("stalk", (0.1, 0.25))[1]))):
            L = 34
            sal = st.get("salute") and k == 1
            if sal:
                tip = (hx0 + dx - 14, hy0 - 22)
            else:
                tip = (hx0 + dx + L * math.sin(ang), hy0 - L * math.cos(ang))
            tube(ctx, [(hx0 + dx, hy0 + 4), ((hx0 + dx + tip[0]) / 2 + 2, (hy0 + tip[1]) / 2), tip], 5, body_c, w=3.2)
            ell(ctx, tip[0], tip[1], 8.5, 8.5)
            ink(ctx, hx("#ffffff"), 3.2)
            e = st.get("eye", 1.0)
            if e > .3:
                ell(ctx, tip[0] + 3 * st.get("look", .4), tip[1] + 3 * st.get("look_y", 0), 4, 4.6 * e)
                flat(ctx, INK)
            else:
                line(ctx, [(tip[0] - 5, tip[1] + 1), (tip[0], tip[1] + 3), (tip[0] + 5, tip[1] + 1)], 2.6)
        # smile + blush
        if st.get("open"):
            ell(ctx, 46, -22, 4, 5)
            ink(ctx, C["mouth"], 2.6)
        else:
            line(ctx, [(39, -22), (44, -18), (49, -21)], 2.6)
        ell(ctx, 38, -27, 4.5, 3)
        flat(ctx, hx("#f49ab2", .8))
        if st.get("hat", True):
            ctx.save()
            ctx.translate(30, -46)
            ctx.rotate(-.25)
            rpoly(ctx, [(-14, 2), (12, 2), (8, -14), (-10, -14)], [3, 3, 5, 5])
            ink(ctx, hx("#f28db2"), 3.0)
            rpoly(ctx, [(-19, 2), (17, 2), (17, 6), (-19, 6)], 3)
            ink(ctx, hx("#f28db2"), 3.0)
            for j in range(6):
                a = j * math.pi / 3
                ell(ctx, -2 + 5.5 * math.cos(a), -7 + 5.5 * math.sin(a), 3.2, 3.2)
                ink(ctx, hx("#ffffff"), 1.8)
            ell(ctx, -2, -7, 2.8, 2.8)
            ink(ctx, hx("#ffd23f"), 1.6)
            ctx.restore()
    elif st.get("bubble"):                                # sleeping inside the shell: a little bubble
        ell(ctx, 30, -16, 7, 7)
        ink(ctx, hx("#e8f6ff", .9), 2.2)
        ell(ctx, 40, -30, 4, 4)
        ink(ctx, hx("#e8f6ff", .9), 2.0)
    ctx.restore()


# ============================================================================================ props
def carpet_roll(ctx, x, y_top, y_bot):
    rpoly(ctx, [(x - 15, y_top), (x + 15, y_top), (x + 15, y_bot), (x - 15, y_bot)], 6)
    ink(ctx, hx("#c8262a"))
    for yy in (y_top + 14, y_bot - 14):
        line(ctx, [(x - 15, yy), (x + 15, yy)], 3.0, C["gold"])
    ell(ctx, x, y_top, 15, 6)
    ink(ctx, hx("#e04a42"))
    line(ctx, [(x, y_top), (x + 7, y_top - 1), (x + 4, y_top + 3), (x - 4, y_top + 1)], 2.0)


def carpet_runner(ctx, near=(470, 590), far=(130, 330), y_near=606, y_far=694):
    """Red runner on the floor from the cab threshold out toward the camera-left."""
    pts = [(near[0], y_near), (near[1], y_near), (far[1], y_far), (far[0], y_far)]
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.close_path()
    ink(ctx, hx("#d0302f"))
    for t in (.06, .94):
        a = (near[0] + (near[1] - near[0]) * t, y_near)
        b = (far[0] + (far[1] - far[0]) * t, y_far)
        line(ctx, [a, b], 3.0, C["gold"])


def lettuce_pole(ctx, a, b, leaf=True):
    tube(ctx, [a, b], 7, C["brass3"], w=3.2)
    ell(ctx, b[0], b[1], 6, 6)
    ink(ctx, C["brass2"], 3)
    end = (b[0] - 4, b[1] + 62)
    line(ctx, [b, (b[0] - 2, b[1] + 30), end], 2.0)
    if leaf:
        lf = [(end[0], end[1]), (end[0] + 22, end[1] + 16), (end[0] + 14, end[1] + 40), (end[0] - 8, end[1] + 44),
              (end[0] - 22, end[1] + 26), (end[0] - 16, end[1] + 8)]
        rpoly(ctx, lf, [3, 10, 8, 8, 10, 6])
        ink(ctx, hx("#8fd05c"), 3.4)
        line(ctx, [(end[0], end[1] + 4), (end[0] - 2, end[1] + 38)], 2.0, hx("#3c7a2a"))
        line(ctx, [(end[0] - 1, end[1] + 20), (end[0] + 12, end[1] + 14)], 2.0, hx("#3c7a2a"))
        line(ctx, [(end[0] - 1, end[1] + 28), (end[0] - 13, end[1] + 22)], 2.0, hx("#3c7a2a"))
    else:                                                  # nibbled to a stub
        rpoly(ctx, [(end[0] - 5, end[1]), (end[0] + 6, end[1] + 2), (end[0] + 3, end[1] + 10), (end[0] - 4, end[1] + 9)], 2)
        ink(ctx, hx("#8fd05c"), 2.6)


def pocket_watch(ctx, x, y, chain_to=None):
    if chain_to:
        line(ctx, [chain_to, ((chain_to[0] + x) / 2, max(chain_to[1], y) + 24), (x, y + 16)], 2.4, C["gold"])
    ell(ctx, x, y - 19, 5, 4)
    ink(ctx, C["gold"], 2.6)
    ell(ctx, x, y, 17, 17)
    ink(ctx, C["gold"], 3.4)
    ell(ctx, x, y, 12, 12)
    ink(ctx, hx("#fbf6e6"), 2.0)
    line(ctx, [(x, y), (x, y - 8)], 2.2)
    line(ctx, [(x, y), (x + 6, y + 3)], 2.2)


def stars(ctx, cx, cy, r=70):
    for k in range(3):
        a = -math.pi / 2 + k * 2.1 + .4
        sx, sy = cx + r * math.cos(a), cy + .45 * r * math.sin(a)
        pts = []
        for j in range(10):
            rr = 13 if j % 2 == 0 else 5.5
            aa = -math.pi / 2 + j * math.pi / 5
            pts.append((sx + rr * math.cos(aa), sy + rr * math.sin(aa)))
        rpoly(ctx, pts, 1)
        ink(ctx, hx("#ffe14d"), 3.0)


# ============================================================================================ Mr. Hoist poses
def pose(**kw):
    p = dict(geo.NEUTRAL)
    p.update(kw)
    return p


def body_pt(world, p):
    return tuple(geo.world_to_body(world, p))


P = {}
P["neutral"] = pose()
P["announce"] = pose(brow=.7, hrot=-.07, hy=-6, eye=1.0)
P["peer"] = pose(lean=-.05, brow=1.0, brow_tilt=-.2, look=-1.0, hrot=-.06, hy=-4,
                 rarm=((40.0, -298.0), 1, "flat", math.pi - .08), rarm_over=True)
P["look_left"] = pose(look=-1.0, brow=.9, brow_tilt=-.3, hrot=-.03)
P["heart"] = pose(sy=1.03, eye=0.0, happy_eyes=True, brow=.9, hrot=-.07, hy=-6, mstyle="smile",
                  rarm=((20.0, -176.0), -1, "flat", math.pi + .35), rarm_over=True)
P["watch"] = pose(eye_r=.45, look=.8, look_y=1.0, mstyle="grit", brow=.4, brow_tilt=.3, stress=True, hrot=.03,
                  rarm=((52.0, -170.0), 1, "fist", -math.pi / 2), rarm_over=True)
P["carpet_hold"] = pose(brow=.8, mstyle="smile", rarm=((112.0, -150.0), 1, "fist", -math.pi / 2), look=-.6)
P["exasperated"] = pose(grip=0.0, larm_free=((-120.0, -268.0), -1, "open", -math.pi / 2 - .5), brow=.9, brow_tilt=-.8,
                        rarm=((120.0, -268.0), 1, "open", -math.pi / 2 + .5), rarm_over=True, look=-.8, sy=1.02,
                        mstyle="grit", sweat=True)
P["sleep"] = pose(eye=0.0, brow=-.25, brow_tilt=0.0, hy=7.0, hrot=.08, mstyle="o", cap_rot=.14, sy=.985)
P["startled"] = pose(eye_big=1.35, brow=1.5, brow_tilt=-.3, cap_dy=26.0, cap_rot=-.1, sy=1.05, hy=-14,
                     rarm=((126.0, -150.0), 1, "open", .2), look=-1.0, look_y=1.0)
P["strain"] = pose(lean=-.04, eye_l=1.0, eye_r=.35, mstyle="grit", brow=-.3, brow_tilt=.9, stress=True, bristle=1.0,
                   hrot=-.06, hy=4, rarm=((-110.0, -150.0), -1, "fist", math.pi), rarm_over=True)
P["composed"] = pose(eye=0.0, happy_eyes=True, brow=.8, hrot=-.08, hy=-6, mstyle="smile", sy=1.03)
P["salute"] = pose(lever=.95, lamp=1.0, eye=0.0, happy_eyes=True, brow=1.0, hrot=.07, hy=-12, sy=1.03,
                   rarm=((84.0, -306.0), 1, "flat", -2.75), rarm_over=True)
P["dismay"] = pose(droop=1.0, brow=.5, brow_tilt=-1.0, eye=.6, look=.6, look_y=1.0, stress=True, hy=6, hrot=.04)
P["faint"] = pose(lean=.42, swirl=True, droop=1.0, mstyle="wavy", brow=.8, brow_tilt=-.6, cap_rot=.5, cap_dx=10, cap_dy=8,
                  grip=0.0, larm_free=((-86.0, -104.0), 1, "fist", math.pi / 2), rarm=((86.0, -104.0), -1, "fist", math.pi / 2))


# ============================================================================================ render a key
def render(spec):
    """spec: dict(cam, variant, clock, night, hoist=pose|None, snails=[(x, y, s, face, st)], props=[(layer, fn)],
    lever_smear, needle)."""
    cam = spec.get("cam", WIDE)
    p = dict(spec["hoist"]) if spec.get("hoist") else None
    bs, bctx = surface(Cam())
    bctx.set_source_surface(background(spec.get("variant", "lobby"), spec.get("clock", (10, 10)), cam), 0, 0)
    bctx.get_source().set_matrix(cairo.Matrix(xx=SS, yy=SS))
    bctx.paint()
    dc = cairo.Context(bs)
    dc.scale(SS, SS)
    dc.set_line_join(cairo.LINE_JOIN_ROUND)
    dc.set_line_cap(cairo.LINE_CAP_ROUND)
    cam.apply(dc)
    dctx = WCtx(dc, cam.wscale)
    st = dict(lamp=(p or {}).get("lamp", spec.get("lamp", 0.0)), needle=spec.get("needle", 0.0), x=(p or {}).get("x", 0.0))
    geo.bg_dynamic(dctx, st)
    cs, cctx = surface(cam)
    for layer, fn in spec.get("props", []):
        if layer == "floor":
            fn(cctx)
    lever = (p or {}).get("lever", 0.0)
    tip = geo.lever_stand(cctx, lever, spec.get("lever_smear", 0.0))
    behind = [sn for sn in spec.get("snails", []) if sn[1] < geo.HERO[1] - 2]
    front = [sn for sn in spec.get("snails", []) if sn[1] >= geo.HERO[1] - 2]
    for sn in behind:
        snail(cctx, *sn)
    for layer, fn in spec.get("props", []):
        if layer == "under":
            fn(cctx, p)
    if p is not None:
        if p.get("grip", 1.0) > .5:
            tb = geo.world_to_body(tip, p)
            p["larm"] = ((tb[0] + 15, tb[1] + 3), -1, "grip", math.pi + .05)
        else:
            p["larm"] = p["larm_free"]
        geo.hoist(cctx, p)
    for layer, fn in spec.get("props", []):
        if layer == "over":
            fn(cctx, p)
    for sn in front:
        snail(cctx, *sn)
    for layer, fn in spec.get("props", []):
        if layer == "top":
            fn(cctx, p)
    bg = geo.surf_np(bs)[..., :3]
    cel = geo.surf_np(cs)
    a = cel[..., 3]
    k = max(1, int(round(geo.SIL * SS * cam.z ** .38)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    d = cv2.GaussianBlur(cv2.dilate(a, ker), (0, 0), .6)
    out = cel[..., :3] + (1 - a)[..., None] * (1 - d)[..., None] * bg + (1 - a)[..., None] * d[..., None] * np.array(INK[:3], np.float32)
    out = cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA)
    if spec.get("night"):
        out = night_grade(out, cam, spec.get("variant", "lobby"))
    img = geo.to8(geo.finish(out, 0))
    return img[8:8 + KH]


# ============================================================================================ the keys
SN0 = dict(stalk=(.05, .25), look=.5)                      # Mrs. Dawdle, default


def _watch(ctx, p):
    w = geo.body_to_world((52, -196), p)
    pocket_watch(ctx, w[0], w[1], chain_to=tuple(geo.body_to_world((24, -140), p)))


def _roll(ctx, p):
    w = geo.body_to_world((114, -150), p)
    carpet_roll(ctx, w[0] + 2, w[1] - 26, geo.HERO[1] - 2)


def _runner(ctx):
    carpet_runner(ctx)


def _pole(leaf):
    def f(ctx, p):
        lettuce_pole(ctx, (640, 418), (262, 548), leaf)
    return f


def _faint_stars(ctx, p):
    w = geo.body_to_world((0, -300), p)
    stars(ctx, w[0] + 10, w[1] - 70, 62)


def pole_pose(**kw):
    """Hoist holding the long lettuce pole with both hands (no lever)."""
    base = pose(grip=0.0, **kw)
    a, b = np.array([640, 418.0]), np.array([262, 548.0])
    front = a + (b - a) * .17
    back = a + (b - a) * .02
    base["larm_free"] = (body_pt(front + [10, 2], base), -1, "grip", math.pi - .33)
    base["rarm"] = (body_pt(back + [12, 8], base), 1, "grip", math.pi - .33)
    base["rarm_over"] = True
    return base


GY = 668                                                  # snail ground line on the lobby floor
SNS = .85                                                 # Mrs. Dawdle's scale in wide/medium shots

KEYS = {
    # S01 wide, day: the ceremony; the tiny passenger is already there, far left
    "s01_a": dict(hoist=P["announce"], snails=[(118, GY, SNS, 1, dict(SN0))]),
    "s01_b": dict(hoist=P["peer"], snails=[(118, GY, SNS, 1, dict(SN0, stalk=(.0, -.5), look=.8))]),
    # S02 snail close-up
    "s02_a": dict(cam=SNAIL_CU, snails=[(150, 680, 1.0, 1, dict(SN0, stalk=(.05, .2), look=.6, look_y=-.6))]),
    "s02_b": dict(cam=SNAIL_CU, snails=[(162, 680, 1.0, 1, dict(SN0, stalk=(-.75, .55), look=.0, look_y=-1.0))]),
    # S03 medium close: "Car Number Seven waits for every passenger."
    "s03_a": dict(cam=MCU, hoist=P["look_left"]),
    "s03_b": dict(cam=MCU, hoist=P["heart"]),
    # S04 wide: waiting; the clock jumps on
    "s04_a": dict(hoist=P["heart"], snails=[(118, GY, SNS, 1, dict(SN0))]),
    "s04_b": dict(hoist=P["watch"], clock=(1, 35), snails=[(170, GY, SNS, 1, dict(SN0))], props=[("over", _watch)]),
    # S05 wide: the red carpet
    "s05_a": dict(hoist=P["carpet_hold"], clock=(1, 35), snails=[(170, GY, SNS, 1, dict(SN0))], props=[("under", _roll)]),
    "s05_b": dict(hoist=P["exasperated"], clock=(1, 40), snails=[(170, GY, SNS, 1, dict(SN0, look_y=1.0, look=.2, stalk=(.4, .6)))],
                  props=[("floor", _runner)]),
    # S06 wide: lettuce on a pole
    "s06_a": dict(hoist=pole_pose(brow=.8, mstyle="smile", look=-1.0), clock=(2, 50),
                  snails=[(196, GY, SNS, 1, dict(SN0, stalk=(.3, .5), look=.9))], props=[("floor", _runner), ("under", _pole(True))]),
    "s06_b": dict(hoist=pole_pose(mouth=1.0, mwide=.8, eye_big=1.25, brow=1.3, look=-1.0, look_y=1.0), clock=(3, 20),
                  snails=[(206, GY, SNS, 1, dict(SN0, tucked=True, bubble=True))], props=[("floor", _runner), ("under", _pole(False))]),
    # S07 medium close: nearly cracks, then composes himself
    "s07_a": dict(cam=MCU, hoist=P["strain"], clock=(4, 45)),
    "s07_b": dict(cam=MCU, hoist=P["composed"], clock=(4, 45)),
    # S08 wide, night: asleep on duty; she crosses the threshold; he jolts awake
    "s08_a": dict(hoist=P["sleep"], clock=(11, 55), night=True, snails=[(440, 640, SNS, 1, dict(SN0))], props=[("floor", _runner)]),
    "s08_b": dict(hoist=P["startled"], clock=(11, 58), night=True, snails=[(566, 604, SNS, 1, dict(SN0, stalk=(-.1, .3), look=.0))],
                  props=[("floor", _runner)]),
    # S09 medium, night: full ceremony, the lever, the salute (and hers)
    "s09_a": dict(cam=MED, hoist=P["announce"], clock=(11, 58), night=True, snails=[(566, 604, SNS, 1, dict(SN0, stalk=(-.1, .3)))],
                  props=[("floor", _runner)]),
    "s09_b": dict(cam=MED, hoist=P["salute"], clock=(11, 58), night=True,
                  snails=[(566, 604, SNS, 1, dict(SN0, salute=True, eye=0.0))], props=[("floor", _runner)]),
    # S10 wide, night, second floor: "Garden Department" ... she exits, slowly
    "s10_a": dict(variant="garden", needle=math.pi / 8, lamp=1.0, night=True, hoist=pose(lamp=1.0, brow=.8, mstyle="smile", hy=-4),
                  snails=[(566, 604, SNS, 1, dict(SN0))]),
    "s10_b": dict(variant="garden", needle=math.pi / 8, lamp=1.0, night=True, hoist=dict(P["dismay"], lamp=1.0),
                  snails=[(612, 618, SNS, 1, dict(SN0))]),
    # S11 (continues from s10_b): she's out ... and here come the babies; he keels over like a plank
    "s11_b": dict(variant="garden", needle=math.pi / 8, lamp=1.0, night=True, hoist=dict(P["faint"], lamp=1.0),
                  snails=[(700, 652, SNS, -1, dict(SN0, look=.6))] + [(772 + j * 50, 656 + (j % 2) * 4, .42, -1, dict(shell=SHELLS[1 + j], hat=False, look=.5, stalk=(.2, .4)))
                                                              for j in range(5)], props=[("top", _faint_stars)]),
}
# babies peeking out from behind the lettuce crate in the S10 keys (they are there from the start)
PEEK = [(928 + j * 31, 578 - (j % 2) * 7, .42, -1 if j % 2 else 1, dict(shell=SHELLS[1 + j], hat=False, look=.5, stalk=(.1, .3)))
        for j in range(5)]


def crate(ctx, p=None):
    """A crate of lettuces: a cel prop (black ink) because the babies hide behind it."""
    cx0, cy0 = 902, 666
    for k in range(5):
        lx, ly = cx0 + 24 + k * 30, cy0 - 92 - (k % 2) * 8
        ell(ctx, lx, ly, 24, 20)
        ink(ctx, hx("#8fd05c") if k % 2 else hx("#79c050"))
        line(ctx, [(lx - 12, ly - 4), (lx, ly + 6), (lx + 12, ly - 4)], 2.4, hx("#3c7a2a"))
    rpoly(ctx, [(cx0, cy0 - 92), (cx0 + 172, cy0 - 92), (cx0 + 172, cy0), (cx0, cy0)], 3)
    ink(ctx, hx("#e2b870"))
    for yy in (cy0 - 62, cy0 - 31):
        line(ctx, [(cx0 + 3, yy), (cx0 + 169, yy)], 2.4, hx("#8a6125"))
    line(ctx, [(cx0 + 12, cy0 - 86), (cx0 + 160, cy0 - 6)], 2.4, hx("#b88a3e"))


for kk in ("s10_a", "s10_b"):
    KEYS[kk]["snails"] = KEYS[kk]["snails"] + PEEK
    KEYS[kk]["props"] = [("floor", lambda ctx: crate(ctx))] + KEYS[kk].get("props", [])
KEYS["s11_b"]["props"] = [("floor", lambda ctx: crate(ctx))] + KEYS["s11_b"]["props"]


def render_key(name):
    return render(KEYS[name])


if __name__ == "__main__":
    import os
    import sys
    out = sys.argv[1]
    names = sys.argv[2:] or list(KEYS)
    os.makedirs(out, exist_ok=True)
    for n in names:
        cv2.imwrite(os.path.join(out, n + ".png"), render_key(n)[:, :, ::-1])
        print(n)
