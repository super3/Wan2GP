"""Characters for the bold_shapes short 'One Little Smudge': Squeaky Doyle (on-model with the prototype) and Mrs. Fig.

Everything is drawn with a camera-aware Cel: a 'base' cairo matrix maps world units to device pixels, so the same drawing
code serves the wide, medium, close and interior set-ups.  Lines drawn by the Cel scale with the camera zoom and with an
optional per-figure line factor (lw); face details scale with the figure."""
import math

import cairo
import numpy as np

from draw import (INK, P, WHITE, cr_path, ellipse, fill, hexc, lerp, poly, rrect, stroke, superellipse, sparkle)
import draw

SS = 2

# extra colours for the short
P2 = {k: hexc(v) for k, v in dict(
    red_face="#ff6b5a", sweat="#7fc8ff", steam="#ffffff", foam="#ffffff", foam_ln="#17121c", chrome="#d6dde5", chrome_dk="#9aa6b3",
    pad="#fff6e8", stripe="#e8343c", cord="#3a3440",
    fig_skin="#ffe1cf", fig_hair="#cfcadf", fig_hair2="#b4aecb", cardigan="#9284d8", cardigan_dk="#7466c0", collar="#fbfbf6",
    pearl="#fffaf0", glasses="#e2405a", lens="#dff3ff", fig_iris="#3aa76a", cheek="#f6a0a8", hanky="#ffffff", hanky_lace="#f4d3e3",
    cup="#ffffff", cup_band="#f07fa0", tea="#b07a45", pencil="#ffd23c", pencil_tip="#f2a07a").items()}


# ================================================================================================== camera-aware cel
class Cel:
    def __init__(self, ctx, base, outer=6.0, inner=3.6, lw=1.0):
        self.ctx, self.base, self.ops = ctx, base, []
        self.outer, self.inner = outer * lw, inner * lw
        self.binv = cairo.Matrix(*base)
        self.binv.invert()
        self.bscale = math.sqrt(abs(base.xx * base.yy - base.xy * base.yx))

    def shape(self, fn, color, sil=True, line=None):
        self.ops.append(("shape", self.ctx.get_matrix(), fn, color, sil, line))

    def tube(self, pts, w, color, sil=True):
        m = self.ctx.get_matrix()
        wp = [np.array(self.binv.transform_point(*m.transform_point(*q))) for q in pts]
        sc = math.sqrt(abs(m.xx * m.yy - m.xy * m.yx)) / self.bscale
        self.ops.append(("tube", None, wp, color, sil, w * sc))

    def line(self, fn, w, color=INK):
        self.ops.append(("line", self.ctx.get_matrix(), fn, color, False, w))

    def raw(self, fn):
        self.ops.append(("raw", self.ctx.get_matrix(), fn, None, False, None))

    def _path(self, op, nudge=None):
        ctx = self.ctx
        if op[0] == "tube":
            m = self.base if nudge is None else cairo.Matrix(*self.base) * nudge
            ctx.set_matrix(m)
            cr_path(ctx, op[2], tension=.9) if len(op[2]) > 2 else poly(ctx, op[2], closed=False)
        else:
            m = op[1] if nudge is None else cairo.Matrix(*op[1]) * nudge
            ctx.set_matrix(m)
            op[2](ctx)
        ctx.set_matrix(self.base)

    def draw(self):
        ctx = self.ctx
        ctx.save()
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_matrix(self.base)
        k = self.outer / 6.0  # the down-right weighting nudge, in device pixels
        nudge = cairo.Matrix(1, 0, 0, 1, 1.1 * k * self.bscale, 1.6 * k * self.bscale)
        for op in self.ops:  # pass 1: heavy silhouette, twice, the second nudged down-right
            if not op[4]:
                continue
            w = self.outer * 2 + (op[5] if op[0] == "tube" else 0)
            for nd in (None, nudge):
                self._path(op, nd)
                stroke(ctx, INK, w)
        for op in self.ops:  # pass 2
            kind = op[0]
            if kind == "shape":
                self._path(op)
                fill(ctx, op[3], True)
                stroke(ctx, INK, (op[5] * self.inner / 3.6) if op[5] else self.inner)
            elif kind == "tube":
                self._path(op)
                stroke(ctx, INK, op[5] + self.inner * 2, True)
                stroke(ctx, op[3], op[5])
            elif kind == "line":
                self._path(op)
                stroke(ctx, op[3], op[5] * self.inner / 3.6)
            else:
                ctx.save()
                ctx.set_matrix(op[1])
                op[2](ctx)
                ctx.restore()
        ctx.restore()
        ctx.set_matrix(self.base)


# ================================================================================================== faces
def eye(ctx, cx, cy, rx, ry, p, side, skin=None, iris=None):
    """Big cartoon eye with modes: normal, blink, closed_happy, spiral (dizzy); lid with slant; pupil size."""
    skin = skin or P["skin"]
    iris = iris or P["iris"]
    lx, ly = p["look"]
    mode = p.get("eyes", "normal")
    if p.get("closed_happy", 0) > .5:
        cr_path(ctx, [(cx - rx * .8, cy + ry * .15), (cx, cy - ry * .45), (cx + rx * .8, cy + ry * .15)], tension=1.0)
        stroke(ctx, INK, 7)
        return
    if p.get("blink", 0) > .7:
        cr_path(ctx, [(cx - rx * .95, cy - ry * .05), (cx, cy + ry * .28), (cx + rx * .95, cy - ry * .05)], tension=1.0)
        stroke(ctx, INK, 7)
        return
    ellipse(ctx, cx, cy, rx, ry)
    fill(ctx, WHITE)
    ctx.save()
    ellipse(ctx, cx, cy, rx, ry)
    ctx.clip()
    if mode == "spiral":
        pts = []
        for j in range(90):
            a = j * .32 + side * .8
            r = (j / 90) * rx * .85
            pts.append((cx + r * math.cos(a) * side, cy + r * math.sin(a) * ry / rx))
        cr_path(ctx, pts, tension=.8)
        stroke(ctx, INK, 4.2)
    else:
        ps = p.get("pupil", 1.0)
        ix, iy = cx + lx * rx * .36, cy + ly * ry * .28 + ry * .14
        ir = rx * .76 * (.72 + .28 * ps) if ps < 1 else rx * .76
        ellipse(ctx, ix, iy, ir, ir * 1.12)
        fill(ctx, iris, True)
        stroke(ctx, INK, 3.0)
        ellipse(ctx, ix, iy + ir * .04, ir * .44 * ps, ir * .5 * ps)
        fill(ctx, INK)
        ellipse(ctx, ix + ir * .34, iy - ir * .34, ir * .3, ir * .34, .35)
        fill(ctx, WHITE)
        ellipse(ctx, ix - ir * .36, iy + ir * .44, ir * .13, ir * .13)
        fill(ctx, WHITE)
    lid = max(p.get("lid", 0), p.get("blink", 0) * 1.2)
    sl = p.get("slant", 6.0)
    if lid > 0:
        ly0 = cy - ry + 2 * ry * lid * .62
        poly(ctx, [(cx - rx - 6, cy - ry - 6), (cx + rx + 6, cy - ry - 6), (cx + rx + 6, ly0 - sl * side), (cx - rx - 6, ly0 + sl * side)])
        fill(ctx, skin)
        poly(ctx, [(cx - rx - 4, ly0 + sl * side), (cx + rx + 4, ly0 - sl * side)], closed=False)
        stroke(ctx, INK, 6)
    if p.get("happy", 0) > 0:
        hy = cy + ry * (1.05 - .75 * p["happy"])
        cr_path(ctx, [(cx - rx * 1.2, cy + ry * 1.4), (cx - rx * 1.2, hy + ry * .2), (cx, hy - ry * .08), (cx + rx * 1.2, hy + ry * .2),
                      (cx + rx * 1.2, cy + ry * 1.4)], closed=True, tension=.8)
        fill(ctx, skin, True)
        stroke(ctx, INK, 4.5)
    ctx.restore()
    ellipse(ctx, cx, cy, rx, ry)
    stroke(ctx, INK, 4.2)
    if lid <= .05:
        ctx.new_sub_path()
        ctx.save()
        ctx.translate(cx, cy)
        ctx.scale(rx, ry)
        ctx.arc(0, 0, 1, math.pi * 1.08, math.pi * 1.92)
        ctx.restore()
        stroke(ctx, INK, 8.5)


def mouth(ctx, mx, my, p):
    kind, o = p["mouth"], p.get("mopen", 0.0)
    if kind == "smile":
        cr_path(ctx, [(mx - 17, my - 3), (mx - 4, my + 4), (mx + 10, my + 2), (mx + 19, my - 6)], tension=1.0)
        stroke(ctx, INK, 4.6)
        poly(ctx, [(mx + 17, my - 9), (mx + 22, my - 3)], closed=False)
        stroke(ctx, INK, 3.4)
        return
    if kind == "set":
        cr_path(ctx, [(mx - 14, my + 1), (mx, my - 1), (mx + 15, my + 2)], tension=1.0)
        stroke(ctx, INK, 4.6)
        return
    if kind == "flat":
        poly(ctx, [(mx - 13, my), (mx + 13, my)], closed=False)
        stroke(ctx, INK, 4.6)
        return
    if kind == "sheepish":  # a wobbly, embarrassed smile
        cr_path(ctx, [(mx - 22, my - 2), (mx - 12, my + 3), (mx - 3, my - 1), (mx + 7, my + 4), (mx + 16, my), (mx + 23, my - 5)], tension=1.0)
        stroke(ctx, INK, 4.4)
        return
    if kind == "whistle":
        ellipse(ctx, mx + 2, my - 1, 6, 6)
        fill(ctx, P["mouth"], True)
        stroke(ctx, INK, 3.6)
        return
    if kind == "puff":  # cheeks full of air: tight little pucker
        ellipse(ctx, mx + 2, my - 2, 7, 5)
        fill(ctx, P["mouth"], True)
        stroke(ctx, INK, 3.6)
        for sx in (-1, 1):
            ctx.new_sub_path()
            ctx.arc(mx + 2 + sx * 10, my - 2, 6, math.pi * (0.6 if sx > 0 else -.4), math.pi * (1.4 if sx > 0 else .4))
            stroke(ctx, INK, 2.6)
        return
    if kind == "teeth":  # gritted teeth
        rrect(ctx, mx - 22, my - 12, mx + 24, my + 6, 7)
        fill(ctx, WHITE, True)
        stroke(ctx, INK, 4.0)
        poly(ctx, [(mx - 20, my - 3), (mx + 22, my - 3)], closed=False)
        stroke(ctx, INK, 2.6)
        for xx in (-8, 4, 14):
            poly(ctx, [(mx + xx, my - 11), (mx + xx, my + 5)], closed=False)
        stroke(ctx, INK, 2.2)
        return
    wdt = {"talk": 20, "grin": 27, "o": 11, "jaw": 22}[kind]
    hgt = {"talk": 8 + 14 * o, "grin": 19, "o": 9 + 8 * o, "jaw": 40}[kind]
    if kind == "o":
        ellipse(ctx, mx, my + 2, wdt, hgt)
    elif kind == "jaw":
        ellipse(ctx, mx, my + 10, wdt, hgt * .62)
    else:
        ctx.move_to(mx - wdt, my - 6)
        ctx.curve_to(mx - wdt * .6, my - 9, mx + wdt * .6, my - 9, mx + wdt, my - 6)
        ctx.curve_to(mx + wdt * .9, my - 6 + hgt, mx - wdt * .9, my - 6 + hgt, mx - wdt, my - 6)
        ctx.close_path()
    if kind == "grin":
        hgt = 26
        ctx.new_path()
        ctx.move_to(mx - wdt - 4, my - 9)
        ctx.curve_to(mx - wdt * .5, my - 4, mx + wdt * .5, my - 4, mx + wdt + 4, my - 10)
        ctx.curve_to(mx + wdt * .8, my + 20, mx - wdt * .8, my + 20, mx - wdt - 4, my - 9)
        ctx.close_path()
        fill(ctx, P["mouth"], True)
        ctx.save()
        ctx.clip_preserve()
        ctx.new_path()
        ctx.rectangle(mx - 60, my - 30, 120, 30)
        fill(ctx, WHITE)
        ellipse(ctx, mx + 2, my + 14, wdt * .5, 7)
        fill(ctx, P["tongue"])
        ctx.restore()
        ctx.new_path()
        ctx.move_to(mx - wdt - 4, my - 9)
        ctx.curve_to(mx - wdt * .5, my - 4, mx + wdt * .5, my - 4, mx + wdt + 4, my - 10)
        ctx.curve_to(mx + wdt * .8, my + 20, mx - wdt * .8, my + 20, mx - wdt - 4, my - 9)
        ctx.close_path()
        stroke(ctx, INK, 4.2)
        return
    fill(ctx, P["mouth"], True)
    ctx.save()
    ctx.clip_preserve()
    yb = my + 10 + hgt * .62 if kind == "jaw" else my - 6 + hgt * 1.02
    ellipse(ctx, mx + 2, yb - 4, wdt * .62, hgt * (.25 if kind == "jaw" else .45))
    fill(ctx, P["tongue"])
    ctx.restore()
    stroke(ctx, INK, 4.2)


def foam_blobs(c, blobs):
    for (x, y, r) in blobs:
        ellipse(c, x, y, r, r * .9)


def sweat_drop(c, x, y, s=1.0):
    c.move_to(x, y - 22 * s)
    c.curve_to(x + 13 * s, y - 2 * s, x + 12 * s, y + 12 * s, x, y + 12 * s)
    c.curve_to(x - 12 * s, y + 12 * s, x - 13 * s, y - 2 * s, x, y - 22 * s)
    c.close_path()
    fill(c, P2["sweat"], True)
    stroke(c, INK, 3.4)
    ellipse(c, x - 4 * s, y + 2 * s, 3 * s, 5 * s)
    fill(c, WHITE)


def puff_cloud(c, x, y, s):
    for dx, dy, r in ((-14, 4, 13), (0, -6, 16), (15, 3, 12), (2, 9, 12)):
        ellipse(c, x + dx * s, y + dy * s, r * s, r * s)
    fill(c, WHITE, True)
    stroke(c, INK, 3.2)
    for dx, dy, r in ((-14, 4, 13), (0, -6, 16), (15, 3, 12), (2, 9, 12)):
        ellipse(c, x + dx * s, y + dy * s, r * s - 1.5, r * s - 1.5)
    fill(c, WHITE)


# ================================================================================================== Squeaky
SQ = dict(draw.REST)
SQ.update(eyes="normal", pupil=1.0, slant=6.0, cheeks=0.0, red=0.0, sweat=False, steam=False, prop="squeegee", hat="hat",
          hat_rot=-.06, hat_dx=0.0, hang_bucket=True, coil=False, foam=False, stars=False, mirror=False, plank_tilt=0.0,
          polisher_at=(182.0, -176.0), bucket_at=(232.0, -214.0), rope_top=-980.0, rig=True)


def squeaky(ctx, base, p, origin, scale=1.0, lw=1.0):
    """Squeaky in his bosun's chair.  origin = world position of the plank-seat centre; local units as in the prototype."""
    cel = Cel(ctx, base, lw=lw)
    ctx.set_matrix(base)
    ctx.translate(*origin)
    ctx.scale(scale * (-1 if p["mirror"] else 1), scale)
    ctx.translate(0, -320)
    ctx.rotate(p["sway"])
    ctx.translate(0, 320)
    rig_m = ctx.get_matrix()
    ring = np.array([0.0, -452.0])
    if not p["rig"]:
        pass
    elif p["coil"]:  # the rope above the ring, wound up tight from spinning
        pts = [(0, -980)]
        for j in range(16):
            y = -760 + j * 18
            pts.append(((-16 if j % 2 else 16), y))
        pts.append(tuple(ring + (0, -10)))
        cel.tube([pts[0], (0, -770)], 9, P["rope"])
        cel.tube(pts[1:], 9, P["rope"])
    else:
        cel.tube([(0, p["rope_top"]), tuple(ring + (0, -10))], 9, P["rope"])
    if p["rig"]:
        for sx in (-1, 1):
            cel.tube([tuple(ring), (sx * 160, 6)], 8, P["rope"])
        cel.shape(lambda c: ellipse(c, ring[0], ring[1], 14, 14), P["steel"])
        cel.shape(lambda c: ellipse(c, ring[0], ring[1], 6, 6), P["rubber"], sil=False, line=2.5)
    bx, by = -198.0, 68.0
    if p["hang_bucket"]:
        cel.line(lambda c: cr_path(c, [(bx - 44, by + 2), (bx - 30, by - 36), (-170, 20), (bx + 30, by - 36), (bx + 44, by + 2)], tension=.9), 4.2)
    cel.shape(lambda c: poly(c, [(-176, -12), (176, -12), (184, 2), (-184, 2)]), hexc("#f0c07e"))
    cel.shape(lambda c: rrect(c, -184, 0, 184, 26, 4), P["wood"])
    cel.line(lambda c: poly(c, [(-150, 9), (-60, 9)], closed=False), 2.2, P["wood_ln"])
    cel.line(lambda c: poly(c, [(40, 15), (140, 15)], closed=False), 2.2, P["wood_ln"])
    for sx in (-1, 1):
        cel.shape(lambda c, sx=sx: ellipse(c, sx * 160, 6, 7, 7), P["steel"], sil=False, line=2.6)
    if p["hang_bucket"]:
        cel.shape(lambda c: poly(c, [(bx - 46, by), (bx + 46, by), (bx + 37, by + 74), (bx - 37, by + 74)]), P["bucket"])
        cel.shape(lambda c: ellipse(c, bx, by, 46, 11), P["bucket_in"], sil=False)
        cel.line(lambda c: poly(c, [(bx - 42, by + 18), (bx + 42, by + 18)], closed=False), 3.0)
        cel.shape(lambda c: foam_blobs(c, [(bx - 28, by - 6, 15), (bx - 8, by - 14, 19), (bx + 16, by - 10, 16), (bx + 33, by - 3, 12),
                                           (bx - 38, by + 2, 9)]), P["suds"], line=3.2)
    if p["prop"] == "polisher_dangle":  # the Polish-o-Matic hanging limp under the plank on its cord
        pc = np.array([130.0, 120.0])
        cel.tube([(96, -70), (120, 0), tuple(pc + (-10, -40))], 6, P2["cord"])
        polisher(cel, pc, -1.35, 1.15)
    # --- body
    ctx.translate(0, -p["hop"])
    ctx.rotate(p["lean"])
    ctx.scale(1 / math.sqrt(p["sq"]), p["sq"])
    body_m = ctx.get_matrix()
    for side, sw in ((-1, p["legL"]), (1, p["legR"])):
        kx = side * 36
        foot = np.array([side * 40 + 24 * math.sin(sw), 82 - 8 * (1 - math.cos(sw))])
        cel.tube([(kx, -6), (kx + side * 2, 12), tuple(foot - (0, 14))], 36, P["suit"])
        fx, fy = foot
        boot = lambda c, fx=fx, fy=fy, side=side: cr_path(c, [(fx - 19, fy - 40), (fx + 19, fy - 40), (fx + 20, fy - 8), (fx + 24 + side * 8, fy - 2),
                                                           (fx + 26 + side * 8, fy + 14), (fx - 4, fy + 18), (fx - 24, fy + 12), (fx - 21, fy - 10)],
                                                          closed=True, tension=.55)
        cel.shape(boot, P["boot"])
        cel.shape(lambda c, fx=fx, fy=fy: rrect(c, fx - 23, fy - 46, fx + 23, fy - 34, 5), hexc("#ff6a6f"), line=3.4)
        cel.line(lambda c, fx=fx, fy=fy, side=side: poly(c, [(fx - 22, fy + 8), (fx + 25 + side * 8, fy + 8)], closed=False), 3.0)
    shirt = [(-46, -152), (46, -152), (60, -120), (64, -60), (-64, -60), (-60, -120)]
    cel.shape(lambda c: cr_path(c, shirt, closed=True, tension=.6), P["tee"])
    pants = [(-66, -70), (66, -70), (72, -34), (76, -8), (40, -2), (0, -4), (-40, -2), (-76, -8), (-72, -34)]
    cel.shape(lambda c: cr_path(c, pants, closed=True, tension=.6), P["suit"])
    for side in (-1, 1):
        kx = side * 38
        cel.shape(lambda c, kx=kx: ellipse(c, kx, 2, 30, 19), P["suit"], sil=False, line=3.6)
    bib = lambda c: (c.move_to(-38, -64), c.line_to(-38, -112), c.curve_to(-38, -122, -32, -124, -24, -124),
                     c.line_to(24, -124), c.curve_to(32, -124, 38, -122, 38, -112), c.line_to(38, -64), c.close_path())
    cel.shape(bib, P["suit"], sil=False)
    for sx in (-1, 1):
        cel.shape(lambda c, sx=sx: poly(c, [(sx * 24, -120), (sx * 38, -120), (sx * 48, -152), (sx * 32, -152)]), P["suit"], sil=False, line=3.2)
        cel.shape(lambda c, sx=sx: ellipse(c, sx * 28, -112, 7, 7), INK, sil=False, line=1.0)

    def bib_detail(c):
        rrect(c, -20, -106, 20, -74, 4)
        fill(c, P["suit"], True)
        stroke(c, INK, 3.0)
        ellipse(c, 0, -90, 11, 11)
        fill(c, WHITE, True)
        stroke(c, INK, 2.6)
        c.move_to(0, -98)
        c.curve_to(5, -91, 6, -87, 0, -83)
        c.curve_to(-6, -87, -5, -91, 0, -98)
        fill(c, P["pole"])
    cel.raw(bib_detail)
    if p["foam"]:  # suds all over his lap and shoulders
        cel.shape(lambda c: foam_blobs(c, [(-50, -14, 22), (-18, -22, 18), (20, -16, 24), (56, -10, 18), (-70, -120, 16), (66, -126, 18),
                                           (-120, 2, 14), (128, 4, 16)]), P2["foam"], line=3.0)
    # left arm (unless the prop needs both hands)
    shL, eL = np.array((-44.0, -128.0)), np.array(p["elbowL"])
    cel.tube([tuple(shL), p["elbowL"], p["handL"]], 25, P["skin"])
    cel.tube([tuple(shL), tuple(shL + (eL - shL) * .5)], 36, P["tee"], sil=False)
    hl = p["handL"]
    cel.shape(lambda c: ellipse(c, hl[0], hl[1] + 2, 21, 24, -.3), P["skin"])
    # --- head
    ctx.translate(p["head_dx"], -150 + p["head_dy"])
    ctx.rotate(p["head_tilt"])
    ctx.translate(0, 150)
    hcx, hcy, hrx, hry = 0.0, -272.0, 152.0, 120.0
    tw = p["turn"]
    sb = 1 + .14 * p["cheeks"]
    head_path = lambda c: superellipse(c, hcx, hcy, hrx, hry, 2.35, sb)
    cel.shape(head_path, P["skin"])
    if p["red"] > 0:  # anger rising from the chin up
        def redden(c):
            c.save()
            head_path(c)
            c.clip()
            lvl = hcy + hry - 2 * hry * p["red"]
            c.rectangle(-400, lvl, 800, 400)
            fill(c, P2["red_face"])
            c.restore()
            head_path(c)
            stroke(c, INK, 3.6)
        cel.raw(redden)
    for side in (-1, 1):
        pts = [(140, -332), (172, -328), (152, -314), (180, -302), (150, -296), (162, -280), (138, -286)]
        cel.shape(lambda c, side=side, pts=pts: poly(c, [(side * x + tw * 6, y) for x, y in pts]), INK, line=2.0)
    fx = tw * 18
    ex, ey, erx, ery = 59.0, -246.0, 52.0, 58.0
    skin_now = P2["red_face"] if p["red"] > .55 else P["skin"]

    def face(c):
        eye(c, fx - ex, ey, erx * (1 + .05 * tw), ery, p, -1, skin=skin_now)
        eye(c, fx + ex, ey, erx * (1 - .13 * tw), ery * (1 - .03 * tw), p, 1, skin=skin_now)
        if p["red"] < .3:
            for sx in (-1, 1):
                for dx, dy in ((0, 0), (11, -4), (8, 8)):
                    ellipse(c, fx + sx * (72 + dx) - (8 if sx > 0 else 0) * tw, -178 + dy, 3.2, 3.2)
            fill(c, hexc("#e7976f"))
        mouth(c, fx + 4, -170, p)
        if p["sweat"]:
            sweat_drop(c, 150 + tw * 10, -300, 1.2)
    cel.raw(face)
    # --- hat (or the bucket that replaced it)
    ctx.translate(p["hat_dx"], -340)
    ctx.rotate(p["hat_rot"])
    ctx.translate(0, 340)
    if p["hat"] == "hat":
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
    else:  # the suds bucket, upside down on his head, foam oozing out
        cel.line(lambda c: cr_path(c, [(-150, -318), (-176, -250), (-150, -200)], tension=1.0), 4.4)  # dangling handle
        cel.shape(lambda c: poly(c, [(-166, -306), (166, -306), (128, -470), (-128, -470)]), P["bucket"])
        cel.shape(lambda c: rrect(c, -174, -320, 174, -298, 8), hexc("#ff5a5f"), line=3.6)
        cel.line(lambda c: poly(c, [(-140, -400), (140, -400)], closed=False), 3.0)
        cel.shape(lambda c: foam_blobs(c, [(-150, -300, 20), (-110, -296, 16), (-60, -300, 14), (40, -298, 15), (100, -296, 18), (152, -302, 20),
                                           (-10, -478, 26), (30, -486, 18), (-48, -476, 16)]), P2["foam"], line=3.0)

        def drips(c):
            for (x, y0, L) in ((-160, -296, 70), (-118, -290, 40), (126, -292, 56), (162, -296, 90)):
                c.move_to(x - 9, y0)
                c.line_to(x - 7, y0 + L)
                c.curve_to(x - 7, y0 + L + 14, x + 7, y0 + L + 14, x + 7, y0 + L)
                c.line_to(x + 9, y0)
                c.close_path()
                fill(c, WHITE, True)
                stroke(c, INK, 3.0)
        cel.raw(drips)
    if p["steam"]:
        cel.raw(lambda c: (puff_cloud(c, -196, -400, 1.3), puff_cloud(c, 196, -410, 1.15), puff_cloud(c, -236, -460, .8)))
    if p["stars"]:
        def stars(c):
            for k, (x, y) in enumerate(((-120, -470), (10, -500), (130, -466))):
                pts = []
                for j in range(10):
                    r = 20 if j % 2 == 0 else 8
                    a = -math.pi / 2 + j * math.pi / 5 + k * .3
                    pts.append((x + r * math.cos(a), y + r * math.sin(a)))
                poly(c, pts)
                fill(c, P["hat"], True)
                stroke(c, INK, 3.2)
            c.new_sub_path()
            c.save()
            c.translate(5, -480)
            c.scale(170, 34)
            c.arc(0, 0, 1, math.pi * .1, math.pi * .9)
            c.restore()
            stroke(c, INK, 2.6)
        cel.raw(stars)
    # --- right arm + prop (body space)
    ctx.set_matrix(body_m)
    if p["prop"] == "squeegee":
        g = np.array(p["grip"])
        hd = np.array(p["sq_head"])
        d = (hd - g) / (np.linalg.norm(hd - g) + 1e-9)
        nrm = np.array([-d[1], d[0]])
        butt = g - d * 78
        ba = p["blade"]
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
        hand_r, hand_ang = g, math.atan2(d[1], d[0])
    elif p["prop"] == "bucket_up":  # heaving the suds bucket toward the window
        bc = np.array(p["bucket_at"])
        a, k = -.5, 1.3
        R = lambda v: bc + k * np.array([v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a)])
        body = [R((-46, -40)), R((46, -40)), R((37, 36)), R((-37, 36))]
        cel.shape(lambda c: poly(c, [tuple(q) for q in body]), P["bucket"])
        cel.shape(lambda c: (ellipse(c, *R((0, -40)), 46 * k, 11 * k, a)), P["bucket_in"], sil=False)
        cel.shape(lambda c: foam_blobs(c, [tuple(R((-26, -52))) + (20,), tuple(R((-2, -62))) + (25,), tuple(R((22, -54))) + (20,),
                                           tuple(R((40, -66))) + (15,), tuple(R((8, -84))) + (17,), tuple(R((62, -80))) + (11,)]), P["suds"], line=3.0)
        cel.line(lambda c: poly(c, [tuple(R((-42, -22))), tuple(R((42, -22)))], closed=False), 3.0)
        hand_r, hand_ang = R((-44, -20)), 0.0
    elif p["prop"] == "polisher":
        pc = np.array(p["polisher_at"])
        cel.tube([tuple(pc + (-14, -50)), tuple(pc + (6, -150)), tuple(pc + (70, -260)), tuple(pc + (60, -420)), tuple(pc + (120, -700))], 6, P2["cord"])
        polisher(cel, pc, 0.0, 1.3)
        hand_r, hand_ang = pc + (-26, -48), 0.0
    else:
        hand_r, hand_ang = np.array(p["grip"]), 0.0
    shR, eR = np.array((44.0, -128.0)), np.array(p["elbowR"])
    cel.tube([tuple(shR), p["elbowR"], tuple(hand_r)], 25, P["skin"])
    cel.tube([tuple(shR), tuple(shR + (eR - shR) * .5)], 36, P["tee"], sil=False)
    cel.shape(lambda c: ellipse(c, hand_r[0], hand_r[1], 22, 24, hand_ang), P["skin"])
    cel.draw()
    return rig_m


def polisher(cel, pc, a, k=1.0):
    """The Polish-o-Matic: a chrome motor drum with a red racing stripe and a big fluffy buffing pad."""
    x, y = pc
    ca, sa = math.cos(a) * k, math.sin(a) * k
    R = lambda v: (x + v[0] * ca - v[1] * sa, y + v[0] * sa + v[1] * ca)

    def pad(c):
        pts = []
        for j in range(18):
            t = j / 18 * 2 * math.pi
            r = 1 + .08 * (j % 2)
            pts.append(R((54 + 18 * math.cos(t) * r, 66 * math.sin(t) * r)))
        cr_path(c, pts, closed=True, tension=.9)
    cel.shape(pad, P2["pad"])
    cel.shape(lambda c: cr_path(c, [R((-20, -40)), R((40, -40)), R((40, 40)), R((-20, 40)), R((-56, 22)), R((-62, 0)), R((-56, -22))],
                                closed=True, tension=.55), P2["chrome"])
    cel.shape(lambda c: poly(c, [R((-60, -9)), R((40, -9)), R((40, 8)), R((-60, 8))]), P2["stripe"], sil=False, line=2.8)
    cel.shape(lambda c: poly(c, [R((40, -30)), R((50, -30)), R((50, 30)), R((40, 30))]), P2["chrome_dk"], sil=False, line=2.8)
    cel.line(lambda c: (c.move_to(*R((-30, -40))), c.curve_to(*R((-30, -68)), *R((10, -68)), *R((10, -40)))), 7.0)
    cel.raw(lambda c: (poly(c, [R((-36, -28)), R((-20, -28))], closed=False), stroke(c, WHITE, 5)))


# ================================================================================================== Mrs. Fig
FIG = dict(lean=0.0, head_tilt=0.0, look=(.4, -.1), blink=0.0, lid=.18, slant=-3.0, happy=0.0, closed_happy=0.0, mouth="smile",
           handL=(-60.0, -30.0), elbowL=(-84.0, -60.0), handR=(70.0, -30.0), elbowR=(92.0, -60.0), hold_l=None, hold_r="cup",
           sweat=False)


def fig_mouth(c, mx, my, kind):
    if kind == "smile":
        cr_path(c, [(mx - 14, my - 2), (mx, my + 5), (mx + 14, my - 2)], tension=1.0)
        stroke(c, INK, 4.2)
    elif kind == "smug":
        cr_path(c, [(mx - 14, my + 1), (mx, my + 4), (mx + 16, my - 4)], tension=1.0)
        stroke(c, INK, 4.2)
        poly(c, [(mx + 14, my - 8), (mx + 19, my - 2)], closed=False)
        stroke(c, INK, 3.0)
    elif kind == "flat":
        poly(c, [(mx - 12, my + 1), (mx + 12, my + 1)], closed=False)
        stroke(c, INK, 4.2)
    elif kind == "o":
        ellipse(c, mx, my + 2, 8, 9)
        fill(c, P["mouth"], True)
        stroke(c, INK, 3.6)


def teacup(cel, x, y, s=1.0, tilt=0.0):
    ca, sa = math.cos(tilt), math.sin(tilt)
    R = lambda vx, vy: (x + (vx * ca - vy * sa) * s, y + (vx * sa + vy * ca) * s)
    cel.line(lambda c: (c.move_to(*R(26, -18)), c.curve_to(*R(48, -20), *R(48, 8), *R(22, 6))), 6.0)
    cel.shape(lambda c: (c.move_to(*R(-30, -26)), c.line_to(*R(30, -26)), c.curve_to(*R(30, 8), *R(18, 24), *R(0, 24)),
                         c.curve_to(*R(-18, 24), *R(-30, 8), *R(-30, -26)), c.close_path()), P2["cup"])
    cel.shape(lambda c: poly(c, [R(-30, -16), R(30, -16), R(29, -8), R(-29, -8)]), P2["cup_band"], sil=False, line=2.4)
    cel.shape(lambda c: ellipse(c, *R(0, -26), 30 * s, 7 * s, tilt), P2["tea"], sil=False, line=2.8)


def hanky(cel, x, y, s=1.0, a=0.0):
    ca, sa = math.cos(a), math.sin(a)
    R = lambda vx, vy: (x + (vx * ca - vy * sa) * s, y + (vx * sa + vy * ca) * s)

    def lace(c):
        pts = []
        corners = [(-34, -30), (30, -38), (36, 30), (-28, 36)]
        for k in range(4):
            p0, p1 = np.array(corners[k]), np.array(corners[(k + 1) % 4])
            for j in range(4):
                q = p0 + (p1 - p0) * (j / 4)
                nrm = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / np.linalg.norm(p1 - p0)
                pts.append(R(*(q)))
                pts.append(R(*(p0 + (p1 - p0) * ((j + .5) / 4) - nrm * 6)))
        cr_path(c, pts, closed=True, tension=.6)
    cel.shape(lace, P2["hanky"])
    cel.shape(lambda c: ellipse(c, *R(20, 22), 7 * s, 7 * s), P2["cup_band"], sil=False, line=2.4)  # a little embroidered rose


def mrs_fig(ctx, base, p, origin, scale=1.0, lw=1.0, part="body"):
    """Mrs. Fig sitting at her desk.  origin = her waist at desk-top height.  part: 'body' (behind the desk) or 'arms'."""
    cel = Cel(ctx, base, lw=lw)
    ctx.set_matrix(base)
    ctx.translate(*origin)
    ctx.scale(scale, scale)
    ctx.rotate(p["lean"])
    body_m = ctx.get_matrix()
    if part == "body":
        torso = [(-58, -128), (58, -128), (74, -60), (84, 30), (-84, 30), (-74, -60)]
        cel.shape(lambda c: cr_path(c, torso, closed=True, tension=.5), P2["cardigan"])
        cel.line(lambda c: poly(c, [(0, -118), (0, 30)], closed=False), 3.0)
        cel.raw(lambda c: ([ellipse(c, 10, yy, 5, 5) for yy in (-80, -46, -12)], fill(c, P2["collar"], True), stroke(c, INK, 2.2)))
        # pearls
        cel.raw(lambda c: ([ellipse(c, 40 * math.sin(t), -118 + 30 * math.cos(t), 6.5, 6.5) for t in np.linspace(-1.15, 1.15, 11)],
                           fill(c, P2["pearl"], True), stroke(c, INK, 2.2)))
        for sx in (-1, 1):  # Peter Pan collar
            cel.shape(lambda c, sx=sx: (c.move_to(0, -132), c.curve_to(sx * 20, -134, sx * 50, -132, sx * 52, -118),
                                        c.curve_to(sx * 52, -100, sx * 20, -98, 0, -116), c.close_path()), P2["collar"], sil=False, line=3.0)
        ctx.translate(0, -128)
        ctx.rotate(p["head_tilt"])
        ctx.translate(0, 128)
        hx, hy, hrx, hry = 0.0, -236.0, 128.0, 104.0
        # bun with a pencil through it, behind the head
        cel.shape(lambda c: poly(c, [(-62, -418), (-48, -426), (58, -334), (46, -326)]), P2["pencil"], line=3.0)
        cel.shape(lambda c: poly(c, [(-62, -418), (-48, -426), (-70, -440)]), P2["pencil_tip"], line=3.0)
        cel.shape(lambda c: ellipse(c, 0, -362, 54, 46), P2["fig_hair"])
        cel.line(lambda c: (c.new_sub_path(), c.arc(0, -362, 30, math.pi * 1.1, math.pi * 1.9)), 3.0, P2["fig_hair2"])
        cel.shape(lambda c: superellipse(c, hx, hy, hrx, hry, 2.35, 1.0), P2["fig_skin"])

        def hair(c):  # a soft grey cap with scalloped waves along the hairline
            c.move_to(-132, -230)
            c.curve_to(-140, -330, -60, -350, 0, -348)
            c.curve_to(60, -350, 140, -330, 132, -230)
            c.curve_to(120, -262, 98, -282, 84, -284)
            for k in range(4):
                xa = 84 - (k + 1) * 42
                c.curve_to(xa + 32, -300, xa + 10, -268, xa, -286)
            c.curve_to(-96, -282, -120, -262, -132, -230)
            c.close_path()
        cel.shape(hair, P2["fig_hair"])
        cel.line(lambda c: (c.move_to(-60, -330), c.curve_to(-30, -340, 10, -340, 40, -330)), 2.6, P2["fig_hair2"])

        def face(c):
            for sx in (-1, 1):
                ellipse(c, sx * 84, -186, 18, 11)
                fill(c, P2["cheek"])
            pf = dict(p)
            for sx in (-1, 1):
                eye(c, sx * 52, -238, 34, 38, pf, sx, skin=P2["fig_skin"], iris=P2["fig_iris"])
                if not p.get("closed_happy") and p.get("blink", 0) < .7:  # two little lashes at the outer corner
                    for j in range(2):
                        a0 = -.35 - j * .38
                        x0, y0 = sx * (52 + 30 * math.cos(a0)), -238 + 34 * math.sin(a0)
                        poly(c, [(x0, y0), (x0 + sx * 14 * math.cos(a0 - .1), y0 + 14 * math.sin(a0 - .1))], closed=False)
                        stroke(c, INK, 3.4)
            fig_mouth(c, 0, -182, p["mouth"])
            for sx in (-1, 1):  # big round red specs
                ellipse(c, sx * 52, -236, 50, 48)
                fill(c, hexc("#dff3ff", .18), True)
                stroke(c, P2["glasses"], 7.5)
                ellipse(c, sx * 52, -236, 50, 48)
                stroke(c, INK, 2.0)
                poly(c, [(sx * 52 - 26, -214), (sx * 52 - 4, -264)], closed=False)
                stroke(c, hexc("#ffffff", .8), 4)
            c.move_to(-4, -246)
            c.curve_to(-2, -256, 2, -256, 4, -246)
            stroke(c, P2["glasses"], 6)
            for sx in (-1, 1):
                poly(c, [(sx * 102, -244), (sx * 128, -252)], closed=False)
                stroke(c, P2["glasses"], 6)
            if p.get("sweat"):
                sweat_drop(c, 140, -290, 1.0)
        cel.raw(face)
    else:  # arms + what the hands hold (drawn in front of the desk top)
        for side, hand, elbow, hold in ((-1, p["handL"], p["elbowL"], p["hold_l"]), (1, p["handR"], p["elbowR"], p["hold_r"])):
            sh = (side * 56, -112)
            cel.tube([sh, elbow, hand], 30, P2["cardigan"])
            cel.shape(lambda c, h=hand: ellipse(c, h[0], h[1], 19, 21), P2["fig_skin"])
            if hold == "cup":
                teacup(cel, hand[0] + side * 8, hand[1] - 18, 1.0)
            elif hold == "cup_sip":
                teacup(cel, hand[0] + side * 20, hand[1] - 20, 1.0, tilt=-.35 * side)
            elif hold == "hanky":
                hanky(cel, hand[0] + side * 6, hand[1] - 26, 1.0, .25 * side)
            if hold in ("cup", "hanky"):
                cel.shape(lambda c, h=hand: ellipse(c, h[0], h[1], 19, 21), P2["fig_skin"], sil=False)
    cel.draw()
