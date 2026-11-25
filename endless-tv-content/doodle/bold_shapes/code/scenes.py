"""Scenes and key frames for 'One Little Smudge' (bold_shapes short).

Set-ups (all 1280x704):
  A  wide exterior: the lilac tower, Squeaky small beside it on his bosun's chair
  B  medium exterior: Squeaky beside the big pane with THE smudge (prototype layout, bigger panes)
  D  close exterior: B's world seen through a 1.6x camera
  C  interior office: Mrs. Fig at her desk, Squeaky outside the picture window
"""
import math

import cairo
import cv2
import numpy as np

import draw
from draw import (INK, P, WHITE, building, cloud, cr_path, ellipse, fill, googie_tower, hexc, poly, rrect, sparkle, stroke, windows)
import squeaky as SQK
from squeaky import P2, SQ, FIG

W, H, SS = 1280, 704, 2
CP = {k: hexc(v) for k, v in dict(
    wall="#f3bf8c", wall_stripe="#eeb47f", wall_ln="#c98a55", frame="#fbf2e0", frame_ln="#c49a6c", sill="#f3e3c4",
    desk_top="#d08d58", desk_top2="#e3a46d", desk_front="#e8ad4c", desk_ln="#9c5f33", knob="#9c5f33",
    lamp="#3fa596", lamp_ln="#2a776c", clock="#e9b23c", clock_ln="#b17d1c", plant="#4f9a5a", plant_ln="#2f6a3a", pot="#e0624f",
    pot_ln="#a8402f", smudge="#7a6556", smudge2="#5e4c40", print_="#8a7f86").items()}


def cam(z=1.0, cx0=0.0, cy0=0.0):
    """World -> device matrix for a camera that shows world x in [cx0, cx0 + W/z], y in [cy0, cy0 + H/z]."""
    return cairo.Matrix(SS * z, 0, 0, SS * z, -SS * z * cx0, -SS * z * cy0)


CAM_A = cam()
CAM_B = cam(1.0, 0, 8)  # the prototype world (designed at 720 high) cropped to 704
CAM_D = cam(1.6, 690 - 640 / 1.6, 262 - 352 / 1.6)
CAM_C = cam()
ORIGIN_B = (560.0, 530.0)
SMUDGE_B = (860.0, 262.0)
SMUDGE_C = (590.0, 262.0)


# ================================================================================================== layered frame
class Frame:
    def __init__(self, base):
        self.base, self.layers = base, []

    def layer(self, textured=False):
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * SS, H * SS)
        ctx = cairo.Context(surf)
        ctx.set_matrix(self.base)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        self.layers.append((surf, textured))
        return ctx

    def render(self, finish=True):
        out = np.zeros((H, W, 3), np.float32)
        tex = texture()
        for surf, textured in self.layers:
            surf.flush()
            a = np.ndarray((H * SS, W * SS, 4), np.uint8, surf.get_data()).astype(np.float32) / 255
            a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
            rgb, al = a[:, :, 2::-1], a[:, :, 3:4]
            if textured:
                rgb = rgb * tex
            out = rgb + out * (1 - al)
        if finish:
            yuv = cv2.cvtColor(np.clip(out, 0, 1), cv2.COLOR_RGB2YCrCb)
            yuv[:, :, 1:] = cv2.GaussianBlur(yuv[:, :, 1:], (0, 0), sigmaX=.9, sigmaY=.25)
            out = cv2.cvtColor(yuv, cv2.COLOR_YCrCb2RGB)
            out = cv2.GaussianBlur(out, (0, 0), sigmaX=.35, sigmaY=.3)
            rng = np.random.default_rng(77)
            out = out + cv2.resize(rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32), (W, H))[..., None] * .004
        return (np.clip(out, 0, 1) * 255 + .5).astype(np.uint8)


_TEX = None


def texture():
    """The faint gouache unevenness of a painted background card (fixed)."""
    global _TEX
    if _TEX is None:
        r = np.random.default_rng(7)
        lo = cv2.resize(r.normal(0, 1, (H // 24, W // 24)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        st = cv2.GaussianBlur(r.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigmaX=9, sigmaY=.8)
        fine = cv2.GaussianBlur(r.normal(0, 1, (H, W)).astype(np.float32), (0, 0), .7)
        _TEX = (1 + lo * .008 + st * .022 + fine * .007)[..., None]
    return _TEX


# ================================================================================================== exterior painting
def sky_and_sun(ctx, sun=(196, 150), r=78):
    ctx.rectangle(-2000, -2000, 6000, 6000)
    fill(ctx, P["sky"])
    ellipse(ctx, sun[0], sun[1], r, r)
    fill(ctx, P["sun"])
    ctx.new_sub_path()
    ctx.arc(sun[0], sun[1], r * 1.23, 0, 2 * math.pi)
    stroke(ctx, hexc("#f8f0c6", .55), 6)


def far_skyline(ctx, x0, x1, ylo, yhi, seed=11, bottom=900):
    x = x0
    rngf = np.random.default_rng(seed)
    while x < x1:
        w = rngf.uniform(36, 80)
        top = rngf.uniform(ylo, yhi)
        kind = rngf.integers(0, 4)
        col = P["far"] if rngf.uniform() < .6 else P["far2"]
        if kind == 0:
            ctx.rectangle(x, top + w / 2, w, bottom)
            ctx.new_sub_path()
            ctx.arc(x + w / 2, top + w / 2, w / 2, math.pi, 2 * math.pi)
        elif kind == 1:
            poly(ctx, [(x, bottom), (x, top + 20), (x + w * .3, top + 20), (x + w * .3, top), (x + w * .7, top), (x + w * .7, top + 20),
                       (x + w, top + 20), (x + w, bottom)])
        else:
            ctx.rectangle(x, top, w, bottom)
        fill(ctx, col)
        if kind != 0 and rngf.uniform() < .6:
            windows(ctx, x + 5, x + w - 5, top + 26, bottom, 6, 8, 10, 14, P["far_win"], .4, rngf)
        x += w + rngf.uniform(-6, 8)


def mid_city(ctx, mids, lows, seed=4):
    rng = np.random.default_rng(seed)
    for x0, x1, top, kind, c in mids:
        if c == "peri":
            building(ctx, x0, x1, top, kind, P["peri"], P["peri_ln"], hexc("#7e83cd"), rng)
        else:
            building(ctx, x0, x1, top, kind, P[c], P["mid_ln"], hexc("#57a493") if c == "mid1" else hexc("#4a9285"), rng)
    for x0, x1, top in lows:
        sk = rng.uniform(-4, 4)
        poly(ctx, [(x0, 760), (x0 + sk, top), (x1 + sk, top), (x1, 760)])
        fill(ctx, P["low"], True)
        stroke(ctx, P["low_ln"], 2.0)
        windows(ctx, x0 + 8, x1 - 8, top + 14, 760, 12, 14, 16, 20, P["low_win"], .35, rng)


def pane(ctx, x, y, w, h, lw=2.2):
    ctx.rectangle(x, y, w, h)
    fill(ctx, P["glass"], True)
    stroke(ctx, P["fac_ln"], lw)
    ctx.save()
    ctx.rectangle(x, y, w, h)
    ctx.clip()
    poly(ctx, [(x - 10, y + h), (x + w, y + h * .27), (x + w + 10, y + h + 10)])
    fill(ctx, P["glass2"])
    k = w / 120
    for j, (a, b) in enumerate(((18, 30), (40, 12))):
        s0 = (a + j * 6) * k
        poly(ctx, [(x + s0, y + 82 * k), (x + s0 + 64 * k, y + 18 * k), (x + s0 + (64 + b) * k, y + 18 * k), (x + s0 + b * k, y + 82 * k)])
        fill(ctx, hexc("#ffffff", .85))
    ctx.restore()


def facade_b(ctx, x0=740):
    ctx.rectangle(x0, -400, 1400, 1600)
    fill(ctx, P["fac"])
    ctx.rectangle(x0, -400, 32, 1600)
    fill(ctx, P["fac_dark"], True)
    stroke(ctx, P["fac_ln"], 2.2)
    for wy in (-230, 80, 390, 700):
        ctx.rectangle(x0 + 32, wy + 276, 1400, 12)
        fill(ctx, P["fac_dark"], True)
        stroke(ctx, P["fac_ln"], 1.8)
        for wx in (x0 + 60, x0 + 300, x0 + 540):
            pane(ctx, wx, wy, 210, 270, 2.4)


def paint_b(ctx):
    sky_and_sun(ctx)
    cloud(ctx, 120, 330, 190, 64, P["cloud_ln"])
    cloud(ctx, 640, 70, 190, 62, P["cloud_ln"])
    cloud(ctx, 360, 92, 120, 40, P["cloud_ln"])
    far_skyline(ctx, -400, 760, 440, 505)
    googie_tower(ctx, 236, 226)
    mids = [(-212, -100, 380, "flat", "mid1"), (-110, -10, 430, "step", "peri"),
            (-12, 92, 392, "tank", "mid2"), (84, 170, 452, "step", "peri"), (268, 350, 372, "dome", "mid1"),
            (346, 432, 456, "flat", "peri"), (430, 520, 420, "ribbon", "mid2"), (512, 602, 392, "flat", "mid1"),
            (596, 690, 448, "step", "peri"), (684, 760, 412, "flat", "mid2")]
    lows = [(-220, -10, 610), (-12, 120, 600), (112, 210, 640), (204, 300, 612), (292, 400, 652), (392, 520, 628), (512, 620, 660),
            (612, 760, 618)]
    mid_city(ctx, mids, lows)
    facade_b(ctx)


def paint_a(ctx):
    """Wide establishing shot: the city at mid-morning, the lilac tower, a davit arm at the roof."""
    sky_and_sun(ctx, (170, 128), 70)
    cloud(ctx, 420, 120, 200, 64, P["cloud_ln"])
    cloud(ctx, 1150, 210, 180, 58, P["cloud_ln"])
    cloud(ctx, 120, 300, 150, 48, P["cloud_ln"])
    far_skyline(ctx, -10, 1300, 430, 500, seed=5)
    googie_tower(ctx, 250, 236)
    mids = [(-12, 80, 420, "tank", "mid2"), (76, 160, 470, "flat", "peri"), (330, 420, 440, "dome", "mid1"), (414, 500, 480, "step", "peri"),
            (494, 600, 410, "ribbon", "mid2"), (1010, 1100, 400, "star", "peri"), (1094, 1190, 450, "flat", "mid1"),
            (1184, 1290, 420, "ribbon", "mid2")]
    lows = [(-12, 130, 600), (122, 240, 640), (232, 360, 612), (352, 470, 650), (462, 600, 620), (980, 1100, 630), (1092, 1200, 606),
            (1192, 1300, 640)]
    mid_city(ctx, mids, lows, seed=9)
    # the tower
    x0, x1, top = 700, 1000, 96
    poly(ctx, [(x0 + 30, top - 36), (x1 - 30, top - 36), (x1 - 30, top), (x0 + 30, top)])  # roof cap
    fill(ctx, P["fac_dark"], True)
    stroke(ctx, P["fac_ln"], 2.2)
    poly(ctx, [(850, top - 36), (850, top - 110)], closed=False)
    stroke(ctx, P["fac_ln"], 3)
    ellipse(ctx, 850, top - 114, 7, 7)
    fill(ctx, P["coral"], True)
    stroke(ctx, P["coral_ln"], 1.8)
    ctx.rectangle(x0, top, x1 - x0, 800)
    fill(ctx, P["fac"], True)
    stroke(ctx, P["fac_ln"], 2.4)
    ctx.rectangle(x0, top, 16, 800)
    fill(ctx, P["fac_dark"], True)
    stroke(ctx, P["fac_ln"], 2.0)
    for wy in range(top + 18, 760, 74):
        ctx.rectangle(x0 + 16, wy + 60, x1 - x0 - 16, 6)
        fill(ctx, P["fac_dark"])
        for wx in range(x0 + 34, x1 - 30, 66):
            pane(ctx, wx, wy, 50, 60, 1.6)
    # davit arm and pulley sticking out from the roof's corner
    poly(ctx, [(x0 + 40, top - 36), (x0 + 40, top - 70), (596, top - 70), (596, top - 58), (x0 + 28, top - 58), (x0 + 28, top - 36)])
    fill(ctx, P["steel"], True)
    stroke(ctx, INK, 2.6)
    ellipse(ctx, 600, top - 52, 11, 11)
    fill(ctx, P["steel_dk"], True)
    stroke(ctx, INK, 2.4)


# ================================================================================================== interior painting
WIN_C = (500, 46, 1222, 520)  # window opening


def paint_c_outside(ctx):
    sky_and_sun(ctx, (1120, 120), 54)
    cloud(ctx, 640, 120, 170, 54, P["cloud_ln"])
    cloud(ctx, 1080, 300, 140, 44, P["cloud_ln"])
    far_skyline(ctx, 480, 1240, 420, 470, seed=21)
    googie_tower(ctx, 1150, 330)
    rng = np.random.default_rng(3)
    for x0, x1, top, c in ((480, 560, 470, "peri"), (556, 640, 450, "mid1"), (640, 720, 480, "mid2"), (1010, 1090, 455, "mid1"),
                           (1086, 1240, 476, "peri")):
        building(ctx, x0, x1, top, "flat", P[c], P["peri_ln"] if c == "peri" else P["mid_ln"],
                 hexc("#7e83cd") if c == "peri" else hexc("#4a9285"), rng)


def paint_c_wall(ctx):
    """The office wall with the picture window cut out of it, the frame and the sill."""
    x0, y0, x1, y1 = WIN_C
    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    ctx.rectangle(-20, -20, W + 40, H + 40)
    ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
    fill(ctx, CP["wall"])
    ctx.save()  # faint vertical stripe wallpaper
    ctx.rectangle(-20, -20, W + 40, H + 40)
    ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
    ctx.clip()
    for xs in range(-20, W + 40, 64):
        ctx.rectangle(xs, -20, 22, H + 40)
    fill(ctx, CP["wall_stripe"])
    ctx.restore()
    ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
    f = 22
    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    ctx.rectangle(x0 - f, y0 - f, x1 - x0 + 2 * f, y1 - y0 + 2 * f)
    ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
    fill(ctx, CP["frame"], True)
    stroke(ctx, CP["frame_ln"], 2.6)
    ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
    rrect(ctx, x0 - 40, y1 + 6, x1 + 40, y1 + 34, 4)  # sill
    fill(ctx, CP["sill"], True)
    stroke(ctx, CP["frame_ln"], 2.6)
    poly(ctx, [(x0 - 36, y1 + 40), (x1 + 36, y1 + 40)], closed=False)
    stroke(ctx, CP["wall_ln"], 3)
    # starburst wall clock (no numerals)
    cx, cy = 150, 150
    for k in range(12):
        a = k * math.pi / 6
        L = 74 if k % 3 == 0 else 58
        poly(ctx, [(cx + 44 * math.cos(a), cy + 44 * math.sin(a)), (cx + L * math.cos(a), cy + L * math.sin(a))], closed=False)
        stroke(ctx, CP["clock_ln"], 7)
        poly(ctx, [(cx + 44 * math.cos(a), cy + 44 * math.sin(a)), (cx + L * math.cos(a), cy + L * math.sin(a))], closed=False)
        stroke(ctx, CP["clock"], 4)
        ellipse(ctx, cx + L * math.cos(a), cy + L * math.sin(a), 5, 5)
        fill(ctx, CP["clock"], True)
        stroke(ctx, CP["clock_ln"], 1.6)
    ellipse(ctx, cx, cy, 42, 42)
    fill(ctx, CP["frame"], True)
    stroke(ctx, CP["clock_ln"], 3)
    poly(ctx, [(cx, cy), (cx + 4, cy - 30)], closed=False)
    stroke(ctx, INK, 4)
    poly(ctx, [(cx, cy), (cx + 20, cy + 6)], closed=False)
    stroke(ctx, INK, 4)
    # snake plant in a coral pot on the sill, right corner
    px, py = 1190, y1 + 6
    for k, (dx, hgt, lean) in enumerate(((-16, 120, -.25), (-4, 150, -.05), (8, 132, .12), (18, 100, .3))):
        bx = px + dx
        tip = (bx + hgt * math.sin(lean), py - 40 - hgt)
        cr_path(ctx, [(bx - 7, py - 36), (bx - 9 + (tip[0] - bx) * .5, py - 40 - hgt * .5), tip, (bx + 9 + (tip[0] - bx) * .5, py - 40 - hgt * .5),
                      (bx + 7, py - 36)], tension=.7)
        fill(ctx, CP["plant"], True)
        stroke(ctx, CP["plant_ln"], 2.2)
    poly(ctx, [(px - 34, py - 44), (px + 34, py - 44), (px + 26, py), (px - 26, py)])
    fill(ctx, CP["pot"], True)
    stroke(ctx, CP["pot_ln"], 2.4)


def paint_c_desk(ctx):
    rrect(ctx, 30, 462, 660, 488, 6)  # top surface
    fill(ctx, CP["desk_top2"], True)
    stroke(ctx, CP["desk_ln"], 2.6)
    ctx.rectangle(44, 486, 602, 260)  # front panel
    fill(ctx, CP["desk_front"], True)
    stroke(ctx, CP["desk_ln"], 2.6)
    ctx.rectangle(30, 482, 630, 10)
    fill(ctx, CP["desk_top"], True)
    stroke(ctx, CP["desk_ln"], 2.2)
    for (x0, x1) in ((70, 330), (360, 620)):  # drawers with round knobs
        rrect(ctx, x0, 512, x1, 590, 6)
        stroke(ctx, CP["desk_ln"], 2.4)
        ellipse(ctx, (x0 + x1) / 2, 551, 9, 9)
        fill(ctx, CP["knob"], True)
        stroke(ctx, CP["desk_ln"], 1.6)
        rrect(ctx, x0, 612, x1, 700, 6)
        stroke(ctx, CP["desk_ln"], 2.4)
        ellipse(ctx, (x0 + x1) / 2, 652, 9, 9)
        fill(ctx, CP["knob"], True)
        stroke(ctx, CP["desk_ln"], 1.6)
    # cone desk lamp on the left end
    lx, ly = 92, 462
    ellipse(ctx, lx, ly - 6, 34, 9)
    fill(ctx, CP["lamp"], True)
    stroke(ctx, CP["lamp_ln"], 2.2)
    poly(ctx, [(lx, ly - 8), (lx + 26, ly - 120)], closed=False)
    stroke(ctx, CP["lamp_ln"], 5)
    poly(ctx, [(lx - 6, ly - 150), (lx + 54, ly - 136), (lx + 60, ly - 96), (lx - 42, ly - 104)])
    fill(ctx, CP["lamp"], True)
    stroke(ctx, CP["lamp_ln"], 2.4)
    ellipse(ctx, lx + 9, ly - 100, 50, 9, .08)
    fill(ctx, hexc("#fff3c4"), True)
    stroke(ctx, CP["lamp_ln"], 2.2)


# ================================================================================================== glass overlays
def smudge(ctx, x, y, s=1.0, alpha=1.0):
    """The smudge: an irregular grey-brown blob with a darker core and two flecks.  Flat, no outline."""
    rng = np.random.default_rng(5)
    pts = []
    for j in range(14):
        a = j / 14 * 2 * math.pi
        r = (1 + rng.uniform(-.22, .22)) * s
        pts.append((x + 21 * r * math.cos(a), y + 15 * r * math.sin(a)))
    cr_path(ctx, pts, closed=True, tension=.9)
    fill(ctx, CP["smudge"][:3] + (alpha * .92,))
    ellipse(ctx, x - 3 * s, y + 1 * s, 10 * s, 7 * s, .3)
    fill(ctx, CP["smudge2"][:3] + (alpha * .9,))
    for dx, dy, r in ((26, -12, 3.6), (-24, 13, 2.8), (30, 9, 2.2)):
        ellipse(ctx, x + dx * s, y + dy * s, r * s, r * s)
    fill(ctx, CP["smudge"][:3] + (alpha * .9,))


def fog(ctx, x, y, r):
    rng = np.random.default_rng(8)
    pts = [(x + r * (1 + rng.uniform(-.12, .12)) * math.cos(a), y + r * .85 * (1 + rng.uniform(-.12, .12)) * math.sin(a))
           for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    cr_path(ctx, pts, closed=True, tension=.9)
    fill(ctx, hexc("#ffffff", .55))


def foam_on_pane(ctx, around, keep_clear=60):
    """Suds splatted on the glass: a few big lumpy patches with short drips, leaving the smudge in a clear spot."""
    rng = np.random.default_rng(14)
    x0, y0, x1, y1 = around
    centres = [(x0 + 40, y0 + 50), (x1 - 46, y0 + 60), (x0 + 150, y0 + 150), (x1 - 36, y1 - 70), (x0 + 50, y1 - 46)]
    blobs, drips = [], []
    for k, (cx, cy) in enumerate(centres):
        for j in range(7):
            a = rng.uniform(0, 2 * math.pi)
            d = rng.uniform(0, 22)
            blobs.append((cx + d * math.cos(a), cy + d * math.sin(a) * .7, rng.uniform(12, 22)))
        for j in range(rng.integers(1, 3)):
            dx = rng.uniform(-18, 18)
            drips.append((cx + dx, cy + 10, rng.uniform(16, 34), rng.uniform(5, 7)))

    def shapes():
        for bx, by, r in blobs:
            ellipse(ctx, bx, by, r, r * .9)
        for dx, dy, L, w in drips:
            rrect(ctx, dx - w, dy, dx + w, dy + 14 + L, w)
    shapes()
    stroke(ctx, INK, 5.4)
    shapes()
    fill(ctx, WHITE)
    for bx, by, r in blobs[::3]:  # a few bubble glints
        ctx.new_sub_path()
        ctx.arc(bx - r * .2, by - r * .2, r * .45, math.pi * 1.1, math.pi * 1.5)
        stroke(ctx, hexc("#a8dcef"), 2.4)


def face_print(ctx, x, y, s=1.0):
    """Squeaky's own face, printed on the glass: head outline, two big eye rings, a squashed mouth, the hat brim."""
    c = CP["print_"]
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(s, s)
    SQK.superellipse(ctx, 0, 0, 150, 116, 2.35, 1.0)
    fill(ctx, c[:3] + (.18,))
    SQK.superellipse(ctx, 0, 0, 150, 116, 2.35, 1.0)
    stroke(ctx, c[:3] + (.55,), 7)
    for sx in (-1, 1):
        ellipse(ctx, sx * 60, 22, 52, 58)
        stroke(ctx, c[:3] + (.6,), 9)
        ellipse(ctx, sx * 60, 30, 22, 24)
        fill(ctx, c[:3] + (.35,))
    cr_path(ctx, [(-30, 92), (0, 100), (30, 90)], tension=1)
    stroke(ctx, c[:3] + (.55,), 7)
    cr_path(ctx, [(-176, -50), (0, -70), (176, -50)], tension=1)
    stroke(ctx, c[:3] + (.5,), 8)
    for sx in (-1, 1):  # smear streaks where he slid off
        poly(ctx, [(sx * 120, 70), (sx * 132, 150)], closed=False)
        stroke(ctx, c[:3] + (.35,), 8)
    ctx.restore()


def glass_c(ctx):
    x0, y0, x1, y1 = WIN_C
    ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
    fill(ctx, hexc("#e8fbff", .1))
    ctx.save()
    ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
    ctx.clip()
    for (a, b, w) in ((x0 + 40, y0 + 160, 70), (x0 + 130, y0 + 160, 26), (x1 - 160, y1, 60), (x1 - 80, y1, 22)):
        poly(ctx, [(a, b), (a + 150, b - 150), (a + 150 + w, b - 150), (a + w, b)])
    fill(ctx, hexc("#ffffff", .3))
    ctx.restore()


# ================================================================================================== poses
def P_(**kw):
    d = dict(SQ)
    d.update(kw)
    return d


def F_(**kw):
    d = dict(FIG)
    d.update(kw)
    return d


REST = P_()
WHISTLE = P_(mouth="whistle", look=(.5, -.4), turn=.3)
PROUD = P_(lean=-.02, sq=1.03, head_dx=-4, head_dy=-6, head_tilt=-.04, turn=0.0, look=(0.0, .05), mouth="smile", happy=.45,
           grip=(118.0, -134.0), elbowR=(96.0, -84.0), sq_head=(198.0, -478.0), blade=0.0)
DETERMINED = P_(lean=.06, head_dx=14, head_tilt=.05, turn=.7, look=(1.0, -.05), lid=.32, slant=13, mouth="set",
                grip=(118.0, -134.0), elbowR=(96.0, -84.0), sq_head=(198.0, -478.0), blade=0.0)
ANTIC = P_(lean=-.07, sq=.93, head_dx=-10, head_dy=6, head_tilt=-.05, turn=.55, look=(1.0, -.6), lid=.3, slant=12, mouth="set",
           grip=(150.0, -176.0), elbowR=(108.0, -158.0), sq_head=(410.0, -400.0), blade=-.04, legL=.35, legR=-.25)
SHOCK = P_(lean=.08, head_dx=10, head_tilt=.06, turn=.6, look=(1.0, -.1), pupil=.42, mouth="o", mopen=.4,
           grip=(172.0, -84.0), elbowR=(116.0, -72.0), sq_head=(380.0, -196.0), blade=.04, legL=-.3, legR=.45)
HUFF = P_(lean=.13, head_dx=24, head_tilt=.08, turn=.8, look=(1.0, 0.0), lid=.26, slant=11, mouth="puff", cheeks=1.0,
          grip=(126.0, -90.0), elbowR=(92.0, -66.0), sq_head=(300.0, 40.0), blade=1.0)
FURY = P_(lean=-.02, head_dy=-4, turn=.55, look=(1.0, 0.0), lid=.36, slant=16, mouth="teeth", red=1.0, steam=True,
          handL=(-124.0, -100.0), grip=(126.0, -90.0), elbowR=(92.0, -66.0), sq_head=(300.0, 40.0), blade=1.0)
HEAVE = P_(lean=-.08, head_dx=-10, turn=.6, look=(1.0, -.2), lid=.22, slant=12, mouth="grin", hang_bucket=False, prop="bucket_up",
           bucket_at=(236.0, -150.0), elbowR=(150.0, -200.0), handL=(196.0, -96.0), elbowL=(60.0, -80.0))
SOGGY = P_(hat="bucket", foam=True, hang_bucket=False, prop="none", turn=.1, look=(.15, .1), lid=.5, slant=-3, mouth="flat",
           grip=(58.0, -44.0), elbowR=(80.0, -84.0), handL=(-58.0, -44.0), elbowL=(-80.0, -84.0), head_dy=4, sq=.97)
POLISH = P_(prop="polisher", polisher_at=(214.0, -150.0), turn=.6, look=(1.0, .1), pupil=.7, mouth="grin", lid=0.0,
            elbowR=(132.0, -178.0), handL=(146.0, -112.0), elbowL=(50.0, -84.0), lean=.04)
DIZZY = P_(prop="polisher_dangle", coil=True, sway=.08, lean=-.1, head_tilt=.16, head_dx=-8, hat_rot=.32, hat_dx=26, eyes="spiral",
           mouth="sheepish", stars=True, turn=0.0, grip=(96.0, -70.0), elbowR=(84.0, -96.0), legL=.5, legR=-.2)


def proto_in_c(**kw):
    """Squeaky outside Mrs. Fig's window (mirrored so his squeegee reaches to the smudge on screen-left)."""
    d = P_(mirror=True)
    d.update(kw)
    return d


# ================================================================================================== keys
def ext_b_frame(p, overlays=(), camera=CAM_B, squeaky_origin=ORIGIN_B):
    fr = Frame(camera)
    paint_b(fr.layer(True))
    gl = fr.layer(False)
    for fn in overlays:
        if fn[0] == "behind":
            fn[1](gl)
    SQK.squeaky(fr.layer(False), camera, p, squeaky_origin, 1.0)
    fx = fr.layer(False)
    for fn in overlays:
        if fn[0] == "front":
            fn[1](fx)
    return fr


def _wide(p, oy):
    fr = Frame(CAM_A)
    paint_a(fr.layer(True))
    p = dict(p, rope_top=(44.0 - oy) / .3)
    SQK.squeaky(fr.layer(False), CAM_A, p, (600.0, oy), .3, lw=.5)
    return fr


def build_keys():
    """Return {name: Frame} for every key frame of the short."""
    K = {}
    sm = ("behind", lambda c: smudge(c, *SMUDGE_B))
    sm_front = ("front", lambda c: smudge(c, *SMUDGE_B))
    # s01: wide establishing -- he lowers one floor, whistling, and sets his squeegee to the glass
    K["s01_a"] = _wide(P_(**{**PROUD, "mouth": "whistle", "happy": 0.0, "look": (.4, -.5)}), 250.0)
    K["s01_b"] = _wide(P_(**{**REST, "mouth": "smile", "sq_head": (390.0, -300.0)}), 330.0)
    # s02: medium -- last pane, 'Spotless!' (the smudge sits right there behind him)
    K["s02_a"] = ext_b_frame(P_(**{**REST, "mouth": "whistle", "look": (.6, -.1), "turn": .4}), [sm])
    K["s02_b"] = ext_b_frame(PROUD, [sm])
    # s03: close -- he notices
    K["s03_a"] = ext_b_frame(PROUD, [sm], CAM_D)
    K["s03_b"] = ext_b_frame(DETERMINED, [sm], CAM_D)
    # s04: medium -- the squeegee attack, twice; nothing
    K["s04_a"] = ext_b_frame(ANTIC, [sm])
    K["s04_b"] = ext_b_frame(SHOCK, [sm])
    # s05: close -- huff and buff; fury
    K["s05_a"] = ext_b_frame(HUFF, [sm, ("front", lambda c: fog(c, SMUDGE_B[0] + 10, SMUDGE_B[1], 70))], CAM_D)
    K["s05_b"] = ext_b_frame(FURY, [sm], CAM_D)
    # s06: medium -- the big bucket; the wind sends it all back
    K["s06_a"] = ext_b_frame(HEAVE, [sm])
    K["s06_b"] = ext_b_frame(SOGGY, [sm, ("behind", lambda c: foam_on_pane(c, (800, 80, 1010, 350), 46))])
    # s07: medium -- the Polish-o-Matic; spin; dizzy
    K["s07_a"] = ext_b_frame(POLISH, [sm])
    K["s07_b"] = ext_b_frame(DIZZY, [sm])
    # s08-s10: inside Mrs. Fig's office
    K["s08_a"] = interior(proto_in_c(lean=.05, turn=.6, look=(1.0, -.3), lid=.3, slant=13, mouth="teeth",
                                     grip=(140.0, -160.0), elbowR=(104.0, -150.0), sq_head=(414.0, -262.0), blade=-.15),
                          F_(look=(.7, -.6), lid=.15, mouth="flat", hold_r="cup", handR=(66.0, -40.0), elbowR=(96.0, -70.0)),
                          smudge_on=True)
    K["s08_b"] = interior(proto_in_c(lean=.02, turn=.4, look=(1.0, .1), pupil=.4, mouth="jaw",
                                     grip=(140.0, -160.0), elbowR=(104.0, -150.0), sq_head=(414.0, -262.0), blade=-.15),
                          F_(lean=.3, look=(.5, -.3), closed_happy=1.0, mouth="smug", hold_r="hanky", hold_l=None,
                             handR=None, elbowR=None, cup_on_desk=True), smudge_on=False, sparkle_at=SMUDGE_C)
    K["s09_b"] = interior(proto_in_c(turn=.15, look=(.6, .2), lid=.42, slant=0, mouth="sheepish", sweat=True),
                          F_(look=(.2, 0), closed_happy=1.0, mouth="smile", hold_r="cup_sip", handR=(14.0, -150.0), elbowR=(70.0, -100.0),
                             hold_l="hanky", handL=(-70.0, -30.0)), smudge_on=False)
    K["s10_b"] = interior(proto_in_c(turn=.35, look=(1.0, .15), lid=.1, mouth="grin", sweat=True, head_dy=4, lean=.04,
                                     grip=(126.0, -90.0), elbowR=(92.0, -66.0), sq_head=(300.0, 40.0), blade=1.0),
                          F_(look=(.9, -.1), lid=.48, slant=0, mouth="flat", hold_r="hanky", handR=(178.0, -212.0), elbowR=(150.0, -118.0),
                             cup_on_desk=True), smudge_on=False, print_at=(712.0, 226.0))
    return K


ORIGIN_C = (930.0, 452.0)
FIG_O = (330.0, 470.0)


def interior(sp, fp, smudge_on=True, sparkle_at=None, print_at=None):
    fr = Frame(CAM_C)
    out = fr.layer(True)
    paint_c_outside(out)
    SQK.squeaky(fr.layer(False), CAM_C, sp, ORIGIN_C, .72, lw=.78)
    gl = fr.layer(False)
    glass_c(gl)
    if smudge_on:
        smudge(gl, *SMUDGE_C, s=1.3)
    if print_at:
        face_print(gl, *print_at, s=.7)
    if sparkle_at:
        sparkle(gl, sparkle_at[0] + 10, sparkle_at[1] - 6, .9, .3)
    paint_c_wall(fr.layer(True))
    fp = dict(fp)
    if fp.get("handR") is None:  # reach for the smudge: solve the hand position in her local space
        lean, s = fp["lean"], .95
        wx, wy = SMUDGE_C[0] - FIG_O[0], SMUDGE_C[1] + 14 - FIG_O[1]
        c, sn = math.cos(-lean), math.sin(-lean)
        lx, ly = (wx * c - wy * sn) / s, (wx * sn + wy * c) / s
        fp["handR"] = (lx - 6, ly + 26)
        fp["elbowR"] = (lx * .55 + 30, ly * .55 - 30)
        fp["handL"] = (-40.0, -12.0)
        fp["elbowL"] = (-80.0, -50.0)
    SQK.mrs_fig(fr.layer(False), CAM_C, fp, FIG_O, .95, part="body")
    dk = fr.layer(True)
    paint_c_desk(dk)
    arms = fr.layer(False)
    if fp.get("cup_on_desk"):
        cel = SQK.Cel(arms, CAM_C)
        arms.set_matrix(CAM_C)
        SQK.teacup(cel, 520, 444, .95)
        cel.draw()
    SQK.mrs_fig(arms, CAM_C, fp, FIG_O, .95, part="arms")
    return fr


def render_all(out_dir, only=None):
    import os
    os.makedirs(out_dir, exist_ok=True)
    K = build_keys()
    for name, fr in K.items():
        if only and name not in only:
            continue
        img = fr.render()
        assert img.shape == (704, 1280, 3)
        cv2.imwrite(os.path.join(out_dir, f"{name}.png"), img[:, :, ::-1])
    return list(K)
