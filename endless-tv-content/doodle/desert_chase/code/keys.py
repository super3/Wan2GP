"""Every start / end key frame of "Clean Sweep", drawn by code.  keys.py OUT_DIR [names...] writes OUT_DIR/<name>.png (1280x704)."""
import math
import os
import sys

import cairo
import cv2
import numpy as np

import cast
import desert as D
import scene
import world

P0 = D.PROUD
B0 = cast.BRAMBLE


def pip(**kw):
    p = dict(P0)
    p.update(kw)
    return p


def bram(**kw):
    p = dict(B0)
    p.update(kw)
    return p


ROLL_FEET = ((-120, -236), (-60, -272), (70, -270), (140, -226))


def pip_layer(p, treat=None, amount=1.0, mud=None, haze=0.0, seed=0, ball=None):
    def draw(ctx):
        if ball is not None:
            cast.pip_ball(ctx, **ball)
        else:
            D.pip(ctx, p)
        if mud:
            ctx.set_operator(cairo.OPERATOR_ATOP)
            cast.splats(ctx, mud, seed=seed + 3)
            ctx.set_operator(cairo.OPERATOR_OVER)
    return dict(draw=draw, treat=treat, amount=amount, seed=seed, haze=haze)


def bram_layer(b, treat=None, amount=1.0, haze=0.0, seed=1, fluff=None):
    def draw(ctx):
        if fluff is not None:
            cast.bramble_fluff(ctx, **fluff)
        else:
            cast.bramble(ctx, b)
    return dict(draw=draw, treat=treat, amount=amount, seed=seed, haze=haze)


def fx_layer(fn, haze=0.0):
    return dict(draw=fn, haze=haze)


def wx(cam, x):
    """world x on the ledge layer for a screen x in this camera (zoom 1)."""
    return x + world.CAMS[cam]["x0"]


# ------------------------------------------------------------------------------------------------ shots
SWEEP_HOME = pip(two_hand=1.0, hn_x=-52, hn_y=-150, broom_rot=.5, crouch=.3, lean=-.08, sy=.97, head_rot=-.06, lid=.42, brow=-.15,
                 pup_x=-.4, pup_y=.6, smile=.6, frown=0)


def s01_a():
    def fx(ctx):
        rng = np.random.default_rng(3)
        for k in range(14):   # a last little pile of grit and pebbles being swept
            x, y = 440 + rng.uniform(-30, 26), 650 + rng.uniform(-6, 8)
            D.fillp(ctx, D.circ(x, y, rng.uniform(1.5, 3.5), None, 10), D.BG["peb"])
        D.dust(ctx, 452, 640, 13, 5, n=6)
    return scene.compose("A", layers=[pip_layer(SWEEP_HOME), fx_layer(fx)])


def s01_b():
    return scene.compose("A", layers=[pip_layer(pip())])


DEADPAN = dict(lid=.55, pup_x=-.05, pup_y=.1, smile=0, frown=.15, brow=-.05)


def s02_a():
    b = bram(x=930, gy=652, sc=.95, rot=-.05, dy=-6, feet=((-104, -16), (-40, 0), (70, 0), (130, -18)), eyes="happy", mouth="open", open_=.5,
             ear_rot=.1)
    return scene.compose("B", layers=[bram_layer(b)])


def s02_b():
    b = bram(x=500, gy=650, sc=.95, roll=True, head_rot=.8, feet=ROLL_FEET, eyes="happy", mouth="open", open_=.85, shadow=False)

    def fx(ctx):
        for k, (x, y, s) in enumerate(((640, 600, 34), (700, 540, 40), (590, 470, 30), (330, 640, 30), (760, 470, 30), (250, 600, 26))):
            D.dust(ctx, x, y, s, 20 + k, n=8)
    return scene.compose("B", layers=[fx_layer(lambda ctx: D.dust(ctx, 560, 636, 44, 9, n=10)), bram_layer(b), fx_layer(fx)])


def s03_b():
    p = pip(**DEADPAN, broom_rot=.12, hn_y=-170, ear_far=-.5, ear_near=.55, head_rot=.05, lean=0.0)

    def fx(ctx):   # the last of the cloud drifting off left, a few motes settling
        for k, (x, y, s) in enumerate(((180, 470, 30), (90, 430, 38), (300, 520, 22))):
            D.dust(ctx, x, y, s, 40 + k, n=7, fade=.85)
    return scene.compose("A", layers=[pip_layer(p, "dust", 1.0, seed=2), fx_layer(fx)])


def s04_a():
    p = pip(**{**DEADPAN, "lid": .6, "pup_x": .15, "pup_y": .05}, broom_rot=.12, hn_y=-170, ear_far=-.5, ear_near=.55, head_rot=.05, lean=0.0)
    return scene.compose("C", layers=[pip_layer(p, "dust", 1.0, seed=2)])


def s04_b():
    p = pip(lid=.42, brow=-.85, brow_tilt=1.0, pup_x=.2, pup_y=.05, smile=0, frown=1.0, two_hand=1.0, hn_x=-30, hn_y=-250, broom_rot=2.75,
            head_rot=-.02, head_dy=4, lean=-.04, ear_far=-.7, ear_near=.75, sy=1.03)
    return scene.compose("C", layers=[pip_layer(p, "dust", 1.0, seed=2)])


# two-shot positions (camera D): Pip on his side of the pebble line, Bramble in his wallow
PX, BX = 330.0, 800.0


def s05_a():
    p = pip(x=PX, sc=.9, facing=-1.0, two_hand=1.0, hn_x=-66, hn_y=-150, broom_rot=1.05, crouch=.35, lean=-.13, sy=.97, head_rot=-.04,
            lid=.45, brow=-.6, brow_tilt=.8, pup_x=-.5, pup_y=.3, smile=0, frown=.9)
    b = bram(x=BX, gy=650, sc=.85, roll=True, head_rot=.8, feet=ROLL_FEET, eyes="happy", mouth="grin", shadow=False)

    def fx(ctx):   # a stream of swept dust flying toward the wallow
        for k in range(7):
            t = k / 6
            D.dust(ctx, 520 + 230 * t, 640 - 40 * math.sin(math.pi * t), 14 + 10 * t, 60 + k, n=6)
        D.speed_lines(ctx, 470, 700, (606, 622, 640), D.alpha(D.PC["dust_line"], .9), 2.0, seed=4)
    return scene.compose("D", layers=[bram_layer(b), pip_layer(p, "dust", .8, seed=2), fx_layer(fx)])


def s05_b():
    p = pip(x=PX, sc=.9, facing=-1.0, **{**DEADPAN, "lid": .62}, broom_rot=.1, hn_y=-172, ear_far=-.5, ear_near=.55, head_rot=.0, lean=0.0)
    b = bram(x=BX, gy=650, sc=.85, roll=True, head_rot=.9, feet=((-110, -250), (-70, -262), (80, -264), (130, -240)), eyes="happy",
             mouth="open", open_=.9, shadow=False)

    def heap(ctx):
        D.dust(ctx, BX + 10, 640, 56, 70, n=12, spread=(1.4, .3))

    def fx(ctx):
        for k, (x, y, s) in enumerate(((600, 560, 44), (560, 420, 34), (690, 470, 36), (200, 470, 30), (150, 560, 30))):
            D.dust(ctx, x, y, s, 80 + k, n=8, fade=.95)
    return scene.compose("D", layers=[fx_layer(heap), bram_layer(b), fx_layer(lambda ctx: D.dust(ctx, BX - 30, 600, 40, 71, n=9)),
                                      pip_layer(p, "dust", 1.35, seed=2), fx_layer(fx)])


P6X, B6X = 470.0, 840.0
MUD_PIP = [(-30, -150, 9), (-40, -205, 7), (20, -120, 8), (-20, -290, 6), (50, -200, 7), (-60, -100, 6), (10, -40, 6), (-75, -260, 5)]


def mud_pts(p, pts):
    M = D.body_matrix(p)
    return [(*D.ap1(M, x, y), r * p.get("sc", 1)) for x, y, r in pts]


def s06_a():
    p = pip(x=P6X, sc=.9, facing=-1.0, prop="canteen", hn_x=-118, hn_y=-318, hf_x=-84, hf_y=-300, canteen_rot=-2.3, spill=1.0, lean=-.1,
            head_rot=.22, lid=.3, brow=.25, pup_x=-.4, pup_y=-.3, smile=.7, frown=0, two_hand=1.0, ear_far=-.3, ear_near=.35)
    b = bram(x=B6X, gy=648, sc=.85, eyes="happy", mouth="grin", grin=1.0, head_rot=-.18, ear_rot=.1)

    def water(ctx):   # the stream from the spout onto Bramble's head
        P = np.array([(612, 330), (622, 380), (628, 440), (630, 500)], float)
        R = D.ribbon(D.crs(P, False, 10), 7, 12)
        D.fillp(ctx, R, D.hexc("#a8dcf2"))
        D.ink(ctx, R, 1.4, D.hexc("#3f7fa6"), closed=True)
        for k, (x, y) in enumerate(((610, 512), (650, 506), (596, 528), (664, 530))):
            D.fillp(ctx, D.circ(x, y, 4, 6, 12), D.hexc("#a8dcf2"))
    return scene.compose("D", layers=[bram_layer(b), pip_layer(p, "dust", .8, seed=2), fx_layer(water)])


def s06_b():
    p = pip(x=P6X, sc=.9, facing=-1.0, prop="canteen", hn_x=-46, hn_y=-182, hf_x=-30, hf_y=-170, canteen_rot=.2, **{**DEADPAN, "lid": .58},
            two_hand=1.0, ear_far=-.5, ear_near=.55)
    b = bram(x=B6X, gy=648, sc=.85, eyes="happy", mouth="open", open_=.7, head_rot=-.1, ear_rot=.1)

    def ground(ctx):
        cast.splats(ctx, [(600, 690, 9), (680, 664, 7), (560, 640, 6), (980, 676, 8), (1040, 640, 6)], seed=5)
    return scene.compose("D", layers=[fx_layer(ground), bram_layer(b, "mud", 1.0), pip_layer(p, "dust", .8, mud=mud_pts(p, MUD_PIP), seed=2)])


def s07_b():
    p = pip(x=P6X, sc=.9, facing=-1.0, prop="none", hn_x=-30, hn_y=-300, hf_x=-70, hf_y=-290, sx=.92, sy=1.1, eye_s=1.4, lid=0, pup_x=-.3,
            pup_y=-.6, brow=1.2, mouth=.8, smile=0, head_rot=.3, head_dy=-12, ear_far=-.05, ear_near=.06, ear_len=1.1, tail_curl=1.0, two_hand=1.0)
    b = bram(x=B6X, gy=648, sc=.85, eyes="open", lid=0, pup_x=-.1, pup_y=-.6, mouth="open", open_=.9, head_rot=.3, ear_rot=-.2, brow=.8)

    def ground(ctx):
        cast.splats(ctx, [(600, 690, 9), (680, 664, 7), (560, 640, 6), (980, 676, 8), (1040, 640, 6)], seed=5)
        cv = D.ap1(D.Tm(P6X + 40, 690), 0, 0)
        M = D.Tm(cv[0], cv[1]) @ D.Rm(1.4) @ D.Sm(.8)
        D.canteen(ctx, M)
    return scene.compose("D", bg=dict(storm=1.0), layers=[fx_layer(ground), bram_layer(b, "mud", 1.0),
                                                       pip_layer(p, "dust", .8, mud=mud_pts(p, MUD_PIP), seed=2)])


def s08_b():
    b = bram(x=B6X, gy=648, sc=.85, eyes="shut", mouth="grin", grin=1.0, head_rot=-.25, head_dy=10, ear_rot=.7, rot=.04)

    def wind(ctx):
        rng = np.random.default_rng(9)
        for k in range(26):
            y = rng.uniform(120, 700)
            x = rng.uniform(-100, 1100)
            D.drybrush(ctx, x + 300, y, x, y + rng.normal(0, 4), rng.uniform(2, 5), D.alpha(D.hexc("#f0d2a0"), .55), rng, broken=.4)
        for k in range(8):
            D.dust(ctx, rng.uniform(0, 1280), rng.uniform(200, 600), rng.uniform(20, 40), 300 + k, n=7, fade=.6)
    return scene.compose("D", bg=dict(storm=1.0, engulf=1.0), layers=[
        bram_layer(b, "mud", .6, haze=.45), pip_layer(None, haze=.45, ball=dict(x=P6X, gy=650, sc=.9, facing=-1.0)), fx_layer(wind, haze=.2)])


# the clean wedge downwind (left / toward camera) of Bramble, in camera-D screen space
WEDGE = [(B6X + 40, 606), (B6X - 120, 596), (P6X - 150, 628), (P6X - 400, 730), (B6X - 20, 730)]


def s09_a():
    return scene.compose("D", bg=dict(after=WEDGE), layers=[
        fx_layer(lambda ctx: lying_broom(ctx, P6X - 150, 690)),
        bram_layer(None, fluff=dict(x=B6X + 20, gy=648, sc=.82, facing=1.0)),
        pip_layer(None, ball=dict(x=P6X, gy=650, sc=.9, facing=-1.0, peek=0.0))])


def lying_broom(ctx, x, y, rot=-1.62):
    M = D.Tm(x, y) @ D.Rm(rot) @ D.Sm(.9)
    D.broom(ctx, M)


def s09_b():
    p = pip(x=P6X, sc=.9, facing=-1.0, eye_s=1.2, lid=0, pup_x=-.2, pup_y=.35, brow=.9, smile=1.0, frown=0, head_rot=-.02, head_dy=-4,
            ear_far=-.1, ear_near=.12, ear_len=1.0, hn_x=126, hn_y=-178, lean=-.02)
    return scene.compose("D", bg=dict(after=WEDGE), layers=[
        bram_layer(None, fluff=dict(x=B6X + 20, gy=648, sc=.82, facing=1.0)), pip_layer(p)])


def s10_b():
    p = pip(x=P6X + 60, sc=.9, facing=-1.0, two_hand=1.0, hn_x=-84, hn_y=-236, broom_rot=1.75, lean=-.1, head_rot=.0, lid=.4, brow=.15,
            pup_x=-.5, pup_y=.25, smile=1.0, frown=0, sy=1.02)
    b = bram(x=B6X + 40, gy=648, sc=.85, facing=-1.0, eyes="happy", mouth="open", open_=.6, head_rot=.12, ear_rot=.25,
             feet=((-82, 0), (-52, 0), (86, 0), (150, -60)))

    def heap(ctx):
        D.dust(ctx, B6X + 10, 646, 40, 90, n=10, spread=(1.8, .25))

    def clumps(ctx):
        for k, (x, y, s) in enumerate(((B6X - 30, 470, 20), (B6X - 70, 500, 16), (B6X + 30, 452, 15))):
            D.dust(ctx, x, y, s, 95 + k, n=5)
    return scene.compose("D", bg=dict(after=WEDGE), layers=[fx_layer(heap), bram_layer(b, "dust", .35), fx_layer(clumps), pip_layer(p)])


P11X, B11X = 500.0, 905.0


def s11_a():
    p = pip(x=P11X)
    b = bram(x=B11X, gy=650, sc=.85, eyes="open", lid=.3, pup_x=-.4, pup_y=.2, mouth="grin", grin=.8, scarf=True, head_rot=.08)
    return scene.compose("A", bg=dict(time="sunset"), layers=[bram_layer(b), pip_layer(p)])


def s11_b():
    p = pip(x=P11X, **{**DEADPAN, "lid": .62}, broom_rot=.12, hn_y=-170, ear_far=-.55, ear_near=.6, head_rot=.0, lean=0.0)
    b = bram(x=B11X, gy=650, sc=.85, eyes="open", lid=.15, pup_x=-.6, pup_y=.1, mouth="sheepish", brow=.7, brow_tilt=-.6, scarf=True,
             head_rot=.0)

    def fx(ctx):
        for k, (x, y, s) in enumerate(((240, 380, 40), (150, 470, 46), (90, 340, 34))):
            D.dust(ctx, x, y, s, 120 + k, n=8, fade=.9)
    return scene.compose("A", bg=dict(time="sunset"), layers=[bram_layer(b), pip_layer(p, "dust", 1.5, seed=7), fx_layer(fx)])


KEYS = dict(s01_a=s01_a, s01_b=s01_b, s02_a=s02_a, s02_b=s02_b, s03_b=s03_b, s04_a=s04_a, s04_b=s04_b, s05_a=s05_a, s05_b=s05_b,
            s06_a=s06_a, s06_b=s06_b, s07_b=s07_b, s08_b=s08_b, s09_a=s09_a, s09_b=s09_b, s10_b=s10_b, s11_a=s11_a, s11_b=s11_b)


def one(args):
    name, out = args
    img = scene.key(KEYS[name]())
    assert img.shape == (704, 1280, 3)
    cv2.imwrite(os.path.join(out, f"{name}.png"), img[:, :, ::-1])
    return name


if __name__ == "__main__":
    from multiprocessing import Pool
    out = sys.argv[1]
    names = sys.argv[2:] or list(KEYS)
    os.makedirs(out, exist_ok=True)
    with Pool(2) as pool:
        for n in pool.imap_unordered(one, [(n, out) for n in names]):
            print(n, end=" ", flush=True)
    print()
