"""Key frames for "The Littlest Crossing": every start/end frame, defined as scenes over the shared drawing code."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bigsby as G  # noqa: E402
from bigsby import bpose, tpose  # noqa: E402

OUT = "/home/user/Wan2GP/outputs/shorts/retro_exaggerated"
KEYS = os.path.join(OUT, "keys")

# cameras: (zoom, world centre x, world centre y)
WIDE = (1.0, 640.0, 360.0)
LOW = (2.0, 650.0, 560.0)          # down at Tilly's level, his belly as a ceiling
MED = (1.7, 650.0, 300.0)          # Bertram chest-up
X_WIDE = (1.0, 640.0, 360.0)
X_ROAD = (1.7, 650.0, 468.0)       # mid-crossing, on Bertram and the hydrant
X_CURB = (1.25, 480.0, 400.0)      # the left curb, Bertram + sign + Tilly

BS = .72                            # Bertram's scale on the crossing set
TS = .7                             # Tilly's scale on the corner set
TSX = .5                            # Tilly's scale on the crossing set
YW = G.WALK_Y


def tilly_on_disc(P, **kw):
    d = G.bertram_disc(P)
    return tpose(X=d[0] - 8 * TSX, Y=d[1] - 43 * P["S"] + 30 * TSX, S=TSX, sit=True, **kw)


def scenes():
    K = {}
    # ---- s01: the corner.  Bertram brags; tiny Tilly waits at his feet.
    tilly0 = tpose(X=640, Y=655, S=TS, face=-1, arms="down", px=-2, py=-3)
    K["s01_a"] = dict(bg="corner", cam=WIDE, items=[("bertram", bpose()), ("tilly", tilly0)])
    b_brag = bpose(near="raise", sx=1.05, sy=1.03, lid=.6, hr=-.07, hy=-6, mouth="smile")
    tilly_up = tpose(X=640, Y=655, S=TS, face=-1, arms="raise", px=-2, py=-3)
    K["s01_b"] = dict(bg="corner", cam=WIDE, items=[("bertram", b_brag), ("tilly", tilly_up)])
    # ---- s02: down at Tilly's level
    K["s02_a"] = dict(bg="corner", cam=LOW, items=[("bertram", b_brag), ("tilly", tilly_up)])
    K["s02_b"] = dict(bg="corner", cam=LOW, items=[("bertram", b_brag),
                                                   ("tilly", tpose(X=640, Y=655, S=TS, face=-1, arms="tap", lid=.45, px=-3, py=-1))])
    # ---- s03: chest-up.  He hears her but cannot see past his own chest
    K["s03_a"] = dict(bg="corner", cam=MED, items=[("bertram", bpose(far="ear", signx=292, px=-7, brow=-6, mouth="mm", lid=.1)),
                                                   ("tilly", tilly0)])
    K["s03_b"] = dict(bg="corner", cam=MED, items=[("bertram", bpose(lean=.13, hy=26, hx=6, hr=.32, eyes=1.25, py=7, px=0, brow=-10,
                                                                     mouth="wavy", lid=0)), ("tilly", tilly0)])
    # ---- the crossing set (one-point perspective street)
    hyd_l = ("hydrant", dict(X=452, Y=YW, S=BS))
    hyd_r = ("hydrant", dict(X=1046, Y=YW, S=BS))
    t_left = tpose(X=398, Y=YW, S=TSX, face=1, arms="down", px=2)
    b_d1 = bpose(X=300, Y=YW, S=BS, lid=.3, hr=-.05, px=0)
    K["s04_a"] = dict(bg="cross", cam=X_WIDE, items=[("bertram", b_d1), hyd_l, ("tilly", t_left)])
    b_d2 = bpose(X=930, Y=YW, S=BS, sign=1.0, near="raise", sx=1.05, lid=.55, hr=-.08, mouth="smile")
    K["s04_b"] = dict(bg="cross", cam=X_WIDE, items=[("bertram", b_d2), hyd_l, ("tilly", t_left)])
    # s05 continues from s04_b
    b_d3 = bpose(X=930, Y=YW, S=BS, far="ear", signx=292, px=-7, brow=-8, mouth="mm", lid=.05, hr=.05)
    K["s05_b"] = dict(bg="cross", cam=X_WIDE, items=[("bertram", b_d3), hyd_l, ("tilly", tpose(X=398, Y=YW, S=TSX, face=1, arms="facepalm"))])
    # ---- s06: attempt 2, he gropes blindly and scoops up the hydrant instead
    b_d4 = bpose(X=300, Y=YW, S=BS, near="sign", far="grope", grope=(214.0, -118.0), lean=.12, hr=.2, hy=14, py=6, px=2, lid=0,
                 mouth="mm", eyes=1.1)
    t_dead = tpose(X=392, Y=YW, S=TSX, face=1, arms="down", lid=.45, py=-3, px=-1)
    K["s06_a"] = dict(bg="cross", cam=X_WIDE, items=[("bertram", b_d4), hyd_l, ("tilly", t_dead)])
    b_d5 = bpose(X=640, Y=YW, S=BS, near="sign", far="cradle", hr=.1, py=6, px=1, lid=.25, mouth="smile")
    t_dead2 = tpose(X=392, Y=YW, S=TSX, face=1, arms="hips", lid=.45, px=2)
    K["s06_b"] = dict(bg="cross", cam=X_WIDE, items=[("bertram", b_d5), ("tilly", t_dead2)])
    # ---- s07: mid-road close-up: he coos at the hydrant; it squirts him
    K["s07_a"] = dict(bg="cross", cam=X_ROAD, items=[("bertram", b_d5), ("tilly", t_dead2)])
    b_wet = bpose(X=640, Y=YW, S=BS, near="sign", far="cradle", capoff=True, wet=True, squeeze=True, mouth="wavy", hr=-.06)
    noz = G.hydrant_nozzle(b_wet)
    face = G.bertram_head(b_wet) + np.array([62 * BS, 22 * BS])
    K["s07_b"] = dict(bg="cross", cam=X_ROAD, items=[("bertram", b_wet), ("tilly", t_dead2)], fx=[G.water_fx(noz, face, s=BS)])
    # ---- s08: back at the curb, soggy and glum; Tilly climbs his sign
    b_glum = bpose(X=300, Y=YW, S=BS, wet=True, near="hang", mouth="frown", lid=.55, px=-5, py=5, hr=.08, hy=4, brow=4)
    K["s08_a"] = dict(bg="cross", cam=X_CURB, items=[("bertram", b_glum), hyd_r,
                                                     ("tilly", tpose(X=472, Y=YW, S=TSX, face=1, arms="raise", px=3, py=-4))])
    K["s08_b"] = dict(bg="cross", cam=X_CURB, items=[("bertram", b_glum), hyd_r, ("tilly", tilly_on_disc(b_glum, face=-1, arms="down", px=-2, py=2))])
    # s09 continues: he lifts the sign to try again and finally sees her up there
    b_see = bpose(X=300, Y=YW, S=BS, wet=True, sign=1.0, near="raise", hr=-.22, hy=-6, py=-7, px=3, eyes=1.22, lid=0, mouth="grin", brow=-10)
    K["s09_b"] = dict(bg="cross", cam=X_CURB, items=[("bertram", b_see), hyd_r,
                                                     ("tilly", tilly_on_disc(b_see, face=-1, arms="wave", mouth="smile", py=3, px=-2))])
    # ---- s10: the triumphant crossing, Tilly riding the sign
    b_march = bpose(X=300, Y=YW, S=BS, sign=1.0, near="raise", sx=1.04, lid=.35, hr=-.08, mouth="smile")
    K["s10_a"] = dict(bg="cross", cam=X_WIDE, items=[("bertram", b_march), hyd_r,
                                                     ("tilly", tilly_on_disc(b_march, face=1, arms="raise", mouth="smile"))])
    b_bow = bpose(X=868, Y=YW, S=BS, sx=1.06, sy=1.03, lid=.6, hr=-.08, mouth="smile", near="raise")
    K["s10_b"] = dict(bg="cross", cam=X_WIDE, items=[("bertram", b_bow), hyd_r,
                                                     ("tilly", tpose(X=1182, Y=YW + 8, S=TSX * 1.04, face=-1, arms="wave", mouth="smile"))])
    # ---- s11: the button.  Back on his corner, alone and proud... with five ducklings waiting at his feet
    ducks = [dict(X=594 + 46 * k, Y=656, S=1.04 + .05 * (k % 2), look=-2) for k in range(5)]
    K["s11_a"] = dict(bg="corner", cam=WIDE, items=[("bertram", bpose()), ("ducks", ducks)])
    ducks_up = [dict(d, wing=1) for d in ducks]
    K["s11_b"] = dict(bg="corner", cam=WIDE, items=[("bertram", bpose(near="raise", sx=1.06, sy=1.03, lid=.62, hr=-.08, hy=-6)),
                                                    ("ducks", ducks_up)])
    return K


def render_all(names=None, ss=3):
    os.makedirs(KEYS, exist_ok=True)
    K = scenes()
    for n, sc in K.items():
        if names and n not in names:
            continue
        img = G.key_png(sc, ss)
        assert img.shape == (704, 1280, 3)
        cv2.imwrite(os.path.join(KEYS, n + ".png"), img[:, :, ::-1])
    return K


def sheet(path, ss=2):
    K = scenes()
    ims = []
    for n, sc in K.items():
        im = G.key_png(sc, ss)
        im = cv2.resize(im, (426, 234), interpolation=cv2.INTER_AREA)
        cv2.putText(im, n, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 3)
        cv2.putText(im, n, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
        ims.append(im)
    while len(ims) % 4:
        ims.append(np.zeros_like(ims[0]))
    cv2.imwrite(path, np.vstack([np.hstack(ims[r * 4:(r + 1) * 4]) for r in range(len(ims) // 4)])[:, :, ::-1])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sheet":
        sheet(sys.argv[2])
    else:
        render_all(sys.argv[1:] or None)
