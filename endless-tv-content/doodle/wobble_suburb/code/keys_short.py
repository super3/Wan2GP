"""Every start/end key frame of the short "One Lousy Dandelion" (1280x704), drawn from the shared rig.

keys_short.py [OUTDIR] [shot ids...]  -> OUTDIR/keys/<id>_a.png, <id>_b.png and a contact sheet."""
import math
import os
import sys

import cv2
import numpy as np

import cast as CA
import chars as CH
import ink as K
import paint

OUT = "/home/user/Wan2GP/outputs/shorts/wobble_suburb"
W, H = K.W, K.H
WIDE = (640, 352, 1.0)
BEA_CAM = (205, 468, 1.9)
DD = dict(x=905, y=642, s=1.45)
BEA_WIDE = dict(x=150, y=612, s=.62)
WX, WY, WS = 470, 628, .9

_BG = {}


def bg(cam=WIDE, **kw):
    key = (cam, tuple(sorted(kw.items())))
    if key not in _BG:
        _BG[key] = paint.backyard(cam=cam, **kw)
    return _BG[key]


def wmul(s, z):
    return (0.9 / (s * z)) ** .75


class Frame:
    def __init__(self, cam=WIDE, boil=0):
        self.cam = cam
        K.set_boil(boil, .5)
        self.surf, self.ctx = K.canvas()
        cx, cy, z = cam
        self.ctx.translate(W / 2, H / 2)
        self.ctx.scale(z, z)
        self.ctx.translate(-cx, -cy)

    def wendell(self, **kw):
        p = CH.pose(**{**dict(x=WX, y=WY, s=WS), **kw})
        K.set_wmul(wmul(p["s"], self.cam[2]))
        CH.wendell(self.ctx, p)
        return p

    def bea(self, **kw):
        p = CA.bpose(**{**BEA_WIDE, **kw})
        K.set_wmul(wmul(p["s"] * 1.45, self.cam[2]))
        CA.bea(self.ctx, p)

    def dd(self, **kw):
        d = {**DD, **kw}
        K.set_wmul(wmul(d["s"] * .62, self.cam[2]))
        CH.dandelion(self.ctx, **d)

    def props(self, fn, s=1.0):
        K.set_wmul(wmul(s * .9, self.cam[2]))
        fn(self.ctx)

    def rgba(self):
        return paint.to_rgba(self.surf)


def crater_front_mask():
    """Screen mask (wide camera) of the lawn in front of the crater's near lip: painted ground occludes whoever is in the pit."""
    cx, cy, rx, ry = paint.CRATER
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = np.clip((xx - cx) / rx, -1, 1)
    edge = cy + ry * np.sqrt(1 - u ** 2) * .92
    m = ((yy > edge) & (np.abs(xx - cx) < rx)).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), 1.0)[..., None]


def comp(bgimg, cel, front=None):
    img = paint.over(bgimg, cel)
    if front is not None:
        img = img * (1 - front) + bgimg * front
    return np.clip(img, 0, 1)


def world(ctx, p):
    """From inside a Wendell prop callback (local coords) back to world coords."""
    ctx.scale(1 / (p["s"] * p["face"] * p["sx"]), 1 / (p["s"] * p["sy"]))
    ctx.translate(-p["x"], -p["y"])


def local_of(x, y, wx=WX, wy=WY, s=WS):
    return ((x - wx) / s, (y - wy) / s)


# ================================================================================================ shots
TAKE = dict(eyesize=1.5, eye=1.0, squint=0, browang=-.6, brow=1.5, hair_up=1.1, mouth="yell", jaw=.95, wide=.9, tongue=1.0)


def s01(which):
    f = Frame()
    f.dd()
    f.bea(hold="sip", mouth="sip", look=(.7, .1))
    if which == "a":
        f.wendell(x=430, lean=-.07, smile=1.0, eye=.9, look=(.2, .35), brow=.5, browang=-.2, mouth="closed")
    else:
        f.wendell(x=430, lean=-.16, hipx=-10, hipy=18, **TAKE, look=(.9, .4), handB=(220, -300), handBs="point", handBa=.32, bendB=1,
                  handF=(-160, -360), handFs="open", footF=(-60, -4), footB=(60, -2))
    return comp(bg(), f.rgba())


def s02(which):
    cam = (780, 452, 1.3)
    f = Frame(cam)
    f.dd()
    base = dict(x=700, y=640, lean=.32, tilt=-.1, hipy=112, hipx=-40, footF=(-10, 0), footB=(50, 0), look=(.95, .5))
    if which == "a":
        f.wendell(**base, handF=(40, -128), handB=(84, -132), handFs="fist", handBs="fist", bendF=-1, bendB=1, squint=.55,
                  browang=1.0, brow=-.2, mouth="frown")
    else:
        f.wendell(**base, handF=(70, -236), handB=(96, -240), handFs="fist", handBs="fist", bendF=-1, bendB=1, squint=.35,
                  browang=.9, brow=0, mouth="grit")
    return comp(bg(cam), f.rgba())


def s03(which):
    f = Frame()
    f.bea(hold="chest", look=(.8, .1), eye=.45)
    if which == "a":
        f.dd(sway=-2.2, stretch=1.25)
        hx, hy = 905 - 2.2 * 18 * 1.45, 642 - 80 * 1.45
        f.wendell(x=690, lean=-.42, hipx=-30, hipy=24, footF=(-40, 0), footB=(70, 0), handF=local_of(hx - 6, hy + 8, 690),
                  handB=local_of(hx + 8, hy + 4, 690), handFs="grip", handBs="grip", bendF=1, bendB=1, mouth="grit", squint=.7,
                  browang=1.1, brow=-.2, hair_up=.35, look=(.8, .2))
    else:
        f.dd()
        # flat on his back, head to the left, dazed, one petal in his raised fist
        f.wendell(x=850, y=600, rot=-math.pi / 2, lean=0, handF=(-170, -300), handFs="open", handB=(150, -420), handBs="fist",
                  footF=(-60, 0), footB=(40, 10), eye=.35, look=(0, -.5), brow=.6, browang=-.4, mouth="open", jaw=.3, wide=.1,
                  tongue=1.0, hair_up=.2)
        def petal(ctx):
            ctx.save()
            ctx.translate(838, 462)
            K.shape(ctx, [(0, 0), (10, -16), (6, -30), (-2, -20)], K.hexc("#f7c51e"), "petal", lw=2.4)
            ctx.restore()
        f.props(petal)
    return comp(bg(), f.rgba())


def bea_med(which, morning=False, line_shot=False):
    f = Frame(BEA_CAM)
    if morning:
        def sw(ctx):
            CA.swarm(ctx, 40, seed=12, y0=470, y1=700, x0=-80, x1=560, avoid=((150, 560, 150, 140),))
        f.props(sw, .6)
        if which == "a":
            f.bea(crown=True, hold="puff", eye=.5, look=(.9, -.6), mouth="closed")
        else:
            f.bea(crown=True, hold="stem", eye=.4, look=(.9, -.4), mouth="closed", smile=.6)
            f.props(lambda ctx: CA.seeds(ctx, 26, (150, 250, 560, 540), seed=5, s=1.5), .7)
        return comp(bg(BEA_CAM, morning=True), f.rgba())
    f.bea(hold="chest", eye=.5, look=(.9, .1), mouth="closed")
    return comp(bg(BEA_CAM), f.rgba())


def s05(which):
    f = Frame()
    f.bea(hold="sip", mouth="sip", look=(.9, .1))
    if which == "a":
        f.dd()
        tip = (782, 650)
        ang = -1.98

        def sp(ctx, p):
            ctx.save()
            world(ctx, p)
            CA.spade(ctx, tip, ang, s=.9)
            ctx.restore()

        top = np.array(tip) + np.array([math.cos(ang), math.sin(ang)]) * 270
        mid = np.array(tip) + np.array([math.cos(ang), math.sin(ang)]) * 170
        f.wendell(x=640, lean=.12, footF=(-40, 0), footB=local_of(774, 606, 640), footBa=.25, front_prop=sp,
                  handF=local_of(*top, 640), handFs="grip", handB=local_of(*mid, 640), handBs="grip", bendF=-1, bendB=1,
                  mouth="closed", smile=.9, squint=.3, browang=.6, look=(.9, .5))
        return comp(bg(), f.rgba())
    # deep in his own hole, only his top half poking out; the dandelion untouched on its island
    f.dd()
    ys = 1000

    f.props(lambda ctx: CA.spade(ctx, (600, 604), -1.78, s=.8, dirt=True), .8)
    f.wendell(x=770, y=ys, lean=0, handF=local_of(706, 662, 770, ys), handFs="fist", handB=local_of(836, 668, 770, ys), handBs="fist",
              bendF=-1, bendB=1, eye=1.0, eyesize=1.15, look=(.9, -.1), brow=.6, browang=-.3, mouth="closed",
              smile=-.4, dirty=.5, shadow=False)
    return comp(bg(lawn_state="crater"), f.rgba(), crater_front_mask())


def s06(which):
    f = Frame()
    f.bea(hold="chest", look=(.9, .1), eye=.4)
    f.dd()
    if which == "a":
        nz = (642, 548)
        f.props(lambda ctx: CA.vacuum(ctx, 300, 644, s=.72, bag=1.0, nozzle=(600, 560)), .72)

        def noz(ctx, p):
            ctx.save()
            world(ctx, p)
            CA.nozzle_tip(ctx, (596, 556), .42, s=.9)
            ctx.restore()

        f.wendell(x=520, lean=.06, handF=local_of(612, 566, 520), handFs="grip", handB=local_of(646, 584, 520), handBs="grip",
                  bendF=-1, bendB=1, front_prop=noz, mouth="closed", smile=1.0, squint=.2, browang=.5, look=(.9, .4),
                  footF=(-50, 0), footB=(60, 0))
        return comp(bg(lawn_state="crater"), f.rgba())
    f.props(lambda ctx: CA.vacuum(ctx, 300, 644, s=.72, burst=True, nozzle=(592, 610)), .72)

    def noz(ctx, p):
        ctx.save()
        world(ctx, p)
        CA.nozzle_tip(ctx, (588, 606), 1.2, s=.9)
        ctx.restore()

    f.wendell(x=520, lean=-.04, handF=local_of(590, 600, 520), handFs="grip", handB=(70, -250), handBs="fist", bendF=-1,
              front_prop=noz, mouth="closed", smile=-.3, eye=1.0, eyesize=1.1, look=(.2, .1), brow=.3, hair_up=1.3, dirty=.85,
              footF=(-50, 0), footB=(60, 0))
    return comp(bg(lawn_state="wrecked"), f.rgba())


def s08(which):
    f = Frame()
    f.bea(hold="chest", look=(.9, .1), eye=.4)
    if which == "a":
        f.dd(state="puff")
        f.wendell(x=700, lean=.1, handF=(-40, -170), handFs="fist", handB=(70, -170), handBs="fist", bendF=-1, bendB=1,
                  mouth="frown", eye=.7, look=(.9, .6), brow=.2, browang=-.6, dirty=.85, tilt=.08)
    else:
        def held(ctx, p):
            wr = p["_wrB"]
            base = wr + (0, 34)
            CH.dandelion(ctx, base[0], base[1], s=1.5, state="puff", tag="held", leaves=False)
            for q in range(5):
                K.line(ctx, [base, base + (-16 + 8 * q, 26 + 8 * (q % 2)), base + (-20 + 10 * q, 44 + 6 * (q % 3))], f"root{q}", lw=1.8)
            K.shape(ctx, K.ellipse_pts(base[0], base[1] + 6, 12, 8, 7), K.hexc("#6b4a35"), "clump", lw=2.2)
            CH.regrip(ctx, p, "B")

        f.wendell(x=780, lean=-.08, handB=(124, -470), handBs="grip", handBa=-1.57, bendB=1, handF=(-110, -330), handFs="open",
                  bendF=-1, mouth="open", jaw=.55, wide=.5, tongue=.5, eye=1.0, eyesize=1.1, look=(.7, -.6), brow=.9, browang=-.5,
                  dirty=.85, prop=held, footF=(-50, 0), footB=(60, 0))
    return comp(bg(lawn_state="wrecked"), f.rgba())


def s09(which):
    cam = (760, 400, 1.45)
    f = Frame(cam)
    if which == "a":
        def held(ctx, p):
            wr = p["_wrB"]
            CH.dandelion(ctx, wr[0], wr[1] + 50, s=1.3, state="puff", tag="held", leaves=False)
            CH.regrip(ctx, p, "B")

        f.wendell(x=700, y=700, lean=0, handB=(150, -360), handBs="grip", handBa=-1.57, bendB=1, handF=(-66, -248), handFs="hip",
                  mouth="closed", smile=1.0, eye=1.0, look=(.9, -.9), brow=.7, browang=-.3, dirty=.85, prop=held)
    else:
        def held(ctx, p):
            wr = p["_wrB"]
            CH.dandelion(ctx, wr[0], wr[1] + 50, s=1.3, state="bare", tag="held", leaves=False)
            CH.regrip(ctx, p, "B")

        f.wendell(x=700, y=700, lean=0, handB=(150, -360), handBs="grip", handBa=-1.57, bendB=1, handF=(-66, -248), handFs="hip",
                  mouth="closed", smile=1.0, eye=.25, look=(0, 0), brow=.9, browang=-.3, dirty=.85, prop=held)
        f.props(lambda ctx: CA.seeds(ctx, 40, (380, 110, 1240, 600), seed=9, s=2.1), 1.2)
    return comp(bg(cam, lawn_state="wrecked"), f.rgba())


def s10(which):
    f = Frame()

    def sw(ctx):
        CA.swarm(ctx, 120, seed=21, y0=462, y1=700, x0=-20, x1=1300, avoid=((470, 640, 120, 40), (150, 600, 120, 60)))
    f.props(sw, .7)
    f.bea(crown=True, hold="sip", mouth="sip", look=(.9, .1))
    if which == "a":
        f.wendell(x=470, lean=-.05, handF=(-90, -470), handFs="fist", handB=(100, -480), handBs="fist", bendF=-1, bendB=1,
                  eye=.0, brow=.5, mouth="closed", smile=.8)
    else:
        f.wendell(x=470, lean=-.16, hipx=-10, hipy=18, **TAKE, look=(.5, .6), handF=(-170, -350), handFs="open", handB=(170, -360),
                  handBs="open", bendF=-1, bendB=1, footF=(-60, -4), footB=(60, -2))
    return comp(bg(morning=True), f.rgba())


SHOTS = {
    "s01": (s01, "ab"), "s02": (s02, "ab"), "s03": (s03, "ab"), "s04": (lambda w: bea_med(w), "a"), "s05": (s05, "ab"),
    "s06": (s06, "ab"), "s08": (s08, "ab"), "s09": (s09, "ab"), "s10": (s10, "ab"), "s11": (lambda w: bea_med(w, morning=True), "ab"),
}


def save(img, path):
    assert img.shape == (H, W, 3)
    cv2.imwrite(path, (np.clip(img, 0, 1) * 255 + .5).astype(np.uint8)[:, :, ::-1])


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else OUT
    want = sys.argv[2:] or list(SHOTS)
    os.makedirs(os.path.join(out, "keys"), exist_ok=True)
    thumbs = []
    for sid in want:
        fn, which = SHOTS[sid]
        for w in which:
            img = fn(w)
            path = os.path.join(out, "keys", f"{sid}_{w}.png")
            save(img, path)
            t = cv2.resize((img * 255).astype(np.uint8), (426, 234), interpolation=cv2.INTER_AREA)
            cv2.putText(t, f"{sid}_{w}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
            thumbs.append(t)
            print(path)
    while len(thumbs) % 3:
        thumbs.append(np.zeros_like(thumbs[0]))
    sheet = np.vstack([np.hstack(thumbs[k:k + 3]) for k in range(0, len(thumbs), 3)])
    cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), "keys_sheet.jpg"), sheet[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
