"""Supporting cast and props for the short.

BEA PRUITT, 6: Wendell's little sister and the yard's unofficial judge.  Round head, two black curly hair puffs with yellow
bobbles, sleepy half-lidded eyes, a wide flat deadpan mouth; yellow tee, grape-purple overalls, pink sneakers.  Lives in a
webbed aluminium lawn chair with a juice box.
Props: garden spade, Mom's upright vacuum, dandelion swarm, floating seeds, the dandelion crown."""
import math

import numpy as np

import chars as CH
import ink as K
from ink import hexc, INK

B = dict(skin=hexc("#b9784f"), hair=hexc("#2b1b16"), bobble=hexc("#f6d04d"), tee=hexc("#f6d04d"), overall=hexc("#7d5ccc"),
         overall_d=hexc("#5f43a6"), shoe=hexc("#f28fb0"), sole=hexc("#fbf3ea"), eye=hexc("#fffef9"), mouth=hexc("#4d1020"),
         tongue=hexc("#ef6a78"), teeth=hexc("#fffdf2"), frame=hexc("#c9d0d8"), web1=hexc("#4fb37a"), web2=hexc("#f4f1e6"),
         juice=hexc("#f39a3b"), label=hexc("#fbf6ea"), straw=hexc("#f07aa0"), shadow=hexc("#2c3a2a", .22))

BDEF = dict(x=150, y=600, s=.62, eye=.5, look=(.6, .1), brow=0.0, mouth="closed", jaw=0.0, smile=0.0, hold="chest", crown=False,
            seeds=False, tilt=0.0)


def bpose(**kw):
    p = dict(BDEF)
    p.update(kw)
    return p


def chair_back(ctx, tag):
    """Back rest of the webbed lawn chair (drawn behind Bea)."""
    K.tube(ctx, [(-70, -70), (-96, -300)], 9, B["frame"], tag + "bk1", lw=2.8)
    K.tube(ctx, [(40, -70), (14, -300)], 9, B["frame"], tag + "bk2", lw=2.8)
    K.tube(ctx, [(-96, -300), (14, -300)], 9, B["frame"], tag + "bk3", lw=2.8)
    for k in range(6):
        y0 = -290 + k * 37
        x0 = -96 + (y0 + 300) * -26 / -230 * -1
        a = -94 + (300 + y0) * .113
        K.shape(ctx, [(a, y0), (a + 108, y0), (a + 106 - 3.3, y0 + 26), (a - 3, y0 + 26)], B["web1"] if k % 2 == 0 else B["web2"],
                tag + f"wb{k}", lw=2.0, tension=.2, n=3)


def chair_front(ctx, tag):
    """Seat webbing, legs and armrests (drawn over Bea's seat, under her arms)."""
    for k, (a, b) in enumerate(((-12, -88), (-88, 0))):
        pass
    K.tube(ctx, [(-80, -70), (-110, 0)], 8, B["frame"], tag + "lg1", lw=2.6)
    K.tube(ctx, [(-70, -70), (60, 0)], 8, B["frame"], tag + "lg2", lw=2.6)
    K.tube(ctx, [(60, -70), (-60, 0)], 8, B["frame"], tag + "lg3", lw=2.6)
    K.tube(ctx, [(50, -70), (80, 0)], 8, B["frame"], tag + "lg4", lw=2.6)
    K.tube(ctx, [(-84, -72), (60, -72)], 9, B["frame"], tag + "seat", lw=2.8)


def bea(ctx, p, tag="b"):
    ctx.save()
    ctx.translate(p["x"], p["y"])
    ctx.scale(p["s"], p["s"])
    ctx.save()
    ctx.scale(1, .2)
    ctx.arc(-10, 0, 140, 0, 2 * math.pi)
    ctx.restore()
    K.fillc(ctx, B["shadow"])
    chair_back(ctx, tag)

    # legs stick straight out over the seat edge (she's small), sneakers dangling
    for k, (hx, fx) in enumerate(((-30, 78), (-8, 104))):
        K.tube(ctx, [(hx, -92), (fx - 40, -96), (fx, -70)], 12, B["skin"], tag + f"leg{k}", n=6)
        ctx.save()
        ctx.translate(fx + 4, -64)
        ctx.rotate(-.5)
        K.shape(ctx, [(-14, -10), (8, -16), (34, -10), (40, 4), (-12, 6)], B["shoe"], tag + f"sh{k}", lw=3.0, tension=.7)
        K.shape(ctx, [(-14, 2), (40, 1), (38, 8), (-12, 9)], B["sole"], tag + f"so{k}", lw=2.4, tension=.3)
        ctx.restore()
    # overall legs (short pants) over the thighs
    K.shape(ctx, [(-60, -126), (40, -126), (56, -104), (52, -76), (-50, -74)], B["overall"], tag + "pants", tension=.6)
    K.line(ctx, [(-6, -118), (40, -86)], tag + "pants2", lw=1.8)

    # torso: yellow tee under overalls bib, sitting back in the chair
    K.shape(ctx, [(-58, -232), (-70, -180), (-64, -118), (20, -110), (34, -170), (24, -232), (-16, -244)], B["tee"], tag + "tee", tension=.8)
    K.shape(ctx, [(-52, -196), (14, -196), (22, -150), (30, -116), (-62, -112), (-58, -150)], B["overall"], tag + "bib", tension=.5)
    K.shape(ctx, [(-30, -184), (-2, -184), (0, -160), (-28, -160)], B["overall_d"], tag + "pocket", lw=2.2, tension=.3)
    for k, (x0, x1) in enumerate(((-48, -54), (10, 14))):
        K.tube(ctx, [(x0, -194), (x1 + (0 if k else 4), -236)], 8, B["overall"], tag + f"strap{k}", lw=2.4)
        K.dot(ctx, x0, -194, 4.2, B["bobble"], tag + f"btn{k}")

    # back arm resting on the armrest
    K.tube(ctx, [(-60, -222), (-84, -170), (-74, -120)], 11, B["skin"], tag + "armb", n=6)
    CH.hand(ctx, (-74, -118), 1.5, "fist", tag + "hb", sc=.85)
    K.tube(ctx, [(-104, -118), (-20, -118)], 8, B["frame"], tag + "arm_rest", lw=2.6)

    # head
    ctx.save()
    ctx.translate(-14, -248)
    ctx.rotate(p["tilt"])
    bhead(ctx, p, tag)
    ctx.restore()

    # front arm + whatever she holds
    hold = p["hold"]
    if hold == "sip":
        hand_at = np.array([40, -270])
    elif hold in ("puff", "stem"):
        hand_at = np.array([70, -206])
    else:
        hand_at = np.array([34, -170])
    el, wr = K.ik2(np.array([18, -226]), hand_at, 50, 52, bend=1)
    K.tube(ctx, [(18, -226), el, wr], 11, B["skin"], tag + "armf", n=6)
    if hold in ("sip", "chest"):
        jb = wr + (6, -6)
        K.shape(ctx, [jb + (-14, -22), jb + (14, -22), jb + (14, 22), jb + (-14, 22)], B["juice"], tag + "jb", lw=3.0, tension=.15, n=3)
        K.shape(ctx, [jb + (-9, -8), jb + (9, -8), jb + (9, 10), jb + (-9, 10)], B["label"], tag + "jbl", lw=1.6, tension=.15, n=3)
        K.dot(ctx, *(jb + (0, 1)), 4, hexc("#e2463d"), tag + "jbd")
        if hold == "sip":
            straw = [jb + (6, -22), jb + (10, -42), jb + (2, -52), (-14 + 64, -248 - 42)]
        else:
            straw = [jb + (6, -22), jb + (10, -40), jb + (18, -46)]
        K.tube(ctx, straw, 4, B["straw"], tag + "straw", lw=1.8)
        CH.hand(ctx, wr, -1.4, "grip", tag + "hf", sc=.85)
    elif hold == "puff":
        CH.dandelion(ctx, wr[0] - 2, wr[1] + 24, s=.8, state="puff", tag=tag + "pdd", leaves=False)
        CH.hand(ctx, wr, -1.4, "grip", tag + "hf", sc=.85)
    elif hold == "stem":
        CH.dandelion(ctx, wr[0] - 2, wr[1] + 24, s=.8, state="bare", tag=tag + "pdd", leaves=False)
        CH.hand(ctx, wr, -1.4, "grip", tag + "hf", sc=.85)
    else:
        CH.hand(ctx, wr, -1.2, "fist", tag + "hf", sc=.85)
    chair_front(ctx, tag)
    ctx.restore()


def bhead(ctx, p, tag):
    sk = B["skin"]
    # hair puffs behind
    for k, (hx, hy, r) in enumerate(((-58, -118, 40), (54, -124, 38))):
        pts = []
        for a in np.linspace(0, 2 * np.pi, 15, endpoint=False):
            rr = r * (1 + .12 * (int(a * 15 / 6.28) % 2))
            pts.append((hx + rr * math.cos(a), hy + rr * math.sin(a)))
        K.shape(ctx, pts, B["hair"], tag + f"puff{k}", tension=.7)
    # ear
    K.shape(ctx, K.ellipse_pts(-76, -50, 14, 18, 8), sk, tag + "ear")
    # head: wide and round with chubby cheeks
    head = [(-60, 0), (-84, -40), (-82, -92), (-52, -126), (0, -136), (52, -124), (80, -92), (88, -50), (78, -14), (40, 6), (-12, 8)]
    K.shape(ctx, head, sk, tag + "head", tension=1.0)
    # hair cap with a curly hairline
    cap = [(-84, -70), (-70, -80), (-60, -96), (-40, -98), (-26, -110), (-6, -100), (10, -112), (28, -100), (44, -110), (62, -96),
           (80, -94), (76, -122), (46, -142), (0, -150), (-50, -140), (-80, -112)]
    K.shape(ctx, cap, B["hair"], tag + "cap", tension=.8)
    for k, (bx, by) in enumerate(((-40, -136), (44, -138))):
        K.shape(ctx, K.ellipse_pts(bx, by, 11, 9, 7), B["bobble"], tag + f"bob{k}", lw=2.6)
    if p.get("crown"):
        for k in range(9):
            a = math.pi * (1.08 + k * .105)
            fx, fy = 10 + 92 * math.cos(a), -96 + 52 * math.sin(a)
            pts = [(fx + (10 if q % 2 == 0 else 6) * math.cos(q * math.pi / 6), fy + (10 if q % 2 == 0 else 6) * math.sin(q * math.pi / 6))
                   for q in range(12)]
            K.shape(ctx, pts, hexc("#f7c51e"), tag + f"cr{k}", lw=2.0, tension=.5)
    # eyes: half-lidded, deadpan
    for k, (ex, ey, rx, ry) in enumerate(((12, -66, 17, 20), (50, -66, 14, 19))):
        ed = K.crs(K.jitter(K.ellipse_pts(ex, ey, rx, ry, 10), tag + f"e{k}"), True, 10)
        K.poly_path(ctx, ed)
        K.fillc(ctx, B["eye"])
        ctx.save()
        K.poly_path(ctx, ed)
        ctx.clip()
        lx, ly = p["look"]
        K.dot(ctx, ex + lx * rx * .45, ey + ry * .3 + ly * 4, 4.4, INK, tag + f"pu{k}")
        lid_y = ey - ry + (1 - p["eye"]) * 2 * ry
        K.poly_path(ctx, np.array([(ex - rx - 4, lid_y), (ex + rx + 4, lid_y), (ex + rx + 4, ey - ry - 30), (ex - rx - 4, ey - ry - 30)]))
        K.fillc(ctx, sk)
        K.line(ctx, [(ex - rx - 2, lid_y), (ex + rx + 2, lid_y)], tag + f"lid{k}", lw=3.0, k=.4)
        ctx.restore()
        K.ink_poly(ctx, ed, tag + f"e{k}", 3.0, True)
        by = ey - ry - 10 - 10 * p["brow"]
        K.line(ctx, [(ex - rx * .7, by + 1), (ex + rx * .7, by - 1)], tag + f"br{k}", lw=3.4)
    # button nose
    K.shape(ctx, K.ellipse_pts(74, -42, 11, 9, 8), sk, tag + "nose", lw=3.2)
    # mouth: wide and flat; opens big when she talks
    if p["mouth"] == "open" and p["jaw"] > .02:
        jd = 34 * p["jaw"]
        mp = K.crs(K.jitter([(8, -26), (40, -28), (74, -26), (70, -18 + jd * .6), (44, -14 + jd), (16, -18 + jd * .6)], tag + "mo"), True, 10)
        K.poly_path(ctx, mp)
        K.fillc(ctx, B["mouth"])
        ctx.save()
        K.poly_path(ctx, mp)
        ctx.clip()
        K.shape(ctx, [(14, -12 + jd), (40, -22 + jd * .7), (70, -14 + jd), (44, 0 + jd)], B["tongue"], tag + "tg", lw=2.0)
        K.shape(ctx, [(10, -32), (72, -32), (70, -22), (12, -22)], B["teeth"], tag + "tt", lw=1.8, tension=.3)
        ctx.restore()
        K.ink_poly(ctx, mp, tag + "mo", K.LW, True)
    elif p["mouth"] == "sip":
        K.line(ctx, [(30, -24), (48, -25)], tag + "ml", lw=3.6)
        K.shape(ctx, K.ellipse_pts(50, -25, 6, 5, 6), B["mouth"], tag + "sp", lw=2.4)
    else:
        sm = p["smile"]
        K.line(ctx, [(6, -24 - 6 * sm), (24, -22), (50, -22), (74, -26 - 8 * sm)], tag + "ml", lw=3.8, taper=(.4, .4))


# ------------------------------------------------------------------------------------------------ props
def spade(ctx, tip, ang, tag="sp", s=1.0, dirt=False):
    """Garden spade: blade tip at `tip` (world), shaft rising along ang (radians from +x)."""
    ctx.save()
    ctx.translate(*tip)
    ctx.rotate(ang + math.pi / 2)
    ctx.scale(s, s)
    K.tube(ctx, [(0, -60), (0, -300)], 11, hexc("#c08a52"), tag + "shaft", lw=3.0)
    K.shape(ctx, [(-22, -300), (22, -300), (16, -336), (-16, -336)], hexc("#c08a52"), tag + "grip", lw=3.0, tension=.4)
    K.shape(ctx, [(-10, -312), (10, -312), (8, -326), (-8, -326)], hexc("#7a5136"), tag + "gh", lw=2.0, tension=.4)
    K.shape(ctx, [(-28, -66), (28, -66), (30, -14), (0, 4), (-30, -14)], hexc("#aab4bf"), tag + "blade", lw=3.2, tension=.4)
    K.line(ctx, [(-18, -54), (-20, -20)], tag + "bl2", lw=1.6, ink=hexc("#ffffff", .8))
    if dirt:
        K.shape(ctx, [(-28, -30), (28, -30), (30, -14), (0, 4), (-30, -14)], hexc("#6b4a35"), tag + "bd", lw=2.4, tension=.4)
    ctx.restore()
    return np.array(tip) + np.array([math.cos(ang), math.sin(ang)]) * 300 * s


def vacuum(ctx, x, y, s=1.0, bag=1.0, burst=False, nozzle=None, tag="vac"):
    """Mom's upright vacuum: red body, cloth bag that swells (bag>1) or bursts, grey ribbed hose to a nozzle."""
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(s, s)
    # cord trailing off to the house (left)
    K.line(ctx, [(-30, -14), (-200, 10), (-500, 0), (-900, 14)], tag + "cord", lw=3.2)
    K.shape(ctx, [(-70, -40), (70, -40), (80, -8), (70, 6), (-70, 6), (-80, -8)], hexc("#d8443c"), tag + "base", tension=.4)
    for k, wx in enumerate((-54, 54)):
        K.shape(ctx, K.ellipse_pts(wx, 4, 12, 12, 8), hexc("#2b2b2b"), tag + f"wh{k}", lw=2.6)
    K.shape(ctx, [(-20, -40), (20, -40), (16, -120), (-16, -120)], hexc("#b8322c"), tag + "neck", tension=.3)
    if burst:
        # tattered bag rags hanging from the handle
        for k in range(5):
            a = -1.0 + k * .5
            K.shape(ctx, [(0, -300), (60 * math.sin(a) - 12, -250 + 20 * abs(a)), (64 * math.sin(a) + 4, -220 + 34 * abs(a) + 10 * (k % 2)),
                          (60 * math.sin(a) + 14, -250 + 20 * abs(a))], hexc("#e8d9b8"), tag + f"rag{k}", lw=2.6, tension=.6)
    else:
        bw = 46 * bag
        K.shape(ctx, [(-16, -118), (-bw, -150 - 10 * bag), (-bw * 1.05, -230), (-bw * .8, -296), (0, -306), (bw * .8, -296),
                      (bw * 1.05, -230), (bw, -150 - 10 * bag), (16, -118)], hexc("#e8d9b8"), tag + "bag", tension=.9)
        K.line(ctx, [(-bw * .5, -170), (-bw * .55, -260)], tag + "bagf", lw=1.6, k=.4)
        if bag > 1.3:
            for q in range(3):
                K.line(ctx, [(bw * (.9 + .1 * q), -200 - 30 * q), (bw * 1.25 + 8 * q, -210 - 34 * q)], tag + f"strain{q}", lw=2.4)
    K.tube(ctx, [(0, -300), (0, -380)], 12, hexc("#d8443c"), tag + "stick", lw=3.0)
    K.tube(ctx, [(-26, -386), (26, -386)], 14, hexc("#2b2b2b"), tag + "handle", lw=3.0)
    if nozzle is not None:
        nz = (np.asarray(nozzle, float) - (x, y)) / s
        hose = [(18, -60), (70, -40), ((70 + nz[0]) / 2, (nz[1] - 30) / 2 + 20), tuple(nz)]
        d = K.tube(ctx, hose, 20, hexc("#8a9098"), tag + "hose", lw=3.0)
        for q in range(4, len(d) - 3, 6):  # ribs
            v = d[min(q + 1, len(d) - 1)] - d[q - 1]
            nrm = np.array([-v[1], v[0]]) / (np.linalg.norm(v) + 1e-6)
            K.line(ctx, [d[q] - nrm * 9, d[q] + nrm * 9], tag + f"rib{q}", lw=1.4, k=.3)
    ctx.restore()


def nozzle_tip(ctx, at, ang, tag="nz", s=1.0):
    ctx.save()
    ctx.translate(*at)
    ctx.rotate(ang)
    ctx.scale(s, s)
    K.shape(ctx, [(-6, -12), (46, -16), (52, -10), (52, 10), (46, 16), (-6, 12)], hexc("#d8443c"), tag + "nz", tension=.3)
    K.shape(ctx, K.ellipse_pts(52, 0, 6, 14, 8), hexc("#2b1216"), tag + "nzh", lw=2.4)
    ctx.restore()


def seeds(ctx, n, box, seed=3, tag="seed", s=1.0):
    """Floating dandelion parachutes."""
    r = np.random.default_rng(seed)
    x0, y0, x1, y1 = box
    for k in range(n):
        x, y = r.uniform(x0, x1), r.uniform(y0, y1)
        sc = s * r.uniform(.7, 1.2)
        a = r.uniform(-.6, .6)
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(a)
        ctx.scale(sc, sc)
        K.line(ctx, [(0, 0), (0, 16)], f"{tag}{k}s", lw=1.5, k=.3, ink=hexc("#55595f"))
        K.dot(ctx, 0, 17, 1.6, hexc("#8a6a55"))
        for q in range(7):
            b = -math.pi / 2 + (q - 3) * .32
            K.line(ctx, [(0, 0), (11 * math.cos(b), 11 * math.sin(b) + 2)], f"{tag}{k}f{q}", lw=1.3, k=.2, ink=hexc("#6d737a"))
        for q in range(7):
            b = -math.pi / 2 + (q - 3) * .32
            K.dot(ctx, 11 * math.cos(b), 11 * math.sin(b) + 2, 1.8, hexc("#ffffff"))
        ctx.restore()


def swarm(ctx, n, seed=8, y0=470, y1=700, x0=-20, x1=1300, avoid=()):
    """A lawn overrun with dandelions, smaller with distance."""
    r = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        x, y = r.uniform(x0, x1), r.uniform(y0, y1)
        if any(abs(x - ax) < aw and abs(y - ay) < ah for ax, ay, aw, ah in avoid):
            continue
        pts.append((x, y))
    pts.sort(key=lambda q: q[1])
    for k, (x, y) in enumerate(pts):
        sc = .35 + .75 * (y - y0) / (y1 - y0)
        CH.dandelion(ctx, x, y, s=sc, sway=r.uniform(-.4, .4), tag=f"sw{k}", leaves=sc > .55)
