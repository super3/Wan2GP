"""Characters for wobble_suburb, drawn as inked cels (flat colour, living ink line, boil).

WENDELL PRUITT, 10: a beanpole with a lantern jaw, a pickle nose, a mop of tangerine hair, ears like jug handles, big buck
teeth and a mouth that opens wider than his head.  Sky-blue polo, khaki shorts to the knee, tall socks, clown-big sneakers.
Pose is a dict; every pose renders from the same rig so the code shot, the H3 keys and the short share one model sheet."""
import math

import numpy as np

import ink as K
from ink import hexc, INK

C = dict(
    skin=hexc("#f7c8a0"), skin_d=hexc("#e2a27c"), hair=hexc("#f2792a"), hair_d=hexc("#c9561a"), shirt=hexc("#58aee6"),
    collar=hexc("#fbf8f0"), shorts=hexc("#d7b46e"), sock=hexc("#fbf7ee"), band=hexc("#e2463d"), shoe=hexc("#bf3b2f"),
    sole=hexc("#f6f0e2"), mouth=hexc("#4d1020"), tongue=hexc("#ef6a78"), teeth=hexc("#fffdf2"), eye=hexc("#fffef9"),
    freckle=hexc("#c9744f"), shadow=hexc("#2c3a2a", .22),
)

DEF = dict(x=640, y=640, s=1.0, face=1, lean=0.0, hipx=0.0, hipy=0.0, tilt=0.0, neck=1.0, sx=1.0, sy=1.0,
           handF=(-62, -250), handB=(64, -250), bendF=-1, bendB=1, handFs="hip", handBs="hip", handFa=None, handBa=None,
           footF=(-42, 0), footB=(52, 0), footFa=0.0, footBa=0.0,
           eye=1.0, look=(.55, .05), brow=0.0, browang=0.0, squint=0.0, eyesize=1.0,
           mouth="closed", jaw=0.0, wide=0.0, smile=0.0, tongue=.6, hair_up=0.0, sweat=0, shadow=True, prop=None, rot=0.0, dirty=0.0,
           cheeks=0.0)


def pose(**kw):
    p = dict(DEF)
    p.update(kw)
    return p


def _ang(v):
    return math.atan2(v[1], v[0])


# ------------------------------------------------------------------------------------------------ hands
def hand(ctx, wrist, ang, kind, tag, sc=1.0, flip=1):
    """A four-fingered cartoon hand at wrist, pointing along ang.  flip mirrors the thumb side."""
    ctx.save()
    ctx.translate(*wrist)
    ctx.rotate(ang)
    ctx.scale(sc, sc * flip)
    sk = C["skin"]
    if kind == "open":
        for k, a in enumerate((-.55, -.18, .18, .52)):
            L = (24, 28, 27, 22)[k]
            K.tube(ctx, [(10, 0), (10 + L * .6 * math.cos(a), L * .6 * math.sin(a)), (10 + L * math.cos(a * 1.1), L * math.sin(a * 1.1))],
                   7.5, sk, f"{tag}f{k}", lw=2.8, n=6)
        K.tube(ctx, [(4, -8), (12, -20), (20, -26)], 8, sk, f"{tag}th", lw=2.8, n=6)
        K.shape(ctx, K.ellipse_pts(8, 0, 14, 13, 9), sk, f"{tag}p", lw=3.0)
    elif kind == "point":
        K.tube(ctx, [(10, -4), (26, -6), (44, -6)], 8, sk, f"{tag}ix", lw=2.8, n=6)
        K.shape(ctx, [(-2, -12), (14, -14), (24, -8), (26, 6), (18, 14), (2, 13), (-4, 2)], sk, f"{tag}p", lw=3.0)
        K.line(ctx, [(14, -1), (22, 0)], f"{tag}k1", lw=1.8)
        K.line(ctx, [(12, 6), (20, 7)], f"{tag}k2", lw=1.8)
        K.tube(ctx, [(6, -10), (14, -18), (20, -16)], 7, sk, f"{tag}th", lw=2.6, n=5)
    elif kind == "grip":  # curled round a handle running across (perpendicular to the forearm)
        K.shape(ctx, [(-2, -12), (14, -15), (26, -10), (29, 4), (22, 14), (4, 14), (-4, 2)], sk, f"{tag}p", lw=3.0)
        for k, y in enumerate((-6, 1, 8)):
            K.line(ctx, [(17, y - 1), (27, y)], f"{tag}k{k}", lw=1.7)
    elif kind == "hip":  # flat mitten planted on the hip, fingers toward the back
        K.shape(ctx, [(-2, -11), (16, -14), (32, -10), (38, 0), (32, 9), (14, 12), (-2, 9)], sk, f"{tag}p", lw=3.0)
        K.line(ctx, [(20, -3), (34, -1)], f"{tag}k1", lw=1.6)
        K.line(ctx, [(18, 4), (31, 5)], f"{tag}k2", lw=1.6)
    else:  # fist
        K.shape(ctx, [(-2, -12), (14, -15), (25, -9), (27, 5), (19, 14), (3, 13), (-4, 2)], sk, f"{tag}p", lw=3.0)
        K.line(ctx, [(15, -4), (24, -3)], f"{tag}k1", lw=1.7)
        K.line(ctx, [(14, 4), (22, 5)], f"{tag}k2", lw=1.7)
        K.line(ctx, [(6, -12), (14, -5)], f"{tag}th", lw=1.8)
    ctx.restore()


# ------------------------------------------------------------------------------------------------ head
def head(ctx, p, tag="w"):
    """Wendell's head in 3/4 facing right; origin at the neck socket under the jaw."""
    J = p["jaw"]
    WD = p["wide"]
    up = p["hair_up"]
    jd = 74 * J  # jaw drop
    sk = C["skin"]

    # ear (back side) first
    K.shape(ctx, [(-58, -116), (-92, -132), (-110, -104), (-102, -72), (-70, -66)], sk, tag + "ear", tension=1.0)
    K.line(ctx, [(-74, -110), (-94, -112), (-96, -92), (-80, -84)], tag + "ear2", lw=2.2)

    # hair behind the skull (back mass)
    back = [(-70, -96), (-92, -100), (-86, -122), (-108, -132), (-96, -152), (-112, -170 - 14 * up), (-84, -180 - 8 * up),
            (-84, -206 - 26 * up), (-52, -204 - 10 * up), (-36, -232 - 40 * up), (-6, -214 - 12 * up), (14, -238 - 42 * up),
            (34, -212 - 10 * up), (62, -226 - 36 * up), (70, -196), (96, -196 - 20 * up), (84, -166), (40, -156), (-30, -150), (-56, -120)]
    K.shape(ctx, back, C["hair"], tag + "hairb", tension=.75)

    # skull + face; lower jaw points slide down when the mouth opens
    jaw_pts = [(118 + 6 * WD, -34 + jd * .45), (112 + 4 * J, -2 + jd), (70, 16 + jd), (20, 14 + jd * .7)]
    outline = [(-30, 8), (-66, -24), (-82, -78), (-76, -134), (-52, -176), (0, -196), (50, -186), (78, -152), (86, -116), (90, -84),
               (112, -64)] + jaw_pts
    K.shape(ctx, outline, sk, tag + "head", tension=.95)

    # open mouth: dark interior, row of big teeth, a tongue
    if p["mouth"] in ("open", "yell", "grin_open") and J > .02:
        x0 = 18 - 34 * WD  # back corner travels toward the ear when he yells
        top = [(x0, -46), (64, -52 - 4 * WD), (112 + 6 * WD, -58)]
        bot = [(112 + 6 * WD, -50 + jd * .35), (92, -40 + jd * .95), (56, -40 + jd), (x0 + 6, -42 + jd * .55)]
        mp = K.crs(K.jitter(top + bot, tag + "mo"), True, 10)
        K.poly_path(ctx, mp)
        K.fillc(ctx, C["mouth"])
        ctx.save()
        K.poly_path(ctx, mp)
        ctx.clip()
        # tongue
        ty = -40 + jd * .92
        K.shape(ctx, [(x0 + 16, ty + 4), (52, ty - 12 - 14 * p["tongue"]), (90, ty - 10 - 10 * p["tongue"]), (104, ty + 6), (60, ty + 22)],
                C["tongue"], tag + "tg", lw=2.4)
        K.line(ctx, [(64, ty - 2 - 10 * p["tongue"]), (74, ty + 6)], tag + "tg2", lw=1.8)
        # upper teeth: big blocky slabs
        tx = np.linspace(x0 + 10, 102, 6)
        for k in range(5):
            a, b = tx[k], tx[k + 1]
            yt = -54 + (a - 30) * -.03
            K.shape(ctx, [(a, yt - 8), (b, yt - 9), (b - 1, yt + 11 + 3 * (k % 2)), (a + 1, yt + 12)], C["teeth"], tag + f"t{k}", lw=2.2,
                    tension=.4, n=4)
        ctx.restore()
        K.ink_poly(ctx, mp, tag + "mo", K.LW, True)
    elif p["mouth"] == "blow":
        pass
    else:
        # closed mouth: a long wavy line from cheek to snout, buck teeth resting on the lower lip
        sm = p["smile"]
        ml = [(14, -50 - 10 * sm), (32, -42 - 3 * sm), (66, -38 + 2 * sm), (100, -40), (116, -47 - 4 * sm)]
        if p["mouth"] == "frown":
            ml = [(32, -34), (50, -42), (76, -44), (100, -40), (106, -34)]
        if p["mouth"] == "grit":
            ml = [(30, -46), (108, -48)]
            K.shape(ctx, [(36, -50), (106, -52), (104, -30), (40, -30)], C["teeth"], tag + "grt", lw=2.6, tension=.3)
            for k, x in enumerate((52, 68, 84, 98)):
                K.line(ctx, [(x, -50), (x + 1, -31)], tag + f"gt{k}", lw=1.6)
            K.line(ctx, [(38, -40), (105, -41)], tag + "gm", lw=1.6)
        if p["mouth"] != "grit":
            # buck teeth hang over the lip line
            K.shape(ctx, [(78, -39), (91, -39.5), (90.5, -23), (79, -23.5)], C["teeth"], tag + "bt1", lw=2.4, tension=.3, n=5)
            K.shape(ctx, [(91, -39.5), (104, -40), (103, -25), (91, -24)], C["teeth"], tag + "bt2", lw=2.4, tension=.3, n=5)
        K.line(ctx, ml, tag + "ml", lw=K.LW, taper=(.35, .5))
        if p["mouth"] != "grit":
            K.line(ctx, [(8, -58 - 12 * sm), (15, -49 - 10 * sm), (13, -41 - 8 * sm)], tag + "crease", lw=2.2)  # cheek crease
    # chin line hint
    K.line(ctx, [(80, -8 + jd), (96, -10 + jd)], tag + "chin", lw=1.8, k=.6)

    if p.get("dirty", 0) > .3:
        dc = hexc("#8a6248", .85)
        for q, (sx_, sy_, a_, b_) in enumerate(((-30, -100, 30, 20), (50, -64, 22, 14), (10, -150, 26, 12), (-50, -50, 18, 22))):
            pts = [(sx_ + a_ * (1 + .25 * math.sin(3 * t + q)) * math.cos(t), sy_ + b_ * (1 + .25 * math.cos(2 * t + q)) * math.sin(t))
                   for t in np.linspace(0, 6.28, 9, endpoint=False)]
            K.poly_path(ctx, K.crs(K.jitter(pts, tag + f"sm{q}"), True, 6))
            K.fillc(ctx, dc)
    if p.get("cheeks", 0) > 0:
        ch = p["cheeks"]
        K.shape(ctx, K.ellipse_pts(48, -62, 30 * ch, 24 * ch, 9), sk, tag + "chk", lw=2.6)
    if p["mouth"] == "blow":
        K.shape(ctx, K.ellipse_pts(112, -44, 9, 8, 8), C["mouth"], tag + "blow", lw=2.8)
    # freckles
    for k, (fx, fy) in enumerate(((44, -78), (56, -72), (40, -66), (104, -86))):
        K.dot(ctx, fx, fy, 2.4, C["freckle"], tag + f"fr{k}")

    # eyes: near eye bigger, far eye overlaps the face edge; lids for squint and blink
    es = p["eyesize"]
    for k, (ex, ey, rx, ry) in enumerate(((36, -122, 22, 29), (74, -124, 17, 27))):
        rx, ry = rx * es, ry * es
        if p["eye"] < .08:
            K.line(ctx, [(ex - rx, ey + 2), (ex, ey - 8), (ex + rx, ey + 2)], tag + f"ce{k}", lw=3.6)
            by = ey - ry - 14 - 14 * p["brow"]
            K.line(ctx, [(ex - rx * .9, by + 2), (ex, by - 4), (ex + rx * .9, by + 1)], tag + f"br{k}", lw=4.2, taper=(.3, .3))
            continue
        e_pts = K.ellipse_pts(ex, ey, rx, ry, 10, rotn=.06)
        ed = K.crs(K.jitter(e_pts, tag + f"e{k}"), True, 10)
        K.poly_path(ctx, ed)
        K.fillc(ctx, C["eye"])
        open_ = p["eye"] * (1 - .55 * p["squint"])
        lx, ly = p["look"]
        ctx.save()
        K.poly_path(ctx, ed)
        ctx.clip()
        if open_ > .08:
            K.dot(ctx, ex + lx * rx * .5, ey + ly * ry * .45 + ry * .05, 5.2 * min(1.0, 1 / es ** .3), INK, tag + f"pu{k}")
        # upper lid as skin coming down, slanted for anger (inner end lower)
        lid_y = ey - ry + (1 - open_) * 2 * ry
        sl = p["browang"] * 10 * (1 if k == 0 else -1) * -1
        lid = np.array([(ex - rx - 4, lid_y - sl), (ex + rx + 4, lid_y + sl), (ex + rx + 4, ey - ry - 40), (ex - rx - 4, ey - ry - 40)])
        if open_ < .98 or abs(sl) > .5:
            K.poly_path(ctx, lid)
            K.fillc(ctx, sk)
            K.line(ctx, [lid[0], lid[1]], tag + f"lid{k}", lw=2.8, k=.5)
        if p["squint"] > .05:  # lower lid pushing up
            lo_y = ey + ry - p["squint"] * ry * .7
            K.poly_path(ctx, np.array([(ex - rx - 4, lo_y + 3), (ex + rx + 4, lo_y - 3), (ex + rx + 4, ey + ry + 20), (ex - rx - 4, ey + ry + 20)]))
            K.fillc(ctx, sk)
            K.line(ctx, [(ex - rx, lo_y + 3), (ex + rx, lo_y - 3)], tag + f"llid{k}", lw=2.2, k=.5)
        ctx.restore()
        K.ink_poly(ctx, ed, tag + f"e{k}", 3.2, True)
        # brows: floating strokes
        by = ey - ry - 14 - 14 * p["brow"]
        ang = p["browang"] * .45 * (1 if k == 0 else -1)
        bx0, bx1 = ex - rx * .9, ex + rx * .9
        cxm = (bx0 + bx1) / 2
        bp = K.rot(np.array([(bx0, by + 2), (cxm, by - 4), (bx1, by + 1)]), ang, (cxm, by))
        K.line(ctx, bp, tag + f"br{k}", lw=4.2, taper=(.3, .3))

    # the pickle nose, drawn over everything on the face
    K.shape(ctx, [(80, -112), (106, -110), (128, -98), (134, -80), (122, -66), (100, -66), (84, -74)], sk, tag + "nose", tension=1.0)
    K.line(ctx, [(106, -74), (112, -70)], tag + "nost", lw=1.8)

    # hair cap over the skull: a scruffy mop with a ragged hairline and fringe clumps over the forehead
    cap = [(-80, -112), (-62, -122), (-58, -140), (-40, -136), (-30, -156), (-8, -150), (6, -168), (18, -150 + 8 * up),
           (30, -170), (44, -146 + 10 * up), (56, -168), (72, -150 + 6 * up), (80, -170), (84, -186), (60, -204), (20, -212),
           (-24, -208), (-60, -194), (-82, -166), (-90, -136)]
    K.shape(ctx, cap, C["hair"], tag + "cap", tension=.7)
    K.line(ctx, [(-50, -184), (-30, -176), (-16, -182)], tag + "hl2", lw=1.8, k=.6)
    K.line(ctx, [(20, -196), (38, -188)], tag + "hl3", lw=1.8, k=.6)


# ------------------------------------------------------------------------------------------------ body
SOIL = hexc("#6b4a35")


def wendell(ctx, p, tag="w"):
    if p.get("dirty", 0) > 0:
        saved = dict(C)
        amt = {"skin": .0, "hair": .35, "shirt": .5, "collar": .55, "shorts": .45, "sock": .55, "band": .3, "shoe": .3, "sole": .55}
        for key, a in amt.items():
            a *= p["dirty"]
            C[key] = tuple(C[key][i] * (1 - a) + SOIL[i] * a for i in range(3)) + (1.0,)
        try:
            _wendell(ctx, p, tag)
        finally:
            C.update(saved)
        return
    _wendell(ctx, p, tag)


def _wendell(ctx, p, tag="w"):
    ctx.save()
    ctx.translate(p["x"], p["y"])
    ctx.rotate(p.get("rot", 0.0))
    s = p["s"]
    ctx.scale(s * p["face"] * p["sx"], s * p["sy"])
    if p["shadow"] and not p.get("rot"):
        ctx.save()
        ctx.scale(1, .2)
        ctx.arc(10, 0, 120, 0, 2 * math.pi)
        ctx.restore()
        K.fillc(ctx, C["shadow"])

    hip = np.array([p["hipx"], -232 + p["hipy"]])
    lean = p["lean"]

    def T(q):  # torso-local -> body space
        return K.rot(np.asarray(q, float), lean) + hip

    shF, shB = T((-30, -122)), T((30, -126))
    neck0 = T((2, -132))
    hipF, hipB = hip + (-24, 0), hip + (26, -2)

    # --- legs
    def leg(hp, foot, fa, which):
        ankle = np.array(foot) + (0, -34)
        knee, ank = K.ik2(hp, ankle, 100, 104, bend=1 if which == "F" else 1, stretch=True)
        # knees bend forward (toward +x), so pick the solution with knee x > line
        knee2, _ = K.ik2(hp, ankle, 100, 104, bend=-1, stretch=True)
        if knee2[0] > knee[0]:
            knee = knee2
        K.tube(ctx, [hp, knee, ank], 13, C["skin"], f"{tag}leg{which}", n=8)
        # sock up the shin
        sock_top = ank + (knee - ank) * .55
        K.tube(ctx, [sock_top, ank + (0, 4)], 17, C["sock"], f"{tag}sock{which}", lw=3.0, cap=False)
        band = ank + (knee - ank) * .5
        K.tube(ctx, [band + (knee - ank) * .06, band - (knee - ank) * .02], 18, C["band"], f"{tag}band{which}", lw=2.6, cap=False)
        # shoe
        ctx.save()
        ctx.translate(*foot)
        ctx.rotate(fa)
        sh = [(-30, -4), (-30, -30), (-8, -40), (26, -34), (56, -24), (70, -10), (66, 2), (-26, 2)]
        K.shape(ctx, sh, C["shoe"], f"{tag}shoe{which}", tension=.8)
        K.shape(ctx, [(-31, -6), (68, -6), (66, 3), (-28, 3)], C["sole"], f"{tag}sole{which}", lw=2.8, tension=.4)
        for k in range(3):
            K.line(ctx, [(2 + 9 * k, -36 + k * 2), (8 + 9 * k, -28 + k * 2)], f"{tag}lace{which}{k}", lw=1.8)
        ctx.restore()
        return knee

    kneeB = leg(hipB, p["footB"], p["footBa"], "B")
    kneeF = leg(hipF, p["footF"], p["footFa"], "F")

    # --- shorts: two baggy legs, then the seat over them
    for which, hp, kn in (("B", hipB, kneeB), ("F", hipF, kneeF)):
        d = kn - hp
        L = np.linalg.norm(d) + 1e-6
        u, nrm = d / L, np.array([-d[1], d[0]]) / L
        end = hp + u * min(L * .78, 86)
        pts = [hp - nrm * 26 - u * 10, end - nrm * 30, end + nrm * 30, hp + nrm * 26 - u * 10]
        K.shape(ctx, pts, C["shorts"], f"{tag}sh{which}", tension=.3)
        K.line(ctx, [end - nrm * 26 - u * 3, end + nrm * 26 - u * 3], f"{tag}shc{which}", lw=1.8, k=.5)  # cuff fold
    seat = [T((-50, -14)), T((52, -16)), hip + (54, 22), hip + (2, 34), hip + (-52, 24)]
    K.shape(ctx, seat, C["shorts"], tag + "seat", tension=.5)
    K.line(ctx, [hip + (0, 6), hip + (2, 30)], tag + "fly", lw=1.8)

    # --- back arm (behind the torso)
    def arm(sh, handp, bend, hkind, hang, which, flip):
        el, wr = K.ik2(sh, handp, 92, 96, bend=bend, stretch=True)
        K.tube(ctx, [sh, el, wr], 12, C["skin"], f"{tag}arm{which}", n=8)
        d = el - sh
        K.tube(ctx, [sh - d * .05, sh + d * .38], 30, C["shirt"], f"{tag}slv{which}", lw=3.2, cap=False)
        a = _ang(wr - el) if hang is None else hang
        hand(ctx, wr, a, hkind, f"{tag}h{which}", sc=1.15, flip=flip)
        p["_wr" + which], p["_ha" + which], p["_hk" + which], p["_fl" + which] = wr, a, hkind, flip
        return el, wr

    arm(shB, np.array(p["handB"], float), p["bendB"], p["handBs"], p["handBa"], "B", 1 if p["bendB"] > 0 else -1)

    # --- torso: a soft pear in a sky-blue polo
    tor = [(-34, -130), (-44, -96), (-52, -40), (-54, -6), (0, 4), (54, -8), (50, -50), (44, -100), (34, -132), (0, -140)]
    K.shape(ctx, [T(q) for q in tor], C["shirt"], tag + "torso", tension=.9)
    K.line(ctx, [T((-50, -18)), T((0, -10)), T((52, -20))], tag + "hem", lw=1.6, k=.5)
    # neck
    ntop = neck0 + K.rot(np.array([0.0, -36 * p["neck"]]), lean * .5)
    K.tube(ctx, [neck0 + K.rot(np.array([0.0, 10]), lean), ntop], 18, C["skin"], tag + "neck")
    # collar flaps
    K.shape(ctx, [T((-26, -136)), T((-2, -128)), T((-14, -108)), T((-32, -118))], C["collar"], tag + "colL", tension=.4)
    K.shape(ctx, [T((24, -138)), T((4, -128)), T((18, -108)), T((34, -120))], C["collar"], tag + "colR", tension=.4)
    K.line(ctx, [T((0, -126)), T((0, -92))], tag + "placket", lw=1.8)
    for k, yy in enumerate((-114, -100)):
        K.dot(ctx, *T((5, yy)), 2.6, INK, tag + f"btn{k}")

    # --- head
    ctx.save()
    ctx.translate(*ntop)
    ctx.rotate(lean * .6 + p["tilt"])
    head(ctx, p, tag)
    ctx.restore()

    if p.get("front_prop") is not None:
        p["front_prop"](ctx, p)
    # --- front arm
    arm(shF, np.array(p["handF"], float), p["bendF"], p["handFs"], p["handFa"], "F", 1 if p["bendF"] < 0 else -1)
    if p["prop"] is not None:
        p["prop"](ctx, p)
    ctx.restore()


def regrip(ctx, p, which, tag="w"):
    """Redraw a hand on top of a prop it holds (call from a prop callback, local coords)."""
    hand(ctx, p["_wr" + which], p["_ha" + which], p["_hk" + which], f"{tag}h{which}", sc=1.15, flip=p["_fl" + which])


# ------------------------------------------------------------------------------------------------ the dandelion (a cel prop)
def dandelion(ctx, x, y, s=1.0, sway=0.0, tag="dd", state="flower", bob=0.0, leaves=True, stretch=1.0):
    """A single stubborn dandelion: jagged leaves at the base, a long stem, a shaggy yellow head (or a puffball)."""
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(s, s)
    leaf = hexc("#5f9d3a")
    for k, (a, L) in enumerate(((-2.9, 58), (-.25, 56), (-2.35, 46), (-.8, 48), (-1.75, 34), (-1.35, 36)) if leaves else ()):
        u = np.array([math.cos(a), math.sin(a) * .62])
        nv = np.array([-u[1], u[0]])
        if nv[1] > 0:
            nv = -nv
        up_edge, lo_edge = [], []
        for t in np.linspace(0, 1, 11):
            wv = 9 * math.sin(math.pi * min(1, t * 1.15)) * (1 - .2 * t)
            tooth = 7 * (1 if int(t * 10) % 2 else 0) * math.sin(math.pi * t)
            up_edge.append(u * L * t + nv * (wv + tooth) + u * tooth * -.6)
            lo_edge.append(u * L * t - nv * wv * .5)
        K.shape(ctx, up_edge + lo_edge[::-1], leaf, f"{tag}lf{k}", lw=2.6, tension=.5, n=4)
        K.line(ctx, [u * 4, u * L * .8], f"{tag}lv{k}", lw=1.3, k=.4)
    top = np.array([sway * 18, -92 * stretch + bob])
    stem = [(0, 0), (sway * 4 + 3, -30 * stretch), (sway * 10 - 2, -62 * stretch), tuple(top)]
    K.tube(ctx, stem, 7, hexc("#78b545"), f"{tag}st", lw=2.8)
    if state == "flower":
        cols = (hexc("#f7c51e"), hexc("#ffdf3f"))
        for ring, (R, nn, c) in enumerate(((27, 16, cols[0]), (18, 12, cols[1]))):
            pts = []
            for k in range(nn * 2):
                a = k * math.pi / nn
                rr = R if k % 2 == 0 else R * .72
                pts.append((top[0] + rr * math.cos(a), top[1] + rr * math.sin(a) * .9))
            K.shape(ctx, pts, c, f"{tag}fl{ring}", lw=2.6 if ring == 0 else 1.8, tension=.5)
        K.line(ctx, [top + (-6, 2), top + (6, 3)], f"{tag}c", lw=1.6)
    elif state == "bare":
        K.shape(ctx, K.ellipse_pts(*top, 6, 5, 7), hexc("#c9b36a"), f"{tag}bare", lw=2.0)
    else:  # puffball
        ctx.arc(top[0], top[1], 26, 0, 2 * math.pi)
        ctx.set_source_rgba(1, 1, 1, .85)
        ctx.fill()
        r = np.random.default_rng(5)
        for k in range(40):
            a = r.uniform(0, 2 * math.pi)
            q0 = top + 4 * np.array([math.cos(a), math.sin(a)])
            q1 = top + 26 * np.array([math.cos(a), math.sin(a)])
            K.line(ctx, [q0, q1], f"{tag}pf{k}", lw=.9, k=.3, ink=hexc("#9aa0a6"))
        K.shape(ctx, K.ellipse_pts(*top, 27, 27, 12), None, f"{tag}pb", lw=1.6, ink=hexc("#6d737a"))
    ctx.restore()
