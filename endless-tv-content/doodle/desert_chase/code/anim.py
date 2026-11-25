"""The 5 s code-only shot (120 frames at 24 fps), timed like a 1950s theatrical pantomime gag.

Beats: proud hold -> a tumbleweed bounces onto Pip's clean ledge -> eyes swivel -> squash -> TAKE (vibrating hold) -> whip-turn
-> glare -> wind-up (anticipation, hold) -> one smeared broom swing -> the tumbleweed sails across the canyon and lands on top of
the needle spire like a little hat -> satisfied nod -> he spots his own dust cloud -> two smeared sweeps -> whip-turn -> proud pose.
Mostly on twos; the fast actions (take, turns, smears, sweeps) are on ones."""
import math

import numpy as np

import desert as D

N = 120
FPS = 24
EV = dict(bounce=(12, 20, 27), settle=32, swivel=24, squash=34, take=36, turn1=48, land1=50, windup=56, strike=66, launch=67,
          spire=90, nod=94, notice=96, sweep=(99, 102), turn2=106, proud=108)

P0 = D.PROUD
WEED_REST = (862.0, 620.0, 30.0)
SPIRE_TOP = (1004.0, 83.0, 11.0)


def P(**kw):
    p = dict(P0)
    p.update(kw)
    return p


ALERT = P(pup_x=.62, pup_y=-.05, lid=.08, brow=.35, ear_far=-.12, ear_near=.12, head_rot=.10, smile=.3)
SQUASH = P(sx=1.08, sy=.88, crouch=.5, head_rot=.0, head_dy=10, lid=.55, brow=-.25, ear_far=-.55, ear_near=.6, smile=0, pup_x=.55,
           hn_y=-170)
TAKE = P(sx=.9, sy=1.14, hop=14, eye_s=1.42, lid=0, pup_x=.72, pup_y=-.15, brow=1.25, ear_far=-.04, ear_near=.06, ear_len=1.12,
         mouth=.75, smile=0, head_rot=.22, head_dy=-18, hn_y=-206, broom_rot=-.08, tail_curl=1.0)
GX = 652.0
GRUMP = P(x=GX, facing=-1.0, sy=.97, crouch=.25, lean=-.04, head_rot=.0, lid=.42, brow=-.6, brow_tilt=1.0, pup_x=-.55, pup_y=.25,
          smile=0, frown=1.0, two_hand=1.0, hn_x=-24, hn_y=-168, broom_rot=.55, ear_far=-.42, ear_near=.46, tail_curl=.3)
WINDUP = P(x=GX, facing=-1.0, sx=1.05, sy=.9, crouch=.65, lean=.17, head_rot=-.02, head_dy=6, lid=.5, brow=-.75, brow_tilt=1.0,
           pup_x=-.6, pup_y=.3, smile=0, frown=1.0, two_hand=1.0, hn_x=46, hn_y=-292, broom_rot=-2.45, ear_far=-.6, ear_near=.65, tail_curl=.7)
STRIKE = P(x=GX, facing=-1.0, sx=.97, sy=1.04, crouch=.35, lean=-.14, head_rot=.0, lid=.35, brow=-.5, brow_tilt=.8, pup_x=-.6, pup_y=.35,
           smile=0, frown=.6, mouth=.35, two_hand=1.0, hn_x=-58, hn_y=-128, broom_rot=1.05, ear_far=-.2, ear_near=.25, tail_curl=.2)
FOLLOW = P(x=GX, facing=-1.0, sy=1.06, crouch=.1, lean=-.17, head_rot=.32, lid=0, eye_s=1.1, brow=.55, brow_tilt=0, pup_x=-.35,
           pup_y=-.6, smile=.3, frown=0, two_hand=1.0, hn_x=-50, hn_y=-236, broom_rot=2.3, ear_far=-.15, ear_near=.2, tail_curl=.4)
WATCH = P(x=GX, facing=-1.0, sy=1.0, lean=-.07, head_rot=.42, lid=.05, eye_s=1.05, brow=.4, pup_x=-.2, pup_y=-.75, smile=.5,
          two_hand=1.0, hn_x=-40, hn_y=-222, broom_rot=2.0, ear_far=-.2, ear_near=.25)
NOD = P(x=GX, facing=-1.0, lean=.0, head_rot=.12, head_dy=8, lid=.55, brow=.1, pup_x=-.3, pup_y=0, smile=1.0, two_hand=1.0, hn_x=-30,
        hn_y=-200, broom_rot=.9)
NOTICE = P(x=GX, facing=-1.0, lean=-.04, head_rot=-.06, head_dy=4, lid=.05, brow=.45, pup_x=-.45, pup_y=.75, smile=0, frown=.4,
           two_hand=1.0, hn_x=-40, hn_y=-176, broom_rot=.6)
SWEEP_A = P(x=GX, facing=-1.0, sy=.97, crouch=.35, lean=-.08, head_rot=-.04, lid=.4, brow=-.35, brow_tilt=.6, pup_x=-.5, pup_y=.6,
            smile=0, frown=.7, two_hand=1.0, hn_x=-52, hn_y=-150, broom_rot=.42)
SWEEP_B = P(**{**SWEEP_A, "broom_rot": 1.05, "lean": -.13, "hn_x": -66})
PROUD_OVER = P(sx=.96, sy=1.05, head_rot=.2, lid=.3)


def lerp(a, b, t):
    out = dict(a)
    for k, v in a.items():
        if isinstance(v, (int, float)) and isinstance(b.get(k), (int, float)):
            out[k] = v + (b[k] - v) * t
    for k in ("two_hand", "facing"):
        out[k] = b[k] if t >= .5 else a[k]
    return out


def seg(f, f0, f1, twos=True):
    if twos:
        f = f // 2 * 2
    return D.ease((f - f0) / max(f1 - f0, 1))


def blink(f):
    return {9: .55, 10: 1.0, 11: 1.0, 12: .5, 115: .55, 116: 1.0, 117: .5}.get(f, 0.0)


def pose_at(f):
    if f < 24:
        p = dict(P0)
        if 18 <= f < 22:  # an ear twitch at the second bounce
            p.update(ear_near=.12, ear_far=-.12)
    elif f < 28:
        p = lerp(P0, ALERT, seg(f, 24, 28, twos=False))
    elif f < 34:
        p = dict(ALERT)
    elif f < 36:
        p = lerp(ALERT, SQUASH, seg(f, 33, 35, twos=False))
    elif f == 36:
        p = P(**{**TAKE, "sy": 1.2, "sx": .86, "hop": 22, "eye_s": 1.5})
    elif f < 48:
        p = dict(TAKE)
        p["x"] += 1.4 if f % 2 else -1.4    # vibrating hold
        p["sy"] += .008 if f % 2 else -.008
    elif f in (48, 49):
        p = lerp(TAKE, GRUMP, .5)
        p["facing"] = .38 if f == 48 else -.38
        p["mouth"] = 0
        p["hop"] = 10
    elif f < 52:
        p = P(**{**GRUMP, "sy": .9, "sx": 1.06, "crouch": .45}) if f < 51 else dict(GRUMP)
    elif f < 56:
        p = dict(GRUMP)
    elif f < 62:
        p = lerp(GRUMP, WINDUP, seg(f, 56, 62))
    elif f < 66:
        p = dict(WINDUP)
        p["x"] += .9 if f % 2 else -.9
    elif f == 66:
        p = dict(STRIKE)
    elif f < 80:
        p = dict(FOLLOW) if f > 67 else P(**{**FOLLOW, "lean": -.22, "sy": 1.1})
    elif f < 90:
        p = lerp(FOLLOW, WATCH, seg(f, 80, 90))
    elif f < 94:
        p = dict(WATCH)
    elif f < 96:
        p = lerp(WATCH, NOD, .7 if f == 94 else 1.0)
    elif f < 99:
        p = lerp(NOD, NOTICE, .6 if f == 96 else 1.0)
    elif f < 106:
        p = {99: SWEEP_A, 100: SWEEP_B, 101: SWEEP_B, 102: SWEEP_A, 103: SWEEP_A, 104: SWEEP_B, 105: SWEEP_B}[f]
        p = dict(p)
    elif f in (106, 107):
        p = lerp(SWEEP_B, P0, .5)
        p["facing"] = -.38 if f == 106 else .38
        p["x"] = GX + (612 - GX) * (.4 if f == 106 else .8)
        p["hop"] = 10
        p["two_hand"] = 0.0
    elif f < 110:
        p = dict(PROUD_OVER)
    elif f < 112:
        p = lerp(PROUD_OVER, P0, .6)
    else:
        p = dict(P0)
    p["blink"] = max(p.get("blink", 0), blink(f))
    return p


def weed_at(f):
    """(x, y, r, rot) or None."""
    if f < 6:
        return None
    x0 = 1360.0
    xr, yr, r = WEED_REST
    if f < 32:
        u = (f - 6) / 26
        x = x0 + (xr - x0) * (1 - (1 - u) ** 1.6)
        # bounces: contacts at f12, f20, f27 then a roll to rest
        hops = ((6, 12, 70), (12, 20, 110), (20, 27, 60), (27, 32, 16))
        y = yr
        for a, b, h in hops:
            if a <= f < b:
                s = (f - a) / (b - a)
                y = yr - h * 4 * s * (1 - s) - (40 * (1 - s) if a == 6 else 0)
        return x, y, r, -(x - x0) / r
    if f < 67:
        wob = 2.0 * math.sin((f - 32) * 1.3) * math.exp(-(f - 32) * .25) if f < 44 else 0.0
        return xr + wob, yr, r, -(xr - x0) / r + wob / r
    if f < 90:
        t = (f - 67 + 1) / (90 - 67 + 1)
        c = (962.0, -300.0)
        x = (1 - t) ** 2 * xr + 2 * t * (1 - t) * c[0] + t * t * SPIRE_TOP[0]
        y = (1 - t) ** 2 * yr + 2 * t * (1 - t) * c[1] + t * t * SPIRE_TOP[1]
        rr = r + (SPIRE_TOP[2] - r) * t ** .65
        return x, y, rr, -(xr - x0) / r - 9 * t
    sq = 1.0
    return SPIRE_TOP[0], SPIRE_TOP[1] + (1.5 if f in (90, 91) else 0), SPIRE_TOP[2] * sq, -(xr - x0) / r - 9


def fx_back(f):
    """Things drawn under Pip: the impact cloud lingers behind him."""
    def draw(ctx):
        pass
    return draw


def fx_front(f, p):
    def draw(ctx):
        w = weed_at(f)
        if w is not None:
            D.tumbleweed(ctx, *w)
        # bounce puffs and the settle puff
        for k, fb in enumerate(EV["bounce"]):
            age = f - fb
            if 0 <= age < 9:
                wx = weed_at(fb)[0]
                D.dust(ctx, wx, WEED_REST[1] + 26, 11, 50 + k, grow=1 + age * .12, drift=(age * 3, -age * .6), fade=1 - age / 9, n=5)
        age = f - EV["settle"]
        if 0 <= age < 18:
            D.dust(ctx, WEED_REST[0] - 6, WEED_REST[1] + 22, 15, 60, grow=1 + age * .06, drift=(-age * .8, -age * .5), fade=1 - age / 18, n=7)
        # turn kick-ups at his feet
        for ft, side in ((EV["turn1"], 1), (EV["turn2"], -1)):
            age = f - ft
            if 0 <= age < 10:
                D.dust(ctx, 640 + side * 20, 648, 13, 70 + ft, grow=1 + age * .1, drift=(side * age * 2, -age), fade=1 - age / 10, n=6)
        # the impact cloud: bursts on the swing, hangs in the air, then gets swept away
        age = f - EV["strike"]
        if age >= 0 and f < 106:
            if f < 99:
                g = .55 + .75 * (1 - math.exp(-age * .45)) + .004 * age
                D.dust(ctx, WEED_REST[0] - 14, WEED_REST[1] + 2, 38, 80, grow=g, drift=(-age * .35, -age * .25), fade=1.0, n=11)
            else:
                k = f - 99
                for j in range(5):  # the sweep breaks it into small puffs that fly off right and vanish
                    D.dust(ctx, WEED_REST[0] + 10 + j * 16 + k * (18 + 6 * j), WEED_REST[1] - 10 + (j % 3) * 14 - k * 3, 13 - j, 90 + j,
                           grow=max(.05, 1 - k / 6.5), fade=1 - k / 7, n=4)
        if EV["strike"] <= f <= EV["strike"] + 3:  # impact flash: a few dry-brush strike lines
            k = f - EV["strike"]
            for j in range(7):
                a = -1.4 + j * .32
                r0, r1 = 40 + 14 * k, 70 + 22 * k
                cx, cy = WEED_REST[0], WEED_REST[1]
                D.tline(ctx, [(cx + r0 * math.cos(a), cy + r0 * math.sin(a)), (cx + r1 * math.cos(a), cy + r1 * math.sin(a))], 3.2 - .7 * k,
                        col=D.INK, t0=.2, t1=.9)
        age = f - EV["spire"]
        if 0 <= age < 10:  # tiny puff on the spire top
            D.dust(ctx, SPIRE_TOP[0], SPIRE_TOP[1] + 7, 6, 99, grow=1 + age * .12, drift=(0, -age * .4), fade=1 - age / 10, n=5)
        # smears
        M0 = D.body_matrix(p)
        if f == EV["strike"]:
            D.smear_arc(ctx, M0, p, WINDUP["broom_rot"] + .3, p["broom_rot"] - .05, n=8)
        if f == EV["launch"]:
            D.smear_arc(ctx, M0, p, STRIKE["broom_rot"], p["broom_rot"] - .1, n=7)
        if f in (100, 104):
            D.smear_arc(ctx, M0, p, SWEEP_A["broom_rot"], SWEEP_B["broom_rot"] - .05, n=6)
        if f == 102:
            D.smear_arc(ctx, M0, p, SWEEP_B["broom_rot"], SWEEP_A["broom_rot"] + .05, n=6)
        if f in (48, 49, 106, 107):  # whip-turn: dry-brush speed streaks across his body
            x = p["x"]
            D.speed_lines(ctx, x - 90, x + 110, np.linspace(430, 620, 9), D.alpha(D.PC["shell"], .9), 3.0, seed=f)
            D.speed_lines(ctx, x - 70, x + 90, np.linspace(380, 600, 6), D.alpha(D.INK, .8), 1.4, seed=f + 9)
    return draw


def frame(i):
    p = pose_at(i)
    w = weed_at(i)

    def front(ctx):
        fx_front(i, p)(ctx)

    # when the tumbleweed sits on the spire it belongs to the background plane (behind nothing of Pip's), fine to draw last
    return D.render(p, front, i)


def key():
    return D.key()
