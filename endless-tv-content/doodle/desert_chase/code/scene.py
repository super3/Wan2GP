"""Compose a frame of the short: painted background for a camera + character cel layers (each optionally dusted or muddied) +
effect cels, then the film finish.  Returns uint8 RGB 1280x720; key() trims to 1280x704."""
import cairo
import cv2
import numpy as np

import cast
import desert as D
import world

W, H, SS = D.W, D.H, D.SS


def cel(draw, cam):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W * SS, H * SS)
    ctx = cairo.Context(surf)
    ctx.scale(SS, SS)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    world.apply_cam(ctx, cam, 0.0)   # characters are placed in screen space; only the zoom applies
    z = world.CAMS[cam]["zoom"] if isinstance(cam, str) else cam["zoom"]
    lw, li = D.LW, D.LI
    D.LW, D.LI = 3.4 / z ** .55, 2.1 / z ** .55   # close-ups: lines get a little heavier on screen, not 1.85x heavier
    try:
        draw(ctx)
    finally:
        D.LW, D.LI = lw, li
    a = np.ndarray((H * SS, W * SS, 4), np.uint8, surf.get_data())
    a = cv2.resize(a.astype(np.float32) / 255, (W, H), interpolation=cv2.INTER_AREA)
    return a[:, :, 2::-1].copy(), a[:, :, 3:4].copy()


def compose(cam, bg=None, layers=(), seed=0, grain=.010):
    """layers: list of dicts {draw: fn(ctx), treat: None|'dust'|'mud', amount: float, haze: float (storm tint)}."""
    img = world.background(cam, **(bg or {})).copy()
    for L in layers:
        rgb, a = cel(L["draw"], cam)
        rgb, a = cast.treat(rgb, a, L.get("treat"), L.get("amount", 1.0), L.get("seed", 0))
        hz = L.get("haze", 0.0)
        if hz > 0:
            rgb = rgb * (1 - hz) + world.STORMC * a * hz
        img = rgb + img * (1 - a)
    return D.to8(D.finish(img, seed, grain))


def key(img):
    return img[8:712]
