"""The style-study key frame: Wendell squares off with the one dandelion on his dad's perfect lawn."""
import sys

import cv2
import numpy as np

import chars as CH
import ink as K
import paint

KEY_POSE = dict(x=470, y=628, s=.9, face=1, lean=-.04, handF=(-66, -248), handB=(66, -250), handFs="hip", handBs="hip",
                footF=(-48, 0), footB=(56, 4), footFa=-.05, footBa=.05, eye=.92, look=(.75, .25), brow=.15, browang=.55,
                squint=.25, mouth="closed", smile=-.1)
DANDE = dict(x=905, y=642, s=1.45)


def cels(i=0, pose=None, dd=None, boil_a=.5):
    """Draw the cel layer (characters + props) onto a transparent canvas; returns premultiplied RGBA."""
    K.set_boil(i, boil_a)
    surf, ctx = K.canvas()
    CH.dandelion(ctx, **(dd or DANDE))
    CH.wendell(ctx, CH.pose(**(pose or KEY_POSE)))
    return paint.to_rgba(surf)


def compose(bg, cel):
    img = paint.over(bg, cel)
    return np.clip(img, 0, 1)


def to8(img):
    return (np.clip(img, 0, 1) * 255 + .5).astype(np.uint8)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "key.png"
    bg = paint.backyard()
    img = compose(bg, cels())
    cv2.imwrite(out, to8(img)[:, :, ::-1])
    face = cv2.resize(to8(img)[60:360, 330:630], (600, 600), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out.replace(".png", "_face.png"), face[:, :, ::-1])
