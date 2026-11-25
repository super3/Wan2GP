"""sheet.py OUT.jpg f1 f2 ... : render the given frames and tile them (4 per row, half size) with frame numbers drawn as tick marks."""
import sys
from multiprocessing import Pool

import cv2
import numpy as np


def one(i):
    import anim
    return i, anim.frame(i)


if __name__ == "__main__":
    out = sys.argv[1]
    fr = [int(a) for a in sys.argv[2:]]
    with Pool(2) as pool:
        res = dict(pool.map(one, fr))
    tiles = []
    for i in fr:
        im = cv2.resize(res[i], (640, 360), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()
        cv2.putText(im, str(i), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 4)
        cv2.putText(im, str(i), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 2)
        tiles.append(im)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
