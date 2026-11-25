"""make_sheet.py: render a contact sheet of selected frames (for checking timing and poses)."""
import sys

import cv2
import numpy as np

import draw

frames = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(0, 120, 8))
tiles = []
for i in frames:
    f = cv2.resize(draw.frame(i), (426, 240), interpolation=cv2.INTER_AREA)
    cv2.putText(f, str(i), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
    tiles.append(f)
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)]
cv2.imwrite(sys.argv[1], np.vstack(rows)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
