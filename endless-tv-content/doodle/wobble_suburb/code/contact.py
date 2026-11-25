"""contact.py MODULE OUT.jpg [frames...]: tile chosen frames of an animation module."""
import importlib, sys
import cv2, numpy as np
m = importlib.import_module(sys.argv[1])
fr = [int(x) for x in sys.argv[3:]] or list(np.linspace(0, m.N - 1, 12).astype(int))
ims = [cv2.resize(m.frame(i), (427, 240), interpolation=cv2.INTER_AREA) for i in fr]
for im, i in zip(ims, fr):
    cv2.putText(im, str(i), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
while len(ims) % 3:
    ims.append(np.zeros_like(ims[0]))
rows = [np.hstack(ims[k:k + 3]) for k in range(0, len(ims), 3)]
cv2.imwrite(sys.argv[2], np.vstack(rows)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
