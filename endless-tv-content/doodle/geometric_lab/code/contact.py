"""contact.py OUT.jpg [frames...]: render a contact sheet of chosen frames (debug)."""
import sys, cv2, numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user-Wan2GP/4e1e6010-b786-5f93-b6fd-87ee44cba9e7/scratchpad/styles/geometric_lab')
import geo
out = sys.argv[1]
fr = [int(x) for x in sys.argv[2:]] or [0, 20, 24, 26, 30, 36, 50, 66, 80, 88, 90, 104, 106, 110, 119]
tiles = []
for f in fr:
    im = geo.frame(f)[:, :, ::-1].copy()
    im = cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA)
    cv2.putText(im, str(f), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    tiles.append(im)
while len(tiles) % 3:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
