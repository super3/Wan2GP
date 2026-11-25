import sys, time, cv2
import desert as D
t = time.time()
k = D.key()
print(k.shape, time.time() - t)
cv2.imwrite(sys.argv[1], k[:, :, ::-1])
