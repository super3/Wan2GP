import sys, time, cv2
sys.path.insert(0, '/tmp/claude-0/-home-user-Wan2GP/4e1e6010-b786-5f93-b6fd-87ee44cba9e7/scratchpad/styles/geometric_lab')
import geo
t=time.time()
img = geo.key_frame()
print('t', time.time()-t)
out = sys.argv[1] if len(sys.argv)>1 else 'k.png'
cv2.imwrite(out, img[:, :, ::-1])
# crops
cv2.imwrite(out.replace('.png','_face.png'), cv2.resize(img[150:420, 480:790, ::-1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
cv2.imwrite(out.replace('.png','_hands.png'), img[340:620, 400:800, ::-1])
