import keys_short as KS, cv2, numpy as np
out = []
for k, (hipy, hipx, lean, ff, fb) in enumerate(((112, -40, .32, (-10, 0), (50, 0)), (100, -50, .45, (-30, 0), (30, 4)))):
    cam = (780, 452, 1.3)
    f = KS.Frame(cam)
    f.dd()
    f.wendell(x=700, y=640, lean=lean, tilt=-.1, hipy=hipy, hipx=hipx, footF=ff, footB=fb, handF=(40, -128), handB=(84, -132),
              handFs="fist", handBs="fist", bendF=-1, bendB=1, squint=.55, browang=1.0, brow=-.2, mouth="frown", look=(.95, .5))
    out.append(KS.comp(KS.bg(cam), f.rgba()))
cv2.imwrite('s02test.jpg', (np.vstack([cv2.resize(o, (640, 352)) for o in out]) * 255).astype(np.uint8)[:, :, ::-1])
