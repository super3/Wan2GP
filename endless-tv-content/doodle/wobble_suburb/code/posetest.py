import cv2, numpy as np, chars as CH, ink as K, paint, key
bg = np.ones((704,1280,3),np.float32)*.92
K.set_boil(0,.5)
surf, ctx = K.canvas()
P = [dict(x=170, mouth="open", jaw=.6, wide=.3, look=(.8,0), brow=.4, handB=(140,-330), handBs="point", bendB=1, handF=(-70,-250), handFs="fist", browang=.6),
     dict(x=470, mouth="yell", jaw=1.0, wide=1.0, tongue=1, eyesize=1.45, eye=1, look=(0,0), brow=1.2, hair_up=1, sy=1.08, sx=.94,
          handF=(-120,-420), handFs="open", handB=(130,-420), handBs="open", bendF=-1, bendB=1),
     dict(x=780, mouth="grit", squint=.6, browang=1, brow=-.2, lean=.35, hipy=30, handF=(90,-170), handB=(120,-175), handFs="grip", handBs="grip", footF=(-80,0), footB=(70,0)),
     dict(x=1080, mouth="closed", smile=1, eye=.15, brow=.6, look=(0,0), handF=(-60,-430), handFs="open", handB=(64,-250)),]
for p in P:
    CH.wendell(ctx, CH.pose(y=650, s=.8, **p))
img = paint.over(bg, paint.to_rgba(surf))
cv2.imwrite('poses.png', key.to8(img)[:,:,::-1])
