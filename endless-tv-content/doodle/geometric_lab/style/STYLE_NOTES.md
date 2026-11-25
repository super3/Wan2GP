# geometric_lab: style notes

Reference: the late-90s Cartoon Network look of *Dexter's Laboratory* (Tartakovsky, 1996-), which draws on UPA's flat,
graphic 1950s design and uses "controlled" limited animation. I borrowed only the general traits. The character, premise and
setting are my own.

## Character and premise
**Mr. Ferdinand Hoist** is a tiny, ramrod-stiff elderly elevator operator. He has run Car No. 7 of a grand 1950s department
store for fifty years and treats every trip like a naval launch.
- Design: a boxy bald head under a red pillbox cap, a block moustache wider than his face, white slab brows, a pink wedge
  nose, a barrel-chested red tunic with gold epaulettes and buttons, and stick legs in navy with pointy black shoes.
- Shot: from the lobby, framed by the brass doorway of his elevator. He yanks the brass car-switch to ring the departure bell,
  raises a finger to make his announcement, snaps off a salute, then settles back to attention.

## Traits observed, and how I implemented them
1. **Geometric construction.** Every shape is a polygon with a set radius on each corner (`rpoly`), so corners can be
   rounded or sharp per vertex. The head is a box, the body a bell, the nose a wedge, the brows slabs, and the moustache has
   crisp notches. There are no freehand curves.
2. **Bold, even black ink plus a heavier outer silhouette.** Inner lines are a uniform 4.4 px. The whole cel layer's alpha
   is dilated by about 1.9 px and filled with ink underneath, so the outer contour reads thicker, the way a cel's outline does.
3. **Flat, saturated cel colour with no rendering.** The red, gold and skin are perfectly flat. The background is
   desaturated by 14% so the character pops.
4. **Painted card backgrounds (UPA influence).** Background lines are drawn in darker tones of each fill, never black.
   - The lobby has a low-contrast harlequin pattern, plus "atomic" sparkles whose fill is printed off-register from its line.
   - Airbrushed light falls off inside the cab, and the sconces throw painted fans of light up the wall.
   - Gouache mottling and paper tooth are applied to the background only.
5. **Strong silhouette and cartoon proportions.** The head plus cap is about half his height. The moustache ends, the cap,
   the epaulettes and the akimbo elbow all break the outline. The door frames him as a frame within a frame.
6. **Small dot eyes; brows and moustache do the acting.** Expressions come from brow lift and tilt, eye squints and blinks,
   a jaw that stretches the head shape, and a moustache that lifts and bristles.
7. **Snappy "controlled" timing.**
   - Everything is on twos at 12 frames per beat (120 bpm), with dead-still holds.
   - Each action runs anticipation → 2-frame smear (tapered speed arcs) → overshoot → settle, with squash and stretch on
     the body.
   - The silence before the yank comes from the "moment of nothing" idea.
8. **Mid-century jazz score with Mickey-Mousing (music hits matched to the action).** The cue uses pizzicato bass, brushes,
   ride cymbal and vibes, with a muted-trumpet "ta-ta-TAAA" landing on the salute.
   - SFX: lever ka-CHUNK, bell ding, whoosh and snap, and a spring "boing".
   - The voice is a formant-synthesised gibberish voice with a reedy old-man quaver. The band ducks under it.

## What I'd improve next
- Add a 3/4 head turn and a second body angle, so the acting isn't always square to camera.
- Push the hands further: real finger overlap on the grip, and a crisper salute silhouette (the raised arm needs cartoon
  stretch to reach the brow).
- Build a mouth chart from real phonemes, and give the voice synth formant glides instead of one formant set per syllable.
- Add a little painted dry-brush texture on the brass, and a richer layout perspective (a slightly tilted camera).
