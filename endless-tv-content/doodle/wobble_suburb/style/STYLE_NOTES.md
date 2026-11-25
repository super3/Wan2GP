# wobble_suburb: style notes

Reference style: late-90s hand-inked TV slapstick (*Ed, Edd n Eddy* era, a.k.a. Cartoon). I studied the general traits only; the characters, premise and setting below are original.

## Traits observed and how I implemented them
- **Boiling ink line.** Each drawing was inked in three passes, so held poses still shimmer. `ink.jitter` re-traces every control point with a seeded jitter. `BOIL["i"]` cycles 0-1-2 on twos, and the pen's width phase changes with each tracing. Key frames use a small fixed jitter, so they still look hand-traced.
- **Heavy, nearly even black outline.** The line breathes a little but never tapers like brush calligraphy. `ink_poly` draws chunked round-capped strokes whose width varies about ±20% around 4.3 px. Detail lines (knuckles, freckle creases, laces) use about 1.8 px. Limbs are "tube" strokes: an ink stroke with a narrower fill stroke on top.
- **Gangly, lopsided kid anatomy.** The head is about a third of the body. Necks and arms are noodle-thin and long enough to reach the knees, the feet are clown-sized, and in 3/4 view the two eyes differ in size. The rig uses two-bone IK that stretches like rubber hose when a target is out of reach.
- **Huge mouths, big teeth, tongues.** At rest the mouth is a long wavy line from cheek to snout, with buck teeth resting on the lip. When open, the lower jaw drops the whole chin outline (`jaw`). On a yell, the back corner of the mouth travels toward the ear (`wide`), showing a row of blocky teeth and a big tongue.
- **Flat cel colour, almost no shading.** One flat fill per area. The only "shadow" is a soft painted ellipse under the feet.
- **Lush painted suburb behind inked cels.** Backgrounds carry no black line. `paint.py` lays in flat shapes, then:
  - scrubs each one with hundreds of translucent, value- and hue-jittered directional brush strokes;
  - dabs the foliage, darker on the shadow side;
  - adds thin darker-local-colour brush lines for siding and fence gaps;
  - finishes with a hand-wobble edge warp, pigment pooling at edges, granulation and paper tooth.
- **Slightly-off hand layout.** The fence top and bottom are a hair off level, the planks lean and wobble, and the house is a little crooked.
- **Elastic takes.** A stretched smear drawing comes first. Then eyes at 1.5x, hair standing up, jaw on the floor, arms flung out, a trembling hold and pop lines. A squashed "gulp" leads back into the pose. Timing is pose-to-pose on twos, with long holds.
- **Jump-blues, rockabilly and boogie score.** `snd.py` synthesises:
  - a swung 150-bpm boogie on honky-tonk piano (two detuned strings);
  - a plucked upright bass, brushes and ride;
  - a harmon-muted "wah" trumpet;
  - formant-synth kid gibberish, and a boing, slide whistle and gulp.

## Original character and premise
**Wendell Pruitt**, a 10-year-old beanpole:
- scruffy tangerine mop of hair, a pickle nose, jug ears, freckles and two big buck teeth;
- sky-blue polo with a white collar, khaki knee shorts, tall white socks with red bands, and huge red sneakers.

His dad's lawn is perfect, striped and spotless, except for **one dandelion**, and Wendell declares war on it.

Key frame: a Western-style standoff. H3 line: "This lawn ain't big enough for the both of us, flower."

## Files
- `key_1280x704.png` / `.jpg`: the key frame.
- `h3_prompt.txt`: the H3 prompt.
- `code_5s.mp4`: the code-only shot. Wendell jabs a finger and jabbers, the dandelion boings, he does an extreme take, then gulps back into his pose.
- Code in `scratchpad/styles/wobble_suburb/`: `ink.py`, `paint.py`, `chars.py`, `key.py`, `anim.py`, `snd.py` and `make.py`.

## Next improvements
- Add hand-drawn smear and multiple-image inbetweens on the fastest moves, and give the boil a little more width wobble on the outline only.
- Add a darker, grimier late-afternoon background variant. Push perspective wonkiness further on the houses, and add painted cast shadows from props.
- Give the hands more drawings (spread, clutch, wave), and add a 3/4-back and profile head for turnarounds.
- Improve the voice synthesis (consonant transitions), and add a real walking-bass variation and a sax for the score.
