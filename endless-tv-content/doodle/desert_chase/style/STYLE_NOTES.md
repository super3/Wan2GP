# desert_chase: style notes

**Target:** the look of the 1950s Warner theatrical desert-chase shorts: Chuck Jones's direction and Maurice Noble's layouts. Everything here is original: the characters, the premise, the setting details and the music.

## Traits observed, and how I implemented them
- **Noble's graphic desert.** Tall flat-topped buttes and a needle spire are built from straight-edged flat planes. Each has a lit face, a hard-edged violet shadow face, a darker cap rock and horizontal strata. The polygons are angular, not wobbly (`butte()`).
- **Off-register line work.** Noble used it to suggest heat shimmer. I draw thin coloured contours a few pixels off the colour shapes, with breaks, only on some edges (`offreg()`). The background has no black outlines.
- **Painted, not flat-digital, backgrounds.** Dry-brush strokes (broken dabs of varying width) and stipple, plus a static gouache mottle and paper tooth over the whole card (`drybrush()`, `background()`).
- **Huge sky, low horizon, vertical depth.** A turquoise-to-cream sky with long tapered streak clouds. The road winds back and forth along a hazy canyon floor far below the ledge. The tiny scale of the road, the shrubs and the floor mesas, plus aerial haze at the ledge edge, sell the drop.
- **Limited warm palette, one cool character.** The palette is ochre, orange, salmon, rose, mauve and violet. Pip's slate-blue shell is the only cool mass, so his silhouette reads instantly against the pale sky.
- **Cel-flat character over a painted card.** The character is a separate flat cel layer composited over the background, with no shading and only a flat cast shadow. Ink lines are near-black and vary in weight: heavier on the shadow side and on the silhouette (3.4 px), lighter inside (2.1 px), with tapered interior strokes (`closed_widths()`, `tline()`).
- **Jones drawing conventions.** Tall touching eyes set high on the face, a half-lidded smug look, floating brows, noodle limbs, big feet and a chest-out line of action in the proud pose.
- **Pantomime timing.** The shot runs: proud hold, eye swivel, squash, extreme take on ones with a vibrating hold, whip-turn, glare, wind-up and hold, then a single dry-brush broom smear. The hit sends the tumbleweed sailing across the canyon onto the spire, and the impact dust cloud hangs in the air. He gives two smeared sweeps, whip-turns back and lands in an overshoot settle. Holds are perfectly still, as cels were, with no line boil. Slow moves are on twos.
- **Sound that hits the action, in the style of Carl Stalling's scores.** A bassoon strut, xylophone notes on each bounce, a brass stab with cymbal and slide whistle on the take, a string tremolo on the hold, a clarinet run and trill on the wind-up, a whack, slide whistles for the flight, xylophone runs on the sweeps and a brass "ta-da" button. Pip's grunts are wordless and formant-synthesised.

## Character and premise
**Pip** is a fussy little armadillo with a slate-blue banded shell, an armour plate running from crown to snout, a red polka-dot neckerchief and a straw broom. He keeps his patch of a high desert ledge spotless in the dustiest place on earth. In the code shot a tumbleweed bounces onto his clean spot. He bats it across the canyon, where it lands on the needle spire like a little hat, and then he sweeps away the dust he made himself. There are no text, signs or catalogue props, and no chase animals from the source shorts.

## What I'd improve next
- Real in-between drawings for the turns. They are currently a squashed, mirrored body with speed streaks.
- Hand poses with open fingers for the take, and a foot lift on the follow-through.
- Richer canyon-floor painting: more floor forms, cast cloud shadows, and a visible cliff face somewhere for an even stronger vertical drop.
- A more voice-like vocal synthesis for the grunts.
