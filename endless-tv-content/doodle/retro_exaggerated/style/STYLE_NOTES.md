# retro_exaggerated: style notes

Reference: the art style of *Johnny Bravo* (Van Partible, Cartoon Network, 1997-), borrowed for general traits only. Partible set out
to evoke early Hanna-Barbera; season 1 had rounder, softer forms and pastel-bright colour, later seasons flat digital colour.

## Traits observed, and how each was implemented
- **Extreme top-heavy proportions.** One giant rounded upper mass (wool hump plus chest), a V-taper to a tiny waist, stick legs, very
  small feet, and a head about a third the width of the torso. `MASS` outline, `hem()`, `draw_legs()` in `bertram.py`.
- **Thick, even black outer contour with thinner interior lines.** Every part is stroked at 3.6 px. The cel's alpha is then
  dilated (distance transform) to add a uniform ~2.6 px ink ring around the whole silhouette, so the outer line is about twice
  as heavy as the inner detail (`cel_layer`). Small details use tapered brush lines (`taper_line`).
- **Flat cel colour, no gradients on the character.** One flat paint per region, no shading, a bright pastel palette (rust wool,
  camel, peach, navy, safety orange, signal red). The only texture is the cel's own outline.
- **Rounded shape language with scalloped edges.** Wool, beard, fringe, clouds and hedges use a scallop generator (round bumps,
  sharp notches, slight skew) (`scallop`), applied only above the wool hem so the waist stays smooth.
- **Hanna-Barbera-style painted background.** Flat side-on (elevation) layers: sky card, rolling hills, a mid-century ranch house,
  a flat-roofed schoolhouse with ribbon windows, lollipop trees with fan-shaped bloom strokes, hedges, a zebra crossing. Foliage is
  sponge-daubed. A thin dark line is laid over each colour shape slightly off-register, with breaks, depth-ordered per layer
  (`BGPainter.overlay_lines`). A light gouache/paper texture goes on the background only, so the cel reads as paint on a card.
- **Retro-modern pattern accents, limited detail.** A breeze-block diamond screen wall, a starburst on the gable, stone base
  shapes. Detail is kept sparse so the character pops.
- **Cartoony posing with big silhouette changes.** Akimbo arm, a planted sign pole, and a chin-up smug face in the neutral pose.
  The action goes squash, then smear, then a raised "HALT" pose with the chest swelling sideways.
- **Snappy limited animation.** Shot on twos, except the snap (on ones). Holds at the start and end, a 4-frame anticipation
  squash, a 2-frame smear (stretched disc plus speed lines), an overshoot, a "take" where the puffed chest pops the cap off, accent
  lines (emanata) on impacts, and a squash on the pole thunk. Frame 119 returns to the key pose.
- **Sound in the era's spirit.** A jaunty cue scored to the action: pizzicato walking bass (Karplus-Strong), brushes, vibraphone,
  a muted-brass "wah" stab, a xylophone run. A formant-synthesised pompous baritone speaks gibberish synced to the mouth chart
  (`SYL`). Foley is slide whistles, a pop, a jaw-harp boing, a woodblock and a thud.

## Character and premise
**Bertram Bigsby**, a vain, enormously top-heavy bison and the neighbourhood's school crossing guard. He is sure the traffic stops
because it admires him. He has a scalloped wool hump, ivory horns, a cap far too small for him, an orange safety sash with a badge,
navy trousers on pin legs and tiny black shoes. He carries a round "lollipop" stop sign with a hand pictogram (no text). Setting:
a pastel 1960s suburban corner across from a mid-century schoolhouse.

## What I'd improve next
- More acting variety in the talk (hand gestures with the free arm, eyebrow shapes) and a proper second mouth chart for the voice
  instead of four vowel shapes.
- The background is a single static card. A second overlay level (a foreground hydrant or lamp) would add depth, and pans would
  need a wider painted card.
- The voice is plausible gibberish but robotic. Glottal-pulse modelling and formant glides between syllables would make it warmer.
- A 4:3-safe composition variant, since the original show aired at 4:3.
