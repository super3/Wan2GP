# Clean Sweep
*A Pip the Armadillo Cartoon*. Original pantomime short in a classic 1950s theatrical cartoon style. Runtime 1:58 (2841 frames at 24 fps), including a 3 s title card and a 2 s end card.

## Logline
A fussy little armadillo keeps the tidiest ledge in the dustiest desert on earth. Then a jolly javelina who lives for dust baths moves in next door.

## Characters
**Pip** is a small, upright armadillo. He has a slate-blue banded shell, a blue armour plate from his crown down his long pink snout, tall pointed ears, a red polka-dot neckerchief, thin pink arms and big feet. He carries a straw broom everywhere. He is proud, fussy and precise, and does slow burns.
- Fixed voice: "Pip's voice: a fussy, slightly nasal, middle-aged male tenor, crisp and precise"

**Bramble** is a big, round javelina on four short legs. He has a grey-violet bristly body, a spiky dark crest along his back, a cream collar stripe over his shoulders, a pink pig-snout disc with one small white tusk, little pointed ears and tiny black hooves. He is cheerful, oblivious and generous, and loves dust above all things.
- Fixed voice: "Bramble's voice: a big, warm, booming male bass, slow, jolly and good-natured"

**Setting.** A high desert ledge in the style of Maurice Noble's painted deserts:
- Pip's burrow at the foot of a tall orange rock. Next to it are a little striped rug and a barrel cactus.
- A neat row of white pebbles marks the property line.
- Bramble's dusty wallow and gnarled scratching post sit on the other side of the line.
- Needle spires and flat-topped mesas rise behind, and a road winds along the canyon floor far below.

## Beat sheet
1. **Premise (0:03–0:23).** Pip sweeps his ledge spotless: "Ahem. Spotless." Next door, Bramble flops into his daily dust bath.
2. **The problem (0:23–0:41).** Bramble's dust cloud drifts over and coats Pip. In close-up, Pip does a slow burn: "That... does it."
3. **Attempt 1 (0:41–0:54).** Pip sweeps the dust back over the line. Bramble thinks it is a gift ("Ooh! Thanks, neighbor!") and rolls harder, and the cloud blows straight back over Pip.
4. **Attempt 2 (0:54–1:04).** Pip washes Bramble with his canteen. Dust plus water makes mud ("Ooh, mud! Even better!"). Bramble shakes like a wet dog, and Pip is splattered.
5. **Complication (1:04–1:26).** A gigantic dust storm rises over the canyon. Pip does a huge take; Bramble is thrilled ("Here comes the big one!"). Pip rolls into a ball, and Bramble plants his hooves and grins into the storm ("Wheee!").
6. **Payoff (1:26–1:48).** The whole desert is buried in dust, except a clean wedge in Bramble's lee, right where Pip is. Pip: "My ledge... it's spotless!" He brushes his giant fluffy neighbour clean, and Bramble's hind leg thumps with bliss: "Ooh... right there!" A deal is struck.
7. **Button (1:48–1:56).** At sunset, Pip proudly shows off his ledge with his new friend, who now wears a matching neckerchief: "Ahem. Spotless." Bramble sneezes, and Pip is coated in dust again, deadpan.

## Shot list
All shots use a static camera. Keys are in `keys/`, and every frame count is one of H3's lengths.

| Shot | Frames (s) | Start → end key | Continues | Camera | Action | Dialogue |
|---|---|---|---|---|---|---|
| s01 | 243 (10.1) | s01_a → s01_b | – | Pip's home, medium-wide | Last broom strokes, then the proud pose | Pip: "Ahem. Spotless." |
| s02 | 243 (10.1) | s02_a → s02_b | – | Bramble's wallow, wide | Bramble trots in and flops on his back in a dust bath | Bramble: "Ahhh. Bath time!" |
| s03 | 192 (8.0) | s01_b → s03_b | – | Pip's home, medium-wide | A dust cloud engulfs Pip; he is left coated, blinks and coughs | (none) |
| s04 | 243 (10.1) | s04_a → s04_b | – | Close-up on Pip | Slow burn, eye twitch, broom raised | Pip: "That... does it." |
| s05 | 294 (12.25) | s05_a → s05_b | – | Two-shot at the pebble line | Pip sweeps the dust back; Bramble rolls in it; the cloud blows back over Pip | Bramble: "Ooh! Thanks, neighbor!" |
| s06 | 243 (10.1) | s06_a → s06_b | – | Two-shot | Canteen bath turns to mud; wet-dog shake splatters Pip | Bramble: "Ooh, mud! Even better!" |
| s07 | 294 (12.25) | s06_b → s07_b | yes | Two-shot | The dust storm rises; Pip's take; Bramble beams | Bramble: "Here comes the big one!" |
| s08 | 243 (10.1) | s07_b → s08_b | yes | Two-shot | Pip rolls into a ball; Bramble braces; everything goes amber | Bramble: "Wheee!" |
| s09 | 294 (12.25) | s09_a → s09_b | – | Two-shot, aftermath | Pip uncurls in the clean wedge, picks up his broom and is amazed | Pip: "My ledge... it's spotless!" |
| s10 | 243 (10.1) | s09_b → s10_b | yes | Two-shot, aftermath | Pip brushes Bramble clean; Bramble's leg thumps | Bramble: "Ooh... right there!" |
| s11 | 192 (8.0) | s11_a → s11_b | – | Pip's home at sunset | The proud line, then a giant sneeze; Pip is dusty and deadpan | Pip: "Ahem. Spotless." |

Each shot has only one speaker, and every line is 5 words or fewer. Each prompt asks for a short pause before and after the line. The full H3 prompts are in `shots.json`; each one sets `non_diegetic_music: N/A`.

## Score (`music_bed.wav`, 118.4 s, 48 kHz stereo)
| Section | Music |
|---|---|
| Title | Brass fanfare and a xylophone run |
| s01 | Pip's theme: a pompous bassoon strut in F over pizzicato |
| s02 | Bramble's theme: a waddling tuba oompah in B-flat with a trombone tune, and a timpani and cymbal hit on the flop |
| s03 | Suspense tremolo |
| s04 | Timpani heartbeat under climbing strings, and a brass stab on "That does it" |
| s05 | Xylophone galop, then wah-wah trombones |
| s06 | Harp pour, tuba "squelch" and woodwind shake |
| s07 | Timpani roll with chromatic low brass, a big chord on the take, and Bramble's theme turned triumphant |
| s08 | Storm: whirling string runs and cymbal swells |
| s09 | Celesta sparkle, then a harp glissando into a warm chord on "spotless" |
| s10 | Friendship waltz: both themes together |
| s11 | Pip's strut, a brass "ah-choo" stab and a deadpan wah-wah |
| End card | Three-chord tag |
