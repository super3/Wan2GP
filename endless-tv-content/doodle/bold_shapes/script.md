# Squeaky Doyle: *One Little Smudge*

**Logline.** Sky-High City's smallest, bravest window-washer declares his last pane spotless - then spends the afternoon losing a war against one tiny smudge, until the little old lady inside wipes it away with her hanky.

**Runtime.** 1:57.50 (2820 frames at 24 fps): 3 s title card, 10 shots, 2 s end card. Style: late-90s 'bold shapes' TV cartoon (UPA-descended): thick black outlines, flat candy colours, geometric characters, flat retro-modern backgrounds.

## Characters

- **Squeaky Doyle**: the smallest, bravest window-washer in Sky-High City. Wide bean head, huge blue eyes, no nose, tilted yellow hard hat with black hair tufts, orange bib overalls over a white tee, red rubber boots, mitten hands. Works from a bosun's chair (plank seat on rope straps) with a red suds bucket and a giant blue squeegee. Treats every pane like a super-mission.  
  *Fixed voice (S1):* "a small boy about seven years old, high, bright and bouncy, brave and cheerful, accent-free"
- **Mrs. Fig** (original supporting character): a tiny elderly office lady who has worked at the same desk for fifty years. Big round red glasses over huge green eyes, grey bun with a pencil stuck through it, lilac cardigan, white collar, pearls, rosy cheeks. Tea, a lace hanky, infinite calm.  
  *Fixed voice (S2):* "a sweet little old lady about eighty, soft, warm and slightly creaky, gentle and unhurried, accent-free"

## Beat sheet

- **Setup (0:03-0:16)**: A hundred floors up, Squeaky finishes the day's last pane and declares it *spotless*. The audience can already see the little brown smudge right behind him.
- **Inciting gag (0:16-0:26)**: He notices. Double take. 'Oh no you don't.'
- **Escalation (0:26-1:15)**: Four attempts, each bigger and each failing: (1) the squeegee, twice, full force; (2) the old huff-and-buff, which turns his face red; (3) extra suds, which the wind blows back so the bucket lands on his head; (4) the Polish-o-Matic, which spins him dizzy on his rope.
- **Twist / payoff (1:15-1:30)**: Cut inside. Mrs. Fig looks up from her tea, and with one dainty wipe of her hanky removes the smudge. It was on the INSIDE the whole time.
- **Save (1:30-1:40)**: Squeaky, jaw on the floor, saves face: 'I... loosened it for you.'
- **Button (1:40-1:55)**: He admires the shining glass, presses his face to it, and leaves a big face print. Mrs. Fig raises her hanky. 'Same time tomorrow?'  End card: her hanky wipes the print away.

## Shot list

| # | in (tc) | length | set-up | start -> end key | action | dialogue |
|---|---|---|---|---|---|---|
| title | 0:00.00 | 72 f (3.0 s) | card | - | Glass with the smudge; a giant squeegee wipes on the title... the smudge survives; Squeaky pops up to glare at it. | - |
| s01 | 0:03.00 | 124 f (5.17 s) | wide establishing | `s01_a` -> `s01_b` | Establish: the smallest window-washer in Sky-High City, a hundred floors up. | (whistling, no words) |
| s02 | 0:08.17 | 192 f (8.00 s) | medium | `s02_a` -> `s02_b` | He declares victory - the audience can already see the smudge. | **Squeaky:** "Last window of the day. Spotless!" |
| s03 | 0:16.17 | 243 f (10.12 s) | close | `s03_a` -> `s03_b` | He notices. It's personal now. | **Squeaky:** "Wait. What is that? Oh no you don't." |
| s04 | 0:26.29 | 294 f (12.25 s) | medium | `s04_a` -> `s04_b` | Attempt 1: the squeegee. Twice. Nothing. | **Squeaky:** "Hi-yah! ... Hi-YAH!" |
| s05 | 0:38.54 | 294 f (12.25 s) | close | `s05_a` -> `s05_b` | Attempt 2: huff and buff. He boils over. | **Squeaky:** "Okay, smudge. The old huff and buff." |
| s06 | 0:50.79 | 294 f (12.25 s) | medium | `s06_a` -> `s06_b` | Attempt 3: the bucket. The wind sends it all back. | **Squeaky:** "Time for extra suds!" |
| s07 | 1:03.04 | 294 f (12.25 s) | medium | `s07_a` -> `s07_b` | Attempt 4: the gadget. He polishes himself into a spin. | **Squeaky:** "Polish-o-Matic. Full power!" |
| s08 | 1:15.29 | 362 f (15.08 s) | interior medium-wide | `s08_a` -> `s08_b` | The reveal: the smudge was on the INSIDE. | **Mrs. Fig:** "Goodness. There we are." |
| s09 | 1:30.38 | 243 f (10.12 s) (continues) | interior medium-wide | `s08_b` -> `s09_b` | Saving face. | **Squeaky:** "I... loosened it for you." |
| s10 | 1:40.46 | 362 f (15.08 s) (continues) | interior medium-wide | `s09_b` -> `s10_b` | Button: his own face print. Same time tomorrow. | **Squeaky:** "Same time tomorrow?" |
| end | 1:55.50 | 48 f (2.0 s) | card | - | 'THE END' over Squeaky's face print; Mrs. Fig's hanky wipes it away. | - |

## Music (music_bed.wav)

An original 60s spy-lounge cartoon score in one piece, silent under the cards (they carry their own stings), changing cue exactly on each cut: Squeaky's whistled theme (s01-s02, with a 'ta-da' on *Spotless*); a double-take tuba sting and pizzicato spy ostinato (s03); a surf-guitar action riff with two brass stabs and a sad trombone (s04); a creeping chromatic boil with a kettle-whistle glide (s05); a fanfare, timpani roll, splash and soggy tuba bwomps (s06); an accelerating gadget motor riff and dizzy wobbly vibes (s07); a tea-time celesta minuet and a record scratch (s08); awkward plunks and a sheepish muted 'wah-wah' (s09); a minuet reprise, a squish 'bloop', an 'uh-oh', and the theme returning into the end card (s10). The assembly ducks it ~9 dB under the H3 dialogue.

## Assembly

`python3 <code folder>/assemble.py RENDERS_DIR OUT.mp4` (stdlib + ffmpeg only). RENDERS_DIR holds `s01.mp4` ... `s10.mp4`. Each render is conformed to its exact frame count, and s09/s10 drop their first frame. The music bed is ducked under the H3 audio and the mix is loudness-normalised to -16 LUFS. Output: 1280x720 H.264/AAC. The code folder is `/tmp/claude-0/-home-user-Wan2GP/4e1e6010-b786-5f93-b6fd-87ee44cba9e7/scratchpad/styles/bold_shapes/`.

