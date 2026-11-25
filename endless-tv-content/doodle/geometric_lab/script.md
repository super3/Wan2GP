# Mr. Hoist in "One Small Passenger"

*Doodle* channel, 2D TV cartoon short in the geometric late-90s style (UPA influence). Runtime about **118.5 s (1:58.5)**: a 3 s title card, 11 shots, and a 2 s end card.

## Logline
Mr. Hoist, the fussiest elevator operator in the city, swears that Car Number Seven waits for every passenger. Then a snail asks for a ride.

## Characters

- **Mr. Ferdinand Hoist** (S1), a tiny elderly elevator operator who runs Car No. 7 like a naval launch.
  - Design: a tiny elderly elevator operator with a boxy bald head, a red pillbox cap with a gold band, a huge white geometric moustache wider than his face, white slab eyebrows over small black dot eyes, a pink triangular nose, and a red double-breasted tunic with gold buttons and fringed gold epaulettes.
  - Fixed voice: *"a fussy, precise elderly man in his seventies, slightly reedy and self-important, crisp and clear, medium energy, no accent."*

- **Mrs. Dawdle**, a passenger who is never in a hurry.
  - Design: a small, polite snail with a pale yellow-green body, a round plum-purple shell with a square spiral, two eye stalks with round white eyes, and a tiny pink hat with a white daisy.
  - Silent: she only makes tiny squeaks, and acts with her eye stalks.

- **The babies**: five tiny baby snails with teal, orange, pink, yellow and blue shells and no hats. They appear only at the end.

## Beat sheet
1. **Setup (0-26 s).** Hoist's departure ceremony. A tiny passenger calls "wait!". His creed: Car Number Seven waits for every passenger.
2. **Complication (26-69 s).** She is a snail. Time passes, and his patience cracks behind a toothy smile.
   - Attempt 1, the red carpet: she stops to admire it.
   - Attempt 2, lettuce on a pole: she eats it and naps.
   - He nearly leaves without her, but recommits.
3. **Climax (69-94 s).** Night. He falls asleep standing up, and she finally makes it aboard. Full ceremony and salutes: going up!
4. **Payoff (94-104 s).** Ding: one floor. She only wanted the second floor, and getting her off will take just as long.
5. **Button (104-116 s).** Five babies line up for a ride. Hoist keels over like a plank. THE END. The dial needle on the end card sits on floor 2.

## Shot list

Each start/end key is in `keys/`. `s11` starts on `s10`'s end frame (`continues: true`).

| # | id | frames | length | timeline | framing | action | dialogue (S1 = Hoist) |
|---|---|---|---|---|---|---|---|
| - | title | 72 | 3.00 s | 0.00 | title card | MR. HOIST in One Small Passenger; the dial needle sweeps; Hoist and the snail pop on | sting |
| 1 | s01 | 243 | 10.12 s | 3.00 | Wide, lobby (day) | The departure ceremony: Hoist grips the lever and announces the trip. A tiny squeak; he shades his eyes and peers at a snail by the planter. | "Car Number Seven, now departing for the top floor!" |
| 2 | s02 | 124 | 5.17 s | 13.12 | Close-up, snail on the floor | Mrs. Dawdle looks up and waves an eye stalk: wait for me! She inches forward. Silent. | none |
| 3 | s03 | 192 | 8.00 s | 18.29 | Medium close-up, Hoist | Hoist puts his hand on his heart and states his creed. | "Car Number Seven waits for every passenger." |
| 4 | s04 | 294 | 12.25 s | 26.29 | Wide, lobby (day) | She creeps. The clock jumps ahead. Hoist's smile turns to a strained toothy grin over his pocket watch, one eye twitching. | "Take your time, madam. No rush whatsoever." |
| 5 | s05 | 294 | 12.25 s | 38.54 | Wide, lobby (day) | Attempt 1: he kicks out a red carpet, which unrolls all the way to her. She stops to admire it. He throws up his hands. | "A red carpet, madam! Do hurry along!" |
| 6 | s06 | 243 | 10.12 s | 50.79 | Wide, lobby (afternoon) | Attempt 2: a lettuce leaf on a long pole. She eats it in three bites and goes to sleep in her shell. His jaw drops. | "Lettuce, madam? Lovely fresh lettuce?" |
| 7 | s07 | 192 | 8.00 s | 60.92 | Medium close-up, Hoist | He grips the lever, about to leave without her... then breathes, closes his eyes and recommits. | "Car Number Seven... waits." |
| 8 | s08 | 362 | 15.08 s | 68.92 | Wide, lobby (night) | Night. Hoist has fallen asleep standing up. She crawls over the threshold; a squeak; he jolts awake, cap popping up. | "Madam! You made it!" |
| 9 | s09 | 243 | 10.12 s | 84.00 | Medium, cab (night) | Full ceremony: the lever, the ding, a crisp salute. She salutes back with an eye stalk. | "All aboard! Car Number Seven, going up!" |
| 10 | s10 | 243 | 10.12 s | 94.12 | Wide, 2nd floor Garden Dept. (night) | Ding. One floor up. She heads out toward the lettuce crate, extremely slowly. Hoist's moustache droops as it dawns on him. | "Second floor! Garden Department." |
| 11 | s11 | 294 | 12.25 s | 104.25 | Wide, 2nd floor (night), continues from s10 | Button: she turns back, and five baby snails climb out of the crate and line up for a ride. Hoist keels over stiff as a plank, seeing stars. | "Oh dear. I shall... wait." |
| - | end | 48 | 2.00 s | 116.46 | end card | THE END; the snail inches across | sting |

Dialogue never exceeds 0.9 words/s (limit 2.3), so every line leaves at least 0.5 s of silence at the start and end of its shot.

## Sound
- H3 generates the dialogue and the sound effects inside each shot.
- `music_bed.wav` is the score, about 118 s, synthesised in code. It is silent under the title and end cards, which carry their own stings.
  - Instruments: pizzicato bass, brushes and ride, vibraphone, a music-box celesta for Mrs. Dawdle, a plodding bassoon for the waiting, and muted-trumpet fanfares for the ceremony.
  - Hits: a harp gliss for the carpet, wah-wah trombones, a slide whistle and a timpani "bonk" for the topple.
- `assemble.py` ducks the score about 9 dB under loud H3 audio and normalises the whole short to -16 LUFS.

