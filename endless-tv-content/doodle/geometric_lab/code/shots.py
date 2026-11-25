"""shots.py: the shot list of "One Small Passenger" (single source of truth for shots.json, the animatic, the score and assembly).
Run it to write shots.json into the deliverables folder."""
import json
import os
import sys

OUT = "/home/user/Wan2GP/outputs/shorts/geometric_lab"
TITLE = "Mr. Hoist in \"One Small Passenger\""
FPS = 24
TITLE_FRAMES, END_FRAMES = 72, 48          # 3.0 s title card, 2.0 s end card

HOIST = ("Mr. Hoist (a tiny elderly elevator operator with a boxy bald head, a red pillbox cap with a gold band, a huge white "
         "geometric moustache wider than his face, white slab eyebrows over small black dot eyes, a pink triangular nose, and a red "
         "double-breasted tunic with gold buttons and fringed gold epaulettes)")
DAWDLE = ("Mrs. Dawdle (a small, polite snail with a pale yellow-green body, a round plum-purple shell with a square spiral, two eye "
          "stalks with round white eyes, and a tiny pink hat with a white daisy)")
BABIES = "five tiny baby snails with teal, orange, pink, yellow and blue shells and no hats"
VOICE = ("Mr. Hoist's voice: a fussy, precise elderly man in his seventies, slightly reedy and self-important, crisp and clear, "
         "medium energy, no accent.")
SILENT = "Mrs. Dawdle never speaks; she only makes tiny squeaks."

LOBBY = ("the 1950s department-store lobby: the open brass elevator doorway with the lavender cab and a brass lever stand inside, "
         "the half-moon floor dial above the door, aqua diamond-patterned walls with cone wall lamps, a starburst wall clock, a potted "
         "plant and an orange-and-cream checkerboard floor")
LOBBY_NIGHT = LOBBY + ", now at night, lit only by the warm cab light and the wall lamps, with deep blue shadows"
GARDEN = ("the store's second-floor Garden Department at night: the same open brass elevator doorway with the lavender cab and brass "
          "lever stand, the half-moon floor dial above the door, green leaf-patterned walls, a white picket-fence wainscot, a pot of "
          "sunflowers, a wooden crate of lettuces and terracotta floor tiles, lit by the warm cab light")
CAB = "the lavender elevator cab, the brass doorway and the aqua lobby wall behind him"
FLOOR_CU = "the lobby's orange-and-cream checkerboard floor beside the white planter"


def prompt(chars, setting, framing, action, end, sfx, voice=True):
    return (f"integrated_multimodal_description: [Shot 1] Continue directly from <Picture 1>, a 2D hand-drawn 1990s TV cartoon frame: "
            f"keep exactly its flat colours, thick black outlines and painted background. Preserve {chars} and {setting}. "
            f"In a static {framing} shot, {action}, ending exactly on <Picture 2>: {end}. The camera does not move. "
            f"Traditional 2D cel animation, not 3D, not photorealistic.\n"
            f"overall_soundscape: {sfx} {VOICE if voice else 'No dialogue in this shot.'} {SILENT}\n"
            f"non_diegetic_music: N/A")


HD = f"{HOIST} and {DAWDLE}"
SHOTS = [
    dict(id="s01", frames=243, start="keys/s01_a.png", end="keys/s01_b.png", continues=False,
         beat="Ceremony; the tiny passenger", line="Car Number Seven, now departing for the top floor!",
         prompt=prompt(HD, LOBBY, "wide",
                       "Mr. Hoist stands at attention in the elevator, grips the brass lever, lifts his chin and proudly announces (S1) "
                       "<d>[English] Car Number Seven, now departing for the top floor!</d> A tiny squeak comes from the far left: he "
                       "freezes, raises one flat hand over his eyebrows and leans to peer at the little snail on the floor beside the "
                       "plant, who lifts an eye stalk",
                       "Mr. Hoist leaning left with one flat hand shading his eyes and his other hand on the upright lever, Mrs. Dawdle "
                       "still beside the plant",
                       "Quiet lobby room tone, a small brass click of the lever, a tiny squeak.")),
    dict(id="s02", frames=124, start="keys/s02_a.png", end="keys/s02_b.png", continues=False,
         beat="Close on Mrs. Dawdle: wait for me!", line="",
         prompt=prompt(DAWDLE, FLOOR_CU, "close-up",
                       "Mrs. Dawdle looks up toward the elevator, then eagerly waves one eye stalk back and forth as if calling 'wait "
                       "for me!', gives a tiny happy squeak and inches forward a few millimetres",
                       "Mrs. Dawdle smiling up, one eye stalk bent back and the other leaning forward",
                       "A tiny eager squeak, a soft wet slither, quiet lobby room tone.", voice=False)),
    dict(id="s03", frames=192, start="keys/s03_a.png", end="keys/s03_b.png", continues=False,
         beat="The creed", line="Car Number Seven waits for every passenger.",
         prompt=prompt(HOIST, CAB, "medium close-up",
                       "Mr. Hoist looks off to the left, then straightens up, puffs out his chest, places one hand on his heart, closes "
                       "his eyes in a proud smile and declares (S1) <d>[English] Car Number Seven waits for every passenger.</d>",
                       "hand on his heart, eyes closed in a proud smile, chest out, his other hand on the lever",
                       "A soft fabric rustle as he puffs up his chest, quiet lobby room tone.")),
    dict(id="s04", frames=294, start="keys/s04_a.png", end="keys/s04_b.png", continues=False,
         beat="Waiting... and waiting", line="Take your time, madam. No rush whatsoever.",
         prompt=prompt(HD, LOBBY, "wide",
                       "Mr. Hoist stands proudly with his hand on his heart while Mrs. Dawdle creeps very slowly to the right across the "
                       "floor. Time passes: the hands of the starburst clock jump forward. His smile turns stiff; he pulls out a gold "
                       "pocket watch on a chain, stares at it with one eye twitching and says through a strained toothy grin (S1) "
                       "<d>[English] Take your time, madam. No rush whatsoever.</d>",
                       "Mr. Hoist holding up the gold pocket watch with a strained toothy grin and one half-closed twitching eye, the "
                       "clock showing a later time, the snail only a little further right",
                       "A loud clock ticking, a slow wet slither, a pocket watch clicking open.")),
    dict(id="s05", frames=294, start="keys/s05_a.png", end="keys/s05_b.png", continues=False,
         beat="Attempt 1: the red carpet", line="A red carpet, madam! Do hurry along!",
         prompt=prompt(HD, LOBBY, "wide",
                       "Mr. Hoist holds a rolled-up red carpet upright beside him, tips it over and gives it a smart kick: it unrolls in "
                       "one swoosh out of the elevator and across the floor all the way to Mrs. Dawdle. She stops to admire it and "
                       "settles down happily on its end. He throws both open hands up, sweating, and says (S1) <d>[English] A red "
                       "carpet, madam! Do hurry along!</d>",
                       "a red carpet with gold edges running from the elevator to the snail, who sits on its end gazing down at it; Mr. "
                       "Hoist with both open hands raised, a strained toothy grin and a sweat drop",
                       "A thump, a long rolling swoosh of carpet unrolling, a tiny delighted squeak.")),
    dict(id="s06", frames=243, start="keys/s06_a.png", end="keys/s06_b.png", continues=False,
         beat="Attempt 2: lettuce on a pole", line="Lettuce, madam? Lovely fresh lettuce?",
         prompt=prompt(HD, LOBBY, "wide",
                       "Mr. Hoist holds a long brass pole out of the elevator with a lettuce leaf dangling on a string in front of Mrs. "
                       "Dawdle and coaxes sweetly (S1) <d>[English] Lettuce, madam? Lovely fresh lettuce?</d> She munches the whole leaf "
                       "down to a stub in three quick bites, yawns, and tucks herself into her shell for a nap. His jaw drops",
                       "the snail asleep inside her shell on the carpet, only a nibbled stub left on the string, Mr. Hoist frozen holding "
                       "the pole with his mouth wide open and his eyes wide",
                       "Three crunchy munching bites, a tiny yawn, a hollow 'plop' as she pulls into her shell.")),
    dict(id="s07", frames=192, start="keys/s07_a.png", end="keys/s07_b.png", continues=False,
         beat="He nearly cracks", line="Car Number Seven... waits.",
         prompt=prompt(HOIST, CAB, "medium close-up",
                       "Mr. Hoist grips the brass lever with both hands, one eye twitching, teeth gritted, moustache bristling, about to "
                       "yank it; then he lets go, takes a long deep breath, closes his eyes, lifts his chin with a calm little smile and "
                       "says (S1) <d>[English] Car Number Seven... waits.</d>",
                       "standing calm and upright, eyes closed, a small composed smile, one hand on the lever and one fist on his hip",
                       "A trembling rattle of the lever, a long deep breath in and out.")),
    dict(id="s08", frames=362, start="keys/s08_a.png", end="keys/s08_b.png", continues=False,
         beat="Night: she makes it aboard", line="Madam! You made it!",
         prompt=prompt(HD, LOBBY_NIGHT, "wide",
                       "Mr. Hoist has fallen asleep standing up at the lever, snoring softly, his moustache fluttering with each snore. "
                       "Along the red carpet, Mrs. Dawdle slowly crawls the last stretch, over the brass threshold and into the elevator "
                       "beside his feet. She gives a tiny squeak: Mr. Hoist jolts awake, his cap popping up off his head, eyes wide, and "
                       "cries with delight (S1) <d>[English] Madam! You made it!</d>",
                       "Mr. Hoist wide awake and delighted with his cap lifted above his head and one open hand out, Mrs. Dawdle inside "
                       "the elevator by his feet",
                       "Night hush, soft snoring with a whistle, a slow wet slither, a tiny squeak, a startled snort.")),
    dict(id="s09", frames=243, start="keys/s09_a.png", end="keys/s09_b.png", continues=False,
         beat="Full ceremony: going up!", line="All aboard! Car Number Seven, going up!",
         prompt=prompt(HD, LOBBY_NIGHT, "medium",
                       "Mr. Hoist beams, puffs out his chest and announces grandly (S1) <d>[English] All aboard! Car Number Seven, "
                       "going up!</d> He yanks the brass lever across with a clunk, the up-arrow lamp above the door lights, and he snaps "
                       "a crisp salute with his eyes proudly closed; beside his feet, Mrs. Dawdle salutes with one eye stalk",
                       "the lever pulled across, the up-arrow lamp lit, Mr. Hoist saluting with his eyes closed, the snail saluting with "
                       "an eye stalk",
                       "A heavy brass lever clunk, a bright bell ding, a crisp salute snap.")),
    dict(id="s10", frames=243, start="keys/s10_a.png", end="keys/s10_b.png", continues=False,
         beat="Second floor already", line="Second floor! Garden Department.",
         prompt=prompt(f"{HD} and {BABIES}", GARDEN, "wide",
                       "Mr. Hoist proudly announces (S1) <d>[English] Second floor! Garden Department.</d> Mrs. Dawdle starts to crawl "
                       "out of the elevator toward the lettuce crate, extremely slowly. Mr. Hoist's smile melts: his moustache droops, "
                       "his eyebrows tilt in dismay and one eye twitches as he realises how long this will take. In the lettuce crate, "
                       "the baby snails wiggle their eye stalks",
                       "Mrs. Dawdle on the brass threshold, Mr. Hoist with a drooping moustache and dismayed eyes, the baby snails "
                       "sitting in the lettuce crate",
                       "A cheerful elevator ding, a slow wet slither, a long weary sigh.")),
    dict(id="s11", frames=294, start="keys/s10_b.png", end="keys/s11_b.png", continues=True,
         beat="Button: the babies want a ride", line="Oh dear. I shall... wait.",
         prompt=prompt(f"{HD} and {BABIES}", GARDEN, "wide",
                       "Mrs. Dawdle finally slides out onto the floor and turns back toward the elevator; the five baby snails climb out "
                       "of the lettuce crate and line up behind her, all looking up eagerly at the elevator. Mr. Hoist stares, whimpers "
                       "(S1) <d>[English] Oh dear. I shall... wait.</d> and keels over stiff as a plank against the side of the doorway, "
                       "his cap knocked askew, little yellow stars circling his head and dizzy swirls in his eyes",
                       "Mr. Hoist tipped over stiff against the right side of the doorway with swirly eyes and stars, the snail and her "
                       "five babies lined up in front of the elevator",
                       "Tiny squeaks from the babies, a little pitter-patter of slithers, a long wooden creak and a thunk as he tips over, "
                       "a tweeting of dizzy stars.")),
]


def timeline():
    """[(shot, start_s, dur_s)] on the assembled timeline (title card first; a 'continues' shot loses its first frame)."""
    t = TITLE_FRAMES
    out = []
    for s in SHOTS:
        n = s["frames"] - (1 if s["continues"] else 0)
        out.append((s, t / FPS, n / FPS))
        t += n
    return out, (t + END_FRAMES) / FPS


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    data = {"title": TITLE, "shots": [{k: s[k] for k in ("id", "frames", "start", "end", "continues", "prompt")} for s in SHOTS]}
    with open(os.path.join(OUT, "shots.json"), "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tl, total = timeline()
    for s, t0, d in tl:
        words = len(s["line"].replace("...", " ").split())
        print(f"{s['id']} start {t0:6.2f}s  dur {d:5.2f}s  words {words:2d}  ({words / d:.2f} w/s)")
    print(f"total {total:.2f}s")
    sys.exit(0)
