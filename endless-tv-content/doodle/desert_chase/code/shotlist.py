"""The shot list of "Clean Sweep" (single source of truth for shots.json, script.md, the score timing, the animatic and assembly)."""
import json
import os

TITLE = "Clean Sweep"
SERIES = "A Pip the Armadillo Cartoon"
FPS = 24
TITLE_FRAMES, END_FRAMES = 72, 48

ERA = ("a 2D hand-drawn classic 1950s theatrical cartoon frame: keep exactly its flat colours, clean black ink outlines and painted "
       "background")
PIP = ("Pip, a small upright armadillo (slate-blue banded shell, a blue armour plate from his crown down his long pink snout, tall pointed "
       "ears, a red polka-dot neckerchief, thin pink arms and big feet)")
BRAM = ("Bramble, a big round javelina on four short legs (grey-violet bristly body, a spiky dark crest along his back, a cream collar "
        "stripe over his shoulders, a pink pig-snout disc with one small white tusk, little pointed ears, tiny black hooves)")
V_PIP = "Pip's voice: a fussy, slightly nasal, middle-aged male tenor, crisp and precise"
V_BRAM = "Bramble's voice: a big, warm, booming male bass, slow, jolly and good-natured"
HOME = ("the painted desert ledge with Pip's burrow at the foot of a tall orange rock, a little striped rug, a barrel cactus, a needle-shaped "
        "rock spire, flat-topped mesas and a road winding along the canyon floor far below")
WALLOW = ("the painted desert ledge with Bramble's shallow dusty wallow, a gnarled dead scratching post, tall orange rock spires and "
          "flat-topped mesas under a pale turquoise sky")
LINE = ("the painted desert ledge with a neat row of little white pebbles marking the property line, Bramble's shallow dusty wallow, "
        "a gnarled dead scratching post, tall orange rock spires and flat-topped mesas")
CLOSE = "the painted desert behind him: a pale mesa, low purple far mesas and the road winding along the canyon floor"
TAIL = "The camera does not move. Traditional 2D cel animation, not 3D, not photorealistic."


def prompt(chars, setting, framing, action, end, sfx, voices):
    return (f"integrated_multimodal_description: [Shot 1] Continue directly from <Picture 1>, {ERA}. Preserve {chars} and {setting}. "
            f"In a static {framing} shot, {action}, ending exactly on <Picture 2>: {end}. {TAIL}\n"
            f"overall_soundscape: {sfx}; {'; '.join(voices)}.\n"
            f"non_diegetic_music: N/A")


SHOTS = [
    dict(id="s01", frames=243, start="s01_a", end="s01_b", continues=False, cam="home",
         beat="Premise: Pip finishes sweeping his ledge spotless and admires his work.",
         line=("Pip", "Ahem. Spotless."),
         prompt=prompt(PIP, HOME, "medium-wide",
                       "Pip finishes sweeping with three brisk, fussy strokes of his straw broom, flicks the last pebble away, then straightens up, "
                       "plants the broom upright at his side, puts his other fist on his hip, lifts his chin proudly and, after a short pause, "
                       "says (S1) <d>[English] Ahem. Spotless.</d> with a smug little nod",
                       "Pip standing proudly with his broom upright at his side, fist on hip, chin up, half-lidded smug eyes, mouth closed",
                       "a dry desert breeze, quick scratchy broom strokes on sandy rock, a tiny pebble skittering away, a satisfied sniff",
                       [V_PIP])),
    dict(id="s02", frames=243, start="s02_a", end="s02_b", continues=False, cam="wallow",
         beat="Inciting incident: next door, Bramble trots to his wallow and flops into a joyous dust bath.",
         line=("Bramble", "Ahhh. Bath time!"),
         prompt=prompt(BRAM, WALLOW, "wide",
                       "Bramble trots happily to the left with a bouncy step, eyes closed in bliss, says (S1) <d>[English] Ahhh. Bath time!</d>, "
                       "then flops onto his back in the dusty wallow and wriggles and kicks his legs in the air with delight, sending big puffy "
                       "clouds of dust billowing up and drifting off to the left on the breeze",
                       "Bramble lying on his back in the wallow with his four legs in the air, laughing with his mouth open, puffy dust clouds "
                       "rising around him",
                       "light hoofbeats on sandy rock, a big happy sigh, a soft whump as he flops down, rhythmic dusty wriggling and scuffing, "
                       "billowing dust", [V_BRAM])),
    dict(id="s03", frames=192, start="s01_b", end="s03_b", continues=False, cam="home",
         beat="The dust cloud drifts over and engulfs Pip mid-pose.",
         line=None,
         prompt=prompt(PIP, HOME, "medium-wide",
                       "Pip holds his proud pose, eyes half-closed and smug; then a huge, slow cloud of pale dust rolls in from the right edge of "
                       "frame and swallows him completely; when it drifts away to the left Pip is coated in grey-beige dust from his ears to his "
                       "tail, still frozen in the same pose; he blinks twice, slowly, and gives one tiny dry cough",
                       "Pip in the same proud pose but covered all over in pale dust, deadpan half-closed eyes, mouth flat, broom drooping "
                       "slightly, the last wisps of the dust cloud drifting away on the left",
                       "a soft rising whoosh of wind, the hiss of drifting dust, a long silence, two tiny blinks, one small dry cough",
                       [V_PIP])),
    dict(id="s04", frames=243, start="s04_a", end="s04_b", continues=False, cam="close",
         beat="Slow burn: Pip's deadpan close-up boils into fury.",
         line=("Pip", "That... does it."),
         prompt=prompt(PIP + ", coated in pale dust", CLOSE, "close-up",
                       "the dusty Pip stares deadpan with heavy lids for a long beat; one eye twitches; his brows slowly lower into a furious "
                       "scowl, his ears flatten back, and he raises his broom in both hands like a banner and says through clenched teeth "
                       "(S1) <d>[English] That... does it.</d>",
                       "dusty Pip glaring sideways with angry slanted brows and narrowed eyes, mouth set in a hard frown, holding his broom up "
                       "high in both hands, bristles at the top",
                       "a quiet breeze, a tiny eyelid twitch, a slow rising tension of silence, the creak of a gripped broom handle",
                       [V_PIP])),
    dict(id="s05", frames=294, start="s05_a", end="s05_b", continues=False, cam="line",
         beat="Attempt 1: Pip sweeps the dust back over the property line; Bramble loves it and rolls harder; the cloud blows back over Pip.",
         line=("Bramble", "Ooh! Thanks, neighbor!"),
         prompt=prompt(f"{PIP} and {BRAM}", LINE, "wide two-character",
                       "the dusty Pip, on the left, sweeps furiously, flinging a stream of dust over the pebble line onto Bramble, who lies "
                       "on his back in his wallow on the right; Bramble rolls happily in the falling dust and says (S1) <d>[English] Ooh! "
                       "Thanks, neighbor!</d>, then wriggles harder, kicking up an even bigger dust cloud that drifts straight back over Pip; "
                       "Pip stops sweeping and stands stiffly, deadpan, as the dust settles on him",
                       "Pip standing on the left, even dustier, deadpan with heavy lids, broom at his side; Bramble on his back in a heap of "
                       "dust in his wallow, legs in the air, laughing; puffy dust clouds drifting between them",
                       "fast scratchy broom strokes, swishing flung dust, happy dusty wriggling, a big puff of dust, dust pattering down on Pip",
                       [V_BRAM])),
    dict(id="s06", frames=243, start="s06_a", end="s06_b", continues=False, cam="line",
         beat="Attempt 2: Pip pours his canteen over Bramble to wash him; dust plus water makes mud; Bramble shakes like a wet dog.",
         line=("Bramble", "Ooh, mud! Even better!"),
         prompt=prompt(f"{PIP}, coated in pale dust, and {BRAM}", LINE, "wide two-character",
                       "Pip, on the left, stands on tiptoe and tips his round green canteen so a stream of water pours onto the head of Bramble, "
                       "who stands on the right with his eyes closed; the dust on Bramble turns into dark brown mud; Bramble opens his eyes, "
                       "says (S1) <d>[English] Ooh, mud! Even better!</d>, and shakes himself all over like a wet dog, spraying mud "
                       "everywhere; brown mud splats land all over Pip, who lowers the empty canteen",
                       "Pip on the left, spattered with brown mud spots, deadpan with heavy lids, holding the empty canteen at his chest; "
                       "Bramble on the right covered in dark brown mud, grinning with his mouth open; mud splats on the ground",
                       "water glugging from a canteen and splashing, a delighted gasp, a long wet flappy shake, a spray of wet mud splats "
                       "hitting Pip, a final drip", [V_BRAM])),
    dict(id="s07", frames=294, start="s06_b", end="s07_b", continues=True, cam="line",
         beat="Complication: a low rumble; a gigantic dust storm rises over the canyon. Pip's take; Bramble is thrilled.",
         line=("Bramble", "Here comes the big one!"),
         prompt=prompt(f"{PIP}, dusty and spattered with mud, and {BRAM}, covered in brown mud", LINE, "wide two-character",
                       "a low rumble grows; both look up at the far horizon, where an enormous wall of billowing brown dust rises behind the "
                       "mesas and towers higher and higher over the canyon while the light turns amber; Pip does a huge take, eyes popping, "
                       "ears shooting straight up and jaw dropping as he lets go of the canteen; Bramble beams with joy and says (S1) "
                       "<d>[English] Here comes the big one!</d>",
                       "a gigantic wall of billowing dust filling the sky behind the rock spires; Pip with huge round eyes, ears straight up, "
                       "mouth open in shock, arms raised; Bramble looking up with a big open-mouthed grin; the canteen lying on the ground",
                       "a deep rumble growing into a roaring wind, a rising whoosh, the canteen clanking onto rock, a gulp from Pip",
                       [V_BRAM])),
    dict(id="s08", frames=243, start="s07_b", end="s08_b", continues=True, cam="line",
         beat="Climax: Pip rolls into a ball; Bramble plants his hooves and grins into the storm as it swallows everything.",
         line=("Bramble", "Wheee!"),
         prompt=prompt(f"{PIP} and {BRAM}", LINE, "wide two-character",
                       "Pip dives down and rolls himself up into a tight banded armadillo ball; Bramble plants his four hooves, closes his "
                       "eyes happily and leans into the wind as the wall of dust sweeps over them; Bramble whoops (S1) <d>[English] "
                       "Wheee!</d>; streaks of wind and dust race across the frame until everything sinks into a swirling amber-brown haze",
                       "a swirling amber-brown dust haze over everything, Pip rolled into a tight blue ball on the left, Bramble standing "
                       "braced on the right with eyes shut and a happy grin, wind streaks racing across",
                       "a roaring dust storm, howling wind, sand hissing across rock, a tight armour clack as Pip rolls up", [V_BRAM])),
    dict(id="s09", frames=294, start="s09_a", end="s09_b", continues=False, cam="line",
         beat="Payoff: the dust settles over the whole desert except a perfectly clean wedge in Bramble's lee, right where Pip is.",
         line=("Pip", "My ledge... it's spotless!"),
         prompt=prompt(f"{PIP} and {BRAM}, now puffed up into a giant fluffy ball of dust with only his snout, eyes, ears and hooves "
                       "showing", LINE + ", now blanketed in pale dust except for a clean wedge of ground", "wide two-character",
                       "the air is calm and still; Pip, rolled into a ball on the clean patch, slowly peeks out, uncurls and stands up, "
                       "picks up his broom, looks around at the clean wedge of ground behind the giant dust-ball Bramble and at the dust "
                       "covering everything else, then says in amazement (S1) <d>[English] My ledge... it's spotless!</d>",
                       "Pip standing on the clean patch holding his broom upright, eyes wide with amazement and a delighted smile, looking at "
                       "the fluffy dust-ball Bramble, who beams at him",
                       "a soft breeze dying away, settling dust, a creaky uncurl of armour plates, a broom lifted from the sand, a small "
                       "amazed gasp", [V_PIP])),
    dict(id="s10", frames=243, start="s09_b", end="s10_b", continues=True, cam="line",
         beat="Resolution: Pip brushes Bramble clean; Bramble turns his back for more, hind leg thumping with bliss. A deal is struck.",
         line=("Bramble", "Ooh... right there!"),
         prompt=prompt(f"{PIP} and {BRAM}", LINE + ", blanketed in pale dust except for a clean wedge of ground", "wide two-character",
                       "Pip steps up and starts brushing the dust off the fluffy Bramble with his broom; big clumps of dust fall away and "
                       "Bramble's grey-violet bristles appear; Bramble turns round to offer his back, closes his eyes in bliss and says (S1) "
                       "<d>[English] Ooh... right there!</d>, one hind leg thumping happily as Pip keeps brushing with a smile",
                       "Pip brushing Bramble's back with his broom, smiling; Bramble facing right, nearly clean, eyes closed in bliss, mouth "
                       "open, one hind leg lifted; a heap of brushed-off dust under him",
                       "soft brushing strokes through bristles, dust clumps plopping onto sand, a blissful rumbling hum, a hind hoof thumping "
                       "the ground", [V_BRAM])),
    dict(id="s11", frames=192, start="s11_a", end="s11_b", continues=False, cam="home",
         beat="Button: at sunset, Pip shows off his ledge with his new friend - and Bramble sneezes.",
         line=("Pip", "Ahem. Spotless."),
         prompt=prompt(f"{PIP} and {BRAM}, now clean and wearing a matching red polka-dot neckerchief", HOME + ", glowing in a pink and "
                       "violet sunset", "medium-wide",
                       "Pip stands proudly with his broom, chin up, and after a short pause says (S1) <d>[English] Ahem. Spotless.</d>; "
                       "beside him Bramble's snout twitches, he draws in a huge breath and lets out an enormous sneeze that blasts a cloud of "
                       "dust over Pip; Pip, coated in dust again, stays perfectly still and deadpan while Bramble smiles sheepishly",
                       "Pip in his proud pose but covered in pale dust, deadpan with heavy lids; Bramble beside him smiling sheepishly with "
                       "raised brows; dust clouds drifting away on the left",
                       "a gentle evening breeze, a smug sniff, a snout twitching, a long rising inhale, a huge booming sneeze, a puff of dust, "
                       "a tiny settling of grit", [V_PIP, V_BRAM])),
]


def timeline():
    """(id, first frame on the assembled timeline, frames kept) for every shot, after the title card."""
    t = TITLE_FRAMES
    out = []
    for s in SHOTS:
        n = s["frames"] - (1 if s["continues"] else 0)
        out.append((s["id"], t, n))
        t += n
    return out, t + END_FRAMES


def write(out_dir):
    data = {"title": TITLE, "shots": [{"id": s["id"], "frames": s["frames"], "start": f"keys/{s['start']}.png", "end": f"keys/{s['end']}.png",
                                       "continues": s["continues"], "prompt": s["prompt"]} for s in SHOTS]}
    with open(os.path.join(out_dir, "shots.json"), "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    write(sys.argv[1])
    tl, total = timeline()
    print("total frames", total, f"{total / FPS:.2f}s")
