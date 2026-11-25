"""Shot list, prompts and timing for "One Lousy Dandelion" (writes shots.json)."""
import json
import os
import sys

TITLE = "Wendell in: One Lousy Dandelion"
FPS = 24
TITLE_S, END_S = 3.0, 2.0

WEN = ("Wendell, a lanky ten-year-old boy with a scruffy tangerine mop of hair, a long pickle-shaped nose, two big buck teeth, "
       "freckles, a sky-blue polo shirt with a white collar, knee-length khaki shorts, tall white socks with red bands and huge red sneakers")
WEN_DIRTY = WEN + ", now smudged all over with dirt"
BEA = ("Bea, his little sister, a small round-headed six-year-old girl with two black curly hair puffs tied with yellow bobbles, "
       "sleepy half-closed eyes, a yellow T-shirt under purple overalls and pink sneakers, sitting in a green-and-white webbed "
       "aluminium lawn chair")
YARD = ("the sunny suburban backyard: a striped, perfectly mowed green lawn, a tall wooden plank fence, a pink house and a yellow "
        "house behind it and a big leafy tree, all softly hand-painted")
DD = "the single yellow dandelion in the lawn"
V_WEN = "Wendell's voice: a ten-year-old boy, nasal, scratchy and over-excited, full of cartoon bravado"
V_BEA = "Bea's voice: a six-year-old girl, flat, dry and slow, completely deadpan"
HEAD = ("integrated_multimodal_description: [Shot 1] Continue directly from <Picture 1>, a 2D hand-drawn 1990s TV cartoon frame: "
        "keep exactly its flat colours, thick black outlines and painted background. ")
TAIL = " The camera does not move. Traditional 2D cel animation, not 3D, not photorealistic."


def P(preserve, action, sound):
    return (HEAD + "Preserve " + preserve + ". " + action + TAIL + "\noverall_soundscape: " + sound + "\nnon_diegetic_music: N/A")


SHOTS = [
    dict(id="s01", frames=243, start="keys/s01_a.png", end="keys/s01_b.png",
         line="Dad's lawn. Perfectly perfect. Not one single weed.", speaker="Wendell",
         prompt=P(f"{WEN}; {BEA}, sipping a juice box; {DD}; and {YARD}",
                  "In a static wide shot, Wendell stands proudly with his hands on his hips, admires the lawn and says (S1) <d>[English] "
                  "Dad's lawn. Perfectly perfect. Not one single weed.</d> He nods, then spots the dandelion: a huge cartoon double take, "
                  "his eyes bulge, his hair springs up and his jaw drops wide open, ending exactly on <Picture 2>: Wendell leaning back in "
                  "shock, mouth wide open, pointing at the dandelion with one hand. Bea just sips her juice box.",
                  f"Birdsong and a soft summer breeze, a smug little hum, then a cartoon zip and a springy boing on the double take, and a "
                  f"loud gasp. {V_WEN}.")),
    dict(id="s02", frames=243, start="keys/s02_a.png", end="keys/s02_b.png",
         line="This lawn ain't big enough for the both of us, flower.", speaker="Wendell",
         prompt=P(f"{WEN}; {DD}; and {YARD}",
                  "In a static medium shot, Wendell squats nose to nose with the dandelion, hands on his knees, glaring at it like a "
                  "cowboy in a showdown. He narrows his eyes and says (S1) <d>[English] This lawn ain't big enough for the both of us, "
                  "flower.</d> The dandelion gives a tiny wobble. He grins through gritted teeth and rubs his hands together, ending "
                  "exactly on <Picture 2>: Wendell still squatting, fists together under his chin, a determined gritted-teeth grin.",
                  f"A lone whistling breeze like a western standoff, a tumbleweed-dry rustle of grass, a squeaky hand rub. {V_WEN}.")),
    dict(id="s03", frames=243, start="keys/s03_a.png", end="keys/s03_b.png",
         line="Nnngh! Let... go... of... my... lawn!", speaker="Wendell",
         prompt=P(f"{WEN}; {BEA}, holding a juice box; {DD}; and {YARD}",
                  "In a static wide shot, Wendell grips the dandelion stem with both hands and pulls with all his might, leaning far back, "
                  "teeth gritted, and strains (S1) <d>[English] Nnngh! Let... go... of... my... lawn!</d> The stem stretches like a rubber "
                  "band, his hands slip, the dandelion snaps back upright and Wendell flies backwards and lands flat on his back on the "
                  "grass with a thud, dazed, tongue out, one yellow petal in his raised fist, ending exactly on <Picture 2>: Wendell lying "
                  "flat on his back, head to the left, the dandelion standing perfectly upright again. Bea watches without moving.",
                  f"Straining grunts, a creaking rubber-band stretch, a loud twang, a whoosh, a heavy thud and a little birdie tweet. "
                  f"{V_WEN}.")),
    dict(id="s04", frames=192, start="keys/s04_a.png", end="keys/s04_a.png", line="That's one.", speaker="Bea",
         prompt=P(f"{BEA}, holding a juice box with a bendy straw; and the corner of the backyard behind her: wooden plank fence, "
                  "round green bushes with orange flowers, the striped lawn",
                  "In a static medium shot, Bea, completely unimpressed, looks off to the right, raises her juice box and says (S1) "
                  "<d>[English] That's one.</d> She takes a long loud sip through the straw, lowers the juice box and returns to the same "
                  "pose, ending exactly on <Picture 2>: Bea sitting still with the juice box at her chest, half-closed eyes.",
                  f"Quiet backyard birdsong, a long gurgling juice-box slurp. {V_BEA}.")),
    dict(id="s05", frames=294, start="keys/s05_a.png", end="keys/s05_b.png", line="Nobody out-digs Wendell Pruitt!", speaker="Wendell",
         prompt=P(f"{WEN}; {BEA}, sipping a juice box; {DD}; and {YARD}",
                  "In a static wide shot, Wendell stomps his foot on a garden spade and shouts (S1) <d>[English] Nobody out-digs Wendell "
                  "Pruitt!</d> He digs in a frantic cartoon blur, dirt flying in every direction, sinking lower and lower into a huge "
                  "crater all around the dandelion, which is never touched, ending exactly on <Picture 2>: only Wendell's dirty head and "
                  "hands poke out of a deep round hole, the spade stuck in a dirt pile behind him, and the dandelion stands untouched on a "
                  "little island of grass in the middle of the hole.",
                  f"A metal spade clang, then rapid-fire digging scrapes and showers of falling dirt, speeding up, then sudden silence and "
                  f"one small cartoon blink. {V_WEN}.")),
    dict(id="s06", frames=294, start="keys/s06_a.png", end="keys/s06_b.png", line="Mom's vacuum. Maximum suck. Say goodbye, flower!",
         speaker="Wendell",
         prompt=P(f"{WEN}; {BEA}, holding a juice box; {DD} on its island in a big dirt hole; Mom's red upright vacuum cleaner with a "
                  f"cloth bag; and {YARD}",
                  "In a static wide shot, Wendell aims the vacuum hose at the dandelion and says (S1) <d>[English] Mom's vacuum. Maximum "
                  "suck. Say goodbye, flower!</d> He switches it on: it roars and sucks up dirt and grass in swirls while the dandelion "
                  "doesn't even bend, the vacuum bag swells up like a balloon and bursts with a bang, and dirt rains down everywhere, "
                  "filling the hole and covering Wendell, ending exactly on <Picture 2>: Wendell standing stiff and dirt-smudged, hair "
                  "blown straight up, holding the limp hose, the bag in tatters, the dandelion untouched on a messy dirt-strewn lawn.",
                  f"A click, a rising vacuum-cleaner roar, a stretching balloon squeak, a loud pop-bang, a long patter of falling dirt, "
                  f"then one blink. {V_WEN}.")),
    dict(id="s07", frames=192, start="keys/s04_a.png", end="keys/s04_a.png", line="Dad gets home in one minute.", speaker="Bea",
         prompt=P(f"{BEA}, holding a juice box with a bendy straw; and the corner of the backyard behind her: wooden plank fence, "
                  "round green bushes with orange flowers, the striped lawn",
                  "In a static medium shot, Bea glances to the right, then says flatly (S1) <d>[English] Dad gets home in one minute.</d> "
                  "She takes a slow sip from her juice box and returns to the same pose, ending exactly on <Picture 2>: Bea sitting still "
                  "with the juice box at her chest, half-closed eyes.",
                  f"A loud ticking clock, a short slurp through a straw. {V_BEA}.")),
    dict(id="s08", frames=294, start="keys/s08_a.png", end="keys/s08_b.png", line="Dad's home?! ... I got it!", speaker="Wendell",
         prompt=P(f"{WEN_DIRTY}; {BEA}, holding a juice box; the dandelion, which has turned into a round white fluffy seed puffball; "
                  f"and {YARD}, now strewn with dirt",
                  "In a static wide shot, Wendell stands slumped and glum next to the puffball. A car horn honks twice off screen; he "
                  "jolts in panic and yelps (S1) <d>[English] Dad's home?!</d> He grabs the puffball and it pops out of the ground "
                  "with its roots as easy as anything; he stares, then lifts it high and cheers (S1) <d>[English] I got it!</d> ending "
                  "exactly on <Picture 2>: Wendell holding the white puffball up beside his face, roots dangling, a huge open-mouthed grin.",
                  f"Two car-horn honks from the street, a panicked yelp, a soft cartoon pop as the root comes free, a triumphant gasp. "
                  f"{V_WEN}.")),
    dict(id="s09", frames=243, start="keys/s09_a.png", end="keys/s09_b.png", line="Victory! So long, sucker!", speaker="Wendell",
         prompt=P(f"{WEN_DIRTY}, holding a white dandelion puffball; and {YARD}, strewn with dirt",
                  "In a static medium shot, Wendell grins at the puffball and says (S1) <d>[English] Victory! So long, sucker!</d> He takes "
                  "an enormous breath, cheeks ballooning, and blows like a birthday candle: hundreds of fluffy dandelion seeds burst off "
                  "and drift across the whole yard, ending exactly on <Picture 2>: Wendell with a smug half-closed-eye grin holding the "
                  "bare green stem, seeds floating everywhere in the air around him.",
                  f"A big inhale, a long cartoon blow like wind, a soft magical whoosh of floating seeds, a satisfied hum. {V_WEN}.")),
    dict(id="s10", frames=294, start="keys/s10_a.png", end="keys/s10_b.png", line="Nooooooo!", speaker="Wendell",
         prompt=P(f"{WEN}; {BEA}, wearing a crown of dandelions and sipping a juice box; and {YARD} the next morning under a warm "
                  "peach-pink sunrise sky, the whole lawn now covered with hundreds of yellow dandelions",
                  "In a static wide shot, Wendell stretches with his arms up and a happy yawn, eyes closed, then opens his eyes and sees "
                  "the lawn. He freezes, his eyes bulge, his hair shoots up and his jaw drops to the floor in an enormous take and he "
                  "wails (S1) <d>[English] Nooooooo!</d> ending exactly on <Picture 2>: Wendell frozen in the take, arms flung out, eyes "
                  "huge, mouth stretched wide open. Bea keeps sipping.",
                  f"A rooster crowing in the distance, morning birdsong, a big contented yawn, a cartoon record-scratch, then a long "
                  f"echoing wail. {V_WEN}.")),
    dict(id="s11", frames=192, start="keys/s11_a.png", end="keys/s11_b.png", line="Best. Lawn. Ever.", speaker="Bea",
         prompt=P(f"{BEA}, wearing a crown of dandelions; and the corner of the backyard behind her in warm morning light: wooden "
                  "plank fence, round green bushes, the lawn crowded with yellow dandelions",
                  "In a static medium shot, Bea holds up a white dandelion puffball, looks right at it and says flatly (S1) <d>[English] "
                  "Best. Lawn. Ever.</d> Then she gently blows: the fluffy seeds float away across the yard, ending exactly on "
                  "<Picture 2>: Bea holding the bare stem with a tiny satisfied smile, seeds drifting in the air beside her.",
                  f"Morning birdsong, a soft gentle blow, a light airy whoosh of floating seeds. {V_BEA}.")),
]


def write(out):
    doc = {"title": TITLE, "shots": [{k: s[k] for k in ("id", "frames", "start", "end")} | {"continues": False, "prompt": s["prompt"]}
                                     for s in SHOTS]}
    json.dump(doc, open(os.path.join(out, "shots.json"), "w"), indent=2)
    return doc


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/home/user/Wan2GP/outputs/shorts/wobble_suburb"
    d = write(out)
    tot = sum(s["frames"] for s in SHOTS)
    print(len(SHOTS), "shots", tot, "frames", tot / FPS, "s; with cards", tot / FPS + TITLE_S + END_S)
    for s in SHOTS:
        words = len(s["line"].replace("...", " ").split())
        print(s["id"], s["frames"], f"{s['frames'] / FPS:.1f}s", words, "words", f"{words / (s['frames'] / FPS - 1):.2f} w/s")
