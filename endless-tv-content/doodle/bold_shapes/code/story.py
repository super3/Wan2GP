"""The script of 'One Little Smudge' as data: shots, lengths, dialogue and the H3 prompts (fixed descriptions reused verbatim)."""

SERIES = "Squeaky Doyle"
TITLE = "One Little Smudge"
LOGLINE = ("Sky-High City's smallest, bravest window-washer declares his last pane spotless - then spends the afternoon losing a war "
           "against one tiny smudge, until the little old lady inside wipes it away with her hanky.")

ERA = "1990s TV"
SQUEAKY = ("Squeaky, a tiny window-washer boy with an oversized round head, huge blue eyes and no nose, a tilted yellow hard hat with "
           "black hair tufts poking out, orange bib overalls over a white T-shirt, red rubber boots and round fingerless hands, "
           "sitting on a wooden plank seat hung from rope straps")
FIG = ("Mrs. Fig, a tiny elderly office lady with an oversized round head, huge green eyes behind big round red glasses and no nose, "
       "a grey hair bun with a yellow pencil stuck through it, a lilac cardigan with a white collar and pearls, rosy cheeks and "
       "round fingerless hands")
V_SQ = "Squeaky's voice: a small boy about seven years old, high, bright and bouncy, brave and cheerful, accent-free"
V_FIG = "Mrs. Fig's voice: a sweet little old lady about eighty, soft, warm and slightly creaky, gentle and unhurried, accent-free"
SET_WIDE = ("the setting: a sunny flat-colour retro-modern city of mint, teal and lilac buildings with a saucer-topped needle tower, "
            "a tall lilac glass skyscraper with a steel davit arm on its roof, a pale mint sky with a cream sun and flat clouds")
SET_EXT = ("the setting: high up beside a lilac skyscraper's big glass windows, a flat mint-and-teal retro-modern skyline with a "
           "saucer-topped needle tower, a cream sun and flat clouds behind him, and one small brown smudge on the window pane to his right")
SET_INT = ("the setting: a cosy retro office with apricot striped wallpaper, a starburst wall clock, a teal cone desk lamp, a mustard "
           "wooden desk with drawers, a snake plant on the sill and a big picture window onto a mint sky and a flat retro skyline, "
           "with Squeaky hanging in his bosun's chair just outside the glass")


def prompt(n, chars, setting, framing, action, end, sounds):
    return (f"integrated_multimodal_description: [Shot {n}] Continue directly from <Picture 1>, a 2D hand-drawn {ERA} cartoon frame: "
            f"keep exactly its flat colours, outlines and painted background. Preserve {chars}; and {setting}. In a static {framing} shot, "
            f"{action}, ending exactly on <Picture 2>: {end}. The camera does not move. Traditional 2D cel animation, not 3D, not "
            f"photorealistic.\noverall_soundscape: {sounds}.\nnon_diegetic_music: N/A")


# id, frames, start key, end key, continues, framing, setting, chars, action, end pose, sounds, dialogue (speaker, line), beat
SHOTS = [
    dict(id="s01", frames=124, start="s01_a", end="s01_b", continues=False, framing="wide establishing", setting=SET_WIDE, chars=SQUEAKY,
         action=("high on the side of the lilac tower, tiny Squeaky whistles a happy tune as his rope lowers him a little way down, "
                 "then he swings his blue-handled squeegee onto the glass with a flourish"),
         end_pose="Squeaky a little lower down the tower, squeegee resting on the glass, smiling",
         sounds=f"soft high-altitude wind, a creaking pulley and rope, Squeaky whistling a cheerful little tune with no words, a rubbery squeegee squeak, distant traffic far below; {V_SQ}",
         lines=[], beat="Establish: the smallest window-washer in Sky-High City, a hundred floors up."),
    dict(id="s02", frames=192, start="s02_a", end="s02_b", continues=False, framing="medium", setting=SET_EXT, chars=SQUEAKY,
         action=("Squeaky finishes wiping the window with his giant blue-handled squeegee, whistling, then lifts the squeegee upright like "
                 "a flag, puffs out his chest and proudly says (S1) <d>[English] Last window of the day. Spotless!</d> without noticing "
                 "the small brown smudge right behind him"),
         end_pose="squeegee held upright beside him like a flag, a proud happy smile, looking straight at us, the smudge still on the pane",
         sounds=f"soft wind, a rubbery squeegee squeak, a short happy whistle, then Squeaky speaks proudly; {V_SQ}",
         lines=[("Squeaky", "Last window of the day. Spotless!")], beat="He declares victory - the audience can already see the smudge."),
    dict(id="s03", frames=243, start="s03_a", end="s03_b", continues=False, framing="close", setting=SET_EXT, chars=SQUEAKY,
         action=("Squeaky beams proudly, then his eyes slide sideways to the window and spot the smudge; his smile drops, he does a quick "
                 "double take, leans toward it, narrows his eyes and says (S1) <d>[English] Wait. What is that? Oh no you don't.</d> and holds the glare"),
         end_pose="leaning toward the smudge with a determined narrow-eyed glare, mouth set in a firm line, squeegee upright",
         sounds=f"soft wind, a comic double-take blink sound, a tiny squeak of the rope as he leans; {V_SQ}",
         lines=[("Squeaky", "Wait. What is that? Oh no you don't.")], beat="He notices. It's personal now."),
    dict(id="s04", frames=294, start="s04_a", end="s04_b", continues=False, framing="medium", setting=SET_EXT, chars=SQUEAKY,
         action=("after a tense beat, Squeaky, squeegee raised high, yells (S1) <d>[English] Hi-yah!</d> and swipes it down across the smudge in one fast "
                 "stroke, checks, then swipes again even harder with another (S1) <d>[English] Hi-YAH!</d>, then freezes and stares: the "
                 "smudge is still exactly there"),
         end_pose="the squeegee at the bottom of the pane, Squeaky wide-eyed with tiny pupils and a small round open mouth, the smudge untouched",
         sounds=f"two fast whooshes with loud rubbery squeegee squeaks, a rope creak, then a stunned silence with a tiny wind gust; {V_SQ}",
         lines=[("Squeaky", "Hi-yah! ... Hi-YAH!")], beat="Attempt 1: the squeegee. Twice. Nothing."),
    dict(id="s05", frames=294, start="s05_a", end="s05_b", continues=False, framing="close", setting=SET_EXT, chars=SQUEAKY,
         action=("Squeaky leans in with puffed cheeks and huffs a big foggy breath onto the glass around the smudge, says "
                 "(S1) <d>[English] Okay, smudge. The old huff and buff.</d> and rubs the fog furiously with his sleeve; the fog clears, "
                 "the smudge is still there, and his face slowly turns bright red from the chin up as steam puffs out beside his hat"),
         end_pose="a bright red face, gritted teeth, angry slanted eyes, little white steam puffs beside his hat, the smudge still on the glass",
         sounds=f"a long breathy huff, fast squeaky rubbing on glass, then a rising kettle-like whistle of steam; {V_SQ}",
         lines=[("Squeaky", "Okay, smudge. The old huff and buff.")], beat="Attempt 2: huff and buff. He boils over."),
    dict(id="s06", frames=294, start="s06_a", end="s06_b", continues=False, framing="medium", setting=SET_EXT, chars=SQUEAKY,
         action=("Squeaky grins and heaves his red bucket of suds back, says (S1) <d>[English] Time for extra suds!</d> and flings the "
                 "suds at the window; a gust of wind blows the whole splash straight back at him and the empty bucket lands upside down "
                 "on his head, while blobs of foam splat on the glass everywhere except on the smudge"),
         end_pose=("the red bucket upside down on his head instead of his hat, foam dripping down his face and lap, droopy deadpan eyes, "
              "foam blobs on the glass and the smudge untouched in a clear spot"),
         sounds=f"a big sloshing heave, a whoosh of wind, a wet splat, a hollow bucket clonk, foam dripping and fizzing; {V_SQ}",
         lines=[("Squeaky", "Time for extra suds!")], beat="Attempt 3: the bucket. The wind sends it all back."),
    dict(id="s07", frames=294, start="s07_a", end="s07_b", continues=False, framing="medium", setting=SET_EXT, chars=SQUEAKY,
         action=("Squeaky presses a chrome Polish-o-Matic buffer with a big fluffy pad against the glass beside the smudge, grins and says "
                 "(S1) <d>[English] Polish-o-Matic. Full power!</d>; the machine roars and spins so hard that it spins Squeaky and his whole "
                 "plank seat around and around on the rope"),
         end_pose=("the rope twisted above him, Squeaky dizzy with spiral eyes and a wobbly smile, hat knocked sideways, little yellow stars "
              "circling his head, the buffer dangling under the seat on its cord, the smudge still there"),
         sounds=f"a click, a rising electric motor whine, a whirring spin with the rope creaking, then a slowing wobble and little dizzy chimes; {V_SQ}",
         lines=[("Squeaky", "Polish-o-Matic. Full power!")], beat="Attempt 4: the gadget. He polishes himself into a spin."),
    dict(id="s08", frames=362, start="s08_a", end="s08_b", continues=False, framing="interior medium-wide", setting=SET_INT,
         chars=f"{FIG}, and {SQUEAKY}",
         action=("Mrs. Fig sits at her desk holding a teacup and watches through her window as Squeaky, outside, scrubs angrily at the "
                 "smudge with his squeegee; she peers closer, sets her teacup down, takes out a white lace hanky, leans over, and with one "
                 "dainty wipe on the inside of the glass the smudge is gone with a tiny sparkle; she says softly (S2) <d>[English] "
                 "Goodness. There we are.</d> while outside Squeaky freezes and his jaw drops"),
         end_pose=("Mrs. Fig leaning toward the window with her hanky on the glass, eyes closed in a pleased little smile, her teacup on the "
              "desk; Squeaky outside wide-eyed with his mouth hanging open"),
         sounds=f"a quiet office: a ticking wall clock, muffled squeegee squeaking through the glass, a teacup clink on a saucer, one soft wipe, a tiny sparkle chime; {V_FIG}; {V_SQ}",
         lines=[("Mrs. Fig", "Goodness. There we are.")], beat="The reveal: the smudge was on the INSIDE."),
    dict(id="s09", frames=243, start="s08_b", end="s09_b", continues=True, framing="interior medium-wide", setting=SET_INT,
         chars=f"{FIG}, and {SQUEAKY}",
         action=("Mrs. Fig sits back, picks up her teacup and sips contentedly; outside the glass Squeaky blinks, closes his mouth, lowers "
                 "his squeegee, gives an awkward little shrug and says (S1) <d>[English] I... loosened it for you.</d> and holds his awkward smile"),
         end_pose=("Squeaky with a wobbly sheepish smile, half-lidded eyes and a sweat drop; Mrs. Fig sipping her tea with her eyes closed, "
              "the hanky in her other hand"),
         sounds=f"a ticking wall clock, a gentle slurp of tea, a small awkward cough; {V_SQ}; {V_FIG}",
         lines=[("Squeaky", "I... loosened it for you.")], beat="Saving face."),
    dict(id="s10", frames=362, start="s09_b", end="s10_b", continues=True, framing="interior medium-wide", setting=SET_INT,
         chars=f"{FIG}, and {SQUEAKY}",
         action=("Squeaky admires the sparkling clean glass, leans in and presses his whole face flat against the window to see his "
                 "reflection, squish, then pulls back and drifts aside, leaving a big greasy print of his face on the glass; Mrs. Fig "
                 "looks up from her tea, unimpressed, sets the cup down and slowly raises her hanky; Squeaky grins sheepishly and says "
                 "(S1) <d>[English] Same time tomorrow?</d> and keeps grinning"),
         end_pose=("Squeaky grinning sheepishly with a sweat drop, his grey face print on the glass beside him; Mrs. Fig with a flat unamused "
              "look, holding her hanky up, her teacup on the desk"),
         sounds=f"a squishy squeak of a face on glass, a sticky peel-off pop, a ticking clock, a teacup set down, one tiny sigh; {V_SQ}; {V_FIG}",
         lines=[("Squeaky", "Same time tomorrow?")], beat="Button: his own face print. Same time tomorrow."),
]


def shot_prompts():
    out = []
    for n, s in enumerate(SHOTS, 1):
        out.append(prompt(n, s["chars"], s["setting"], s["framing"], s["action"], s["end_pose"], s["sounds"]))
    return out
