# SPDX-License-Identifier: Apache-2.0
"""The Biscuit adventure: a fixed choose-your-path story tree.

Why the tree lives in code: the story IS the product configuration. Scenes
are generated once, stored durably, and shared by every player -- so the
definition wants to be reviewed in a diff, not edited in a database.

Structure: 4 layers deep -- 1 opening, 2 branches, 4 middles, 7 endings
(end_roll is reachable from two parents, so 14 unique scenes). ORDER is the
sequence a player could first encounter each scene (breadth-first, reading
order within a depth); the background renderer works through it top to
bottom so the earliest-needed clips exist first.

Every prompt is a single continuous 15-second shot and repeats the same
character description, because each scene is an independent generation and
the corgi must look like the same corgi in all of them.
"""

from __future__ import annotations

BISCUIT = ("Biscuit, a small cheerful corgi with a copper-and-white coat, "
           "oversized upright ears, and short eager legs")

STORY_ID = "biscuit"

#: Seconds per scene and the pixel size every scene is rendered at.
SCENE_DURATION_S = 15
SCENE_RESOLUTION = "832x480"


def _p(shot: str, sound: str) -> str:
    return (f"integrated_multimodal_description: [Shot 1] {shot}\n"
            f"overall_soundscape: {sound}\n"
            "non_diegetic_music: None.")


#: id -> {title, desc, choices [(label, next_id), ...], prompt}
NODES: dict[str, dict] = {
    "n0": {
        "title": "The Open Gate",
        "desc": ("You are Biscuit, a corgi of considerable ambition. This morning the "
                 "garden gate swings open in the breeze. The mail van idles at the curb, "
                 "and somewhere down the street the bakery is pulling cinnamon rolls out "
                 "of the oven."),
        "choices": [["Chase the mail van", "n_van"],
                    ["Follow the cinnamon smell", "n_bakery"]],
        "prompt": _p(
            f"A sunlit suburban garden in the early morning. {BISCUIT} stands at a white "
            "wooden garden gate as it slowly swings open in the breeze. The camera starts "
            "low at dog height behind the corgi, then pushes gently past the gate to "
            "reveal a quiet street: a boxy mail van idling at the curb on the left, and "
            "far down the sidewalk on the right, a small bakery with a striped awning and "
            "a faint curl of steam from its door. The corgi's ears perk and its head "
            "swivels between the two, tail wagging fast, deciding.",
            "Morning birdsong, a soft breeze, the creak of the gate hinge, the low idle "
            "of a van engine, the corgi's excited panting."),
    },
    "n_van": {
        "title": "The Mail Van",
        "desc": ("You catch the van at the corner. Dave the mail carrier knows you. He "
                 "holds a dog treat in one hand and a package addressed to YOUR house in "
                 "the other — and it is leaking glitter."),
        "choices": [["Ride along with Dave", "n_ride"],
                    ["Investigate the glitter package", "n_glitter"]],
        "prompt": _p(
            f"A tracking shot follows {BISCUIT} sprinting joyfully along a suburban "
            "sidewalk, ears flattened by speed, catching up to a boxy white mail van "
            "stopped at a corner. A friendly middle-aged mail carrier in a blue uniform "
            "crouches by the open door, holding out a small bone-shaped treat in one hand "
            "and a cardboard package in the other; a thin sparkle of golden glitter leaks "
            "from the package seam and drifts in the sunlight. The corgi skids to a stop, "
            "tail a blur, looking back and forth between the treat and the sparkling box.",
            "Quick paw taps on pavement, panting, a van door sliding, a friendly chuckle, "
            "the faint papery hiss of leaking glitter, light traffic ambience."),
    },
    "n_bakery": {
        "title": "The Bakery Door",
        "desc": ("Flour dust, warm sugar, and Marisol the baker blocking the doorway with "
                 "a broom and a grin. Behind her: forty cinnamon rolls cooling on a rack, "
                 "guarded by a cat named Espresso."),
        "choices": [["Deploy maximum puppy eyes", "n_eyes"],
                    ["Team up with Espresso the cat", "n_cat"]],
        "prompt": _p(
            f"{BISCUIT} trots up to the open door of a small warm bakery, nose high and "
            "sniffing visibly. In the doorway stands a smiling baker in a flour-dusted "
            "apron, gently blocking the way with a broom held sideways. Behind her, "
            "racks of golden cinnamon rolls steam under warm pendant lights, and a sleek "
            "black cat sits on the counter like a guard, tail curled, watching the corgi "
            "with slow-blinking eyes. Flour motes drift through shafts of morning light "
            "as the camera slowly dollies in over the corgi's shoulder.",
            "A bakery oven humming, trays clinking, the soft whisk of a broom, a "
            "curious sniffing snout, one low meow, cozy indoor ambience."),
    },
    "n_ride": {
        "title": "Ride Along",
        "desc": ("You ride shotgun with your head out the window. Every house on the "
                 "route has a dog, and every dog watches you pass like you have just "
                 "been elected mayor."),
        "choices": [["Bark the good news to everyone", "end_mayor"],
                    ["Hop out at the park", "end_park"]],
        "prompt": _p(
            f"Interior of a mail van, bright day. {BISCUIT} rides in the passenger seat "
            "with its head out the open window, ears streaming in the wind, tongue out, "
            "utterly delighted. The camera rides alongside outside the van, tracking the "
            "corgi's face, then pulls wide to show a friendly suburban street sliding by: "
            "in front yard after front yard, dogs of every size stand at their fences and "
            "turn their heads in unison to watch the corgi pass like a dignitary.",
            "Wind flutter over ears, a steady van engine, passing sprinklers, a chorus of "
            "distant assorted barks doppler-shifting by, the corgi's happy panting."),
    },
    "n_glitter": {
        "title": "The Glitter Package",
        "desc": ("You nose the box open. Inside: a birthday banner, party hats, and an "
                 "invitation with your name on it — today, 3 PM, your own backyard."),
        "choices": [["Sprint home to investigate", "end_party"],
                    ["Pretend you saw nothing", "end_surprise"]],
        "prompt": _p(
            f"Close on a cardboard package on a sunny sidewalk. {BISCUIT} noses the flap "
            "open and golden glitter puffs up in a little sparkling cloud around its "
            "muzzle. The camera cranes slowly over the corgi's head to look down into the "
            "box: a folded rainbow birthday banner, a stack of striped cone party hats, "
            "and a card with a bone motif lying on top. The corgi's head tilts hard to "
            "one side, ears at full alert, glitter settling on its copper fur.",
            "Cardboard flaps rustling, a fine glittery shimmer like soft sand pouring, "
            "one surprised snort and sneeze, quiet street ambience, birdsong."),
    },
    "n_eyes": {
        "title": "Maximum Puppy Eyes",
        "desc": ("You sit. You tilt your head. You deploy the full arsenal. Marisol "
                 "lasts eleven seconds — a personal best for her — before reaching for "
                 "the day-old tray."),
        "choices": [["Accept the roll graciously", "end_roll"],
                    ["Share it with Espresso", "end_friends"]],
        "prompt": _p(
            f"Inside the warm bakery doorway, {BISCUIT} sits perfectly straight on the "
            "tile floor and performs devastating puppy eyes at the baker: huge shining "
            "dark eyes, a slow head tilt to the left, one small paw lift. The camera cuts "
            "between the corgi's imploring face in close-up and the baker's crumbling "
            "resolve as she leans on her broom, laughing, then turns to reach for a tray "
            "of day-old cinnamon rolls behind the counter. Warm light, drifting flour "
            "dust, the black cat watching from the counter with grudging respect.",
            "A soft pleading whine, a woman's warm laughter, apron fabric rustling, a "
            "metal tray sliding off a rack, bakery oven hum."),
    },
    "n_cat": {
        "title": "An Unlikely Alliance",
        "desc": ("Espresso blinks slowly, which in cat means \"I'm listening.\" The plan: "
                 "you provide the distraction, Espresso works the latch. Nobody explains "
                 "how a cat knows latches. You don't ask."),
        "choices": [["Execute the heist", "end_heist"],
                    ["Abort — Marisol saw everything", "end_roll"]],
        "prompt": _p(
            f"A conspiratorial low-angle shot inside the bakery. {BISCUIT} and a sleek "
            "black cat sit side by side behind a flour sack, heads close together like "
            "two spies planning a job. The corgi glances toward the counter where "
            "cinnamon rolls cool on a wire rack; the cat follows the look, then slowly "
            "blinks and flexes one paw. The camera slides slowly around the pair, "
            "framing the rolls in the background between their silhouettes, dramatic "
            "warm side-light, flour dust hanging like fog in a heist movie.",
            "Hushed bakery hum, a fridge compressor, one quiet meow, a soft "
            "determined dog grumble, a distant timer ding."),
    },
    "end_mayor": {
        "title": "Mayor Biscuit",
        "desc": ("By noon, every dog in town has heard. By dinner, someone has made you "
                 "a tiny sash. You didn't run for office, but you won something better: "
                 "name recognition."),
        "choices": [],
        "prompt": _p(
            f"A triumphant slow-motion-free parade moment on a suburban street at golden "
            f"afternoon. {BISCUIT} trots proudly down the middle of the sidewalk wearing "
            "a tiny red sash across its chest, chin high, ears bouncing. Dogs line the "
            "route at their fences, tails wagging; a kid on a bike rings a bell; someone "
            "waves from a porch. The camera tracks backward in front of the corgi at its "
            "eye level, then rises into a wide hero shot of the whole cheerful street.",
            "A chorus of happy barks near and far, a bicycle bell, porch wind chimes, "
            "warm afternoon breeze, proud little paw steps."),
    },
    "end_park": {
        "title": "The Park",
        "desc": ("Six tennis balls, two new friends, and one heroic catch that will be "
                 "discussed at the dog park for years. You come home tired, muddy, and "
                 "completely satisfied."),
        "choices": [],
        "prompt": _p(
            f"A wide green dog park in bright afternoon light. {BISCUIT} rockets across "
            "the grass after a tennis ball, mud flecks flying, alongside a bounding "
            "golden retriever and a tiny terrier. The ball arcs high; the camera whips "
            "up to follow it against the blue sky, then back down as the corgi leaps -- "
            "an improbably heroic catch mid-air -- and tumbles over in the grass, ball "
            "in mouth, immediately mobbed by its two delighted new friends.",
            "Galloping paws on turf, three different happy barks, a ball bouncing, "
            "distant fetch whistles, wind through park trees."),
    },
    "end_party": {
        "title": "The Party",
        "desc": ("You burst through the gate at full speed and skid into a yard full of "
                 "balloons. Everyone you love is there. The cake is peanut butter. It "
                 "was for you all along."),
        "choices": [],
        "prompt": _p(
            f"A backyard birthday party in warm afternoon light. {BISCUIT} bursts "
            "through the garden gate at full sprint and skids to a stop on the lawn, "
            "ears flying, as a yard full of balloons and paper streamers is revealed. "
            "A small table holds a peanut-butter cake with a single candle shaped like "
            "a bone. Family and neighborhood friends turn with a cheer; a toddler "
            "throws confetti. The camera swings around the corgi in a joyful half-orbit "
            "as it spins in a circle, tail helicoptering, under a banner reading a "
            "single word: BISCUIT.",
            "A burst of party horns and laughter, balloons squeaking, confetti "
            "fluttering, one ecstatic bark, garden birdsong beneath it all."),
    },
    "end_surprise": {
        "title": "The Best Actor",
        "desc": ("At 3 PM you act SO surprised. Oscar-worthy. Nobody suspects a thing, "
                 "and the party hat fits perfectly between your ears."),
        "choices": [],
        "prompt": _p(
            f"A backyard party at three in the afternoon. {BISCUIT} walks in through the "
            "gate and performs an enormous theatrical double-take at the balloons and "
            "cake: freezing mid-step, eyes going wide, one ear flopping, then a dramatic "
            "gasp-like head rear. Guests applaud the performance. Someone gently sets a "
            "tiny striped cone party hat between the corgi's big ears, where it fits "
            "perfectly. The camera pushes slowly in on the corgi's proud, terrible "
            "acting, ending on a close-up wink-like blink under the party hat.",
            "A crowd's warm 'surpriiise!', laughter and applause, a paper hat elastic "
            "snap, camera shutter clicks, festive backyard ambience."),
    },
    "end_roll": {
        "title": "One Perfect Roll",
        "desc": ("Marisol serves it on a real plate, because you are a regular now. Warm "
                 "cinnamon in a sunbeam outside the bakery. Some days are simply "
                 "undefeated."),
        "choices": [],
        "prompt": _p(
            f"Outside the bakery, a single perfect sunbeam falls on the doorstep where "
            f"{BISCUIT} sits before a white ceramic plate holding one warm cinnamon "
            "roll, steam curling up through the light. The baker leans in the doorway "
            "with her broom, watching fondly. The corgi takes one careful, reverent "
            "bite, then closes its eyes in pure bliss, crumbs on its nose, tail "
            "sweeping slowly across the pavement. The camera settles into a gentle "
            "close-up of the world's most contented dog in the sunbeam.",
            "A soft satisfied dog sigh, gentle chewing, a broom settling against a "
            "doorframe, quiet street ambience, one church bell far away."),
    },
    "end_friends": {
        "title": "Espresso & Biscuit",
        "desc": ("You split it fifty-fifty, dog and cat, sitting on the same doorstep. "
                 "The whole street stops to take pictures. A legendary friendship "
                 "begins."),
        "choices": [],
        "prompt": _p(
            f"On the bakery doorstep in soft afternoon light, {BISCUIT} and a sleek "
            "black cat sit shoulder to shoulder, a cinnamon roll split neatly in two "
            "between them on a napkin. They eat their halves in parallel, the cat "
            "dainty, the corgi enthusiastic. Passers-by stop on the sidewalk to raise "
            "their phones; the pair ignore the fame completely. The camera slowly pulls "
            "back from a close two-shot into a wide frame of the little bakery, the odd "
            "couple silhouetted together in its doorway.",
            "Two very different eating sounds side by side, a cat's purr, phone camera "
            "shutters, pedestrians murmuring 'look at them', mellow street ambience."),
    },
    "end_heist": {
        "title": "The Great Roll Heist",
        "desc": ("It works. It absolutely should not have worked, but it works. Marisol "
                 "laughs so hard she gives you a second one, legally."),
        "choices": [],
        "prompt": _p(
            f"A gleeful caper inside the bakery. {BISCUIT} performs a loud distraction "
            "by the door -- spinning after its own tail -- while a black cat pads along "
            "the counter and expertly flips a wire rack latch with one paw. A cinnamon "
            "roll rolls off the rack, bounces once, and the corgi catches it mid-spin "
            "like a trained acrobat. The baker turns, sees everything, and doubles over "
            "laughing, holding out a second roll on her palm. The camera whip-pans "
            "between distraction, heist, and laughing baker, ending on dog and cat "
            "frozen in perfect innocent poses.",
            "Skittering paws on tile, a metal latch click, a roll bouncing, a woman's "
            "helpless laughter filling the room, one proud bark, one smug meow."),
    },
}

#: The order a player could FIRST encounter each scene: breadth-first through
#: the tree, reading order within a depth. The renderer works top to bottom.
ORDER: list[str] = [
    "n0",
    "n_van", "n_bakery",
    "n_ride", "n_glitter", "n_eyes", "n_cat",
    "end_mayor", "end_park", "end_party", "end_surprise",
    "end_roll", "end_friends", "end_heist",
]


# ---------------------------------------------------------------------------
# Space Picnic
# ---------------------------------------------------------------------------

PIP = ("Pip, a golden hamster astronaut with round cheeks and tiny paws, "
       "wearing a small white spacesuit with orange stripes and an open "
       "clear bubble helmet")

SP_NODES: dict[str, dict] = {
    "sp0": {
        "title": "The Floating Basket",
        "desc": ("You are Pip, a hamster with a picnic basket and a window seat "
                 "over Earth. The moment you open the lid, everything floats: "
                 "the sandwich, the grapes, one runaway strawberry, and a whole "
                 "squadron of juice bubbles."),
        "choices": [["Chase the runaway strawberry", "sp_berry"],
                    ["Wrangle the juice bubbles", "sp_juice"]],
        "prompt": _p(
            "The bright observation dome of a space station, Earth glowing "
            f"through the wide windows. {PIP} floats beside a wicker picnic "
            "basket strapped to the floor, opening the lid. A checkered blanket "
            "unfurls in slow motion; a sandwich, a bunch of grapes, and a "
            "bright red strawberry drift out weightlessly in different "
            "directions while wobbling spheres of purple juice escape a bottle. "
            "The camera slowly orbits as the hamster looks between the fleeing "
            "strawberry and the shimmering juice bubbles, whiskers twitching, "
            "deciding.",
            "A soft station hum, gentle ventilation, the creak of the wicker "
            "lid, wet wobbles of floating juice, tiny excited hamster squeaks."),
    },
    "sp_berry": {
        "title": "The Runaway Strawberry",
        "desc": ("The strawberry has a head start and zero respect. It pinballs "
                 "down the corridor toward the greenhouse hatch, right past the "
                 "station's friendly robot arm, which just woke up and would "
                 "love a job."),
        "choices": [["Follow it into the greenhouse", "sp_green"],
                    ["Ask the robot arm for help", "sp_arm"]],
        "prompt": _p(
            f"A tracking shot follows {PIP} paddling through the air down a "
            "white station corridor, chasing a bright red strawberry that "
            "tumbles lazily ahead, glinting in the light. The strawberry "
            "bounces off a porthole, spins past handrails, and drifts toward a "
            "junction: on one side glows the leafy light of a greenhouse module "
            "hatch, on the other a friendly yellow robot arm folded against the "
            "wall blinks awake with a curious tilt. The hamster glances between "
            "them mid-float.",
            "Air whooshing softly, paw taps on handrails, the ping of the "
            "strawberry off glass, a gentle servo whir."),
    },
    "sp_juice": {
        "title": "The Juice Bubbles",
        "desc": ("Dozens of purple juice spheres wobble around the dome like a "
                 "tiny solar system. You could drink your dessert the fun way, "
                 "or there is a perfectly good fan clipped to the wall."),
        "choices": [["Slurp them out of the air", "sp_slurp"],
                    ["Herd them with a fan", "sp_fan"]],
        "prompt": _p(
            f"{PIP} floats in the middle of the observation dome surrounded by "
            "dozens of wobbling purple juice spheres of every size, each "
            "catching the light of Earth below. The hamster spins slowly, boots "
            "over head, gently batting the largest bubble, which splits into a "
            "cloud of smaller ones. On the wall a small handheld fan is clipped "
            "beside the hatch. The camera drifts closer as the hamster looks "
            "from its own open mouth to the fan and back, cheeks puffed with "
            "mischief.",
            "Deep wet wobbles and blips of floating liquid, the station hum, "
            "one long thoughtful hamster squeak."),
    },
    "sp_green": {
        "title": "The Greenhouse",
        "desc": ("The greenhouse smells like tomatoes and rain. Somewhere "
                 "between the vines, the strawberry has finally run out of "
                 "places to hide, and a cherry tomato seems to have joined "
                 "your side."),
        "choices": [["Set the picnic among the plants", "spend_vines"],
                    ["Take it back to the big window", "spend_window"]],
        "prompt": _p(
            f"{PIP} glides through the hatch into a lush greenhouse module "
            "where tomato vines and lettuce grow in spiral racks under warm "
            "pink grow-lights, water droplets floating like tiny lenses. The "
            "strawberry drifts ahead through the leaves and the hamster weaves "
            "after it, brushing past a floating cherry tomato that joins the "
            "chase. The camera follows through the foliage as the hamster "
            "finally cups the strawberry in both paws, triumphant, surrounded "
            "by drifting greenery.",
            "Soft irrigation mist, leaves brushing, a small victorious squeak, "
            "droplets pattering on leaves."),
    },
    "sp_arm": {
        "title": "An Unlikely Butler",
        "desc": ("The robot arm catches the strawberry on the first try, then "
                 "looks at you, if an arm can look, clearly asking: what else "
                 "needs catching?"),
        "choices": [["Let it set the table", "spend_table"],
                    ["Share the picnic with it", "spend_crumbs"]],
        "prompt": _p(
            "The friendly yellow robot arm unfolds from the wall with three "
            "smooth joints and a two-fingered gripper, its status light "
            f"blinking like a wink at {PIP}. In one fluid motion it plucks the "
            "tumbling strawberry from the air and holds it out. The hamster "
            "claps its paws; the arm gently pats the hamster's helmet, then "
            "begins collecting the drifting sandwich and grapes, stacking them "
            "neatly on its gripper like a tray while the camera slowly pulls "
            "back.",
            "Precise servo whirs and soft clicks, a happy squeak, the faint "
            "bump of the strawberry into the gripper, station hum."),
    },
    "sp_slurp": {
        "title": "The Great Slurp",
        "desc": ("You are getting good at this. Cheeks full, mustache purple, "
                 "one last enormous bubble left, wobbling right in front of "
                 "your nose."),
        "choices": [["Take a victory lap", "spend_mustache"],
                    ["Gather everything for the feast", "spend_feast"]],
        "prompt": _p(
            f"{PIP} darts from bubble to bubble in the observation dome, "
            "slurping each purple sphere out of the air with comic little "
            "gulps, cheeks growing rounder with every catch. Each slurp sends "
            "the hamster into a gentle backwards spin against the glow of "
            "Earth. The final and largest bubble wobbles right in front of the "
            "lens; the hamster looks straight into it, its reflection stretched "
            "and upside down, and grins before the last enormous slurp leaves a "
            "purple mustache across its whiskers.",
            "Comic wet slurps and pops, tiny gulps, giggly squeaks between "
            "catches, one big final gulp."),
    },
    "sp_fan": {
        "title": "The Bubble Shepherd",
        "desc": ("The fan hums, and suddenly you are not chasing bubbles "
                 "anymore, you are conducting them. The whole glittering school "
                 "drifts wherever you point."),
        "choices": [["Spin them into a ring", "spend_ring"],
                    ["Round them up for the feast", "spend_feast"]],
        "prompt": _p(
            f"{PIP} clips its boots to a rail, aims the small handheld fan, and "
            "switches it on: a gentle breeze catches the field of purple juice "
            "bubbles and they drift together in a slow glittering current "
            "across the dome. The hamster conducts them like an orchestra, "
            "sweeping the fan in wide arcs, herding stragglers away from air "
            "vents, until the whole shimmering school of spheres streams in "
            "formation past the window with Earth turning below.",
            "A small fan's whirr rising and falling, dozens of soft wobbles "
            "moving together, satisfied squeaks, the deep station hum."),
    },
    "spend_vines": {
        "title": "Picnic in the Vines",
        "desc": ("Dinner is served on four leaves and a floating blanket. The "
                 "tomato stays for dessert. Best seat in orbit, no contest."),
        "choices": [],
        "prompt": _p(
            f"{PIP} spreads the checkered blanket in mid-air between the tomato "
            "vines, pinning its corners to four leaves, and lays out the "
            "strawberry, a floating cherry tomato, and a drifting lettuce leaf "
            "like fine dining. The hamster tucks a leaf under its chin as a "
            "napkin and nibbles the strawberry while grow-lights glow pink "
            "through the foliage and droplets drift like tiny chandeliers. The "
            "camera slowly circles the tiny impossible garden party.",
            "Leaves rustling, soft mist, dainty little nibbles, a contented "
            "squeak melody hummed between bites."),
    },
    "spend_window": {
        "title": "Strawberry Over Earth",
        "desc": ("Just you, one strawberry, and the whole blue planet rolling "
                 "by underneath. Some picnics are worth chasing."),
        "choices": [],
        "prompt": _p(
            f"{PIP} floats cross-legged in front of the observation dome's "
            "widest window, the checkered picnic blanket drifting behind like a "
            "cape, and takes a slow happy bite of the strawberry while Earth "
            "rolls blue and bright below. Crumbs sparkle and drift like tiny "
            "stars around the hamster's whiskers. The camera pulls back gently "
            "until the small round silhouette hangs framed against the planet, "
            "completely content.",
            "A tiny crunch and happy chewing, the deep peaceful hum of the "
            "station, one soft satisfied sigh of a squeak."),
    },
    "spend_table": {
        "title": "The Perfect Table",
        "desc": ("The arm holds the blanket flat like a proper table and pours "
                 "juice into a perfect floating sphere. Fine dining, 400 "
                 "kilometers up."),
        "choices": [],
        "prompt": _p(
            "In the observation dome the yellow robot arm holds the picnic "
            f"blanket perfectly flat like a floating table while {PIP} arranges "
            "the rescued sandwich, grapes, and strawberry on top, everything "
            "gently bobbing in place. The arm extends a second small gripper "
            "holding the juice bottle and pours: the juice forms a neat "
            "wobbling sphere above a cup as the hamster applauds. The camera "
            "circles the absurdly elegant zero-gravity table setting with "
            "Earth beyond the glass.",
            "Smooth servos, the wobble of poured juice, delighted applause of "
            "tiny paws, one proud beep from the arm."),
    },
    "spend_crumbs": {
        "title": "Crumbs for the Robot",
        "desc": ("You share your sandwich with a robot arm that cannot eat, and "
                 "it rocks you like a swing anyway. New best friend: "
                 "confirmed."),
        "choices": [],
        "prompt": _p(
            f"{PIP} sits on the robot arm's gripper like a swing, sharing the "
            "picnic: the hamster takes a bite of the sandwich, then holds a "
            "grape up to the arm's camera lens as if feeding it, and the arm's "
            "status light blinks pink. Together they watch drifting crumbs "
            "sparkle in a sunbeam through the corridor porthole, the arm gently "
            "rocking the hamster back and forth. The camera slowly pulls away "
            "down the corridor as they sway.",
            "A soft servo rocking rhythm, tiny bites, a friendly synth beep "
            "answering each squeak, warm station hum."),
    },
    "spend_mustache": {
        "title": "The Purple Mustache",
        "desc": ("Victory laps, purple mustache, hiccups that spin you faster. "
                 "Nobody in orbit has ever been happier."),
        "choices": [],
        "prompt": _p(
            f"{PIP} tumbles in slow victorious somersaults across the "
            "observation dome, purple juice mustache across its whiskers and "
            "cheeks packed round, giggling between hiccups that each send it "
            "spinning a little faster. The empty juice bottle drifts past; the "
            "hamster salutes it. The camera rotates with the spin so Earth "
            "wheels around behind, until the hamster spreads all four paws wide "
            "and simply floats, grinning up into the lens with the messiest, "
            "happiest face in orbit.",
            "Helpless squeaky giggles interrupted by tiny hiccup pops, the "
            "bottle clinking off glass, one long happy exhale."),
    },
    "spend_feast": {
        "title": "The Zero-G Feast",
        "desc": ("Everything you rescued now orbits you like a tiny solar "
                 "system of snacks. You eat like a ringmaster. The crumbs "
                 "sparkle."),
        "choices": [],
        "prompt": _p(
            f"The whole picnic reassembled: {PIP} floats at the center of the "
            "observation dome with the checkered blanket spread beneath like a "
            "magic carpet, the sandwich, grapes, strawberry, and a "
            "constellation of small juice spheres arranged in a slow orbit "
            "around it like a tiny solar system of snacks. The hamster plucks "
            "items from orbit one by one, taking bites as they pass, arms wide "
            "like a ringmaster, while Earth glows through the window behind.",
            "A gentle carousel of wobbles, cheerful nibbles and gulps, tiny "
            "ringmaster squeaks, the deep contented hum of the station."),
    },
    "spend_ring": {
        "title": "The Bubble Ring",
        "desc": ("The bubbles form a slow golden ring with you in the middle, "
                 "just as the sun rises over Earth. You made a planet ring out "
                 "of juice."),
        "choices": [],
        "prompt": _p(
            f"With one last sweep of the fan, {PIP} bends the stream of juice "
            "bubbles into a slowly turning ring, a tiny glittering planet ring "
            "right there in the dome, and floats into its center. The hamster "
            "hangs at the middle of the orbiting spheres, arms out, as the ring "
            "catches the sunrise coming over Earth's horizon and every bubble "
            "ignites gold at once. The camera pulls back to frame the hamster "
            "crowned by its ring against the window.",
            "The fan winding down to silence, a chorus of soft synchronized "
            "wobbles, an awed tiny squeak, a sunrise-warm hum."),
    },
}

SP_ORDER: list[str] = [
    "sp0",
    "sp_berry", "sp_juice",
    "sp_green", "sp_arm", "sp_slurp", "sp_fan",
    "spend_vines", "spend_window", "spend_table", "spend_crumbs",
    "spend_mustache", "spend_feast", "spend_ring",
]


# ---------------------------------------------------------------------------
# The registry and story-scoped helpers
# ---------------------------------------------------------------------------

#: Every playable story, by URL slug: /adventures/{slug} is the player page
#: and all of a story's API routes hang off it. A new story is a new entry
#: here plus its node definitions -- the routes never change. "card" is the
#: one-liner under its shelf card; "blurb" is the title-card pitch line.
STORIES: dict[str, dict] = {
    "biscuit": {
        "title": "Biscuit",
        "page_title": "Biscuit: A Choose-Your-Path Adventure",
        "card": "A corgi's big day out",
        "blurb": ("A corgi, an open gate, and one big morning. Every choice is "
                  "a new 15-second scene, generated in real time."),
        "nodes": NODES, "order": ORDER,
    },
    "space-picnic": {
        "title": "Space Picnic",
        "page_title": "Space Picnic: A Choose-Your-Path Adventure",
        "card": "A picnic in zero gravity",
        "blurb": ("A hamster, a picnic basket, and zero gravity. Every choice "
                  "is a new 15-second scene, generated in real time."),
        "nodes": SP_NODES, "order": SP_ORDER,
    },
}


def nodes_of(slug: str) -> dict[str, dict]:
    return STORIES[slug]["nodes"]


def order_of(slug: str) -> list[str]:
    return STORIES[slug]["order"]


def depth_of(slug: str, node_id: str) -> int:
    """Computed from the tree: ORDER lists parents before children, so one
    pass assigns every node its first-encounter depth."""
    depths = {order_of(slug)[0]: 1}
    for nid in order_of(slug):
        for _label, target in nodes_of(slug)[nid]["choices"]:
            depths.setdefault(target, depths[nid] + 1)
    return depths[node_id]


def parent_of(slug: str, node_id: str) -> str | None:
    """The scene this one continues from. A shared ending is reachable from
    two parents; the FIRST in encounter order is canonical, so the other
    transition is a cut rather than a continuation."""
    for pid in order_of(slug):
        for _label, target in nodes_of(slug)[pid]["choices"]:
            if target == node_id:
                return pid
    return None


def public_tree(slug: str) -> list[dict]:
    """The story without prompts, in encounter order -- what the page needs."""
    nodes = nodes_of(slug)
    return [{"id": nid, "depth": depth_of(slug, nid),
             "title": nodes[nid]["title"], "desc": nodes[nid]["desc"],
             "choices": nodes[nid]["choices"]}
            for nid in order_of(slug)]
