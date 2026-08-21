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


def depth_of(node_id: str) -> int:
    depths = {"n0": 1, "n_van": 2, "n_bakery": 2,
              "n_ride": 3, "n_glitter": 3, "n_eyes": 3, "n_cat": 3}
    return depths.get(node_id, 4)


def parent_of(node_id: str) -> str | None:
    """The scene whose last frame this one starts from (FL2V continuity).
    end_roll is reachable from two parents; the FIRST in encounter order is
    canonical, so the other transition is a cut rather than a match."""
    for pid in ORDER:
        for _label, target in NODES[pid]["choices"]:
            if target == node_id:
                return pid
    return None


def public_tree() -> list[dict]:
    """The story without prompts, in encounter order -- what the page needs."""
    return [{"id": nid, "depth": depth_of(nid),
             "title": NODES[nid]["title"], "desc": NODES[nid]["desc"],
             "choices": NODES[nid]["choices"]}
            for nid in ORDER]
