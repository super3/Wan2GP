"""The shot list for "The Littlest Crossing": writes shots.json and script.md from one source."""
import json
import os

OUT = "/home/user/Wan2GP/outputs/shorts/retro_exaggerated"
TITLE = "Bertram Bigsby in The Littlest Crossing"
SERIES = "Bertram Bigsby"
EPISODE = "The Littlest Crossing"
LOGLINE = ("The proudest crossing guard in town is so magnificently top-heavy that he cannot see the tiny tortoise who needs his "
           "help, so he heroically escorts everyone and everything except her.")
TITLE_FRAMES, END_FRAMES = 72, 48

V_BERTRAM = ("Bertram (S1) has a deep, booming, pompous middle-aged male baritone voice, slow and theatrical, very pleased with "
             "himself.")
V_TILLY = "Tilly (S2) has a small, calm, polite little girl's voice, soft, dry and matter-of-fact."
D_BERTRAM = ("Bertram, an enormous top-heavy brown bison crossing guard (a giant scalloped wool hump and chest on a tiny tan waist, "
             "a small head with ivory horns and a woolly beard, a tiny navy cap with a gold star, an orange diagonal sash with a "
             "gold badge, navy trousers on pin-thin legs with tiny black shoes)")
D_SIGN = "his round red stop sign with a white hand on a white pole"
D_TILLY = ("Tilly, a tiny green tortoise schoolgirl (an amber hexagon-patterned shell, round glasses, a red bobble hat with a "
           "white pompom, a yellow satchel)")
D_HYDRANT = "a stubby red fire hydrant"
D_DUCKS = "five small yellow ducklings in little blue caps"
SET_CORNER = ("the pastel 1960s street corner: a beige sidewalk, the blue-grey road with a zebra crossing, a yellow house, a salmon "
              "schoolhouse with a diamond-pattern wall, lollipop trees and lumpy white clouds")
SET_LOW = ("the beige sidewalk at ground level, with the blue-grey road, a zebra crossing and green hedges behind")
SET_CROSS = ("the pastel 1960s street seen straight down the road: a zebra crossing in the foreground, beige sidewalks on both "
             "sides, a long salmon schoolhouse on the right, a green house and lollipop trees on the left, hills on the horizon")

HEAD = ("[Shot 1] Continue directly from <Picture 1>, a 2D hand-drawn 1990s TV cartoon frame: keep exactly its flat colours, "
        "thick black outlines and painted background.")
TAIL = "The camera does not move. Traditional 2D cel animation, not 3D, not photorealistic."

SHOTS = [
    dict(id="s01", frames=243, start="s01_a", end="s01_b", continues=False, beat="Setup",
         keep=f"Preserve {D_BERTRAM} holding {D_SIGN}, {D_TILLY}, and {SET_CORNER}.",
         action=("In a static wide shot, Bertram stands proudly at the curb while tiny Tilly waits quietly beside his feet. After a "
                 "short pause he puffs out his huge chest, raises one fist high, lifts his chin, half-closes his eyes and booms "
                 "(S1) <d>[English] Bertram Bigsby, crossing guard! Nobody crosses this street without me.</d> Below him Tilly "
                 "politely raises her hand, but he never looks down, ending exactly on <Picture 2>: Bertram puffed up with his fist "
                 "raised and a smug smile, Tilly holding her hand up."),
         sound="a quiet suburban street with distant birds, a rubbery stretch as his chest swells. " + V_BERTRAM,
         speaker="Bertram", line="Bertram Bigsby, crossing guard! Nobody crosses this street without me."),
    dict(id="s02", frames=243, start="s02_a", end="s02_b", continues=False, beat="Premise: he cannot see her",
         keep=f"Preserve {D_TILLY}, the navy trouser legs and tiny black shoes of Bertram the bison, his tan belly overhead and {D_SIGN}, and {SET_LOW}.",
         action=("In a static low shot at sidewalk level, under Bertram's enormous belly, tiny Tilly looks up with her hand raised. "
                 "After a short pause she says (S2) <d>[English] Excuse me, sir? Down here. I'd like to cross, please.</d> Nothing "
                 "happens above her. She lowers her hand, gives a flat, unimpressed look and reaches out to tap his trouser leg, "
                 "ending exactly on <Picture 2>: Tilly reaching toward his leg with a deadpan, half-lidded look. Bertram's legs do not move."),
         sound="soft street ambience, two small taps on fabric. " + V_TILLY,
         speaker="Tilly", line="Excuse me, sir? Down here. I'd like to cross, please."),
    dict(id="s03", frames=192, start="s03_a", end="s03_b", continues=False, beat="He tries to look down",
         keep=f"Preserve {D_BERTRAM}, {D_SIGN} standing beside him, and {SET_CORNER}.",
         action=("In a static medium shot, Bertram cups a hand to his ear and glances left and right, then grabs his sign, leans "
                 "forward and strains to look down past his own enormous chest, eyes bulging, and says (S1) <d>[English] I hear a "
                 "pedestrian... but all I can see is magnificence.</d> ending exactly on <Picture 2>: Bertram bent forward with his "
                 "head tilted down, eyes wide, mouth in a wobbly strain, one hand on the sign pole."),
         sound="a creak of effort as he leans, a small grunt. " + V_BERTRAM,
         speaker="Bertram", line="I hear a pedestrian... but all I can see is magnificence."),
    dict(id="s04", frames=362, start="s04_a", end="s04_b", continues=False, beat="Attempt 1: he crosses alone",
         keep=f"Preserve {D_BERTRAM} with {D_SIGN}, {D_TILLY}, {D_HYDRANT}, and {SET_CROSS}.",
         action=("In a static wide shot looking straight down the street, Bertram stands on the left sidewalk with tiny Tilly and the "
                 "red fire hydrant beside his feet. He lifts his sign high and booms (S1) <d>[English] Fear not, citizen! Halt, "
                 "traffic!</d> at the completely empty road, then marches grandly across the zebra crossing to the right sidewalk, all "
                 "alone, and announces (S1) <d>[English] Safely across! You're welcome!</d> Tilly and the hydrant stay where they were on "
                 "the left curb, ending exactly on <Picture 2>: Bertram on the right sidewalk with his sign raised and a fist in the "
                 "air, tiny Tilly still waiting on the left curb."),
         sound="a big proud intake of breath, heavy stomping footsteps across the road, a single distant bird tweet. " + V_BERTRAM,
         speaker="Bertram", line="Fear not, citizen! Halt, traffic! ... Safely across! You're welcome!"),
    dict(id="s05", frames=192, start="s04_b", end="s05_b", continues=True, beat="She's still over there",
         keep=f"Preserve {D_BERTRAM} with {D_SIGN}, {D_TILLY}, {D_HYDRANT}, and {SET_CROSS}.",
         action=("In a static wide shot, continuing the same moment: on the left curb tiny Tilly sighs and says (S2) <d>[English] "
                 "Mr. Bigsby? I'm still over here.</d> then puts her hand over her glasses. Across the road Bertram plants his sign, "
                 "cups a hand to his ear and looks around in confusion, ending exactly on <Picture 2>: Tilly facepalming on the left "
                 "curb, Bertram on the right sidewalk with a hand cupped to his ear and his sign standing beside him."),
         sound="a small sigh, a light tap of the sign pole on the pavement. " + V_TILLY,
         speaker="Tilly", line="Mr. Bigsby? I'm still over here."),
    dict(id="s06", frames=294, start="s06_a", end="s06_b", continues=False, beat="Attempt 2: wrong pedestrian",
         keep=f"Preserve {D_BERTRAM} with {D_SIGN}, {D_TILLY}, {D_HYDRANT}, and {SET_CROSS}.",
         action=("In a static wide shot, Bertram is back on the left sidewalk, holding his sign in one hand and leaning forward, "
                 "groping blindly below his own huge chest with the other hand. His hand closes on the red fire hydrant instead of "
                 "Tilly; he lifts it gently, cradles it in his arm like a baby and says (S1) <d>[English] Gently does it... there we "
                 "are. Mind the curb, little one.</d> then tiptoes out onto the zebra crossing with it, while Tilly stays on the curb "
                 "with her hands on her hips, ending exactly on <Picture 2>: Bertram in the middle of the road cradling the hydrant "
                 "and smiling down at it, Tilly deadpan on the left curb."),
         sound="fumbling pats on metal, a clunk as the hydrant comes loose, soft tiptoe footsteps. " + V_BERTRAM,
         speaker="Bertram", line="Gently does it... there we are. Mind the curb, little one."),
    dict(id="s07", frames=243, start="s07_a", end="s07_b", continues=False, beat="The hydrant squirts back",
         keep=f"Preserve {D_BERTRAM} holding {D_SIGN} in one hand and {D_HYDRANT} in the other arm, {D_TILLY} small on the curb, and {SET_CROSS}.",
         action=("In a static medium shot in the middle of the road, Bertram cradles the red fire hydrant like a baby and coos "
                 "(S1) <d>[English] My, what a quiet little pedestrian you are.</d> He leans closer to it; suddenly a cap on the "
                 "hydrant pops off and a jet of water squirts him square in the face. He freezes, eyes squeezed shut, soaked and "
                 "dripping, ending exactly on <Picture 2>: Bertram drenched, his wool flattened and dark, water dripping, eyes "
                 "squeezed shut, the hydrant still spraying at his face."),
         sound="a soft coo, a metallic pop, a loud water gush and splash, then dripping. " + V_BERTRAM,
         speaker="Bertram", line="My, what a quiet little pedestrian you are."),
    dict(id="s08", frames=243, start="s08_a", end="s08_b", continues=False, beat="Glum; Tilly takes charge",
         keep=f"Preserve {D_BERTRAM}, soaked with dark flattened wool and dripping, holding {D_SIGN}, {D_TILLY}, and {SET_CROSS}.",
         action=("In a static medium-wide shot at the left curb, soggy Bertram stands slumped and glum, dripping, his free arm "
                 "hanging, holding his sign upright beside him, and mutters (S1) <d>[English] Perhaps there was no pedestrian at "
                 "all.</d> Right beside him, unseen, tiny Tilly grabs the sign pole and calmly climbs all the way up it, then sits on "
                 "top of the round sign, ending exactly on <Picture 2>: Tilly sitting on top of the sign, Bertram still glum and "
                 "dripping, looking down at the ground."),
         sound="steady water drips, a sad squelch, small scrabbling climbing sounds on the pole. " + V_BERTRAM,
         speaker="Bertram", line="Perhaps there was no pedestrian at all."),
    dict(id="s09", frames=192, start="s08_b", end="s09_b", continues=True, beat="He finally sees her",
         keep=f"Preserve {D_BERTRAM}, soaked and dripping, holding {D_SIGN}, {D_TILLY} sitting on top of the sign, and {SET_CROSS}.",
         action=("In a static medium-wide shot, continuing the same moment: Bertram sighs and lifts his sign high to try once more, "
                 "raising Tilly with it, and raises his other fist. He looks up, his eyes pop wide and his face bursts into a huge "
                 "grin as Tilly waves and says (S2) <d>[English] Hello. Can we go now?</d> ending exactly on <Picture 2>: Bertram "
                 "holding the sign high and grinning up at Tilly, who sits on top of the sign waving."),
         sound="a heavy sigh, a whoosh as the sign goes up, a cartoon eye-pop. " + V_TILLY,
         speaker="Tilly", line="Hello. Can we go now?"),
    dict(id="s10", frames=294, start="s10_a", end="s10_b", continues=False, beat="Payoff: the grand crossing",
         keep=f"Preserve {D_BERTRAM} holding {D_SIGN}, {D_TILLY}, {D_HYDRANT} on the right sidewalk, and {SET_CROSS}.",
         action=("In a static wide shot looking down the street, Bertram marches proudly across the zebra crossing from the left "
                 "sidewalk to the right, holding his sign high with tiny Tilly riding on top of it and his other fist in the air, "
                 "booming (S1) <d>[English] Make way! Make way for the littlest pedestrian in town!</d> On the right sidewalk he "
                 "lowers the sign, Tilly hops down beside the fire hydrant and waves goodbye, ending exactly on <Picture 2>: "
                 "Bertram puffed up proudly with his fist raised and eyes closed, Tilly waving from the sidewalk."),
         sound="a jaunty proud march of heavy footsteps, a small hop and a light landing. " + V_BERTRAM,
         speaker="Bertram", line="Make way! Make way for the littlest pedestrian in town!"),
    dict(id="s11", frames=243, start="s11_a", end="s11_b", continues=False, beat="Button: more tiny pedestrians",
         keep=f"Preserve {D_BERTRAM} holding {D_SIGN}, {D_DUCKS}, and {SET_CORNER}.",
         action=("In a static wide shot back at his corner, Bertram stands proudly with his sign; at his feet, unnoticed, five small "
                 "yellow ducklings in little blue caps wait in a neat row. He puffs out his chest, raises a fist, closes his eyes "
                 "smugly and declares (S1) <d>[English] Another flawless crossing. Not a single pedestrian left in town!</d> Below "
                 "him all five ducklings raise a wing and quack hopefully, ending exactly on <Picture 2>: Bertram puffed up with his "
                 "fist raised and eyes closed, the five ducklings with their wings raised."),
         sound="a quiet suburban street, a rubbery stretch as his chest swells, then five tiny quacks together. " + V_BERTRAM,
         speaker="Bertram", line="Another flawless crossing. Not a single pedestrian left in town!"),
]


def prompt(s):
    return (f"integrated_multimodal_description: {HEAD} {s['keep']} {s['action']} {TAIL}\n"
            f"overall_soundscape: {s['sound']}\n"
            f"non_diegetic_music: N/A")


def timeline():
    """(id, start_frame, n_frames_on_screen) on the assembled timeline, after the title card."""
    t = TITLE_FRAMES
    out = []
    for s in SHOTS:
        n = s["frames"] - (1 if s["continues"] else 0)
        out.append((s["id"], t, n))
        t += n
    return out, t + END_FRAMES


def write():
    os.makedirs(OUT, exist_ok=True)
    js = {"title": TITLE, "shots": [{"id": s["id"], "frames": s["frames"], "start": f"keys/{s['start']}.png",
                                     "end": f"keys/{s['end']}.png", "continues": s["continues"], "prompt": prompt(s)} for s in SHOTS]}
    with open(os.path.join(OUT, "shots.json"), "w") as f:
        json.dump(js, f, indent=2)
    tl, total = timeline()
    lines = [f"# {EPISODE}", "", f"*A {SERIES} cartoon. Series: {SERIES}. Episode: {EPISODE}.*", "",
             f"**Logline.** {LOGLINE}", "", f"**Runtime.** {total / 24:.1f} s ({total} frames at 24 fps): title card "
             f"{TITLE_FRAMES / 24:.0f} s + {len(SHOTS)} shots + end card {END_FRAMES / 24:.0f} s.", "",
             "## Characters (fixed designs and voices)", "",
             f"- **Bertram Bigsby** (S1): {D_BERTRAM[len('Bertram, '):]}. Vain, booming, sure the traffic stops because it admires "
             f"him. His chest is so big he cannot see anything below it. Voice: {V_BERTRAM[len('Bertram (S1) has '):]}",
             f"- **Tilly Tuckett** (S2): {D_TILLY[len('Tilly, '):]}. Unflappable, polite and patient. Voice: "
             f"{V_TILLY[len('Tilly (S2) has '):]}",
             "- Props and extras: the round red hand-sign on a pole, a stubby red fire hydrant, five ducklings in blue caps.", "",
             "## Beat sheet", "",
             "1. **Setup (s01).** Bertram brags that nobody crosses without him; tiny Tilly waits right under his chest, unseen.",
             "2. **Premise (s02-s03).** Tilly asks to cross; he hears her, but when he tries to look down all he can see is his own chest.",
             "3. **Attempt 1 (s04-s05).** He halts an empty road and marches across alone. Tilly is still on the curb.",
             "4. **Attempt 2 (s06-s07).** He gropes blindly, scoops up a fire hydrant instead, cradles it across the road, and it squirts him.",
             "5. **Low point (s08).** Soggy and glum, he decides there was no pedestrian. Tilly quietly climbs his sign.",
             "6. **Payoff (s09-s10).** He lifts the sign and finally sees her; triumphant crossing with Tilly riding on top.",
             "7. **Button (s11).** Back on his corner he declares the town free of pedestrians, as five ducklings wait unseen at his feet.",
             "", "## Shot list", "", "| # | Frames (s) | Start -> end key | Beat | Speaker | Dialogue |", "|---|---|---|---|---|---|"]
    for s, (_, t0, n) in zip(SHOTS, tl):
        cont = " (continues)" if s["continues"] else ""
        lines.append(f"| {s['id']} | {s['frames']} ({s['frames'] / 24:.1f}){cont} | {s['start']} -> {s['end']} | {s['beat']} | "
                     f"{s['speaker']} | \"{s['line']}\" |")
    lines += ["", "Title card (3 s) before s01, end card (2 s) after s11. Music is a separate synthesised score (`music_bed.wav`);"
              " every shot prompt has `non_diegetic_music: N/A`.", ""]
    with open(os.path.join(OUT, "script.md"), "w") as f:
        f.write("\n".join(lines))
    return js


if __name__ == "__main__":
    js = write()
    tl, total = timeline()
    for s in SHOTS:
        words = len(s["line"].replace("...", " ").split())
        print(s["id"], s["frames"], f"{s['frames'] / 24:.1f}s", words, "words", f"{words / (s['frames'] / 24 - 1):.2f} w/s")
    print("total", total, "frames", total / 24, "s")
