#!/usr/bin/env python3
"""The four 2-minute Laugh Track sets (8 x 15 s clips each) for the other comedian stills, as bakeoff_box.py jobs.

    comic_sets.py WHO[,WHO...] SCENE[,SCENE...] RUN_ID OUT.json   (SCENE: the box's still slot for each comedian)"""
import json
import sys

STAGE = "the red brick wall, the warm spotlight, the mic stand and the wooden stool with a glass of water"
WHO = {
    "bob": dict(she=True, seed=2000, keep="her sharp black bob, the small gold hoop earrings, the oversized camel blazer over a white t-shirt "
                "and the handheld microphone"),
    "cardigan": dict(she=False, seed=3000, keep="his shaved head, the neatly trimmed grey-flecked beard, the round tortoiseshell glasses, "
                     "the burgundy cardigan over a collared shirt and the handheld microphone"),
    "silver": dict(she=True, seed=4000, keep="her silver pixie cut, the bold red lipstick, the black leather jacket over a sequined top and "
                   "the handheld microphone"),
    "flannel": dict(she=False, seed=5000, keep="his short dark wavy hair, the thin moustache, the unbuttoned red flannel shirt over a grey "
                    "t-shirt and the handheld microphone with its cord"),
}
# (action before line 1, line 1, action before line 2, line 2) per clip; clip 8 is the closer
BITS = {
    "bob": [
        ("smiles as the applause settles, one eyebrow raised", "Thank you! So, my mom just learned how to text. Which is great. For her.",
         "tilts her head", "She signs every message. Love, Mom. In case I think it's a different mom."),
        ("gives the audience a dry, knowing look", "She also discovered emojis. Now every text ends with a skull. Happy birthday, skull.",
         "shrugs", "Dinner on Sunday? Skull. I don't know if that's a threat or a menu."),
        ("leans toward the audience", "Then she joined the family group chat. Twenty-two people. Nobody knows who added my dentist.",
         "nods slowly", "He's very active. He reacts to every baby photo with a reminder to floss."),
        ("smiles with one eyebrow raised", "I work from home now. My commute is eleven steps. Twelve if I stop at the fridge.",
         "holds up a finger", "Which I do. Every time. My coworkers think I'm dedicated."),
        ("gives the audience a dry look", "You lose track of time at home. Yesterday I said good morning in a meeting. It was four p.m.",
         "shrugs", "Nobody corrected me. Everybody just said good morning back."),
        ("raises one eyebrow", "My mom calls during my meetings. I decline. She calls again. Then she texts: are you dead? Skull.",
         "deadpans", "I said, Mom, I'm working. She said, so you're not dead. Good. Call your aunt."),
        ("leans in as if sharing a secret", "My aunt knows everything. I bought new shoes online, at midnight, in my bedroom.",
         "spreads her free hand", "The next morning she called to ask why I need new shoes."),
        ("smiles at the audience", "Anyway, I have to go. My mom is texting me right now. I can feel it.",
         "raises the microphone", "You've been wonderful. Goodnight! Skull!"),
    ],
    "cardigan": [
        ("smiles warmly as the applause settles", "Thank you, thank you. I'm forty-six. I make a noise every time I sit down. Not a word. Just a noise.",
         "chuckles", "My kids think it's a language. They call it Dad."),
        ("leans toward the audience", "I have two teenagers. They don't talk to me. They talk near me.",
         "nods slowly", "Like the weather. I just get the forecast. Mostly cloudy. Needs twenty dollars."),
        ("adjusts his glasses", "My son is learning to drive. I'm in the passenger seat holding the handle like a parachute.",
         "shakes his head", "He said, Dad, relax, I did this in a video game. I said, son, that's what worries me."),
        ("sighs with a smile", "My daughter says everything I do is cringe. I waved at her school. Cringe.",
         "points at his own smile", "I said, sweetheart, I paid for those braces. Half of that smile is mine."),
        ("smiles fondly", "My wife and I have date night every Friday. We walk around the hardware store holding hands.",
         "raises his eyebrows", "Very romantic. Aisle twelve. Lighting."),
        ("tilts his head back to peer through his glasses", "I got progressive lenses. Now I read a menu like a confused pigeon.",
         "chuckles", "The waiter thinks I'm judging him. I'm just looking for the prices."),
        ("leans in", "I joined a gym. My trainer is twenty-two. His name is Tyler. Tyler says, push through the pain.",
         "spreads his free hand", "Tyler, the pain is where I live now."),
        ("smiles at the audience", "Alright, I gotta go. It's nine-thirty. That's late for me.",
         "raises the microphone", "You've been wonderful. Drive safe, use your blinkers. Goodnight!"),
    ],
    "silver": [
        ("gives the audience a wry smirk as the applause settles", "Thank you, darlings. I'm sixty-three, and I just got my first tattoo.",
         "taps her wrist", "It's a little reminder of who I am. It says, take your pills."),
        ("smirks", "I'm retired now. People ask what I do all day. I tell them, whatever I want. Slowly.",
         "shrugs", "Yesterday I spent an hour choosing a sandwich. No regrets."),
        ("rolls her eyes", "My grandson set up my smartphone. Now it listens to everything I say.",
         "deadpans", "This morning I said I need a nap. It ordered me a mattress."),
        ("leans toward the audience", "I tried online dating. Every man my age has a photo holding a fish.",
         "raises an eyebrow", "Sir, I don't need a man who can catch dinner. I need one who can find his glasses."),
        ("smirks", "My doctor told me to cut back on wine. So I bought smaller glasses.",
         "pauses, then nods", "Then I bought a lot more of them."),
        ("puts her hand back on her hip", "My kids worry about me living alone. I'm not alone. I have a cat and a neighbor named Doreen.",
         "narrows her eyes", "Doreen knows when I get home before I do."),
        ("gives a knowing look", "People say sixty is the new forty. So I tried it. I stayed up past ten.",
         "shakes her head", "It took me three days to recover. Sixty is the new sixty."),
        ("smiles at the audience", "Well, that's my time. Literally. It's almost nine.",
         "raises the microphone", "You've been lovely. Call your mothers. Goodnight, darlings!"),
    ],
    "flannel": [
        ("grins eagerly as the applause settles", "What's up, everybody! I just got my first real job. I have a desk. I have a badge.",
         "shrugs", "I don't know what I do yet, but I do it from nine to five."),
        ("widens his eyes", "My boss keeps saying, let's circle back. We circle back so much I get dizzy.",
         "spins one finger in the air", "Last week we had a meeting about circling back. We circled back to it."),
        ("leans toward the audience", "I live with three roommates. One fridge. A very strict label system.",
         "holds up a hand", "Somebody labeled the milk, not yours. Which is true for everybody."),
        ("grins sheepishly", "I started cooking to save money. Day one, I set off the smoke alarm making cereal.",
         "spreads his hands", "Don't ask me how. The firefighter didn't know either."),
        ("smiles fondly", "My mom calls every Sunday. Are you eating? Yes, Mom. What are you eating? Food.",
         "mimics a stern look", "She says, describe it. Like it's a police lineup."),
        ("widens his eyes", "I joined a gym. First day, I watched a guy deadlift four hundred pounds.",
         "flexes proudly", "I went home and carried all my groceries in one trip. That's my deadlift."),
        ("grins", "Being an adult is weird. I got so excited about a new vacuum, I vacuumed at midnight. Twice.",
         "nods", "My roommates called a meeting. They said, let's circle back."),
        ("smiles at the audience", "Alright, that's my time. I gotta go. I have work in the morning. I think.",
         "raises the microphone", "You guys are awesome. Thank you so much. Goodnight!"),
    ],
}


def prompt(who, n, bit):
    spec = WHO[who]
    he, his = ("she", "her") if spec["she"] else ("he", "his")
    a1, l1, a2, l2 = bit
    head = ("integrated_multimodal_description: [Shot 1] Continue directly from <Picture 1>, preserving the comedian, "
            f"{spec['keep']}, and {STAGE}. In a static medium shot {he} {a1} and says (S1) <d>[English] {l1}</d> ")
    if n == 8:
        body = (f"The audience laughs; {he} {a2} and adds (S1) <d>[English] {l2}</d> The audience cheers and applauds; {he} gives a "
                f"small wave, then returns to the same stance, matching <Picture 2> at the end. The camera does not move.\n"
                f"overall_soundscape: Small comedy club room tone; {his} clear voice through the club's speakers; the unseen audience "
                "laughs after the punchline, then cheers and applauds at the end.\nnon_diegetic_music: N/A")
    else:
        body = (f"The audience laughs; {he} {a2} and adds (S1) <d>[English] {l2}</d> {he.capitalize()} lets the laugh land, then "
                f"returns to the same stance, matching <Picture 2> at the end. The camera does not move.\n"
                f"overall_soundscape: Small comedy club room tone; {his} clear voice through the club's speakers; the unseen audience "
                "laughs and claps after each punchline.\nnon_diegetic_music: N/A")
    return head + body


def jobs(who, scene):
    return [{"scene": scene, "variant": "larry4", "resolution": "1280x704", "frames": 362, "seed": WHO[who]["seed"] + n,
             "audio_refinement": "enabled", "prompt": prompt(who, n, bit), "name": f"set_{who}_clip{n}"}
            for n, bit in enumerate(BITS[who], 1)]


if __name__ == "__main__":
    whos, scenes, run_id, out = sys.argv[1].split(","), sys.argv[2].split(","), sys.argv[3], sys.argv[4]
    all_jobs = [job for who, scene in zip(whos, scenes) for job in jobs(who, scene)]
    json.dump({"id": run_id, "cmd": "jobs", "jobs": all_jobs}, open(out, "w"))
    print(run_id, len(all_jobs), "jobs", max(len(j["prompt"]) for j in all_jobs), "max prompt chars")
