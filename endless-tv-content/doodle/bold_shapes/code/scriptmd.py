"""Writes script.md for the short from story.py and the real timeline."""
import os

import story


def tc(sec):
    return f"{int(sec // 60)}:{sec % 60:05.2f}"


def make(out, timeline_total):
    tl, total = timeline_total
    T = {name: (f0 / 24, n / 24) for name, f0, n in tl}
    L = []
    L.append(f"# {story.SERIES}: *{story.TITLE}*\n")
    L.append(f"**Logline.** {story.LOGLINE}\n")
    L.append(f"**Runtime.** {tc(total / 24)} ({total} frames at 24 fps): 3 s title card, 10 shots, 2 s end card. "
             "Style: late-90s 'bold shapes' TV cartoon (UPA-descended): thick black outlines, flat candy colours, "
             "geometric characters, flat retro-modern backgrounds.\n")
    L.append("## Characters\n")
    L.append("- **Squeaky Doyle**: the smallest, bravest window-washer in Sky-High City. Wide bean head, huge blue eyes, no nose, "
             "tilted yellow hard hat with black hair tufts, orange bib overalls over a white tee, red rubber boots, mitten hands. "
             "Works from a bosun's chair (plank seat on rope straps) with a red suds bucket and a giant blue squeegee. "
             "Treats every pane like a super-mission.  \n"
             f"  *Fixed voice (S1):* \"{story.V_SQ.split(': ', 1)[1]}\"")
    L.append("- **Mrs. Fig** (original supporting character): a tiny elderly office lady who has worked at the same desk for fifty years. "
             "Big round red glasses over huge green eyes, grey bun with a pencil stuck through it, lilac cardigan, white collar, "
             "pearls, rosy cheeks. Tea, a lace hanky, infinite calm.  \n"
             f"  *Fixed voice (S2):* \"{story.V_FIG.split(': ', 1)[1]}\"\n")
    L.append("## Beat sheet\n")
    beats = [
        ("Setup (0:03-0:16)", "A hundred floors up, Squeaky finishes the day's last pane and declares it *spotless*. The audience can "
                              "already see the little brown smudge right behind him."),
        ("Inciting gag (0:16-0:26)", "He notices. Double take. 'Oh no you don't.'"),
        ("Escalation (0:26-1:15)", "Four attempts, each bigger and each failing: (1) the squeegee, twice, full force; (2) the old "
                                   "huff-and-buff, which turns his face red; (3) extra suds, which the wind blows back so the "
                                   "bucket lands on his head; (4) the Polish-o-Matic, which spins him dizzy on his rope."),
        ("Twist / payoff (1:15-1:30)", "Cut inside. Mrs. Fig looks up from her tea, and with one dainty wipe of her hanky removes the "
                                       "smudge. It was on the INSIDE the whole time."),
        ("Save (1:30-1:40)", "Squeaky, jaw on the floor, saves face: 'I... loosened it for you.'"),
        ("Button (1:40-1:55)", "He admires the shining glass, presses his face to it, and leaves a big face print. Mrs. Fig raises "
                               "her hanky. 'Same time tomorrow?'  End card: her hanky wipes the print away."),
    ]
    for h, b in beats:
        L.append(f"- **{h}**: {b}")
    L.append("\n## Shot list\n")
    L.append("| # | in (tc) | length | set-up | start -> end key | action | dialogue |")
    L.append("|---|---|---|---|---|---|---|")
    L.append(f"| title | 0:00.00 | 72 f (3.0 s) | card | - | Glass with the smudge; a giant squeegee wipes on the title... the "
             "smudge survives; Squeaky pops up to glare at it. | - |")
    for s in story.SHOTS:
        t0, d = T[s["id"]]
        dl = "<br>".join(f"**{w}:** \"{ln}\"" for w, ln in s["lines"]) or "(whistling, no words)"
        cont = " (continues)" if s["continues"] else ""
        L.append(f"| {s['id']} | {tc(t0)} | {s['frames']} f ({s['frames'] / 24:.2f} s){cont} | {s['framing']} | "
                 f"`{s['start']}` -> `{s['end']}` | {s['beat']} | {dl} |")
    t0, d = T["end"]
    L.append(f"| end | {tc(t0)} | 48 f (2.0 s) | card | - | 'THE END' over Squeaky's face print; Mrs. Fig's hanky wipes it away. | - |")
    L.append("\n## Music (music_bed.wav)\n")
    L.append("An original 60s spy-lounge cartoon score in one piece, silent under the cards (they carry their own stings), "
             "changing cue exactly on each cut: Squeaky's whistled theme (s01-s02, with a 'ta-da' on *Spotless*); a "
             "double-take tuba sting and pizzicato spy ostinato (s03); a surf-guitar action riff with two brass stabs and a sad "
             "trombone (s04); a creeping chromatic boil with a kettle-whistle glide (s05); a fanfare, timpani roll, splash and "
             "soggy tuba bwomps (s06); an accelerating gadget motor riff and dizzy wobbly vibes (s07); a tea-time celesta minuet "
             "and a record scratch (s08); awkward plunks and a sheepish muted 'wah-wah' (s09); a minuet reprise, a squish 'bloop', "
             "an 'uh-oh', and the theme returning into the end card (s10). The assembly ducks it ~9 dB under the H3 dialogue.\n")
    L.append("## Assembly\n")
    L.append("`python3 <code folder>/assemble.py RENDERS_DIR OUT.mp4` (stdlib + ffmpeg only). RENDERS_DIR holds `s01.mp4` ... "
             "`s10.mp4`. Each render is conformed to its exact frame count, and s09/s10 drop their first frame. The music bed is ducked "
             "under the H3 audio and the mix is loudness-normalised to -16 LUFS. Output: 1280x720 H.264/AAC. The code folder is "
             "`/tmp/claude-0/-home-user-Wan2GP/4e1e6010-b786-5f93-b6fd-87ee44cba9e7/scratchpad/styles/bold_shapes/`.\n")
    with open(os.path.join(out, "script.md"), "w") as f:
        f.write("\n".join(L) + "\n")
