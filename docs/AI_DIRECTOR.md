# The AI director brief

A constraint sheet for an LLM asked to write a multi-shot film for MiniMax H3.
The premise: **do not fight the model's discontinuities — spend them as cuts.**
A real film cuts every few seconds anyway, so a boundary where motion and audio
restart is normal grammar, not a defect. The director's job is to never write a
shot that needs continuity the pipeline cannot deliver.

## Hard limits

| Constraint | Value | Consequence for the script |
|---|---|---|
| clip length | ≤ **362 frames = 15.08 s** (24 fps) | no beat may run longer than 15 s without a cut |
| legal lengths | `17n + 5`, min 107 | 107 (4.5 s), 124 (5.2 s), 243 (10.1 s), 362 (15.1 s) |
| resolution | 832×480 | wide framing; faces small in wide shots read poorly |
| motion across clips | **none** | never write "continuing the move from the last shot" |
| audio across clips | **none** | score, ambience and room tone restart every clip |
| keyframe consistency | style yes, identity **maybe** | same palette/lens across a grid; the same *face* is not guaranteed |
| start frame | anchors shot 1 reliably | the strongest control you have |
| start **and** end frame | over 15 s this is a long interpolation | **unverified** — the middle may go slack; prefer ≤ 5 s spans until measured |

## What the director controls well

- **The first frame of every clip.** A keyframe fixes composition, palette,
  wardrobe and lighting exactly. Write shots that *begin* on an image.
- **Cuts inside a clip.** 3–4 shots inside 10 s works today. Internal cuts get
  full motion and audio continuity — the opposite of cross-clip cuts. Put the
  continuity-dependent moments *inside* a clip, always.
- **Per-shot sound design.** Each clip's audio is coherent within itself.

## Rules that follow

1. **Every clip boundary is a hard cut.** Change angle, scale or location across
   it. A boundary between two near-identical framings reads as a glitch; between
   a wide and a close-up it reads as editing.
2. **Never split an action across a boundary.** A punch, a fall, a door opening
   must complete inside one clip. Cross-clip means cause and effect land in
   different generations and the physics will not agree.
3. **Put the cause and its consequence in the same clip,** cutting *on* the
   impact rather than between impact and result. (Learned the hard way: a cut
   placed between a collision and the reaction produced a bird strike where the
   rider fell off a second later, unrelated.)
4. **Do not write sustaining music.** `non_diegetic_music` restarts every clip.
   Either keep it to a stab or sting that resolves inside the shot, or omit it
   and lay a continuous bed over the whole film in post.
5. **Prefer per-shot ambience that plausibly changes** — a new location, a new
   distance — so the audio restart is motivated.
6. **Budget identity risk.** A recurring hero across twelve clips will drift.
   Either accept it as stylisation, keep the hero in fewer shots, or shoot them
   away from camera in the risky ones.
7. **One idea per clip.** 15 s is roughly one beat.

## Prompt format (per clip)

Three labelled blocks, in this order:

```
integrated_multimodal_description: [Shot 1] … [Shot 2] At 00:04.000, cut to …
overall_soundscape: …
non_diegetic_music: …
```

- Internal cuts: `[Shot N]` plus `At MM:SS.mmm`, strictly increasing, all inside
  the clip's runtime.
- Dialogue: `(S1) <d>[English] line here.</d>` — omit entirely for silent shots.
- Describe what is *visible*, not backstory the camera cannot see.

## Shape of a one-minute film

Four keyframes from one 2×2 board → four 15 s clips → 60 s, all generated in
parallel. Wall-clock is one clip (~80 s), not four. Scale by adding boards: three
boards → twelve clips → ~3 minutes, still one clip of wall-clock given twelve
workers. Cost per minute is unchanged — parallelism buys latency, not money.
