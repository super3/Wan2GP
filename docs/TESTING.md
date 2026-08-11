# Testing

WanGP is a large application built around GPU-bound generative models, which makes
"just add tests" harder than it sounds: the interesting code paths want CUDA, tens of
gigabytes of checkpoints, and a browser session. This document describes the testing
strategy that works around that, what is covered today, and how to extend it.

## Guiding principle

**The test suite must stay runnable by anyone, on any machine, in seconds.**

That means the suite depends on nothing but the standard library and `pytest`. It does
not import `torch`, `numpy`, `gradio`, `diffusers`, `opencv`, or anything else from
`requirements.txt`. A contributor can clone the repo, `pip install pytest`, and get a
green run without a GPU, without downloading a checkpoint, and without a network
connection.

This is a real constraint, not an aspiration — CI installs only `requirements-test.txt`,
so a stray `import numpy` in a test fails the build immediately.

## Running the tests

```bash
pip install -r requirements-test.txt
pytest
```

To run one file, or one test:

```bash
pytest tests/test_prompt_parser.py
pytest tests/test_prompt_parser.py -k paragraph -v
```

The suite has no ordering requirements and no shared mutable state, so `pytest -x`,
`-k`, and parallel runners all behave.

## What is covered today

The suite targets the pure-python logic that sits between the user and the models — the
code that decides *what* to generate before any tensor is allocated. This is where
user-visible bugs are both most likely and cheapest to catch.

| Test file | Tests | Covers |
| --- | --- | --- |
| `tests/test_prompt_parser.py` | 110 | Splitting the prompt box into queued requests vs. sliding windows; comments, paragraph breaks, CRLF, the macro templating |
| `tests/test_loras_multipliers.py` | 110 | Per-step LoRA strength schedules; phase separators, step interpolation, and the text surgery that preserves user comments through a merge |
| `tests/test_model_configs.py` | 50 | Every bundled model definition parses, resolves to a real handler, and declares no property the code never reads |
| `tests/test_regressions.py` | 27 | One focused test per bug fix whose module-level suite is still in review |
| `tests/test_package_imports.py` | 14 | The pure logic stays importable without torch, so the suite keeps working |

**311 tests, about 1.4 seconds.**

The count is deliberate. An earlier revision of this suite ran to 1589 tests, and an
adversarial review of it found roughly 40% were noise: parametrize lists that expanded
one assertion into thirty-five, expectations rebuilt with the same expression as the
implementation, and — the worst pattern — tests parametrised over the very constant they
validated, so removing an entry made the case *vanish from collection* rather than fail.
The files that survived were pruned against mutation testing rather than taste: each
retained file was checked by reverting real one-line changes in a sandbox copy and
confirming the smaller suite still went red.

Five module-level test files are held back for follow-up pull requests so this change
stays hand-reviewable: `frame_scheduler`, `filename_formatter`, `resolutions`,
`audio_metadata` and `lora_mapper`. Their fixes ship here, guarded by
`tests/test_regressions.py`; each of those tests was verified to fail when its fix is
reverted. As each module-level file lands, its regression tests move into it.

`tests/test_model_configs.py` is worth calling out: with ~212 model definitions in
`defaults/`, a single typo breaks model discovery at startup for everyone. It is a
data-integrity check rather than a unit test, and it is cheap insurance. Its headline
assertion is that every `architecture` is backed by a handler — a mistyped one does not
crash, `init_model_def` sets `visible = False` and the model silently disappears from
the UI. The valid set is recovered by parsing each module in `wgp.py`'s
`family_handlers` with `ast`, because importing a handler would pull in torch. It also
found four properties in `defaults/` that no source file reads, listed in that file.

## Defects the suite surfaced

Writing the tests turned up nine pre-existing defects. All of them are now fixed, and
each has a regression test alongside it. They are recorded here because the *shape* of
these bugs is the argument for the suite: none crashed, most produced quietly wrong
output, and several had clearly been present for a long time.

1. **Multiplier fusion** — `shared/utils/loras_mutipliers.py` `_strip_bars_outside_comments`
   removed phase bars without leaving a separator, so adjacent multipliers fused.
   Through `merge_loras_settings`, `"1|2|3"` yielded the multiplier string `"23"`,
   silently applying a LoRA strength of twenty-three. The bar now becomes a space,
   matching `preparse_loras_multipliers`. It survived because the spaced spelling
   `1 | 2` always worked — only `1|2` fused.

2. **Eager package import** — `shared/utils/__init__.py` re-exported the scheduler
   classes from `fm_solvers` eagerly, pulling in torch and diffusers and making every
   module in the package unreachable without the CUDA stack. Now lazy via PEP 562
   `__getattr__`; see the next section.

3. **Forged checksums** — `shared/tools/sha256_verify.py` a `chunk_size` of 0 made
   `f.read(0)` return immediately, so `compute_sha256` returned the empty-string digest
   for *any* file — and that digest then "verified successfully" against it. A
   non-positive `chunk_size` is now rejected.

4. **Doubled spaces** — `shared/utils/loras_mutipliers.py` `preparse_loras_multipliers`
   split on `" "` rather than whitespace, so `"1.0  0.5"` produced an empty token and
   failed with *"Lora Multiplier no 2 () is invalid"*. Whitespace-only and comment-only
   input now mean "no multipliers given", like an empty box.

5. **`None` prompt mode** — `shared/utils/prompt_parser.py` `serialize_prompt_units`
   lacked the `or ""` guard its sibling splitters have, so a `None` mode raised
   `TypeError`.

6. **Date tokens eating each other** — `shared/utils/filename_formatter.py`
   `_parse_date_format` substituted tokens sequentially over its own output. `MM`
   became `%m`, and the later `mm` token matched the `m` just written: `MMmm` passed
   validation and compiled to `%%Mm`, losing the month. Substitution is now a single
   pass.

7. **Leaked file handles** — `shared/utils/audio_metadata.py` read files with
   `open(path, 'rb').read()` in two places, never closing them. The suite runs clean
   under `-W error::ResourceWarning`.

8. **Stale resolution cache** — `shared/resolutions.py` `_custom_resolutions` was not
    keyed on the filename, so the `resolution_file` argument was ignored after the
    first call and a second file returned the first one's contents. The cache now
    records which file it came from.

9. **Lax architecture conditions** — `shared/match_archi.py` `eval_condition` used
    `re.match` rather than `fullmatch`, which anchors only at the start. That made it
    lax at the end and strict in the middle: `">=89garbage"` silently parsed as
    `">=89"`, while the natural `">= 89"` failed outright. Now a full match, with
    whitespace around the operator accepted. No condition shipped in this repository
    changes meaning: the only two are `<89` and `<999`.

A tenth candidate was investigated and **rejected**: `format()` applies no overall
length cap, which looks like an `ENAMETOOLONG` waiting to happen. It is not reachable.
The sole caller, `wgp.py:7938`, already pipes the result through
`truncate_for_filesystem()` (100 bytes on Linux, 50 on Windows), so a cap inside the
formatter would be dead code. An earlier revision of this branch added one anyway and
introduced a real crash with it — `value.encode('utf-8')` raised `UnicodeEncodeError`
on the lone surrogates that reach filenames through `surrogateescape`. Both were
removed.

## Keeping the pure modules importable

Every test uses a plain `import`. That is only possible because the packages holding
this logic stay light at import time.

`shared/utils/__init__.py` used to re-export the scheduler classes from `fm_solvers`
eagerly, which imports `torch` and `diffusers`. That made `import
shared.utils.prompt_parser` fail even though `prompt_parser.py` needs nothing but `re`,
and it charged every CLI entry point the cost of loading torch just to reach a string
helper. Nothing in the application actually imported those names from the package —
every caller reaches for `shared.utils.fm_solvers` directly — so the cost bought
nothing.

The package now re-exports them lazily via PEP 562 `__getattr__`, so the heavy import
happens on first attribute access rather than at package import. The public names are
unchanged and still show up in `__all__` and `dir()`.

`tests/test_package_imports.py` guards this. It imports each pure module in a
subprocess with `torch`, `diffusers`, `numpy`, `gradio` and `transformers` poisoned at
the meta-path, so a reintroduced top-level import fails loudly instead of passing
quietly on a developer machine that happens to have torch installed.

If you add a module to the suite, add it to `PURE_MODULES` there too.

## Continuous integration

`.github/workflows/tests.yml` runs on every push to `main`, every pull request, and on
demand. Three jobs:

- **pytest** — the suite on Python 3.10, 3.11 and 3.12, the versions `requirements.txt`
  ships wheels for.
- **syntax check** — byte-compiles all first-party sources on each of those versions.
  It imports nothing, so it needs no dependencies, and it catches both plain syntax
  errors and syntax that is not valid on the oldest supported interpreter. `models/` is
  excluded because it holds vendored upstream code, one file of which
  (`models/longcat/modules/block_sparse_attention/flash_attn_bsa_varlen_mask.py`) does
  not parse as valid Python and is not ours to fix.
- **config integrity** — runs the `defaults/*.json` validation on its own so a bad model
  definition shows up as a distinct red check rather than being buried in the unit tests.

The whole workflow completes in well under a minute.

## Adding a test

1. Check the module is importable without the heavy stack:
   `python -c "import shared.your_module"`. If it needs a heavy package `__init__`
   to be made lazy first, do that rather than working around it, and add the module to
   `PURE_MODULES` in `tests/test_package_imports.py`.
2. Add `tests/test_<module>.py`. Plain `pytest` functions; `class Test<Area>:` for
   grouping; `@pytest.mark.parametrize` for table-driven cases.
3. Assert **actual current behaviour**, derived from reading the source. If you find a
   genuine bug, pin the current behaviour with a comment explaining it and open an
   issue — don't leave a red test in `main`.
4. Keep it hermetic: no network, no writes outside `tmp_path`, no dependence on the
   clock (`monkeypatch` it), on the working directory, or on the host GPU.

## Where to go next

The first tier deliberately stops at the dependency boundary. Natural follow-ups, in
increasing order of cost:

**Tier 2 — allow `numpy`.** A single lightweight wheel unlocks another group of modules
that are otherwise pure logic, including `shared/utils/video_metadata.py`,
`shared/utils/hdr.py`, `shared/utils/motion.py`, `shared/utils/audio_cleaning.py`,
`shared/utils/vace_preprocessor.py` and the scheduler maths in
`shared/utils/euler_scheduler.py`. CI cost is a few seconds.

**Tier 3 — allow CPU `torch`.** This makes the schedulers (`fm_solvers`,
`fm_solvers_unipc`, `basic_flowmatch`, `lcm_scheduler`) and the tensor-shaping helpers
testable with tiny synthetic tensors. Worth a separate, slower workflow job that does
not gate every PR — the CPU wheel is a few hundred megabytes.

**Tier 4 — end-to-end smoke tests.** A self-hosted GPU runner generating a handful of
frames from the smallest available model, on a schedule rather than per-PR, would catch
the integration breakage that unit tests structurally cannot.

**Other eager package `__init__` files.** `shared/utils` was the worst offender and is
now lazy, but the same pattern may be hiding elsewhere. Any package whose `__init__`
imports the runtime stack keeps otherwise-pure modules out of reach; making it lazy is
usually a few lines and pays for itself in start-up time.
