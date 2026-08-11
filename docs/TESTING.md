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

The first tier targets the pure-python logic that sits between the user and the
models — the code that decides *what* to generate before any tensor is allocated.
This is where user-visible bugs are both most likely and cheapest to catch.

| Test file | Covers | Why it matters |
| --- | --- | --- |
| `tests/test_prompt_parser.py` | `shared/utils/prompt_parser.py` | Splitting the prompt box into queued requests vs. sliding windows; mode aliases, comments, paragraph breaks, CRLF |
| `tests/test_loras_multipliers.py` | `shared/utils/loras_mutipliers.py` | Per-step LoRA strength schedules — phase separators, step interpolation, malformed input |
| `tests/test_frame_scheduler.py` | `shared/utils/frame_scheduler.py` | Sliding-window frame maths and in-prompt slash commands; rounding and overlap boundaries |
| `tests/test_filename_formatter.py` | `shared/utils/filename_formatter.py` | Output filename templating, sanitisation of unsafe characters, length limits |
| `tests/test_resolutions.py` | `shared/resolutions.py`, `shared/match_archi.py` | Resolution parsing/grouping and GPU architecture detection |
| `tests/test_lora_mapper.py` | `shared/lora_mapper.py`, `shared/utils/gguf_mapping.py`, `shared/tools/sha256_verify.py` | Key remapping between checkpoint formats; checksum verification |
| `tests/test_audio_metadata.py` | `shared/utils/audio_metadata.py` | Binary metadata chunk round-tripping, truncated and non-audio files |
| `tests/test_model_configs.py` | `defaults/*.json`, `plugins.json`, `setup_config.json` | Every bundled model definition parses and has the shape the loader expects |

`tests/test_model_configs.py` is worth calling out: with ~212 model definitions in
`defaults/`, a single typo breaks model discovery at startup for everyone. It is a
data-integrity check rather than a unit test, and it is cheap insurance.

## The `import_pure_module` helper

Several stdlib-only modules live inside packages whose `__init__` eagerly imports the
heavy stack. `shared/utils/__init__.py`, for example, imports `fm_solvers`, which
imports `torch` and `diffusers`. A plain `import shared.utils.prompt_parser` therefore
fails even though `prompt_parser.py` itself only needs `re`.

`tests/conftest.py` provides:

```python
from conftest import import_pure_module

prompt_parser = import_pure_module("shared.utils.prompt_parser")
```

It registers a lightweight stand-in for the parent package in `sys.modules` before
importing the submodule, so the submodule loads normally (relative imports included)
while the expensive `__init__` never runs.

Modules that already import cleanly — `shared.resolutions`, `shared.lora_mapper`,
`shared.match_archi`, `shared.tools.sha256_verify` — should just use a plain `import`.

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
   `python -c "import shared.your_module"`, or via `import_pure_module` if it lives
   under `shared/utils/`.
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

**Refactor opportunity.** Making `shared/utils/__init__.py` import lazily would remove
the need for `import_pure_module` and would also speed up every CLI entry point that
currently pays for `torch` + `diffusers` just to reach a string helper. That is a source
change rather than a test change, so it is deliberately out of scope here.
