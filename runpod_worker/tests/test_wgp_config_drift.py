"""The regression test for the boot-time ``KeyError: 'attention_mode'`` trap.

CPU only. No torch, no wgp import, no CUDA, no weights, no network: ``wgp.py`` is
read as *text* and parsed with :mod:`ast`, never imported (importing it calls
``torch.cuda.get_device_capability`` at module scope, ``wgp.py:2508``).

WHY THIS FILE EXISTS
--------------------
``wgp.py`` builds its full ``server_config`` default dict only when the config
file is ABSENT (``wgp.py:2576-2617``, written out at ``wgp.py:2618-2619``). When
the file EXISTS it does ``server_config = json.loads(text)`` (``wgp.py:2623``)
and **replaces the
defaults wholesale**; exactly two keys are ``setdefault``ed afterwards
(``wgp.py:2625``, ``wgp.py:2631``). Every other module-scope *bare subscript* of
``server_config`` is therefore a live ``KeyError`` against any hand-written
config file — and ``import wgp`` happens inside ``shared/api.py:1082``, so the
traceback surfaces as a worker that will not boot, with a stack that looks
nothing like a config problem.

Today that set is ``attention_mode`` (``wgp.py:3301``) plus the three
``*_profile`` keys (``wgp.py:3310-3312``, protected only by a helper call at
``wgp.py:2678`` that a text scan cannot see through). ``config.REQUIRED_WGP_KEYS``
lists all four and ``config.ensure_wgp_config`` refuses to write a file without
them.

This test re-derives that set from the current source on every CI run, so an
upstream bump that adds a fifth bare read fails here instead of in production.

    pytest runpod_worker/tests/test_wgp_config_drift.py -v
"""

from __future__ import annotations

import ast
import copy
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import pytest

from runpod_worker import config as C

# --------------------------------------------------------------------------
# Source location
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
WGP_PY = REPO_ROOT / "wgp.py"
API_PY = REPO_ROOT / "shared" / "api.py"

pytestmark = pytest.mark.skipif(
    not WGP_PY.is_file(),
    reason=f"{WGP_PY} not found: this test only runs inside a WanGP checkout",
)


@lru_cache(maxsize=1)
def _wgp_source() -> str:
    """wgp.py as text. Cached: it is ~14k lines and every test re-scans it."""
    return WGP_PY.read_text(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------
# The scan, done twice: the spec's text scan, and an ast cross-check
# --------------------------------------------------------------------------

#: ``server_config["key"]`` — the *bare subscript* form. ``.get("key", default)``
#: is safe by construction and is deliberately not matched.
SUBSCRIPT_RE = re.compile(r"""server_config\[\s*(["'])([^"']+)\1\s*\]""")


def _is_write(line: str, end: int) -> bool:
    """Whether the subscript ending at ``end`` is the target of an assignment.

    ``server_config["k"] = v`` is a write (it *creates* the key);
    ``server_config["k"] == v`` is a read.
    """
    rest = line[end:].lstrip()
    return rest.startswith("=") and not rest.startswith("==")


@lru_cache(maxsize=2)
def scan_text(source: str) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """The spec's scan: ``(reads, writes)`` -> ``{key: [line, ...]}``.

    Restricted to statements starting in column 0, which is the text-only
    approximation of "executes at import time". The ast scan below is the
    stricter version; :func:`test_text_and_ast_scans_agree` keeps them honest.
    """
    reads: dict[str, list[int]] = {}
    writes: dict[str, list[int]] = {}
    for number, line in enumerate(source.splitlines(), start=1):
        if line[:1].isspace():
            continue  # indented: inside a def/class/if body
        for match in SUBSCRIPT_RE.finditer(line):
            key = match.group(2)
            bucket = writes if _is_write(line, match.end()) else reads
            bucket.setdefault(key, []).append(number)
    return reads, writes


def _body_line_ranges(tree: ast.AST) -> list[tuple[int, int]]:
    """Line ranges that do NOT execute at import: function and class bodies."""
    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            end = getattr(node, "end_lineno", None) or node.lineno
            ranges.append((node.lineno, end))
    return ranges


def _module_scope(line: int, ranges: list[tuple[int, int]]) -> bool:
    return not any(start <= line <= end for start, end in ranges)


def _const_str(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


@lru_cache(maxsize=2)
def scan_ast(source: str) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """``(reads, guards)`` for every module-scope ``server_config`` access.

    A *guard* is anything that guarantees the key exists from that line on:

    * ``server_config["k"] = ...``            (assignment target)
    * ``server_config.setdefault("k", ...)``
    * ``"k" in server_config`` / ``"k" not in server_config`` (the
      ``wgp.py:3331-3349`` idiom, which always assigns in its body)

    A guard on line *g* protects reads on lines ``>= g`` only: ``wgp.py:3331+``
    guards nothing that ``wgp.py:3301`` reads.
    """
    tree = ast.parse(source, filename=str(WGP_PY))
    ranges = _body_line_ranges(tree)
    reads: dict[str, list[int]] = {}
    guards: dict[str, list[int]] = {}

    for node in ast.walk(tree):
        if not _module_scope(getattr(node, "lineno", 0), ranges):
            continue
        if isinstance(node, ast.Subscript):
            target = node.value
            if not (isinstance(target, ast.Name) and target.id == "server_config"):
                continue
            key = _const_str(node.slice)
            if key is None:
                continue
            if isinstance(node.ctx, ast.Store):
                guards.setdefault(key, []).append(node.lineno)
            else:
                reads.setdefault(key, []).append(node.lineno)
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "setdefault"
                and isinstance(func.value, ast.Name)
                and func.value.id == "server_config"
                and node.args
            ):
                key = _const_str(node.args[0])
                if key is not None:
                    guards.setdefault(key, []).append(node.lineno)
        elif isinstance(node, ast.Compare):
            if not node.ops or not isinstance(node.ops[0], (ast.In, ast.NotIn)):
                continue
            container = node.comparators[0]
            if isinstance(container, ast.Name) and container.id == "server_config":
                key = _const_str(node.left)
                if key is not None:
                    guards.setdefault(key, []).append(node.lineno)
    return reads, guards


def unguarded_reads() -> dict[str, int]:
    """``{key: first unguarded read line}`` — the set our config MUST supply."""
    source = _wgp_source()
    # The scans are cached, so copy before merging: mutating the cached dicts
    # would make a later call in the same session see this call's union.
    reads, guards = (copy.deepcopy(part) for part in scan_ast(source))
    text_reads, text_writes = scan_text(source)
    for key, lines in text_reads.items():  # union: whichever scan sees more wins
        reads.setdefault(key, []).extend(line for line in lines if line not in reads.get(key, []))
    for key, lines in text_writes.items():
        guards.setdefault(key, []).extend(line for line in lines if line not in guards.get(key, []))

    out: dict[str, int] = {}
    for key, lines in reads.items():
        for line in sorted(lines):
            if not any(guard <= line for guard in guards.get(key, ())):
                out[key] = line
                break
    return out


FIX_INSTRUCTIONS = """
HOW TO FIX (in this order):

  1. runpod_worker/wgp_config.json.tmpl  — add the key with the same default
     wgp.py itself uses in its default-config block (wgp.py:2576-2617).
  2. runpod_worker/config.py             — add the key to REQUIRED_WGP_KEYS so
     ensure_wgp_config() refuses to write a config without it.
  3. runpod_worker/config.py             — if the worker must OWN the value
     (not merely supply a default), also add it to authoritative_keys().

WHY IT MATTERS: wgp.py replaces its entire default server_config with the parsed
file when config/wgp_config.json exists (wgp.py:2620-2623). A bare subscript that
the file does not carry raises KeyError during `import wgp`, inside
shared/api.py:1082 -- i.e. the worker never boots and RunPod reports an unhealthy
worker with a traceback that mentions neither the config nor the missing key's
purpose.
"""


# --------------------------------------------------------------------------
# The test that matters
# --------------------------------------------------------------------------

def test_required_wgp_keys_cover_every_unguarded_read():
    derived = unguarded_reads()
    missing = {key: line for key, line in derived.items() if key not in C.REQUIRED_WGP_KEYS}
    assert not missing, (
        "wgp.py now reads server_config[...] at module scope for "
        f"{sorted(missing)}, which runpod_worker/config.py does not guarantee.\n"
        + "\n".join(
            f"  wgp.py:{line}  server_config[{key!r}]"
            for key, line in sorted(missing.items())
        )
        + "\n"
        + FIX_INSTRUCTIONS
    )


def test_the_scan_itself_still_finds_attention_mode():
    """Canary: a subset assertion passes vacuously if the scan silently breaks.

    ``attention_mode`` is the read the whole file exists for. If it stops being
    detected, the scan is broken (regex drift, wgp.py restructured, the key
    renamed) — not fixed.
    """
    derived = unguarded_reads()
    assert "attention_mode" in derived, (
        "the wgp.py scan no longer finds the unguarded server_config['attention_mode'] "
        "read (expected around wgp.py:3301). Either upstream guarded it — in which "
        "case relax this canary — or this test has gone blind and is no longer "
        "protecting anything."
    )
    assert derived["attention_mode"] > 0


def test_derived_set_is_the_documented_four():
    """Informational pin: today the answer is exactly these four keys.

    Not a subset assertion — this one fails when upstream *removes* a read too,
    which is the moment to re-check the comment block in config.py.
    """
    derived = set(unguarded_reads())
    assert derived == {"attention_mode", "video_profile", "image_profile", "audio_profile"}, (
        f"the unguarded module-scope server_config reads changed to {sorted(derived)}.\n"
        "If keys were ADDED see the instructions below; if keys were REMOVED, update "
        "the comment block in runpod_worker/config.py (and optionally trim "
        "REQUIRED_WGP_KEYS, though keeping an extra key costs nothing).\n"
        + FIX_INSTRUCTIONS
    )


def test_text_and_ast_scans_agree():
    """The spec's column-0 text scan and the ast scan must see the same reads.

    They can legitimately diverge (a read nested in a module-scope ``if`` is
    invisible to the text scan), and when they do the ast scan is authoritative —
    but the divergence should be noticed, not silent.
    """
    source = _wgp_source()
    text_reads, _ = scan_text(source)
    ast_reads, _ = scan_ast(source)
    only_text = set(text_reads) - set(ast_reads)
    assert not only_text, (
        f"the text scan sees module-scope reads the ast scan does not: {sorted(only_text)}. "
        "That means the ast scan is under-reporting and the derived set cannot be trusted."
    )


def test_guard_detection_recognises_the_wgp_idioms():
    """The three guard forms must all be recognised, or the scan over-reports."""
    _, guards = scan_ast(_wgp_source())
    # wgp.py:2626-2629 assignment, read at 2630
    assert "multi_prompts_gen_type" in guards
    # wgp.py:2625 setdefault
    assert "prompt_enhancer_quantization" in guards
    # wgp.py:3331 `if not "video_output_codec" in server_config: ...`
    assert "video_output_codec" in guards
    assert "multi_prompts_gen_type" not in unguarded_reads()


SYNTHETIC = '''
import json
server_config = json.loads(text)
server_config.setdefault("guarded_by_setdefault", 1)
server_config["guarded_by_assignment"] = 2
if "guarded_by_membership" not in server_config: server_config["guarded_by_membership"] = 3
a = server_config["guarded_by_setdefault"]
b = server_config["guarded_by_assignment"]
c = server_config["guarded_by_membership"]
d = server_config["brand_new_unguarded_key"]
if True:
    e = server_config["unguarded_inside_a_module_level_if"]
f = server_config.get("safe_because_get", 0)
g = server_config["guarded_too_late"]
server_config["guarded_too_late"] = 4

def later():
    h = server_config["read_inside_a_function_is_not_import_time"]
'''


def test_the_scan_machinery_works_on_a_synthetic_module():
    """Prove the detector detects, without waiting for upstream to regress.

    Every branch of the guard logic is exercised here: the three guard forms,
    ordering (a guard below the read does not count), ``.get()`` immunity,
    module-level ``if`` bodies, and function bodies.
    """
    reads, guards = scan_ast(SYNTHETIC)
    assert set(guards) == {
        "guarded_by_setdefault",
        "guarded_by_assignment",
        "guarded_by_membership",
        "guarded_too_late",
    }
    assert "read_inside_a_function_is_not_import_time" not in reads
    assert "safe_because_get" not in reads

    unguarded = {
        key: min(lines)
        for key, lines in reads.items()
        if not any(guard <= min(lines) for guard in guards.get(key, ()))
    }
    assert set(unguarded) == {
        "brand_new_unguarded_key",
        "unguarded_inside_a_module_level_if",
        "guarded_too_late",
    }
    # The column-0 text scan is blind to the nested one; that is exactly the
    # divergence test_text_and_ast_scans_agree exists to keep an eye on.
    text_reads, _ = scan_text(SYNTHETIC)
    assert "unguarded_inside_a_module_level_if" not in text_reads
    assert "brand_new_unguarded_key" in text_reads


def test_wgp_still_replaces_the_default_config_wholesale():
    """The premise of this whole file, asserted rather than assumed.

    If upstream ever merges the parsed file into its defaults instead of
    replacing them, the KeyError class of bug disappears and this suite can be
    relaxed. Until then it must stay.
    """
    source = _wgp_source()
    assert re.search(r"^\s*server_config\s*=\s*json\.loads\(text\)", source, re.M), (
        "wgp.py no longer does `server_config = json.loads(text)`. Re-read "
        "wgp.py:2620-2631 and re-derive the guarantees documented in "
        "runpod_worker/config.py before touching this test."
    )


# --------------------------------------------------------------------------
# The other half: our config actually supplies those keys
# --------------------------------------------------------------------------

@pytest.fixture()
def config_dir(tmp_path, monkeypatch):
    """Point ensure_wgp_config() at a throwaway directory.

    ``C.CONFIG_DIR`` is resolved at import time, so setting the env var here
    would be ignored and the real /opt/wangp/config would be written.
    """
    target = tmp_path / "config"
    monkeypatch.setattr(C, "CONFIG_DIR", target)
    monkeypatch.delenv("WANGP_ATTENTION", raising=False)
    monkeypatch.delenv("WANGP_PROFILE", raising=False)
    return target


def test_template_renders_and_carries_every_required_key():
    rendered = C.render_wgp_config(("--attention", "sdpa", "--profile", "4"))
    for key in C.REQUIRED_WGP_KEYS:
        assert key in rendered, f"{C.TEMPLATE_PATH} has no {key!r}: {FIX_INSTRUCTIONS}"
    # __-prefixed template metadata must never reach wgp.
    assert not [key for key in rendered if key.startswith("__")]
    # It must survive a JSON round-trip: wgp.py:2623 json.loads()es this file.
    assert json.loads(json.dumps(rendered)) == rendered


def test_ensure_wgp_config_writes_a_usable_file(config_dir):
    path = C.ensure_wgp_config(("--attention", "sdpa", "--profile", "4", "--verbose", "1"))
    # shared/api.py:1071-1072 raises ValueError on any other name.
    assert path.name == "wgp_config.json"
    written = json.loads(path.read_text(encoding="utf-8"))
    for key in C.REQUIRED_WGP_KEYS:
        assert key in written
    assert written["attention_mode"] == "sdpa"
    assert written["preload_model_policy"] == [], "a worker must never load weights at import"
    assert os.path.isabs(written["loras_root"]), (
        "get_lora_dir() returns a RELATIVE path (wgp.py:2498-2499), so a relative "
        "loras_root never finds volume-staged LoRAs"
    )
    assert all(os.path.isabs(item) or item == "." for item in written["checkpoints_paths"])
    # Idempotent: a second call must not rewrite a different file.
    again = C.ensure_wgp_config(("--attention", "sdpa", "--profile", "4", "--verbose", "1"))
    assert json.loads(again.read_text(encoding="utf-8")) == written


def test_ensure_wgp_config_repairs_a_stale_file(config_dir):
    """A config written by an older image, missing the key that kills boot."""
    config_dir.mkdir(parents=True, exist_ok=True)
    stale = config_dir / "wgp_config.json"
    stale.write_text(json.dumps({"video_container": "mkv", "some_migrated_key": 7}), "utf-8")

    path = C.ensure_wgp_config(("--attention", "sdpa"))
    written = json.loads(path.read_text(encoding="utf-8"))
    for key in C.REQUIRED_WGP_KEYS:
        assert key in written
    # wgp's own migrations write into this file; they must survive our next boot.
    assert written["some_migrated_key"] == 7
    # ...but keys the worker owns are re-asserted, not inherited.
    assert written["video_container"] == "mp4"


def test_ensure_wgp_config_survives_a_corrupt_file(config_dir):
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "wgp_config.json").write_text("{not json at all", encoding="utf-8")
    written = json.loads(
        C.ensure_wgp_config(("--attention", "sdpa")).read_text(encoding="utf-8")
    )
    assert written["attention_mode"] == "sdpa"


def test_attention_mode_whitelist_matches_wgp():
    """``config.ATTENTION_MODES`` must equal the literal wgp.py:3303 enforces.

    An unlisted value raises ``Exception(f"Unknown attention mode ...")`` during
    import (wgp.py:3306), i.e. another way to build an image that cannot boot.
    """
    source = _wgp_source()
    match = re.search(r"if args\.attention in (\[[^\]]*\])", source)
    assert match, "wgp.py no longer validates args.attention against a list literal"
    assert sorted(ast.literal_eval(match.group(1))) == sorted(C.ATTENTION_MODES)


def test_unknown_attention_mode_is_rejected_before_the_image_is_built(monkeypatch):
    monkeypatch.setenv("WANGP_ATTENTION", "sage3000")
    with pytest.raises(RuntimeError) as excinfo:
        C.attention_mode(())
    assert "sage3000" in str(excinfo.value)


def test_cli_attention_wins_over_the_env_value(monkeypatch):
    """wgp.py:3302-3305 lets --attention overwrite server_config; agree with it."""
    monkeypatch.setenv("WANGP_ATTENTION", "sdpa")
    assert C.attention_mode(("--attention", "sage2")) == "sage2"
    assert C.attention_mode(("--attention=sage2",)) == "sage2"
    monkeypatch.delenv("WANGP_ATTENTION", raising=False)
    assert C.attention_mode(()) == "sdpa"


def test_config_path_name_requirement_is_still_in_shared_api():
    """We hard-code the filename; assert the constraint that forces it still exists."""
    if not API_PY.is_file():  # pragma: no cover - only in a partial checkout
        pytest.skip(f"{API_PY} not found")
    source = API_PY.read_text(encoding="utf-8", errors="replace")
    assert "config_path must point to a file named 'wgp_config.json'" in source


def test_config_module_is_cpu_only():
    """No torch / wgp / gradio may be dragged in by importing the config module."""
    import sys

    for module in ("torch", "wgp", "gradio", "numpy"):
        assert module not in sys.modules, (
            f"{module} was imported by the runpod_worker test session; the CPU tier "
            f"must stay importable on a plain runner"
        )


# --------------------------------------------------------------------------
# constraints.txt vs requirements.txt
#
# constraints.txt mirrors a dozen pins from the repo's requirements.txt BY HAND
# so the `runpod` install layer cannot re-resolve the WanGP stack underneath
# itself. Nothing kept the two files agreeing: a bump in requirements.txt left a
# stale pin here, which either fails `pip check` an hour into the image build or
# -- worse -- pins the OLD version and nothing notices until `import wgp`.
# --------------------------------------------------------------------------

CONSTRAINTS = Path(__file__).resolve().parents[1] / "constraints.txt"
REQUIREMENTS = REPO_ROOT / "requirements.txt"

#: Pins that deliberately do NOT come from requirements.txt. The torch triple is
#: installed from download.pytorch.org in an earlier Docker layer, so its
#: authority is the Dockerfile, not requirements.txt.
_NOT_FROM_REQUIREMENTS = {"torch", "torchvision", "torchaudio"}


def _pins(text: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;]+)$", line)
        if match:
            pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


def test_constraints_agree_with_requirements():
    constraints = _pins(CONSTRAINTS.read_text(encoding="utf-8"))
    requirements = _pins(REQUIREMENTS.read_text(encoding="utf-8"))
    assert constraints, "constraints.txt parsed to nothing"

    mismatched = {
        name: (version, requirements[name])
        for name, version in constraints.items()
        if name not in _NOT_FROM_REQUIREMENTS
        and name in requirements
        and requirements[name] != version
    }
    assert not mismatched, (
        "runpod_worker/constraints.txt disagrees with requirements.txt "
        f"(name: constraints, requirements): {mismatched}"
    )

    unknown = sorted(
        name for name in constraints
        if name not in _NOT_FROM_REQUIREMENTS and name not in requirements
    )
    assert not unknown, (
        f"constraints.txt pins {unknown}, which requirements.txt does not pin at "
        "all; either the package was dropped upstream or the name drifted"
    )


def test_constraints_still_pin_the_torch_triple():
    """The triple must stay pinned somewhere, or the runpod layer can move it."""
    constraints = _pins(CONSTRAINTS.read_text(encoding="utf-8"))
    for name in sorted(_NOT_FROM_REQUIREMENTS):
        assert constraints.get(name), f"constraints.txt no longer pins {name}"


def test_runtime_stage_does_not_pre_install_torch():
    """requirements.txt owns the torch version -- pre-installing it is wasted work.

    Measured on a real RunPod pod (2026-08-19): `pip install -r requirements.txt`
    resolves torch to 2.13.0 on CUDA 13 wheels and REPLACES anything installed
    before it. The old `torch==2.10.0+cu128` line downloaded ~2.5 GB per build
    and was discarded seconds later, while leaving the false impression that the
    image ran a cu128 build.
    """
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(
        encoding="utf-8"
    )
    # Strip comments -- the reasoning above is allowed to mention the old pin.
    code = "\n".join(
        line for line in dockerfile.splitlines() if not line.lstrip().startswith("#")
    )
    runtime = code.split("requirements.txt")[0]
    # The only torch install permitted before requirements.txt is the WITH_SAGE
    # builder-stage one, which is guarded by an `if`.
    for m in re.finditer(r"pip install[^\n]*\btorch==", runtime):
        context = runtime[max(0, m.start() - 200):m.start()]
        assert "WITH_SAGE" in context, (
            "the runtime stage pre-installs torch; requirements.txt will "
            "overwrite it (verified 2026-08-19) -- delete the line"
        )


def test_sage_builder_torch_matches_constraints():
    """A SageAttention wheel is compiled against one torch ABI.

    If the builder stage compiles against a different torch than the runtime
    ends up with, the wheel installs perfectly and dies at the first kernel
    launch. Keep the builder pin equal to the constrained version.
    """
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(
        encoding="utf-8"
    )
    constraints = _pins(CONSTRAINTS.read_text(encoding="utf-8"))
    want = constraints["torch"]
    builder = [
        line for line in dockerfile.splitlines()
        if "pip install" in line and "torch==" in line and not line.lstrip().startswith("#")
    ]
    assert builder, "no torch install found in the Dockerfile at all"
    for line in builder:
        got = re.search(r"torch==([0-9][^\s;+]*)", line).group(1)
        assert got == want, (
            f"Dockerfile compiles SageAttention against torch {got} but "
            f"constraints.txt pins torch {want}; the wheel would ABI-mismatch "
            "at the first kernel launch"
        )


# ---------------------------------------------------------------------------
# hf_transfer reconciliation (regression: real RunPod pod, 2026-08-19)
# ---------------------------------------------------------------------------

def test_hf_transfer_disabled_when_package_missing(monkeypatch):
    """HF_HUB_ENABLE_HF_TRANSFER=1 without the package must be forced to "0".

    huggingface_hub raises on EVERY download in that state rather than falling
    back, which cost a partial 28 GB warm before it was caught.
    """
    import builtins
    from runpod_worker import config

    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    real_import = builtins.__import__

    def no_hf_transfer(name, *a, **kw):
        if name == "hf_transfer":
            raise ImportError("simulated: not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_hf_transfer)
    assert config.ensure_hf_transfer_sane() == "disabled"
    assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "0"


def test_hf_transfer_left_alone_when_unset(monkeypatch):
    from runpod_worker import config

    monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising=False)
    assert config.ensure_hf_transfer_sane() == "off"
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ


def test_hf_transfer_kept_when_importable(monkeypatch):
    """If the package IS importable the fast path must stay enabled."""
    import sys, types
    from runpod_worker import config

    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    monkeypatch.setitem(sys.modules, "hf_transfer", types.ModuleType("hf_transfer"))
    assert config.ensure_hf_transfer_sane() == "fast"
    assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "1"
