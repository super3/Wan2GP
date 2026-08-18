"""CPU-only tests for ``runpod_worker.schema``.

No torch, no wgp import, no CUDA, no weights, no network. WanGP's source is read
as *text* (``ast.literal_eval`` on the literals we mirror) so an upstream bump
breaks CI instead of production.

    pytest runpod_worker/tests/test_schema.py -v

The four ``minimax_h3`` variants are exercised with the worked examples from
``docs/RUNPOD_SERVERLESS.md`` ("Worked examples — verified field names, one per
model_type"), trimmed to the fields that carry validation meaning.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

from runpod_worker import config as C
from runpod_worker import schema as S
from runpod_worker.errors import WorkerError

# --------------------------------------------------------------------------
# Source locations (text-scanned, never imported)
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
WGP_PY = REPO_ROOT / "wgp.py"
HANDLER_PY = REPO_ROOT / "models" / "minimax_h3" / "minimax_h3_handler.py"
SETTINGS_JSON = REPO_ROOT / "models" / "_settings.json"
PROFILE_NAME = "Turbo Lightx2v FL2V 4 Steps v1.0 768p"

FL2VA = "minimax_h3_fl2va"
FL2VA_PRUNED = "minimax_h3_fl2va_pruned"
REF2VA = "minimax_h3_ref2va"
REF2VA_PRUNED = "minimax_h3_ref2va_pruned"
ALL_TYPES = (FL2VA, FL2VA_PRUNED, REF2VA, REF2VA_PRUNED)

DEMO_PROMPT = (
    "integrated_multimodal_description: [Shot 1] A five-second cinematic single take.\n"
    "overall_soundscape: Rain on a dome, a low electrical hum.\n"
    "non_diegetic_music: One quiet bowed-glass chord."
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

_ENV_KNOBS = (
    "WANGP_MODEL_TYPE",
    "WANGP_MODEL_CONFIG",
    "WANGP_MAX_FRAMES",
    "WANGP_MAX_STEPS",
    "WANGP_MAX_BUDGET_S",
    "WANGP_DEFAULT_BUDGET_S",
    "WANGP_ALLOWED_LORAS",
    "ALLOW_MODEL_SWITCH",
    "ALLOW_URL_INPUTS",
)


@pytest.fixture()
def env(monkeypatch):
    """A deterministic environment: no endpoint-specific overrides."""
    for name in _ENV_KNOBS:
        monkeypatch.delenv(name, raising=False)
    # load_profile_fragment() falls back to the checkout this package lives in,
    # but an inherited WANGP_ROOT would point at an image path that is not here.
    monkeypatch.setenv("WANGP_ROOT", str(REPO_ROOT))
    return monkeypatch


@pytest.fixture()
def cfg(env):
    return C.WorkerConfig()


def default_settings(model_type: str) -> dict:
    """What ``session.get_default_settings(model_type)`` returns for MiniMax H3.

    Mirrors ``wgp.py:3155-3168`` (settings_version / prompt / resolution /
    flow_shift) plus ``MinimaxH3Handler.update_default_settings``
    (``minimax_h3_handler.py:513-534``). ~18 keys — deliberately NOT the settings
    universe, which is why ``schema.PRIMARY_SETTINGS`` exists.

    The prompt comes from ``defaults/<model_type>.json`` so this drifts with the
    repo rather than freezing a copy.
    """
    prompt = DEMO_PROMPT
    model_json = REPO_ROOT / "defaults" / f"{model_type}.json"
    if model_json.is_file():
        prompt = json.loads(_read(model_json)).get("prompt") or DEMO_PROMPT
    settings = {
        "settings_version": 2.68,
        "prompt": prompt,
        "resolution": "832x480",
        "video_length": 124,
        "sliding_window_size": 362,
        "sliding_window_overlap": 18,
        "num_inference_steps": 20,
        "guidance_scale": 1.0,
        "flow_shift": 12.0,
        "sample_solver": "euler",
        "skip_steps_start_step_perc": 25,
        "skip_steps_multiplier": 0.08,
        "denoising_strength": 1.0,
        "audio_prompt_type": "",
        "video_prompt_type": "",
        "image_mode": 0,
    }
    if model_type in S.REF2VA_TYPES:
        settings.update({"image_refs_relative_size": 100, "remove_background_images_ref": 0})
    return settings


def parse(payload: dict, model_type: str = FL2VA_PRUNED, *, cfg=None, **kwargs):
    """``schema.parse`` with the endpoint pinned to ``model_type``."""
    return S.parse(
        payload,
        model_type=model_type,
        allowed_settings=default_settings(model_type),
        cfg=cfg if cfg is not None else C.WorkerConfig(),
        **kwargs,
    )


def raises(code: str, payload: dict, model_type: str = FL2VA_PRUNED, **kwargs) -> WorkerError:
    with pytest.raises(WorkerError) as excinfo:
        parse(payload, model_type, **kwargs)
    assert excinfo.value.code == code, (
        f"expected {code}, got {excinfo.value.code}: {excinfo.value.message}"
    )
    return excinfo.value


def b64_stub() -> dict:
    """A media spec that is structurally valid; schema never decodes it."""
    return {"b64": "aVZCT1J3MEtHZ28="}


# ==========================================================================
# Source drift: the literals we mirror must still be the literals upstream has
# ==========================================================================

def test_attachment_keys_match():
    """``schema.ATTACHMENT_KEYS`` must equal ``wgp.ATTACHMENT_KEYS`` exactly.

    Parsed out of ``wgp.py:167-168`` with ``ast.literal_eval`` — no import, no
    torch. An upstream addition that slipped through would mean a media slot the
    worker never validates, never materializes and never cleans up, which is how
    a caller-supplied local path reaches ``_absolutize_setting_path``
    (``shared/api.py:1028-1043``).
    """
    if not WGP_PY.is_file():  # pragma: no cover - partial checkout
        pytest.skip(f"{WGP_PY} not found")
    match = re.search(r"^ATTACHMENT_KEYS\s*=\s*(\[[^\]]*\])", _read(WGP_PY), re.M | re.S)
    assert match, "wgp.py no longer defines ATTACHMENT_KEYS as a list literal"
    upstream = ast.literal_eval(match.group(1))
    assert tuple(upstream) == tuple(S.ATTACHMENT_KEYS), (
        "wgp.ATTACHMENT_KEYS drifted.\n"
        f"  wgp.py:    {list(upstream)}\n"
        f"  schema.py: {list(S.ATTACHMENT_KEYS)}\n"
        "FIX: update ATTACHMENT_KEYS in runpod_worker/schema.py, give every new key "
        "an entry in MEDIA_KIND (image/video/audio), and confirm media_in.py can "
        "materialize it. Until then the new slot is unvalidated passthrough."
    )


def test_every_attachment_key_has_a_media_kind_and_is_forbidden_in_settings():
    for key in S.ATTACHMENT_KEYS:
        assert key in S.MEDIA_KIND, f"{key} has no media kind"
        assert S.MEDIA_KIND[key] in S.EXTS_BY_KIND
        assert key in S.FORBIDDEN_KEYS, f"{key} must not be settable through input.settings"
    assert set(S.MEDIA_KIND) == set(S.ATTACHMENT_KEYS)
    assert S.LIST_KEYS <= set(S.ATTACHMENT_KEYS)


def test_primary_settings_matches_models_settings_json():
    """The baked 112-key allow-list vs. the file it was transcribed from."""
    if not SETTINGS_JSON.is_file():  # pragma: no cover
        pytest.skip(f"{SETTINGS_JSON} not found")
    on_disk = S.read_primary_settings(REPO_ROOT)
    assert on_disk == S.PRIMARY_SETTINGS, (
        "models/_settings.json drifted from schema.PRIMARY_SETTINGS.\n"
        f"  only on disk: {sorted(on_disk - S.PRIMARY_SETTINGS)}\n"
        f"  only baked:   {sorted(S.PRIMARY_SETTINGS - on_disk)}\n"
        "FIX: update _PRIMARY_SETTINGS_KEYS in runpod_worker/schema.py. A key missing "
        "from the baked list is rejected as unknown_setting even though WanGP accepts it."
    )


def test_frame_lattice_matches_the_handler_source():
    """107 / 17 / 5 come from ``minimax_h3_handler.py:185-187``, not from us."""
    if not HANDLER_PY.is_file():  # pragma: no cover
        pytest.skip(f"{HANDLER_PY} not found")
    source = _read(HANDLER_PY)
    found = {
        name: int(re.search(rf'"{name}"\s*:\s*(\d+)', source).group(1))
        for name in ("frames_minimum", "frames_steps", "frames_offset", "block_size")
    }
    model_def = S.fallback_model_def(FL2VA_PRUNED)
    assert (found["frames_minimum"], found["frames_steps"], found["frames_offset"]) == (107, 17, 5)
    assert S.frame_lattice(model_def) == (107, 17, 5)
    assert model_def["block_size"] == found["block_size"] == 32
    # frames_maximum is Ref2VA-only (minimax_h3_handler.py:251).
    assert "frames_maximum" not in S.fallback_model_def(FL2VA)
    assert S.fallback_model_def(REF2VA)["frames_maximum"] == int(
        re.search(r'"frames_maximum"\s*:\s*(\d+)', source).group(1)
    )


def test_first_block_cache_thresholds_match_the_handler_source():
    if not HANDLER_PY.is_file():  # pragma: no cover
        pytest.skip(f"{HANDLER_PY} not found")
    match = re.search(r"^FIRST_BLOCK_CACHE_THRESHOLDS\s*=\s*(\([^)]*\))", _read(HANDLER_PY), re.M)
    assert match, "minimax_h3_handler.py no longer defines FIRST_BLOCK_CACHE_THRESHOLDS"
    assert tuple(ast.literal_eval(match.group(1))) == tuple(S.FIRST_BLOCK_CACHE_THRESHOLDS)


def test_letter_whitelists_match_the_handler_source():
    """Every ``letters_filter`` in the handler must be accounted for."""
    if not HANDLER_PY.is_file():  # pragma: no cover
        pytest.skip(f"{HANDLER_PY} not found")
    source = _read(HANDLER_PY)
    upstream = set(re.findall(r'"letters_filter"\s*:\s*"([^"]*)"', source))
    assert upstream == {"KI", "PDEV+-", "ABK", "GVKFI", "AK2"}, (
        f"minimax_h3_handler.py letters_filter set changed to {sorted(upstream)}; "
        "update _FALLBACK_LETTERS / _FL2VA_MODEL_DEF / _REF2VA_MODEL_DEF in schema.py"
    )
    fl2va = S.letters_allowed(FL2VA_PRUNED, S.fallback_model_def(FL2VA_PRUNED))
    ref2va = S.letters_allowed(REF2VA, S.fallback_model_def(REF2VA))
    assert set(fl2va["video_prompt_type"]) == set("GVKFI")
    assert set(fl2va["audio_prompt_type"]) == set("AK2")
    assert set(ref2va["video_prompt_type"]) == set("PDEV+-") | set("KI")
    assert set(ref2va["audio_prompt_type"]) == set("ABK")
    for letters in (fl2va, ref2va):
        assert set(letters["image_prompt_type"]) == set("TSEVL")


def test_schema_stays_cpu_only():
    for module in ("torch", "wgp", "gradio", "numpy", "runpod"):
        assert module not in sys.modules, (
            f"importing runpod_worker.schema pulled in {module}; the CPU tier must run "
            f"on a plain runner with only pytest installed"
        )


# ==========================================================================
# Valid requests — one per model_type (spec "Worked examples")
# ==========================================================================

def example_a() -> dict:
    """(a) FL2VA pruned, text-only, 4-step turbo."""
    return {
        "model_type": FL2VA_PRUNED,
        "profile": PROFILE_NAME,
        "settings": {
            "prompt": DEMO_PROMPT,
            "resolution": "832x480",
            "video_length": 124,
            "sample_solver": "euler",
            "image_prompt_type": "",
            "video_prompt_type": "",
            "audio_prompt_type": "",
            "sliding_window_size": 362,
            "sliding_window_overlap": 18,
            "seed": 918273645,
        },
        "runtime": {"timeout_s": 900},
    }


def example_b() -> dict:
    """(b) FL2VA full, first+last frame, 20 steps, First Block Cache."""
    return {
        "model_type": FL2VA,
        "settings": {
            "prompt": DEMO_PROMPT,
            "resolution": "832x480",
            "video_length": 209,
            "num_inference_steps": 20,
            "flow_shift": 12.0,
            "sample_solver": "euler",
            "skip_steps_cache_type": "first_block",
            "skip_steps_multiplier": 0.08,
            "skip_steps_start_step_perc": 25,
            "image_prompt_type": "SE",
            "seed": 4242,
        },
        "media": {"image_start": b64_stub(), "image_end": b64_stub()},
        "runtime": {"timeout_s": 2400},
    }


def example_c() -> dict:
    """(c) FL2VA pruned, control video, generate a new soundtrack only."""
    return {
        "model_type": FL2VA_PRUNED,
        "profile": PROFILE_NAME,
        "settings": {
            "prompt": DEMO_PROMPT,
            "video_prompt_type": "GV",
            "audio_prompt_type": "2",
            "denoising_strength": 1.0,
            "video_length": 124,
            "resolution": "832x480",
            "seed": 77,
        },
        "media": {
            "video_guide": {
                "volume": "clips/plate.mp4",
                "range": {"start_frame": 0, "end_frame": 240},
            }
        },
    }


def example_d() -> dict:
    """(d) Ref2VA, two reference images + one audio reference."""
    return {
        "model_type": REF2VA,
        "settings": {
            "prompt": DEMO_PROMPT,
            "video_prompt_type": "KI",
            "audio_prompt_type": "A",
            "image_refs_relative_size": 100,
            "remove_background_images_ref": 0,
            "resolution": "832x480",
            "video_length": 226,
            "num_inference_steps": 20,
            "flow_shift": 12.0,
            "sliding_window_size": 362,
            "sliding_window_overlap": 18,
            "seed": 1234,
        },
        "media": {
            "image_refs": [b64_stub(), b64_stub()],
            "audio_guide": {"volume": "refs/voice.wav"},
        },
        "runtime": {"timeout_s": 2600},
    }


def example_e() -> dict:
    """(e) Ref2VA pruned, two reference videos + two audio references."""
    return {
        "model_type": REF2VA_PRUNED,
        "settings": {
            "prompt": DEMO_PROMPT,
            "video_prompt_type": "IV+-",
            "audio_prompt_type": "AB",
            "image_refs_relative_size": 120,
            "resolution": "832x480",
            "video_length": 124,
            "num_inference_steps": 20,
            "flow_shift": 12.0,
            "seed": 99,
        },
        "media": {
            "image_refs": [b64_stub()],
            "video_guide": {"volume": "refs/motion_a.mp4"},
            "video_guide2": {"volume": "refs/motion_b.mp4"},
            "audio_guide": {"volume": "refs/voice_a.wav"},
            "audio_guide2": {"volume": "refs/voice_b.wav"},
        },
    }


EXAMPLES = {
    FL2VA_PRUNED: example_a,
    FL2VA: example_b,
    REF2VA: example_d,
    REF2VA_PRUNED: example_e,
}


@pytest.mark.parametrize("model_type", ALL_TYPES)
def test_valid_request_for_each_model_type(cfg, model_type):
    payload = EXAMPLES[model_type]()
    req = parse(payload, model_type, cfg=cfg)

    assert req.model_type == model_type
    assert req.settings["model_type"] == model_type
    # Worker pins: exactly one video per job, never an image.
    assert req.settings["batch_size"] == 1
    assert req.settings["repeat_generation"] == 1
    assert req.settings["image_mode"] == 0
    # The seed is resolved before the GPU is touched, so the response can echo it
    # and a retry lands on the same object key.
    assert req.settings["seed"] == payload["settings"]["seed"]
    assert req.resolved["seed"] == req.settings["seed"]
    assert req.resolved["model_type"] == model_type
    assert S.is_legal_frame_count(req.settings["video_length"], S.fallback_model_def(model_type))
    # Media is validated, not materialized: the specs are still specs.
    assert set(req.media) == set(payload.get("media", {}))
    assert all(key not in req.settings for key in S.ATTACHMENT_KEYS)
    assert req.output["mode"] == "auto"
    assert S.MIN_BUDGET_S <= req.budget_s <= cfg.max_budget_s


def test_image_refs_is_always_a_list(cfg):
    payload = example_d()
    payload["media"]["image_refs"] = b64_stub()  # a single object, not a list
    req = parse(payload, REF2VA, cfg=cfg)
    assert isinstance(req.media["image_refs"], list)
    assert len(req.media["image_refs"]) == 1
    assert isinstance(req.media["audio_guide"], dict)


def test_media_range_is_preserved_for_media_in(cfg):
    """The virtual-media suffix is media_in's job; schema must not eat it."""
    req = parse(example_c(), FL2VA_PRUNED, cfg=cfg)
    assert req.media["video_guide"]["range"] == {"start_frame": 0, "end_frame": 240}


def test_accelerator_profile_is_applied_before_user_settings(cfg):
    """``input.profile`` is opt-in and loses to explicit ``input.settings``."""
    req = parse(example_a(), FL2VA_PRUNED, cfg=cfg)
    assert req.profile == PROFILE_NAME
    assert req.settings["num_inference_steps"] == 4, "the profile must beat the 20-step default"
    assert req.settings["flow_shift"] == 6
    loras = req.settings["activated_loras"]
    assert loras and "lightx2v_fl2v_turbo_4step" in loras[0]
    assert any("profile" in item for item in req.warnings)

    payload = example_a()
    payload["settings"]["num_inference_steps"] = 8
    assert parse(payload, FL2VA_PRUNED, cfg=cfg).settings["num_inference_steps"] == 8


def test_profile_is_not_forced_on(cfg):
    """No ``profile`` key -> the model's own 20-step default, no LoRA."""
    payload = example_a()
    payload.pop("profile")
    req = parse(payload, FL2VA_PRUNED, cfg=cfg)
    assert req.profile is None
    assert req.settings["num_inference_steps"] == 20
    assert not req.settings.get("activated_loras")


def test_unknown_profile_and_traversal_are_rejected(cfg):
    payload = example_a()
    payload["profile"] = "no such profile"
    error = raises("bad_request", payload, FL2VA_PRUNED, cfg=cfg)
    assert "unknown accelerator profile" in error.message

    payload["profile"] = "../../defaults/minimax_h3_fl2va"
    error = raises("bad_request", payload, FL2VA_PRUNED, cfg=cfg)
    assert "may only contain" in error.message, "a profile name must never reach the filesystem"


def test_prompt_alias_and_precedence(cfg):
    req = parse({"prompt": "a top level prompt"}, FL2VA_PRUNED, cfg=cfg)
    assert req.settings["prompt"] == "a top level prompt"

    req = parse(
        {"prompt": "ignored", "settings": {"prompt": "settings wins"}},
        FL2VA_PRUNED,
        cfg=cfg,
    )
    assert req.settings["prompt"] == "settings wins"

    # No prompt anywhere -> the model's demo prompt runs. That is a real, billed
    # generation of something the caller never asked for, so it must be flagged.
    req = parse({}, FL2VA_PRUNED, cfg=cfg)
    assert req.settings["prompt"]
    assert any("demo prompt" in item for item in req.warnings)


def test_prompt_is_required_when_there_are_no_defaults(cfg):
    with pytest.raises(WorkerError) as excinfo:
        S.parse({"settings": {}}, model_type=FL2VA_PRUNED, cfg=cfg)
    assert excinfo.value.code == "bad_request"
    assert "prompt" in excinfo.value.message


# ==========================================================================
# Forbidden keys and unknown keys
# ==========================================================================

def test_forbidden_keys(cfg):
    """The spec's two named cases: an attachment path and ``mode``."""
    error = raises("bad_request", {"settings": {"image_start": "/etc/hostname"}}, cfg=cfg)
    assert "image_start" in error.message and "input.media" in error.message

    error = raises("bad_request", {"settings": {"mode": "edit_postprocessing"}}, cfg=cfg)
    assert "mode" in error.message


@pytest.mark.parametrize("key", sorted(S.FORBIDDEN_KEYS))
def test_every_forbidden_key_is_rejected(cfg, key):
    """`mode` flips validate_task into the edit branch (wgp.py:1871-1872, 8567),
    which reads ``video_source`` straight off disk; the attachment keys let a
    caller name any local file; the rest steer WanGP off the generation path."""
    raises("bad_request", {"settings": {key: "x"}}, cfg=cfg)


def test_unknown_setting_is_its_own_code(cfg):
    error = raises("unknown_setting", {"settings": {"not_a_wangp_setting": 1}}, cfg=cfg)
    assert "not_a_wangp_setting" in error.message
    assert error.retryable is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("seed", 5),
        ("activated_loras", []),
        ("frames_positions", ""),
        ("masking_strength", 0.5),
        ("skip_steps_cache_type", ""),
        ("negative_prompt", ""),
        ("override_attention", ""),
        ("config", ""),
    ],
)
def test_keys_absent_from_get_default_settings_are_still_accepted(cfg, key, value):
    """``get_default_settings`` is NOT the settings universe.

    None of these keys appear in it for MiniMax H3; all of them are in
    ``models/_settings.json`` and are merged in by ``clean_settings``
    (``wgp.py:1747-1760``). Validating against the defaults alone would reject
    every one of them as ``unknown_setting`` — the bug the allow-list union fixes.
    """
    assert key not in default_settings(FL2VA_PRUNED)
    assert key in S.PRIMARY_SETTINGS
    req = parse({"settings": {"prompt": DEMO_PROMPT, key: value}}, cfg=cfg)
    assert req.settings.get(key) == value


def test_settings_must_be_an_object(cfg):
    raises("bad_request", {"settings": []}, cfg=cfg)
    with pytest.raises(WorkerError) as excinfo:
        S.parse("not a dict", model_type=FL2VA_PRUNED, cfg=cfg)
    assert excinfo.value.code == "bad_request"


# ==========================================================================
# Frame arithmetic
# ==========================================================================

def test_frame_math():
    """The spec's table. ``round_overlap`` ROUNDS TO NEAREST; 27/30 -> 35 are the
    cases that catch a floor-instead-of-round implementation."""
    assert S.floor_frames(124, 107, 17, 5) == 124
    assert S.floor_frames(130, 107, 17, 5) == 124
    assert S.floor_frames(50, 107, 17, 5) == 107
    assert S.floor_frames(209, 107, 17, 5) == 209
    assert S.round_overlap(18, 17, 1) == 18
    assert S.round_overlap(20, 17, 1) == 18
    assert S.round_overlap(30, 17, 1) == 35
    assert S.round_overlap(27, 17, 1) == 35
    assert S.round_overlap(0, 17, 1) == 0


def test_frame_math_matches_the_real_frame_scheduler():
    """Differential test against ``shared/utils/frame_scheduler.py`` itself.

    Loaded by file path: importing ``shared.utils`` as a package pulls numpy in,
    and the CPU tier must not need it.
    """
    module_path = REPO_ROOT / "shared" / "utils" / "frame_scheduler.py"
    if not module_path.is_file():  # pragma: no cover
        pytest.skip(f"{module_path} not found")
    import importlib.util

    spec = importlib.util.spec_from_file_location("_frame_scheduler_under_test", module_path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)

    for value in list(range(0, 400)) + [737, 1000, 99999]:
        assert S.floor_frames(value, 107, 17, 5) == upstream.floor_frame_count(value, 107, 17, 5)
        assert S.normalize_frames(value, 107, 17, 5) == upstream.normalize_frame_count(
            value, 107, 17, 5
        )
        overlap, error = upstream.normalize_overlap(value, 17, 1)
        assert not error
        assert S.round_overlap(value, 17, 1) == overlap


def test_legal_frame_counts_are_5_mod_17_from_107():
    model_def = S.fallback_model_def(FL2VA_PRUNED)
    counts = S.legal_frame_counts(model_def, 362)
    assert counts[0] == 107 and counts[-1] == 362
    assert all((value - 5) % 17 == 0 for value in counts)
    assert counts == tuple(range(107, 363, 17))
    assert S.is_legal_frame_count(124, model_def)
    assert not S.is_legal_frame_count(125, model_def)
    assert not S.is_legal_frame_count(90, model_def)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(107, 107), (124, 124), (130, 124), (50, 107), (209, 209), (363, 362), (362, 362)],
)
def test_video_length_is_floored_onto_the_lattice(cfg, requested, expected):
    req = parse(
        {"settings": {"prompt": DEMO_PROMPT, "video_length": requested}},
        FL2VA_PRUNED,
        cfg=cfg,
    )
    assert req.settings["video_length"] == expected
    assert req.resolved["video_length"] == expected
    if requested != expected:
        assert any("video_length" in item for item in req.warnings), (
            "a silently changed frame count must be reported: the caller is billed for it"
        )


def test_video_length_cap(cfg):
    """FL2VA has no ``frames_maximum`` anywhere in the headless path, so the
    worker's own cap is the only thing between a caller and hundreds of sliding
    windows on a billed GPU."""
    error = raises(
        "invalid_setting",
        {"settings": {"prompt": DEMO_PROMPT, "video_length": 100000}},
        cfg=cfg,
    )
    assert "cap" in error.message
    assert any("WANGP_MAX_FRAMES" in item for item in error.details)
    # 363 floors to 362 and is accepted; 364 floors to 362 too.
    assert parse({"settings": {"prompt": DEMO_PROMPT, "video_length": 363}},
                 cfg=cfg).settings["video_length"] == 362


def test_video_length_cap_is_raisable_but_never_above_the_model(env):
    env.setenv("WANGP_MAX_FRAMES", "800")
    cfg = C.WorkerConfig()
    # Ref2VA declares frames_maximum 737 (minimax_h3_handler.py:251); the model
    # wins over a larger worker cap. Note 737 is NOT itself on the lattice
    # (737 - 5 = 732, and 732 % 17 == 6), so the real ceiling is 736 = 5 + 17*43 —
    # the kind of off-by-one a hand-written cap would get wrong.
    req = parse({"settings": {"prompt": DEMO_PROMPT, "video_length": 737}}, REF2VA, cfg=cfg)
    assert req.settings["video_length"] == 736
    raises("invalid_setting", {"settings": {"prompt": DEMO_PROMPT, "video_length": 760}},
           REF2VA, cfg=cfg)
    # FL2VA has no model maximum, so the worker cap is what applies.
    req = parse({"settings": {"prompt": DEMO_PROMPT, "video_length": 800}}, FL2VA, cfg=cfg)
    assert req.settings["video_length"] == 787  # 5 + 17*46, floored under 800


@pytest.mark.parametrize("value", [0, -17, "many", 12.5])
def test_video_length_must_be_a_positive_int(cfg, value):
    raises("invalid_setting", {"settings": {"prompt": DEMO_PROMPT, "video_length": value}},
           cfg=cfg)


def test_sliding_window_size_and_overlap_are_normalized(cfg):
    req = parse(
        {
            "settings": {
                "prompt": DEMO_PROMPT,
                "sliding_window_size": 500,     # above window_max 481
                "sliding_window_overlap": 30,   # rounds UP to 35
            }
        },
        cfg=cfg,
    )
    assert req.settings["sliding_window_size"] == 481
    assert req.settings["sliding_window_overlap"] == 35
    assert len([item for item in req.warnings if "sliding_window" in item]) == 2

    req = parse(
        {"settings": {"prompt": DEMO_PROMPT, "sliding_window_size": 50,
                      "sliding_window_overlap": 999}},
        cfg=cfg,
    )
    assert req.settings["sliding_window_size"] == 124   # window_min
    assert req.settings["sliding_window_overlap"] == 120  # overlap_max

    # 0 means "sliding windows off" and must survive untouched (wgp.py:6930).
    req = parse({"settings": {"prompt": DEMO_PROMPT, "sliding_window_size": 0,
                              "sliding_window_overlap": 0}}, cfg=cfg)
    assert req.settings["sliding_window_size"] == 0
    assert req.settings["sliding_window_overlap"] == 0


# ==========================================================================
# Seed resolution
# ==========================================================================

def test_seed_resolution(cfg):
    """A resolved seed is what makes a re-run reproducible and an idempotency key
    derivable before any GPU work. WanGP never reports the seed it drew."""
    req = parse({"settings": {"prompt": DEMO_PROMPT, "seed": -1}}, cfg=cfg)
    assert 1 <= req.settings["seed"] <= S.SEED_MAX
    assert any("seed" in item for item in req.warnings)
    assert req.resolved["seed"] == req.settings["seed"]

    req = parse({"settings": {"prompt": DEMO_PROMPT, "seed": 918273645}}, cfg=cfg)
    assert req.settings["seed"] == 918273645
    assert not [item for item in req.warnings if "seed" in item]

    # Absent seed: get_default_settings does not carry one, so it must still resolve.
    req = parse({"settings": {"prompt": DEMO_PROMPT}}, cfg=cfg)
    assert 1 <= req.settings["seed"] <= S.SEED_MAX

    # Any negative value is "random" to WanGP (wgp.py:6924), not just -1.
    assert 1 <= parse({"settings": {"prompt": DEMO_PROMPT, "seed": -99}},
                      cfg=cfg).settings["seed"] <= S.SEED_MAX


def test_resolve_seed_is_in_wgps_own_range():
    """wgp.py:5775 draws ``random.randint(0, 999999999)``; the UI slider is
    ``gr.Slider(-1, 999999999)`` (wgp.py:12036). 0 is excluded here only so a
    resolved seed is never falsy."""
    drawn = {S.resolve_seed(-1) for _ in range(200)}
    assert all(1 <= value <= S.SEED_MAX for value in drawn)
    assert len(drawn) > 1, "resolve_seed must not be constant"
    assert S.resolve_seed(0) == 0
    assert S.resolve_seed("42") == 42


@pytest.mark.parametrize("value", ["abc", 1.5, True, [1]])
def test_bad_seed_is_rejected(cfg, value):
    raises("invalid_setting", {"settings": {"prompt": DEMO_PROMPT, "seed": value}}, cfg=cfg)


# ==========================================================================
# image_prompt_type / video_prompt_type / audio_prompt_type combinations
# ==========================================================================

def fl2va(settings: dict, media: dict | None = None) -> dict:
    payload = {"settings": {"prompt": DEMO_PROMPT, **settings}}
    if media is not None:
        payload["media"] = media
    return payload


def test_letters_outside_the_variant_alphabet_are_rejected(cfg):
    # "P"/"D" are Ref2VA guide letters; FL2VA's filter is "GVKFI".
    error = raises("invalid_setting", fl2va({"video_prompt_type": "PD"}), cfg=cfg)
    assert "not supported by FL2VA" in error.message
    # "B" (second audio reference) is Ref2VA-only; FL2VA's filter is "AK2".
    raises("invalid_setting", fl2va({"audio_prompt_type": "AB"}), cfg=cfg)
    # "F"/"G" are FL2VA guide letters; Ref2VA's filter is "PDEV+-" + "KI".
    error = raises(
        "invalid_setting",
        {"settings": {"prompt": DEMO_PROMPT, "video_prompt_type": "GF"}},
        REF2VA,
        cfg=cfg,
    )
    assert "not supported by Ref2VA" in error.message
    # image_prompt_types_allowed is "TSEVL" for both variants.
    raises("invalid_setting", fl2va({"image_prompt_type": "X"}), cfg=cfg)


def test_fl2va_rejects_ref2va_only_attachments(cfg):
    error = raises(
        "bad_request",
        fl2va({"video_prompt_type": "GV"},
              {"video_guide": b64_stub(), "video_guide2": b64_stub()}),
        cfg=cfg,
    )
    assert "Ref2VA-only" in error.message
    raises("bad_request", fl2va({}, {"audio_guide2": b64_stub()}), cfg=cfg)


def test_audio_from_control_video_combination_rules(cfg):
    """minimax_h3_handler.py:360-374, pre-flighted before the model loads."""
    ok = parse(fl2va({"video_prompt_type": "GV", "audio_prompt_type": "2"},
                     {"video_guide": b64_stub()}), cfg=cfg)
    assert ok.settings["audio_prompt_type"] == "2"

    # "2" cannot combine with "A" or "K".
    error = raises(
        "invalid_setting",
        fl2va({"video_prompt_type": "GV", "audio_prompt_type": "2A"},
              {"video_guide": b64_stub(), "audio_guide": b64_stub()}),
        cfg=cfg,
    )
    assert "cannot combine" in error.message
    raises(
        "invalid_setting",
        fl2va({"video_prompt_type": "GV", "audio_prompt_type": "2K"},
              {"video_guide": b64_stub()}),
        cfg=cfg,
    )

    # "2" and "K" both need G + V + a real video_guide.
    for letter in ("2", "K"):
        error = raises("invalid_setting",
                       fl2va({"video_prompt_type": "GV", "audio_prompt_type": letter}), cfg=cfg)
        assert "video_guide" in error.message
        raises("invalid_setting",
               fl2va({"video_prompt_type": "G", "audio_prompt_type": letter},
                     {"video_guide": b64_stub()}), cfg=cfg)


def test_frame_injection_needs_one_position_per_reference_image(cfg):
    payload = fl2va(
        {"video_prompt_type": "KFI", "frames_positions": "1 40"},
        {"image_refs": [b64_stub(), b64_stub()]},
    )
    assert parse(payload, cfg=cfg).settings["frames_positions"] == "1 40"

    payload["settings"]["frames_positions"] = "1"
    error = raises("invalid_setting", payload, cfg=cfg)
    assert "one frames_positions entry" in error.message
    # Commas are a legal separator upstream ((x or "").replace(",", " ").split()).
    payload["settings"]["frames_positions"] = "1,40"
    assert parse(payload, cfg=cfg)


def test_one_image_ref_only_for_fl2va(cfg):
    """wgp.py:1336 — plain "I" takes a single reference image; "KFI" is the
    multi-image (frame injection) mode."""
    error = raises(
        "invalid_setting",
        fl2va({"video_prompt_type": "I"}, {"image_refs": [b64_stub(), b64_stub()]}),
        cfg=cfg,
    )
    assert "one reference image" in error.message
    assert parse(fl2va({"video_prompt_type": "I"}, {"image_refs": [b64_stub()]}), cfg=cfg)


def test_required_media_is_demanded_before_the_model_loads(cfg):
    """Every ``You must provide a ...`` in validate_settings, pre-flighted."""
    for letters, key in (
        ({"image_prompt_type": "S"}, "image_start"),
        ({"image_prompt_type": "E"}, "image_end"),
        ({"image_prompt_type": "V"}, "video_source"),
        ({"video_prompt_type": "GV"}, "video_guide"),
        ({"video_prompt_type": "I"}, "image_refs"),
    ):
        error = raises("invalid_setting", fl2va(letters), cfg=cfg)
        assert key in error.message
        assert parse(fl2va(letters, {key: b64_stub()}), cfg=cfg)


def test_media_that_would_be_silently_dropped_is_reported(cfg):
    """WanGP nulls an attachment whose letter is absent; say so rather than let a
    caller believe their reference image was used."""
    req = parse(fl2va({"video_prompt_type": ""}, {"image_refs": [b64_stub()]}), cfg=cfg)
    assert any("image_refs" in item and "ignored" in item for item in req.warnings)
    req = parse(fl2va({}, {"image_guide": b64_stub()}), cfg=cfg)
    assert any("image_guide" in item for item in req.warnings)


def test_ref2va_reference_counting(cfg):
    """minimax_h3_handler.py:376-445 — counts only; the duration rules need
    ffprobe/librosa and stay with WanGP."""
    # <= 9 reference images
    error = raises(
        "invalid_setting",
        {"settings": {"prompt": DEMO_PROMPT, "video_prompt_type": "I"},
         "media": {"image_refs": [b64_stub() for _ in range(10)]}},
        REF2VA,
        cfg=cfg,
    )
    assert "9 reference images" in error.message

    # #audio must not exceed #images + #videos
    error = raises(
        "invalid_setting",
        {"settings": {"prompt": DEMO_PROMPT, "video_prompt_type": "", "audio_prompt_type": "A"},
         "media": {"audio_guide": b64_stub()}},
        REF2VA,
        cfg=cfg,
    )
    assert "visual" in error.message

    # "K" (use the reference-video soundtracks) needs at least one video
    error = raises(
        "invalid_setting",
        {"settings": {"prompt": DEMO_PROMPT, "video_prompt_type": "I",
                      "audio_prompt_type": "K"},
         "media": {"image_refs": [b64_stub()]}},
        REF2VA,
        cfg=cfg,
    )
    assert "reference video" in error.message

    # ...and with one, K is fine and does not count toward the 12-file total.
    assert parse(
        {"settings": {"prompt": DEMO_PROMPT, "video_prompt_type": "IV",
                      "audio_prompt_type": "K"},
         "media": {"image_refs": [b64_stub()], "video_guide": b64_stub()}},
        REF2VA,
        cfg=cfg,
    )

    # <= 12 reference files in total (9 images + 2 videos + 2 audio = 13)
    error = raises(
        "invalid_setting",
        {"settings": {"prompt": DEMO_PROMPT, "video_prompt_type": "IV+",
                      "audio_prompt_type": "AB"},
         "media": {"image_refs": [b64_stub() for _ in range(9)],
                   "video_guide": b64_stub(), "video_guide2": b64_stub(),
                   "audio_guide": b64_stub(), "audio_guide2": b64_stub()}},
        REF2VA,
        cfg=cfg,
    )
    assert "12 reference files" in error.message


def test_ref2va_accepts_two_videos_and_two_audio_references(cfg):
    req = parse(example_e(), REF2VA_PRUNED, cfg=cfg)
    assert req.settings["video_prompt_type"] == "IV+-"
    assert req.settings["audio_prompt_type"] == "AB"
    assert not [item for item in req.warnings if "ignored" in item]


# ==========================================================================
# Step-skipping cache
# ==========================================================================

@pytest.mark.parametrize("value", [0.06, 0.08, 0.10, 0.12, 0.14])
def test_skip_steps_multiplier_accepts_every_declared_threshold(cfg, value):
    req = parse(
        {"settings": {"prompt": DEMO_PROMPT, "skip_steps_cache_type": "first_block",
                      "skip_steps_multiplier": value}},
        cfg=cfg,
    )
    assert req.settings["skip_steps_multiplier"] == value


@pytest.mark.parametrize("value", [0.07, 0.2, 1.5, 0])
def test_skip_steps_multiplier_threshold_membership(cfg, value):
    """wgp.py:1215 rejects anything outside first_block_cache_thresholds with
    "Unsupported First Block Cache threshold" — after the model has loaded."""
    error = raises(
        "invalid_setting",
        {"settings": {"prompt": DEMO_PROMPT, "skip_steps_cache_type": "first_block",
                      "skip_steps_multiplier": value}},
        cfg=cfg,
    )
    assert "skip_steps_multiplier" in error.message
    assert "0.06" in error.message


def test_skip_steps_cache_type_must_be_supported(cfg):
    """wgp.py:1208-1214. MiniMax H3 declares spectrum_cache + first_block_cache
    and neither tea nor mag."""
    assert parse({"settings": {"prompt": DEMO_PROMPT, "skip_steps_cache_type": "spectrum"}},
                 cfg=cfg)
    assert parse({"settings": {"prompt": DEMO_PROMPT, "skip_steps_cache_type": ""}}, cfg=cfg)
    error = raises("invalid_setting",
                   {"settings": {"prompt": DEMO_PROMPT, "skip_steps_cache_type": "tea"}},
                   cfg=cfg)
    assert "does not support" in error.message
    # The threshold rule only bites for first_block: a spectrum cache ignores it.
    assert parse({"settings": {"prompt": DEMO_PROMPT, "skip_steps_cache_type": "spectrum",
                               "skip_steps_multiplier": 1.75}}, cfg=cfg)


def test_skip_steps_start_step_perc_range(cfg):
    raises("invalid_setting",
           {"settings": {"prompt": DEMO_PROMPT, "skip_steps_start_step_perc": 250}}, cfg=cfg)


# ==========================================================================
# Solver, steps, resolution
# ==========================================================================

def test_sample_solver_must_be_declared_by_the_model(cfg):
    for solver in ("euler", "res_multistep"):
        assert parse({"settings": {"prompt": DEMO_PROMPT, "sample_solver": solver}},
                     cfg=cfg).settings["sample_solver"] == solver
    error = raises("invalid_setting",
                   {"settings": {"prompt": DEMO_PROMPT, "sample_solver": "dpm++"}}, cfg=cfg)
    assert "res_multistep" in str(error.details)


def test_num_inference_steps_is_capped(env):
    cfg = C.WorkerConfig()
    assert parse({"settings": {"prompt": DEMO_PROMPT, "num_inference_steps": 4}},
                 cfg=cfg).settings["num_inference_steps"] == 4
    raises("invalid_setting",
           {"settings": {"prompt": DEMO_PROMPT, "num_inference_steps": 5000}}, cfg=cfg)
    raises("invalid_setting",
           {"settings": {"prompt": DEMO_PROMPT, "num_inference_steps": 0}}, cfg=cfg)
    env.setenv("WANGP_MAX_STEPS", "8")
    raises("invalid_setting",
           {"settings": {"prompt": DEMO_PROMPT, "num_inference_steps": 20}},
           cfg=C.WorkerConfig())


def test_resolution_must_sit_on_the_vae_block_grid(cfg):
    """wgp.py:6760 floors both dimensions silently; rejecting is the honest
    alternative, and the message carries the nearest legal pair."""
    assert parse({"settings": {"prompt": DEMO_PROMPT, "resolution": "832x480"}},
                 cfg=cfg).settings["resolution"] == "832x480"
    error = raises("invalid_setting",
                   {"settings": {"prompt": DEMO_PROMPT, "resolution": "833x481"}}, cfg=cfg)
    assert "832x480" in str(error.details)
    raises("invalid_setting",
           {"settings": {"prompt": DEMO_PROMPT, "resolution": "not a size"}}, cfg=cfg)
    # 1280x720: 720 is not a multiple of 32 (22.5), which surprises people.
    raises("invalid_setting",
           {"settings": {"prompt": DEMO_PROMPT, "resolution": "1280x720"}}, cfg=cfg)


def test_inert_settings_are_reported_not_rejected(cfg):
    """guidance_scale, negative_prompt and keep_frames_video_guide are read and
    then ignored by this model; a silent no-op is worse than a warning."""
    req = parse({"settings": {"prompt": DEMO_PROMPT, "guidance_scale": 7.5,
                              "negative_prompt": "blurry",
                              "keep_frames_video_guide": "1:10"}}, cfg=cfg)
    joined = " ".join(req.warnings)
    assert "guidance_scale" in joined
    assert "negative_prompt" in joined
    assert "keep_frames_video_guide" in joined


# ==========================================================================
# LoRA guards
# ==========================================================================

def test_lora_guards(env):
    """get_lora_local_path (wgp.py:3670-3677) returns an absolute entry verbatim
    and maps a URL to lora_dir/basename(url); allow-list by basename instead."""
    cfg = C.WorkerConfig()
    error = raises("bad_request",
                   {"settings": {"prompt": DEMO_PROMPT,
                                 "activated_loras": ["/etc/shadow"]}}, cfg=cfg)
    assert "absolute" in error.message
    raises("bad_request", {"settings": {"prompt": DEMO_PROMPT,
                                        "activated_loras": ["../../etc/shadow"]}}, cfg=cfg)
    raises("bad_request", {"settings": {"prompt": DEMO_PROMPT,
                                        "activated_loras": ["file:///etc/shadow"]}}, cfg=cfg)
    raises("invalid_setting", {"settings": {"prompt": DEMO_PROMPT,
                                            "activated_loras": "one.safetensors"}}, cfg=cfg)

    env.setenv("WANGP_ALLOWED_LORAS", "staged.safetensors")
    cfg = C.WorkerConfig()
    error = raises("bad_request",
                   {"settings": {"prompt": DEMO_PROMPT,
                                 "activated_loras": ["not_staged.safetensors"]}}, cfg=cfg)
    assert "not staged" in error.message
    assert parse({"settings": {"prompt": DEMO_PROMPT,
                               "activated_loras": ["staged.safetensors"]}}, cfg=cfg)
    # A URL whose basename is staged resolves to the staged file with no network.
    assert parse(
        {"settings": {"prompt": DEMO_PROMPT,
                      "activated_loras": ["https://example.com/loras/staged.safetensors"]}},
        cfg=cfg,
    )


def test_a_baked_profile_lora_is_exempt_from_the_allow_list(env):
    """The profile ships inside the image; locking it out would make the turbo
    path unusable on any endpoint that sets WANGP_ALLOWED_LORAS."""
    env.setenv("WANGP_ALLOWED_LORAS", "something_else.safetensors")
    req = parse(example_a(), FL2VA_PRUNED, cfg=C.WorkerConfig())
    assert req.settings["activated_loras"]


# ==========================================================================
# media block structure
# ==========================================================================

def test_media_must_name_a_source_object(cfg):
    """A bare string would be a filesystem path on the worker."""
    error = raises("bad_request", {"settings": {"prompt": DEMO_PROMPT},
                                   "media": {"image_start": "/etc/hostname"}}, cfg=cfg)
    assert "bare string" in error.message
    raises("bad_request", {"settings": {"prompt": DEMO_PROMPT},
                           "media": {"image_start": {}}}, cfg=cfg)
    error = raises("bad_request", {"settings": {"prompt": DEMO_PROMPT},
                                   "media": {"not_a_slot": b64_stub()}}, cfg=cfg)
    assert "attachment key" in error.message


def test_media_path_traversal_is_rejected_at_the_schema_layer(cfg):
    """media_in guards this too; schema refuses it first so a hostile payload
    never reaches the filesystem code at all."""
    for spec in ({"volume": "../../etc/passwd"}, {"path": "/etc/passwd"},
                 {"path": "../secrets"}):
        raises("bad_request", {"settings": {"prompt": DEMO_PROMPT},
                               "media": {"image_start": spec}}, cfg=cfg)


def test_url_inputs_are_off_by_default(env):
    payload = {"settings": {"prompt": DEMO_PROMPT},
               "media": {"image_start": {"url": "https://example.com/a.png"}}}
    error = raises("bad_request", payload, cfg=C.WorkerConfig())
    assert "ALLOW_URL_INPUTS=1" in str(error.details)
    env.setenv("ALLOW_URL_INPUTS", "1")
    assert parse(payload, cfg=C.WorkerConfig())


def test_only_image_refs_takes_a_list(cfg):
    raises("bad_request", {"settings": {"prompt": DEMO_PROMPT},
                           "media": {"image_start": [b64_stub()]}}, cfg=cfg)


# ==========================================================================
# model_type pinning, runtime and output blocks
# ==========================================================================

def test_model_switch_is_refused_unless_explicitly_allowed(env):
    payload = {"model_type": REF2VA, "settings": {"prompt": DEMO_PROMPT}}
    error = raises("bad_request", payload, FL2VA_PRUNED, cfg=C.WorkerConfig())
    assert "pinned" in error.message
    assert "ALLOW_MODEL_SWITCH=1" in str(error.details)

    env.setenv("ALLOW_MODEL_SWITCH", "1")
    req = S.parse(payload, model_type=FL2VA_PRUNED,
                  allowed_settings=default_settings(REF2VA), cfg=C.WorkerConfig())
    assert req.model_type == REF2VA


def test_a_non_minimax_model_type_is_rejected(cfg):
    error = raises("bad_request", {"model_type": "t2v", "settings": {"prompt": DEMO_PROMPT}},
                   cfg=cfg)
    assert "minimax_h3" in error.message


def test_budget_is_clamped_into_the_endpoint_range(cfg):
    assert parse({"settings": {"prompt": DEMO_PROMPT}},
                 cfg=cfg).budget_s == cfg.default_budget_s
    req = parse({"settings": {"prompt": DEMO_PROMPT}, "runtime": {"timeout_s": 900}}, cfg=cfg)
    assert req.budget_s == 900 and req.timeout_s == 900

    req = parse({"settings": {"prompt": DEMO_PROMPT}, "runtime": {"timeout_s": 5}}, cfg=cfg)
    assert req.budget_s == S.MIN_BUDGET_S
    assert any("timeout_s" in item for item in req.warnings)

    req = parse({"settings": {"prompt": DEMO_PROMPT}, "runtime": {"timeout_s": 99999}}, cfg=cfg)
    assert req.budget_s == cfg.max_budget_s


def test_idempotency_key_charset(cfg):
    req = parse({"settings": {"prompt": DEMO_PROMPT},
                 "runtime": {"idempotency_key": "order-42:take_1"}}, cfg=cfg)
    assert req.idempotency_key == "order-42:take_1"
    # It becomes part of an S3 object key, so slashes and spaces are out.
    for bad in ("../../evil", "a/b", "with space", "", 7):
        payload = {"settings": {"prompt": DEMO_PROMPT}, "runtime": {"idempotency_key": bad}}
        if bad == "":
            assert parse(payload, cfg=cfg).idempotency_key is None
        else:
            raises("bad_request", payload, cfg=cfg)


def test_output_modes_are_normalized_for_media_out(cfg):
    """media_out branches on the canonical names only, so aliases are rewritten
    here rather than duplicated there."""
    assert parse({"settings": {"prompt": DEMO_PROMPT}}, cfg=cfg).output == {"mode": "auto"}
    assert parse({"settings": {"prompt": DEMO_PROMPT}, "output": {}},
                 cfg=cfg).output == {"mode": "auto", "content_type": "video/mp4"}
    for spelling, canonical in (("s3", "rp_bucket"), ("bucket", "rp_bucket"),
                                ("b64", "base64"), ("inline", "base64")):
        req = parse({"settings": {"prompt": DEMO_PROMPT}, "output": {"mode": spelling}}, cfg=cfg)
        assert req.output["mode"] == canonical
    raises("bad_request", {"settings": {"prompt": DEMO_PROMPT},
                           "output": {"mode": "carrier pigeon"}}, cfg=cfg)


def test_presigned_mode_requires_a_url(cfg):
    error = raises("bad_request", {"settings": {"prompt": DEMO_PROMPT},
                                   "output": {"mode": "presigned"}}, cfg=cfg)
    assert "presigned_url" in error.message
    raises("bad_request", {"settings": {"prompt": DEMO_PROMPT},
                           "output": {"mode": "presigned",
                                      "presigned_url": "file:///tmp/x"}}, cfg=cfg)
    assert parse({"settings": {"prompt": DEMO_PROMPT},
                  "output": {"mode": "presigned",
                             "presigned_url": "https://s3.example.com/x?sig=1"}}, cfg=cfg)


# ==========================================================================
# Nothing escapes as an untyped exception
# ==========================================================================

HOSTILE = [
    None,
    [],
    "",
    {"settings": {"prompt": DEMO_PROMPT}, "media": []},
    {"settings": {"prompt": DEMO_PROMPT}, "media": {"image_refs": "x"}},
    {"settings": {"prompt": DEMO_PROMPT}, "runtime": "soon"},
    {"settings": {"prompt": DEMO_PROMPT}, "output": 3},
    {"settings": {"prompt": 42}},
    {"settings": {"prompt": DEMO_PROMPT, "config": ["int8"]}},
    {"settings": {"prompt": DEMO_PROMPT, "config": "not_a_config_id"}},
    {"settings": {"prompt": DEMO_PROMPT, "flow_shift": "fast"}},
    {"settings": {"prompt": DEMO_PROMPT, "denoising_strength": 4}},
    {"settings": {"prompt": DEMO_PROMPT, "activated_loras": ["a.safetensors"],
                  "loras_multipliers": {"a": 1}}},
    {"model_type": 5, "settings": {"prompt": DEMO_PROMPT}},
    {"profile": 12, "settings": {"prompt": DEMO_PROMPT}},
    {"settings": {"prompt": DEMO_PROMPT}, "runtime": {"priority": 99}},
    {"settings": {"prompt": DEMO_PROMPT}, "runtime": {"timeout_s": "soon"}},
    {"settings": {"prompt": DEMO_PROMPT, "image_mode": 1}},
]


@pytest.mark.parametrize("payload", HOSTILE, ids=range(len(HOSTILE)))
def test_hostile_payloads_raise_typed_errors_only(cfg, payload):
    """``parse`` promises bad_request / unknown_setting / invalid_setting and
    nothing else — the handler turns anything else into ``internal_error``."""
    with pytest.raises(WorkerError) as excinfo:
        parse(payload, cfg=cfg)
    assert excinfo.value.code in ("bad_request", "unknown_setting", "invalid_setting")
    assert excinfo.value.message
    assert excinfo.value.retryable is False


def test_null_settings_is_treated_as_no_settings(cfg):
    """``"settings": null`` is not an error; it runs the model's demo prompt, and
    that is exactly the case the demo-prompt warning exists for."""
    req = parse({"settings": None}, cfg=cfg)
    assert req.settings["prompt"]
    assert any("demo prompt" in item for item in req.warnings)


def test_parse_does_not_mutate_the_caller_payload(cfg):
    payload = example_b()
    snapshot = json.loads(json.dumps(payload))
    parse(payload, FL2VA, cfg=cfg)
    assert payload == snapshot


def test_defaults_are_not_shared_between_requests(cfg):
    """``parse`` deep-copies the defaults; a leak here would let one request's
    settings bleed into the next on a warm worker."""
    defaults = default_settings(FL2VA_PRUNED)
    first = S.parse({"settings": {"prompt": DEMO_PROMPT, "video_length": 209}},
                    model_type=FL2VA_PRUNED, allowed_settings=defaults, cfg=cfg)
    second = S.parse({"settings": {"prompt": DEMO_PROMPT}},
                     model_type=FL2VA_PRUNED, allowed_settings=defaults, cfg=cfg)
    assert first.settings["video_length"] == 209
    assert second.settings["video_length"] == 124
    assert defaults["video_length"] == 124


def test_request_to_dict_round_trips_as_json(cfg):
    req = parse(example_d(), REF2VA, cfg=cfg)
    body = json.dumps(req.to_dict())
    assert json.loads(body)["model_type"] == REF2VA
