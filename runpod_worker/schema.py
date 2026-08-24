"""Request validation and settings assembly for the RunPod WanGP worker.

Standard library only: no torch, no wgp, no CUDA, no third-party imports, and no
network. ``parse()`` is a pure function of its arguments -- everything it needs
about the model (default settings, ``model_def``) is *passed in* by the caller,
which is what makes this module and its tests runnable on a plain CPU runner.

Two escape hatches exist and are the only impure paths in the file:

* ``parse(..., session=...)`` -- the handler may hand us a live
  ``shared.api.WanGPSession``; we then call ``session.get_model_schema(mt)``
  (``shared/api.py:543-556``) instead of requiring ``allowed_settings``.
  Nothing is imported from WanGP to do it; it is duck-typed.
* ``profile`` in the payload -- accelerator profiles are JSON fragments on disk
  (``profiles/<profiles_dir>/<name>.json``, ``wgp.py:8893``). ``parse`` reads one
  *only* when the caller asked for a profile, and a ``profile_loader`` callable
  can be injected to keep tests off the filesystem entirely.

Everything here mirrors validation WanGP performs anyway. The point of doing it
twice is *when*: WanGP's own checks run inside ``validate_settings``
(``wgp.py:983``) after the task is queued, and its model-dependent ones run in
``MinimaxH3Handler.validate_generative_settings``
(``models/minimax_h3/minimax_h3_handler.py:345-445``). Failing there is cheap in
absolute terms but expensive on a serverless GPU that has already spent minutes
loading 20-33B of weights. Every rule below is cited to the line it mirrors.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from .errors import BAD_REQUEST, INVALID_SETTING, UNKNOWN_SETTING, WorkerError

__all__ = [
    # media / attachment vocabulary
    "ATTACHMENT_KEYS",
    "IMAGE_EXTS",
    "AUDIO_EXTS",
    "VIDEO_EXTS",
    "EXTS_BY_KIND",
    "MEDIA_KIND",
    "LIST_KEYS",
    "MEDIA_SOURCE_KEYS",
    # model vocabulary
    "MINIMAX_H3_TYPES",
    "FL2VA_TYPES",
    "REF2VA_TYPES",
    "DEFAULT_MODEL_TYPE",
    "FORBIDDEN_KEYS",
    "PRIMARY_SETTINGS",
    "POISON_MARKERS",
    "OUTPUT_MODES",
    "OUTPUT_MODE_ALIASES",
    "SEED_MAX",
    "MIN_BUDGET_S",
    "FIRST_BLOCK_CACHE_THRESHOLDS",
    # frame math
    "floor_frames",
    "normalize_frames",
    "round_overlap",
    "frame_lattice",
    "legal_frame_counts",
    "is_legal_frame_count",
    # model-def helpers
    "fallback_model_def",
    "letters_allowed",
    "resolve_seed",
    "read_primary_settings",
    "load_profile_fragment",
    "check_cross_variant",
    "RESOLVED_ECHO_KEYS",
    # the entry point
    "Request",
    "parse",
]


# ---------------------------------------------------------------------------
# Attachment vocabulary
# ---------------------------------------------------------------------------

#: Verified against ``wgp.py:167-168`` -- 15 keys, same order. CI re-derives this
#: by ``ast.literal_eval``ing the list literal out of ``wgp.py`` (no import, no
#: torch); see ``tests/test_schema.py::test_attachment_keys_match``.
#: ``engine._assert_attachment_keys`` makes the same comparison at boot against
#: the live module, so a silent upstream addition fails fast instead of turning
#: into an unvalidated passthrough.
ATTACHMENT_KEYS: tuple[str, ...] = (
    "image_start",
    "image_end",
    "image_refs",
    "image_guide",
    "image_mask",
    "video_guide",
    "video_guide2",
    "video_mask",
    "video_source",
    "audio_guide",
    "audio_guide2",
    "audio_source",
    "replace_voice_sample",
    "replace_voice_sample2",
    "custom_guide",
)

# Extension whitelists WanGP itself enforces (``shared/utils/utils.py:36-49``).
# These are what ``has_image_file_extension`` / ``has_video_file_extension`` /
# ``has_audio_file_extension`` accept, and WanGP dispatches on the *extension*,
# never on content -- which is exactly why media_in.py sniffs magic bytes and
# then names the temp file itself.
# NOTE: ``.webm`` is NOT accepted; ``.avi`` IS. ``.flac``/``.ogg``/``.m4a`` are
# NOT accepted either -- the audio list is only wav/mp3/aac.
IMAGE_EXTS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".jfif", ".pjpeg"}
)
AUDIO_EXTS: frozenset[str] = frozenset({".wav", ".mp3", ".aac"})
VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".mkv", ".avi", ".mov"})

EXTS_BY_KIND: dict[str, frozenset[str]] = {
    "image": IMAGE_EXTS,
    "audio": AUDIO_EXTS,
    "video": VIDEO_EXTS,
}

#: Attachment key -> media kind. Drives media_in's sniff/extension decision and
#: the "is this slot even a video slot" checks below.
MEDIA_KIND: dict[str, str] = {}
MEDIA_KIND.update(
    {k: "image" for k in ("image_start", "image_end", "image_refs", "image_guide", "image_mask")}
)
MEDIA_KIND.update(
    {k: "video" for k in ("video_guide", "video_guide2", "video_mask", "video_source")}
)
MEDIA_KIND.update(
    {
        k: "audio"
        for k in (
            "audio_guide",
            "audio_guide2",
            "audio_source",
            "replace_voice_sample",
            "replace_voice_sample2",
        )
    }
)
#: ``custom_guide`` is model-defined (``wgp.py:1286-1291``: ``model_def["custom_guide"]``).
#: MiniMax H3 declares no ``custom_guide``, so WanGP nulls it (``wgp.py:1291``);
#: it is accepted here as a video-ish slot only so the key is not silently
#: unknown. ``_warn_ignored_media`` tells the caller it will be dropped.
MEDIA_KIND.setdefault("custom_guide", "video")

#: Attachment keys whose value is a *list* of media specs. WanGP treats
#: ``image_refs`` as a gallery (``wgp.py:1330-1337``); every other slot is one file.
LIST_KEYS: frozenset[str] = frozenset({"image_refs"})

#: Keys a media spec object may carry to name its source. media_in.py owns the
#: semantics; schema only checks that at least one is present and that ``url``
#: is permitted by policy (``ALLOW_URL_INPUTS``, default off).
#:
#: This tuple MUST stay a superset-free match for ``media_in._normalize_item``'s
#: ``accepted`` set: a key accepted here and rejected there is a request that
#: passes validation and then dies in materialization. ``base64`` is the one
#: alias (media_in renames it to ``b64``). There is deliberately no ``path``
#: key -- it is the only source form with no scheme, i.e. indistinguishable
#: from a path on the worker's own filesystem.
MEDIA_SOURCE_KEYS: tuple[str, ...] = ("b64", "base64", "volume", "url")

#: The ``scheme://`` string shorthands ``media_in._normalize_item`` accepts, and
#: the source key each one becomes. A *bare* string stays refused: it would name
#: a path on the worker's filesystem.
MEDIA_STRING_PREFIXES: tuple[tuple[str, str], ...] = (
    ("volume://", "volume"),
    ("http://", "url"),
    ("https://", "url"),
    ("data:", "b64"),
)

#: Hard ceiling on how many attachments one request may carry, over every key.
#: The byte budget alone cannot stop an entry-count attack: 20k valid 14-byte
#: GIFs cost 280 KB against a 7 MB budget and 20k inodes on the container disk,
#: plus a 20k-element ``image_refs`` list handed to WanGP's validator. MiniMax
#: H3 takes one reference image (FL2VA, ``one_image_ref_only``) or at most nine
#: (Ref2VA), so a low cap costs nothing real.
DEFAULT_MAX_MEDIA_ITEMS = 16


# ---------------------------------------------------------------------------
# Model vocabulary
# ---------------------------------------------------------------------------

FL2VA_TYPES: frozenset[str] = frozenset({"minimax_h3_fl2va", "minimax_h3_fl2va_pruned"})
REF2VA_TYPES: frozenset[str] = frozenset({"minimax_h3_ref2va", "minimax_h3_ref2va_pruned"})
MINIMAX_H3_TYPES: frozenset[str] = FL2VA_TYPES | REF2VA_TYPES

DEFAULT_MODEL_TYPE = "minimax_h3_fl2va_pruned"

# Keys a caller may never set: they steer WanGP away from the generation path or
# let a caller name an arbitrary local file. ``mode`` in particular flips
# ``_is_edit_task_params`` / ``validate_task`` into the edit branch
# (``wgp.py:1871-1872``, ``wgp.py:8567``) which reads ``video_source`` straight
# off disk. All 15 ATTACHMENT_KEYS are here because media may arrive ONLY through
# ``input.media``, where it is materialized into a job-scoped temp directory --
# ``settings.image_start = "/etc/hostname"`` would otherwise sail through
# ``_absolutize_setting_path`` (``shared/api.py:1028-1043``) into WanGP.
#: Post-processing selectors a caller may never set. Every one of them is read
#: by ``download_requested_postprocessing_assets`` (``wgp.py:3532-3539``), which
#: ``generate_media`` calls ON THE REQUEST PATH at ``wgp.py:6786`` -- after the
#: model is loaded and the clock is running. A caller could therefore reintroduce
#: a multi-GB download plus a second model load into a billed generation, which
#: is exactly what the boot-time weight gate exists to prevent
#: (``engine.assert_weights_complete`` proves only the pinned model's core files).
#: ``prompt_enhancer`` is here for the same reason plus a privacy one: WanGP
#: prints the enhanced prompt to stdout (``wgp.py:7276``), which
#: ``WANGP_CONSOLE=1`` routes to the container log.
POSTPROCESS_KEYS: frozenset[str] = frozenset(
    {
        "postprocess_audio",
        "prompt_enhancer",
        "replace_voice_method",
        "spatial_upsampling",
        "temporal_upsampling",
    }
)

FORBIDDEN_KEYS: frozenset[str] = (
    frozenset(ATTACHMENT_KEYS)
    | frozenset({"mode", "_api", "client_id", "state", "type", "base_model_type", "priority"})
    | POSTPROCESS_KEYS
)

#: Substrings that mean the CUDA context is poisoned rather than the request
#: being wrong. The handler scans WanGP's error messages with these and sets
#: ``refresh_worker`` (failure mode 16). Lives here so handler.py needs no
#: additional import.
POISON_MARKERS: tuple[str, ...] = (
    "cuda error",
    "out of memory",
    "cublas_status",
    "device-side assert",
    "illegal memory access",
    "nccl",
)

#: Output transports, in the order ``media_out.deliver`` tries them for "auto".
OUTPUT_MODES: tuple[str, ...] = ("auto", "presigned", "rp_bucket", "volume", "base64")
#: Spellings we accept and rewrite, so ``media_out`` only ever sees the canonical
#: name it branches on.
OUTPUT_MODE_ALIASES: dict[str, str] = {
    "s3": "rp_bucket",
    "bucket": "rp_bucket",
    "rp_upload": "rp_bucket",
    "b64": "base64",
    "inline": "base64",
    "presigned_url": "presigned",
    "put": "presigned",
    "network_volume": "volume",
}

#: WanGP's own random-seed range (``wgp.py:5775``: ``random.randint(0, 999999999)``;
#: the UI slider at ``wgp.py:12036`` is ``gr.Slider(-1, 999999999)``). We resolve
#: into ``[1, SEED_MAX]`` -- excluding 0 only so a resolved seed is never falsy.
SEED_MAX = 999_999_999

#: Floor for a per-request wall-clock budget. Below this nothing can finish, so
#: accepting it would only produce a guaranteed ``timeout``.
MIN_BUDGET_S = 60

#: ``models/minimax_h3/minimax_h3_handler.py:30``. Enforced by WanGP at
#: ``wgp.py:1215`` -- ``float(skip_steps_multiplier) not in
#: model_def["first_block_cache_thresholds"]`` -> "Unsupported First Block Cache
#: threshold". Note that ``wgp.py:1207`` blanks ``skip_steps_cache_type`` when the
#: model supports no caching at all, so the check only bites for ``first_block``.
FIRST_BLOCK_CACHE_THRESHOLDS: tuple[float, ...] = (0.06, 0.08, 0.10, 0.12, 0.14)


# ---------------------------------------------------------------------------
# The settings universe
# ---------------------------------------------------------------------------

# CRITICAL: ``get_default_settings()`` is NOT the settings universe. For
# minimax_h3 it returns only the keys ``update_default_settings`` writes
# (``minimax_h3_handler.py:513-534``) plus ``settings_version``/``prompt``/
# ``resolution``/``flow_shift`` (``wgp.py:3155-3164``) -- 18-20 keys. ``seed``,
# ``activated_loras``, ``frames_positions``, ``masking_strength``,
# ``skip_steps_cache_type``, ``negative_prompt``, ``config`` and
# ``override_attention`` are all ABSENT from it. The real universe is
# ``models/_settings.json`` (112 keys), merged in by ``clean_settings``
# (``wgp.py:1747-1760``).
#
# The list is baked rather than read at import so this module does zero I/O and
# cannot fail to import on a runner without the repo laid out at WANGP_ROOT.
# ``read_primary_settings()`` re-reads the file for a drift test.
_PRIMARY_SETTINGS_KEYS: tuple[str, ...] = (
    "NAG_alpha",
    "NAG_scale",
    "NAG_tau",
    "RIFLEx_setting",
    "activated_loras",
    "alt_guidance_scale",
    "alt_prompt",
    "alt_scale",
    "apg_switch",
    "attention_sparsity",
    "audio_guidance_scale",
    "audio_guide",
    "audio_guide2",
    "audio_prompt_type",
    "audio_scale",
    "audio_source",
    "batch_size",
    "cfg_star_switch",
    "cfg_zero_step",
    "client_id",
    "config",
    "control_net_weight",
    "control_net_weight2",
    "control_net_weight_alt",
    "custom_guide",
    "custom_settings",
    "denoising_strength",
    "duration_seconds",
    "embedded_guidance_scale",
    "film_grain_intensity",
    "film_grain_saturation",
    "flow_shift",
    "force_fps",
    "frames_positions",
    "guidance2_scale",
    "guidance3_scale",
    "guidance_phases",
    "guidance_scale",
    "image_end",
    "image_guide",
    "image_mask",
    "image_mode",
    "image_prompt_type",
    "image_refs",
    "image_refs_relative_size",
    "image_start",
    "input_video_strength",
    "keep_frames_video_guide",
    "keep_frames_video_source",
    "loras_multipliers",
    "mask_expand",
    "masking_strength",
    "min_frames_if_references",
    "model_mode",
    "model_switch_phase",
    "motion_amplitude",
    "multi_images_gen_type",
    "multi_prompts_gen_type",
    "negative_prompt",
    "num_inference_steps",
    "output_filename",
    "override_attention",
    "override_profile",
    "pause_seconds",
    "perturbation_end_perc",
    "perturbation_layers",
    "perturbation_start_perc",
    "perturbation_switch",
    "postprocess_audio",
    "postprocess_audio_neg_prompt",
    "postprocess_audio_prompt",
    "prompt",
    "prompt_enhancer",
    "remove_background_images_ref",
    "repeat_generation",
    "replace_voice_method",
    "replace_voice_sample",
    "replace_voice_sample2",
    "resolution",
    "sample_solver",
    "seed",
    "self_refiner_certain_percentage",
    "self_refiner_f_uncertainty",
    "self_refiner_plan",
    "self_refiner_setting",
    "skip_steps_cache_type",
    "skip_steps_multiplier",
    "skip_steps_start_step_perc",
    "sliding_window_color_correction_strength",
    "sliding_window_discard_last_frames",
    "sliding_window_overlap",
    "spatial_upsampler_face_count",
    "spatial_upsampler_prompt",
    "spatial_upsampler_reference_images",
    "sliding_window_overlap_noise",
    "sliding_window_size",
    "sliding_window_trim_first_frames",
    "spatial_upsampling",
    "speakers_locations",
    "sub_parallel_window_overlap",
    "sub_parallel_window_size",
    "switch_threshold",
    "switch_threshold2",
    "temperature",
    "temporal_upsampling",
    "top_k",
    "top_p",
    "video_guide",
    "video_guide2",
    "video_guide_outpainting",
    "video_guide_outpainting_ratio",
    "video_length",
    "video_mask",
    "video_prompt_type",
    "video_source",)

PRIMARY_SETTINGS: frozenset[str] = frozenset(_PRIMARY_SETTINGS_KEYS)


def read_primary_settings(root: str | os.PathLike[str] | None = None) -> frozenset[str]:
    """Re-read ``models/_settings.json`` from disk (drift test helper).

    Not used by :func:`parse`; the baked :data:`PRIMARY_SETTINGS` is. A CI test
    asserts the two agree, which is how an upstream key addition is noticed.
    """
    base = Path(root) if root is not None else Path(os.environ.get("WANGP_ROOT") or _repo_root())
    with open(base / "models" / "_settings.json", "r", encoding="utf-8") as handle:
        return frozenset(json.load(handle))


def _repo_root() -> Path:
    """The WanGP checkout this package lives inside (``runpod_worker/..``)."""
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# A model_def we can validate against with no WanGP process alive
# ---------------------------------------------------------------------------
#
# Every value below is copied from ``MinimaxH3Handler.query_model_def``
# (``models/minimax_h3/minimax_h3_handler.py:176-352``). When a live session is
# available its real ``model_def`` wins key-by-key; this only fills the gaps, so
# a CPU test (and a caller who passes nothing but ``model_type``) still gets the
# true frame lattice and the true letter whitelists.

_SLIDING_WINDOW_DEFAULTS = {
    "window_min": 124,
    "window_max": 481,
    "window_step": 17,
    "window_default": 362,
    "overlap_min": 1,
    "overlap_max": 120,
    "overlap_step": 17,
    "overlap_offset": 1,
    "overlap_default": 18,
}

_MINIMAX_H3_COMMON: dict[str, Any] = {
    "dtype": "bf16",
    "fps": 24,
    # The frame lattice: video_length must be >= 107 and == 5 (mod 17).
    # minimax_h3_handler.py:185-187. WanGP floors to it at wgp.py:6929.
    "frames_minimum": 107,
    "frames_steps": 17,
    "frames_offset": 5,
    "block_size": 32,
    "vae_block_size": 32,
    # guidance_max_phases == 0: MiniMaxH3Pipeline.generate takes no CFG argument,
    # so guidance_scale is inert (minimax_h3_handler.py:190).
    "guidance_max_phases": 0,
    "inference_steps": True,
    "flow_shift": True,
    "spectrum_cache": True,
    "first_block_cache": True,
    "first_block_cache_thresholds": FIRST_BLOCK_CACHE_THRESHOLDS,
    # upstream 238e25f added Ralston 2S: two full transformer predictions per
    # step, so ~2x slower than Euler at the same step count.
    "sample_solvers": [("Euler", "euler"), ("RES Multistep", "res_multistep"),
                      ("Ralston 2S (~2x slower)", "ralston_2s")],
    "no_negative_prompt": True,
    "returns_audio": True,
    "multimedia_generation": True,
    "profiles_dir": ["minimax_h3"],
    "keep_frames_video_guide_not_supported": True,
    "sliding_window": True,
    "video_continuation": True,
    "sliding_window_defaults": _SLIDING_WINDOW_DEFAULTS,
    "image_prompt_types_allowed": "TSEVL",
    "end_frames_always_enabled": True,
    "audio_guide_window_slicing": True,
    "video_length_not_limited_by_audio": True,
    # The `config` selection groups (shared/config_groups.py:1-3). Values are
    # the ids a caller may name; the dicts are empty because only the ids matter
    # for validation.
    "system_configs": {
        "_name": "Text Encoder",
        "bf16": {},
        "int8": {},
        "nvfp4_awq": {},
        "gguf_q4_k_m": {},
        "gguf_q2_k": {},
    },
    "system_configs2": {"_name": "Video VAE", "_default_label": "Original VAE", "fp8mix": {}},
    "system_configs3": {
        "_name": "DiT Denoising Priority",
        "_default_label": "Lower VRAM",
        "lower_ram": {},
    },
}

_FL2VA_MODEL_DEF: dict[str, Any] = {
    **_MINIMAX_H3_COMMON,
    # minimax_h3_handler.py:313-321 -- letters_filter "GVKFI".
    # NOTE there is no frames_maximum here: FL2VA has no upper frame bound
    # anywhere in the headless path, which is why WANGP_MAX_FRAMES exists.
    "guide_custom_choices": {"letters_filter": "GVKFI", "default": ""},
    "audio_prompt_type_sources": {"letters_filter": "AK2", "default": ""},
    "mask_preprocessing": {"selection": ["", "A", "NA"]},
    "custom_frames_injection": True,
    "one_image_ref_only": True,
    "no_background_removal": True,
    "any_audio_prompt": True,
    "output_audio_is_input_audio": True,
}

_REF2VA_MODEL_DEF: dict[str, Any] = {
    **_MINIMAX_H3_COMMON,
    # minimax_h3_handler.py:258 -- upstream 238e25f renamed this from
    # "frames_maximum" and demoted it to a UI-only slider bound (wgp.py:11897).
    # The headless path no longer reads it at all: shared/api.py:109-118 derives
    # its maximum from the sliding-window default instead. Keep declaring it so
    # the cap message can still cite the model's own number, but WANGP_MAX_FRAMES
    # is now the ONLY real bound for Ref2VA as well as FL2VA.
    "frames_selection_maximum": 737,
    # minimax_h3_handler.py:265-279 -- "PDEV+-" for the guide, "KI" for refs.
    "guide_custom_choices": {"letters_filter": "PDEV+-", "default": ""},
    "image_ref_choices": {"letters_filter": "KI", "default": ""},
    "audio_prompt_type_sources": {"letters_filter": "ABK", "default": ""},
    "reference_image_enabled": True,
    "any_image_refs_relative_size": True,
    "image_refs_relative_size": {"min": 50, "max": 400, "step": 1},
    "preprocess_video_guide2": True,
    "reference_video_max_frames": 15 * 24,
    "any_audio_prompt": True,
}


def fallback_model_def(model_type: str) -> dict[str, Any]:
    """A source-derived ``model_def`` for a MiniMax H3 variant.

    Used when no live session is available. Returns a fresh dict every call.
    """
    base = _REF2VA_MODEL_DEF if model_type in REF2VA_TYPES else _FL2VA_MODEL_DEF
    out = copy.deepcopy(base)
    out["model_type"] = str(model_type)
    return out


# ---------------------------------------------------------------------------
# Frame arithmetic -- exact mirrors of shared/utils/frame_scheduler.py
# ---------------------------------------------------------------------------


def normalize_frames(frame_count: int, minimum: int, step: int, offset: int = 1) -> int:
    """Mirror ``normalize_frame_count`` (``frame_scheduler.py:15-19``): round UP."""
    frame_count = max(int(minimum), int(frame_count))
    step = max(1, int(step))
    offset = max(0, int(offset))
    if step <= 1:
        return frame_count
    return math.ceil(max(0, frame_count - offset) / step) * step + offset


# NOTE on the step used to floor `video_length`. wgp.py:6929 floors with
# `latent_size`, not `frames_steps`:
#     frames_minimum, frames_steps, latent_size = get_model_min_frames_and_step(...)
#     latent_size = model_def.get("latent_size", frames_steps)     # wgp.py:2853
# MiniMax H3 declares no `latent_size`, so latent_size == frames_steps == 17 for
# all four types and `frame_lattice`'s use of frames_steps is exact. A future
# model that declares a different `latent_size` would make the two diverge --
# read it out of model_def here rather than assuming they stay equal.
def floor_frames(frame_count: int, minimum: int, step: int, offset: int = 1) -> int:
    """Mirror ``floor_frame_count`` (``frame_scheduler.py:22-29``): round DOWN,
    but never below ``minimum`` -- in which case it rounds the minimum UP onto
    the lattice, exactly as upstream does.

    This is the function WanGP applies to ``video_length`` and
    ``sliding_window_size`` at ``wgp.py:6929-6931``, so mirroring it is what
    makes the ``resolved`` block we echo back match what actually ran.
    """
    frame_count = max(int(minimum), int(frame_count))
    step = max(1, int(step))
    offset = max(0, int(offset))
    if step <= 1:
        return frame_count
    lower = ((frame_count - offset) // step) * step + offset
    if lower >= minimum:
        return lower
    return normalize_frames(minimum, minimum, step, offset)


def round_overlap(frame_count: int, step: int, offset: int = 1) -> int:
    """Mirror ``normalize_overlap`` (``frame_scheduler.py:41-49``).

    This ROUNDS TO NEAREST, it does not floor: with step 17 / offset 1,
    30 -> 35 and 27 -> 35, while 20 -> 18. Upstream returns
    ``(value, error)``; a negative count is its only error and we reject that
    before calling, so this returns the int alone.

    ``MinimaxH3Handler.validate_generative_settings`` applies exactly this to
    ``sliding_window_overlap`` on every request (``minimax_h3_handler.py:346-349``).
    """
    frame_count = int(frame_count)
    if frame_count <= 0:
        return 0
    step = max(1, int(step))
    offset = max(0, int(offset))
    overlap = ((frame_count - offset + step // 2) // step) * step + offset
    return max(step if offset == 0 else offset, overlap)


def frame_lattice(model_def: Mapping[str, Any] | None) -> tuple[int, int, int]:
    """``(minimum, step, offset)`` for a model. Defaults match a step-1 model.

    The step is ``latent_size`` when the model declares one, exactly as
    ``get_model_min_frames_and_step`` does (``wgp.py:2853``:
    ``latent_size = model_def.get("latent_size", frames_steps)``), because that
    -- not ``frames_steps`` -- is what ``wgp.py:6929`` floors ``video_length``
    with. The two are equal for all four MiniMax H3 types (neither declares
    ``latent_size``, both declare ``frames_steps: 17``), so this changes nothing
    today; it is here so a future model that separates them cannot make this
    module quietly predict the wrong lattice.
    """
    md = model_def or {}
    minimum = max(1, int(md.get("frames_minimum", 1) or 1))
    step = max(1, int(md.get("latent_size") or md.get("frames_steps", 1) or 1))
    offset = max(0, int(md.get("frames_offset", 0) or 0))
    return minimum, step, offset


def legal_frame_counts(
    model_def: Mapping[str, Any] | None, cap: int, *, limit: int = 4096
) -> tuple[int, ...]:
    """Every legal ``video_length`` from the lattice minimum up to ``cap``.

    For MiniMax H3 that is ``107, 124, 141, ... `` -- i.e. ``5 + 17k`` for the
    smallest ``k`` giving at least 107 (``k = 6`` -> 107) and up. ``limit`` keeps
    a pathological cap from materializing a huge tuple.
    """
    minimum, step, offset = frame_lattice(model_def)
    start = normalize_frames(minimum, minimum, step, offset)
    if cap < start:
        return ()
    if step <= 1:
        return tuple(range(start, min(cap, start + limit - 1) + 1))
    count = min(limit, (cap - start) // step + 1)
    return tuple(start + index * step for index in range(count))


def is_legal_frame_count(value: int, model_def: Mapping[str, Any] | None) -> bool:
    """Whether ``value`` sits on the model's frame lattice."""
    minimum, step, offset = frame_lattice(model_def)
    value = int(value)
    if value < normalize_frames(minimum, minimum, step, offset):
        return False
    return step <= 1 or (value - offset) % step == 0


# ---------------------------------------------------------------------------
# Letter whitelists (image_prompt_type / video_prompt_type / audio_prompt_type)
# ---------------------------------------------------------------------------
#
# WanGP encodes modes as letter soups. The legal alphabet per model is declared
# in model_def, not hard-coded: ``image_prompt_types_allowed`` (checked at
# wgp.py:1396-1398 and :1421-1423), and the letters of the FOUR video groups
# wgp.py builds the video_prompt_type dropdowns from:
#
#   group                    filter / source                       wgp.py
#   guide_custom_choices     model_def["letters_filter"]           11542
#   custom_video_selection   model_def["letters_filter"]           11568
#   mask_preprocessing       model_def["selection"] & "XYZWNA"     11602
#   image_ref_choices        model_def["letters_filter"]           11630
#
# Omitting any of them makes a documented mode unreachable: FL2VA declares
# ``mask_preprocessing: {"selection": ["", "A", "NA"]}``
# (minimax_h3_handler.py:322) -> "Masked Area" / "Non Masked Area"
# (wgp.py:11580-11582), which wgp.py:1350-1356 then enforces with
# "You must provide a Video Mask".
#
# "U" ("identity", wgp.py:4584) is deliberately NOT allowed: it belongs to the
# ``guide_preprocessing`` group, which neither MiniMax H3 variant declares, so it
# is unreachable from the UI for these models. It only ever appears as the opt-out
# half of ``"A" in vpt and not "U" in vpt``; allowing it would let a caller ask for
# masked processing and then suppress the mask requirement WanGP enforces.
#
# An illegal letter is not always an error upstream -- some are silently dropped
# -- so rejecting here is strictly more informative than letting it through.

_FALLBACK_LETTERS: dict[str, dict[str, str]] = {
    "fl2va": {
        "image_prompt_type": "TSEVL",
        # "GVKFI" from guide_custom_choices + "AN" from mask_preprocessing
        # (selection ["", "A", "NA"], filtered through wgp.py:11602's "XYZWNA").
        "video_prompt_type": "GVKFIAN",
        "audio_prompt_type": "AK2",
    },
    "ref2va": {
        "image_prompt_type": "TSEVL",
        "video_prompt_type": "KIPDEV+-",
        "audio_prompt_type": "ABK",
    },
}


#: ``mask_letter_filter`` at wgp.py:11602. A mask ``selection`` entry may only
#: contribute letters from this set.
_MASK_LETTERS = "XYZWNA"


def _dedupe(text: str) -> str:
    seen: list[str] = []
    for char in text:
        if char not in seen:
            seen.append(char)
    return "".join(seen)


def letters_allowed(model_type: str, model_def: Mapping[str, Any] | None = None) -> dict[str, str]:
    """The legal alphabet for each ``*_prompt_type`` setting of ``model_type``."""
    md: Mapping[str, Any] = model_def or {}
    variant = "ref2va" if model_type in REF2VA_TYPES else "fl2va"
    fallback = _FALLBACK_LETTERS[variant]

    image = str(md.get("image_prompt_types_allowed") or "") or fallback["image_prompt_type"]

    video = ""
    for group_key in ("guide_custom_choices", "custom_video_selection", "image_ref_choices"):
        group = md.get(group_key)
        if isinstance(group, Mapping):
            video += str(group.get("letters_filter") or "")
    # The mask group names whole *modes* ("", "A", "NA"), not a letter filter;
    # wgp.py:11602 constrains them with "XYZWNA" on the way in.
    mask = md.get("mask_preprocessing")
    if isinstance(mask, Mapping):
        for choice in mask.get("selection") or ():
            video += "".join(char for char in str(choice) if char in _MASK_LETTERS)
    video = _dedupe(video) or fallback["video_prompt_type"]

    audio_group = md.get("audio_prompt_type_sources")
    audio = ""
    if isinstance(audio_group, Mapping):
        audio = str(audio_group.get("letters_filter") or "")
    audio = _dedupe(audio) or fallback["audio_prompt_type"]

    return {
        "image_prompt_type": _dedupe(image),
        "video_prompt_type": video,
        "audio_prompt_type": audio,
    }


# ---------------------------------------------------------------------------
# The parsed request
# ---------------------------------------------------------------------------


class Request:
    """A validated job. ``settings`` is ready for ``session.submit_task`` once
    media_in has replaced the ``media`` specs with absolute temp paths."""

    __slots__ = (
        "model_type",
        "settings",
        "media",
        "output",
        "runtime",
        "budget_s",
        "priority",
        "idempotency_key",
        "profile",
        "warnings",
        "resolved",
    )

    def __init__(self) -> None:
        self.model_type: str = ""
        self.settings: dict[str, Any] = {}
        self.media: dict[str, Any] = {}
        self.output: dict[str, Any] = {}
        self.runtime: dict[str, Any] = {}
        self.budget_s: int = MIN_BUDGET_S
        self.priority: int = 0
        self.idempotency_key: str | None = None
        self.profile: str | None = None
        self.warnings: list[str] = []
        self.resolved: dict[str, Any] = {}

    # ``timeout_s`` is the wire spelling of ``budget_s``; keep both readable.
    @property
    def timeout_s(self) -> int:
        return self.budget_s

    @property
    def seed(self) -> int:
        return int(self.settings.get("seed", -1))

    def to_dict(self) -> dict[str, Any]:
        """Debug/echo view. Not the response envelope -- handler builds that."""
        return {
            "model_type": self.model_type,
            "profile": self.profile,
            "settings": copy.deepcopy(self.settings),
            "media": copy.deepcopy(self.media),
            "output": copy.deepcopy(self.output),
            "runtime": copy.deepcopy(self.runtime),
            "budget_s": self.budget_s,
            "priority": self.priority,
            "idempotency_key": self.idempotency_key,
            "warnings": list(self.warnings),
            "resolved": copy.deepcopy(self.resolved),
        }

    def __repr__(self) -> str:
        return (
            f"Request(model_type={self.model_type!r}, seed={self.settings.get('seed')!r}, "
            f"video_length={self.settings.get('video_length')!r}, "
            f"media={sorted(self.media)!r}, budget_s={self.budget_s!r})"
        )


#: Keys echoed back to the client so a job can be reproduced exactly.
RESOLVED_ECHO_KEYS: tuple[str, ...] = (
    "model_type",
    "seed",
    "video_length",
    "num_inference_steps",
    "resolution",
    "flow_shift",
    "sample_solver",
    "sliding_window_size",
    "sliding_window_overlap",
    "config",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _default_cfg() -> Any:
    """The process-wide :class:`config.WorkerConfig` (imported lazily so this
    module has no import-time dependency on anything but ``errors``)."""
    from . import config as _config

    return _config.CONFIG


def _flag(cfg: Any, attr: str, env: str, default: str = "0") -> bool:
    value = getattr(cfg, attr, None)
    if value is None:
        return os.environ.get(env, default) == "1"
    return bool(value)


def _as_int(value: Any, field: str, *, code: str = INVALID_SETTING) -> int:
    """Strict int coercion. Accepts ``5``/``5.0``/``"5"``; rejects ``5.5``."""
    if isinstance(value, bool):
        raise WorkerError(code, f"{field} must be an integer, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise WorkerError(code, f"{field} must be an integer (got {value!r})") from None
    if not math.isfinite(number) or number != int(number):
        raise WorkerError(code, f"{field} must be a whole number (got {value!r})")
    return int(number)


def _as_float(value: Any, field: str, *, code: str = INVALID_SETTING) -> float:
    if isinstance(value, bool):
        raise WorkerError(code, f"{field} must be a number, not a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise WorkerError(code, f"{field} must be a number (got {value!r})") from None
    if not math.isfinite(number):
        raise WorkerError(code, f"{field} must be finite (got {value!r})")
    return number


def resolve_seed(value: Any = None, *, warnings: list[str] | None = None) -> int:
    """Resolve a request seed to a concrete int so the response can echo it.

    WanGP treats any negative seed as "random" (``wgp.py:6924`` nulls -1 and
    ``wgp.py:5775`` draws ``random.randint(0, 999999999)``) and never tells the
    caller what it drew. Resolving here is what makes a generation reproducible
    and what lets the idempotency key be derived before the GPU is touched.
    """
    if value is None or value == "":
        seed = -1
    else:
        seed = _as_int(value, "settings.seed")
    if seed < 0:
        seed = random.SystemRandom().randint(1, SEED_MAX)
        if warnings is not None:
            warnings.append(f"seed was unset or -1; resolved to {seed}")
    return seed


_CONFIG_GROUP_KEYS = ("system_configs", "system_configs2", "system_configs3", "configs")
_CONFIG_METADATA_KEYS = frozenset({"_name", "_default_label"})


def _serialize_config_selection(values: Sequence[str]) -> str:
    """Mirror ``serialize_config_selection`` (``shared/config_groups.py:18-20``):
    four comma-joined ids with trailing commas stripped. Storing the rstripped
    form matters -- ``load_models`` records ``config_id or ""`` verbatim
    (``wgp.py:4082``) and the reload gate at ``wgp.py:6773`` compares strings."""
    return ",".join(str(value or "") for value in list(values)[:4]).rstrip(",")


def _split_config_selection(selection: str) -> list[str]:
    """Mirror ``split_config_selection`` (``shared/config_groups.py:14-16``)."""
    values = str(selection or "").split(",")
    return (values + [""] * 4)[:4]


def _normalize_config_selection(selection: str) -> str:
    return _serialize_config_selection(_split_config_selection(selection))


def _validate_config_selection(selection: str, model_def: Mapping[str, Any]) -> None:
    """Reject a config id the model does not define.

    ``selected_model_configs`` raises a bare ``ValueError`` for an unknown id
    (``shared/config_groups.py:36``), which would surface as an opaque
    ``generation_failed`` minutes into a job.
    """
    for group_key, config_id in zip(_CONFIG_GROUP_KEYS, _split_config_selection(selection)):
        if not config_id:
            continue
        group = model_def.get(group_key)
        if not isinstance(group, Mapping) or not group:
            continue
        available = sorted(key for key in group if key not in _CONFIG_METADATA_KEYS)
        if config_id in _CONFIG_METADATA_KEYS or config_id not in group:
            raise WorkerError(
                INVALID_SETTING,
                f"config selection '{config_id}' is not defined in {group_key} for this model",
                details=[f"available {group_key}: {available}"],
            )


_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+()-]{0,127}$")
_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def load_profile_fragment(
    name: str,
    *,
    model_def: Mapping[str, Any] | None = None,
    root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Read an accelerator-profile settings fragment off disk.

    Profiles live at ``<profile_root>/<profiles_dir>/<name>.json``
    (``wgp.py:8891-8907``); ``profiles_dir`` for MiniMax H3 is ``["minimax_h3"]``
    (``minimax_h3_handler.py:220``) and the profile roots come from
    ``model_def["_profile_roots"]`` (``wgp.py:3205``, default ``["profiles"]``).
    The six shipped files are plain settings fragments -- ``activated_loras``,
    ``loras_multipliers``, ``num_inference_steps``, ``guidance_scale``,
    ``flow_shift``.

    ``name`` is caller-controlled, so it is charset-checked before it is ever
    joined onto a path: without that, ``profile: "../../etc/passwd"`` would be a
    file-read primitive that reports its result through the error message.
    """
    label = str(name).strip()
    if not _SAFE_NAME_RE.match(label) or ".." in label:
        raise WorkerError(
            BAD_REQUEST,
            "profile name may only contain letters, digits, spaces and '._+-()'",
            details=[f"got {label!r}"],
        )

    md: Mapping[str, Any] = model_def or {}
    base = Path(root) if root is not None else Path(os.environ.get("WANGP_ROOT") or _repo_root())

    profile_roots = md.get("_profile_roots") or ["profiles"]
    if isinstance(profile_roots, str):
        profile_roots = [profile_roots]
    profile_dirs = md.get("profiles_dir") or ["minimax_h3"]
    if isinstance(profile_dirs, str):
        profile_dirs = [profile_dirs]

    searched: list[str] = []
    available: list[str] = []
    for profile_root in profile_roots:
        root_path = Path(str(profile_root))
        if not root_path.is_absolute():
            root_path = base / root_path
        for folder in profile_dirs:
            folder_path = root_path / str(folder)
            candidate = folder_path / f"{label}.json"
            searched.append(str(candidate))
            if candidate.is_file():
                try:
                    with open(candidate, "r", encoding="utf-8") as handle:
                        fragment = json.load(handle)
                except (OSError, ValueError) as exc:
                    raise WorkerError(
                        BAD_REQUEST,
                        f"accelerator profile '{label}' could not be read: {exc}",
                        cause=exc,
                    ) from exc
                if not isinstance(fragment, dict):
                    raise WorkerError(
                        BAD_REQUEST,
                        f"accelerator profile '{label}' is not a JSON object",
                    )
                return fragment
            if folder_path.is_dir():
                available.extend(sorted(item.stem for item in folder_path.glob("*.json")))

    raise WorkerError(
        BAD_REQUEST,
        f"unknown accelerator profile '{label}'",
        details=[f"available: {sorted(set(available))}", f"searched: {searched}"],
    )


def _load_accel_profile(model_def: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Spec-sketch spelling of :func:`load_profile_fragment`."""
    return load_profile_fragment(name, model_def=model_def)


# ---------------------------------------------------------------------------
# Media block
# ---------------------------------------------------------------------------


def _max_media_items(cfg: Any) -> int:
    raw = getattr(cfg, "max_media_items", None)
    if raw in (None, ""):
        raw = os.environ.get("WANGP_MAX_MEDIA_ITEMS", DEFAULT_MAX_MEDIA_ITEMS)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_MEDIA_ITEMS
    return value if value > 0 else DEFAULT_MAX_MEDIA_ITEMS


def _validate_media(media: Any, cfg: Any) -> dict[str, Any]:
    """Structural validation of ``input.media``. Materialization is media_in's job.

    A *bare* string is refused on purpose: ``{"image_start": "/etc/hostname"}``
    would otherwise be indistinguishable from a legitimate path once media_in
    normalized it, and ``_absolutize_setting_path`` (``shared/api.py:1028-1043``)
    would happily hand it to WanGP. The ``scheme://`` shorthands
    (``MEDIA_STRING_PREFIXES``) are accepted and rewritten into their object
    form here, so exactly one contract reaches media_in.

    The attachment COUNT is capped (``WANGP_MAX_MEDIA_ITEMS``): the byte budget
    in media_in counts bytes, not entries, so thousands of tiny valid images fit
    inside it and would still cost thousands of files on the container disk and
    a list of the same length handed to WanGP's validator.
    """
    if media is None:
        return {}
    if not isinstance(media, Mapping):
        raise WorkerError(BAD_REQUEST, "input.media must be an object")

    allow_urls = _flag(cfg, "allow_url_inputs", "ALLOW_URL_INPUTS", "0")
    item_cap = _max_media_items(cfg)
    total_items = 0
    out: dict[str, Any] = {}

    for key, value in media.items():
        name = str(key)
        if name not in MEDIA_KIND:
            raise WorkerError(
                BAD_REQUEST,
                f"'{name}' is not a WanGP attachment key",
                details=[f"valid: {sorted(MEDIA_KIND)}"],
            )
        if value is None:
            continue
        if name in LIST_KEYS:
            items = list(value if isinstance(value, (list, tuple)) else [value])
            total_items += len(items)
            if total_items > item_cap:
                raise WorkerError(
                    BAD_REQUEST,
                    f"input.media carries more than {item_cap} attachments",
                    details=[
                        f"media['{name}'] alone has {len(items)}",
                        "raise WANGP_MAX_MEDIA_ITEMS if this endpoint really needs more",
                    ],
                )
            entries = [_validate_media_entry(f"{name}[{i}]", item, allow_urls)
                       for i, item in enumerate(items)]
            if not entries:
                continue
            out[name] = entries
        else:
            total_items += 1
            if total_items > item_cap:
                raise WorkerError(
                    BAD_REQUEST,
                    f"input.media carries more than {item_cap} attachments",
                    details=["raise WANGP_MAX_MEDIA_ITEMS if this endpoint really "
                             "needs more"],
                )
            if isinstance(value, (list, tuple)):
                raise WorkerError(
                    BAD_REQUEST,
                    f"media['{name}'] takes a single media object, not a list",
                    details=[f"only {sorted(LIST_KEYS)} accept a list"],
                )
            out[name] = _validate_media_entry(name, value, allow_urls)
    return out


def _expand_media_string(label: str, text: str) -> dict[str, Any]:
    """Rewrite a ``scheme://`` shorthand into the object form media_in expects.

    Mirrors ``media_in._normalize_item``'s string branch exactly, so the two
    modules cannot disagree about which strings are legal.
    """
    stripped = text.strip()
    lowered = stripped.lower()
    for prefix, source in MEDIA_STRING_PREFIXES:
        if lowered.startswith(prefix):
            if source == "volume":
                return {"volume": stripped[len(prefix):]}
            return {source: stripped}
    raise WorkerError(
        BAD_REQUEST,
        f"media['{label}'] must be an object such as "
        f'{{"b64": "..."}} or {{"volume": "clips/plate.mp4"}}, or a string starting '
        f"with {', '.join(prefix for prefix, _ in MEDIA_STRING_PREFIXES)}",
        details=["a bare string would name a path on the worker's filesystem"],
    )


def _validate_media_entry(label: str, entry: Any, allow_urls: bool) -> dict[str, Any]:
    if isinstance(entry, bytes):
        raise WorkerError(
            BAD_REQUEST,
            f"media['{label}'] must be an object or a string, not raw bytes",
        )
    if isinstance(entry, str):
        entry = _expand_media_string(label, entry)
    if not isinstance(entry, Mapping):
        raise WorkerError(BAD_REQUEST, f"media['{label}'] must be an object")
    unknown = sorted(set(entry) - set(MEDIA_SOURCE_KEYS) - {"range"})
    if unknown:
        raise WorkerError(
            BAD_REQUEST,
            f"media['{label}'] has unknown fields {unknown}",
            details=[f"accepted: {sorted(set(MEDIA_SOURCE_KEYS) | {'range'})}"],
        )
    present = [key for key in MEDIA_SOURCE_KEYS if entry.get(key) not in (None, "")]
    if not present:
        raise WorkerError(
            BAD_REQUEST,
            f"media['{label}'] names no source",
            details=[f"one of {list(MEDIA_SOURCE_KEYS)} is required"],
        )
    # media_in._normalize_item refuses more than one source; say so here rather
    # than after the request has been accepted.
    if len({"b64" if key == "base64" else key for key in present}) > 1:
        raise WorkerError(
            BAD_REQUEST,
            f"media['{label}'] names {len(present)} sources ({present}); exactly one "
            f"of b64 / volume / url is allowed",
        )
    if "url" in present and not allow_urls:
        raise WorkerError(
            BAD_REQUEST,
            f"media['{label}'] uses a URL, but URL inputs are disabled on this endpoint",
            details=["set ALLOW_URL_INPUTS=1 to enable them, or send b64/volume instead"],
        )
    value = entry.get("volume")
    if value not in (None, "") and ".." in str(value).split("|", 1)[0].split("/"):
        raise WorkerError(BAD_REQUEST, f"media['{label}'].volume may not contain '..'")
    return dict(entry)


def _media_count(media: Mapping[str, Any], key: str) -> int:
    value = media.get(key)
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return len(value)
    return 1


# ---------------------------------------------------------------------------
# Cross-field rules
# ---------------------------------------------------------------------------

#: ``letter in <setting>`` -> the media slot WanGP then demands. Every entry is a
#: literal transcription of a ``return err(...)`` in ``validate_settings``. The
#: fifth field is a letter that must ALSO be present for the rule to apply,
#: because upstream nests the check: ``"+" -> video_guide2`` lives inside
#: ``if "V" in video_prompt_type:`` (wgp.py:1341/1347), so a "+" without a "V"
#: is never reached there and must not be rejected here either.
_REQUIRED_MEDIA: tuple[tuple[str, str, str, str, str], ...] = (
    ("image_prompt_type", "S", "image_start", "wgp.py:1409", ""),
    ("image_prompt_type", "E", "image_end", "wgp.py:1425", ""),
    ("image_prompt_type", "V", "video_source", "wgp.py:1294-1295", ""),
    ("video_prompt_type", "I", "image_refs", "wgp.py:1330-1331", ""),
    ("video_prompt_type", "V", "video_guide", "wgp.py:1341-1346", ""),
    ("video_prompt_type", "+", "video_guide2", "wgp.py:1347-1348", "V"),
    ("audio_prompt_type", "A", "audio_guide", "wgp.py:1302-1304", ""),
    ("audio_prompt_type", "B", "audio_guide2", "wgp.py:1310-1312", ""),
)


def check_cross_variant(
    model_type: str,
    settings: Mapping[str, Any],
    media: Mapping[str, Any],
    *,
    model_def: Mapping[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> None:
    """Reject combinations WanGP would reject anyway -- before the GPU spends
    2-5 minutes loading a model to say so.

    Mirrors ``MinimaxH3Handler.validate_generative_settings``
    (``minimax_h3_handler.py:345-445``) and the letter/attachment rules in
    ``validate_settings`` (``wgp.py:1285-1440``). The duration rules (each
    reference video >= 2 s and truncated to 15 s, total <= 15 s; each audio
    2-15 s, total <= 15 s) are deliberately NOT replicated: they need
    ffprobe/librosa on the real files. WanGP enforces them and they surface as
    ``wangp_validation`` within seconds of submit.
    """
    md: Mapping[str, Any] = model_def or fallback_model_def(model_type)
    warn = warnings if warnings is not None else []
    legal = letters_allowed(model_type, md)
    variant = "Ref2VA" if model_type in REF2VA_TYPES else "FL2VA"

    for key in ("image_prompt_type", "video_prompt_type", "audio_prompt_type"):
        value = str(settings.get(key) or "")
        illegal = sorted(set(value) - set(legal[key]))
        if illegal:
            raise WorkerError(
                INVALID_SETTING,
                f"{key} uses letters {illegal} not supported by {variant} "
                f"(allowed: {sorted(set(legal[key]))})",
            )

    ipt = str(settings.get("image_prompt_type") or "")
    vpt = str(settings.get("video_prompt_type") or "")
    apt = str(settings.get("audio_prompt_type") or "")
    n_refs = _media_count(media, "image_refs")

    if model_type in FL2VA_TYPES:
        for key in ("video_guide2", "audio_guide2"):
            if media.get(key):
                raise WorkerError(
                    BAD_REQUEST,
                    "video_guide2/audio_guide2 are Ref2VA-only",
                    details=[f"'{key}' was supplied for {model_type}"],
                )
        # minimax_h3_handler.py:355-359
        if "F" in vpt:
            n_pos = len(str(settings.get("frames_positions") or "").replace(",", " ").split())
            if n_pos != n_refs:
                raise WorkerError(
                    INVALID_SETTING,
                    "frame injection requires one frames_positions entry per image_refs entry "
                    f"({n_pos} positions, {n_refs} images)",
                )
        # minimax_h3_handler.py:360-368
        if "2" in apt and ("A" in apt or "K" in apt):
            raise WorkerError(
                INVALID_SETTING,
                "audio_prompt_type '2' (generate audio from the control video) cannot combine "
                "with 'A' or 'K'",
            )
        for letter in ("2", "K"):
            if letter in apt and not ("G" in vpt and "V" in vpt and media.get("video_guide")):
                raise WorkerError(
                    INVALID_SETTING,
                    f"audio_prompt_type '{letter}' requires video_prompt_type 'GV' and a "
                    f"video_guide file",
                )
    else:
        # minimax_h3_handler.py:376-445, in source order.
        n_vid = (1 if "V" in vpt else 0) + (1 if "+" in vpt else 0)
        soundtrack = "K" in apt
        if n_refs > 9:
            raise WorkerError(INVALID_SETTING, "Ref2VA accepts at most 9 reference images")
        if n_vid > 2:
            raise WorkerError(INVALID_SETTING, "Ref2VA accepts at most 2 reference videos")
        if soundtrack:
            if n_vid == 0:
                raise WorkerError(
                    INVALID_SETTING,
                    "audio_prompt_type 'K' (use the reference-video soundtracks) requires at "
                    "least one reference video",
                )
            n_aud = n_vid
        else:
            n_aud = (1 if "A" in apt else 0) + (1 if "B" in apt else 0)
        if n_aud > 2:
            raise WorkerError(INVALID_SETTING, "Ref2VA accepts at most 2 audio references")
        visual = n_refs + n_vid
        if n_aud > visual:
            raise WorkerError(
                INVALID_SETTING,
                "Ref2VA needs at least as many reference images+videos as audio references "
                f"({visual} visual, {n_aud} audio)",
            )
        if visual + (0 if soundtrack else n_aud) > 12:
            raise WorkerError(INVALID_SETTING, "Ref2VA accepts at most 12 reference files")

    # wgp.py:1336 -- one_image_ref_only bites only when "I" is used without K/F.
    if (
        md.get("one_image_ref_only")
        and "I" in vpt
        and (md.get("one_image_ref_only_with_background") or not set("KF") & set(vpt))
        and n_refs > 1
    ):
        raise WorkerError(
            INVALID_SETTING,
            f"only one reference image is supported by this model mode ({n_refs} supplied)",
            details=["use video_prompt_type 'KFI' (frame injection) for multiple images"],
        )

    _check_media_requirements(model_type, settings, media, model_def=md, warnings=warn)


#: Spec-sketch spelling.
_check_cross_variant = check_cross_variant


def _check_media_requirements(
    model_type: str,
    settings: Mapping[str, Any],
    media: Mapping[str, Any],
    *,
    model_def: Mapping[str, Any],
    warnings: list[str],
) -> None:
    """Every ``You must provide a ...`` in ``validate_settings``, pre-flighted;
    plus a warning for each attachment WanGP would silently drop."""
    values = {
        "image_prompt_type": str(settings.get("image_prompt_type") or ""),
        "video_prompt_type": str(settings.get("video_prompt_type") or ""),
        "audio_prompt_type": str(settings.get("audio_prompt_type") or ""),
    }
    legal = {
        key: set(letters)
        for key, letters in letters_allowed(model_type, model_def).items()
    }

    for setting_key, letter, media_key, citation, prereq in _REQUIRED_MEDIA:
        if letter not in legal[setting_key]:
            continue  # this variant has no such mode at all
        if prereq and prereq not in values[setting_key]:
            continue  # upstream nests this check under `prereq`; so do we
        wanted = letter in values[setting_key]
        supplied = _media_count(media, media_key) > 0
        if wanted and not supplied:
            raise WorkerError(
                INVALID_SETTING,
                f"{setting_key} contains '{letter}', so media.{media_key} is required "
                f"({citation})",
            )
        if supplied and not wanted:
            warnings.append(
                f"media.{media_key} will be ignored: {setting_key} does not contain "
                f"'{letter}' ({citation})"
            )

    # wgp.py:1350-1358 -- masked inpainting needs a mask; "U" opts out. Nested
    # under "V" upstream (wgp.py:1341), so the mask is only demanded when a
    # control video is in play.
    vpt = values["video_prompt_type"]
    masked = "V" in vpt and "A" in vpt and "U" not in vpt
    if "A" in legal["video_prompt_type"]:
        if masked and not media.get("video_mask"):
            raise WorkerError(
                INVALID_SETTING,
                "video_prompt_type contains 'A', so media.video_mask is required "
                "(wgp.py:1350-1356)",
            )
        if media.get("video_mask") and not masked:
            warnings.append(
                "media.video_mask will be ignored: video_prompt_type does not select "
                "masked processing (needs 'V' and 'A', without 'U') (wgp.py:1350-1358)"
            )
    elif media.get("video_mask"):
        warnings.append(
            "media.video_mask will be ignored: this model has no masked-inpainting mode"
        )

    # wgp.py:1387-1394 -- image_guide/image_mask are image-output slots only, and
    # image_mode is pinned to 0 for a video endpoint.
    for key in ("image_guide", "image_mask"):
        if media.get(key):
            warnings.append(f"media.{key} will be ignored: it only applies to image output modes")
    # wgp.py:1286-1291 -- custom_guide is nulled unless model_def declares one.
    if media.get("custom_guide") and not model_def.get("custom_guide"):
        warnings.append("media.custom_guide will be ignored: this model declares no custom guide")
    if media.get("audio_source") and not str(settings.get("audio_prompt_type") or ""):
        warnings.append(
            "media.audio_source is only used by post-processing modes, not by generation"
        )


# ---------------------------------------------------------------------------
# Model schema resolution
# ---------------------------------------------------------------------------


def _resolve_model_schema(
    model_type: str,
    *,
    allowed_settings: Any,
    model_def: Mapping[str, Any] | None,
    session: Any,
) -> tuple[dict[str, Any], set[str], dict[str, Any]]:
    """``(default_settings, allow_listed_names, model_def)``.

    ``allowed_settings`` may be the model's default-settings mapping (the usual
    case -- ``session.get_default_settings(model_type)``) or just an iterable of
    key names. ``session`` is duck-typed: anything exposing
    ``get_model_schema`` / ``get_default_settings`` / ``get_model_def`` works,
    which keeps this module free of any WanGP import.
    """
    defaults: dict[str, Any] = {}
    names: set[str] | None = None

    if isinstance(allowed_settings, Mapping):
        defaults = copy.deepcopy(dict(allowed_settings))
        names = set(defaults)
    elif allowed_settings is not None:
        names = {str(key) for key in allowed_settings}

    mdef: dict[str, Any] = dict(model_def) if model_def else {}

    if session is not None and (not defaults or not mdef):
        schema = _session_schema(session, model_type)
        if not defaults and isinstance(schema.get("default_settings"), Mapping):
            defaults = copy.deepcopy(dict(schema["default_settings"]))
            if names is None:
                names = set(defaults)
        if not mdef and isinstance(schema.get("model_def"), Mapping):
            mdef = dict(schema["model_def"])

    # Fill anything still missing from the source-derived fallback so a caller
    # who passes nothing but a model_type is still validated against the real
    # frame lattice and the real letter whitelists.
    merged = fallback_model_def(model_type)
    merged.update(mdef)
    return defaults, (names if names is not None else set(defaults)), merged


def _session_schema(session: Any, model_type: str) -> dict[str, Any]:
    getter = getattr(session, "get_model_schema", None)
    try:
        schema = getter(model_type) if callable(getter) else None
        if schema is None and callable(getattr(session, "get_default_settings", None)):
            schema = {
                "default_settings": session.get_default_settings(model_type),
                "model_def": (
                    session.get_model_def(model_type)
                    if callable(getattr(session, "get_model_def", None))
                    else {}
                ),
            }
    except WorkerError:
        raise
    except Exception as exc:  # noqa: BLE001 - any backend failure is a bad model_type
        raise WorkerError(
            BAD_REQUEST,
            f"unknown or unusable model_type '{model_type}': {exc}",
            cause=exc,
        ) from exc
    if not isinstance(schema, Mapping):
        raise WorkerError(BAD_REQUEST, f"unknown model_type '{model_type}'")
    return dict(schema)


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------


def parse(
    job_input: Any,
    *,
    model_type: str | None = None,
    allowed_settings: Any = None,
    model_def: Mapping[str, Any] | None = None,
    cfg: Any = None,
    session: Any = None,
    profile_loader: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    extra_allowed: Iterable[str] = (),
) -> Request:
    """Validate one job payload and assemble the settings dict WanGP will run.

    ``job_input`` is the ``input`` object of a RunPod job.

    Merge order (later wins):
        ``allowed_settings`` (= ``get_default_settings(model_type)``)
        -> accelerator-profile fragment (``input.profile``)
        -> ``input.settings``
        -> ``input.prompt`` (only if ``settings.prompt`` was absent)
        -> worker pins (``model_type``, ``config``, resolved ``seed``,
           ``batch_size=1``, ``repeat_generation=1``, ``image_mode=0``)

    ``input.media`` is validated but NOT materialized -- media_in replaces those
    specs with absolute paths and merges them into ``settings`` afterwards.

    Raises :class:`errors.WorkerError` with ``bad_request`` / ``unknown_setting``
    / ``invalid_setting`` and never anything else.
    """
    if not isinstance(job_input, Mapping):
        raise WorkerError(BAD_REQUEST, "input must be a JSON object")
    payload: Mapping[str, Any] = job_input
    cfg = cfg if cfg is not None else _default_cfg()
    warnings: list[str] = []

    # ---- model_type -------------------------------------------------------
    pinned = str(
        model_type
        or getattr(cfg, "model_type", "")
        or os.environ.get("WANGP_MODEL_TYPE")
        or DEFAULT_MODEL_TYPE
    ).strip()
    requested = payload.get("model_type")
    mt = str(requested).strip() if requested not in (None, "") else pinned
    if mt not in MINIMAX_H3_TYPES:
        raise WorkerError(
            BAD_REQUEST,
            f"model_type must be one of {sorted(MINIMAX_H3_TYPES)}",
            details=[f"got {mt!r}"],
        )
    if mt != pinned and not _flag(cfg, "allow_model_switch", "ALLOW_MODEL_SWITCH", "0"):
        raise WorkerError(
            BAD_REQUEST,
            f"this endpoint is pinned to '{pinned}'; a switch to '{mt}' costs a full "
            f"release_model()+reload (wgp.py:6773)",
            details=["set ALLOW_MODEL_SWITCH=1 to permit it"],
        )

    defaults, allow_names, mdef = _resolve_model_schema(
        mt, allowed_settings=allowed_settings, model_def=model_def, session=session
    )

    # ---- settings allow-list ---------------------------------------------
    user = payload.get("settings")
    if user is None:
        user = {}
    if not isinstance(user, Mapping):
        raise WorkerError(BAD_REQUEST, "input.settings must be an object")
    user = dict(user)

    bad = sorted(set(user) & FORBIDDEN_KEYS)
    if bad:
        raise WorkerError(
            BAD_REQUEST,
            f"settings may not contain {bad}; media goes in input.media",
            details=[
                "attachment keys, `mode`, `_api`, `client_id`, `state`, `type`, "
                "`base_model_type` and `priority` are worker-controlled"
            ],
        )
    universe = (
        set(PRIMARY_SETTINGS)
        | set(allow_names)
        | set(defaults)
        | {str(key) for key in extra_allowed}
    )
    unknown = sorted(set(user) - universe)
    if unknown:
        raise WorkerError(
            UNKNOWN_SETTING,
            f"unknown settings for '{mt}': {unknown}",
            details=[f"{len(universe)} keys are accepted; see models/_settings.json"],
        )

    media = _validate_media(payload.get("media"), cfg)

    # ---- merge ------------------------------------------------------------
    settings: dict[str, Any] = copy.deepcopy(defaults)

    profile_name = payload.get("profile")
    profile_loras: set[str] = set()
    profile_lora_entries: set[str] = set()
    if profile_name not in (None, ""):
        loader = profile_loader
        fragment = (
            dict(loader(str(profile_name), mdef))
            if callable(loader)
            else load_profile_fragment(str(profile_name), model_def=mdef)
        )
        forbidden_in_profile = sorted(set(fragment) & FORBIDDEN_KEYS)
        if forbidden_in_profile:
            raise WorkerError(
                BAD_REQUEST,
                f"accelerator profile '{profile_name}' sets worker-controlled keys "
                f"{forbidden_in_profile}",
            )
        for entry in fragment.get("activated_loras") or []:
            profile_loras.add(os.path.basename(str(entry).split("|")[0]))
            # The FULL entry too, verbatim: the shipped MiniMax H3 turbo profiles
            # name their LoRA by https:// URL, which get_lora_local_path
            # (wgp.py:3670-3677) maps to <lora_dir>/<basename(url)> so a staged
            # file resolves with zero network. Only an exact match is honoured --
            # a basename match would let a caller point the same filename at their
            # own host (check_loras_exist downloads when the local file is absent,
            # wgp.py:3697-3706).
            profile_lora_entries.add(str(entry))
        settings.update(fragment)
        warnings.append(f"accelerator profile '{profile_name}' applied before input.settings")

    settings.update(user)

    # ``prompt`` is a convenience alias. NOTE the spec sketch used
    # ``settings.setdefault("prompt", payload["prompt"])`` here, which can never
    # fire: get_default_settings ALWAYS carries a prompt (wgp.py:3157), so the
    # alias would be silently dropped and the model's demo prompt used instead.
    if payload.get("prompt") not in (None, "") and "prompt" not in user:
        settings["prompt"] = payload["prompt"]

    settings["model_type"] = mt

    prompt = settings.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise WorkerError(BAD_REQUEST, "prompt is required")
    if "prompt" not in user and payload.get("prompt") in (None, ""):
        warnings.append(
            "no prompt was supplied; the model's built-in demo prompt is being used"
        )

    _apply_config_selection(settings, user, cfg, mdef)
    _validate_frames(settings, user, mdef, cfg, warnings)
    _validate_resolution(settings, mdef)
    _validate_solver_and_steps(settings, user, mdef, warnings)
    _validate_cache(settings, mdef)
    _validate_loras(settings, mdef, cfg, profile_loras,
                    profile_entries=profile_lora_entries, warnings=warnings)
    _note_inert_settings(settings, user, mdef, warnings)

    settings["seed"] = resolve_seed(settings.get("seed"), warnings=warnings)
    # One video per job: the response schema, the idempotency key and the
    # transport chain all assume exactly one output file.
    settings["batch_size"] = 1
    settings["repeat_generation"] = 1
    if _as_int(settings.get("image_mode", 0) or 0, "settings.image_mode") != 0:
        raise WorkerError(
            INVALID_SETTING,
            "image_mode must be 0: this endpoint produces video, and a non-zero image_mode "
            "makes WanGP emit images that the video transport cannot deliver",
        )
    settings["image_mode"] = 0

    check_cross_variant(mt, settings, media, model_def=mdef, warnings=warnings)

    # ---- runtime / output -------------------------------------------------
    runtime = payload.get("runtime") or {}
    if not isinstance(runtime, Mapping):
        raise WorkerError(BAD_REQUEST, "input.runtime must be an object")
    runtime = dict(runtime)
    output = _validate_output(payload.get("output"))

    req = Request()
    req.model_type = mt
    req.settings = settings
    req.media = media
    req.output = output
    req.runtime = runtime
    req.profile = str(profile_name) if profile_name not in (None, "") else None
    req.budget_s = _resolve_budget(runtime.get("timeout_s"), cfg, warnings)
    req.priority = _resolve_priority(runtime.get("priority"), warnings)
    req.idempotency_key = _resolve_idempotency_key(runtime.get("idempotency_key"))
    req.warnings = warnings
    req.resolved = {key: settings[key] for key in RESOLVED_ECHO_KEYS if key in settings}
    return req


# ---------------------------------------------------------------------------
# Individual settings validators
# ---------------------------------------------------------------------------

#: ``update_default_settings`` writes this (``minimax_h3_handler.py:514``); used
#: only when the caller supplied no defaults at all.
_DEFAULT_VIDEO_LENGTH = 124
#: ``wgp.py:3159`` -- "1280x720" if the model name says 720, else "832x480".
_DEFAULT_RESOLUTION = "832x480"


def _apply_config_selection(
    settings: dict[str, Any], user: Mapping[str, Any], cfg: Any, mdef: Mapping[str, Any]
) -> None:
    """Pin the ``config`` selection string (text encoder / VAE / DiT priority).

    ``wgp.py:6773`` reloads the whole model when ``config`` differs from the
    loaded one, so a per-request config change costs a full
    ``release_model()`` + reload -- the same price as a model switch, and gated
    the same way.
    """
    endpoint = str(getattr(cfg, "model_config", "") or "").rstrip(",")
    supplied = user.get("config")

    if supplied is None:
        current = str(settings.get("config") or "")
        chosen = endpoint or current
    else:
        if not isinstance(supplied, str):
            raise WorkerError(INVALID_SETTING, "settings.config must be a string")
        chosen = supplied
        # Validate the ids BEFORE comparing against the pin, so a typo reports
        # "not defined in system_configs" rather than "endpoint is pinned".
        _validate_config_selection(_normalize_config_selection(chosen), mdef)
        if (
            endpoint
            and _normalize_config_selection(chosen) != _normalize_config_selection(endpoint)
            and not _flag(cfg, "allow_model_switch", "ALLOW_MODEL_SWITCH", "0")
        ):
            raise WorkerError(
                BAD_REQUEST,
                f"this endpoint is pinned to config '{endpoint}'; changing it forces a full "
                f"model reload (wgp.py:6773)",
                details=["set ALLOW_MODEL_SWITCH=1 to permit it"],
            )

    normalized = _normalize_config_selection(chosen)
    _validate_config_selection(normalized, mdef)
    if normalized or "config" in settings or endpoint:
        settings["config"] = normalized


#: ``WANGP_MAX_FRAMES`` when it is unset or unusable. 362 frames is ~15.1 s at
#: 24 fps -- the longest clip either MiniMax H3 variant is documented for.
DEFAULT_MAX_FRAMES = 362


def _worker_frame_cap(cfg: Any) -> int:
    """The endpoint's ``video_length`` ceiling, never 0.

    ``0`` is NOT "unlimited": FL2VA declares no frame maximum anywhere in
    the headless path, so an uncapped endpoint lets one request schedule
    hundreds of sliding windows on a billed GPU. A non-positive or unparseable
    value therefore falls back to :data:`DEFAULT_MAX_FRAMES` rather than
    collapsing the cap to ``frames_minimum`` (which is what
    ``int(cfg.max_frames or os.environ[...])`` used to do for ``0``: falsy ->
    env -> ``"0"`` -> a 107-frame endpoint that rejects the model's own default
    ``video_length``).
    """
    raw = getattr(cfg, "max_frames", None)
    if raw in (None, ""):
        raw = os.environ.get("WANGP_MAX_FRAMES", DEFAULT_MAX_FRAMES)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_FRAMES
    return value if value > 0 else DEFAULT_MAX_FRAMES


def _validate_frames(
    settings: dict[str, Any],
    user: Mapping[str, Any],
    mdef: Mapping[str, Any],
    cfg: Any,
    warnings: list[str],
) -> None:
    """Put ``video_length`` on the model's frame lattice and under the cap.

    The lattice is ``frames_minimum`` / ``frames_steps`` / ``frames_offset``
    (107 / 17 / 5 for MiniMax H3, ``minimax_h3_handler.py:185-187``), so the
    legal values are 107, 124, 141, 158, ... WanGP floors onto it silently at
    ``wgp.py:6929``; we do it here so the ``resolved`` block we return is the
    truth rather than the request.

    ``frames_selection_maximum`` exists ONLY for Ref2VA (=737,
    ``minimax_h3_handler.py:258``) and, since upstream 238e25f, bounds only the
    gradio slider (``wgp.py:11897``). No variant has an upper bound in the
    headless path -- ``validate_settings`` never caps it -- so the worker MUST,
    or one request can schedule hundreds of sliding windows on a billed GPU.
    """
    minimum, step, offset = frame_lattice(mdef)
    fps = float(mdef.get("fps", 24) or 24)
    worker_max = _worker_frame_cap(cfg)
    model_max = int(mdef.get("frames_selection_maximum") or 0)
    hard_max = min(model_max, worker_max) if model_max > 0 else worker_max
    cap = floor_frames(max(minimum, hard_max), minimum, step, offset)

    raw = settings.get("video_length", _DEFAULT_VIDEO_LENGTH)
    requested = _as_int(raw, "settings.video_length")
    if requested < 1:
        raise WorkerError(INVALID_SETTING, "video_length must be positive")
    value = floor_frames(requested, minimum, step, offset)
    if value > cap:
        raise WorkerError(
            INVALID_SETTING,
            f"video_length={requested} exceeds this endpoint's cap of {cap} frames "
            f"({cap / fps:.1f}s at {fps:g} fps)",
            details=[
                f"legal values are {minimum} then every +{step} frames ({offset} mod {step})",
                "raise WANGP_MAX_FRAMES to allow longer clips"
                + (f"; the model's own maximum is {hard_max}"
                   if mdef.get("frames_selection_maximum") else ""),
            ],
        )
    if value != requested:
        warnings.append(
            f"video_length {requested} -> {value} (must be >= {minimum} and = {offset} "
            f"mod {step}; WanGP floors at wgp.py:6929)"
        )
    settings["video_length"] = value

    # Sliding windows share the frame quantum but carry their own bounds
    # (minimax_h3_handler.py:249-250 / :307-308): window 124..481, overlap
    # rounded to nearest on the same 17/1 lattice, max 120.
    swd = mdef.get("sliding_window_defaults") or {}
    if "sliding_window_size" in settings and swd:
        requested_window = _as_int(settings["sliding_window_size"], "settings.sliding_window_size")
        if requested_window <= 0:
            # 0 does NOT disable sliding windows. wgp.py:6930 only skips the
            # FLOORING for 0; whether windows run is decided by
            # test_any_sliding_window(model_type), which reads
            # model_def["sliding_window"] -- True for both MiniMax H3 variants
            # (minimax_h3_handler.py:249, :304). With size 0 upstream computes
            # default_reuse_frames = min(0 - latent_size, overlap) = -17
            # (wgp.py:7158), sliding_window = video_length > 0 -> True
            # (wgp.py:7164), and then
            # compute_sliding_window_no(124, 0, 0, -17) = 9 windows of
            # current_video_length = 0 (wgp.py:7196-7197). Nine zero-length
            # windows on a billed GPU.
            raise WorkerError(
                INVALID_SETTING,
                "sliding_window_size=0 does not disable sliding windows; this model "
                "always uses them (model_def['sliding_window'] is true)",
                details=[
                    f"legal window sizes are {int(swd.get('window_min', minimum))}.."
                    f"{int(swd.get('window_max', cap))} on the {offset} mod {step} lattice",
                    "to generate a single window, set video_length <= sliding_window_size",
                ],
            )
        window = floor_frames(requested_window, minimum, step, offset)
        window = max(
            int(swd.get("window_min", minimum)),
            min(window, int(swd.get("window_max", cap))),
        )
        if window != requested_window:
            warnings.append(f"sliding_window_size {requested_window} -> {window}")
        settings["sliding_window_size"] = window

    if "sliding_window_overlap" in settings and swd:
        requested_overlap = _as_int(
            settings["sliding_window_overlap"], "settings.sliding_window_overlap"
        )
        if requested_overlap < 0:
            raise WorkerError(
                INVALID_SETTING, "sliding_window_overlap must be 0 or a positive frame count"
            )
        overlap = round_overlap(
            requested_overlap,
            int(swd.get("overlap_step", 17)),
            int(swd.get("overlap_offset", 1)),
        )
        if overlap:
            overlap = max(int(swd.get("overlap_min", 1)), min(overlap, int(swd.get("overlap_max", 120))))
        if overlap != requested_overlap:
            warnings.append(f"sliding_window_overlap {requested_overlap} -> {overlap}")
        settings["sliding_window_overlap"] = overlap


def _validate_resolution(settings: dict[str, Any], mdef: Mapping[str, Any]) -> None:
    """``WxH``, both multiples of the VAE block size.

    WanGP floors both dimensions itself (``wgp.py:6760``:
    ``int(width) // block_size * block_size`` with
    ``block_size = model_def["vae_block_size"]`` = 32), i.e. it would silently
    give you a different frame size than you asked for. Rejecting is the honest
    alternative: the caller can compute the nearest legal pair from the message.
    """
    raw = settings.get("resolution")
    if raw in (None, ""):
        settings["resolution"] = _DEFAULT_RESOLUTION
        return
    text = str(raw).strip().lower().replace("*", "x")
    match = re.match(r"^(\d{2,5})\s*x\s*(\d{2,5})$", text)
    if not match:
        raise WorkerError(
            INVALID_SETTING,
            f"resolution '{raw}' must look like '832x480'",
        )
    width, height = int(match.group(1)), int(match.group(2))
    block = int(mdef.get("vae_block_size") or mdef.get("block_size") or 0)
    if block:
        new_width, new_height = (width // block) * block, (height // block) * block
        if new_width <= 0 or new_height <= 0:
            raise WorkerError(
                INVALID_SETTING,
                f"resolution '{raw}' is smaller than one {block}px block",
            )
        if (new_width, new_height) != (width, height):
            raise WorkerError(
                INVALID_SETTING,
                f"resolution '{raw}' is not a multiple of block_size={block}",
                details=[f"nearest valid: {new_width}x{new_height}"],
            )
    settings["resolution"] = f"{width}x{height}"


def _validate_solver_and_steps(
    settings: dict[str, Any],
    user: Mapping[str, Any],
    mdef: Mapping[str, Any],
    warnings: list[str],
) -> None:
    """``sample_solver`` against ``model_def["sample_solvers"]`` and a hard step cap.

    ``sample_solvers`` is ``[("Euler", "euler"), ("RES Multistep", "res_multistep")]``
    (``minimax_h3_handler.py:200``); the second element of each pair is the wire
    value. An unknown solver is not rejected upstream -- it silently falls back --
    so a typo would quietly cost a different result.
    """
    solvers: list[str] = []
    for entry in mdef.get("sample_solvers") or []:
        if isinstance(entry, (list, tuple)) and len(entry) > 1:
            solvers.append(str(entry[1]))
        elif isinstance(entry, str):
            solvers.append(entry)
    solver = settings.get("sample_solver")
    if solvers and solver not in (None, ""):
        if str(solver) not in solvers:
            raise WorkerError(
                INVALID_SETTING,
                f"sample_solver '{solver}' is not supported by this model",
                details=[f"supported: {solvers}"],
            )

    max_steps = int(os.environ.get("WANGP_MAX_STEPS", "100"))
    steps = _as_int(settings.get("num_inference_steps", 20), "settings.num_inference_steps")
    if steps < 1:
        raise WorkerError(INVALID_SETTING, "num_inference_steps must be at least 1")
    if steps > max_steps:
        raise WorkerError(
            INVALID_SETTING,
            f"num_inference_steps={steps} exceeds this endpoint's cap of {max_steps}",
            details=["raise WANGP_MAX_STEPS to allow more"],
        )
    settings["num_inference_steps"] = steps

    if "flow_shift" in settings and settings["flow_shift"] is not None:
        settings["flow_shift"] = _as_float(settings["flow_shift"], "settings.flow_shift")
    if "denoising_strength" in settings and settings["denoising_strength"] is not None:
        strength = _as_float(settings["denoising_strength"], "settings.denoising_strength")
        if not 0.0 <= strength <= 1.0:
            raise WorkerError(INVALID_SETTING, "denoising_strength must be between 0 and 1")
        settings["denoising_strength"] = strength


def _validate_cache(settings: dict[str, Any], mdef: Mapping[str, Any]) -> None:
    """Step-skipping type and threshold -- mirrors ``wgp.py:1208-1216``."""
    supported = {""}
    for flag, name in (
        ("tea_cache", "tea"),
        ("mag_cache", "mag"),
        ("spectrum_cache", "spectrum"),
        ("first_block_cache", "first_block"),
    ):
        if mdef.get(flag):
            supported.add(name)
    cache_type = settings.get("skip_steps_cache_type")
    cache_type = "" if cache_type is None else str(cache_type)
    if cache_type not in supported:
        raise WorkerError(
            INVALID_SETTING,
            f"this model does not support step-skipping type '{cache_type}'",
            details=[f"supported: {sorted(supported)}"],
        )
    if cache_type == "first_block":
        allowed = [float(value) for value in (mdef.get("first_block_cache_thresholds") or ())]
        multiplier = _as_float(
            settings.get("skip_steps_multiplier", 0), "settings.skip_steps_multiplier"
        )
        if allowed and multiplier not in allowed:
            raise WorkerError(
                INVALID_SETTING,
                f"skip_steps_multiplier must be one of {allowed} for the first_block cache "
                f"(wgp.py:1215)",
                details=[f"got {multiplier}"],
            )
    if "skip_steps_start_step_perc" in settings:
        percent = _as_float(
            settings["skip_steps_start_step_perc"], "settings.skip_steps_start_step_perc"
        )
        if not 0.0 <= percent <= 100.0:
            raise WorkerError(
                INVALID_SETTING, "skip_steps_start_step_perc must be between 0 and 100"
            )


def _validate_loras(
    settings: dict[str, Any],
    mdef: Mapping[str, Any],
    cfg: Any,
    profile_loras: set[str],
    *,
    profile_entries: set[str] | None = None,
    warnings: list[str] | None = None,
) -> None:
    """Only basenames that are staged on this endpoint.

    ``get_lora_local_path`` (``wgp.py:3670-3677``) returns the entry verbatim
    when ``os.path.isabs(lora)``, maps an ``https://`` entry to
    ``lora_dir/basename(url)``, and otherwise joins the entry onto ``lora_dir``
    as a relative path. So an absolute path is an arbitrary-file primitive, a
    URL is an arbitrary-download primitive, and ``..`` escapes the LoRA
    directory. Allow-list by basename instead.

    LoRAs contributed by a baked accelerator profile are exempt from the
    allow-list: they ship inside the image and are staged by the same build that
    wrote ``WANGP_ALLOWED_LORAS``.
    """
    entries = settings.get("activated_loras")
    if entries in (None, ""):
        return
    if isinstance(entries, str):
        raise WorkerError(
            INVALID_SETTING, "activated_loras must be a list of LoRA names, not a string"
        )
    if not isinstance(entries, (list, tuple)):
        raise WorkerError(INVALID_SETTING, "activated_loras must be a list")

    allowed = {str(name) for name in (getattr(cfg, "allowed_loras", None) or ())}
    if not allowed:
        allowed = {
            chunk.strip()
            for chunk in os.environ.get("WANGP_ALLOWED_LORAS", "").split(",")
            if chunk.strip()
        }

    for entry in entries:
        text = str(entry)
        head = text.split("|", 1)[0]
        if os.path.isabs(head) or head.startswith("\\\\") or (len(head) > 1 and head[1] == ":"):
            raise WorkerError(BAD_REQUEST, "absolute LoRA paths are not allowed")
        lowered = head.lower()
        if ("://" in lowered or lowered.startswith(("http:", "https:"))) and \
                text not in (profile_entries or set()):
            # NOT a naming quibble: an http(s) entry is a caller-steered fetch
            # primitive. validate_settings -> update_loras_url_cache
            # (wgp.py:1181-1182, :9980-9996) files the URL in loras_url_cache,
            # then check_loras_exist(..., download=True) (wgp.py:6901, :3689-3706)
            # calls download_file -> urlretrieve (shared/utils/download.py:241)
            # with no scheme, IP, port or redirect validation -- from inside the
            # RunPod network, bypassing every control media_in.check_url_target
            # implements, and writing attacker-chosen bytes to
            # <lora_dir>/<basename(url)> on the SHARED network volume, where the
            # next job loads them as model weights. The worker never needs a
            # remote LoRA: scripts/prefetch_weights.py stages them.
            raise WorkerError(
                BAD_REQUEST,
                f"remote LoRA URLs are not accepted: {head!r}",
                details=["stage the LoRA on the volume and name it by basename",
                         "see runpod_worker/scripts/prefetch_weights.py",
                         "the only exception is a URL contributed verbatim by a "
                         "shipped accelerator profile (input.profile)"],
            )
        if "://" in lowered:
            # A profile URL, i.e. repo-controlled. Still worth saying out loud:
            # if the file is not staged, check_loras_exist downloads it inside
            # the billed generation (wgp.py:3697-3706).
            if warnings is not None:
                warnings.append(
                    f"LoRA '{os.path.basename(head)}' is named by URL; it must be "
                    f"staged in the loras directory or WanGP will download it "
                    f"during the generation"
                )
        if ".." in head.replace("\\", "/").split("/"):
            raise WorkerError(BAD_REQUEST, "LoRA paths may not contain '..'")
        name = os.path.basename(head)
        if not name:
            raise WorkerError(BAD_REQUEST, f"LoRA entry {text!r} names no file")
        if name in profile_loras:
            continue
        # An EMPTY allow-list means "no caller-supplied LoRAs", not "any LoRA".
        # The permissive reading made the shipped default (WANGP_ALLOWED_LORAS
        # unset, per the Dockerfile) accept every name a caller invented.
        if not allowed:
            raise WorkerError(
                BAD_REQUEST,
                "this endpoint accepts no caller-supplied LoRAs",
                details=["set WANGP_ALLOWED_LORAS to the basenames staged on the "
                         "volume, or select a baked accelerator profile with "
                         "input.profile"],
            )
        if name not in allowed:
            raise WorkerError(
                BAD_REQUEST,
                f"LoRA '{name}' is not staged on this endpoint",
                details=[f"allowed: {sorted(allowed)}"],
            )

    multipliers = settings.get("loras_multipliers")
    if multipliers is not None and not isinstance(multipliers, (str, int, float)):
        raise WorkerError(
            INVALID_SETTING,
            "loras_multipliers must be a string such as '1.0' or '1.0 0.8' (wgp.py:1198-1200)",
        )


def _note_inert_settings(
    settings: dict[str, Any],
    user: Mapping[str, Any],
    mdef: Mapping[str, Any],
    warnings: list[str],
) -> None:
    """Warn about settings this model reads and then ignores. Derived from
    model_def, never hard-coded, so a model change cannot make these lie."""
    if int(mdef.get("guidance_max_phases", 1) or 0) == 0 and "guidance_scale" in user:
        warnings.append(
            "guidance_scale is ignored: guidance_max_phases=0 "
            "(MiniMaxH3Pipeline.generate takes no CFG argument)"
        )
    if mdef.get("no_negative_prompt") and settings.get("negative_prompt"):
        warnings.append("negative_prompt is ignored: model declares no_negative_prompt")
    if mdef.get("keep_frames_video_guide_not_supported") and settings.get(
        "keep_frames_video_guide"
    ):
        warnings.append("keep_frames_video_guide is not supported by this model")


# ---------------------------------------------------------------------------
# runtime / output blocks
# ---------------------------------------------------------------------------


def _validate_output(output: Any) -> dict[str, Any]:
    if output is None:
        return {"mode": "auto"}
    if not isinstance(output, Mapping):
        raise WorkerError(BAD_REQUEST, "input.output must be an object")
    out = dict(output)

    mode = str(out.get("mode") or "auto").strip().lower()
    mode = OUTPUT_MODE_ALIASES.get(mode, mode)
    if mode not in OUTPUT_MODES:
        raise WorkerError(
            BAD_REQUEST,
            f"output.mode must be one of {list(OUTPUT_MODES)}",
            details=[f"got {out.get('mode')!r}"],
        )
    out["mode"] = mode

    url = out.get("presigned_url")
    if url not in (None, ""):
        parts = urlsplit(str(url))
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise WorkerError(
                BAD_REQUEST, "output.presigned_url must be an http(s) URL"
            )
    elif mode == "presigned":
        raise WorkerError(
            BAD_REQUEST, "output.mode='presigned' requires output.presigned_url"
        )

    content_type = out.get("content_type")
    if content_type in (None, ""):
        out["content_type"] = "video/mp4"
    elif not isinstance(content_type, str) or "/" not in content_type:
        raise WorkerError(BAD_REQUEST, "output.content_type must be a MIME type")
    return out


def _resolve_budget(raw: Any, cfg: Any, warnings: list[str]) -> int:
    default_budget = int(getattr(cfg, "default_budget_s", 1400) or 1400)
    max_budget = int(getattr(cfg, "max_budget_s", 2600) or 2600)
    if raw in (None, ""):
        requested = default_budget
    else:
        requested = _as_int(raw, "runtime.timeout_s", code=BAD_REQUEST)
    budget = max(MIN_BUDGET_S, min(requested, max_budget))
    if budget != requested:
        warnings.append(
            f"runtime.timeout_s {requested} -> {budget} "
            f"(endpoint range {MIN_BUDGET_S}..{max_budget}s)"
        )
    return budget


def _resolve_priority(raw: Any, warnings: list[str] | None = None) -> int:
    """Range-check ``runtime.priority``. It is INERT on this worker.

    WanGP only reads a task ``priority`` on the webui-queue path
    (``WanGPSession._ensure_task_client_ids``, ``shared/api.py:686-692``), which
    this worker never uses -- and ``priority`` is in :data:`FORBIDDEN_KEYS` as a
    settings key anyway. At concurrency 1 there is no queue to order, so the
    field is accepted, validated and then ignored; say so rather than let a
    caller believe it did something.
    """
    if raw in (None, ""):
        return 0
    priority = _as_int(raw, "runtime.priority", code=BAD_REQUEST)
    if not 0 <= priority <= 9:
        raise WorkerError(BAD_REQUEST, "runtime.priority must be between 0 and 9")
    if warnings is not None:
        warnings.append(
            "runtime.priority is inert on this endpoint: one generation per worker "
            "means there is no queue to order (scale with max_workers instead)"
        )
    return priority


def _resolve_idempotency_key(raw: Any) -> str | None:
    if raw in (None, ""):
        return None
    if not isinstance(raw, str) or not _IDEMPOTENCY_RE.match(raw):
        raise WorkerError(
            BAD_REQUEST,
            "runtime.idempotency_key must be 1-128 chars of [A-Za-z0-9._:-] starting "
            "alphanumeric",
            details=["it becomes part of the output object key"],
        )
    return raw
