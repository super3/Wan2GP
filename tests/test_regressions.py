"""Focused regression tests for the source fixes shipped in this PR.

Five large per-module unit-test files (``test_frame_scheduler.py``,
``test_filename_formatter.py``, ``test_resolutions.py``, ``test_audio_metadata.py``,
``test_lora_mapper.py``) are deferred to follow-up PRs so this change stays
hand-reviewable, but the source fixes they covered ship here. This file exists so those
fixes stay guarded in the meantime: one small class per fix, each test pinning the fixed
behaviour and failing if the fix is reverted.

These tests are expected to move into their module-specific file once that file lands;
this module is a holding pen, not the permanent home for any of them.

Like the rest of the suite, nothing here imports torch, numpy or gradio.
"""

from __future__ import annotations

import datetime as datetime_module
import gc
import json
import struct
import types
import warnings

import pytest

import shared.match_archi as match_archi
import shared.resolutions as resolutions
import shared.tools.sha256_verify as sha256_verify
import shared.utils.audio_metadata as audio_metadata
import shared.utils.filename_formatter as filename_formatter
import shared.utils.prompt_parser as prompt_parser


# --------------------------------------------------------------------------------------
# Fix 1: FilenameFormatter._parse_date_format substitutes date tokens in a single pass.
# --------------------------------------------------------------------------------------

# 2025-01-15 14:30:45 -> year 2025/25, month 01, day 15, hour 14, minute 30, second 45.
FROZEN_NOW = datetime_module.datetime(2025, 1, 15, 14, 30, 45)


class TestDateTokenSubstitution:
    """Sequential replacement let a token match the strftime code an earlier one wrote."""

    @pytest.fixture(autouse=True)
    def _frozen_clock(self, monkeypatch):
        """Pin the module-level ``time`` and ``datetime`` the formatter reads."""

        monkeypatch.setattr(
            filename_formatter, "time", types.SimpleNamespace(time=lambda: 0.0)
        )
        monkeypatch.setattr(
            filename_formatter,
            "datetime",
            types.SimpleNamespace(fromtimestamp=lambda _timestamp: FROZEN_NOW),
        )

    def test_adjacent_month_and_minute_tokens_both_survive(self):
        # "MM" -> "%m" used to leave an "m" that the later "mm" token matched, so
        # "MMmm" compiled to "%%Mm" and strftime rendered the literal "%Mm".
        assert filename_formatter.FilenameFormatter.format_filename("{date(MMmm)}", {}) == "0130"

    def test_adjacent_year_tokens_both_survive(self):
        # Same failure in the year family: "YYYYYY" compiled to "%%yY" -> "%yY".
        assert filename_formatter.FilenameFormatter.format_filename("{date(YYYYYY)}", {}) == "202525"

    def test_ordinary_separated_format_is_unaffected(self):
        # Control: with separators between the tokens the old code was already correct,
        # so the single-pass rewrite must not change this result.
        assert (
            filename_formatter.FilenameFormatter.format_filename(
                "{date(YYYY-MM-DD_HH-mm-ss)}", {}
            )
            == "2025-01-15_14-30-45"
        )

    @pytest.mark.parametrize(
        "user_format, expected_strftime",
        [
            ("MMmm", "%m%M"),
            ("YYYYYY", "%Y%y"),
            ("DD.MM.YYYY", "%d.%m.%Y"),
        ],
    )
    def test_parse_date_format_emits_expected_strftime_codes(self, user_format, expected_strftime):
        formatter = filename_formatter.FilenameFormatter("{date}")
        assert formatter._parse_date_format(user_format) == expected_strftime


# --------------------------------------------------------------------------------------
# Fix 2: compute_sha256 rejects a non-positive chunk_size.
# --------------------------------------------------------------------------------------

HELLO_WORLD_SHA256 = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


class TestSha256ChunkSize:
    """chunk_size=0 made ``f.read(0)`` end the loop before it started."""

    @pytest.fixture
    def hello_file(self, tmp_path):
        path = tmp_path / "hello.bin"
        path.write_bytes(b"hello world")
        return path

    def test_zero_chunk_size_is_rejected(self, hello_file):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            sha256_verify.compute_sha256(hello_file, chunk_size=0)

    def test_zero_chunk_size_cannot_verify_the_empty_digest(self, hello_file):
        # The real damage: the empty-string digest came back for any file, and passing
        # it as expected_hash then "verified successfully" against arbitrary content.
        with pytest.raises(ValueError) as excinfo:
            sha256_verify.compute_sha256(hello_file, expected_hash=EMPTY_SHA256, chunk_size=0)
        assert "chunk_size must be positive" in str(excinfo.value)

    def test_negative_chunk_size_is_rejected(self, hello_file):
        # -1 reads the whole file and was never wrong; it is refused because
        # "read everything" is not a chunk size.
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            sha256_verify.compute_sha256(hello_file, chunk_size=-1)

    # Smaller than, not a divisor of, and exactly the 11-byte payload.
    @pytest.mark.parametrize("chunk_size", [1, 7, 11])
    def test_positive_chunk_sizes_hash_the_whole_file(self, hello_file, chunk_size):
        assert sha256_verify.compute_sha256(hello_file, chunk_size=chunk_size) == HELLO_WORLD_SHA256
        assert (
            sha256_verify.compute_sha256(
                hello_file, expected_hash=HELLO_WORLD_SHA256, chunk_size=chunk_size
            )
            == HELLO_WORLD_SHA256
        )


# --------------------------------------------------------------------------------------
# Fix 3: audio_metadata reads whole files inside a with-block instead of leaking the handle.
# --------------------------------------------------------------------------------------


def _minimal_wave_bytes() -> bytes:
    """A smallest-possible little-endian RIFF/WAVE file, built byte by byte."""

    # 16-byte PCM 'fmt ' payload: format=1, channels=1, 8000 Hz, 16000 B/s, align 2, 16 bits.
    fmt_payload = struct.pack("<HHIIHH", 1, 1, 8000, 16000, 2, 16)
    data_payload = b"\x00\x00\x01\x00"
    body = b"".join(
        [
            b"WAVE",
            b"fmt ",
            struct.pack("<I", len(fmt_payload)),
            fmt_payload,
            b"data",
            struct.pack("<I", len(data_payload)),
            data_payload,
        ]
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _call_recording_resource_warnings(func, *args, **kwargs):
    """Call ``func`` and return ``(result, resource_warnings)``."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func(*args, **kwargs)
        gc.collect()  # unclosed handles warn from the deallocator
    return result, [entry for entry in caught if issubclass(entry.category, ResourceWarning)]


class TestAudioMetadataFileHandles:
    """``open(path, "rb").read()`` left the handle to be collected later."""

    @pytest.fixture
    def wave_path(self, tmp_path):
        path = tmp_path / "clip.wav"
        path.write_bytes(_minimal_wave_bytes())
        return path

    def test_reading_a_wav_chunk_leaks_no_handle(self, wave_path):
        result, resource_warnings = _call_recording_resource_warnings(
            audio_metadata.read_wav_text_chunk, str(wave_path)
        )
        assert result is None  # no 'json' chunk in a freshly built file
        assert resource_warnings == []

    def test_writing_a_wav_chunk_leaks_no_handle(self, wave_path, tmp_path):
        out_path = tmp_path / "tagged.wav"
        _, resource_warnings = _call_recording_resource_warnings(
            audio_metadata.write_wav_text_chunk, str(wave_path), str(out_path), '{"seed": 1234}'
        )
        assert resource_warnings == []

    def test_round_trip_still_returns_the_stored_text(self, wave_path, tmp_path):
        out_path = tmp_path / "tagged.wav"
        audio_metadata.write_wav_text_chunk(str(wave_path), str(out_path), '{"seed": 1234}')
        assert audio_metadata.read_wav_text_chunk(str(out_path)) == '{"seed": 1234}'

    def test_in_place_save_and_read_leak_no_handle(self, wave_path):
        _, save_warnings = _call_recording_resource_warnings(
            audio_metadata.save_audio_metadata, str(wave_path), {"seed": 1234}
        )
        metadata, read_warnings = _call_recording_resource_warnings(
            audio_metadata.read_audio_metadata, str(wave_path)
        )
        assert metadata == {"seed": 1234}
        assert save_warnings == []
        assert read_warnings == []


# --------------------------------------------------------------------------------------
# Fix 4: load_custom_resolution_choices keys its cache on the resolution_file argument.
# --------------------------------------------------------------------------------------


class TestCustomResolutionCacheKey:
    """The cache used to answer with whatever file was loaded first."""

    @pytest.fixture(autouse=True)
    def _isolated_cache(self):
        resolutions.reset_custom_resolution_cache()
        yield
        resolutions.reset_custom_resolution_cache()

    @pytest.fixture
    def file_a(self, tmp_path):
        path = tmp_path / "resolutions_a.json"
        path.write_text(json.dumps([["Custom A", "111x222"]]), encoding="utf-8")
        return str(path)

    @pytest.fixture
    def file_b(self, tmp_path):
        path = tmp_path / "resolutions_b.json"
        path.write_text(json.dumps([["Custom B", "333x444"]]), encoding="utf-8")
        return str(path)

    def test_each_file_returns_its_own_choices(self, file_a, file_b):
        assert resolutions.load_custom_resolution_choices(file_a) == [("Custom A", "111x222")]
        assert resolutions.load_custom_resolution_choices(file_b) == [("Custom B", "333x444")]

    def test_switching_between_files_keeps_working(self, file_a, file_b):
        assert resolutions.load_custom_resolution_choices(file_b) == [("Custom B", "333x444")]
        assert resolutions.load_custom_resolution_choices(file_a) == [("Custom A", "111x222")]
        assert resolutions.load_custom_resolution_choices(file_b) == [("Custom B", "333x444")]

    def test_repeated_calls_with_one_path_stay_cached(self, file_a, tmp_path):
        assert resolutions.load_custom_resolution_choices(file_a) == [("Custom A", "111x222")]
        (tmp_path / "resolutions_a.json").unlink()
        # Served from the cache: an uncached load of a missing file would return [].
        assert resolutions.load_custom_resolution_choices(file_a) == [("Custom A", "111x222")]


# --------------------------------------------------------------------------------------
# Fix 5: match_archi.eval_condition uses re.fullmatch, not re.match.
# --------------------------------------------------------------------------------------


class TestArchitectureConditionAnchoring:
    """re.match was lax at the end and strict in the middle."""

    def test_trailing_junk_after_an_operator_is_rejected(self):
        # arch 89 is chosen so the two behaviours differ: ">=89garbage" used to parse
        # as ">=89" and match, and only an architecture satisfying ">=89" shows that.
        assert match_archi.match_nvidia_architecture({">=89garbage": "params"}, 89) == []

    def test_trailing_junk_after_a_bare_value_is_rejected(self):
        # "89x" at arch 89 used to match as "89"; at arch 90 both old and new return [],
        # so it would not discriminate.
        assert match_archi.match_nvidia_architecture({"89x": "params"}, 89) == []
        assert match_archi.match_nvidia_architecture({"89 x": "params"}, 89) == []

    def test_whitespace_around_the_operator_is_accepted(self):
        assert match_archi.match_nvidia_architecture({">= 89": "params"}, 89) == ["params"]
        assert match_archi.match_nvidia_architecture({"<= 50": "params"}, 50) == ["params"]

    def test_well_formed_conditions_are_unchanged(self):
        conditions = {
            "<89": "below",
            ">=89": "ada_plus",
            "89": "exactly",
            "<=50+>89": "edges",
            ">=70&<90": "ampere_ada",
        }
        assert match_archi.match_nvidia_architecture(conditions, 89) == [
            "ada_plus",
            "exactly",
            "ampere_ada",
        ]


# --------------------------------------------------------------------------------------
# Fix 6: serialize_prompt_units guards multi_prompts_gen_type with `or ""`.
# --------------------------------------------------------------------------------------


class TestSerializePromptUnitsNoneMode:
    """``"P" in None`` raised TypeError before the guard."""

    def test_none_mode_joins_lines_with_a_newline(self):
        assert prompt_parser.serialize_prompt_units("a\nb", ["a", "b"], None) == "a\nb"

    def test_none_mode_with_a_single_prompt(self):
        assert prompt_parser.serialize_prompt_units("solo", ["solo"], None) == "solo"

    def test_paragraph_mode_still_joins_with_a_blank_line(self):
        assert prompt_parser.serialize_prompt_units("a\n\nb", ["a", "b"], "PG") == "a\n\nb"

    def test_full_prompt_mode_still_returns_the_first_prompt(self):
        assert prompt_parser.serialize_prompt_units("a\nb", ["a", "b"], "FG") == "a"
