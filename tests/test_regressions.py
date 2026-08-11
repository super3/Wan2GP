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
