"""Tests for ``shared.utils.audio_metadata``.

The module stores WanGP generation settings inside audio files and recovers a
creation date from them.  Everything exercised here is pure python (``struct``,
``json``, ``os``, ``re``, ``datetime``); the WAV files are synthesised
byte-by-byte in ``tmp_path`` so no real audio is needed.

Covered:

* ``write_wav_text_chunk`` / ``read_wav_text_chunk`` -- the RIFF chunk walker:
  round trips, appending vs. replacing a chunk, RIFF size bookkeeping, the
  even-length padding rule, custom fourccs and encodings;
* the failure modes: non-RIFF data, RIFX/RF64, truncated headers, a chunk whose
  declared size runs past EOF, a missing path and fourcc validation;
* ``save_audio_metadata`` / ``read_audio_metadata`` -- JSON payloads, extension
  dispatch (including the mp3 branch when ``mutagen`` is unavailable);
* ``_parse_datetime_value`` -- the many accepted date spellings, the
  epoch/"bare year" heuristics and what it rejects;
* ``extract_creation_datetime_from_metadata`` -- key priority, the
  ``extra_info`` nesting and the include/exclude substring rules;
* ``_iter_tag_values``, ``_write_mp3_text_tag`` / ``_read_mp3_text_tag`` and
  ``_extract_native_audio_datetime`` -- driven through a hand-built fake
  ``mutagen`` module so the tests never depend on it being installed;
* ``resolve_audio_creation_datetime`` -- the metadata -> native tag -> file
  mtime fallback chain.

``datetime.fromtimestamp`` is local-time dependent, so expected values for epoch
inputs are computed with ``fromtimestamp`` rather than hard coded.  ``mutagen``
is always forced into a known state (absent, or a fake) so results never depend
on the machine's site-packages.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import types
from datetime import datetime, timedelta, timezone

import pytest

from conftest import import_pure_module

am = import_pure_module("shared.utils.audio_metadata")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

FMT_PAYLOAD = struct.pack("<HHIIHH", 1, 1, 44100, 88200, 2, 16)  # canonical PCM 'fmt '
DATA_PAYLOAD = b"\x00\x01\x02\x03"


def build_wav(chunks=((b"fmt ", FMT_PAYLOAD), (b"data", DATA_PAYLOAD)), trailer=b"") -> bytes:
    """Assemble a little-endian RIFF/WAVE file from ``(fourcc, payload)`` pairs."""

    body = bytearray(b"WAVE")
    for cid, payload in chunks:
        body += cid + struct.pack("<I", len(payload)) + payload
        if len(payload) & 1:
            body += b"\x00"  # RIFF chunks are padded to an even length
    return b"RIFF" + struct.pack("<I", len(body)) + bytes(body) + trailer


def write_wav(tmp_path, name="sound.wav", **kwargs):
    path = tmp_path / name
    path.write_bytes(build_wav(**kwargs))
    return str(path)


def parse_chunks(data: bytes):
    """Independent RIFF walker used to assert on what the writer produced."""

    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    assert struct.unpack_from("<I", data, 4)[0] == len(data) - 8
    out, pos = [], 12
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        out.append((cid, data[pos + 8:pos + 8 + size]))
        pos += 8 + size + (size & 1)
    return out


def install_no_mutagen(monkeypatch):
    """Make every ``import mutagen`` inside the module fail, whatever is installed."""

    monkeypatch.setitem(sys.modules, "mutagen", None)
    monkeypatch.setitem(sys.modules, "mutagen.id3", None)


class FakeFrame:
    """Stand-in for an ID3 text frame: a ``desc`` plus a ``text`` list."""

    def __init__(self, desc="", text=(), encoding=0):
        self.desc = desc
        self.text = list(text)
        self.encoding = encoding


def install_fake_mutagen(monkeypatch, store=None, file_factory=None):
    """Install a minimal fake ``mutagen`` / ``mutagen.id3`` in ``sys.modules``.

    ``store`` maps path -> ``{frame_key: frame}`` and stands in for the tags
    persisted on disk.  ``file_factory`` backs ``mutagen.File``.
    """

    store = {} if store is None else store

    class ID3NoHeaderError(Exception):
        pass

    class TXXX(FakeFrame):
        pass

    class COMM(FakeFrame):  # deliberately *not* a TXXX subclass, as in mutagen
        pass

    class ID3(dict):
        def __init__(self, path=None):
            super().__init__()
            self.path = path
            if path is not None:
                if path not in store:
                    raise ID3NoHeaderError(path)
                self.update(store[path])

        def add(self, frame):
            self[f"{type(frame).__name__}:{frame.desc}"] = frame

        def getall(self, name):
            return [frame for key, frame in self.items() if key.split(":")[0] == name]

        def save(self, path):
            store[path] = dict(self)

    id3_module = types.ModuleType("mutagen.id3")
    id3_module.ID3 = ID3
    id3_module.ID3NoHeaderError = ID3NoHeaderError
    id3_module.TXXX = TXXX
    id3_module.COMM = COMM

    root = types.ModuleType("mutagen")
    root.id3 = id3_module
    root.File = file_factory if file_factory is not None else (lambda path, easy=False: None)

    monkeypatch.setitem(sys.modules, "mutagen", root)
    monkeypatch.setitem(sys.modules, "mutagen.id3", id3_module)
    return types.SimpleNamespace(store=store, ID3=ID3, TXXX=TXXX, COMM=COMM,
                                 ID3NoHeaderError=ID3NoHeaderError)


# --------------------------------------------------------------------------- #
# write_wav_text_chunk / read_wav_text_chunk
# --------------------------------------------------------------------------- #

class TestWavTextChunkRoundTrip:
    def test_chunk_is_appended_after_the_existing_ones(self, tmp_path):
        path = write_wav(tmp_path)
        am.write_wav_text_chunk(path, path, "hello")

        chunks = parse_chunks((tmp_path / "sound.wav").read_bytes())
        assert chunks == [(b"fmt ", FMT_PAYLOAD), (b"data", DATA_PAYLOAD), (b"json", b"hello")]
        assert am.read_wav_text_chunk(path) == "hello"

    def test_read_returns_none_when_chunk_absent(self, tmp_path):
        assert am.read_wav_text_chunk(write_wav(tmp_path)) is None

    def test_existing_chunk_payload_is_replaced_in_place(self, tmp_path):
        path = write_wav(tmp_path, chunks=((b"json", b"a much longer old payload"),
                                           (b"data", DATA_PAYLOAD)))
        am.write_wav_text_chunk(path, path, "new")

        # the replacement keeps its original position, it is not moved to the end
        assert parse_chunks((tmp_path / "sound.wav").read_bytes()) == [
            (b"json", b"new"), (b"data", DATA_PAYLOAD),
        ]
        assert am.read_wav_text_chunk(path) == "new"

    def test_only_the_first_matching_chunk_is_replaced(self, tmp_path):
        path = write_wav(tmp_path, chunks=((b"json", b"first"), (b"json", b"second")))
        am.write_wav_text_chunk(path, path, "X")

        assert parse_chunks((tmp_path / "sound.wav").read_bytes()) == [
            (b"json", b"X"), (b"json", b"second"),
        ]
        # ...and the reader still only ever sees the first one
        assert am.read_wav_text_chunk(path) == "X"

    def test_odd_length_payload_is_padded_so_later_chunks_stay_aligned(self, tmp_path):
        path = write_wav(tmp_path, chunks=((b"json", b"odd"), (b"data", DATA_PAYLOAD)))
        raw = (tmp_path / "sound.wav").read_bytes()

        assert raw[12:12 + 8 + 4] == b"json" + struct.pack("<I", 3) + b"odd\x00"
        assert am.read_wav_text_chunk(path) == "odd"
        # writing again re-emits the pad byte and keeps the file even-sized
        am.write_wav_text_chunk(path, path, "12345")
        assert len((tmp_path / "sound.wav").read_bytes()) % 2 == 0

    def test_odd_length_source_chunk_survives_a_rewrite(self, tmp_path):
        path = write_wav(tmp_path, chunks=((b"data", b"\xaa\xbb\xcc"),))
        am.write_wav_text_chunk(path, path, "x")

        assert parse_chunks((tmp_path / "sound.wav").read_bytes()) == [
            (b"data", b"\xaa\xbb\xcc"), (b"json", b"x"),
        ]

    def test_out_path_may_differ_from_in_path(self, tmp_path):
        src = write_wav(tmp_path, name="in.wav")
        dst = str(tmp_path / "out.wav")
        am.write_wav_text_chunk(src, dst, "copied")

        assert am.read_wav_text_chunk(dst) == "copied"
        assert am.read_wav_text_chunk(src) is None
        assert (tmp_path / "in.wav").read_bytes() == build_wav()

    def test_custom_fourcc_is_independent_of_the_default(self, tmp_path):
        path = write_wav(tmp_path)
        am.write_wav_text_chunk(path, path, "blobby", fourcc=b"blob")

        assert am.read_wav_text_chunk(path, fourcc=b"blob") == "blobby"
        assert am.read_wav_text_chunk(path) is None

    def test_empty_text_round_trips_as_empty_string(self, tmp_path):
        path = write_wav(tmp_path)
        am.write_wav_text_chunk(path, path, "")

        assert (b"json", b"") in parse_chunks((tmp_path / "sound.wav").read_bytes())
        assert am.read_wav_text_chunk(path) == ""

    def test_unicode_payload_round_trips_through_utf8(self, tmp_path):
        path = write_wav(tmp_path)
        text = "prompt: éàü 日本語 🎧"
        am.write_wav_text_chunk(path, path, text)

        assert am.read_wav_text_chunk(path) == text
        # the byte payload is the utf-8 encoding, not the code points
        payload = dict(parse_chunks((tmp_path / "sound.wav").read_bytes()))[b"json"]
        assert payload == text.encode("utf-8")

    def test_non_default_encoding_round_trips(self, tmp_path):
        path = write_wav(tmp_path)
        am.write_wav_text_chunk(path, path, "café", encoding="latin-1")

        assert am.read_wav_text_chunk(path, encoding="latin-1") == "café"

    def test_decoding_with_the_wrong_encoding_raises(self, tmp_path):
        path = write_wav(tmp_path)
        am.write_wav_text_chunk(path, path, "café")  # utf-8 bytes

        with pytest.raises(UnicodeDecodeError):
            am.read_wav_text_chunk(path, encoding="ascii")

    def test_zero_sized_chunks_are_preserved(self, tmp_path):
        path = write_wav(tmp_path, chunks=((b"junk", b""), (b"data", DATA_PAYLOAD)))
        am.write_wav_text_chunk(path, path, "kept")

        assert parse_chunks((tmp_path / "sound.wav").read_bytes()) == [
            (b"junk", b""), (b"data", DATA_PAYLOAD), (b"json", b"kept"),
        ]

    def test_trailing_bytes_too_short_for_a_header_are_dropped(self, tmp_path):
        # Known quirk (pinned, not endorsed): the chunk walker stops as soon as
        # fewer than 8 bytes remain, so any short tail is silently discarded on
        # rewrite instead of being carried over or reported.
        path = write_wav(tmp_path, chunks=((b"data", DATA_PAYLOAD),), trailer=b"XYZ")
        assert (tmp_path / "sound.wav").read_bytes().endswith(b"XYZ")

        am.write_wav_text_chunk(path, path, "q")
        assert b"XYZ" not in (tmp_path / "sound.wav").read_bytes()


class TestWavTextChunkErrors:
    BAD_HEADERS = {
        "empty": b"",
        "too_short": b"RIFF\x04\x00\x00",
        "plain_text": b"this is not audio at all, just some text",
        "rifx_big_endian": b"RIFX" + struct.pack(">I", 4) + b"WAVE",
        "rf64": b"RF64" + struct.pack("<I", 4) + b"WAVE",
        "riff_but_not_wave": b"RIFF" + struct.pack("<I", 4) + b"AVI ",
    }

    @pytest.mark.parametrize("name", sorted(BAD_HEADERS))
    def test_read_rejects_non_riff_wave(self, tmp_path, name):
        path = tmp_path / f"{name}.wav"
        path.write_bytes(self.BAD_HEADERS[name])

        with pytest.raises(ValueError, match="Not a standard little-endian RIFF/WAVE"):
            am.read_wav_text_chunk(str(path))

    @pytest.mark.parametrize("name", sorted(BAD_HEADERS))
    def test_write_rejects_non_riff_wave(self, tmp_path, name):
        path = tmp_path / f"{name}.wav"
        path.write_bytes(self.BAD_HEADERS[name])

        with pytest.raises(ValueError, match="Not a standard little-endian RIFF/WAVE"):
            am.write_wav_text_chunk(str(path), str(path), "x")
        assert path.read_bytes() == self.BAD_HEADERS[name]  # left untouched

    def test_chunk_size_past_eof_is_reported_as_corrupt(self, tmp_path):
        path = tmp_path / "corrupt.wav"
        body = b"WAVE" + b"data" + struct.pack("<I", 4096) + b"\x00\x00\x00\x00"
        path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)

        with pytest.raises(ValueError, match="chunk size exceeds file length"):
            am.read_wav_text_chunk(str(path))
        with pytest.raises(ValueError, match="chunk size exceeds file length"):
            am.write_wav_text_chunk(str(path), str(path), "x")

    def test_truncated_after_a_valid_chunk_still_reads_that_chunk(self, tmp_path):
        # a whole chunk survives, the trailing 5-byte stub is simply ignored
        path = tmp_path / "trunc.wav"
        path.write_bytes(build_wav(chunks=((b"json", b"survivor"),), trailer=b"data\x10"))

        assert am.read_wav_text_chunk(str(path)) == "survivor"

    def test_missing_file_raises_file_not_found(self, tmp_path):
        missing = str(tmp_path / "nope.wav")

        with pytest.raises(FileNotFoundError):
            am.read_wav_text_chunk(missing)
        with pytest.raises(FileNotFoundError):
            am.write_wav_text_chunk(missing, missing, "x")

    @pytest.mark.parametrize("fourcc", [b"", b"abc", b"abcde", b"ab\x01d", b"ab\x7fd"])
    def test_write_requires_four_printable_ascii_bytes(self, tmp_path, fourcc):
        path = write_wav(tmp_path)

        with pytest.raises(ValueError, match="4 printable ASCII bytes"):
            am.write_wav_text_chunk(path, path, "x", fourcc=fourcc)

    @pytest.mark.parametrize("fourcc", [b"", b"abc", b"abcde"])
    def test_read_requires_a_four_byte_fourcc(self, tmp_path, fourcc):
        path = write_wav(tmp_path)

        with pytest.raises(ValueError, match="fourcc must be 4 bytes"):
            am.read_wav_text_chunk(path, fourcc=fourcc)

    def test_read_accepts_non_printable_fourcc_unlike_write(self, tmp_path):
        # asymmetric on purpose: only the writer enforces printability
        path = write_wav(tmp_path)
        assert am.read_wav_text_chunk(path, fourcc=b"\x00\x01\x02\x03") is None

    def test_header_check_runs_before_the_fourcc_check(self, tmp_path):
        path = tmp_path / "bad.wav"
        path.write_bytes(b"nope")

        with pytest.raises(ValueError, match="Not a standard little-endian RIFF/WAVE"):
            am.write_wav_text_chunk(str(path), str(path), "x", fourcc=b"nope!")


# --------------------------------------------------------------------------- #
# save_audio_metadata / read_audio_metadata
# --------------------------------------------------------------------------- #

class TestSaveReadAudioMetadata:
    CONFIGS = {
        "type": "WanGP by DeepBeepMeep",
        "prompt": "a cat playing piano",
        "seed": 42,
        "loras": ["a.safetensors", "b.safetensors"],
        "extra_info": {"nested": {"flag": True, "ratio": 0.5}},
        "unicode": "éàü 🎹",
    }

    def test_wav_round_trip(self, tmp_path):
        path = write_wav(tmp_path)
        am.save_audio_metadata(path, self.CONFIGS)

        assert am.read_audio_metadata(path) == self.CONFIGS

    def test_saving_twice_replaces_rather_than_duplicates(self, tmp_path):
        path = write_wav(tmp_path)
        am.save_audio_metadata(path, {"seed": 1})
        am.save_audio_metadata(path, {"seed": 2})

        chunks = parse_chunks((tmp_path / "sound.wav").read_bytes())
        assert [cid for cid, _ in chunks].count(b"json") == 1
        assert am.read_audio_metadata(path) == {"seed": 2}

    def test_audio_payload_is_untouched_by_a_metadata_write(self, tmp_path):
        path = write_wav(tmp_path)
        am.save_audio_metadata(path, self.CONFIGS)

        chunks = dict(parse_chunks((tmp_path / "sound.wav").read_bytes()))
        assert chunks[b"data"] == DATA_PAYLOAD
        assert chunks[b"fmt "] == FMT_PAYLOAD

    def test_extension_matching_is_case_insensitive(self, tmp_path):
        path = write_wav(tmp_path, name="LOUD.WAV")
        am.save_audio_metadata(path, {"seed": 7})

        assert am.read_audio_metadata(path) == {"seed": 7}

    @pytest.mark.parametrize("configs", [{}, [], [1, 2, 3], "just a string", 0, None])
    def test_non_dict_payloads_round_trip_through_json(self, tmp_path, configs):
        # empty containers serialise to the truthy strings "{}"/"[]", so they
        # survive the falsy-payload short circuit in ``read_audio_metadata``
        path = write_wav(tmp_path)
        am.save_audio_metadata(path, configs)

        assert am.read_audio_metadata(path) == configs

    def test_read_returns_none_when_no_metadata_chunk_present(self, tmp_path):
        assert am.read_audio_metadata(write_wav(tmp_path)) is None

    def test_empty_chunk_reads_back_as_none_not_a_json_error(self, tmp_path):
        # ``read_audio_metadata`` short-circuits on a falsy payload
        path = write_wav(tmp_path, chunks=((b"json", b""), (b"data", DATA_PAYLOAD)))

        assert am.read_audio_metadata(path) is None

    def test_malformed_json_propagates(self, tmp_path):
        path = write_wav(tmp_path, chunks=((b"json", b"{not json"),))

        with pytest.raises(json.JSONDecodeError):
            am.read_audio_metadata(path)

    def test_corrupt_wav_propagates_from_read(self, tmp_path):
        path = tmp_path / "junk.wav"
        path.write_bytes(b"definitely not a wav file")

        with pytest.raises(ValueError):
            am.read_audio_metadata(str(path))

    @pytest.mark.parametrize("name", ["clip.flac", "clip.ogg", "clip.m4a", "clip", "clip.WAV.txt"])
    def test_unsupported_extensions_read_as_none(self, tmp_path, name):
        path = tmp_path / name
        path.write_bytes(b"whatever")

        assert am.read_audio_metadata(str(path)) is None

    @pytest.mark.parametrize("name", ["clip.flac", "clip.ogg", "clip"])
    def test_unsupported_extensions_cannot_be_written(self, tmp_path, name):
        path = tmp_path / name
        path.write_bytes(b"whatever")

        with pytest.raises(ValueError, match="Unsupported audio metadata format"):
            am.save_audio_metadata(str(path), {"seed": 1})

    def test_mp3_read_returns_none_without_mutagen(self, tmp_path, monkeypatch):
        install_no_mutagen(monkeypatch)
        path = tmp_path / "song.mp3"
        path.write_bytes(b"\xff\xfb\x90\x00")

        assert am.read_audio_metadata(str(path)) is None

    def test_mp3_write_requires_mutagen(self, tmp_path, monkeypatch):
        install_no_mutagen(monkeypatch)
        path = tmp_path / "song.mp3"
        path.write_bytes(b"\xff\xfb\x90\x00")

        with pytest.raises(RuntimeError, match="mutagen is required for mp3 metadata"):
            am.save_audio_metadata(str(path), {"seed": 1})


# --------------------------------------------------------------------------- #
# _parse_datetime_value
# --------------------------------------------------------------------------- #

class TestParseDatetimeValue:
    @pytest.mark.parametrize("text,expected", [
        ("2024-01-02 03:04:05", datetime(2024, 1, 2, 3, 4, 5)),
        ("2024-01-02 03:04", datetime(2024, 1, 2, 3, 4)),
        ("2024-01-02", datetime(2024, 1, 2)),
        ("2024/01/02 03:04:05", datetime(2024, 1, 2, 3, 4, 5)),
        ("2024/01/02", datetime(2024, 1, 2)),
        ("2024-01-02-03h04m05s", datetime(2024, 1, 2, 3, 4, 5)),  # WanGP filename style
        ("20240102", datetime(2024, 1, 2)),
        ("2024", datetime(2024, 1, 1)),
        ("2024:01:02 03:04:05", datetime(2024, 1, 2, 3, 4, 5)),  # EXIF style
        ("  2024-01-02  ", datetime(2024, 1, 2)),  # stripped first
        ("2024-01-02T03:04:05", datetime(2024, 1, 2, 3, 4, 5)),  # via fromisoformat
    ])
    def test_naive_string_formats(self, text, expected):
        assert am._parse_datetime_value(text) == expected

    @pytest.mark.parametrize("text,offset_hours", [
        ("2024-01-02T03:04:05Z", 0),
        ("2024-01-02T03:04:05+00:00", 0),
        ("2024-01-02T03:04:05+02:00", 2),
        ("2024-01-02T03:04:05-05:00", -5),
    ])
    def test_timezone_aware_iso_strings_keep_their_offset(self, text, offset_hours):
        parsed = am._parse_datetime_value(text)

        assert parsed.utcoffset() == timedelta(hours=offset_hours)
        assert parsed.replace(tzinfo=None) == datetime(2024, 1, 2, 3, 4, 5)
        assert parsed == datetime(2024, 1, 2, 3, 4, 5,
                                  tzinfo=timezone(timedelta(hours=offset_hours)))

    @pytest.mark.parametrize("value", [
        None, "", "   ", "\t\n", "not a date", "12:30", "tomorrow",
        "2024-13-45", [2024], {"year": 2024}, 0, 0.0, -1, -1700000000,
        "2024:01:02",  # EXIF date without a time part: the ':' fixup needs \s
        "1000000000000",  # 13 digits but not > 1e12, so used as raw seconds -> overflows
    ])
    def test_unparseable_values_return_none(self, value):
        assert am._parse_datetime_value(value) is None

    def test_datetime_instances_pass_through_unchanged(self):
        dt = datetime(2020, 5, 6, 7, 8, 9)
        assert am._parse_datetime_value(dt) is dt

    @pytest.mark.parametrize("value", [1700000000, 1700000000.5, "1700000000", "1700000000.5"])
    def test_ten_digit_epoch_seconds(self, value):
        assert am._parse_datetime_value(value) == datetime.fromtimestamp(float(value))

    def test_thirteen_digit_epoch_string_is_treated_as_milliseconds(self):
        assert am._parse_datetime_value("1700000000000") == datetime.fromtimestamp(1700000000.0)

    @pytest.mark.parametrize("value,year", [(1900, 1900), (2024, 2024), (3000, 3000), (2024.7, 2024)])
    def test_bare_numbers_in_1900_3000_are_years(self, value, year):
        assert am._parse_datetime_value(value) == datetime(year, 1, 1)

    @pytest.mark.parametrize("value", [1899, 3001])
    def test_numbers_just_outside_the_year_window_are_epoch_seconds(self, value):
        # boundary of the "looks like a year" heuristic
        assert am._parse_datetime_value(value) == datetime.fromtimestamp(float(value))

    def test_numeric_strings_below_1900_are_years_but_ints_are_not(self):
        # Inconsistency pinned, not endorsed: the year window only guards the
        # numeric branch, so the string and the int disagree for the same value.
        assert am._parse_datetime_value("1234") == datetime(1234, 1, 1)
        assert am._parse_datetime_value(1234) == datetime.fromtimestamp(1234.0)

    def test_booleans_follow_the_numeric_branch(self):
        # ``bool`` is an ``int`` subclass: True is epoch second 1, False is 0 -> None
        assert am._parse_datetime_value(True) == datetime.fromtimestamp(1.0)
        assert am._parse_datetime_value(False) is None


# --------------------------------------------------------------------------- #
# extract_creation_datetime_from_metadata
# --------------------------------------------------------------------------- #

class TestExtractCreationDatetimeFromMetadata:
    @pytest.mark.parametrize("metadata", [None, "2024-01-02", ["2024-01-02"], 12345, set()])
    def test_non_dict_metadata_returns_none(self, metadata):
        assert am.extract_creation_datetime_from_metadata(metadata) is None

    @pytest.mark.parametrize("key", list(am._CREATION_KEYS))
    def test_every_explicit_creation_key_is_honoured(self, key):
        assert am.extract_creation_datetime_from_metadata({key: "2024-01-02"}) == datetime(2024, 1, 2)

    def test_creation_keys_win_in_their_declared_order(self):
        metadata = {"created_at": "2024-03-04", "creation_date": "2024-01-02"}
        # dict insertion order is irrelevant: _CREATION_KEYS order decides
        assert am.extract_creation_datetime_from_metadata(metadata) == datetime(2024, 1, 2)

    def test_unparseable_creation_key_falls_through_to_the_next_candidate(self):
        metadata = {"creation_date": "not a date", "date": "2024-05-06"}
        assert am.extract_creation_datetime_from_metadata(metadata) == datetime(2024, 5, 6)

    def test_extra_info_creation_keys_beat_generic_top_level_keys(self):
        metadata = {"date": "2024-09-10", "extra_info": {"created_at": "2024-07-08"}}
        assert am.extract_creation_datetime_from_metadata(metadata) == datetime(2024, 7, 8)

    def test_non_dict_extra_info_is_ignored(self):
        metadata = {"extra_info": "2020-01-01", "date": "2024-01-02"}
        assert am.extract_creation_datetime_from_metadata(metadata) == datetime(2024, 1, 2)

    @pytest.mark.parametrize("key", ["date", "Creation Date", "timestamp", "created", "MOD_TIME"])
    def test_generic_keys_matching_a_date_substring_are_used(self, key):
        assert am.extract_creation_datetime_from_metadata({key: "2024-01-02"}) == datetime(2024, 1, 2)

    @pytest.mark.parametrize("key", list(am._DATE_KEY_EXCLUDE))
    def test_excluded_keys_are_skipped(self, key):
        assert am.extract_creation_datetime_from_metadata({key: "2024-01-02"}) is None

    def test_exclusion_is_a_substring_match(self):
        metadata = {"total_generation_time_str": "2024-01-02", "date": "2024-05-06"}
        assert am.extract_creation_datetime_from_metadata(metadata) == datetime(2024, 5, 6)

    @pytest.mark.parametrize("metadata", [
        {}, {"seed": 12345}, {"prompt": "a cat"}, {"video_length": 81},
        {"extra_info": {}}, {"date": "nonsense"},
    ])
    def test_returns_none_when_nothing_looks_like_a_date(self, metadata):
        assert am.extract_creation_datetime_from_metadata(metadata) is None

    def test_generic_scan_also_walks_extra_info(self):
        metadata = {"extra_info": {"shoot_date": "2024-11-12"}}
        assert am.extract_creation_datetime_from_metadata(metadata) == datetime(2024, 11, 12)

    def test_datetime_objects_are_accepted_directly(self):
        dt = datetime(2021, 5, 5, 6, 7)
        assert am.extract_creation_datetime_from_metadata({"creation_datetime": dt}) is dt

    def test_small_numeric_time_fields_are_misread_as_epoch_seconds(self):
        # Known trap (pinned, not endorsed): any un-excluded key containing
        # "time"/"date"/"created"/"timestamp" whose value is a small number is
        # converted with fromtimestamp, so a 12 second duration reads as 1970.
        assert am.extract_creation_datetime_from_metadata({"sampling_time": 12}) == \
            datetime.fromtimestamp(12.0)


# --------------------------------------------------------------------------- #
# _iter_tag_values
# --------------------------------------------------------------------------- #

class TestIterTagValues:
    def test_none_yields_nothing(self):
        assert list(am._iter_tag_values(None)) == []

    def test_scalars_yield_themselves(self):
        assert list(am._iter_tag_values("2024")) == ["2024"]
        assert list(am._iter_tag_values(7)) == [7]

    def test_sequences_are_flattened_recursively(self):
        assert list(am._iter_tag_values(["a", ["b", ("c",)]])) == ["a", "b", "c"]

    def test_nested_none_contributes_nothing(self):
        assert list(am._iter_tag_values(["a", None, ["b"]])) == ["a", "b"]

    def test_objects_with_a_text_attribute_are_unwrapped(self):
        assert list(am._iter_tag_values(FakeFrame(text=["x", "y"]))) == ["x", "y"]

    def test_scalar_text_attribute_is_yielded_as_is(self):
        frame = FakeFrame()
        frame.text = "z"
        assert list(am._iter_tag_values(frame)) == ["z"]

    def test_frames_inside_a_list_are_unwrapped_too(self):
        frames = [FakeFrame(text=["q"]), FakeFrame(text=["r", "s"])]
        assert list(am._iter_tag_values(frames)) == ["q", "r", "s"]

    def test_text_none_yields_a_single_none(self):
        frame = FakeFrame()
        frame.text = None
        assert list(am._iter_tag_values(frame)) == [None]

    def test_dicts_are_treated_as_opaque_scalars(self):
        value = {"a": 1}
        assert list(am._iter_tag_values(value)) == [value]


# --------------------------------------------------------------------------- #
# mp3 / mutagen backed helpers -- driven through a fake mutagen module
# --------------------------------------------------------------------------- #

class TestMp3TextTag:
    def test_write_then_read_round_trip(self, monkeypatch):
        fake = install_fake_mutagen(monkeypatch)
        am._write_mp3_text_tag("/song.mp3", '{"seed": 1}')

        assert am._read_mp3_text_tag("/song.mp3") == '{"seed": 1}'

    def test_write_creates_a_fresh_tag_when_the_file_has_no_id3_header(self, monkeypatch):
        fake = install_fake_mutagen(monkeypatch)
        am._write_mp3_text_tag("/no-header.mp3", "payload")

        assert list(fake.store["/no-header.mp3"]) == ["TXXX:WanGP"]

    def test_rewriting_replaces_the_previous_frame_instead_of_duplicating(self, monkeypatch):
        fake = install_fake_mutagen(monkeypatch)
        am._write_mp3_text_tag("/song.mp3", "first")
        am._write_mp3_text_tag("/song.mp3", "second")

        assert len(fake.store["/song.mp3"]) == 1
        assert am._read_mp3_text_tag("/song.mp3") == "second"

    def test_other_frames_are_left_alone(self, monkeypatch):
        fake = install_fake_mutagen(monkeypatch)
        fake.store["/song.mp3"] = {
            "TXXX:Other": fake.TXXX(desc="Other", text=["keep me"]),
            "COMM:WanGP": fake.COMM(desc="WanGP", text=["comment"]),
        }
        am._write_mp3_text_tag("/song.mp3", "payload")

        assert set(fake.store["/song.mp3"]) == {"TXXX:Other", "COMM:WanGP", "TXXX:WanGP"}

    def test_custom_tag_key_is_isolated(self, monkeypatch):
        install_fake_mutagen(monkeypatch)
        am._write_mp3_text_tag("/song.mp3", "payload", tag_key="Custom")

        assert am._read_mp3_text_tag("/song.mp3", tag_key="Custom") == "payload"
        assert am._read_mp3_text_tag("/song.mp3") is None

    def test_read_falls_back_to_a_matching_comm_frame(self, monkeypatch):
        fake = install_fake_mutagen(monkeypatch)
        fake.store["/song.mp3"] = {"COMM:WanGP": fake.COMM(desc="WanGP", text=["from comm"])}

        assert am._read_mp3_text_tag("/song.mp3") == "from comm"

    def test_txxx_is_preferred_over_comm(self, monkeypatch):
        fake = install_fake_mutagen(monkeypatch)
        fake.store["/song.mp3"] = {
            "COMM:WanGP": fake.COMM(desc="WanGP", text=["from comm"]),
            "TXXX:WanGP": fake.TXXX(desc="WanGP", text=["from txxx"]),
        }

        assert am._read_mp3_text_tag("/song.mp3") == "from txxx"

    def test_read_returns_none_without_an_id3_header(self, monkeypatch):
        install_fake_mutagen(monkeypatch)
        assert am._read_mp3_text_tag("/never-written.mp3") is None

    def test_read_returns_none_when_the_frame_text_is_empty(self, monkeypatch):
        fake = install_fake_mutagen(monkeypatch)
        fake.store["/song.mp3"] = {"TXXX:WanGP": fake.TXXX(desc="WanGP", text=[])}

        assert am._read_mp3_text_tag("/song.mp3") is None

    def test_read_returns_none_when_mutagen_is_unavailable(self, monkeypatch):
        install_no_mutagen(monkeypatch)
        assert am._read_mp3_text_tag("/song.mp3") is None

    def test_write_raises_runtime_error_when_mutagen_is_unavailable(self, monkeypatch):
        install_no_mutagen(monkeypatch)

        with pytest.raises(RuntimeError, match="mutagen is required"):
            am._write_mp3_text_tag("/song.mp3", "payload")

    def test_save_and_read_audio_metadata_use_the_mp3_backend(self, tmp_path, monkeypatch):
        install_fake_mutagen(monkeypatch)
        path = str(tmp_path / "song.mp3")
        am.save_audio_metadata(path, {"seed": 3, "prompt": "x"})

        assert am.read_audio_metadata(path) == {"seed": 3, "prompt": "x"}


class TestExtractNativeAudioDatetime:
    def test_returns_none_without_mutagen(self, monkeypatch):
        install_no_mutagen(monkeypatch)
        assert am._extract_native_audio_datetime("/song.mp3") is None

    def test_returns_none_when_file_cannot_be_opened(self, monkeypatch):
        def boom(path, easy=False):
            raise OSError("unreadable")

        install_fake_mutagen(monkeypatch, file_factory=boom)
        assert am._extract_native_audio_datetime("/song.mp3") is None

    @pytest.mark.parametrize("audio", [None, types.SimpleNamespace(tags=None)])
    def test_returns_none_without_tags(self, monkeypatch, audio):
        install_fake_mutagen(monkeypatch, file_factory=lambda path, easy=False: audio)
        assert am._extract_native_audio_datetime("/song.mp3") is None

    def _id3_audio(self, monkeypatch, frames):
        """An audio object whose ``tags`` is an ID3-like mapping of frames."""

        fake = install_fake_mutagen(monkeypatch)
        tags = fake.ID3()
        tags.update(frames)
        monkeypatch.setitem(sys.modules, "mutagen",
                            sys.modules["mutagen"])  # keep the same fake in place
        sys.modules["mutagen"].File = lambda path, easy=False: types.SimpleNamespace(tags=tags)
        return fake

    def test_tdrc_frame_is_used(self, monkeypatch):
        self._id3_audio(monkeypatch, {"TDRC:": FakeFrame(text=["2024-01-02 03:04:05"])})
        assert am._extract_native_audio_datetime("/song.mp3") == datetime(2024, 1, 2, 3, 4, 5)

    def test_tdrc_wins_over_tyer(self, monkeypatch):
        self._id3_audio(monkeypatch, {
            "TYER:": FakeFrame(text=["1999"]),
            "TDRC:": FakeFrame(text=["2024-01-02"]),
        })
        assert am._extract_native_audio_datetime("/song.mp3") == datetime(2024, 1, 2)

    def test_falls_through_to_a_later_frame_when_the_first_is_unparseable(self, monkeypatch):
        self._id3_audio(monkeypatch, {
            "TDRC:": FakeFrame(text=["garbage"]),
            "TDEN:": FakeFrame(text=["2024-06-07"]),
        })
        assert am._extract_native_audio_datetime("/song.mp3") == datetime(2024, 6, 7)

    def test_txxx_frames_are_matched_on_a_date_like_description(self, monkeypatch):
        self._id3_audio(monkeypatch, {
            "TXXX:Comment": FakeFrame(desc="Comment", text=["2020-01-01"]),
            "TXXX:Creation Date": FakeFrame(desc="Creation Date", text=["2024-08-09"]),
        })
        assert am._extract_native_audio_datetime("/song.mp3") == datetime(2024, 8, 9)

    @pytest.mark.parametrize("key,expected", [
        ("ICRD", datetime(2024, 3, 4)),
        ("\xa9day", datetime(2024, 3, 4)),
        ("year", datetime(2024, 3, 4)),
        ("DATE", datetime(2024, 3, 4)),
    ])
    def test_plain_mapping_tags_are_scanned_by_key_name(self, monkeypatch, key, expected):
        tags = {key: ["2024-03-04"]}
        install_fake_mutagen(monkeypatch,
                            file_factory=lambda path, easy=False: types.SimpleNamespace(tags=tags))
        assert am._extract_native_audio_datetime("/song.flac") == expected

    @pytest.mark.parametrize("key", ["artist", "album", "generation_time", "duration_seconds"])
    def test_plain_mapping_tags_with_unrelated_keys_are_ignored(self, monkeypatch, key):
        tags = {key: ["2024-03-04"]}
        install_fake_mutagen(monkeypatch,
                            file_factory=lambda path, easy=False: types.SimpleNamespace(tags=tags))
        assert am._extract_native_audio_datetime("/song.flac") is None

    def test_getall_failures_are_swallowed(self, monkeypatch):
        class ExplodingTags(dict):
            def getall(self, name):
                raise RuntimeError("boom")

        tags = ExplodingTags({"ICRD": ["2024-03-04"]})
        install_fake_mutagen(monkeypatch,
                            file_factory=lambda path, easy=False: types.SimpleNamespace(tags=tags))
        # the ID3 branch blows up but the generic items() scan still succeeds
        assert am._extract_native_audio_datetime("/song.mp3") == datetime(2024, 3, 4)


# --------------------------------------------------------------------------- #
# resolve_audio_creation_datetime
# --------------------------------------------------------------------------- #

MTIME = 1_600_000_000  # fixed epoch; never read from the wall clock


@pytest.fixture
def wav_with_mtime(tmp_path, monkeypatch):
    """A minimal WAV whose mtime is pinned, with mutagen forced absent."""

    install_no_mutagen(monkeypatch)

    def _make(name="sound.wav", **kwargs):
        path = write_wav(tmp_path, name=name, **kwargs)
        import os

        os.utime(path, (MTIME, MTIME))
        return path

    return _make


class TestResolveAudioCreationDatetime:
    def test_explicit_metadata_wins(self, wav_with_mtime):
        path = wav_with_mtime()
        assert am.resolve_audio_creation_datetime(path, {"creation_date": "2024-01-02"}) == \
            datetime(2024, 1, 2)

    def test_embedded_metadata_is_read_from_the_file(self, wav_with_mtime):
        import os

        path = wav_with_mtime()
        am.save_audio_metadata(path, {"creation_date": "2023-06-07 08:09:10"})
        os.utime(path, (MTIME, MTIME))  # the write refreshed mtime

        assert am.resolve_audio_creation_datetime(path) == datetime(2023, 6, 7, 8, 9, 10)

    def test_falls_back_to_file_mtime_when_there_is_no_metadata(self, wav_with_mtime):
        path = wav_with_mtime()
        assert am.resolve_audio_creation_datetime(path) == datetime.fromtimestamp(MTIME)

    def test_metadata_without_a_date_falls_back_to_mtime(self, wav_with_mtime):
        path = wav_with_mtime()
        assert am.resolve_audio_creation_datetime(path, {"seed": 1}) == datetime.fromtimestamp(MTIME)

    def test_passing_metadata_explicitly_suppresses_the_file_read(self, wav_with_mtime):
        # the on-disk chunk says 2020 but the caller-supplied dict has priority
        path = wav_with_mtime()
        am.save_audio_metadata(path, {"creation_date": "2020-01-01"})
        import os

        os.utime(path, (MTIME, MTIME))

        assert am.resolve_audio_creation_datetime(path, {"creation_date": "2024-01-02"}) == \
            datetime(2024, 1, 2)

    def test_corrupt_json_chunk_degrades_to_mtime(self, wav_with_mtime):
        path = wav_with_mtime(chunks=((b"json", b"{not json"),))
        assert am.resolve_audio_creation_datetime(path) == datetime.fromtimestamp(MTIME)

    def test_non_riff_file_degrades_to_mtime(self, tmp_path, monkeypatch):
        import os

        install_no_mutagen(monkeypatch)
        path = tmp_path / "fake.wav"
        path.write_bytes(b"this is not audio")
        os.utime(path, (MTIME, MTIME))

        assert am.resolve_audio_creation_datetime(str(path)) == datetime.fromtimestamp(MTIME)

    def test_unsupported_extension_degrades_to_mtime(self, tmp_path, monkeypatch):
        import os

        install_no_mutagen(monkeypatch)
        path = tmp_path / "clip.ogg"
        path.write_bytes(b"OggS")
        os.utime(path, (MTIME, MTIME))

        assert am.resolve_audio_creation_datetime(str(path)) == datetime.fromtimestamp(MTIME)

    def test_missing_file_raises(self, tmp_path, monkeypatch):
        install_no_mutagen(monkeypatch)

        with pytest.raises(FileNotFoundError):
            am.resolve_audio_creation_datetime(str(tmp_path / "gone.wav"))

    def test_native_tags_are_tried_before_the_mtime(self, tmp_path, monkeypatch):
        import os

        tags = {"ICRD": ["2022-02-02"]}
        install_fake_mutagen(monkeypatch,
                             file_factory=lambda path, easy=False: types.SimpleNamespace(tags=tags))
        path = tmp_path / "sound.wav"
        path.write_bytes(build_wav())
        os.utime(path, (MTIME, MTIME))

        assert am.resolve_audio_creation_datetime(str(path)) == datetime(2022, 2, 2)

    def test_wangp_metadata_beats_native_tags(self, tmp_path, monkeypatch):
        tags = {"ICRD": ["2022-02-02"]}
        install_fake_mutagen(monkeypatch,
                             file_factory=lambda path, easy=False: types.SimpleNamespace(tags=tags))
        path = tmp_path / "sound.wav"
        path.write_bytes(build_wav())

        assert am.resolve_audio_creation_datetime(str(path), {"created_at": "2024-04-04"}) == \
            datetime(2024, 4, 4)


class TestFileCreationDatetime:
    def test_uses_the_modification_time(self, tmp_path):
        import os

        path = tmp_path / "any.bin"
        path.write_bytes(b"x")
        os.utime(path, (123456789, 987654321))  # (atime, mtime) -- mtime is the one used

        assert am._get_file_creation_datetime(str(path)) == datetime.fromtimestamp(987654321)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            am._get_file_creation_datetime(str(tmp_path / "absent"))
