"""Tests for ``shared.utils.filename_formatter.FilenameFormatter``.

The formatter expands a user template such as ``"{date}-{prompt(50)}-{seed}"``
into a filesystem-safe filename (without extension).  Covered here:

* template validation -- which placeholder keys are accepted and the
  ``ValueError`` raised for anything else;
* the ``{date}`` placeholder: default format, the user-friendly token
  language (``YYYY``/``MM``/``DD``/``HH``/``hh``/``mm``/``ss``), separator
  validation and the fallback to the default format for invalid formats;
* value placeholders and their aliases (``steps``/``frames``/``cfg``);
* truncation via ``{prompt(N)}`` and its boundary values;
* sanitisation of path separators, Windows-illegal characters, control
  characters and whitespace runs;
* the empty-result fallback and unicode handling.

The module reads the wall clock through ``time.time()`` and
``datetime.fromtimestamp()``; the ``frozen_clock`` fixture replaces both
module-level lookups so every date assertion is a literal string.  The module
never touches the filesystem or the network.
"""

from __future__ import annotations

import types
from datetime import datetime

import pytest


import shared.utils.filename_formatter as ff
FilenameFormatter = ff.FilenameFormatter

# 2025-01-15 14:30:45 local time -- the exact epoch value is irrelevant because
# the frozen ``datetime`` below ignores it, it only has to be stable.
FIXED_TIMESTAMP = 1736951445.0
FROZEN_NOW = datetime(2025, 1, 15, 14, 30, 45)
DEFAULT_DATE = "2025-01-15-14h30m45s"

SETTINGS = {
    "prompt": "A beautiful sunset over the ocean",
    "seed": 12345,
    "resolution": "1280x720",
    "num_inference_steps": 30,
    "flow_shift": 5.0,
    "video_length": 81,
    "guidance_scale": 7.5,
}

# What each alias must render to given ``SETTINGS`` -- written out as literals so the
# alias tests compare against a fixed string rather than against another call to the
# function under test.
ALIAS_EXPECTATIONS = {"steps": "30", "frames": "81", "cfg": "7.5"}


@pytest.fixture
def frozen_clock(monkeypatch):
    """Freeze the module's clock; yields the list of timestamps it converted."""

    converted = []

    class _FrozenDatetime:
        @staticmethod
        def fromtimestamp(timestamp):
            converted.append(timestamp)
            return FROZEN_NOW

    monkeypatch.setattr(ff, "time", types.SimpleNamespace(time=lambda: FIXED_TIMESTAMP))
    monkeypatch.setattr(ff, "datetime", _FrozenDatetime)
    return converted


def fmt(template, settings=None):
    return FilenameFormatter.format_filename(template, settings if settings is not None else SETTINGS)


class TestTemplateValidation:
    @pytest.mark.parametrize("key", sorted(FilenameFormatter.ALLOWED_KEYS))
    def test_every_allowed_key_is_accepted(self, key):
        assert FilenameFormatter("{%s}" % key).template == "{%s}" % key

    @pytest.mark.parametrize("template", ["{unknown}", "{prompt}-{nope(3)}", "{Seed}", "{PROMPT}"])
    def test_unknown_placeholder_raises(self, template):
        with pytest.raises(ValueError, match="Unknown placeholder"):
            FilenameFormatter(template)

    def test_error_message_lists_allowed_keys_sorted(self):
        with pytest.raises(ValueError) as excinfo:
            FilenameFormatter("{bogus}")
        message = str(excinfo.value)
        assert message.startswith("Unknown placeholder: {bogus}.")
        assert message.endswith(", ".join(sorted(FilenameFormatter.ALLOWED_KEYS)))

    def test_validation_happens_in_constructor_not_at_format_time(self, frozen_clock):
        # Constructing is what raises...
        with pytest.raises(ValueError):
            FilenameFormatter("{bogus}")
        # ...and nothing re-checks the template afterwards: swapping in an invalid
        # one post-construction formats happily (the unknown key resolves to '',
        # leaving an empty result, which falls back to the default date).
        formatter = FilenameFormatter("{seed}")
        formatter.template = "{bogus}"
        assert formatter.format({"seed": 1}) == DEFAULT_DATE

    @pytest.mark.parametrize("template", ["{seed", "seed}", "{}", "{ seed }", "no placeholders"])
    def test_text_that_is_not_a_placeholder_is_not_validated(self, template):
        # The placeholder regex needs ``{word}`` with no spaces; anything else is
        # literal text and never reaches the allow-list check.
        FilenameFormatter(template)

    def test_double_braces_expand_the_inner_placeholder(self, frozen_clock):
        assert fmt("{{seed}}") == "{12345}"

    def test_spaced_braces_stay_literal(self, frozen_clock):
        # ``{ seed }`` is not a placeholder; the inner spaces become an underscore
        # and the braces survive sanitisation.
        assert fmt("{ seed }") == "{_seed_}"


class TestDatePlaceholder:
    def test_default_format(self, frozen_clock):
        assert fmt("{date}") == DEFAULT_DATE

    def test_clock_is_read_through_time_then_datetime(self, frozen_clock):
        fmt("{date}")
        assert frozen_clock == [FIXED_TIMESTAMP]

    @pytest.mark.parametrize(
        "date_format, expected",
        [
            ("YYYY-MM-DD", "2025-01-15"),
            ("YYYY-MM-DD_HH-mm-ss", "2025-01-15_14-30-45"),
            ("YYYYMMDD", "20250115"),
            ("YY", "25"),
            ("DD.MM.YYYY", "15.01.2025"),
            ("HHhmm", "14h30"),
            ("HH-mm-ss", "14-30-45"),
            ("hh", "02"),  # 12-hour clock -> %I
            ("HH", "14"),  # 24-hour clock -> %H
        ],
    )
    def test_custom_formats(self, frozen_clock, date_format, expected):
        assert fmt("{date(%s)}" % date_format) == expected

    @pytest.mark.parametrize("separator", ["/", ":", " "])
    def test_allowed_date_separators_are_sanitised_out_of_the_result(self, frozen_clock, separator):
        # ``/``, ``:`` and space are accepted by ``_is_valid_date_format`` and reach
        # strftime, but the final whole-result sanitisation rewrites them to ``_``.
        # The module docstring advertising ``{date(YYYY/MM/DD)} -> 2025/01/15`` is
        # therefore stale -- the real output is underscore separated.
        assert fmt("{date(YYYY%sMM%sDD)}" % (separator, separator)) == "2025_01_15"

    @pytest.mark.parametrize("date_format", ["bogus", "YYYY年", "YYYY|MM", "%Y", "YYYY!MM"])
    def test_invalid_format_falls_back_to_the_default(self, frozen_clock, date_format):
        assert fmt("{date(%s)}" % date_format) == DEFAULT_DATE

    def test_empty_arg_yields_an_empty_date(self, frozen_clock):
        # ``{date()}`` validates (nothing left over after token removal) and maps to
        # an empty strftime format, so it contributes nothing to the filename.
        assert fmt("x{date()}x") == "xx"

    def test_empty_arg_alone_falls_back_to_the_default_date(self, frozen_clock):
        # The empty expansion leaves an empty result, which triggers the
        # non-empty-filename fallback rather than the invalid-format fallback.
        assert fmt("{date()}") == DEFAULT_DATE

    def test_numeric_arg_is_not_a_truncation_for_date(self, frozen_clock):
        # ``{date}`` returns from ``_format_date`` before the ``arg.isdigit()``
        # truncation branch is reached, and "50" is not a valid date format, so the
        # whole default timestamp comes back rather than a 50-char cut of it.
        assert fmt("{date(50)}") == DEFAULT_DATE

    def test_separator_only_format_renders_literally(self, frozen_clock):
        # 'h' is in DATE_SEPARATORS, so a format made of nothing but separators
        # validates and strftime hands it straight back.
        assert fmt("{date(h)}") == "h"

    def test_strftime_failure_falls_back_to_the_default_format(self, monkeypatch):
        # The ``except`` arm of ``_format_date`` is otherwise unreachable through the
        # public API: every format that survives ``_is_valid_date_format`` is a legal
        # strftime string on CPython.
        class _ExplodingStrftime:
            def strftime(self, strftime_fmt):
                if strftime_fmt == "%Y-%m-%d-%Hh%Mm%Ss":
                    return DEFAULT_DATE
                raise ValueError("strftime rejected %r" % strftime_fmt)

        class _FrozenDatetime:
            @staticmethod
            def fromtimestamp(timestamp):
                return _ExplodingStrftime()

        monkeypatch.setattr(ff, "time", types.SimpleNamespace(time=lambda: FIXED_TIMESTAMP))
        monkeypatch.setattr(ff, "datetime", _FrozenDatetime)

        # "YYYY-MM-DD" is valid, so the first strftime("%Y-%m-%d") is attempted,
        # raises, and the retry with the default pattern is what we see.
        assert fmt("{date(YYYY-MM-DD)}") == DEFAULT_DATE

    def test_adjacent_month_and_minute_tokens_both_survive(self, frozen_clock):
        # ``_parse_date_format`` substitutes in a single pass. Rewriting tokens
        # sequentially over its own output turned ``MMmm`` into ``%m`` + ``mm``, whose
        # trailing ``mm`` was then rewritten to ``%M`` -- yielding the literal ``%Mm``
        # with the month lost.
        assert fmt("{date(MMmm)}") == "0130"  # January, 30 minutes past

    @pytest.mark.parametrize(
        "date_format, expected",
        [
            ("MMmm", "0130"),   # month then minute
            ("mmMM", "3001"),   # minute then month
            ("MMMM", "0101"),   # the same token twice
            ("HHhh", "1402"),   # 24h then 12h
        ],
    )
    def test_a_token_never_rematches_what_an_earlier_one_wrote(
        self, frozen_clock, date_format, expected
    ):
        assert fmt("{date(%s)}" % date_format) == expected

    @pytest.mark.parametrize(
        "date_format, expected_valid",
        [("", True), ("YYYY", True), ("YYYY-MM-DD", True), ("HHhmm", True), ("YYYY年", False), ("abc", False)],
    )
    def test_is_valid_date_format(self, date_format, expected_valid):
        assert FilenameFormatter("{date}")._is_valid_date_format(date_format) is expected_valid

    @pytest.mark.parametrize(
        "date_format, strftime_format",
        [("YYYY-MM-DD", "%Y-%m-%d"), ("HHhmm", "%Hh%M"), ("YY.ss", "%y.%S"), ("hh:mm", "%I:%M")],
    )
    def test_parse_date_format(self, date_format, strftime_format):
        assert FilenameFormatter("{date}")._parse_date_format(date_format) == strftime_format


class TestValuePlaceholders:
    @pytest.mark.parametrize(
        "template, expected",
        [
            ("{seed}", "12345"),
            ("{resolution}", "1280x720"),
            ("{num_inference_steps}", "30"),
            ("{prompt}", "A_beautiful_sunset_over_the_ocean"),
            ("{flow_shift}", "5.0"),
            ("{video_length}", "81"),
            ("{guidance_scale}", "7.5"),
        ],
    )
    def test_direct_keys(self, frozen_clock, template, expected):
        assert fmt(template) == expected

    def test_alias_map_is_exactly_the_documented_three(self):
        assert FilenameFormatter.KEY_ALIASES == {
            "steps": "num_inference_steps",
            "frames": "video_length",
            "cfg": "guidance_scale",
        }

    @pytest.mark.parametrize(
        "alias, canonical", sorted(FilenameFormatter.KEY_ALIASES.items())
    )
    def test_aliases_read_the_canonical_setting(self, frozen_clock, alias, canonical):
        # Both spellings must produce the same *literal* string; comparing the two
        # calls to each other would also pass if aliases silently resolved to ''.
        expected = ALIAS_EXPECTATIONS[alias]
        assert fmt("{%s}" % alias) == expected
        assert fmt("{%s}" % canonical) == expected

    def test_alias_ignores_a_setting_stored_under_the_alias_name(self, frozen_clock):
        # ``{steps}`` resolves to ``num_inference_steps``; a literal ``steps`` entry
        # in the settings dict is never consulted.
        assert fmt("a{steps}b", {"steps": 30}) == "ab"

    def test_missing_key_becomes_empty_string(self, frozen_clock):
        assert fmt("x{seed}y", {}) == "xy"

    def test_none_value_becomes_empty_string(self, frozen_clock):
        assert fmt("x{seed}y", {"seed": None}) == "xy"

    @pytest.mark.parametrize("value, expected", [(0, "0"), (False, "False"), (-1, "-1"), (0.0, "0.0")])
    def test_falsy_non_none_values_are_stringified(self, frozen_clock, value, expected):
        assert fmt("{seed}", {"seed": value}) == expected

    def test_all_placeholders_empty_falls_back_to_date(self, frozen_clock):
        assert fmt("{seed}{prompt}", {}) == DEFAULT_DATE

    def test_literal_separator_survives_when_values_are_missing(self, frozen_clock):
        # Only underscores and whitespace are collapsed/stripped, so a bare "-"
        # is considered a valid filename and no date fallback happens.
        assert fmt("{seed}-{prompt}", {}) == "-"

    def test_documented_example(self, frozen_clock):
        assert fmt("{date}-{prompt(50)}-{seed}") == (
            "2025-01-15-14h30m45s-A_beautiful_sunset_over_the_ocean-12345"
        )

    def test_settings_dict_is_not_mutated(self, frozen_clock):
        settings = dict(SETTINGS)
        fmt("{date}-{steps}-{cfg}-{frames}", settings)
        assert settings == SETTINGS

    def test_formatter_is_reusable_across_settings(self, frozen_clock):
        formatter = FilenameFormatter("{seed}")
        assert formatter.format({"seed": 1}) == "1"
        assert formatter.format({"seed": 2}) == "2"


class TestTruncation:
    @pytest.mark.parametrize(
        "arg, expected",
        [
            ("10", "A_beautifu"),
            ("1", "A"),
            ("05", "A_bea"),  # leading zeros still parse as an int
            ("50", "A_beautiful_sunset_over_the_ocean"),  # longer than the value
            ("0", "A_beautiful_sunset_over_the_ocean"),  # max_len <= 0 disables truncation
            ("-5", "A_beautiful_sunset_over_the_ocean"),  # not isdigit() -> no truncation
            ("", "A_beautiful_sunset_over_the_ocean"),  # not isdigit() -> no truncation
            ("abc", "A_beautiful_sunset_over_the_ocean"),  # non-numeric arg is ignored
        ],
    )
    def test_prompt_truncation_arguments(self, frozen_clock, arg, expected):
        assert fmt("{prompt(%s)}" % arg) == expected

    def test_truncation_happens_before_sanitisation(self, frozen_clock):
        # "A beautiful"[:2] == "A " -> rstrip -> "A": the cut is made on the raw
        # text, so the trailing space never becomes an underscore.
        assert fmt("{prompt(2)}") == "A"

    def test_truncation_applies_to_non_text_values_too(self, frozen_clock):
        assert fmt("{seed(3)}") == "123"

    @pytest.mark.parametrize(
        "value, max_len, expected",
        [
            ("hello", 3, "hel"),
            ("hello", 5, "hello"),
            ("hello", 99, "hello"),
            ("hello", 0, "hello"),
            ("hello", -1, "hello"),
            ("he llo", 3, "he"),  # trailing whitespace stripped after the cut
            ("", 3, ""),
        ],
    )
    def test_truncate_helper(self, value, max_len, expected):
        assert FilenameFormatter("{prompt}")._truncate(value, max_len) == expected

    def test_the_whole_name_is_capped_not_just_each_placeholder(self, frozen_clock):
        # Without a cap a 5000-character prompt produced a 5000-character name, which
        # fails with ENAMETOOLONG on every common filesystem.
        result = fmt("{prompt}", {"prompt": "x" * 5000})
        assert result == "x" * FilenameFormatter.MAX_FILENAME_LENGTH
        assert len(result) == FilenameFormatter.MAX_FILENAME_LENGTH

    def test_the_cap_counts_bytes_for_multi_byte_text(self, frozen_clock):
        # 200 characters of multi-byte text still overruns the 255-*byte* component
        # limit on ext4, so the byte budget is enforced as well.
        result = fmt("{prompt}", {"prompt": "é" * 5000})
        assert len(result.encode("utf-8")) <= FilenameFormatter.MAX_FILENAME_BYTES
        assert len(result) <= FilenameFormatter.MAX_FILENAME_LENGTH
        assert set(result) == {"é"}

    def test_a_name_under_the_cap_is_untouched(self, frozen_clock):
        assert fmt("{seed}", {"seed": 12345}) == "12345"
        short = "y" * (FilenameFormatter.MAX_FILENAME_LENGTH - 1)
        assert fmt("{prompt}", {"prompt": short}) == short

    def test_the_cut_does_not_leave_a_dangling_separator(self, frozen_clock):
        # The prompt is sanitised to underscore-separated words, so an unlucky cut
        # would otherwise end the filename on "_".
        result = fmt("{prompt}", {"prompt": "ab " * 5000})
        assert not result.endswith("_")


class TestSanitisation:
    @pytest.mark.parametrize("char", list('<>:"/\\|?*'))
    def test_windows_illegal_characters_become_underscores(self, frozen_clock, char):
        assert fmt("{prompt}", {"prompt": "a%sb" % char}) == "a_b"

    @pytest.mark.parametrize("char", ["\x00", "\x1f", "\n", "\r", "\t", "\x07"])
    def test_control_characters_become_underscores(self, frozen_clock, char):
        assert fmt("{prompt}", {"prompt": "a%sb" % char}) == "a_b"

    def test_path_separators_in_literal_template_text_are_removed(self, frozen_clock):
        assert fmt("out/{seed}") == "out_12345"

    def test_backslash_traversal_in_literal_template_text_is_neutralised(self, frozen_clock):
        assert fmt("..\\..\\{seed}") == ".._.._12345"

    def test_a_name_of_only_dots_falls_back_to_the_date(self, frozen_clock):
        # "." and ".." used to come back verbatim, which is not a usable filename on
        # any OS. The trailing-separator trim in _enforce_max_length empties them, and
        # the existing non-empty fallback then supplies a timestamp.
        assert fmt(".") == DEFAULT_DATE
        assert fmt("..") == DEFAULT_DATE

    def test_a_trailing_dot_is_trimmed(self, frozen_clock):
        # Windows rejects a name ending in '.', and it is invisible to the user.
        assert fmt("{prompt}", {"prompt": "report."}) == "report"

    def test_dots_inside_a_name_are_preserved(self, frozen_clock):
        assert fmt("{prompt}", {"prompt": "v1.2.3"}) == "v1.2.3"

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("a b", "a_b"),
            ("a__b", "a_b"),
            ("a   b", "a_b"),
            ("a _ b", "a_b"),
            ("a\t \nb", "a_b"),
            ("_leading", "leading"),
            ("trailing_", "trailing"),
            ("  padded  ", "padded"),
            ("__both__", "both"),
        ],
    )
    def test_underscore_and_whitespace_runs_collapse_and_strip(self, frozen_clock, value, expected):
        assert fmt("{prompt}", {"prompt": value}) == expected

    def test_delete_character_is_not_sanitised(self, frozen_clock):
        # GAP (pinned, not fixed): UNSAFE_FILENAME_CHARS covers \x00-\x1f but stops
        # short of \x7f (DEL), so that control character reaches the filename.
        assert fmt("{prompt}", {"prompt": "a\x7fb"}) == "a\x7fb"

    @pytest.mark.parametrize("value, expected", [("", ""), (None, ""), (12, "12"), ("a/b", "a_b")])
    def test_sanitize_helper(self, value, expected):
        assert FilenameFormatter("{prompt}")._sanitize_for_filename(value) == expected

    @pytest.mark.parametrize("value", [0, False, 0.0])
    def test_sanitize_helper_swallows_falsy_non_strings(self, value):
        # QUIRK (pinned): the ``if not value`` guard is applied *before* ``str()``, so
        # a falsy non-string is turned into '' instead of "0"/"False". ``format`` is
        # unaffected because it stringifies first -- see
        # TestValuePlaceholders.test_falsy_non_none_values_are_stringified.
        assert FilenameFormatter("{prompt}")._sanitize_for_filename(value) == ""

    def test_hyphens_and_dots_are_preserved(self, frozen_clock):
        assert fmt("-{seed}-", {"seed": 1}) == "-1-"
        assert fmt("{guidance_scale}") == "7.5"

    def test_literal_spaces_around_a_placeholder_collapse(self, frozen_clock):
        assert fmt("{seed} - {seed}", {"seed": 7}) == "7_-_7"


class TestEmptyTemplateFallback:
    @pytest.mark.parametrize("template", ["", "   ", "___", "_ _ _", "\t"])
    def test_templates_that_sanitise_to_nothing_fall_back_to_the_date(self, frozen_clock, template):
        assert fmt(template) == DEFAULT_DATE

    def test_fallback_fires_after_literal_text_is_stripped(self, frozen_clock):
        # The empty date expansion leaves "__", which sanitises to '' -- so the
        # fallback re-formats with the *default* pattern rather than the (empty)
        # one the template asked for.
        assert fmt("_{date()}_") == DEFAULT_DATE


class TestUnicode:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("café über naïve", "café_über_naïve"),
            ("一只猫 🐱", "一只猫_🐱"),
            ("Ω≈ç√", "Ω≈ç√"),
            ("emoji🚀only", "emoji🚀only"),
        ],
    )
    def test_unicode_characters_are_preserved(self, frozen_clock, value, expected):
        assert fmt("{prompt}", {"prompt": value}) == expected

    def test_unicode_whitespace_is_collapsed(self, frozen_clock):
        # ``\s`` is unicode-aware for str patterns, so NBSP and ideographic space
        # collapse into a single underscore just like ASCII whitespace.
        assert fmt("{prompt}", {"prompt": "a\xa0　b"}) == "a_b"

    def test_truncation_counts_characters_not_bytes(self, frozen_clock):
        assert fmt("{prompt(3)}", {"prompt": "日本語のテキスト"}) == "日本語"


def _help_text_examples():
    """The template lines from the ``Examples:`` block of the help text."""

    body = FilenameFormatter.get_help_text().split("Examples:", 1)[1]
    return [line.strip() for line in body.splitlines() if line.strip().startswith("{")]


class TestHelpText:
    @pytest.mark.parametrize("key", sorted(FilenameFormatter.ALLOWED_KEYS))
    def test_help_text_documents_every_allowed_key(self, key):
        assert "{%s}" % key in FilenameFormatter.get_help_text()

    def test_help_text_lists_the_expected_examples(self):
        assert _help_text_examples() == [
            "{date}-{prompt(50)}-{seed}",
            "{date(YYYYMMDD)}_{resolution}_{steps}steps",
            "{date(YYYY-MM-DD_HH-mm-ss)}_{seed}",
            "{date(DD.MM.YYYY)}_{prompt(30)}",
        ]

    @pytest.mark.parametrize("template, expected", [
        ("{date}-{prompt(50)}-{seed}", "2025-01-15-14h30m45s-A_beautiful_sunset_over_the_ocean-12345"),
        ("{date(YYYYMMDD)}_{resolution}_{steps}steps", "20250115_1280x720_30steps"),
        ("{date(YYYY-MM-DD_HH-mm-ss)}_{seed}", "2025-01-15_14-30-45_12345"),
        # the 30-char cut lands mid-word: "...over the oc"
        ("{date(DD.MM.YYYY)}_{prompt(30)}", "15.01.2025_A_beautiful_sunset_over_the_oc"),
    ])
    def test_documented_examples_produce_the_documented_shape(self, frozen_clock, template, expected):
        assert fmt(template) == expected


class TestFormatFilenameClassmethod:
    def test_matches_the_instance_api(self, frozen_clock):
        template = "{date(YYYYMMDD)}_{resolution}_{steps}steps"
        expected = "20250115_1280x720_30steps"
        # Pinned against a literal as well as against each other, so a mutation that
        # broke *both* paths identically would still be caught.
        assert FilenameFormatter.format_filename(template, SETTINGS) == expected
        assert FilenameFormatter(template).format(SETTINGS) == expected

    def test_propagates_validation_errors(self):
        with pytest.raises(ValueError, match="Unknown placeholder"):
            FilenameFormatter.format_filename("{nope}", SETTINGS)

    def test_result_never_contains_a_path_separator(self, frozen_clock):
        messy = {
            "prompt": "a/b\\c:d",
            "seed": "1/2",
            "resolution": "12/34",
            "num_inference_steps": "3\\4",
            "flow_shift": "5:6",
            "video_length": "7|8",
            "guidance_scale": "9*0",
        }
        result = fmt("{date}/{prompt}/{seed}/{resolution}/{steps}/{cfg}", messy)
        assert "/" not in result and "\\" not in result
        # Pinned exactly: every separator (literal or from a value) becomes '_' and
        # the resulting runs collapse, so nothing survives as a directory boundary.
        assert result == "2025-01-15-14h30m45s_a_b_c_d_1_2_12_34_3_4_9_0"
