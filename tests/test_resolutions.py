"""Tests for ``shared/resolutions.py`` and ``shared/match_archi.py``.

``shared/resolutions.py`` owns everything the UI knows about resolutions: parsing
``"WIDTHxHEIGHT"`` strings, loading the optional user ``resolutions.json``, aligning
dimensions onto a VAE block grid, bucketing a resolution into a ``256p``..``2160p``
group, filtering those groups with model-supplied expressions, and picking the
closest available resolution when the user switches model.

``shared/match_archi.py`` is a tiny expression evaluator (``'>=70&<90'``,
``'<=50+>89'``) originally written to select an attention backend from an Nvidia
compute-capability number; ``resolutions.py`` reuses it to evaluate resolution
category expressions, so both call sites are covered here.

Covered: ``is_resolution_value``, ``parse_resolution``,
``normalize_resolution_choices``, ``load_custom_resolution_choices`` /
``reset_custom_resolution_cache``, ``dedupe_resolution_choices``, the block
alignment family, ``builtin_resolution_choices`` and friends,
``categorize_resolution``, the category-expression helpers,
``closest_resolution``, ``resolve_resolution_choices``,
``resolve_model_switch_resolution``, the grouping helpers,
``keep_resolution_on_model_switch_enabled`` and ``match_nvidia_architecture``.

Expectations were derived by reading the source. A few surprising-but-real
behaviours (division by zero on a zero-height resolution, the sticky custom
resolution cache, ``""`` as a category expression rejecting everything) are pinned
explicitly and flagged with a comment.
"""

from __future__ import annotations

import json

import pytest

import shared.resolutions as res
from shared.match_archi import match_nvidia_architecture


@pytest.fixture(autouse=True)
def no_custom_resolutions(monkeypatch):
    """Pre-seed the custom-resolution cache so nothing ever touches the disk.

    ``load_custom_resolution_choices`` defaults to reading ``resolutions.json``
    *relative to the current working directory*; seeding the cache with an empty
    list short-circuits that and keeps every test independent of where pytest was
    started from.  ``monkeypatch`` restores the module global afterwards.
    """

    monkeypatch.setattr(res, "_custom_resolutions", [])


@pytest.fixture
def cleared_resolution_cache(monkeypatch):
    """Undo the autouse seeding for the tests that exercise the loader itself."""

    monkeypatch.setattr(res, "_custom_resolutions", None)


@pytest.fixture
def printed():
    """Collector standing in for the module's ``printer=print`` default."""

    return []


BUILTIN = res.DEFAULT_RESOLUTION_CHOICES


class TestIsResolutionValue:
    @pytest.mark.parametrize(
        "value",
        [
            "1280x720",
            "1280X720",  # the "x" separator is case insensitive
            " 1280x720 ",  # surrounding whitespace is stripped
            "12X34\n",
            "0x0",  # syntactically valid even though it is nonsense
            "9x9",
        ],
    )
    def test_accepts_well_formed_values(self, value):
        assert res.is_resolution_value(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "x",
            "1280",
            "1280x",
            "1280 x 720",  # inner whitespace is *not* tolerated
            "12x34x56",
            "-12x34",  # a minus sign is not part of the pattern
            "1280x720p",
            "1280*720",
            None,
            123,
            b"1280x720",
        ],
    )
    def test_rejects_malformed_values(self, value):
        assert res.is_resolution_value(value) is False


class TestParseResolution:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1280x720", (1280, 720)),
            ("720x1280", (720, 1280)),
            (" 1280X720 ", (1280, 720)),
            ("0x0", (0, 0)),
        ],
    )
    def test_parses_dimensions(self, value, expected):
        assert res.parse_resolution(value) == expected

    @pytest.mark.parametrize("value", ["abc", "1280", "1280x", "12x34x56"])
    def test_malformed_values_raise_value_error(self, value):
        # parse_resolution does no validation of its own: callers are expected to
        # gate it behind is_resolution_value.
        with pytest.raises(ValueError):
            res.parse_resolution(value)


class TestNormalizeResolutionChoices:
    def test_none_passes_through(self, printed):
        assert res.normalize_resolution_choices(None, "src", printed.append) is None
        assert printed == []

    def test_empty_list_normalizes_to_empty_list(self, printed):
        assert res.normalize_resolution_choices([], "src", printed.append) == []
        assert printed == []

    def test_accepts_lists_and_tuples_and_lowercases_the_value(self, printed):
        assert res.normalize_resolution_choices(
            [["A", "1280X720"], ("B", "640x480")], "src", printed.append
        ) == [("A", "1280x720"), ("B", "640x480")]
        assert printed == []

    def test_non_list_input_is_rejected_with_a_message(self, printed):
        assert res.normalize_resolution_choices({"a": 1}, "src", printed.append) is None
        assert printed == ['"src" should be a list of 2 elements lists ["Label","WxH"]']

    @pytest.mark.parametrize(
        "entry",
        [
            ["A"],  # wrong arity
            ["A", "1280x720", "extra"],
            ["A", 720],  # value not a string
            [720, "1280x720"],  # label not a string
            "1280x720",  # not a pair at all
        ],
    )
    def test_invalid_pairs_are_rejected(self, entry, printed):
        assert res.normalize_resolution_choices([entry], "src", printed.append) is None
        assert printed == [f'"src" contains an invalid list of two elements: {entry}']

    def test_bad_resolution_string_rejects_the_whole_list(self, printed):
        # One bad entry discards every other entry, good ones included.
        assert (
            res.normalize_resolution_choices(
                [["A", "1280x720"], ["B", "oops"]], "src", printed.append
            )
            is None
        )
        assert printed == ['"src" contains a resolution value that is not in the format "WxH": oops']

    def test_default_printer_writes_to_stdout(self, capsys):
        assert res.normalize_resolution_choices("nope", "model.resolutions") is None
        assert "model.resolutions" in capsys.readouterr().out


class TestCustomResolutionFile:
    def test_missing_file_returns_empty_and_does_not_cache(self, tmp_path, cleared_resolution_cache):
        missing = str(tmp_path / "absent.json")
        assert res.load_custom_resolution_choices(missing) == []
        assert res._custom_resolutions is None

    def test_valid_file_is_loaded_and_cached(self, tmp_path, cleared_resolution_cache):
        path = tmp_path / "resolutions.json"
        path.write_text(json.dumps([["Wide", "1280X720"], ["Tall", "720x1280"]]), encoding="utf-8")

        loaded = res.load_custom_resolution_choices(str(path))
        assert loaded == [("Wide", "1280x720"), ("Tall", "720x1280")]
        assert res._custom_resolutions == loaded

    def test_cache_wins_over_the_file_argument(self, tmp_path, cleared_resolution_cache):
        first = tmp_path / "first.json"
        first.write_text(json.dumps([["Wide", "1280x720"]]), encoding="utf-8")
        second = tmp_path / "second.json"
        second.write_text(json.dumps([["Square", "512x512"]]), encoding="utf-8")

        assert res.load_custom_resolution_choices(str(first)) == [("Wide", "1280x720")]
        # The cache is global and keyed on nothing: the second path is ignored.
        assert res.load_custom_resolution_choices(str(second)) == [("Wide", "1280x720")]

        res.reset_custom_resolution_cache()
        assert res._custom_resolutions is None
        assert res.load_custom_resolution_choices(str(second)) == [("Square", "512x512")]

    def test_unparseable_json_is_reported_and_yields_empty(self, tmp_path, cleared_resolution_cache, printed):
        path = tmp_path / "broken.json"
        path.write_text("{ not json", encoding="utf-8")

        assert res.load_custom_resolution_choices(str(path), printed.append) == []
        assert len(printed) == 1
        assert printed[0].startswith(f'Invalid "{path}" :')
        assert res._custom_resolutions is None

    def test_structurally_invalid_json_yields_empty(self, tmp_path, cleared_resolution_cache, printed):
        path = tmp_path / "bad_shape.json"
        path.write_text(json.dumps([["Label", "1280*720"]]), encoding="utf-8")

        assert res.load_custom_resolution_choices(str(path), printed.append) == []
        assert res._custom_resolutions is None


class TestDedupeResolutionChoices:
    def test_first_label_wins_for_a_repeated_resolution(self):
        assert res.dedupe_resolution_choices(
            [("A", "1x1"), ("B", "1x1"), ("C", "2x2")]
        ) == [("A", "1x1"), ("C", "2x2")]

    def test_empty_input(self):
        assert res.dedupe_resolution_choices([]) == []

    def test_accepts_any_iterable(self):
        assert res.dedupe_resolution_choices(iter([("A", "1x1")])) == [("A", "1x1")]


class TestBlockAlignment:
    @pytest.mark.parametrize("raw,expected", [(16, 16), ("32", 32), (16.9, 16), (1, 1)])
    def test_normalize_block_size_coerces_to_int(self, raw, expected):
        assert res.normalize_block_size(raw) == expected

    @pytest.mark.parametrize("raw", [0, -8, "abc"])
    def test_normalize_block_size_rejects_non_positive_and_garbage(self, raw):
        with pytest.raises(ValueError):
            res.normalize_block_size(raw)

    @pytest.mark.parametrize(
        "value,block,expected",
        [
            (720, 16, 720),
            (721, 16, 720),
            (1000, 16, 992),
            (100, 16, 96),
            (8, 16, 16),  # rounds *up* to one block rather than down to zero
            (0, 16, 16),
            (-5, 16, 16),
            (1000, 1, 1000),  # block sizes <= 1 disable alignment entirely
            (1000, 0, 1000),
        ],
    )
    def test_align_dimension_to_block(self, value, block, expected):
        assert res.align_dimension_to_block(value, block) == expected

    @pytest.mark.parametrize(
        "resolution,block,expected",
        [
            ("1280x720", 16, "1280x720"),
            ("1280x720", 32, "1280x704"),
            ("1000x100", 16, "992x96"),
            ("8x8", 16, "16x16"),
            ("1281X721", 16, "1280x720"),
        ],
    )
    def test_align_resolution_value(self, resolution, block, expected):
        assert res.align_resolution_value(resolution, block) == expected

    def test_align_resolution_label_returns_label_unchanged_when_nothing_moved(self):
        assert res.align_resolution_label("1280x720 (16:9)", "1280x720", "1280x720") == "1280x720 (16:9)"

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("1280x720 (16:9)", "1280x704 (16:9)"),
            ("1280X720 label", "1280x704 label"),  # match is case insensitive
            ("HD (1280x720) 1280x720", "HD (1280x704) 1280x720"),  # only the first hit
        ],
    )
    def test_align_resolution_label_substitutes_the_exact_resolution(self, label, expected):
        assert res.align_resolution_label(label, "1280x720", "1280x704") == expected

    def test_align_resolution_label_falls_back_to_any_wxh_token(self):
        assert res.align_resolution_label("legacy 640x480 preset", "1280x720", "1280x704") == (
            "legacy 1280x704 preset"
        )

    def test_align_resolution_label_leaves_labels_without_a_token_alone(self):
        # "1280 x 720" is not a \d+x\d+ token, so nothing is rewritten.
        assert res.align_resolution_label("HD 1280 x 720", "1280x720", "1280x704") == "HD 1280 x 720"

    def test_align_resolution_choices_rewrites_values_and_labels(self):
        choices = [("A", "1280x720"), ("B", "1279x719"), ("C", "640x480")]
        assert res.align_resolution_choices(choices, 16) == [
            ("A", "1280x720"),
            ("B", "1264x704"),
            ("C", "640x480"),
        ]

    def test_align_resolution_choices_dedupes_collisions(self):
        choices = [("A", "1280x720"), ("B", "1281x721")]
        assert res.align_resolution_choices(choices, 16) == [("A", "1280x720")]

    def test_align_resolution_choices_is_a_no_op_for_block_size_one(self):
        choices = [("A", "1280x720"), ("B", "1280x720")]
        # Block size 1 short-circuits before the dedupe, so duplicates survive.
        assert res.align_resolution_choices(choices, 1) is choices

    def test_align_resolution_choices_rejects_a_zero_block_size(self):
        with pytest.raises(ValueError):
            res.align_resolution_choices([("A", "1280x720")], 0)

    def test_alignment_of_the_builtin_list_keeps_labels_in_sync(self):
        aligned = dict(
            (resolution, label) for label, resolution in res.align_resolution_choices(BUILTIN, 32)
        )
        assert aligned["1280x704"] == "1280x704 (16:9)"
        assert aligned["704x1280"] == "704x1280 (9:16)"


class TestBuiltinChoices:
    def test_every_builtin_entry_is_a_valid_pair(self):
        for label, resolution in res.DEFAULT_RESOLUTION_CHOICES_4K + res.DEFAULT_RESOLUTION_CHOICES:
            assert res.is_resolution_value(resolution)
            assert label.startswith(resolution)

    def test_builtin_entries_are_unique(self):
        combined = res.DEFAULT_RESOLUTION_CHOICES_4K + res.DEFAULT_RESOLUTION_CHOICES
        assert len(res.dedupe_resolution_choices(combined)) == len(combined)

    def test_builtin_choices_default_to_no_4k(self):
        assert res.builtin_resolution_choices() == list(res.DEFAULT_RESOLUTION_CHOICES)

    def test_builtin_choices_prepend_4k_when_requested(self):
        assert res.builtin_resolution_choices(include_4k=True) == (
            list(res.DEFAULT_RESOLUTION_CHOICES_4K) + list(res.DEFAULT_RESOLUTION_CHOICES)
        )

    def test_builtin_choices_returns_a_fresh_list(self):
        first = res.builtin_resolution_choices()
        first.append(("X", "1x1"))
        assert res.builtin_resolution_choices() == list(res.DEFAULT_RESOLUTION_CHOICES)

    def test_all_global_choices_always_include_4k(self):
        assert res.all_global_resolution_choices() == res.builtin_resolution_choices(include_4k=True)

    def test_default_global_choices_follow_the_flag(self):
        assert res.default_global_resolution_choices(False) == list(res.DEFAULT_RESOLUTION_CHOICES)
        assert res.default_global_resolution_choices(True) == res.all_global_resolution_choices()

    def test_custom_choices_are_appended_and_deduped(self, monkeypatch):
        monkeypatch.setattr(
            res, "_custom_resolutions", [("Mine", "1234x576"), ("Dup", "1280x720")]
        )
        choices = res.default_global_resolution_choices(False)
        assert ("Mine", "1234x576") in choices
        # "1280x720" is already builtin, so the custom label loses.
        assert ("Dup", "1280x720") not in choices
        assert ("1280x720 (16:9)", "1280x720") in choices


class TestCategorizeResolution:
    @pytest.mark.parametrize(
        "resolution,group",
        [
            ("0x0", "256p"),
            ("320x320", "256p"),
            ("448x256", "256p"),  # exactly on the 256p threshold
            ("448x257", "320p"),
            ("448x448", "320p"),
            ("512x512", "384p"),  # exactly on the 384p threshold
            ("672x384", "384p"),
            ("832x480", "480p"),
            ("832x624", "480p"),  # exactly on the 480p threshold
            ("960x544", "540p"),
            ("544x960", "540p"),  # portrait lands in the same group
            ("1024x1024", "720p"),
            ("1280x720", "720p"),
            ("1600x400", "720p"),  # grouping is by pixel count, not by height
            ("1088x1088", "1080p"),
            ("1920x1088", "1080p"),
            ("1920x1920", "1440p"),
            ("2560x1440", "1440p"),
            ("3840x2176", "2160p"),
            ("9999x9999", "2160p"),  # anything above the top threshold clamps to 2160p
        ],
    )
    def test_group_for_resolution(self, resolution, group):
        assert res.categorize_resolution(resolution) == group

    def test_orientation_does_not_change_the_group(self):
        for label, resolution in BUILTIN:
            width, height = res.parse_resolution(resolution)
            flipped = f"{height}x{width}"
            assert res.categorize_resolution(flipped) == res.categorize_resolution(resolution)

    def test_every_group_name_has_a_tier(self):
        assert set(res.GROUP_THRESHOLDS) == set(res.GROUP_TIERS)

    def test_malformed_resolution_raises(self):
        with pytest.raises(ValueError):
            res.categorize_resolution("nope")


class TestCategoryExpressions:
    @pytest.mark.parametrize(
        "category,tier",
        [("4k", 2160), ("2K", 1440), ("720p", 720), ("720", 720), (" 1080P ", 1080), (720, 720)],
    )
    def test_category_tier_recognised(self, category, tier):
        assert res._category_tier(category) == tier

    @pytest.mark.parametrize("category", ["999", "999p", "abc", "", "8k", None])
    def test_category_tier_unknown(self, category):
        assert res._category_tier(category) is None

    @pytest.mark.parametrize(
        "expression,normalized",
        [
            (">=720p", ">=720"),
            (">=2k&<4k", ">=1440&<2160"),
            ("480p+720p", "480+720"),
            (" <=1080P ", "<=1080"),
            ("hd", "hd"),  # unknown tokens are left alone
            (">=720", ">=720"),
        ],
    )
    def test_normalize_category_expression(self, expression, normalized):
        assert res._normalize_category_expression(expression) == normalized

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, []),
            ("720p", ["720p"]),
            (["720p", "1080p"], ["720p", "1080p"]),
            (("720p", 1080), ["720p", "1080"]),
            (5, []),  # anything else degrades to "no constraint"
            ({"720p": True}, []),
        ],
    )
    def test_normalize_category_expressions(self, raw, expected):
        assert res.normalize_category_expressions(raw) == expected

    @pytest.mark.parametrize(
        "category,expressions,allowed",
        [
            ("720p", None, True),
            ("720p", [], True),
            ("720p", 5, True),  # unusable input means "allow everything"
            ("720p", ">=720", True),
            ("720p", ">=720p", True),
            ("480p", ">=720p", False),
            ("2160p", "<=4k", True),
            ("1440p", "2k", True),
            ("1440p", "4k", False),
            ("720p", "<=480p+>=720p", True),
            ("540p", "<=480p+>=720p", False),
            ("720p", ">=480p&<=1080p", True),
            ("1440p", ">=480p&<=1080p", False),
            ("720p", ["<=480p", ">=1080p"], False),
            ("720p", ["<=480p", "720p"], True),
            ("720p", "hd", False),  # an unparseable expression matches nothing
            ("720p", "", False),  # ... and so does the empty string
        ],
    )
    def test_category_allowed(self, category, expressions, allowed):
        assert res.category_allowed(category, expressions) is allowed

    def test_category_allowed_requires_a_known_group_name(self):
        with pytest.raises(KeyError):
            res.category_allowed("9999p", ">=720")

    def test_filter_by_categories_keeps_matching_groups(self):
        choices = [("a", "832x480"), ("b", "480x832"), ("c", "1280x720")]
        assert res.filter_resolution_choices_by_categories(choices, ">=720p") == [("c", "1280x720")]

    def test_filter_without_expressions_keeps_everything(self):
        choices = [("a", "832x480"), ("c", "1280x720")]
        assert res.filter_resolution_choices_by_categories(choices, None) == choices

    def test_filter_the_global_list_by_an_upper_bound(self):
        filtered = res.filter_resolution_choices_by_categories(
            res.all_global_resolution_choices(), "<=720p"
        )
        groups = {res.categorize_resolution(resolution) for _, resolution in filtered}
        assert groups == {"256p", "320p", "384p", "480p", "540p", "720p"}


class TestClosestResolution:
    def test_empty_choices_return_the_target_untouched(self):
        assert res.closest_resolution("123x456", []) == "123x456"

    @pytest.mark.parametrize("target", [None, "", "nope", "1280 x 720"])
    def test_unparseable_target_falls_back_to_the_first_choice(self, target):
        assert res.closest_resolution(target, BUILTIN) == BUILTIN[0][1]

    @pytest.mark.parametrize(
        "target,expected",
        [
            ("1280x720", "1280x720"),  # exact match survives
            ("1280x718", "1280x720"),
            ("1920x1080", "1920x1088"),
            ("1080x1920", "1088x1920"),  # portrait stays portrait
            ("640x480", "832x624"),
            ("480x640", "624x832"),
            ("2560x1080", "1920x832"),  # ultrawide keeps its aspect ratio
            ("100x100", "320x320"),  # tiny targets fall to the smallest group
            ("3840x2160", "1920x1088"),  # 4K is absent, so the nearest group wins
            ("9999x9999", "1440x1440"),
        ],
    )
    def test_closest_from_the_builtin_list(self, target, expected):
        assert res.closest_resolution(target, BUILTIN) == expected

    def test_aspect_ratio_beats_pixel_count_within_a_group(self):
        # Both candidates sit in the 1080p group. "1440x1440" matches the target
        # pixel count exactly, yet the 16:9 candidate wins on aspect ratio.
        choices = [("square", "1440x1440"), ("wide", "1920x1088")]
        assert res.closest_resolution("1920x1080", choices) == "1920x1088"

    def test_group_proximity_is_considered_before_aspect_ratio(self):
        # "640x360" has the exact target ratio but lives four groups away, so the
        # square 720p candidate is preferred.
        choices = [("square", "1024x1024"), ("wide", "640x360")]
        assert res.closest_resolution("1920x1080", choices) == "1024x1024"

    def test_pixel_count_breaks_ties_between_equal_ratios(self):
        choices = [("small", "1440x810"), ("big", "1920x1080")]
        assert res.closest_resolution("1760x990", choices) == "1920x1080"

    def test_zero_height_target_raises_zero_division(self):
        # BUG (pinned): closest_resolution computes width/height without guarding
        # against a zero height, so a syntactically valid "800x0" blows up rather
        # than falling back to a sensible choice.
        with pytest.raises(ZeroDivisionError):
            res.closest_resolution("800x0", BUILTIN)


class TestResolveResolutionChoices:
    def test_defaults_when_the_model_says_nothing(self):
        choices, current = res.resolve_resolution_choices(None, {})
        assert choices == list(res.DEFAULT_RESOLUTION_CHOICES)
        assert current == res.DEFAULT_RESOLUTION_CHOICES[0][1]

    def test_4k_flag_extends_the_default_list(self):
        choices, current = res.resolve_resolution_choices(None, {}, enable_4k_resolutions=True)
        assert choices[0] == res.DEFAULT_RESOLUTION_CHOICES_4K[0]
        assert current == res.DEFAULT_RESOLUTION_CHOICES_4K[0][1]

    def test_model_resolutions_replace_the_global_list(self):
        model_def = {"resolutions": [["A", "1280x720"], ["B", "640x480"]]}
        choices, current = res.resolve_resolution_choices("1280x720", model_def)
        assert choices == [("A", "1280x720"), ("B", "640x480")]
        assert current == "1280x720"

    def test_unavailable_current_resolution_snaps_to_the_closest(self):
        model_def = {"resolutions": [["A", "1280x720"], ["B", "640x480"]]}
        _, current = res.resolve_resolution_choices("999x999", model_def)
        assert current == "1280x720"

    def test_invalid_model_resolutions_yield_no_choices(self, capsys):
        choices, current = res.resolve_resolution_choices("1280x720", {"resolutions": "garbage"})
        assert (choices, current) == ([], None)
        assert "model.resolutions" in capsys.readouterr().out

    def test_model_categories_filter_the_global_list(self):
        choices, current = res.resolve_resolution_choices(None, {"resolutions_categories": "<=480p"})
        groups = {res.categorize_resolution(resolution) for _, resolution in choices}
        assert groups == {"256p", "320p", "384p", "480p"}
        assert current == choices[0][1]

    def test_model_resolutions_and_categories_are_merged(self):
        model_def = {"resolutions": [["Native", "1280x720"]], "resolutions_categories": "256p"}
        choices, current = res.resolve_resolution_choices(None, model_def)
        assert choices[0] == ("Native", "1280x720")
        assert {res.categorize_resolution(r) for _, r in choices[1:]} == {"256p"}
        assert current == "1280x720"

    def test_vae_block_size_aligns_the_list(self):
        choices, current = res.resolve_resolution_choices("1280x720", {"vae_block_size": 32})
        assert all(
            w % 32 == 0 and h % 32 == 0
            for w, h in (res.parse_resolution(r) for _, r in choices)
        )
        assert current == "1280x704"

    def test_explicit_block_size_overrides_the_model_value(self):
        _, current = res.resolve_resolution_choices(
            "1280x720", {"vae_block_size": 16}, block_size=32
        )
        assert current == "1280x704"

    def test_default_block_size_is_sixteen(self):
        choices, _ = res.resolve_resolution_choices(None, {})
        assert all(
            w % 16 == 0 and h % 16 == 0
            for w, h in (res.parse_resolution(r) for _, r in choices)
        )

    def test_zero_block_size_is_rejected(self):
        with pytest.raises(ValueError):
            res.resolve_resolution_choices("1280x720", {"vae_block_size": 0})

    def test_unparseable_current_choice_falls_back_to_the_first_entry(self):
        _, current = res.resolve_resolution_choices("garbage", {})
        assert current == res.DEFAULT_RESOLUTION_CHOICES[0][1]


class TestResolveModelSwitchResolution:
    @pytest.mark.parametrize("source", [None, "", "junk", "1280 x 720"])
    def test_invalid_source_returns_none(self, source):
        assert res.resolve_model_switch_resolution(source, {}) is None

    def test_supported_resolution_is_kept(self):
        assert res.resolve_model_switch_resolution("1280x720", {}) == "1280x720"

    def test_resolution_is_realigned_for_the_target_model(self):
        assert res.resolve_model_switch_resolution("1280x720", {"vae_block_size": 32}) == "1280x704"

    def test_resolution_snaps_into_a_restricted_model_list(self):
        target = {"resolutions": [["Only", "512x512"]]}
        assert res.resolve_model_switch_resolution("1280x720", target) == "512x512"

    def test_4k_source_downgrades_when_4k_is_disabled(self):
        assert res.resolve_model_switch_resolution("3840x2176", {}) == "1920x1088"
        assert (
            res.resolve_model_switch_resolution("3840x2176", {}, enable_4k_resolutions=True)
            == "3840x2176"
        )

    def test_zero_height_source_propagates_the_zero_division(self):
        # BUG (pinned): is_resolution_value accepts "800x0", so the guard at the top
        # of resolve_model_switch_resolution lets it through into closest_resolution.
        with pytest.raises(ZeroDivisionError):
            res.resolve_model_switch_resolution("800x0", {})


class TestGrouping:
    def test_groups_are_listed_largest_first(self):
        groups, choices, selected = res.group_resolution_choices(BUILTIN, None)
        assert groups == ["1080p", "720p", "540p", "480p", "384p", "320p", "256p"]
        assert selected == "1080p"
        assert all(res.categorize_resolution(r) == "1080p" for _, r in choices)

    def test_selected_resolution_picks_its_own_group(self):
        _, choices, selected = res.group_resolution_choices(BUILTIN, "832x480")
        assert selected == "480p"
        assert choices == res.group_choices(BUILTIN, "480p")

    def test_selection_outside_the_available_groups_falls_back_to_the_largest(self):
        subset = [("a", "832x480"), ("b", "480x832"), ("c", "1280x720")]
        groups, choices, selected = res.group_resolution_choices(subset, "1920x1088")
        assert groups == ["720p", "480p"]
        assert selected == "720p"
        assert choices == [("c", "1280x720")]

    def test_empty_choices(self):
        assert res.group_resolution_choices([], None) == ([], [], None)
        assert res.group_resolution_choices([], "1280x720") == ([], [], None)

    def test_malformed_selection_raises(self):
        with pytest.raises(ValueError):
            res.group_resolution_choices(BUILTIN, "abc")

    def test_group_choices_filters_by_group(self):
        assert res.group_choices(BUILTIN, "540p") == [
            ("960x544 (16:9)", "960x544"),
            ("544x960 (9:16)", "544x960"),
        ]

    def test_group_choices_for_an_absent_group(self):
        assert res.group_choices([("a", "832x480")], "1080p") == []

    def test_every_group_partition_is_complete(self):
        groups, _, _ = res.group_resolution_choices(BUILTIN, None)
        rebuilt = [choice for group in groups for choice in res.group_choices(BUILTIN, group)]
        assert sorted(rebuilt) == sorted(BUILTIN)


class TestRememberLastResolution:
    def test_stores_under_the_group_and_returns_the_mapping(self):
        store: dict[str, str] = {}
        returned = res.remember_last_resolution(store, "1280x720")
        assert returned is store
        assert store == {"720p": "1280x720"}

    def test_later_values_overwrite_the_same_group(self):
        store = {"720p": "1280x720"}
        res.remember_last_resolution(store, "1024x1024")
        res.remember_last_resolution(store, "832x480")
        assert store == {"720p": "1024x1024", "480p": "832x480"}


class TestKeepResolutionOnModelSwitch:
    @pytest.mark.parametrize("value", ["0", "false", "FALSE", " no ", "off", "No"])
    def test_falsey_strings_disable(self, value):
        assert res.keep_resolution_on_model_switch_enabled(value) is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "", " ", "None"])
    def test_every_other_string_enables(self, value):
        assert res.keep_resolution_on_model_switch_enabled(value) is True

    @pytest.mark.parametrize(
        "value,expected",
        [
            (True, True),
            (False, False),
            (0, False),
            (0.0, False),
            (1, True),
            ([], True),
            # None means "unset", which the module treats as enabled.
            (None, True),
        ],
    )
    def test_non_string_values(self, value, expected):
        assert res.keep_resolution_on_model_switch_enabled(value) is expected


class TestMatchNvidiaArchitecture:
    """``match_nvidia_architecture(conditions_dict, architecture)``.

    ``architecture`` is the compute capability as ``major * 10 + minor`` (see
    ``get_overridden_attention`` in ``wgp.py``): GTX 1080 Ti = 61, RTX 3060 = 86,
    A100 = 80, RTX 4090 = 89, H100 = 90.  The return value is the list of dict
    *values* whose condition matched, in dict order.
    """

    ADA = 89  # RTX 4090
    AMPERE_CONSUMER = 86  # RTX 3060
    AMPERE_DATACENTER = 80  # A100
    HOPPER = 90  # H100
    PASCAL = 61  # GTX 1080 Ti

    @pytest.mark.parametrize(
        "condition,arch,matches",
        [
            # equality, with and without the explicit "="
            ("89", ADA, True),
            ("=89", ADA, True),
            ("89", HOPPER, False),
            ("089", ADA, True),  # leading zeros are fine
            # comparisons
            ("<89", AMPERE_CONSUMER, True),
            ("<89", ADA, False),
            (">=75", ADA, True),
            (">=75", PASCAL, False),
            (">90", HOPPER, False),
            ("<=61", PASCAL, True),
            # OR via '+'
            ("<=50+>89", HOPPER, True),
            ("<=50+>89", PASCAL, False),
            ("<=61+>=89", PASCAL, True),
            # AND via '&'
            (">=70&<90", AMPERE_CONSUMER, True),
            (">=70&<90", ADA, True),
            (">=70&<90", HOPPER, False),
            (">=80&<=86", AMPERE_DATACENTER, True),
            # combined
            ("<70+>=90", PASCAL, True),
            ("<70+>=90", AMPERE_DATACENTER, False),
            # boundaries
            (">=0", 0, True),
            ("<=50", 50, True),
        ],
    )
    def test_condition_matching(self, condition, arch, matches):
        assert match_nvidia_architecture({condition: "sage"}, arch) == (["sage"] if matches else [])

    @pytest.mark.parametrize(
        "condition",
        [
            "",  # empty condition
            "   ",
            "abc",  # no number at all
            "-5",  # a leading minus is not part of the grammar
            ">= 89",  # the operator may not be separated from the value
            "+",  # nothing but a separator
            ">=70&",  # a dangling AND term is false, so the whole AND is false
        ],
    )
    def test_unparseable_conditions_never_match(self, condition):
        assert match_nvidia_architecture({condition: "sage"}, 89) == []

    def test_surrounding_whitespace_is_stripped(self):
        assert match_nvidia_architecture({"  >=89  ": "sage"}, 90) == ["sage"]

    def test_trailing_garbage_after_the_number_is_ignored(self):
        # The parser uses re.match, not fullmatch, so anything after the digits is
        # silently dropped rather than rejected.
        assert match_nvidia_architecture({">=89nonsense": "sage"}, 90) == ["sage"]

    def test_empty_condition_dict(self):
        assert match_nvidia_architecture({}, 89) == []

    def test_all_matching_values_are_returned_in_dict_order(self):
        conditions = {">=80": "sage2", ">=89": "sage3", "<70": "sdpa"}
        assert match_nvidia_architecture(conditions, 89) == ["sage2", "sage3"]
        assert match_nvidia_architecture(conditions, 61) == ["sdpa"]
        assert match_nvidia_architecture(conditions, 75) == []

    def test_values_are_returned_verbatim(self):
        assert match_nvidia_architecture({">=70": None}, 89) == [None]
        assert match_nvidia_architecture({">=70": ["a", "b"]}, 89) == [["a", "b"]]

    def test_used_as_the_category_expression_evaluator(self):
        # This is how resolutions.category_allowed drives the same evaluator, with
        # a resolution tier standing in for a compute capability.
        assert match_nvidia_architecture({">=720&<=1080": True}, 1080) == [True]
        assert match_nvidia_architecture({">=720&<=1080": True}, 1440) == []
