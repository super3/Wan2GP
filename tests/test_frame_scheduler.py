"""Tests for ``shared/utils/frame_scheduler.py``.

The module is pure python: it rounds frame counts onto a model's ``minimum`` /
``step`` / ``offset`` grid, parses the in-prompt ``[/command]`` slash blocks and
lays out the sliding windows used to generate a long video from several prompts.

Covered here: ``has_slash_commands``, the rounding family
(``normalize_frame_count``, ``floor_frame_count``, ``normalize_output_frame_count``,
``normalize_overlap``), ``_parse_duration``, ``_parse_options``, ``_window`` /
``build_extension_window``, ``clone_loras_slists``, the early-exit paths of
``prepare_loras_mult_windows`` and the window layout produced by
``build_frame_scheduler``.

Expectations were derived by reading the source; a handful of surprising-but-real
behaviours are pinned explicitly and flagged with a comment.
"""

from __future__ import annotations

import pytest


import shared.utils.frame_scheduler as fs


# A realistic Wan-style grid: windows of 81 frames on a 4n+1 latent grid.
WAN = dict(
    total_frames=200,
    fps=16.0,
    window_size=81,
    default_overlap=9,
    minimum=5,
    step=4,
)


def parse_options(prompt: str, **overrides):
    """``_parse_options`` with sensible defaults, mirroring a Wan i2v model."""

    kwargs = dict(
        supported_model_commands=set(),
        allow_new_shot=False,
        fps=16.0,
        total_frames=100,
        step=4,
        overlap_offset=1,
        default_overlap=9,
    )
    kwargs.update(overrides)
    return fs._parse_options(prompt, **kwargs)


def schedule(prompts, **overrides):
    kwargs = dict(WAN)
    kwargs.update(overrides)
    return fs.build_frame_scheduler(prompts, **kwargs)


class TestHasSlashCommands:
    @pytest.mark.parametrize(
        "prompts, expected",
        [
            ([], False),
            (["a plain prompt"], False),
            (["[/duration=5s] a cat"], True),
            (["plain", "[ / new_shot ]"], True),
            (["[  /duration=1]"], True),
            (["line one\n[/overlap]\nline two"], True),
            # A bracket without a leading slash is ordinary prompt text.
            (["[foo]"], False),
            # The command name group requires at least one character, so a bare
            # "[/]" does not register, while "[//]" (group == "/") does.
            (["[/]"], False),
            (["[//]"], True),
            # "[/ ]" matches: the single space satisfies the name group.
            (["[/ ]"], True),
        ],
    )
    def test_detection(self, prompts, expected):
        assert fs.has_slash_commands(prompts) is expected

    def test_none_entries_are_tolerated(self):
        assert fs.has_slash_commands([None, None]) is False
        assert fs.has_slash_commands([None, "[/overlap]"]) is True


class TestNormalizeFrameCount:
    @pytest.mark.parametrize(
        "frame_count, expected",
        [
            (5, 5),  # the minimum is already on the grid
            (6, 9),  # anything above a grid point rounds up to the next one
            (9, 9),
            (50, 53),
            (80, 81),
            (81, 81),  # exact multiple + offset is left alone
            (82, 85),
            (1, 5),  # below the minimum: clamped up first, then rounded
            (-100, 5),
        ],
    )
    def test_rounds_up_onto_the_4n_plus_1_grid(self, frame_count, expected):
        assert fs.normalize_frame_count(frame_count, 5, 4, 1) == expected

    @pytest.mark.parametrize("step", [0, 1, -3])
    def test_step_of_one_or_less_only_applies_the_minimum(self, step):
        assert fs.normalize_frame_count(7, 5, step, 1) == 7
        assert fs.normalize_frame_count(2, 5, step, 1) == 5

    def test_offset_zero_rounds_onto_plain_multiples(self):
        assert fs.normalize_frame_count(0, 5, 4, 0) == 8  # max(5, 0) -> ceil(5/4)*4
        assert fs.normalize_frame_count(8, 5, 4, 0) == 8

    def test_negative_offset_is_clamped_to_zero(self):
        assert fs.normalize_frame_count(6, 5, 4, -3) == 8

    def test_an_offset_above_the_frame_count_cannot_produce_a_negative_ceil(self):
        # ``max(0, frame_count - offset)`` guards the division, so the result is the
        # offset itself rather than a value below it.
        assert fs.normalize_frame_count(0, 0, 4, 10) == 10
        assert fs.normalize_frame_count(3, 0, 4, 10) == 10

    @pytest.mark.parametrize("frame_count", range(1, 90))
    def test_result_is_never_below_the_input_or_the_minimum(self, frame_count):
        result = fs.normalize_frame_count(frame_count, 5, 4, 1)
        assert result >= frame_count
        assert result >= 5
        assert (result - 1) % 4 == 0


class TestFloorFrameCount:
    @pytest.mark.parametrize(
        "frame_count, expected",
        [
            (5, 5),
            (6, 5),
            (8, 5),
            (9, 9),
            (50, 49),
            (53, 53),
            (84, 81),
        ],
    )
    def test_rounds_down_onto_the_grid(self, frame_count, expected):
        assert fs.floor_frame_count(frame_count, 5, 4, 1) == expected

    def test_below_minimum_falls_back_to_rounding_up(self):
        # minimum=7 is not on the 4n+1 grid, so flooring 7 yields 5 which is below
        # the minimum; the code then rounds the minimum *up* instead -- the result
        # is larger than the input despite the "floor" name.
        assert fs.floor_frame_count(3, 7, 4, 1) == 9
        assert fs.floor_frame_count(7, 7, 4, 1) == 9

    def test_below_minimum_fallback_with_offset_zero(self):
        # offset -3 is clamped to 0, so the grid is 0, 4, 8...; flooring 6 gives 4,
        # which is under the minimum, so the round-up fallback returns 8.
        assert fs.floor_frame_count(6, 5, 4, -3) == 8

    def test_an_offset_above_the_frame_count_also_uses_the_fallback(self):
        # lower would be ((0 - 10) // 4) * 4 + 10 == -2, i.e. below the minimum.
        assert fs.floor_frame_count(0, 0, 4, 10) == 10

    @pytest.mark.parametrize("step", [0, 1])
    def test_step_of_one_or_less_only_applies_the_minimum(self, step):
        assert fs.floor_frame_count(7, 5, step, 1) == 7
        assert fs.floor_frame_count(3, 5, step, 1) == 5


class TestNormalizeOutputFrameCount:
    @pytest.mark.parametrize(
        "frame_count, expected",
        [
            (49, 49),
            (50, 49),
            (51, 49),  # exact tie between 49 and 53 -> the lower grid point wins
            (52, 53),
            (53, 53),
            (48, 49),
        ],
    )
    def test_picks_the_nearest_grid_point_ties_going_down(self, frame_count, expected):
        assert fs.normalize_output_frame_count(frame_count, 5, 4, 1) == expected

    def test_below_minimum_is_lifted_to_the_minimum(self):
        assert fs.normalize_output_frame_count(1, 5, 4, 1) == 5
        # minimum=7 is off-grid: both neighbours collapse onto 9, so the result
        # overshoots the minimum rather than landing on it.
        assert fs.normalize_output_frame_count(3, 7, 4, 1) == 9

    @pytest.mark.parametrize("step", [0, 1])
    def test_step_of_one_or_less_only_applies_the_minimum(self, step):
        assert fs.normalize_output_frame_count(40, 5, step, 1) == 40
        assert fs.normalize_output_frame_count(2, 5, step, 1) == 5

    @pytest.mark.parametrize("frame_count", range(5, 90))
    def test_always_lands_on_the_nearer_neighbouring_grid_point(self, frame_count):
        # The neighbours are derived here arithmetically rather than by calling
        # floor_frame_count/normalize_frame_count, so a regression in those helpers
        # cannot move this test's goalposts along with the result.
        lower = frame_count - (frame_count - 1) % 4
        upper = lower if lower == frame_count else lower + 4
        result = fs.normalize_output_frame_count(frame_count, 5, 4, 1)
        assert result in (lower, upper)
        if frame_count - lower < upper - frame_count:
            assert result == lower
        elif frame_count - lower > upper - frame_count:
            assert result == upper
        else:
            assert result == lower  # exact tie -> the lower grid point wins


class TestNormalizeOverlap:
    @pytest.mark.parametrize(
        "frame_count, expected",
        [
            (0, 0),  # zero is passed through untouched (means "new shot")
            (1, 1),
            (2, 1),
            (3, 5),  # half a step away rounds up
            (5, 5),
            (9, 9),
            (10, 9),
            (11, 13),
            (13, 13),
        ],
    )
    def test_rounds_to_the_nearest_grid_point(self, frame_count, expected):
        assert fs.normalize_overlap(frame_count, 4, 1) == (expected, None)

    def test_negative_overlap_is_rejected(self):
        overlap, error = fs.normalize_overlap(-1, 4, 1)
        assert overlap is None
        assert error == "/overlap must be 0 or a positive frame count."

    def test_offset_zero_floors_at_a_full_step(self):
        assert fs.normalize_overlap(1, 4, 0) == (4, None)
        assert fs.normalize_overlap(5, 4, 0) == (4, None)
        assert fs.normalize_overlap(8, 4, 0) == (8, None)

    def test_negative_offset_behaves_like_zero(self):
        assert fs.normalize_overlap(9, 4, -1) == (8, None)

    @pytest.mark.parametrize("step", [0, 1])
    def test_step_of_one_or_less_keeps_the_value(self, step):
        assert fs.normalize_overlap(7, step, 1) == (7, None)


class TestParseDuration:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("48", 48),
            ("  48  ", 48),
            ("5s", 80),  # seconds * fps
            ("1S", 16),  # case insensitive
            ("2.5s", 40),
            ("3.7s", 59),  # rounded to the nearest frame
            ("10%", 10),  # percentage of total_frames
            (" 20% ", 20),
            ("100%", 100),
        ],
    )
    def test_accepted_values(self, raw, expected):
        assert fs._parse_duration(raw, fps=16.0, total_frames=100) == (expected, None)

    def test_uses_the_supplied_fps_and_total_frames(self):
        assert fs._parse_duration("2s", fps=24.0, total_frames=100) == (48, None)
        assert fs._parse_duration("25%", fps=24.0, total_frames=321) == (80, None)

    def test_half_frames_use_pythons_banker_rounding(self):
        # round(160.5) == 160, not 161.
        assert fs._parse_duration("50%", fps=24.0, total_frames=321) == (160, None)
        assert fs._parse_duration("0.5s", fps=15.0, total_frames=100) == (8, None)
        assert fs._parse_duration("0.5s", fps=13.0, total_frames=100) == (6, None)

    @pytest.mark.parametrize("raw", ["abc", "", "   ", "5.5", "s", "%", "5 frames"])
    def test_unparsable_values_report_the_raw_input(self, raw):
        frames, error = fs._parse_duration(raw, fps=16.0, total_frames=100)
        assert frames is None
        assert error == (
            f"Invalid /duration value '{raw}'. "
            "Use frames, seconds like 5s, or a percentage like 20%."
        )

    def test_none_is_reported_as_the_string_none(self):
        # ``str(raw_value or "")`` turns None into "", but the error message
        # interpolates the original ``raw_value`` -- so it reads 'None'.
        frames, error = fs._parse_duration(None, fps=16.0, total_frames=100)
        assert frames is None
        assert "value 'None'" in error

    @pytest.mark.parametrize("raw", ["0", "-4", "0%", "-2s"])
    def test_non_positive_durations_are_rejected(self, raw):
        assert fs._parse_duration(raw, fps=16.0, total_frames=100) == (
            None,
            "/duration must be a positive frame count.",
        )


class TestParseOptions:
    def test_prompt_without_a_block_is_untouched(self):
        stripped, wgp, model, has_options, error = parse_options("a cat running")
        assert (stripped, wgp, model, has_options, error) == ("a cat running", {}, {}, False, None)

    def test_block_is_removed_from_the_prompt(self):
        stripped, wgp, _, has_options, error = parse_options("a cat [/duration=5s] running")
        assert stripped == "a cat  running"
        assert wgp == {"duration_frames": 80}
        assert has_options is True
        assert error is None

    def test_several_blocks_and_comma_separated_options(self):
        stripped, wgp, _, _, error = parse_options("[/duration=48][/overlap=9] hello")
        assert (stripped, wgp, error) == (" hello", {"duration_frames": 48, "overlap_frames": 9}, None)
        _, wgp, _, _, error = parse_options("[/duration=48,/overlap=9]")
        assert (wgp, error) == ({"duration_frames": 48, "overlap_frames": 9}, None)

    def test_whitespace_and_case_are_normalized(self):
        _, wgp, _, _, error = parse_options("[ / DURATION = 5S ]")
        assert (wgp, error) == ({"duration_frames": 80}, None)

    def test_overlap_without_a_value_uses_the_default(self):
        _, wgp, _, _, error = parse_options("[/overlap]", default_overlap=9)
        assert (wgp, error) == ({"overlap_frames": 9}, None)

    def test_overlap_value_is_normalized_onto_the_grid(self):
        _, wgp, _, _, error = parse_options("[/overlap=10]")
        assert (wgp, error) == ({"overlap_frames": 9}, None)

    def test_overlap_zero_requires_t2v_support(self):
        _, wgp, _, _, error = parse_options("[/overlap=0]", allow_new_shot=False)
        assert wgp == {}
        assert error == "/overlap=0 is only supported by text-to-video capable models."
        _, wgp, _, _, error = parse_options("[/overlap=0]", allow_new_shot=True)
        assert (wgp, error) == ({"overlap_frames": 0, "new_shot": True}, None)

    def test_valueless_overlap_bypasses_the_t2v_gate_when_the_default_is_zero(self):
        # QUIRK: the ``allow_new_shot`` check only runs when an explicit "=" was
        # given, so "[/overlap]" with a zero default silently enables new_shot on
        # a model that rejects "[/overlap=0]".
        _, wgp, _, _, error = parse_options("[/overlap]", default_overlap=0, allow_new_shot=False)
        assert (wgp, error) == ({"overlap_frames": 0, "new_shot": True}, None)

    def test_new_shot_requires_t2v_support(self):
        _, wgp, _, _, error = parse_options("[/new_shot]", allow_new_shot=False)
        assert wgp == {}
        assert error == "/new_shot is only supported by text-to-video capable models."

    def test_new_shot_sets_a_zero_overlap(self):
        _, wgp, _, _, error = parse_options("[/new_shot]", allow_new_shot=True)
        assert (wgp, error) == ({"overlap_frames": 0, "new_shot": True}, None)

    def test_loras_mult_keeps_the_raw_value_including_commas(self):
        _, wgp, _, _, error = parse_options("[/loras_mult=1,2 0.5,3]")
        assert (wgp, error) == ({"loras_multipliers": "1,2 0.5,3"}, None)
        _, wgp, _, _, error = parse_options("[/loras_mult=0.8;1]")
        assert (wgp, error) == ({"loras_multipliers": "0.8;1"}, None)

    def test_comma_value_that_looks_like_a_command_is_split_off(self):
        # QUIRK: the comma re-joining heuristic keeps any fragment whose name is a
        # known command as a separate option, truncating the multiplier value.
        _, wgp, _, _, error = parse_options("[/loras_mult=1,new_shot]", allow_new_shot=True)
        assert (wgp, error) == (
            {"loras_multipliers": "1", "overlap_frames": 0, "new_shot": True},
            None,
        )

    def test_comma_value_matching_a_model_command_is_also_split_off(self):
        # QUIRK, same heuristic as above but through the ``supported_model_commands``
        # half of the guard: the fragment becomes its own model option.
        _, wgp, model, _, error = parse_options(
            "[/loras_mult=1,zoom]", supported_model_commands={"zoom"}
        )
        assert (wgp, model, error) == ({"loras_multipliers": "1"}, {"zoom": True}, None)

    def test_comma_fragment_carrying_its_own_value_is_never_rejoined(self):
        # A fragment containing "=" is split off whatever its name, so a multiplier
        # value with an "=" in it ends up parsed as an unknown command.
        _, wgp, _, _, error = parse_options("[/loras_mult=1,x=2]")
        assert wgp == {"loras_multipliers": "1"}
        assert error.startswith("Unknown prompt command '/x'.")

    def test_model_commands_with_and_without_values(self):
        _, wgp, model, _, error = parse_options(
            "[/motion=fast][/style]", supported_model_commands={"motion", "style"}
        )
        assert (wgp, model, error) == ({}, {"motion": "fast", "style": True}, None)

    def test_unsupported_model_command_is_an_error(self):
        _, _, _, _, error = parse_options("[/motion=fast]", supported_model_commands={"style"})
        assert error == (
            "Unknown prompt command '/motion'. "
            "Supported / commands: /duration, /loras_mult, /new_shot, /overlap, /style."
        )

    @pytest.mark.parametrize(
        "prompt, expected_error",
        [
            ("[/duration]", "/duration requires a value, e.g. [/duration=5s]."),
            ("[/duration=]", "/duration requires a value, e.g. [/duration=5s]."),
            ("[/overlap=]", "/overlap value cannot be empty. Use [/overlap] or [/overlap=9]."),
            ("[/overlap=abc]", "Invalid /overlap value 'abc'. Use an integer frame count."),
            ("[/overlap=-1]", "/overlap must be 0 or a positive frame count."),
            ("[/new_shot=1]", "/new_shot does not take a value."),
            ("[/loras_mult]", "/loras_mult requires a value, e.g. [/loras_mult=1;3]."),
        ],
    )
    def test_malformed_options(self, prompt, expected_error):
        stripped, wgp, _, has_options, error = parse_options(prompt, allow_new_shot=True)
        assert error == expected_error
        assert stripped == ""
        assert has_options is True
        assert wgp == {}

    def test_invalid_duration_error_is_forwarded(self):
        _, wgp, _, _, error = parse_options("[/duration=abc]")
        assert error.startswith("Invalid /duration value 'abc'.")
        # The failed parse still leaves the (None) key behind in the options dict.
        assert wgp == {"duration_frames": None}

    def test_first_error_wins_and_later_options_are_skipped(self):
        _, wgp, _, _, error = parse_options("[/bogus][/duration=5s]")
        assert error.startswith("Unknown prompt command '/bogus'.")
        assert wgp == {}

    def test_options_accepted_before_the_failing_one_are_kept(self):
        # The error short-circuits everything after it, but what was already parsed
        # stays in the dict. Harmless today because build_frame_scheduler throws the
        # whole result away and returns {} on any error.
        _, wgp, _, _, error = parse_options("[/duration=5s][/bogus]")
        assert wgp == {"duration_frames": 80}
        assert error.startswith("Unknown prompt command '/bogus'.")

    @pytest.mark.parametrize("prompt", ["[/ ]", "[/=5]"])
    def test_empty_option_names_are_ignored_but_still_count_as_a_block(self, prompt):
        stripped, wgp, model, has_options, error = parse_options(prompt)
        assert (stripped, wgp, model, has_options, error) == ("", {}, {}, True, None)


class TestWindow:
    def test_frame_num_is_the_sum_of_the_three_parts(self):
        window = fs._window("p", 50, 9, 0, {"a": 1}, 5, 4)
        assert window == {
            "prompt": "p",
            "output_frames": 49,  # 50 rounded to the nearest grid point
            "overlap_frames": 9,
            "discard_last_frames": 3,  # grown by the rounding of frame_num
            "frame_num": 61,  # normalize(49 + 9 + 0)
            "new_shot": False,
            "model_options": {"a": 1},
        }

    @pytest.mark.parametrize(
        "output, overlap, discard, expected",
        [
            # requested -> (output_frames, overlap_frames, discard_last_frames, frame_num)
            (50, 9, 0, (49, 9, 3, 61)),
            (52, 9, 3, (53, 9, 3, 65)),
            (81, 0, 0, (81, 0, 0, 81)),
            (7, 13, 4, (5, 13, 7, 25)),  # 7 is a tie between 5 and 9 -> rounds down
            (100, 5, 8, (101, 5, 11, 117)),
        ],
    )
    def test_layout(self, output, overlap, discard, expected):
        window = fs._window("p", output, overlap, discard, None, 5, 4)
        assert (
            window["output_frames"],
            window["overlap_frames"],
            window["discard_last_frames"],
            window["frame_num"],
        ) == expected
        # ``discard_last_frames`` is *defined* as ``frame_num - output - overlap``, so
        # the sum identity below can never fail; it documents the layout, while the
        # literal expectations above are what actually pin the rounding.
        assert window["frame_num"] == (
            window["output_frames"] + window["overlap_frames"] + window["discard_last_frames"]
        )
        # Rounding frame_num up can only ever grow the discard, never shrink it.
        assert window["discard_last_frames"] >= discard
        assert (window["frame_num"] - 1) % 4 == 0
        assert (window["output_frames"] - 1) % 4 == 0

    def test_negative_inputs_are_clamped(self):
        window = fs._window("p", 20, -5, -5, None, 5, 4)
        assert window["overlap_frames"] == 0
        assert window["discard_last_frames"] == 0
        assert window["frame_num"] == 21

    def test_model_options_are_copied_not_aliased(self):
        source = {"x": 1}
        window = fs._window("p", 20, 0, 0, source, 5, 4)
        source["x"] = 2
        assert window["model_options"] == {"x": 1}
        assert window["model_options"] is not source

    def test_model_options_none_becomes_an_empty_dict(self):
        assert fs._window("p", 20, 0, 0, None, 5, 4)["model_options"] == {}

    def test_new_shot_is_coerced_to_a_bool(self):
        assert fs._window("p", 20, 0, 0, None, 5, 4, new_shot=1)["new_shot"] is True
        assert fs._window("p", 20, 0, 0, None, 5, 4)["new_shot"] is False


class TestBuildExtensionWindow:
    def test_window_size_is_split_between_output_and_overlap(self):
        window = fs.build_extension_window("p", window_size=81, overlap_frames=9, minimum=5, step=4)
        # 81 - 9 = 72 output frames requested, rounded to 73; frame_num then has to
        # be rounded up as well, so it exceeds the requested window size.
        assert window == {
            "prompt": "p",
            "output_frames": 73,
            "overlap_frames": 9,
            "discard_last_frames": 3,
            "frame_num": 85,
            "new_shot": False,
            "model_options": {},
        }

    def test_discarded_frames_reduce_the_output(self):
        window = fs.build_extension_window(
            "p", window_size=81, overlap_frames=9, discard_last_frames=4, minimum=5, step=4
        )
        assert window["output_frames"] == 69
        assert window["discard_last_frames"] == 7
        assert window["frame_num"] == 85

    def test_oversized_overlap_still_yields_at_least_one_output_frame(self):
        window = fs.build_extension_window(
            "p", window_size=10, overlap_frames=20, discard_last_frames=5, minimum=5, step=4
        )
        assert window["output_frames"] == 5  # max(1, ...) then lifted to the minimum
        assert window["frame_num"] == 33

    def test_frame_offset_zero_uses_plain_multiples(self):
        window = fs.build_extension_window(
            "p", window_size=81, overlap_frames=0, minimum=5, step=4, frame_offset=0
        )
        assert window["output_frames"] == 80
        assert window["frame_num"] == 80


class TestCloneLorasSlists:
    def test_none_stays_none(self):
        assert fs.clone_loras_slists(None) is None

    def test_top_level_lists_and_nested_dicts_are_copied(self):
        source = {"a": [1.0, 2.0], "b": {"c": [3.0]}, "d": 7}
        cloned = fs.clone_loras_slists(source)
        assert cloned == source
        assert cloned is not source
        assert cloned["a"] is not source["a"]
        assert cloned["b"] is not source["b"]
        assert cloned["b"]["c"] is not source["b"]["c"]
        assert cloned["d"] == 7

    def test_mutating_the_clone_leaves_the_source_alone(self):
        source = {"a": [1.0, 2.0], "b": {"c": [3.0]}}
        cloned = fs.clone_loras_slists(source)
        cloned["a"].append(9.0)
        cloned["b"]["c"][0] = 99.0
        cloned["new"] = 1
        assert source == {"a": [1.0, 2.0], "b": {"c": [3.0]}}

    def test_lists_are_copied_only_one_level_deep(self):
        # The clone is per-level: elements of a list are never copied, so nested
        # lists remain shared with the source.
        source = {"a": [[1.0], [2.0]]}
        cloned = fs.clone_loras_slists(source)
        assert cloned["a"] is not source["a"]
        assert cloned["a"][0] is source["a"][0]

    def test_empty_mapping(self):
        assert fs.clone_loras_slists({}) == {}


class TestPrepareLorasMultWindows:
    @pytest.mark.parametrize("scheduler", [None, {}, {"active": False}])
    def test_inactive_schedulers_are_a_no_op(self, scheduler):
        assert fs.prepare_loras_mult_windows(scheduler, [], 10, 1) is None

    def test_windows_without_multipliers_are_skipped(self):
        scheduler = {"active": True, "windows": [{"prompt": "a"}, {"loras_multipliers": ""}]}
        assert fs.prepare_loras_mult_windows(scheduler, [], 10, 1) is None

    def test_multipliers_without_a_selected_lora_are_reported(self):
        scheduler = {"active": True, "windows": [{"loras_multipliers": "1;2"}]}
        assert fs.prepare_loras_mult_windows(scheduler, [], 10, 2) == (
            "Sliding window 1 uses /loras_mult but no LoRA is selected."
        )

    def test_unparsable_multipliers_are_reported_with_the_window_number(self):
        scheduler = {
            "active": True,
            "windows": [{"loras_multipliers": "1"}, {"loras_multipliers": "abc"}],
        }
        assert fs.prepare_loras_mult_windows(scheduler, ["lora"], 10, 2) == (
            "Error parsing /loras_mult for Sliding window 2: "
            "Lora Multiplier no 1 (abc) is invalid"
        )
        # The failing window is reported before anything is stored on it.
        assert "loras_slists" not in scheduler["windows"][1]

    def test_stored_slists_are_only_written_when_requested(self):
        scheduler = {"active": True, "windows": [{"loras_multipliers": "1;2"}]}
        assert fs.prepare_loras_mult_windows(scheduler, ["lora"], 10, 2) is None
        assert "loras_slists" not in scheduler["windows"][0]
        assert fs.prepare_loras_mult_windows(scheduler, ["lora"], 10, 2, store_slists=True) is None
        assert "loras_slists" in scheduler["windows"][0]


class TestBuildFrameSchedulerInactive:
    def test_prompts_without_commands_produce_an_inactive_scheduler(self):
        scheduler, error = schedule(["a cat", "a dog"])
        assert error is None
        assert scheduler == {"active": False, "prompts": ["a cat", "a dog"], "model_commands": []}
        assert "windows" not in scheduler

    def test_inactive_prompts_are_stripped(self):
        scheduler, error = schedule(["  a cat  ", "\tb\n"])
        assert (scheduler["prompts"], error) == (["a cat", "b"], None)

    def test_no_prompts_at_all(self):
        assert schedule([]) == ({"active": False, "prompts": [], "model_commands": []}, None)

    def test_model_commands_are_normalized_and_reported(self):
        scheduler, error = schedule(["a cat"], supported_model_commands=["/Motion", "  ", "style"])
        assert (scheduler["model_commands"], error) == (["motion", "style"], None)

    def test_an_empty_block_still_activates_the_scheduler(self):
        scheduler, _ = schedule(["[/ ]a cat"], total_frames=40)
        assert scheduler["active"] is True


class TestBuildFrameSchedulerLayout:
    def test_explicit_durations_lay_out_one_window_per_prompt(self):
        scheduler, error = schedule(["[/duration=48] a cat", "[/duration=3s] a dog"])
        assert error is None
        assert scheduler["active"] is True
        assert scheduler["prompts"] == ["a cat", "a dog"]
        assert scheduler["windows"] == [
            {
                "prompt": "a cat",
                # The first window has nothing to continue from, so its overlap is
                # clamped to first_window_overlap_frames (0 by default).
                "output_frames": 49,
                "overlap_frames": 0,
                "discard_last_frames": 0,
                "frame_num": 49,
                "new_shot": False,
                "model_options": {},
            },
            {
                "prompt": "a dog",
                "output_frames": 49,
                "overlap_frames": 9,
                "discard_last_frames": 3,
                "frame_num": 61,
                "new_shot": False,
                "model_options": {},
            },
        ]
        assert scheduler["predicted_total_frames"] == 98
        assert scheduler["requested_total_frames"] == 200

    def test_metadata_of_an_active_scheduler(self):
        scheduler, _ = schedule(["[/duration=48] a"], window_size=80, supported_model_commands=["Zoom"])
        assert scheduler["default_window_size"] == 81  # normalized onto the grid
        assert scheduler["default_overlap_frames"] == 9
        assert scheduler["overlap_offset"] == 1
        assert scheduler["minimum"] == 5
        assert scheduler["step"] == 4
        assert scheduler["frame_offset"] == 1
        assert scheduler["model_commands"] == ["zoom"]

    def test_missing_durations_are_filled_up_to_the_requested_total(self):
        scheduler, error = schedule(["[/ ] a cat"])
        assert error is None
        # window 1 has no overlap so it uses the whole window; the extension
        # windows each give up ``default_overlap`` frames to the previous one.
        assert [w["output_frames"] for w in scheduler["windows"]] == [81, 73, 45, 5]
        assert [w["overlap_frames"] for w in scheduler["windows"]] == [0, 9, 9, 9]
        assert scheduler["prompts"] == ["a cat"] * 4
        assert scheduler["predicted_total_frames"] == 204
        assert scheduler["predicted_total_frames"] >= scheduler["requested_total_frames"]

    def test_extension_windows_repeat_the_last_prompt(self):
        scheduler, _ = schedule(["[/overlap=13] cat", "dog"])
        assert scheduler["prompts"] == ["cat", "dog", "dog", "dog"]
        # The /overlap on the *first* prompt is still clamped away by
        # first_window_overlap_frames.
        assert scheduler["windows"][0]["overlap_frames"] == 0

    def test_any_explicit_duration_disables_the_auto_extension(self):
        scheduler, _ = schedule(["[/duration=48] a", "b"], total_frames=200)
        assert len(scheduler["windows"]) == 2
        assert scheduler["predicted_total_frames"] < scheduler["requested_total_frames"]

    def test_explicit_durations_are_not_clamped_to_the_requested_total(self):
        scheduler, error = schedule(["[/duration=200] a"], total_frames=50)
        assert error is None
        assert scheduler["windows"][0]["output_frames"] == 201
        assert scheduler["predicted_total_frames"] == 201

    def test_auto_duration_is_clamped_to_what_is_left(self):
        scheduler, _ = schedule(["[/ ] a"], total_frames=40)
        assert len(scheduler["windows"]) == 1
        assert scheduler["windows"][0]["output_frames"] == 41  # 40 rounded onto the grid

    def test_first_window_overlap_frames_is_an_upper_bound(self):
        scheduler, _ = schedule(["[/duration=48] a", "b"], total_frames=100, first_window_overlap_frames=13)
        assert scheduler["windows"][0]["overlap_frames"] == 9  # min(default 9, 13)
        scheduler, _ = schedule(["[/duration=48] a", "b"], total_frames=100, first_window_overlap_frames=5)
        assert scheduler["windows"][0]["overlap_frames"] == 5

    def test_overlap_offset_is_applied_to_both_the_default_and_the_prompt_overlap(self):
        scheduler, error = schedule(
            ["a", "[/overlap=10] b"], overlap_offset=0, first_window_overlap_frames=100
        )
        assert error is None
        # With offset 0 the overlap grid is plain multiples of the step, floored at
        # one full step: the default 9 becomes 8 and the explicit 10 becomes 12.
        assert scheduler["default_overlap_frames"] == 8
        assert [w["overlap_frames"] for w in scheduler["windows"]][:2] == [8, 12]
        assert scheduler["overlap_offset"] == 0

    def test_frame_offset_zero_puts_the_windows_on_plain_multiples(self):
        scheduler, error = schedule(["[/duration=48] a", "[/duration=48] b"], frame_offset=0)
        assert error is None
        assert [(w["output_frames"], w["frame_num"]) for w in scheduler["windows"]] == [
            (48, 48),
            (48, 60),
        ]
        assert scheduler["default_window_size"] == 84  # 81 rounded onto 4n
        assert scheduler["frame_offset"] == 0

    def test_negative_discard_last_frames_is_clamped_to_zero(self):
        negative, error = schedule(["[/duration=48] a", "b"], discard_last_frames=-4)
        assert error is None
        zero, _ = schedule(["[/duration=48] a", "b"], discard_last_frames=0)
        assert negative == zero
        # Without the clamp the auto window would have been sized 81 - 9 + 4 = 76
        # frames of payload (rounding to 77) instead of 72 (rounding to 73).
        assert [w["output_frames"] for w in negative["windows"]] == [49, 73]

    def test_realistic_multi_prompt_schedule_with_discard(self):
        scheduler, error = schedule(
            ["[/duration=48]a", "[/overlap=13]b", "[/duration=1s]c"],
            first_window_overlap_frames=5,
            discard_last_frames=4,
        )
        assert error is None
        assert [
            (w["output_frames"], w["overlap_frames"], w["discard_last_frames"], w["frame_num"])
            for w in scheduler["windows"]
        ] == [(49, 5, 7, 61), (65, 13, 7, 85), (17, 9, 7, 33)]
        assert scheduler["predicted_total_frames"] == 131

    def test_new_shot_window_has_no_overlap(self):
        scheduler, error = schedule(
            ["a[/duration=48]", "[/new_shot] b"], total_frames=100, allow_new_shot=True
        )
        assert error is None
        assert [w["new_shot"] for w in scheduler["windows"]] == [False, True]
        assert [w["overlap_frames"] for w in scheduler["windows"]] == [0, 0]

    def test_loras_multipliers_are_attached_to_the_window(self):
        scheduler, _ = schedule(["[/duration=48][/loras_mult=0.8;1] a"])
        assert scheduler["windows"][0]["loras_multipliers"] == "0.8;1"
        assert "loras_multipliers" not in fs._window("a", 48, 0, 0, {}, 5, 4)

    def test_model_options_only_apply_to_their_own_window(self):
        scheduler, _ = schedule(["[/style] a"], total_frames=100, supported_model_commands=["style"])
        assert scheduler["windows"][0]["model_options"] == {"style": True}
        assert all(w["model_options"] == {} for w in scheduler["windows"][1:])

    def test_step_one_models_get_exact_frame_counts(self):
        scheduler, error = schedule(
            ["[/ ]a"], total_frames=10, window_size=4, default_overlap=2, minimum=1, step=1
        )
        assert error is None
        assert [w["output_frames"] for w in scheduler["windows"]] == [4, 2, 2, 2]
        assert scheduler["predicted_total_frames"] == 10

    def test_window_smaller_than_the_overlap_still_terminates(self):
        scheduler, error = schedule(
            ["[/ ]a"], total_frames=10, window_size=3, default_overlap=9, minimum=1, step=1
        )
        assert error is None
        # max(1, window_size - overlap - discard) floors the payload at one frame.
        assert [w["output_frames"] for w in scheduler["windows"]] == [3, 1, 1, 1, 1, 1, 1, 1]
        assert scheduler["predicted_total_frames"] == 10

    def test_every_window_satisfies_the_frame_num_invariant(self):
        scheduler, _ = schedule(["[/ ] a"], discard_last_frames=4)
        # Spelled out, because the sum identity below holds by construction:
        # discard_last_frames is derived from frame_num, not checked against it.
        assert [
            (w["output_frames"], w["overlap_frames"], w["discard_last_frames"], w["frame_num"])
            for w in scheduler["windows"]
        ] == [(77, 0, 4, 81), (69, 9, 7, 85), (53, 9, 7, 69), (5, 9, 7, 21)]
        for window in scheduler["windows"]:
            assert window["frame_num"] == (
                window["output_frames"] + window["overlap_frames"] + window["discard_last_frames"]
            )
            assert window["discard_last_frames"] >= 4
            assert (window["frame_num"] - 1) % 4 == 0


class TestBuildFrameSchedulerErrors:
    def test_negative_default_overlap(self):
        assert schedule(["a"], default_overlap=-1) == (
            {},
            "/overlap must be 0 or a positive frame count.",
        )

    def test_unknown_command_aborts_the_whole_schedule(self):
        scheduler, error = schedule(["ok", "[/bogus] a"])
        assert scheduler == {}
        assert error == (
            "Unknown prompt command '/bogus'. "
            "Supported / commands: /duration, /loras_mult, /new_shot, /overlap."
        )

    def test_new_shot_without_t2v_support(self):
        scheduler, error = schedule(["[/new_shot] a"], allow_new_shot=False)
        assert (scheduler, error) == (
            {},
            "/new_shot is only supported by text-to-video capable models.",
        )

    def test_exhausted_frame_budget(self):
        scheduler, error = schedule(["[/duration=100] a", "b"], total_frames=100)
        assert scheduler == {}
        assert error.startswith(
            "Sliding window 2 would generate no frame because previous windows already "
            "consume the requested frame count."
        )

    def test_invalid_duration_aborts_the_whole_schedule(self):
        scheduler, error = schedule(["[/duration=nope] a"])
        assert scheduler == {}
        assert error.startswith("Invalid /duration value 'nope'.")


def test_build_frame_scheduler_does_not_mutate_its_input():
    prompts = ["[/duration=48] a", "b"]
    original = list(prompts)
    schedule(prompts)
    assert prompts == original


def test_build_frame_scheduler_is_deterministic():
    first = schedule(["[/ ] a cat", "[/overlap=13] a dog"], supported_model_commands=["b", "a"])
    second = schedule(["[/ ] a cat", "[/overlap=13] a dog"], supported_model_commands=["a", "b"])
    assert first == second
