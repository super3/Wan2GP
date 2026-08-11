"""Tests for ``shared/utils/loras_mutipliers.py``.

That module turns the user-supplied "Loras Multipliers" text box into per-step
multiplier schedules.  It is pure python (stdlib only) and covers three fairly
separate concerns, all exercised here:

* **parsing** -- ``preparse_loras_multipliers`` / ``parse_loras_multipliers``
  handle space- and newline-separated values, ``#`` comment lines, the ``;``
  phase separator, the ``,`` per-step separator, the ``:`` branch separator and
  the ``|`` before/after separator.
* **expansion** -- ``expand_slist`` stretches a short list of values over
  ``num_inference_steps``, honouring the model-switch step boundaries, and
  ``get_model_switch_steps`` derives those boundaries from a timestep schedule.
* **text surgery** -- ``merge_loras_settings`` / ``extract_loras_side`` and the
  ``_spans``-based helpers edit the multiplier *text* in place so that user
  comments and layout survive a merge.

Every expectation below is written out as a literal derived from reading the
implementation, never by calling the function under test or by re-deriving the
value with the same expression the implementation uses.
"""

from __future__ import annotations

import pytest


import shared.utils.loras_mutipliers as lm


def tokens(text: str) -> list[str]:
    """The substrings ``_spans`` considers to be multiplier tokens."""

    return [text[start:end] for start, end in lm._spans(text)]


class TestPreparseLorasMultipliers:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("  1.0 0.5  ", ["1.0", "0.5"]),
            ("1.0\r\n0.5", ["1.0", "0.5"]),
            # '|' is only a separator here (the before/after split happens later),
            # while ';' is kept inside the token and parsed downstream.
            ("1.0|0.5;0.3", ["1.0", "0.5;0.3"]),
        ],
    )
    def test_splits_on_whitespace_newlines_and_bars(self, raw, expected):
        assert lm.preparse_loras_multipliers(raw) == expected

    def test_whole_comment_lines_are_dropped(self):
        raw = "  1.0\n\n# a comment line\n0.5  "
        assert lm.preparse_loras_multipliers(raw) == ["1.0", "0.5"]

    def test_trailing_comments_are_not_stripped(self):
        # Only lines *starting* with '#' are removed; an inline comment survives
        # preparsing and becomes bogus tokens (parse_loras_multipliers then
        # rejects them).  The _spans() based helpers do understand inline
        # comments -- the two comment models disagree.
        assert lm.preparse_loras_multipliers("1 # trailing") == ["1", "#", "trailing"]

    @pytest.mark.parametrize("raw", ["", "# only a comment"])
    def test_input_without_values_yields_no_tokens(self, raw):
        # Whitespace-only and comment-only input mean "no multipliers given", the same
        # as the empty string.  These used to yield [""] -- a single empty token that
        # parse_loras_multipliers then rejected as an invalid multiplier.
        assert lm.preparse_loras_multipliers(raw) == []

    def test_input_without_values_is_accepted_downstream(self):
        for raw in ("", "   ", "# only a comment"):
            nums, _slists, error = lm.parse_loras_multipliers(raw, 2, 4)
            assert error == "", f"{raw!r} was rejected: {error}"
            assert nums == [1.0, 1.0]

    def test_consecutive_spaces_do_not_produce_empty_tokens(self):
        # The split is `.split()` rather than `.split(" ")`; the latter left an empty
        # token behind for every doubled space, which parse_loras_multipliers reported
        # as "Lora Multiplier no 2 () is invalid".
        assert lm.preparse_loras_multipliers("1.0  0.5") == ["1.0", "0.5"]
        assert lm.preparse_loras_multipliers("1.0 \t 0.5\n\n0.25") == ["1.0", "0.5", "0.25"]
        assert lm.parse_loras_multipliers("1.0  0.5", 2, 4)[0] == [1.0, 0.5]
        assert lm.parse_loras_multipliers("1.0  0.5", 2, 4)[2] == ""

    def test_list_input_is_stripped_elementwise_and_non_strings_kept(self):
        assert lm.preparse_loras_multipliers(["1.0 ", " 0.5\n", 3]) == ["1.0", "0.5", 3]


def _slists(phase1, phase2=1.0, phase3=1.0, shared=False):
    return {"phase1": [phase1], "phase2": [phase2], "phase3": [phase3], "shared": [shared]}


class TestExpandSlist:
    def test_shared_scalar_stays_a_scalar(self):
        assert lm.expand_slist(_slists(0.5, shared=True), 0, 4, 2, 4) == 0.5

    @pytest.mark.parametrize(
        "values, steps, expected",
        [
            ([0.0, 1.0], 4, [0.0, 0.0, 1.0, 1.0]),
            # More values than steps: the list is sub-sampled, not averaged.
            ([0.0, 1.0, 2.0, 3.0, 4.0], 3, [0.0, 1.0, 3.0]),
            ([0.7], 3, [0.7, 0.7, 0.7]),
        ],
    )
    def test_shared_list_is_stretched_over_the_steps(self, values, steps, expected):
        assert lm.expand_slist(_slists(values, shared=True), 0, steps, 0, steps) == expected

    def test_zero_steps_gives_an_empty_schedule(self):
        assert lm.expand_slist(_slists([1.0, 2.0], shared=True), 0, 0, 0, 0) == []

    def test_identical_unshared_float_phases_collapse_to_a_scalar(self):
        assert lm.expand_slist(_slists(0.7, 0.7, 0.7), 0, 6, 2, 4) == 0.7

    def test_two_phases_split_at_the_model_switch_step(self):
        # phase3 is left at its 1.0 default and model_switch_step2 == steps, so
        # the third segment has zero length.
        got = lm.expand_slist(_slists(0.8, 0.5, 1.0), 0, 6, 3, 6)
        assert got == [0.8, 0.8, 0.8, 0.5, 0.5, 0.5]

    def test_per_phase_lists_are_expanded_independently(self):
        got = lm.expand_slist(_slists([0.1, 0.2], 0.5, 0.9), 0, 6, 2, 4)
        assert got == [0.1, 0.2, 0.5, 0.5, 0.9, 0.9]

    def test_switch_step_equal_to_total_steps_uses_phase1_only(self):
        assert lm.expand_slist(_slists(0.8, 0.5, 0.2), 0, 4, 4, 4) == [0.8] * 4

    def test_selects_the_requested_lora_index(self):
        slists = {
            "phase1": [0.1, 0.2],
            "phase2": [0.1, 0.2],
            "phase3": [0.1, 0.2],
            "shared": [True, True],
        }
        assert lm.expand_slist(slists, 1, 4, 2, 4) == 0.2


class TestParseLorasMultipliers:
    def test_returns_first_step_value_slists_and_empty_error(self):
        nums, slists, error = lm.parse_loras_multipliers("1.0 0.5", 2, 4)
        assert error == ""
        assert nums == [1.0, 0.5]
        assert slists["phase1"] == [1.0, 0.5]
        assert slists["shared"] == [True, True]
        # Both switch steps default to num_inference_steps.
        assert slists["model_switch_step"] == 4
        assert slists["model_switch_step2"] == 4

    def test_empty_string_defaults_every_lora_to_one(self):
        nums, slists, error = lm.parse_loras_multipliers("", 2, 4)
        assert (nums, error) == ([1.0, 1.0], "")
        assert slists["phase1"] == [1.0, 1.0]
        # Nothing was parsed, so the loras never got flagged as shared even
        # though all three phases hold the same default.
        assert slists["shared"] == [False, False]

    def test_missing_multipliers_fall_back_to_one(self):
        nums, slists, error = lm.parse_loras_multipliers("0.5", 3, 4)
        assert (nums, error) == ([0.5, 1.0, 1.0], "")
        assert slists["shared"] == [True, False, False]

    def test_extra_multipliers_are_truncated_to_nb_loras(self):
        nums, _, error = lm.parse_loras_multipliers("0.1 0.2 0.3", 2, 4)
        assert (nums, error) == ([0.1, 0.2], "")

    def test_explicit_switch_steps_are_echoed_back(self):
        _, slists, _ = lm.parse_loras_multipliers("0.5", 1, 10, model_switch_step=3, model_switch_step2=6)
        assert (slists["model_switch_step"], slists["model_switch_step2"]) == (3, 6)

    def test_returned_number_is_the_first_entry_of_a_step_list(self):
        nums, slists, error = lm.parse_loras_multipliers("0.1,0.9", 1, 4)
        assert (nums, error) == ([0.1], "")
        assert slists["phase1"] == [[0.1, 0.9]]

    def test_semicolon_splits_phases_and_disables_sharing(self):
        _, slists, error = lm.parse_loras_multipliers("0.8;0.5", 1, 6, model_switch_step=3)
        assert error == ""
        assert slists["phase1"] == [0.8]
        assert slists["phase2"] == [0.5]
        assert slists["phase3"] == [1.0]  # untouched default
        assert slists["shared"] == [False]

    def test_three_phases_are_accepted_when_nb_phases_is_three(self):
        _, slists, error = lm.parse_loras_multipliers(
            "0.8;0.5;0.2", 1, 6, nb_phases=3, model_switch_step=2, model_switch_step2=4
        )
        assert error == ""
        assert (slists["phase1"], slists["phase2"], slists["phase3"]) == ([0.8], [0.5], [0.2])

    def test_short_phase_list_is_padded_by_repeating_the_last_value(self):
        _, slists, _ = lm.parse_loras_multipliers(
            "0.8;0.5", 1, 6, nb_phases=3, model_switch_step=2, model_switch_step2=4
        )
        assert (slists["phase1"], slists["phase2"], slists["phase3"]) == ([0.8], [0.5], [0.5])

    def test_model_switch_phase_two_pads_by_repeating_the_first_value(self):
        _, slists, _ = lm.parse_loras_multipliers(
            "0.8;0.5", 1, 6, nb_phases=3, model_switch_step=2, model_switch_step2=4, model_switch_phase=2
        )
        assert (slists["phase1"], slists["phase2"], slists["phase3"]) == ([0.8], [0.8], [0.5])

    def test_per_phase_step_lists_are_kept_separate(self):
        _, slists, _ = lm.parse_loras_multipliers("0.1,0.9;0.3,0.7", 1, 4, model_switch_step=2)
        assert slists["phase1"] == [[0.1, 0.9]]
        assert slists["phase2"] == [[0.3, 0.7]]
        assert lm.expand_slist(slists, 0, 4, 2, 4) == [0.1, 0.9, 0.3, 0.7]

    def test_a_single_bar_is_accepted_and_acts_as_a_separator(self):
        nums, _, error = lm.parse_loras_multipliers("1|2", 2, 4)
        assert (nums, error) == ([1.0, 2.0], "")

    def test_negative_multipliers_are_allowed(self):
        nums, _, error = lm.parse_loras_multipliers("-1", 1, 4)
        assert (nums, error) == ([-1.0], "")

    def test_list_input_of_floats_is_accepted(self):
        nums, slists, error = lm.parse_loras_multipliers([0.5, 0.25], 2, 4)
        assert (nums, error) == ([0.5, 0.25], "")
        assert slists["shared"] == [True, True]

    def test_list_input_of_strings_still_honours_the_phase_syntax(self):
        _, slists, error = lm.parse_loras_multipliers(["0.5;0.25"], 1, 4, model_switch_step=2)
        assert error == ""
        assert (slists["phase1"], slists["phase2"], slists["shared"]) == ([0.5], [0.25], [False])

    def test_merge_slist_prepends_previously_parsed_loras(self):
        previous = {"phase1": [0.3], "phase2": [0.3], "phase3": [0.3], "shared": [True]}
        nums, slists, error = lm.parse_loras_multipliers("0.9", 1, 4, merge_slist=previous)
        assert (nums, error) == ([0.3, 0.9], "")
        assert slists["phase1"] == [0.3, 0.9]
        assert slists["shared"] == [True, True]


class TestParseLorasMultipliersErrors:
    """Errors are returned as a ``("", "", message)`` triple, never raised."""

    def test_two_bars_are_rejected(self):
        assert lm.parse_loras_multipliers("1|2|3", 2, 4) == (
            "",
            "",
            "There can be only one '|' character in Loras Multipliers Sequence",
        )

    def test_non_numeric_multiplier(self):
        assert lm.parse_loras_multipliers("abc", 1, 4) == (
            "",
            "",
            "Lora Multiplier no 1 (abc) is invalid",
        )

    def test_colon_without_declared_branches_is_not_special(self):
        # The ':' branch syntax is only recognised when lora_multiplier_branches is
        # supplied; otherwise the whole token is handed to float() and fails.
        assert lm.parse_loras_multipliers("0.5:0.25", 1, 4) == (
            "",
            "",
            "Lora Multiplier no 1 (0.5:0.25) is invalid",
        )


class TestParseLorasMultipliersBranches:
    def test_a_single_value_is_broadcast_to_every_branch(self):
        _, slists, error = lm.parse_loras_multipliers(
            "0.5", 1, 4, lora_multiplier_branches=["high", "low"]
        )
        assert error == ""
        assert slists["high"]["phase1"] == [0.5]
        assert slists["low"]["phase1"] == [0.5]

    def test_colon_assigns_one_value_per_branch(self):
        _, slists, error = lm.parse_loras_multipliers(
            "0.5:0.25", 1, 4, lora_multiplier_branches=["high", "low"]
        )
        assert error == ""
        assert slists["high"]["phase1"] == [0.5]
        assert slists["low"]["phase1"] == [0.25]
        # The top-level keys mirror the first branch.
        assert slists["phase1"] == [0.5]

    def test_branch_and_phase_syntax_combine(self):
        _, slists, error = lm.parse_loras_multipliers(
            "0.5:0.25;0.1:0.05", 1, 4, lora_multiplier_branches=["high", "low"], model_switch_step=2
        )
        assert error == ""
        assert (slists["high"]["phase1"], slists["high"]["phase2"]) == ([0.5], [0.1])
        assert (slists["low"]["phase1"], slists["low"]["phase2"]) == ([0.25], [0.05])

    def test_wrong_branch_count_is_reported(self):
        assert lm.parse_loras_multipliers(
            "0.5:0.25:0.1", 1, 4, lora_multiplier_branches=["high", "low"]
        ) == ("", "", "Lora Multiplier no 1 (0.5:0.25:0.1) should define 2 branch values separated by ':'")

    def test_merge_slist_is_applied_per_branch(self):
        previous = {
            "high": {"phase1": [0.3], "phase2": [0.3], "phase3": [0.3], "shared": [True]},
            "low": {"phase1": [0.1], "phase2": [0.1], "phase3": [0.1], "shared": [True]},
        }
        nums, slists, error = lm.parse_loras_multipliers(
            "0.5:0.25", 1, 4, merge_slist=previous, lora_multiplier_branches=["high", "low"]
        )
        assert (nums, error) == ([0.3, 0.5], "")
        assert slists["high"]["phase1"] == [0.3, 0.5]
        assert slists["low"]["phase1"] == [0.1, 0.25]

    def test_blank_branch_names_are_discarded(self):
        _, slists, error = lm.parse_loras_multipliers(
            "0.5", 1, 4, lora_multiplier_branches=["  ", ""]
        )
        assert error == ""
        assert "phase1" in slists and slists["phase1"] == [0.5]
        assert "  " not in slists


TIMESTEPS = [1000, 800, 600, 400, 200]


class TestGetModelSwitchSteps:
    def test_single_phase_never_switches(self):
        assert lm.get_model_switch_steps(TIMESTEPS, 1, 1, 700, 300) == (5, 5, "")

    def test_two_phases(self):
        step, step2, desc = lm.get_model_switch_steps(TIMESTEPS, 2, 1, 700, 300)
        assert (step, step2) == (2, 5)
        assert desc == "Denoising Steps:  Phase 1 = 1:2, Phase 2 = 3:5"

    @pytest.mark.parametrize(
        "switch, switch2, expected",
        [
            # 800 is itself a timestep: the comparison is `t <= threshold`, so the step
            # *at* the threshold already belongs to the next phase (index 1, not 2).
            (800, 300, (1, 5)),
            (200, 200, (4, 5)),  # the very last timestep equals the threshold
        ],
    )
    def test_a_timestep_equal_to_the_threshold_switches_on_that_step(self, switch, switch2, expected):
        # Boundary guard: with a strict `<` the switch would slip one step later, and
        # every other case in this class uses a threshold that falls between timesteps,
        # so nothing else here would notice.
        assert lm.get_model_switch_steps(TIMESTEPS, 2, 1, switch, switch2)[:2] == expected

    def test_second_threshold_boundary_is_also_inclusive(self):
        assert lm.get_model_switch_steps(TIMESTEPS, 3, 1, 800, 400) == (
            1,
            3,
            "Denoising Steps:  Phase 1 = 1:1, Phase 2 = 2:3, Phase 3 = 4:5",
        )

    def test_threshold_never_reached_falls_back_to_the_step_count(self):
        step, step2, desc = lm.get_model_switch_steps(TIMESTEPS, 2, 1, 100, 50)
        assert (step, step2) == (5, 5)
        assert desc == "Denoising Steps:  Phase 1 = 1:5"

    def test_threshold_reached_immediately_reports_an_empty_first_phase(self):
        step, step2, desc = lm.get_model_switch_steps(TIMESTEPS, 2, 1, 2000, 2000)
        assert (step, step2) == (0, 5)
        assert desc == "Denoising Steps:  Phase 1 = None, Phase 2 = 1:5"

    def test_equal_thresholds_report_an_empty_second_phase(self):
        step, step2, desc = lm.get_model_switch_steps(TIMESTEPS, 3, 1, 700, 700)
        assert (step, step2) == (2, 2)
        assert desc == "Denoising Steps:  Phase 1 = 1:2, Phase 2 = None, Phase 3 = 3:5"

    def test_third_phase_is_omitted_when_its_threshold_is_never_reached(self):
        # guide_phases == 3 but switch2_threshold sits below every timestep, so
        # model_switch_step2 falls back to the step count and the "Phase 3" clause is
        # skipped -- phase 2 absorbs the tail.
        step, step2, desc = lm.get_model_switch_steps(TIMESTEPS, 3, 1, 700, 50)
        assert (step, step2) == (2, 5)
        assert desc == "Denoising Steps:  Phase 1 = 1:2, Phase 2 = 3:5"

    def test_empty_timesteps(self):
        assert lm.get_model_switch_steps([], 3, 1, 700, 300) == (0, 0, "Denoising Steps:  Phase 1 = None")


class TestNumberTokenGrammar:
    """_spans has to agree with what parse_loras_multipliers accepts.

    It used to match a run of characters from a fixed set that omitted '-' and 'e', so a
    signed or exponent value was split mid-number. Because merge_loras_settings edits the
    original text by these offsets, a split number got half of it deleted.
    """

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("-1 -2", ["-1", "-2"]),
            ("1e5", ["1e5"]),
            ("1.5e-3", ["1.5e-3"]),
            ("+2 3", ["+2", "3"]),
            ("-0.5;-0.25", ["-0.5;-0.25"]),
        ],
    )
    def test_signed_and_exponent_values_are_single_tokens(self, text, expected):
        assert tokens(text) == expected

    @pytest.mark.parametrize(
        "multiplier, expected_value",
        [("-1", -1.0), ("1e5", 100000.0), ("-0.5", -0.5), ("1.5e-3", 0.0015)],
    )
    def test_values_the_parser_accepts_survive_a_merge(self, multiplier, expected_value):
        # The end-to-end defect: merging used to corrupt these into text that no longer
        # parses -- "1e5|" came back as "1e|0.5", and "-1 -2|3" as "-1 -|0.5".
        loras, merged = lm.merge_loras_settings(
            ["a"], f"{multiplier}|", ["b"], "0.5", "merge after"
        )
        assert loras == ["a", "b"]
        values, _slists, error = lm.parse_loras_multipliers(merged, len(loras), 4)
        assert error == "", f"{merged!r} no longer parses"
        assert values == [expected_value, 0.5]

    def test_a_negative_multiplier_keeps_its_sign_through_a_merge(self):
        _loras, merged = lm.merge_loras_settings(
            ["a", "b"], "-1 -2|", ["c"], "0.5", "merge after"
        )
        assert "-1" in merged, merged
        assert not merged.rstrip().endswith("-"), f"dangling sign left behind: {merged!r}"


class TestSpans:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("1 2 3", ["1", "2", "3"]),
            ("1.0,0.5;0.2", ["1.0,0.5;0.2"]),  # ':;,.' and digits are all token chars
            ("1|2", ["1", "2"]),
            ("   ", []),
            ("1 # comment 2 3", ["1"]),  # comment runs to end of line
            ("1\n#c\n2", ["1", "2"]),  # ... and a newline ends it
        ],
    )
    def test_tokenisation(self, text, expected):
        assert tokens(text) == expected

    def test_spans_are_offsets_into_the_original_text(self):
        assert lm._spans("ab 12 cd") == [(3, 5)]

    @pytest.mark.parametrize(
        "text, expected",
        [("a|b", 1), ("# a|b", -1), ("#a|b\nc|d", 6)],
    )
    def test_find_bar_ignores_bars_inside_comments(self, text, expected):
        assert lm._find_bar(text) == expected

    def test_choose_sep_looks_at_the_gap_before_the_last_token(self):
        # The separator is copied from the gap between the last two tokens, not from
        # the text as a whole: here the text does contain a newline, but the final gap
        # is a space, so appending must use a space.
        assert lm._choose_sep("1\n2 3", lm._spans("1\n2 3")) == " "
        # ... and the mirror image, where only the final gap is a newline.
        assert lm._choose_sep("1 2\n3", lm._spans("1 2\n3")) == "\n"

    @pytest.mark.parametrize(
        "text, expected",
        [("1 #c", True), ("1 #c\n2", False)],
    )
    def test_ends_in_comment_line(self, text, expected):
        assert lm._ends_in_comment_line(text) == expected


class TestEnforceCount:
    @pytest.mark.parametrize(
        "text, target, expected",
        [
            ("1 2 3", 2, "1 2"),
            ("1 2 3", 0, ""),
            ("1 2", 4, "1 2 1 1"),
            ("1\n2", 4, "1\n2\n1\n1"),  # the separator is copied from the text
        ],
    )
    def test_pads_with_ones_and_trims_from_the_end(self, text, target, expected):
        assert lm._enforce_count(text, target) == expected

    def test_trailing_comment_survives_trimming(self):
        assert lm._enforce_count("1 2 3 # c", 2) == "1 2 # c"

    def test_appending_after_a_comment_starts_a_new_line(self):
        assert lm._enforce_count("1 # c", 2) == "1 # c\n1"

    def test_append_tokens_does_not_double_the_separator(self):
        assert lm._append_tokens("1 ", 1, " ") == "1 1"
        assert lm._append_tokens("", 1, " ") == "1"


class TestTokenEditing:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("1 | 2", "1   2"),
            # Bars inside a comment are left alone; a newline ends the comment.
            ("1|2 # c|d\n3|4", "1 2 # c|d\n3 4"),
        ],
    )
    def test_strip_bars_outside_comments(self, text, expected):
        assert lm._strip_bars_outside_comments(text) == expected

    def test_stripping_a_bar_keeps_adjacent_tokens_apart(self):
        # A bar separates two multipliers, so removing it leaves a space behind.  It
        # used to be dropped with a bare `continue`, which fused the neighbours: "1|2"
        # became the single multiplier twelve.  This now agrees with
        # preparse_loras_multipliers, which tokenises via `.replace("|", " ")`.
        assert lm._strip_bars_outside_comments("1|2") == "1 2"
        assert len(lm._spans(lm._strip_bars_outside_comments("1|2"))) == 2

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("0.5|0.25|0.125", ["0.5", "0.25", "0.125"]),
            # A bar inside a comment is not a separator and must not split anything.
            ("1|2 # a|b", ["1", "2"]),
        ],
    )
    def test_stripping_bars_never_alters_a_multiplier_value(self, text, expected):
        # Expectations are written out literally rather than derived from the input:
        # computing them as `text.replace("|", " ").split()` would just re-implement the
        # function under test and assert it against itself.
        assert tokens(lm._strip_bars_outside_comments(text)) == expected

    def test_replace_tokens_with_longer_text_keeps_later_indices_valid(self):
        assert lm._replace_tokens("1 2 3", {0: "0.125", 1: "0.5"}) == "0.125 0.5 3"

    def test_replace_tokens_skips_over_comments(self):
        assert lm._replace_tokens("1 # c\n2", {1: "0.5"}) == "1 # c\n0.5"

    @pytest.mark.parametrize(
        "text, idxs, expected",
        [
            ("1 2 3", [1], "1 3"),
            ("1 2 3", [0, 2], "2"),
            ("1\n2\n3", [1], "1\n\n3"),  # newline separators are not reclaimed
        ],
    )
    def test_drop_tokens_by_indices(self, text, idxs, expected):
        assert lm._drop_tokens_by_indices(text, idxs) == expected

    @pytest.mark.parametrize(
        "path, expected",
        [
            (r" a\b//c/ ", "a/b/c"),
            ("/", "/"),  # a lone slash is not stripped
            ("x", "x"),
        ],
    )
    def test_default_path_key_normalises_separators(self, path, expected):
        assert lm._default_path_key(path) == expected


class TestSelectNewSide:
    @pytest.mark.parametrize(
        "loras, mult, mode, expected",
        [
            (["x", "y"], "0.1|0.2", "merge after", (["y"], "0.2")),
            # Loras with no matching token become "extras" appended to the side.
            (["x", "y", "z"], "0.1|0.2", "merge before", (["x", "z"], "0.1")),
        ],
    )
    def test_splits_new_set_on_the_bar(self, loras, mult, mode, expected):
        assert lm._select_new_side(loras, mult, mode) == expected

    def test_a_second_bar_becomes_a_separator_rather_than_fusing(self):
        # Only the first bar splits the two sides; _strip_bars_outside_comments reduces
        # any further one to a separator.  It used to delete it outright, collapsing
        # "2|3" into the single token "23" -- a 23x LoRA strength -- leaving two loras
        # on the "after" side with only one multiplier to describe them.
        loras, mult = lm._select_new_side(["x", "y", "z"], "1|2|3", "merge after")
        assert loras == ["y", "z"]
        assert tokens(mult) == ["2", "3"]


class TestMergeLorasSettings:
    def test_merge_before_into_an_empty_set_marks_the_before_side(self):
        assert lm.merge_loras_settings([], "", ["a"], "0.5", "merge before") == (["a"], "0.5|")

    def test_merge_before_keeps_the_unbarred_old_set_as_the_after_side(self):
        assert lm.merge_loras_settings(["a"], "1", ["b"], "0.5", "merge before") == (["b", "a"], "0.5|1")

    def test_merge_after_replaces_an_unbarred_old_set_entirely(self):
        # Without a bar every old lora belongs to the "after" side, which is the
        # side being replaced.
        assert lm.merge_loras_settings(["a"], "1", ["b"], "0.5", "merge after") == (["b"], "0.5")

    def test_duplicate_updates_the_preserved_side_and_drops_the_new_entry(self):
        assert lm.merge_loras_settings(["a", "b"], "1|2", ["a"], "0.7", "merge after") == (
            ["a"],
            "0.7|",
        )

    def test_duplicate_on_the_preserved_after_side(self):
        assert lm.merge_loras_settings(["a", "b"], "1|2", ["b"], "0.7", "merge before") == (
            ["b"],
            "0.7",
        )

    def test_partial_duplicate_keeps_the_genuinely_new_lora(self):
        assert lm.merge_loras_settings(["a", "b"], "1 2|3", ["b", "z"], "0.7 0.8", "merge after") == (
            ["a", "b", "z"],
            "1 0.7|0.8",
        )

    def test_bar_in_the_new_multipliers_selects_the_matching_side(self):
        assert lm.merge_loras_settings(["a"], "1", ["x", "y"], "0.1|0.2", "merge before") == (
            ["x", "a"],
            "0.1|1",
        )
        assert lm.merge_loras_settings(["a"], "1", ["x", "y"], "0.1|0.2", "merge after") == (
            ["y"],
            "0.2",
        )

    def test_missing_new_multipliers_are_padded_with_one(self):
        assert lm.merge_loras_settings(["a"], "1|", ["x", "y", "z"], "0.5", "merge after") == (
            ["a", "x", "y", "z"],
            "1|0.5 1 1",
        )

    def test_more_old_before_tokens_than_old_loras_collapses_the_after_side(self):
        # The `n_b_old > total_old` branch: the before side is trimmed down to the whole
        # lora list and the after side is emptied, so "merge after" has nothing to
        # preserve on the right and "merge before" has nothing to preserve at all.
        assert lm.merge_loras_settings(["a"], "1 2|3", ["c"], "0.5", "merge after") == (
            ["a", "c"],
            "1|0.5",
        )
        assert lm.merge_loras_settings(["a"], "1 2|3", ["c"], "0.5", "merge before") == (
            ["c"],
            "0.5|",
        )

    def test_a_second_bar_in_the_new_multipliers_keeps_its_value(self):
        # The user-visible end of the _strip_bars_outside_comments fix.  Lora "y" keeps
        # the 2x it was given; this used to return "1|23", silently applying a 23x
        # strength.  parse_loras_multipliers rejects "1|2|3" outright ("There can be
        # only one '|' character") -- merge_loras_settings accepts it, and now no
        # longer corrupts it.
        loras, mult = lm.merge_loras_settings(["a"], "1|", ["x", "y"], "1|2|3", "merge after")
        assert (loras, mult) == (["a", "y"], "1|2")
        assert "23" not in mult

    def test_missing_old_multipliers_are_padded_before_splitting(self):
        assert lm.merge_loras_settings(["a", "b", "c"], "1|2", ["d"], "0.5", "merge before") == (
            ["d", "b", "c"],
            "0.5|2 1",
        )

    def test_comments_on_the_preserved_side_survive(self):
        assert lm.merge_loras_settings(
            ["a", "b"], "1 # first\n|2 # second", ["c"], "0.5", "merge after"
        ) == (["a", "c"], "1 # first\n|0.5")
        assert lm.merge_loras_settings(
            ["a", "b"], "1 # first\n|2 # second", ["c"], "0.5", "merge before"
        ) == (["c", "b"], "0.5|2 # second")

    def test_whitespace_around_the_multipliers_is_stripped(self):
        assert lm.merge_loras_settings(["a"], "  1|  ", ["b"], "  0.5  ", "merge after") == (
            ["a", "b"],
            "1|0.5",
        )

    def test_a_custom_path_key_controls_dedupe(self):
        assert lm.merge_loras_settings(["A"], "1|", ["a"], "0.7", "merge after", str.lower) == (
            ["A"],
            "0.7|",
        )
        assert lm.merge_loras_settings(["A"], "1|", ["a"], "0.7", "merge after") == (
            ["A", "a"],
            "1|0.7",
        )

    def test_result_is_reparseable(self):
        loras, mult = lm.merge_loras_settings(["a", "b"], "1|2", ["c"], "0.5", "merge after")
        nums, _, error = lm.parse_loras_multipliers(mult, len(loras), 4)
        assert error == ""
        assert nums == [1.0, 0.5]


class TestExtractLorasSide:
    def test_splits_on_the_bar(self):
        assert lm.extract_loras_side(["a", "b", "c"], "1 2|3", "before") == (["a", "b"], "1 2")
        assert lm.extract_loras_side(["a", "b", "c"], "1 2|3", "after") == (["c"], "3")

    def test_without_a_bar_everything_is_on_the_after_side(self):
        assert lm.extract_loras_side(["a", "b", "c"], "1 2 3", "before") == ([], "")
        assert lm.extract_loras_side(["a", "b", "c"], "1 2 3", "after") == (["a", "b", "c"], "1 2 3")

    def test_more_before_tokens_than_loras_truncates_and_empties_the_after_side(self):
        assert lm.extract_loras_side(["a", "b"], "1 2 3 4|5", "before") == (["a", "b"], "1 2")
        assert lm.extract_loras_side(["a", "b"], "1 2 3 4|5", "after") == ([], "")

    def test_missing_after_multipliers_are_padded_with_one(self):
        assert lm.extract_loras_side(["a", "b", "c"], "1|", "after") == (["b", "c"], "1 1")
