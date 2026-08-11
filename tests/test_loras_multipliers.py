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

Several expectations below pin behaviour that looks accidental rather than
intended; those are flagged with a ``BUG:`` comment.
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
            ("1.0 0.5", ["1.0", "0.5"]),
            ("  1.0 0.5  ", ["1.0", "0.5"]),
            ("1.0\n0.5", ["1.0", "0.5"]),
            ("1.0\r\n0.5", ["1.0", "0.5"]),
            # '|' is only a separator here, the before/after split happens later.
            ("1.0|0.5", ["1.0", "0.5"]),
            # ';' and ',' are kept inside the token, they are parsed downstream.
            ("1.0;0.5 0.3", ["1.0;0.5", "0.3"]),
            ("0.1,0.9", ["0.1,0.9"]),
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

    @pytest.mark.parametrize("raw", ["", "\n", "   ", "# only a comment"])
    def test_input_without_values_yields_a_single_empty_token(self, raw):
        # Note the empty *token* rather than an empty list: "".split(" ") == [""].
        # parse_loras_multipliers skips preparse entirely for "" but not for "   " or a
        # comment-only string, so those two reach it and are rejected -- see
        # TestParseLorasMultipliersErrors.test_input_with_no_values_is_rejected.
        assert lm.preparse_loras_multipliers(raw) == [""]

    def test_consecutive_spaces_produce_empty_tokens(self):
        # BUG: the split is `.split(" ")` rather than `.split()`, so a double
        # space leaves an empty token behind which parse_loras_multipliers
        # later reports as an invalid multiplier.
        assert lm.preparse_loras_multipliers("1.0  0.5") == ["1.0", "", "0.5"]

    def test_non_numeric_tokens_are_passed_through_untouched(self):
        assert lm.preparse_loras_multipliers("a b") == ["a", "b"]

    def test_list_input_is_stripped_elementwise_and_non_strings_kept(self):
        assert lm.preparse_loras_multipliers(["1.0 ", " 0.5\n", 3]) == ["1.0", "0.5", 3]

    def test_none_raises(self):
        with pytest.raises(AttributeError):
            lm.preparse_loras_multipliers(None)


def _slists(phase1, phase2=1.0, phase3=1.0, shared=False):
    return {"phase1": [phase1], "phase2": [phase2], "phase3": [phase3], "shared": [shared]}


class TestExpandSlist:
    def test_shared_scalar_stays_a_scalar(self):
        assert lm.expand_slist(_slists(0.5, shared=True), 0, 4, 2, 4) == 0.5

    @pytest.mark.parametrize(
        "values, steps, expected",
        [
            ([0.0, 1.0], 4, [0.0, 0.0, 1.0, 1.0]),
            ([0.0, 1.0, 2.0], 4, [0.0, 0.0, 1.0, 2.0]),
            ([0.1, 0.9], 5, [0.1, 0.1, 0.1, 0.9, 0.9]),
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

    def test_three_phases_split_at_both_switch_steps(self):
        got = lm.expand_slist(_slists(0.8, 0.5, 0.2), 0, 6, 2, 4)
        assert got == [0.8, 0.8, 0.5, 0.5, 0.2, 0.2]

    def test_per_phase_lists_are_expanded_independently(self):
        got = lm.expand_slist(_slists([0.1, 0.2], 0.5, 0.9), 0, 6, 2, 4)
        assert got == [0.1, 0.2, 0.5, 0.5, 0.9, 0.9]

    def test_switch_step_equal_to_total_steps_uses_phase1_only(self):
        assert lm.expand_slist(_slists(0.8, 0.5, 0.2), 0, 4, 4, 4) == [0.8] * 4

    def test_switch_step_of_zero_skips_straight_to_phase3(self):
        assert lm.expand_slist(_slists(0.8, 0.5, 0.2), 0, 4, 0, 0) == [0.2] * 4

    def test_int_multipliers_are_not_treated_as_scalars(self):
        # The scalar short-circuits test `isinstance(x, float)`, so plain ints
        # always go through the list expansion path.
        assert lm.expand_slist(_slists(1, 1, 1, shared=True), 0, 3, 1, 2) == [1, 1, 1]
        assert lm.expand_slist(_slists(1, 1, 1), 0, 3, 1, 2) == [1, 1, 1]

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

    def test_zero_loras_gives_empty_schedules(self):
        nums, slists, error = lm.parse_loras_multipliers("0.5", 0, 4)
        assert (nums, error) == ([], "")
        assert slists["phase1"] == []

    def test_switch_steps_default_to_num_inference_steps(self):
        _, slists, _ = lm.parse_loras_multipliers("0.5", 1, 7)
        assert slists["model_switch_step"] == 7
        assert slists["model_switch_step2"] == 7

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

    def test_semicolon_schedule_expands_across_the_switch_step(self):
        _, slists, _ = lm.parse_loras_multipliers("0.8;0.5", 1, 6, model_switch_step=3)
        assert lm.expand_slist(slists, 0, 6, 3, 6) == [0.8, 0.8, 0.8, 0.5, 0.5, 0.5]

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

    def test_comment_lines_are_ignored(self):
        nums, _, error = lm.parse_loras_multipliers("# a note\n0.5\n0.25", 2, 4)
        assert (nums, error) == ([0.5, 0.25], "")

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

    def test_non_numeric_step_value_reports_the_split_list(self):
        assert lm.parse_loras_multipliers("0.1,abc", 1, 4) == (
            "",
            "",
            "Lora sub value no 1 (abc) in Multiplier definition '['0.1', 'abc']' is invalid in Phase 1",
        )

    def test_double_space_is_reported_as_an_invalid_multiplier(self):
        # BUG: consequence of preparse splitting on a single space -- "1.0  0.5"
        # is a reasonable thing for a user to type but is rejected.
        assert lm.parse_loras_multipliers("1.0  0.5", 2, 4) == (
            "",
            "",
            "Lora Multiplier no 2 () is invalid",
        )

    @pytest.mark.parametrize("nb_phases", [1, 2])
    def test_more_phases_than_nb_phases(self, nb_phases):
        _, _, error = lm.parse_loras_multipliers("0.8;0.5;0.2", 1, 6, nb_phases=nb_phases)
        assert error == (
            "if the ';' syntax is used for one Lora multiplier, there should be "
            f"at most {nb_phases} phases for this multiplier"
        )

    def test_none_raises_a_type_error(self):
        with pytest.raises(TypeError):
            lm.parse_loras_multipliers(None, 1, 4)

    @pytest.mark.parametrize("raw", ["   ", "# only a comment", "# a\n# b"])
    def test_input_with_no_values_is_rejected(self, raw):
        # BUG: the `len(loras_multipliers) > 0` guard is applied to the *raw* string, so
        # whitespace-only and comment-only boxes get past it, preparse hands back [""],
        # and the user sees a confusing "no 1 () is invalid".  An entirely empty box is
        # fine (see test_empty_string_defaults_every_lora_to_one) -- adding a comment to
        # it is what breaks it.
        assert lm.parse_loras_multipliers(raw, 1, 4) == (
            "",
            "",
            "Lora Multiplier no 1 () is invalid",
        )

    def test_colon_without_declared_branches_is_not_special(self):
        # The ':' branch syntax is only recognised when lora_multiplier_branches is
        # supplied; otherwise the whole token is handed to float() and fails.
        assert lm.parse_loras_multipliers("0.5:0.25", 1, 4) == (
            "",
            "",
            "Lora Multiplier no 1 (0.5:0.25) is invalid",
        )

    def test_list_containing_a_bar_raises(self):
        # BUG: the "only one '|'" guard uses `in` (element test on a list) and
        # then `.find`, which lists do not have.
        with pytest.raises(AttributeError):
            lm.parse_loras_multipliers(["|", "1"], 2, 4)


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

    def test_three_phases(self):
        step, step2, desc = lm.get_model_switch_steps(TIMESTEPS, 3, 1, 700, 300)
        assert (step, step2) == (2, 4)
        assert desc == "Denoising Steps:  Phase 1 = 1:2, Phase 2 = 3:4, Phase 3 = 5:5"

    def test_second_threshold_is_ignored_below_three_phases(self):
        assert lm.get_model_switch_steps(TIMESTEPS, 2, 1, 700, 300)[1] == 5

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

    @pytest.mark.parametrize("model_switch_phase", [1, 2, 3])
    def test_model_switch_phase_is_ignored(self, model_switch_phase):
        # BUG (harmless): the parameter is accepted but never read.
        assert lm.get_model_switch_steps(TIMESTEPS, 3, model_switch_phase, 700, 300) == (
            2,
            4,
            "Denoising Steps:  Phase 1 = 1:2, Phase 2 = 3:4, Phase 3 = 5:5",
        )

    def test_switch_step_feeds_expand_slist(self):
        step, step2, _ = lm.get_model_switch_steps(TIMESTEPS, 2, 1, 700, 300)
        assert lm.expand_slist(_slists(0.8, 0.5, 0.5), 0, len(TIMESTEPS), step, step2) == [
            0.8,
            0.8,
            0.5,
            0.5,
            0.5,
        ]


class TestSpans:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("1 2 3", ["1", "2", "3"]),
            ("1.0,0.5;0.2", ["1.0,0.5;0.2"]),  # ':;,.' and digits are all token chars
            ("1|2", ["1", "2"]),
            ("", []),
            ("   ", []),
            ("#c", []),
            ("1 # comment 2 3", ["1"]),  # comment runs to end of line
            ("1 # c\n2", ["1", "2"]),
            ("1\n#c\n2", ["1", "2"]),
        ],
    )
    def test_tokenisation(self, text, expected):
        assert tokens(text) == expected

    def test_minus_sign_is_not_part_of_a_token(self):
        # BUG: '-' is missing from the allowed character set, so a negative
        # multiplier is tokenised as its absolute value by the merge helpers.
        assert tokens("-1 2") == ["1", "2"]

    def test_exponent_notation_is_split(self):
        # BUG: same root cause as above -- 'e' is not in _ALWD either, so "1e5" is two
        # tokens to the merge helpers even though float() accepts it as one number.
        assert tokens("1e5") == ["1", "5"]

    def test_spans_are_offsets_into_the_original_text(self):
        assert lm._spans("ab 12 cd") == [(3, 5)]

    @pytest.mark.parametrize(
        "text, expected",
        [("a|b", 1), ("no bar", -1), ("|", 0), ("# a|b", -1), ("#a|b\nc|d", 6)],
    )
    def test_find_bar_ignores_bars_inside_comments(self, text, expected):
        assert lm._find_bar(text) == expected

    @pytest.mark.parametrize(
        "text, expected",
        [("1 2", " "), ("1\n2", "\n"), ("1", " "), ("", " "), ("1 2\n3", "\n")],
    )
    def test_choose_sep_copies_the_last_separator(self, text, expected):
        assert lm._choose_sep(text, lm._spans(text)) == expected

    @pytest.mark.parametrize(
        "text, expected",
        [("1 #c", True), ("1 #c\n2", False), ("1", False), ("#c\n", False), ("", False)],
    )
    def test_ends_in_comment_line(self, text, expected):
        assert lm._ends_in_comment_line(text) == expected


class TestEnforceCount:
    @pytest.mark.parametrize(
        "text, target, expected",
        [
            ("1 2 3", 3, "1 2 3"),
            ("1 2 3", 2, "1 2"),
            ("1 2 3", 0, ""),
            ("1 2", 4, "1 2 1 1"),
            ("", 3, "1 1 1"),
            ("", 0, ""),
            ("1\n2", 4, "1\n2\n1\n1"),
        ],
    )
    def test_pads_with_ones_and_trims_from_the_end(self, text, target, expected):
        assert lm._enforce_count(text, target) == expected

    def test_trailing_comment_survives_trimming(self):
        assert lm._enforce_count("1 2 3 # c", 2) == "1 2 # c"

    def test_appending_after_a_comment_starts_a_new_line(self):
        assert lm._enforce_count("1 # c", 2) == "1 # c\n1"

    def test_trimming_can_leave_a_dangling_comment_line(self):
        assert lm._enforce_count("1 # c\n2 3", 1) == "1 # c\n"

    @pytest.mark.parametrize(
        "text, drop, expected",
        [
            ("1 2 3", 0, "1 2 3"),
            ("1 2 3", 1, "1 2"),
            ("1 2 3", 2, "1"),
            ("1 2 3", 5, ""),
            ("1 2 3 ", 1, "1 2"),
            ("1\n2\n3", 1, "1\n2\n"),  # newline separators are not reclaimed
        ],
    )
    def test_trim_last_tokens(self, text, drop, expected):
        assert lm._trim_last_tokens(text, lm._spans(text), drop) == expected

    @pytest.mark.parametrize(
        "text, span, expected",
        [
            ("1 2 3", (2, 3), "1 3"),  # eats the space after
            ("1 2 3", (4, 5), "1 2"),  # no space after, eats the one before
            ("12", (0, 2), ""),
        ],
    )
    def test_erase_span_and_one_sep(self, text, span, expected):
        assert lm._erase_span_and_one_sep(text, *span) == expected

    def test_append_tokens_is_a_noop_for_non_positive_counts(self):
        assert lm._append_tokens("1 2", 0, " ") == "1 2"
        assert lm._append_tokens("1 2", -1, " ") == "1 2"

    def test_append_tokens_does_not_double_the_separator(self):
        assert lm._append_tokens("1 ", 1, " ") == "1 1"
        assert lm._append_tokens("", 1, " ") == "1"


class TestTokenEditing:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("#a|b", "#a|b"),  # bars inside a comment are left alone
            ("1 | 2", "1   2"),
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

    def test_stripping_bars_keeps_the_count_however_they_were_spaced(self):
        # The old fusion only showed up in the unspaced spelling, which is how it
        # survived unnoticed; both forms now yield the same tokens.
        assert tokens(lm._strip_bars_outside_comments("1 | 2")) == ["1", "2"]
        assert tokens(lm._strip_bars_outside_comments("1|2")) == ["1", "2"]
        assert tokens(lm._strip_bars_outside_comments("1|2\n3|4")) == ["1", "2", "3", "4"]

    def test_stripping_bars_never_alters_a_multiplier_value(self):
        for text in ("1|2", "0.5|0.25|0.125", "1 | 2", "1|2\n3|4", "10|20"):
            assert tokens(lm._strip_bars_outside_comments(text)) == text.replace("|", " ").split()

    def test_replace_tokens_by_index(self):
        assert lm._replace_tokens("1 2 3", {0: "9", 2: "7"}) == "9 2 7"

    def test_replace_tokens_with_longer_text_keeps_later_indices_valid(self):
        assert lm._replace_tokens("1 2 3", {0: "0.125", 1: "0.5"}) == "0.125 0.5 3"

    def test_replace_tokens_ignores_empty_map_and_out_of_range_indices(self):
        assert lm._replace_tokens("1 2 3", {}) == "1 2 3"
        assert lm._replace_tokens("1 2 3", {5: "9"}) == "1 2 3"
        assert lm._replace_tokens("1 2 3", {-1: "9"}) == "1 2 3"

    def test_replace_tokens_skips_over_comments(self):
        assert lm._replace_tokens("1 # c\n2", {1: "0.5"}) == "1 # c\n0.5"

    @pytest.mark.parametrize(
        "text, idxs, expected",
        [
            ("1 2 3", [], "1 2 3"),
            ("1 2 3", [1], "1 3"),
            ("1 2 3", [0, 2], "2"),
            ("1 2 3", [9], "1 2 3"),
            ("1 2 3", [1, 1], "1 3"),  # duplicates are de-duplicated
            ("1\n2\n3", [1], "1\n\n3"),
        ],
    )
    def test_drop_tokens_by_indices(self, text, idxs, expected):
        assert lm._drop_tokens_by_indices(text, idxs) == expected

    @pytest.mark.parametrize(
        "path, expected",
        [
            (r" a\b//c/ ", "a/b/c"),
            ("a/b/", "a/b"),
            ("/", "/"),
            ("x", "x"),
            ("", ""),
        ],
    )
    def test_default_path_key_normalises_separators(self, path, expected):
        assert lm._default_path_key(path) == expected


class TestSelectNewSide:
    @pytest.mark.parametrize(
        "loras, mult, mode, expected",
        [
            (["x", "y"], "0.1|0.2", "merge before", (["x"], "0.1")),
            (["x", "y"], "0.1|0.2", "merge after", (["y"], "0.2")),
            # Loras with no matching token become "extras" appended to the side.
            (["x", "y", "z"], "0.1|0.2", "merge before", (["x", "z"], "0.1")),
            (["x", "y", "z"], "0.1|0.2", "merge after", (["y", "z"], "0.2")),
            (["x"], "0.1", "merge after", (["x"], "0.1")),
            (["x"], "0.1", "merge before", (["x"], "0.1")),
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

    def test_a_second_bar_is_harmless_on_the_side_that_precedes_it(self):
        # The "before" side stops at the first bar, so it never sees the second one.
        assert lm._select_new_side(["x", "y", "z"], "1|2|3", "merge before") == (["x"], "1")


class TestMergeLorasSettings:
    def test_rejects_an_unknown_mode(self):
        with pytest.raises(AssertionError):
            lm.merge_loras_settings(["a"], "1", ["b"], "1", "nope")

    def test_merge_before_into_an_empty_set_marks_the_before_side(self):
        assert lm.merge_loras_settings([], "", ["a"], "0.5", "merge before") == (["a"], "0.5|")

    def test_merge_after_into_an_empty_set_needs_no_bar(self):
        assert lm.merge_loras_settings([], "", ["a"], "0.5", "merge after") == (["a"], "0.5")

    def test_merge_before_keeps_the_unbarred_old_set_as_the_after_side(self):
        assert lm.merge_loras_settings(["a"], "1", ["b"], "0.5", "merge before") == (["b", "a"], "0.5|1")

    def test_merge_after_replaces_an_unbarred_old_set_entirely(self):
        # Without a bar every old lora belongs to the "after" side, which is the
        # side being replaced.
        assert lm.merge_loras_settings(["a"], "1", ["b"], "0.5", "merge after") == (["b"], "0.5")

    def test_merge_before_replaces_only_the_before_side(self):
        assert lm.merge_loras_settings(["a", "b"], "1|2", ["c"], "0.5", "merge before") == (
            ["c", "b"],
            "0.5|2",
        )

    def test_merge_after_replaces_only_the_after_side(self):
        assert lm.merge_loras_settings(["a", "b"], "1|2", ["c"], "0.5", "merge after") == (
            ["a", "c"],
            "1|0.5",
        )

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

    def test_empty_new_multipliers_default_to_one(self):
        assert lm.merge_loras_settings(["a"], "1|", ["b", "c"], "", "merge after") == (
            ["a", "b", "c"],
            "1|1 1",
        )

    def test_surplus_old_multipliers_are_trimmed_to_the_lora_count(self):
        assert lm.merge_loras_settings(["a"], "1 2 3", ["c"], "0.5", "merge after") == (["c"], "0.5")

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

    def test_dedupe_uses_the_normalised_path_key(self):
        assert lm.merge_loras_settings(
            ["loras/a.safetensors"], "1|", ["loras//a.safetensors"], "0.7", "merge after"
        ) == (["loras/a.safetensors"], "0.7|")

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
    def test_rejects_an_unknown_side(self):
        with pytest.raises(AssertionError):
            lm.extract_loras_side(["a"], "1", "sideways")

    def test_splits_on_the_bar(self):
        assert lm.extract_loras_side(["a", "b", "c"], "1 2|3", "before") == (["a", "b"], "1 2")
        assert lm.extract_loras_side(["a", "b", "c"], "1 2|3", "after") == (["c"], "3")

    def test_without_a_bar_everything_is_on_the_after_side(self):
        assert lm.extract_loras_side(["a", "b", "c"], "1 2 3", "before") == ([], "")
        assert lm.extract_loras_side(["a", "b", "c"], "1 2 3", "after") == (["a", "b", "c"], "1 2 3")

    def test_empty_input(self):
        assert lm.extract_loras_side([], "", "before") == ([], "")
        assert lm.extract_loras_side([], "", "after") == ([], "")

    def test_more_before_tokens_than_loras_truncates_and_empties_the_after_side(self):
        assert lm.extract_loras_side(["a", "b"], "1 2 3 4|5", "before") == (["a", "b"], "1 2")
        assert lm.extract_loras_side(["a", "b"], "1 2 3 4|5", "after") == ([], "")

    def test_missing_after_multipliers_are_padded_with_one(self):
        assert lm.extract_loras_side(["a", "b", "c"], "1|", "after") == (["b", "c"], "1 1")

    def test_the_two_sides_partition_the_lora_list(self):
        # Asserting only `before + after == loras` would hold for *any* split point, so
        # pin where the cut actually falls (after the two "before" tokens) as well.
        loras = ["a", "b", "c", "d"]
        before_loras, before_mult = lm.extract_loras_side(loras, "1 2|3 4", "before")
        after_loras, after_mult = lm.extract_loras_side(loras, "1 2|3 4", "after")
        assert (before_loras, before_mult) == (["a", "b"], "1 2")
        assert (after_loras, after_mult) == (["c", "d"], "3 4")
        assert before_loras + after_loras == loras
