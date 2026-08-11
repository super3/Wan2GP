"""Tests for ``shared/utils/prompt_parser.py``.

The module turns the contents of the prompt box into generation-queue requests.
Covered here:

* ``normalize_multi_prompts_mode`` -- the "G"/"PG"/"W"/"PW"/"FG" mode codes plus
  the legacy int/str aliases and the fallback to ``default``.
* ``get_multi_prompts_gen_choices`` -- the dropdown labels/values.
* ``split_prompt_units`` / ``split_prompt_original_units`` -- line vs paragraph
  vs single-prompt splitting, ``#`` comment lines, CRLF and lone-CR
  normalisation, the ``!enhanced!`` prefix and the ``#!PROMPT!:`` unit markers.
* ``serialize_prompt_units`` / ``serialize_prompt_blocks_with_prefix`` -- the
  inverse operations, including round-trips.
* ``is_speaker_options_line`` -- the ``Speaker 1 {..}:`` recogniser.
* ``process_template`` and the macro helpers (``process_current_template``,
  ``extract_variable_names``, ``extract_variable_values``,
  ``generate_macro_line``).

A handful of tests deliberately pin behaviour that looks wrong; each one carries
a comment saying so.  See the accompanying report for the details.
"""

import pytest

from conftest import import_pure_module

prompt_parser = import_pure_module("shared.utils.prompt_parser")

PREFIX = prompt_parser.PROMPT_UNIT_PREFIX
ENHANCED = prompt_parser.ENHANCED_PROMPT_PREFIX


class TestConstants:
    def test_marker_constants(self):
        assert PREFIX == "#!PROMPT!:"
        assert ENHANCED == "!enhanced!\n"
        assert prompt_parser.DEFAULT_MULTI_PROMPTS_MODE == "PG"

    def test_unit_prefix_also_looks_like_a_comment(self):
        # Every helper that drops "#" comment lines therefore also hides the
        # marker lines from the user-visible prompt.
        assert PREFIX.startswith("#")


class TestNormalizeMultiPromptsMode:
    @pytest.mark.parametrize("value", ["G", "PG", "W", "PW", "FG"])
    def test_canonical_codes_pass_through(self, value):
        assert prompt_parser.normalize_multi_prompts_mode(value) == value

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("g", "G"),
            (" pw ", "PW"),
            ("\tfg\n", "FG"),
            ("Pg", "PG"),
        ],
    )
    def test_strings_are_stripped_and_upper_cased(self, value, expected):
        assert prompt_parser.normalize_multi_prompts_mode(value) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0, "G"),
            (1, "W"),
            (2, "FG"),
            ("0", "G"),
            ("1", "W"),
            ("2", "FG"),
            ("P", "PG"),
            ("p", "PG"),
        ],
    )
    def test_legacy_aliases(self, value, expected):
        assert prompt_parser.normalize_multi_prompts_mode(value) == expected

    @pytest.mark.parametrize("value", ["", "   ", "\n"])
    def test_blank_string_means_full_prompt(self, value):
        # The empty string is an alias for "FG", *not* a fallback to `default`.
        assert prompt_parser.normalize_multi_prompts_mode(value, default="W") == "FG"

    @pytest.mark.parametrize("value", [1.0, 1.9, 1.2])
    def test_floats_are_truncated_to_int(self, value):
        assert prompt_parser.normalize_multi_prompts_mode(value) == "W"

    @pytest.mark.parametrize("value", [True, False])
    def test_bools_follow_the_int_aliases(self, value):
        # bool is a subclass of int, so True == 1 -> "W" and False == 0 -> "G".
        assert prompt_parser.normalize_multi_prompts_mode(value) == ("W" if value else "G")

    @pytest.mark.parametrize("value", [None, [], {}, object(), 3, -1, "3", "nope", "PGW"])
    def test_unknown_values_fall_back_to_default(self, value):
        assert prompt_parser.normalize_multi_prompts_mode(value) == "G"
        assert prompt_parser.normalize_multi_prompts_mode(value, default="PW") == "PW"

    def test_default_is_not_validated(self):
        # An unrecognised value returns `default` verbatim, whatever it is.
        assert prompt_parser.normalize_multi_prompts_mode("nope", default=None) is None


class TestGetMultiPromptsGenChoices:
    def test_all_modes_offered_by_default(self):
        choices = prompt_parser.get_multi_prompts_gen_choices()
        assert [code for _, code in choices] == ["G", "PG", "W", "PW", "FG"]

    def test_sliding_window_modes_can_be_hidden(self):
        choices = prompt_parser.get_multi_prompts_gen_choices(include_sliding_window=False)
        assert [code for _, code in choices] == ["G", "PG", "FG"]

    def test_medium_is_interpolated_into_the_queue_labels(self):
        labels = [label for label, _ in prompt_parser.get_multi_prompts_gen_choices(medium="Image")]
        assert labels[0] == "Each New Line Will Add a new Image Request to the Generation Queue"
        assert "new Image Request" in labels[1]
        # The sliding-window and full-prompt labels are fixed text.
        assert "Image" not in labels[2]
        assert labels[-1] == "All the Lines are Part of the Same Prompt"

    def test_every_code_normalizes_to_itself(self):
        for _, code in prompt_parser.get_multi_prompts_gen_choices():
            assert prompt_parser.normalize_multi_prompts_mode(code) == code


class TestSplitPromptUnits:
    def test_one_request_per_line(self):
        assert prompt_parser.split_prompt_units("a\nb\nc", "G") == ["a", "b", "c"]

    def test_line_mode_strips_and_drops_blank_lines(self):
        assert prompt_parser.split_prompt_units("  a  \n\n\t\n b ", "G") == ["a", "b"]

    def test_sliding_window_mode_splits_per_line_too(self):
        assert prompt_parser.split_prompt_units("a\nb", "W") == ["a", "b"]

    @pytest.mark.parametrize("mode", ["PG", "PW"])
    def test_paragraph_modes_split_on_blank_lines(self, mode):
        text = "a\nb\n\n\nc\n \nd"
        assert prompt_parser.split_prompt_units(text, mode) == ["a\nb", "c", "d"]

    def test_full_prompt_mode_keeps_everything_as_one_unit(self):
        assert prompt_parser.split_prompt_units("a  \n\nb", "FG") == ["a\n\nb"]

    def test_single_prompt_overrides_the_mode(self):
        assert prompt_parser.split_prompt_units("a\nb", "W", single_prompt=True) == ["a\nb"]
        assert prompt_parser.split_prompt_units("a\n\nb", "PG", single_prompt=True) == ["a\n\nb"]

    @pytest.mark.parametrize("mode", ["G", "PG", "W", "PW", "FG", "", None])
    @pytest.mark.parametrize("text", ["", "   ", "\n\n", "\r\n \r\n", "# only a comment"])
    def test_empty_input_yields_no_units(self, text, mode):
        assert prompt_parser.split_prompt_units(text, mode) == []

    def test_missing_mode_behaves_like_line_mode(self):
        assert prompt_parser.split_prompt_units("a\nb", None) == ["a", "b"]
        assert prompt_parser.split_prompt_units("a\nb", "") == ["a", "b"]

    def test_mode_matching_is_case_sensitive(self):
        # Callers are expected to feed a normalized code; "pg"/"fg" silently
        # degrade to line mode.
        assert prompt_parser.split_prompt_units("a\n\nb", "pg") == ["a", "b"]
        assert prompt_parser.split_prompt_units("a\nb", "fg") == ["a", "b"]

    @pytest.mark.parametrize("mode", ["G", "PG", "FG"])
    def test_crlf_and_lone_cr_are_normalized(self, mode):
        crlf = prompt_parser.split_prompt_units("a\r\nb\rc", mode)
        assert crlf == prompt_parser.split_prompt_units("a\nb\nc", mode)

    def test_comment_lines_are_dropped(self):
        text = "# leading note\n  #  indented note\nkeep me\n#\nalso keep"
        assert prompt_parser.split_prompt_units(text, "G") == ["keep me", "also keep"]

    def test_hash_in_the_middle_of_a_line_is_not_a_comment(self):
        assert prompt_parser.split_prompt_units("shot #3 of the scene", "G") == ["shot #3 of the scene"]

    def test_comment_line_does_not_split_a_paragraph(self):
        text = "a\n# note\nb\n\nc"
        assert prompt_parser.split_prompt_units(text, "PG") == ["a\nb", "c"]

    def test_enhanced_prefix_is_stripped(self):
        assert prompt_parser.split_prompt_units(ENHANCED + "hello", "G") == ["hello"]
        # ...also after CRLF normalisation.
        assert prompt_parser.split_prompt_units("!enhanced!\r\nhello", "G") == ["hello"]

    @pytest.mark.parametrize("text", ["!enhanced!hello", " !enhanced!\nhello", "x\n!enhanced!\ny"])
    def test_enhanced_prefix_only_stripped_at_the_very_start_of_a_line_of_its_own(self, text):
        assert "!enhanced!" in "".join(prompt_parser.split_prompt_units(text, "FG"))

    def test_trailing_whitespace_is_rstripped_inside_a_unit(self):
        assert prompt_parser.split_prompt_units("a   \n   b", "FG") == ["a\n   b"]

    def test_none_text_is_not_supported(self):
        # Current behaviour: the caller must pass a string.
        with pytest.raises(AttributeError):
            prompt_parser.split_prompt_units(None, "G")

    def test_originals_flag_delegates_to_the_original_splitter(self):
        text = f"{ENHANCED}{PREFIX} original\nenhanced text"
        assert prompt_parser.split_prompt_units(text, "G", originals=True) == ["original"]
        assert prompt_parser.split_prompt_units(text, "G") == ["enhanced text"]


class TestSerializePromptUnits:
    def test_line_mode_joins_with_single_newlines(self):
        assert prompt_parser.serialize_prompt_units("", ["a", "b"], "G") == "a\nb"

    @pytest.mark.parametrize("mode", ["PG", "PW"])
    def test_paragraph_modes_join_with_blank_lines(self, mode):
        assert prompt_parser.serialize_prompt_units("", ["a", "b"], mode) == "a\n\nb"

    def test_blank_prompts_are_dropped_and_the_rest_stripped(self):
        assert prompt_parser.serialize_prompt_units("", [" a ", "  ", "", "b\n"], "G") == "a\nb"

    @pytest.mark.parametrize("prompts", [[], ["", "   "]])
    def test_no_usable_prompt_serializes_to_empty_string(self, prompts):
        assert prompt_parser.serialize_prompt_units("whatever", prompts, "G") == ""

    def test_full_prompt_mode_keeps_only_the_first_unit(self):
        # BUG (pinned): in "FG" mode everything after prompts[0] is silently
        # discarded instead of being joined back together.
        assert prompt_parser.serialize_prompt_units("", ["a", "b"], "FG") == "a"

    @pytest.mark.parametrize("prompt_text", ["", "unrelated", ENHANCED + "x\r\ny", None])
    def test_prompt_text_argument_is_ignored(self, prompt_text):
        # BUG (pinned): `prompt_text` is normalized then never used, so even
        # None -- which would crash a real use -- has no effect... except that
        # None.replace() does raise, so guard the None case separately.
        if prompt_text is None:
            with pytest.raises(AttributeError):
                prompt_parser.serialize_prompt_units(prompt_text, ["a"], "G")
        else:
            assert prompt_parser.serialize_prompt_units(prompt_text, ["a"], "G") == "a"

    def test_none_mode_raises_unlike_the_splitter(self):
        # BUG (pinned): split_prompt_units() guards with `or ""` but this one
        # does not, so a None mode blows up as soon as there is a prompt.
        assert prompt_parser.serialize_prompt_units("", [], None) == ""
        with pytest.raises(TypeError):
            prompt_parser.serialize_prompt_units("", ["a"], None)

    @pytest.mark.parametrize("mode", ["G", "PG", "W", "PW"])
    def test_round_trip_split_then_serialize(self, mode):
        text = "first prompt\n\nsecond prompt"
        units = prompt_parser.split_prompt_units(text, mode)
        again = prompt_parser.split_prompt_units(
            prompt_parser.serialize_prompt_units("", units, mode), mode
        )
        assert again == units


class TestSplitPromptOriginalUnits:
    def test_without_markers_it_matches_the_visible_split(self):
        text = "a\n\n# note\nb"
        for mode in ("G", "W", "PG", "PW", "FG"):
            assert prompt_parser.split_prompt_original_units(text, mode) == (
                prompt_parser.split_prompt_units(text, mode)
            )

    def test_line_mode_marker_replaces_the_following_line(self):
        text = f"{PREFIX} original one\nenhanced one\n{PREFIX} original two\nenhanced two"
        assert prompt_parser.split_prompt_original_units(text, "G") == ["original one", "original two"]

    def test_line_mode_marker_only_covers_one_line(self):
        text = f"{PREFIX} O\nline one\nline two"
        assert prompt_parser.split_prompt_original_units(text, "G") == ["O", "line two"]

    def test_line_mode_trailing_marker_is_still_emitted(self):
        assert prompt_parser.split_prompt_original_units(f"{PREFIX} only original", "G") == [
            "only original"
        ]

    def test_line_mode_bare_marker_alone_produces_nothing(self):
        assert prompt_parser.split_prompt_original_units(PREFIX, "G") == []
        assert prompt_parser.split_prompt_original_units(f"{PREFIX}\n   ", "G") == []

    def test_line_mode_bare_marker_after_a_real_one_duplicates_it(self):
        # BUG (pinned): the pending original is appended but not cleared before
        # `pending_original = marker or pending_original`, so an empty marker
        # re-emits the previous original.
        text = f"{PREFIX} A\n{PREFIX}\nvisible"
        assert prompt_parser.split_prompt_original_units(text, "G") == ["A", "A"]

    def test_indented_marker_is_swallowed_as_a_comment(self):
        # BUG (pinned): the marker starts with '#', so an indented marker line
        # is treated as a comment and its original is lost.
        text = f"  {PREFIX} O\nvisible"
        assert prompt_parser.split_prompt_original_units(text, "G") == ["visible"]

    def test_paragraph_mode_marker_covers_the_whole_paragraph(self):
        text = f"{PREFIX} O1\nline a\nline b\n\nplain c\nplain d"
        assert prompt_parser.split_prompt_original_units(text, "PG") == ["O1", "plain c\nplain d"]

    def test_paragraph_mode_marker_mid_paragraph_starts_a_new_unit(self):
        text = f"a\n{PREFIX} O\nb"
        assert prompt_parser.split_prompt_original_units(text, "PG") == ["a", "O"]

    def test_paragraph_mode_consecutive_markers_flush_each_other(self):
        # The visible "x" is shadowed by the still-pending original "B".
        text = f"{PREFIX} A\n{PREFIX} B\nx"
        assert prompt_parser.split_prompt_original_units(text, "PG") == ["A", "B"]

    def test_paragraph_mode_bare_marker_falls_back_to_visible_text(self):
        text = f"{PREFIX} A\nx\n\n{PREFIX}\ny"
        assert prompt_parser.split_prompt_original_units(text, "PG") == ["A", "y"]

    def test_paragraph_mode_ignores_comments(self):
        assert prompt_parser.split_prompt_original_units("# c\na\n\n# d\nb", "PG") == ["a", "b"]

    def test_full_prompt_mode_joins_every_original(self):
        text = f"{PREFIX} A\nvisible\n{PREFIX} B\nmore"
        assert prompt_parser.split_prompt_original_units(text, "FG") == ["A\nB"]

    def test_full_prompt_mode_falls_back_to_visible_lines(self):
        assert prompt_parser.split_prompt_original_units("a\n# c\nb  ", "FG") == ["a\nb"]

    @pytest.mark.parametrize("text", ["   \n  ", "", f"{PREFIX}\n  ", "# only a comment"])
    def test_full_prompt_mode_empty_input(self, text):
        assert prompt_parser.split_prompt_original_units(text, "FG") == []

    def test_single_prompt_uses_the_full_prompt_branch(self):
        text = f"{PREFIX} A\nvisible"
        assert prompt_parser.split_prompt_original_units(text, "W", single_prompt=True) == ["A"]

    def test_enhanced_prefix_and_crlf_are_normalized(self):
        text = f"{ENHANCED}{PREFIX} A\r\nvisible\r"
        assert prompt_parser.split_prompt_original_units(text, "G") == ["A"]

    def test_missing_mode_behaves_like_line_mode(self):
        assert prompt_parser.split_prompt_original_units("a\nb", None) == ["a", "b"]

    def test_none_text_is_not_supported(self):
        with pytest.raises(AttributeError):
            prompt_parser.split_prompt_original_units(None, "G")


class TestSerializePromptBlocksWithPrefix:
    def test_default_placeholder_originals(self):
        assert prompt_parser.serialize_prompt_blocks_with_prefix(["a", "b"]) == (
            f"{PREFIX} Prompt 1\na\n\n{PREFIX} Prompt 2\nb"
        )

    def test_supplied_originals_are_used_then_placeholders(self):
        assert prompt_parser.serialize_prompt_blocks_with_prefix([" a ", "b"], ["O1"]) == (
            f"{PREFIX} O1\na\n\n{PREFIX} Prompt 2\nb"
        )

    @pytest.mark.parametrize("original", [None, "", "   "])
    def test_falsy_originals_become_an_empty_marker(self, original):
        assert prompt_parser.serialize_prompt_blocks_with_prefix(["a"], [original]) == (
            f"{PREFIX} \na"
        )

    def test_newlines_in_an_original_are_flattened_to_one_space(self):
        assert prompt_parser.serialize_prompt_blocks_with_prefix(["a"], ["x\r\n\ny"]) == (
            f"{PREFIX} x y\na"
        )

    def test_non_string_originals_are_coerced(self):
        assert prompt_parser.serialize_prompt_blocks_with_prefix(["a"], [42]) == f"{PREFIX} 42\na"

    @pytest.mark.parametrize("prompts", [[], ["", "   "]])
    def test_no_usable_prompt_gives_an_empty_string(self, prompts):
        assert prompt_parser.serialize_prompt_blocks_with_prefix(prompts) == ""

    def test_blank_prompts_do_not_consume_an_original(self):
        assert prompt_parser.serialize_prompt_blocks_with_prefix(["", "b"], ["O1"]) == (
            f"{PREFIX} O1\nb"
        )

    @pytest.mark.parametrize("mode", ["G", "PG"])
    def test_round_trip_with_the_original_splitter(self, mode):
        prompts = ["enhanced one", "enhanced two"]
        originals = ["orig one", "orig two"]
        blocks = prompt_parser.serialize_prompt_blocks_with_prefix(prompts, originals)
        assert prompt_parser.split_prompt_units(blocks, mode, originals=True) == originals
        assert prompt_parser.split_prompt_units(blocks, mode) == prompts

    def test_round_trip_keeps_multiline_enhanced_prompts_together(self):
        blocks = prompt_parser.serialize_prompt_blocks_with_prefix(["line1\nline2"], ["orig"])
        assert prompt_parser.split_prompt_units(blocks, "PG", originals=True) == ["orig"]
        assert prompt_parser.split_prompt_units(blocks, "PG") == ["line1\nline2"]

    def test_round_trip_of_an_empty_marker_falls_back_to_the_visible_line(self):
        blocks = prompt_parser.serialize_prompt_blocks_with_prefix(["visible"], [None])
        assert prompt_parser.split_prompt_units(blocks, "G", originals=True) == ["visible"]


class TestIsSpeakerOptionsLine:
    @pytest.mark.parametrize(
        "line",
        [
            "Speaker 1 {pitch=2}: hello",
            "speaker2{}:x",
            "  Speaker 10 {a b} : text",
            "SPEAKER 3 {v}:",
        ],
    )
    def test_recognised(self, line):
        assert prompt_parser.is_speaker_options_line(line) is True

    @pytest.mark.parametrize(
        "line",
        [
            "Speaker 1: hello",  # no option braces
            "Speaker {x}: hi",  # no speaker number
            "Speaker 1 {a}",  # no colon
            "Speaker 1 {a{b}}: x",  # nested braces
            "text\nSpeaker 1 {a}: hi",  # anchored to the start of the string
            "a Speaker 1 {x}: hi",
            "",
            None,
        ],
    )
    def test_rejected(self, line):
        assert prompt_parser.is_speaker_options_line(line) is False


class TestProcessTemplate:
    def test_plain_text_passes_through(self):
        assert prompt_parser.process_template("hello\nworld") == ("hello\nworld", "")

    def test_lines_are_stripped(self):
        assert prompt_parser.process_template("   hello   ") == ("hello", "")

    def test_single_variable_expands_once_per_value(self):
        out, err = prompt_parser.process_template('!{color}="red","blue"\na {color} car')
        assert (out, err) == ("a red car\na blue car", "")

    def test_every_template_line_is_repeated_per_value(self):
        out, err = prompt_parser.process_template('!{a}="1","2"\nfirst {a}\nsecond {a}')
        assert (out, err) == ("first 1\nsecond 1\nfirst 2\nsecond 2", "")

    def test_shorter_variables_cycle_with_modulo(self):
        out, err = prompt_parser.process_template('!{a}="1","2","3" : {b}="p","q"\n{a}{b}')
        assert (out, err) == ("1p\n2q\n3p", "")

    def test_a_second_macro_starts_a_new_section(self):
        out, err = prompt_parser.process_template('!{a}="1","2"\nx {a}\n!{b}="9"\ny {b}')
        assert (out, err) == ("x 1\nx 2\ny 9", "")

    def test_lines_before_the_first_macro_are_emitted_verbatim(self):
        out, err = prompt_parser.process_template('plain\n!{a}="1"\nx {a}')
        assert (out, err) == ("plain\nx 1", "")

    def test_crlf_is_normalized(self):
        assert prompt_parser.process_template('!{a}="1"\r\nx {a}') == ("x 1", "")

    def test_comments_are_dropped_by_default(self):
        assert prompt_parser.process_template("# note\nhello") == ("hello", "")

    def test_comments_can_be_kept_and_are_substituted(self):
        out, err = prompt_parser.process_template('!{a}="1"\n# {a} note\nx {a}', keep_comments=True)
        assert (out, err) == ("# 1 note\nx 1", "")

    def test_empty_lines_are_dropped_by_default(self):
        assert prompt_parser.process_template("a\n\nb") == ("a\nb", "")

    def test_empty_lines_can_be_kept(self):
        assert prompt_parser.process_template("a\n\nb", keep_empty_lines=True) == ("a\n\nb", "")

    def test_kept_empty_lines_are_repeated_with_the_template(self):
        out, err = prompt_parser.process_template(
            '!{a}="1","2"\nx {a}\n\n', keep_empty_lines=True
        )
        assert (out, err) == ("x 1\n\n\nx 2\n\n", "")

    @pytest.mark.parametrize("text", [None, "", "   \n  "])
    def test_blank_input(self, text):
        assert prompt_parser.process_template(text) == ("", "")

    def test_blank_input_with_kept_empty_lines(self):
        # The whole text is no longer stripped, so the two blank lines survive.
        assert prompt_parser.process_template("   \n  ", keep_empty_lines=True) == ("\n", "")

    def test_speaker_option_lines_skip_the_unknown_variable_check(self):
        assert prompt_parser.process_template("Speaker 1 {vol=2}: hello") == (
            "Speaker 1 {vol=2}: hello",
            "",
        )

    def test_a_macro_only_section_produces_nothing(self):
        assert prompt_parser.process_template('!{a}="1","2"') == ("", "")

    @pytest.mark.parametrize(
        "text, expected_error",
        [
            ("a {b} c", "Unknown variable '{b}' in template"),
            ('!{a="1"', "Unmatched braces: 1 opening '{' and 0 closing '}' braces"),
            ('!{a}="1', "Unclosed double quotes"),
            ('!{a}"1"', "Missing '=' after variable '{a}'"),
            ('!{a}=1', "No quoted values found for variable '{a}'"),
            ('!{a}="1""2"', "Missing comma between values for variable '{a}'"),
            ('!{ }="1"', "Empty variable name"),
            ('!}x{="1"', "Malformed variable declaration"),
        ],
    )
    def test_errors_blank_the_output_and_quote_the_line(self, text, expected_error):
        out, err = prompt_parser.process_template(text)
        assert out == ""
        assert err.startswith(expected_error)
        assert err.endswith(f"Line: '{text.splitlines()[-1]}'")

    def test_a_macro_section_with_no_variables_is_ignored(self):
        assert prompt_parser.process_template("! : ") == ("", "")

    def test_values_containing_a_colon_are_rejected(self):
        # BUG (pinned): the macro line is split on every ':' -- including ones
        # inside quoted values -- so "x:y" cannot be used as a value.
        out, err = prompt_parser.process_template('!{a}="x:y"\n{a}')
        assert out == ""
        assert err.startswith("No quoted values found for variable '{a}'")

    def test_spaces_after_commas_are_accepted(self):
        assert prompt_parser.process_template('!{a}="1", "2"\n{a}') == ("1\n2", "")

    def test_a_value_may_contain_a_comma(self):
        assert prompt_parser.process_template('!{a}="x,y"\n{a}') == ("x,y", "")


class TestProcessCurrentTemplate:
    def test_no_variables_returns_the_lines_unchanged(self):
        lines = ["a", "b"]
        out, err = prompt_parser.process_current_template(lines, {})
        assert (out, err) == (lines, "")

    def test_no_lines_returns_empty(self):
        assert prompt_parser.process_current_template([], {"a": ["1"]}) == ([], "")

    def test_substitution_is_repeated_per_value(self):
        out, err = prompt_parser.process_current_template(["<{a}>"], {"a": ["1", "2"]})
        assert (out, err) == (["<1>", "<2>"], "")

    def test_unreferenced_variables_still_drive_the_repeat_count(self):
        out, err = prompt_parser.process_current_template(["fixed"], {"a": ["1", "2", "3"]})
        assert (out, err) == (["fixed", "fixed", "fixed"], "")


class TestExtractVariableNames:
    @pytest.mark.parametrize("line", ['!{a}="1","2" : {b}="x"', '{a}="1","2" : {b}="x"'])
    def test_leading_bang_is_optional(self, line):
        assert prompt_parser.extract_variable_names(line) == (["a", "b"], "")

    def test_names_are_stripped_and_deduplicated_in_order(self):
        assert prompt_parser.extract_variable_names('!{ b }="1" : {a}="2" : {b}="3"') == (
            ["b", "a"],
            "",
        )

    def test_unmatched_braces_error(self):
        names, err = prompt_parser.extract_variable_names('!{a="1"')
        assert names == []
        assert err == "Unmatched braces: 1 opening '{' and 0 closing '}' braces"

    @pytest.mark.parametrize("line", ["", "!", "!hello"])
    def test_no_variables(self, line):
        assert prompt_parser.extract_variable_names(line) == ([], "")

    def test_braces_on_the_value_side_are_also_reported(self):
        # BUG (pinned): the value part is not excluded, so "{b}" used as a value
        # is reported as a declared variable name.
        assert prompt_parser.extract_variable_names("!{a}={b}") == (["a", "b"], "")


class TestExtractVariableValues:
    def test_parses_names_and_values(self):
        assert prompt_parser.extract_variable_values('!{a}="1","2" : {b}="x"') == (
            {"a": ["1", "2"], "b": ["x"]},
            "",
        )

    def test_leading_bang_is_optional(self):
        assert prompt_parser.extract_variable_values('{a}="1"') == ({"a": ["1"]}, "")

    def test_empty_value_is_allowed(self):
        assert prompt_parser.extract_variable_values('!{a}=""') == ({"a": [""]}, "")

    @pytest.mark.parametrize("line", ["", "!", "!hello"])
    def test_no_variables(self, line):
        assert prompt_parser.extract_variable_values(line) == ({}, "")

    @pytest.mark.parametrize(
        "line, expected_error",
        [
            ('!{a="1"', "Unmatched braces: 1 opening '{' and 0 closing '}' braces"),
            ('!{a}="1', "Unclosed double quotes"),
            ('!{a}"1"', "Missing '=' after variable '{a}'"),
            ("!{a}=", "No quoted values found for variable '{a}'"),
            ('!{a}="1""2"', "Missing comma between values for variable '{a}'"),
            ('!{ }="1"', "Empty variable name"),
            ('!}x{="1"', "Malformed variable declaration"),
        ],
    )
    def test_errors(self, line, expected_error):
        variables, err = prompt_parser.extract_variable_values(line)
        assert variables == {}
        assert err == expected_error

    def test_errors_match_the_ones_process_template_reports(self):
        line = '!{a}="1'
        _, macro_err = prompt_parser.extract_variable_values(line)
        _, template_err = prompt_parser.process_template(line)
        assert template_err.startswith(macro_err)


class TestGenerateMacroLine:
    def test_formats_a_macro_line(self):
        assert prompt_parser.generate_macro_line({"a": ["1", "2"], "b": ["x"]}) == (
            '! {a}="1","2" : {b}="x"'
        )

    def test_empty_dict(self):
        assert prompt_parser.generate_macro_line({}) == "! "

    def test_round_trips_through_the_extractors(self):
        variables = {"color": ["red", "blue"], "mood": ["calm"]}
        line = prompt_parser.generate_macro_line(variables)
        assert prompt_parser.extract_variable_values(line) == (variables, "")
        assert prompt_parser.extract_variable_names(line) == (["color", "mood"], "")

    def test_generated_line_is_accepted_by_process_template(self):
        line = prompt_parser.generate_macro_line({"col": ["red", "blue"]})
        assert prompt_parser.process_template(f"{line}\na {{col}} car") == (
            "a red car\na blue car",
            "",
        )
