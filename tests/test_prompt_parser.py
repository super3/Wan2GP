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

Every expectation below is a literal read off the implementation; a handful of
tests deliberately pin behaviour that looks wrong, and each one carries a
comment saying so.  Parametrisation is kept to inputs that reach a *different*
line of the module -- one case per branch, not one case per spelling.
"""

import pytest


import shared.utils.prompt_parser as prompt_parser

PREFIX = prompt_parser.PROMPT_UNIT_PREFIX
ENHANCED = prompt_parser.ENHANCED_PROMPT_PREFIX


class TestConstants:
    def test_marker_constants(self):
        assert PREFIX == "#!PROMPT!:"
        assert ENHANCED == "!enhanced!\n"
        assert prompt_parser.DEFAULT_MULTI_PROMPTS_MODE == "PG"


class TestNormalizeMultiPromptsMode:
    def test_canonical_codes_pass_through(self):
        assert prompt_parser.normalize_multi_prompts_mode("PW") == "PW"

    def test_strings_are_stripped_and_upper_cased(self):
        assert prompt_parser.normalize_multi_prompts_mode("\tfg\n") == "FG"

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0, "G"),
            (2, "FG"),
            ("1", "W"),
            ("p", "PG"),
        ],
    )
    def test_legacy_aliases(self, value, expected):
        assert prompt_parser.normalize_multi_prompts_mode(value) == expected

    def test_blank_string_means_full_prompt(self):
        # The empty string is an alias for "FG", *not* a fallback to `default`.
        assert prompt_parser.normalize_multi_prompts_mode("", default="W") == "FG"

    def test_floats_are_truncated_to_int(self):
        assert prompt_parser.normalize_multi_prompts_mode(1.9) == "W"

    def test_bools_follow_the_int_aliases(self):
        # bool is a subclass of int, so True == 1 -> "W".
        assert prompt_parser.normalize_multi_prompts_mode(True) == "W"

    # One case per rejection branch: `is None`, the non-str/non-number `else`,
    # and a string that survives the alias table but fails the final whitelist.
    @pytest.mark.parametrize("value", [None, object(), "PGW"])
    def test_unknown_values_fall_back_to_default(self, value):
        assert prompt_parser.normalize_multi_prompts_mode(value) == "G"
        assert prompt_parser.normalize_multi_prompts_mode(value, default="PW") == "PW"


class TestGetMultiPromptsGenChoices:
    def test_all_modes_offered_by_default(self):
        choices = prompt_parser.get_multi_prompts_gen_choices()
        assert [code for _, code in choices] == ["G", "PG", "W", "PW", "FG"]

    def test_sliding_window_modes_can_be_hidden(self):
        choices = prompt_parser.get_multi_prompts_gen_choices(include_sliding_window=False)
        assert [code for _, code in choices] == ["G", "PG", "FG"]

    def test_medium_is_interpolated_into_the_queue_labels(self):
        default_labels = [label for label, _ in prompt_parser.get_multi_prompts_gen_choices()]
        assert default_labels[0] == (
            "Each New Line Will Add a new Video Request to the Generation Queue"
        )
        labels = [label for label, _ in prompt_parser.get_multi_prompts_gen_choices(medium="Image")]
        assert labels[0] == "Each New Line Will Add a new Image Request to the Generation Queue"
        assert labels[1] == (
            "Each new Paragraph separated by an Empty Line Will Add a new Image "
            "Request to the Generation Queue"
        )
        # The sliding-window and full-prompt labels are fixed text.
        assert labels[2] == (
            "Each Line Will be used for a new Sliding Window of the same Video Generation"
        )
        assert labels[-1] == "All the Lines are Part of the Same Prompt"


class TestSplitPromptUnits:
    def test_one_request_per_line(self):
        assert prompt_parser.split_prompt_units("a\nb\nc", "G") == ["a", "b", "c"]

    def test_line_mode_strips_and_drops_blank_lines(self):
        assert prompt_parser.split_prompt_units("  a  \n\n\t\n b ", "G") == ["a", "b"]

    def test_paragraph_modes_split_on_blank_lines(self):
        # "PW" rather than "PG": it is the case that fails when the
        # `"P" in multi_prompts_gen_type` membership test is narrowed to an
        # equality check against "PG".
        text = "a\nb\n\n\nc\n \nd"
        assert prompt_parser.split_prompt_units(text, "PW") == ["a\nb", "c", "d"]

    def test_full_prompt_mode_keeps_everything_as_one_unit(self):
        assert prompt_parser.split_prompt_units("a  \n\nb", "FG") == ["a\n\nb"]

    def test_single_prompt_overrides_the_mode(self):
        assert prompt_parser.split_prompt_units("a\nb", "W", single_prompt=True) == ["a\nb"]
        assert prompt_parser.split_prompt_units("a\n\nb", "PG", single_prompt=True) == ["a\n\nb"]

    @pytest.mark.parametrize("text", ["   ", "# only a comment"])
    def test_empty_input_yields_no_units(self, text):
        # The early `if not prompt_text: return []` is reached before the mode
        # is ever looked at, so there is nothing to gain from a mode axis here.
        assert prompt_parser.split_prompt_units(text, "G") == []

    def test_missing_mode_behaves_like_line_mode(self):
        assert prompt_parser.split_prompt_units("a\nb", None) == ["a", "b"]
        assert prompt_parser.split_prompt_units("a\nb", "") == ["a", "b"]

    def test_crlf_and_lone_cr_are_normalized(self):
        assert prompt_parser.split_prompt_units("a\r\nb\rc", "G") == ["a", "b", "c"]

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

    def test_enhanced_prefix_only_stripped_at_the_very_start_of_a_line_of_its_own(self):
        assert prompt_parser.split_prompt_units("x\n!enhanced!\ny", "FG") == ["x\n!enhanced!\ny"]

    def test_trailing_whitespace_is_rstripped_inside_a_unit(self):
        assert prompt_parser.split_prompt_units("a   \n   b", "FG") == ["a\n   b"]

    def test_originals_flag_delegates_to_the_original_splitter(self):
        text = f"{ENHANCED}{PREFIX} original\nenhanced text"
        assert prompt_parser.split_prompt_units(text, "G", originals=True) == ["original"]
        assert prompt_parser.split_prompt_units(text, "G") == ["enhanced text"]


class TestSerializePromptUnits:
    def test_line_mode_joins_with_single_newlines(self):
        assert prompt_parser.serialize_prompt_units("", ["a", "b"], "G") == "a\nb"

    def test_paragraph_modes_join_with_blank_lines(self):
        # "PW" for the same reason as in the splitter: it is the case that
        # distinguishes `"P" in mode` from `mode == "PG"`.
        assert prompt_parser.serialize_prompt_units("", ["a", "b"], "PW") == "a\n\nb"

    def test_blank_prompts_are_dropped_and_the_rest_stripped(self):
        assert prompt_parser.serialize_prompt_units("", [" a ", "  ", "", "b\n"], "G") == "a\nb"

    def test_full_prompt_mode_keeps_only_the_first_unit(self):
        # BUG (pinned): in "FG" mode everything after prompts[0] is silently
        # discarded instead of being joined back together.
        assert prompt_parser.serialize_prompt_units("", ["a", "b"], "FG") == "a"

    def test_prompt_text_argument_is_ignored(self):
        # BUG (pinned): `prompt_text` is normalized (CRLF, "!enhanced!", strip)
        # and then never read again -- the result depends only on `prompts` and
        # the mode.
        assert prompt_parser.serialize_prompt_units(ENHANCED + "x\r\ny", ["a"], "G") == "a"

    @pytest.mark.parametrize(
        "mode, expected_units, expected_text",
        [
            ("G", ["first prompt", "second prompt"], "first prompt\nsecond prompt"),
            ("PG", ["first prompt", "second prompt"], "first prompt\n\nsecond prompt"),
        ],
    )
    def test_round_trip_split_then_serialize(self, mode, expected_units, expected_text):
        # Both intermediate values are pinned to literals: a bare
        # split(serialize(split(x))) == split(x) round trip would also pass if
        # the splitter always returned [] and the serializer always returned "".
        text = "first prompt\n\nsecond prompt"
        units = prompt_parser.split_prompt_units(text, mode)
        assert units == expected_units
        serialized = prompt_parser.serialize_prompt_units("", units, mode)
        assert serialized == expected_text
        assert prompt_parser.split_prompt_units(serialized, mode) == expected_units


class TestSplitPromptOriginalUnits:
    @pytest.mark.parametrize("mode", ["G", "PW"])
    def test_without_markers_it_matches_the_visible_split(self, mode):
        # Marker-free text takes the same shape as the visible splitter's
        # output; "PW" also pins the `"P" in mode` membership test in this
        # function, which the marker tests below only exercise through "PG".
        text = "a\n\n# note\nb"
        assert prompt_parser.split_prompt_original_units(text, mode) == ["a", "b"]

    def test_line_mode_marker_replaces_the_following_line(self):
        text = f"{PREFIX} original one\nenhanced one\n{PREFIX} original two\nenhanced two"
        assert prompt_parser.split_prompt_original_units(text, "G") == ["original one", "original two"]

    def test_line_mode_trailing_marker_is_still_emitted(self):
        assert prompt_parser.split_prompt_original_units(f"{PREFIX} only original", "G") == [
            "only original"
        ]

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

    def test_full_prompt_mode_falls_back_to_visible_lines(self):
        assert prompt_parser.split_prompt_original_units("a\n# c\nb  ", "FG") == ["a\nb"]

    def test_full_prompt_mode_empty_input(self):
        # A marker with no text leaves both `originals` and the visible lines
        # empty, which is the `return [prompt] if prompt else []` branch.
        assert prompt_parser.split_prompt_original_units(f"{PREFIX}\n  ", "FG") == []

    def test_single_prompt_joins_every_original_unlike_line_mode(self):
        # A single marker cannot tell the `single_prompt` branch from plain line
        # mode -- both yield ["A"].  With two markers they diverge: the
        # full-prompt branch joins the originals into a single unit.
        text = f"{PREFIX} A\nv1\n{PREFIX} B\nv2"
        assert prompt_parser.split_prompt_original_units(text, "G") == ["A", "B"]
        assert prompt_parser.split_prompt_original_units(text, "G", single_prompt=True) == ["A\nB"]
        # ...and split_prompt_units forwards the flag to the original splitter.
        assert prompt_parser.split_prompt_units(
            text, "G", single_prompt=True, originals=True
        ) == ["A\nB"]

    def test_enhanced_prefix_and_crlf_are_normalized(self):
        text = f"{ENHANCED}{PREFIX} A\r\nvisible\r"
        assert prompt_parser.split_prompt_original_units(text, "G") == ["A"]

    def test_missing_mode_behaves_like_line_mode(self):
        assert prompt_parser.split_prompt_original_units("a\nb", None) == ["a", "b"]


class TestSerializePromptBlocksWithPrefix:
    def test_default_placeholder_originals(self):
        assert prompt_parser.serialize_prompt_blocks_with_prefix(["a", "b"]) == (
            f"{PREFIX} Prompt 1\na\n\n{PREFIX} Prompt 2\nb"
        )

    def test_supplied_originals_are_used_then_placeholders(self):
        assert prompt_parser.serialize_prompt_blocks_with_prefix([" a ", "b"], ["O1"]) == (
            f"{PREFIX} O1\na\n\n{PREFIX} Prompt 2\nb"
        )

    def test_falsy_originals_become_an_empty_marker(self):
        assert prompt_parser.serialize_prompt_blocks_with_prefix(["a"], [None]) == f"{PREFIX} \na"

    def test_newlines_in_an_original_are_flattened_to_one_space(self):
        assert prompt_parser.serialize_prompt_blocks_with_prefix(["a"], ["x\r\n\ny"]) == (
            f"{PREFIX} x y\na"
        )

    def test_non_string_originals_are_coerced(self):
        assert prompt_parser.serialize_prompt_blocks_with_prefix(["a"], [42]) == f"{PREFIX} 42\na"

    def test_blank_prompts_do_not_consume_an_original(self):
        assert prompt_parser.serialize_prompt_blocks_with_prefix(["", "b"], ["O1"]) == (
            f"{PREFIX} O1\nb"
        )

    def test_round_trip_keeps_multiline_enhanced_prompts_together(self):
        blocks = prompt_parser.serialize_prompt_blocks_with_prefix(["line1\nline2"], ["orig"])
        assert blocks == f"{PREFIX} orig\nline1\nline2"
        assert prompt_parser.split_prompt_units(blocks, "PG", originals=True) == ["orig"]
        assert prompt_parser.split_prompt_units(blocks, "PG") == ["line1\nline2"]


class TestIsSpeakerOptionsLine:
    @pytest.mark.parametrize(
        "line",
        [
            "Speaker 1 {pitch=2}: hello",
            "  speaker10{} : text",  # case-insensitive, multi-digit, loose spacing
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
            "a Speaker 1 {x}: hi",  # anchored to the start of the string
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

    def test_crlf_is_normalized(self):
        assert prompt_parser.process_template('!{a}="1"\r\nx {a}') == ("x 1", "")

    def test_comments_are_dropped_by_default(self):
        assert prompt_parser.process_template("# note\nhello") == ("hello", "")

    def test_comments_can_be_kept_and_are_substituted(self):
        out, err = prompt_parser.process_template('!{a}="1"\n# {a} note\nx {a}', keep_comments=True)
        assert (out, err) == ("# 1 note\nx 1", "")

    def test_kept_comments_skip_the_unknown_variable_check(self):
        # A kept comment is exempt from the unknown-variable check (the
        # `not line.startswith('#')` guard), so it passes through with the
        # braces intact instead of erroring...
        assert prompt_parser.process_template("# {b} note\nhello", keep_comments=True) == (
            "# {b} note\nhello",
            "",
        )
        # ...while the identical reference on a normal line is an error.
        out, err = prompt_parser.process_template("{b} note\nhello", keep_comments=True)
        assert out == ""
        assert err.startswith("Unknown variable '{b}' in template")

    def test_empty_lines_are_dropped_by_default(self):
        assert prompt_parser.process_template("a\n\nb") == ("a\nb", "")

    def test_empty_lines_can_be_kept(self):
        assert prompt_parser.process_template("a\n\nb", keep_empty_lines=True) == ("a\n\nb", "")

    def test_kept_empty_lines_are_repeated_with_the_template(self):
        out, err = prompt_parser.process_template(
            '!{a}="1","2"\nx {a}\n\n', keep_empty_lines=True
        )
        assert (out, err) == ("x 1\n\n\nx 2\n\n", "")

    def test_blank_input(self):
        assert prompt_parser.process_template(None) == ("", "")

    def test_speaker_option_lines_skip_the_unknown_variable_check(self):
        assert prompt_parser.process_template("Speaker 1 {vol=2}: hello") == (
            "Speaker 1 {vol=2}: hello",
            "",
        )

    @pytest.mark.parametrize(
        "text, expected_error",
        [
            # The unknown-variable error exists only here...
            ("a {b} c", "Unknown variable '{b}' in template\nLine: 'a {b} c'"),
            # ...and this one pins the "\nLine: '<orig_line>'" suffix that
            # extract_variable_values() (tested exhaustively below) omits.
            ('!{a}="1', "Unclosed double quotes\nLine: '!{a}=\"1'"),
        ],
    )
    def test_errors_blank_the_output_and_quote_the_line(self, text, expected_error):
        assert prompt_parser.process_template(text) == ("", expected_error)

    def test_values_containing_a_colon_are_rejected(self):
        # BUG (pinned): the macro line is split on every ':' -- including ones
        # inside quoted values -- so "x:y" cannot be used as a value.
        out, err = prompt_parser.process_template('!{a}="x:y"\n{a}')
        assert out == ""
        assert err.startswith("No quoted values found for variable '{a}'")

    def test_a_value_may_contain_a_comma(self):
        assert prompt_parser.process_template('!{a}="x,y"\n{a}') == ("x,y", "")


class TestProcessCurrentTemplate:
    def test_no_variables_returns_the_lines_unchanged(self):
        lines = ["a", "b"]
        out, err = prompt_parser.process_current_template(lines, {})
        assert (out, err) == (["a", "b"], "")

    def test_unreferenced_variables_still_drive_the_repeat_count(self):
        out, err = prompt_parser.process_current_template(["fixed"], {"a": ["1", "2", "3"]})
        assert (out, err) == (["fixed", "fixed", "fixed"], "")


class TestExtractVariableNames:
    def test_leading_bang_is_optional(self):
        # Every other test in this class passes the '!' form.
        assert prompt_parser.extract_variable_names('{a}="1","2" : {b}="x"') == (["a", "b"], "")

    def test_names_are_stripped_and_deduplicated_in_order(self):
        assert prompt_parser.extract_variable_names('!{ b }="1" : {a}="2" : {b}="3"') == (
            ["b", "a"],
            "",
        )

    def test_unmatched_braces_error(self):
        names, err = prompt_parser.extract_variable_names('!{a="1"')
        assert names == []
        assert err == "Unmatched braces: 1 opening '{' and 0 closing '}' braces"

    def test_no_variables(self):
        assert prompt_parser.extract_variable_names("!hello") == ([], "")

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

    def test_no_variables(self):
        assert prompt_parser.extract_variable_values("!hello") == ({}, "")

    # One case per error `return` in the function -- this is the canonical
    # place where the macro-parsing diagnostics are pinned.
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
        assert line == '! {color}="red","blue" : {mood}="calm"'
        assert prompt_parser.extract_variable_values(line) == (
            {"color": ["red", "blue"], "mood": ["calm"]},
            "",
        )
        assert prompt_parser.extract_variable_names(line) == (["color", "mood"], "")
