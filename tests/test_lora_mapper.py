"""Tests for the pure key-remapping helpers.

Covered here:

* ``shared/lora_mapper.py`` -- ``LoraKeyMapper``: the alias/flattened indexes
  built from a set of target module names, ``map_key`` (adapter-suffix
  splitting, ``lora_unet_`` flattened names, the ``transformer.`` /
  ``diffusion_model.`` namespaces, ambiguous aliases) and ``map_state_dict``
  (order, value pass-through, collision detection).
* ``shared/utils/gguf_mapping.py`` -- ``has_standard_gguf_tensor_names``,
  ``remap_named_mapping`` (unmapped keys, list values, mapping-class
  preservation, duplicate detection) and ``remap_state_dict_triplet``.
* ``shared/tools/sha256_verify.py`` -- ``compute_sha256`` against real files
  created under ``tmp_path``: chunking, verification success/failure and the
  missing-file error.

Two tests deliberately pin behaviour that looks wrong; each one carries a
comment saying so.
"""

import hashlib
from collections import OrderedDict

import pytest

import shared.lora_mapper as lora_mapper
import shared.tools.sha256_verify as sha256_verify

import shared.utils.gguf_mapping as gguf_mapping

LoraKeyMapper = lora_mapper.LoraKeyMapper


@pytest.fixture
def mapper():
    """Mapper with a single canonical target module."""

    return LoraKeyMapper(["blocks.0.q_proj"])


class TestModuleConstants:
    def test_public_surface(self):
        assert lora_mapper.__all__ == ["COMMON_LORA_ALIASES", "LoraKeyMapper"]

    def test_common_aliases(self):
        assert lora_mapper.COMMON_LORA_ALIASES == (
            ("transformer_blocks", "blocks"),
            ("to_q", "q_proj"),
            ("to_k", "k_proj"),
            ("to_v", "v_proj"),
            ("to_out.0", "out_proj"),
        )


class TestMapperIndexes:
    def test_alias_index_lists_every_non_identity_variant(self, mapper):
        assert mapper.aliases == {
            "blocks.0.to_q": "blocks.0.q_proj",
            "transformer_blocks.0.q_proj": "blocks.0.q_proj",
            "transformer_blocks.0.to_q": "blocks.0.q_proj",
        }

    def test_flattened_index_includes_the_target_itself(self, mapper):
        assert mapper.flattened == {
            "blocks_0_q_proj": "blocks.0.q_proj",
            "blocks_0_to_q": "blocks.0.q_proj",
            "transformer_blocks_0_q_proj": "blocks.0.q_proj",
            "transformer_blocks_0_to_q": "blocks.0.q_proj",
        }

    def test_module_names_is_a_frozenset_without_falsy_entries(self):
        built = LoraKeyMapper(["", None, "blocks.0.q_proj", "blocks.0.q_proj"])
        assert isinstance(built.module_names, frozenset)
        assert built.module_names == frozenset({"blocks.0.q_proj"})

    def test_multi_part_alias_expands_both_ways(self):
        built = LoraKeyMapper(["blocks.0.attn.out_proj"])
        assert built.aliases["blocks.0.attn.to_out.0"] == "blocks.0.attn.out_proj"
        assert built.flattened["transformer_blocks_0_attn_to_out_0"] == "blocks.0.attn.out_proj"

    def test_aliases_apply_in_the_reverse_direction_too(self):
        # Target named the "diffusers" way: keys named the "native" way resolve.
        built = LoraKeyMapper(["transformer_blocks.0.to_k"])
        assert built.map_key("blocks.0.k_proj.alpha") == "transformer_blocks.0.to_k.alpha"

    def test_extra_aliases_compose_with_the_common_ones(self):
        built = LoraKeyMapper(
            ["blocks.0.ffn.net"], aliases=[("ff", "ffn"), ("proj_out", "net")]
        )
        assert built.map_key("transformer_blocks.0.ff.proj_out.lora_A.weight") == (
            "blocks.0.ffn.net.lora_A.weight"
        )

    def test_empty_module_names_leaves_every_key_alone(self):
        built = LoraKeyMapper([])
        assert built.aliases == {}
        assert built.flattened == {}
        assert built.map_key("transformer_blocks.0.to_q.alpha") == "transformer_blocks.0.to_q.alpha"

    def test_ambiguous_variants_are_recorded_as_none_and_never_remapped(self):
        # Both targets generate the *same* four variants, so the two variants
        # that are not themselves module names are ambiguous.
        built = LoraKeyMapper(["blocks.0.q_proj", "transformer_blocks.0.to_q"])
        assert built.aliases["blocks.0.to_q"] is None
        assert built.aliases["transformer_blocks.0.q_proj"] is None
        assert built.map_key("blocks.0.to_q.alpha") == "blocks.0.to_q.alpha"
        assert built.map_key("transformer_blocks.0.q_proj.alpha") == (
            "transformer_blocks.0.q_proj.alpha"
        )

    def test_ambiguous_flattened_names_are_not_remapped(self):
        built = LoraKeyMapper(["a.to_q", "a.q_proj"])
        assert built.flattened == {"a_to_q": None, "a_q_proj": None}
        assert built.map_key("lora_unet_a_to_q.alpha") == "lora_unet_a_to_q.alpha"


class TestMapKey:
    @pytest.mark.parametrize(
        "key, expected",
        [
            (
                "transformer_blocks.0.to_q.lora_A.weight",
                "blocks.0.q_proj.lora_A.weight",
            ),
            (
                "transformer_blocks.0.to_q.lora_down.weight",
                "blocks.0.q_proj.lora_down.weight",
            ),
            ("blocks.0.to_q.lora_up.weight", "blocks.0.q_proj.lora_up.weight"),
            (
                "transformer_blocks.0.q_proj.lora_B.weight",
                "blocks.0.q_proj.lora_B.weight",
            ),
            # ".lora." separator instead of ".lora_".
            ("blocks.0.to_q.lora.up.weight", "blocks.0.q_proj.lora.up.weight"),
        ],
    )
    def test_known_variants_are_rewritten(self, mapper, key, expected):
        assert mapper.map_key(key) == expected

    @pytest.mark.parametrize(
        "suffix", [".alpha", ".diff", ".diff_b", ".dora_scale", ".lokr_w1", ".lokr_w2"]
    )
    def test_terminal_suffixes_are_recognised(self, mapper, suffix):
        assert mapper.map_key("transformer_blocks.0.to_q" + suffix) == (
            "blocks.0.q_proj" + suffix
        )

    def test_canonical_key_is_returned_unchanged(self, mapper):
        assert mapper.map_key("blocks.0.q_proj.lora_A.weight") == (
            "blocks.0.q_proj.lora_A.weight"
        )

    def test_unknown_module_is_returned_unchanged(self, mapper):
        assert mapper.map_key("unknown.module.lora_A.weight") == "unknown.module.lora_A.weight"

    @pytest.mark.parametrize(
        "key",
        [
            # No adapter marker and no terminal suffix at all.
            "transformer_blocks.0.to_q.weight",
            "transformer_blocks.0.to_q",
            "",
            # The marker sits at index 0, which the splitter requires to be > 0.
            ".lora_up.weight",
            ".lora.up.weight",
        ],
    )
    def test_keys_without_a_usable_adapter_suffix_pass_through(self, mapper, key):
        assert mapper.map_key(key) == key

    def test_only_the_last_adapter_marker_is_used(self, mapper):
        # ".lora." here is not the split point -- the later ".lora_down" wins,
        # which leaves an unknown module name, so the key is left alone.
        key = "transformer_blocks.0.to_q.lora.up.lora_down.weight"
        assert mapper.map_key(key) == key

    @pytest.mark.parametrize("namespace", ["transformer.", "diffusion_model."])
    def test_known_namespaces_are_preserved_around_the_rewrite(self, mapper, namespace):
        key = namespace + "transformer_blocks.0.to_q.lora_A.weight"
        assert mapper.map_key(key) == namespace + "blocks.0.q_proj.lora_A.weight"

    @pytest.mark.parametrize("namespace", ["transformer.", "diffusion_model."])
    def test_namespaced_canonical_key_keeps_its_prefix(self, mapper, namespace):
        # Wrapper-prefix removal is documented as MMGP's job, so a key that is
        # already canonical under a namespace is returned verbatim.
        key = namespace + "blocks.0.q_proj.alpha"
        assert mapper.map_key(key) == key

    def test_unknown_namespace_is_not_stripped(self, mapper):
        assert mapper.map_key("model.blocks.0.to_q.alpha") == "model.blocks.0.to_q.alpha"

    @pytest.mark.parametrize(
        "key, expected",
        [
            (
                "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight",
                "blocks.0.attn.q_proj.lora_down.weight",
            ),
            (
                "lora_unet_blocks_0_attn_q_proj.alpha",
                "blocks.0.attn.q_proj.alpha",
            ),
            # Everything after the first "." is kept verbatim as the suffix.
            (
                "lora_unet_blocks_0_attn_to_q.anything.at.all",
                "blocks.0.attn.q_proj.anything.at.all",
            ),
        ],
    )
    def test_flattened_lora_unet_keys_are_expanded(self, key, expected):
        built = LoraKeyMapper(["blocks.0.attn.q_proj"])
        assert built.map_key(key) == expected

    @pytest.mark.parametrize(
        "key",
        [
            # No "." after the prefix -> no module/suffix split at all.
            "lora_unet_blocks_0_attn_to_q",
            # Empty module name after the prefix.
            "lora_unet_.lora_down.weight",
            # Unknown flattened module.
            "lora_unet_nope_0.lora_down.weight",
        ],
    )
    def test_unresolvable_flattened_keys_pass_through(self, key):
        built = LoraKeyMapper(["blocks.0.attn.q_proj"])
        assert built.map_key(key) == key

    def test_flattened_lookup_ignores_namespaces(self):
        # "lora_unet_" names are matched against the flattened index only, so a
        # namespace baked into the flat name simply fails to resolve.
        built = LoraKeyMapper(["blocks.0.q_proj"])
        key = "lora_unet_transformer_blocks_0_to_q.alpha"
        assert built.map_key(key) == "blocks.0.q_proj.alpha"
        assert built.map_key("lora_unet_transformer_blocks_0_to_q_extra.alpha") == (
            "lora_unet_transformer_blocks_0_to_q_extra.alpha"
        )


class TestMapStateDict:
    def test_values_pass_through_by_identity_and_order_is_preserved(self, mapper):
        first, second = object(), object()
        state_dict = {
            "transformer_blocks.0.to_q.lora_A.weight": first,
            "untouched.key.alpha": second,
        }
        mapped = mapper.map_state_dict(state_dict)
        assert list(mapped) == ["blocks.0.q_proj.lora_A.weight", "untouched.key.alpha"]
        assert mapped["blocks.0.q_proj.lora_A.weight"] is first
        assert mapped["untouched.key.alpha"] is second

    def test_input_is_not_mutated(self, mapper):
        state_dict = {"transformer_blocks.0.to_q.alpha": 1}
        mapped = mapper.map_state_dict(state_dict)
        assert state_dict == {"transformer_blocks.0.to_q.alpha": 1}
        assert mapped is not state_dict

    def test_empty_state_dict_yields_empty_dict(self, mapper):
        assert mapper.map_state_dict({}) == {}

    def test_colliding_keys_raise(self, mapper):
        with pytest.raises(ValueError, match="collide after mapping to 'blocks.0.q_proj.alpha'"):
            mapper.map_state_dict(
                {
                    "blocks.0.q_proj.alpha": 1,
                    "transformer_blocks.0.to_q.alpha": 2,
                }
            )

    def test_instance_is_callable(self, mapper):
        state_dict = {"transformer_blocks.0.to_q.alpha": 7}
        assert mapper(state_dict) == mapper.map_state_dict(state_dict)

    def test_accepts_any_mapping_and_returns_a_plain_dict(self, mapper):
        source = OrderedDict([("transformer_blocks.0.to_q.alpha", 1)])
        mapped = mapper.map_state_dict(source)
        assert type(mapped) is dict
        assert mapped == {"blocks.0.q_proj.alpha": 1}


class TestHasStandardGgufTensorNames:
    @pytest.mark.parametrize("prefix", ["blk.", "enc.blk.", "dec.blk."])
    def test_recognised_block_prefixes(self, prefix):
        state_dict = {"token_embd.weight": 1, prefix + "0.attn_q.weight": 2}
        assert gguf_mapping.has_standard_gguf_tensor_names(state_dict) is True

    @pytest.mark.parametrize(
        "state_dict",
        [
            None,
            {},
            [],
            # Token embedding present but no block-style names.
            {"token_embd.weight": 1, "output_norm.weight": 2},
            # Block-style names but no token embedding.
            {"blk.0.attn_q.weight": 1},
            # "blk." must be a prefix, not just a substring.
            {"token_embd.weight": 1, "model.blk.0.weight": 2},
        ],
    )
    def test_rejected_inputs(self, state_dict):
        assert gguf_mapping.has_standard_gguf_tensor_names(state_dict) is False

    def test_works_on_a_plain_sequence_of_names(self):
        assert gguf_mapping.has_standard_gguf_tensor_names(
            ["token_embd.weight", "blk.0.attn_q.weight"]
        ) is True


class TestRemapNamedMapping:
    def test_none_mapping_returns_none(self):
        assert gguf_mapping.remap_named_mapping(None, {"a": "A"}) is None

    def test_empty_mapping_returns_empty_mapping(self):
        assert gguf_mapping.remap_named_mapping({}, {"a": "A"}) == {}

    def test_unmapped_names_are_kept_by_default(self):
        source = {"a": 1, "b": 2}
        assert gguf_mapping.remap_named_mapping(source, {"a": "A"}) == {"A": 1, "b": 2}

    def test_unmapped_names_are_dropped_when_asked(self):
        source = {"a": 1, "b": 2}
        assert gguf_mapping.remap_named_mapping(source, {"a": "A"}, keep_unmapped=False) == {"A": 1}

    def test_empty_name_map_is_a_copy(self):
        source = {"a": 1}
        result = gguf_mapping.remap_named_mapping(source, {})
        assert result == source
        assert result is not source

    def test_input_is_not_mutated(self):
        source = {"a": 1, "b": 2}
        gguf_mapping.remap_named_mapping(source, {"a": "A", "b": "B"})
        assert source == {"a": 1, "b": 2}

    def test_mapping_class_is_preserved_along_with_order(self):
        source = OrderedDict([("b", 1), ("a", 2)])
        result = gguf_mapping.remap_named_mapping(source, {"b": "B"})
        assert type(result) is OrderedDict
        assert list(result.items()) == [("B", 1), ("a", 2)]

    def test_list_values_are_remapped_elementwise(self):
        source = {"tied": ["a", "b", "unknown"]}
        result = gguf_mapping.remap_named_mapping(source, {"a": "A", "b": "B", "tied": "TIED"})
        assert result == {"TIED": ["A", "B", "unknown"]}

    def test_non_list_values_are_left_alone(self):
        source = {"a": ("a", "b"), "c": "a"}
        result = gguf_mapping.remap_named_mapping(source, {"a": "A", "b": "B", "c": "C"})
        assert result == {"A": ("a", "b"), "C": "a"}

    def test_two_names_mapped_onto_one_raise(self):
        with pytest.raises(ValueError, match="Duplicate weight after GGUF mapping: X"):
            gguf_mapping.remap_named_mapping({"a": 1, "b": 2}, {"a": "X", "b": "X"})

    def test_collision_with_a_kept_unmapped_name_raises(self):
        with pytest.raises(ValueError, match="Duplicate weight after GGUF mapping: b"):
            gguf_mapping.remap_named_mapping({"a": 1, "b": 2}, {"a": "b"})

    def test_collision_avoided_when_unmapped_names_are_dropped(self):
        assert gguf_mapping.remap_named_mapping(
            {"a": 1, "b": 2}, {"a": "b"}, keep_unmapped=False
        ) == {"b": 1}


class TestRemapStateDictTriplet:
    def test_all_three_mappings_are_remapped(self):
        state_dict, quant_map, tied_map = gguf_mapping.remap_state_dict_triplet(
            {"a": 1, "b": 2},
            {"a": "int8"},
            {"a": ["b"]},
            {"a": "A", "b": "B"},
        )
        assert state_dict == {"A": 1, "B": 2}
        assert quant_map == {"A": "int8"}
        assert tied_map == {"A": ["B"]}

    def test_none_members_stay_none(self):
        result = gguf_mapping.remap_state_dict_triplet({"a": 1}, None, None, {"a": "A"})
        assert result == ({"A": 1}, None, None)

    def test_keep_unmapped_applies_to_every_member(self):
        # NOTE: pins current behaviour -- with keep_unmapped=False the *whole*
        # tied-weights entry is dropped because its key ("tied") is absent from
        # the name map, even though its list values would have remapped fine.
        state_dict, quant_map, tied_map = gguf_mapping.remap_state_dict_triplet(
            {"a": 1, "b": 2},
            {"a": "int8", "b": "int8"},
            {"tied": ["a"]},
            {"a": "A"},
            keep_unmapped=False,
        )
        assert state_dict == {"A": 1}
        assert quant_map == {"A": "int8"}
        assert tied_map == {}


@pytest.fixture
def sample_file(tmp_path):
    payload = b"wan2gp sample payload\n" * 500
    path = tmp_path / "model.safetensors"
    path.write_bytes(payload)
    return path, payload, hashlib.sha256(payload).hexdigest()


class TestComputeSha256:
    def test_matches_hashlib(self, sample_file):
        path, _payload, digest = sample_file
        assert sha256_verify.compute_sha256(path) == digest

    def test_accepts_a_string_path(self, sample_file):
        path, _payload, digest = sample_file
        assert sha256_verify.compute_sha256(str(path)) == digest

    @pytest.mark.parametrize("chunk_size", [1, 7, 8192, 1 << 20])
    def test_chunk_size_does_not_change_the_digest(self, sample_file, chunk_size):
        path, _payload, digest = sample_file
        assert sha256_verify.compute_sha256(path, chunk_size=chunk_size) == digest

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.bin"
        path.write_bytes(b"")
        assert sha256_verify.compute_sha256(path) == hashlib.sha256(b"").hexdigest()

    def test_zero_chunk_size_silently_hashes_nothing(self, sample_file):
        # BUG (pinned, not fixed): ``f.read(0)`` returns b"" immediately, so the
        # walrus loop never runs and the digest of the *empty* string is
        # returned for any file content.
        path, _payload, digest = sample_file
        result = sha256_verify.compute_sha256(path, chunk_size=0)
        assert result == hashlib.sha256(b"").hexdigest()
        assert result != digest

    def test_no_output_when_no_expected_hash_is_given(self, sample_file, capsys):
        path, _payload, _digest = sample_file
        sha256_verify.compute_sha256(path)
        assert capsys.readouterr().out == ""

    def test_matching_expected_hash_returns_the_digest_and_reports(self, sample_file, capsys):
        path, _payload, digest = sample_file
        assert sha256_verify.compute_sha256(path, digest) == digest
        assert digest in capsys.readouterr().out

    @pytest.mark.parametrize("decorate", [str.upper, lambda h: "  " + h + "\n"])
    def test_expected_hash_is_normalised_before_comparing(self, sample_file, decorate):
        path, _payload, digest = sample_file
        assert sha256_verify.compute_sha256(path, decorate(digest)) == digest

    def test_mismatching_expected_hash_raises_with_both_hashes(self, sample_file, capsys):
        path, _payload, digest = sample_file
        wrong = "0" * 64
        with pytest.raises(ValueError) as excinfo:
            sha256_verify.compute_sha256(path, wrong)
        message = str(excinfo.value)
        assert "Hash mismatch!" in message
        assert wrong in message
        assert digest in message
        assert capsys.readouterr().out == ""

    def test_near_miss_expected_hash_still_fails(self, sample_file):
        path, _payload, digest = sample_file
        with pytest.raises(ValueError):
            sha256_verify.compute_sha256(path, digest[:-1] + ("0" if digest[-1] != "0" else "1"))

    def test_missing_file_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "nope.safetensors"
        with pytest.raises(FileNotFoundError, match="File not found"):
            sha256_verify.compute_sha256(missing)

    def test_missing_file_is_checked_before_the_expected_hash(self, tmp_path):
        missing = tmp_path / "nope.safetensors"
        with pytest.raises(FileNotFoundError):
            sha256_verify.compute_sha256(missing, "0" * 64)
