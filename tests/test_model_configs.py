"""Integrity checks for the JSON configuration bundled with the repository.

``defaults/`` holds ~212 model definitions. They are not documentation: ``wgp.py``
loads every one of them at startup, and ``refresh_model_defs`` re-raises on anything
it cannot parse or destructure::

    json_def = json.load(f)            # malformed JSON  -> Exception for defaults/
    model_def = json_def.pop("model")  # missing "model" -> Exception for defaults/

So a single typo in this directory is a hard startup failure for every user, and the
kind of typo that is easy to make and invisible in review. Other mistakes are worse
than a crash because they are silent: an unknown ``architecture`` makes the model
disappear from the UI, an unknown ``group`` is ignored, a duplicated display name
makes two entries indistinguishable in the dropdown.

Covered here:

* every ``defaults/*.json`` parses, is an object, and carries a ``model`` object;
* the keys the loader dereferences without a default (``name``, ``description``,
  ``architecture``) are present and sane, and typed fields really have that type;
* every architecture is claimed by exactly one handler, every ``group`` is a family
  some handler declares;
* ``URLs``/``URLs2``/``text_encoder_URLs``/``preload_URLs``/``loras``/``modules``
  hold either downloadable ``https://`` entries or a reference to another model
  definition, and those references resolve and terminate;
* ``plugins.json`` and ``setup_config.json`` parse and cross-reference correctly.

These are pure data checks: no project module is imported and **no network request is
made** -- URLs are inspected as text only. The architecture whitelist, the family
whitelist, the plugin type whitelist and the GPU profile keys are recovered by parsing
the relevant sources with ``ast`` rather than importing them, since importing a handler
or ``wgp.py`` would pull in torch.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Static analysis helpers
# ---------------------------------------------------------------------------


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _module_level_constants(tree: ast.Module) -> dict[str, object]:
    """Literal-valued assignments in a module, used to resolve names in a return."""

    constants: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                constants[target.id] = value
    return constants


def _functions_named(tree: ast.Module, name: str):
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]


def _returns_of(tree: ast.Module, function_name: str):
    for function in _functions_named(tree, function_name):
        for node in ast.walk(function):
            if isinstance(node, ast.Return) and node.value is not None:
                yield node.value


def _resolve_strings(node: ast.AST, constants: dict[str, object]):
    """Yield the string values an expression evaluates to, as far as we can tell.

    Handlers return their supported types in several shapes: a list literal, a name
    bound to a list, a list containing names, or a concatenation of those.
    """

    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        pass
    else:
        yield from [value] if isinstance(value, str) else value
        return

    if isinstance(node, ast.Name):
        value = constants.get(node.id)
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            # e.g. `return list(QWEN3_TTS_VARIANTS)` -- iterating a dict yields its keys,
            # and the keys are the architecture names.
            yield from value
        elif isinstance(value, (list, tuple, set)):
            yield from value
    elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        for element in node.elts:
            yield from _resolve_strings(element, constants)
    elif isinstance(node, ast.Call):
        for arg in node.args:
            yield from _resolve_strings(arg, constants)
    elif isinstance(node, ast.BinOp):
        yield from _resolve_strings(node.left, constants)
        yield from _resolve_strings(node.right, constants)


def _supported_types(handler_path: Path) -> set[str]:
    tree = _parse(handler_path)
    constants = _module_level_constants(tree)
    found: set[str] = set()
    for returned in _returns_of(tree, "query_supported_types"):
        found.update(v for v in _resolve_strings(returned, constants) if isinstance(v, str))
    return found


def _declared_families(handler_path: Path) -> set[str]:
    """Keys of the dict returned by ``query_family_infos`` -- the UI family names."""

    families: set[str] = set()
    for returned in _returns_of(_parse(handler_path), "query_family_infos"):
        try:
            value = ast.literal_eval(returned)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, dict):
            families.update(k for k in value if isinstance(k, str))
    return families


def _returned_string_constants(tree: ast.Module, function_name: str) -> set[str]:
    """Every string literal a function can return (handles `return a if c else b`)."""

    found: set[str] = set()
    for returned in _returns_of(tree, function_name):
        for node in ast.walk(returned):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.add(node.value)
    return found


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
#
# wgp.py maps an architecture name to a handler via map_family_handlers(), which calls
# family_handler.query_supported_types() on each module listed in `family_handlers`.
# An architecture that no handler claims is silently dropped:
#
#     family_handler = model_types_handlers.get(base_model_type, None)
#     if family_handler is None:
#         ...
#         model_def["visible"] = False
#
# That is the failure mode worth catching -- a mistyped architecture does not crash,
# it makes the model quietly vanish from the UI.


@pytest.fixture(scope="session")
def handler_modules(repo_root: Path) -> list[Path]:
    """The handler modules listed in wgp.py's `family_handlers`."""

    source = (repo_root / "wgp.py").read_text(encoding="utf-8")
    match = re.search(r"^family_handlers = (\[.*?\])$", source, re.MULTILINE | re.DOTALL)
    assert match, "could not locate the `family_handlers` list in wgp.py"

    paths = []
    for dotted in ast.literal_eval(match.group(1)):
        path = repo_root / Path(*dotted.split(".")).with_suffix(".py")
        assert path.is_file(), f"wgp.py lists handler {dotted!r} but {path} does not exist"
        paths.append(path)
    return paths


@pytest.fixture(scope="session")
def registered_architectures(handler_modules: list[Path]) -> set[str]:
    architectures: set[str] = set()
    for path in handler_modules:
        architectures |= _supported_types(path)
    assert architectures, "recovered no architectures at all -- the ast walk is broken"
    return architectures


@pytest.fixture(scope="session")
def registered_families(handler_modules: list[Path]) -> set[str]:
    # map_family_handlers() seeds families_infos with "unknown" before merging in the
    # handlers' own query_family_infos().
    families = {"unknown"}
    for path in handler_modules:
        families |= _declared_families(path)
    assert len(families) > 1, "recovered no families at all -- the ast walk is broken"
    return families


@pytest.fixture(scope="session")
def setup_config(repo_root: Path) -> dict:
    return json.loads((repo_root / "setup_config.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def plugin_catalog(repo_root: Path) -> list:
    return json.loads((repo_root / "plugins.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Shape helpers for the model definitions
# ---------------------------------------------------------------------------

# get_model_recursive_prop() is called for these properties (wgp.py, ltx2.py,
# flux_main.py, model_dropdowns.py, ...). Each holds either a list of download
# entries or the *name of another model definition* to inherit them from.
_EXPLICIT_URL_PROPS = frozenset({"preload_URLs", "loras"})


def _url_props(model: dict):
    """Yield the (key, value) pairs of every download-bearing property."""

    for key, value in model.items():
        if key in _EXPLICIT_URL_PROPS or re.fullmatch(r"URLs\d*", key) or key.endswith("_URLs"):
            yield key, value


def _flatten_urls(value):
    """Yield every ``https://`` entry reachable from a property value."""

    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(current)
        elif isinstance(current, str) and current.startswith("https://"):
            yield current


# defaults/steadydancer.json lists the same int8 checkpoint twice in "URLs". It is
# harmless today (get_model_filename picks sub_choices[0]) but it is a copy/paste slip
# and almost certainly meant to be a second quantization. Pinned so the rest of the
# directory is still guarded; drop the entry once the source is fixed.
_KNOWN_DUPLICATE_URLS = {
    (
        "steadydancer.json",
        "URLs",
        "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/"
        "wan2.1_steadydancer_14B_quanto_mbf16_int8.safetensors",
    ),
}


class TestHandlerRegistry:
    def test_every_listed_handler_declares_supported_types(self, handler_modules):
        """A handler that declares nothing would make its models unreachable."""

        silent = [p.name for p in handler_modules if not _supported_types(p)]
        assert not silent, f"handlers declaring no supported types: {silent}"

    def test_no_architecture_is_claimed_by_two_handlers(self, handler_modules):
        # map_family_handlers() raises outright on a duplicate:
        #   raise Exception(f"Model type {model_type} supported by {prev} and ...")
        owners: dict[str, list[str]] = {}
        for path in handler_modules:
            for architecture in _supported_types(path):
                owners.setdefault(architecture, []).append(path.name)
        clashes = {a: o for a, o in owners.items() if len(o) > 1}
        assert not clashes, f"architectures claimed by multiple handlers: {clashes}"


class TestDefaultsParse:
    def test_defaults_directory_is_populated(self, default_model_configs):
        # Guards the rest of this file: every other test below iterates this fixture,
        # so an empty glob would make them all vacuously pass.
        assert len(default_model_configs) > 100

    def test_every_file_is_valid_json(self, repo_root):
        # The fixture already parses these, but it fails at collection time with a
        # traceback that names only the first bad file. Report all of them at once.
        failures = []
        for path in sorted((repo_root / "defaults").glob("*.json")):
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                failures.append(f"{path.name}: {exc}")
        assert not failures, "malformed JSON in defaults/:\n" + "\n".join(failures)

    def test_every_file_is_a_json_object(self, default_model_configs):
        # refresh_model_defs() calls json_def.pop("model"); a list or scalar top level
        # would blow up with AttributeError.
        wrong = [
            f"{p.name}: {type(cfg).__name__}"
            for p, cfg in default_model_configs
            if not isinstance(cfg, dict)
        ]
        assert not wrong, f"top level of a model definition must be an object: {wrong}"

    def test_every_file_has_a_model_block(self, default_model_configs):
        # refresh_model_defs() does json_def.pop("model") and re-raises for defaults/.
        missing = [p.name for p, cfg in default_model_configs if "model" not in cfg]
        assert not missing, f"files with no top-level 'model' key: {missing}"

    def test_model_block_is_an_object(self, default_model_configs):
        wrong = [
            f"{p.name}: {type(cfg['model']).__name__}"
            for p, cfg in default_model_configs
            if not isinstance(cfg.get("model"), dict)
        ]
        assert not wrong, f"'model' must be an object: {wrong}"

    def test_filenames_do_not_shadow_each_other(self, default_model_configs):
        """The filename stem *is* the model type, so a collision merges two models.

        ``model_type = os.path.basename(file_path)[:-5]``. Stems are unique within one
        directory by construction, but two stems differing only in case collide on
        Windows and on the default macOS filesystem.
        """

        by_lowercase: dict[str, list[str]] = {}
        for path, _ in default_model_configs:
            by_lowercase.setdefault(path.stem.lower(), []).append(path.name)
        clashes = {stem: names for stem, names in by_lowercase.items() if len(names) > 1}
        assert not clashes, f"model definition filenames differing only in case: {clashes}"


class TestModelMetadata:
    def test_every_model_has_a_non_empty_name(self, default_model_configs):
        # get_model_name() reads model_def["name"] with no default.
        bad = [
            p.name
            for p, cfg in default_model_configs
            if not isinstance(cfg["model"].get("name"), str) or not cfg["model"]["name"].strip()
        ]
        assert not bad, f"files with a missing or blank model name: {bad}"

    def test_every_model_has_a_non_empty_description(self, default_model_configs):
        # get_model_name() also reads model_def["description"] with no default and
        # hands it straight to the UI.
        bad = [
            p.name
            for p, cfg in default_model_configs
            if not isinstance(cfg["model"].get("description"), str)
            or not cfg["model"]["description"].strip()
        ]
        assert not bad, f"files with a missing or blank description: {bad}"

    def test_model_names_are_unique(self, default_model_configs):
        # Two models sharing a display name are indistinguishable in the UI dropdown.
        seen: dict[str, list[str]] = {}
        for path, cfg in default_model_configs:
            seen.setdefault(cfg["model"]["name"], []).append(path.name)
        duplicates = {name: files for name, files in seen.items() if len(files) > 1}
        assert not duplicates, f"duplicate model names: {duplicates}"

    def test_every_model_declares_a_non_empty_architecture(self, default_model_configs):
        # get_base_model_type() does `return model_def["architecture"]` -- a definition
        # without one raises KeyError the moment anything asks for its family.
        bad = []
        for path, cfg in default_model_configs:
            architecture = cfg["model"].get("architecture")
            if not isinstance(architecture, str) or not architecture.strip():
                bad.append(f"{path.name}: {architecture!r}")
        assert not bad, f"missing or invalid architecture values: {bad}"

    def test_every_architecture_is_backed_by_a_handler(
        self, default_model_configs, registered_architectures
    ):
        """The headline check: a mistyped architecture hides the model silently.

        init_model_def() falls back to the filename stem when "architecture" is absent
        (via get_base_model_type), so that is the value validated for those files.
        """

        unbacked = []
        for path, cfg in default_model_configs:
            effective = cfg["model"].get("architecture") or path.stem
            if effective not in registered_architectures:
                unbacked.append(f"{path.name} -> {effective!r}")
        assert not unbacked, (
            "these models would be silently marked invisible at startup because no "
            f"handler claims their architecture:\n" + "\n".join(unbacked)
        )

    def test_group_when_present_is_a_family_some_handler_declares(
        self, default_model_configs, registered_families
    ):
        """get_model_family(for_ui=True) ignores a "group" that no handler declares.

            model_family = model_def.get("group", None)
            if model_family is not None and model_family in families_infos:
                return model_family

        A typo here does not crash, it just silently files the model under its
        handler's default family.
        """

        unknown = []
        for path, cfg in default_model_configs:
            group = cfg["model"].get("group")
            if group is None:
                continue
            if not isinstance(group, str) or group not in registered_families:
                unknown.append(f"{path.name} -> {group!r}")
        assert not unknown, f"'group' values no handler declares: {unknown}"

    # Flags read through `model_def.get(<key>, False)` and used directly in a boolean
    # test. A string "false" would read as True, so the JSON type matters.
    BOOLEAN_FLAGS = (
        "visible",
        "image_outputs",
        "auto_quantize",
        "audio_only",
        "returns_audio",
        "i2v_class",
        "t2v_class",
        "vace_class",
        "inpaint_support",
        "v2i_switch_supported",
        "any_audio_prompt",
        "end_frames_always_enabled",
        "one_image_ref_needed",
        "one_image_ref_only",
        "one_speaker_only",
        "multi_speakers_only",
        "reference_image_enabled",
        "lock_guidance_phases",
        "unified_solver",
        "ltx2_msr",
    )

    def test_boolean_flags_are_json_booleans(self, default_model_configs):
        wrong = []
        for path, cfg in default_model_configs:
            for flag in self.BOOLEAN_FLAGS:
                if flag in cfg["model"] and not isinstance(cfg["model"][flag], bool):
                    wrong.append(f"{path.name}:{flag} = {cfg['model'][flag]!r}")
        assert not wrong, f"flags that must be true/false: {wrong}"

    def test_model_resolution_choices_are_label_value_pairs(self, default_model_configs):
        """normalize_resolution_choices() rejects anything but ["Label", "WxH"] pairs.

        It only prints and returns None, so a malformed entry silently throws away the
        model's whole custom resolution list.
        """

        bad = []
        for path, cfg in default_model_configs:
            choices = cfg["model"].get("resolutions")
            if choices is None:
                continue
            if not isinstance(choices, list) or not choices:
                bad.append(f"{path.name}: resolutions is {choices!r}")
                continue
            for choice in choices:
                if (
                    not isinstance(choice, list)
                    or len(choice) != 2
                    or not all(isinstance(part, str) for part in choice)
                    or not re.fullmatch(r"\d+x\d+", choice[1])
                ):
                    bad.append(f"{path.name}: {choice!r}")
        assert not bad, f"invalid model 'resolutions' entries: {bad}"

    def test_default_resolution_setting_has_the_wxh_format(self, default_model_configs):
        # is_resolution_value() -> re.fullmatch(r"\d+x\d+", value.strip().lower()).
        # A value that fails it is silently replaced by the first available choice.
        bad = []
        for path, cfg in default_model_configs:
            resolution = cfg.get("resolution")
            if resolution is None:
                continue
            if not isinstance(resolution, str) or not re.fullmatch(
                r"\d+x\d+", resolution.strip().lower()
            ):
                bad.append(f"{path.name}: {resolution!r}")
        assert not bad, f"default 'resolution' settings that are not WxH: {bad}"


class TestModelUrls:
    """``URLs`` & co are either a list of sources or a string naming another model.

    get_model_recursive_prop() follows the string form recursively, so a dangling
    reference raises at download time and a cycle trips its depth guard.
    """

    def test_every_model_declares_urls(self, default_model_configs):
        missing = [p.name for p, cfg in default_model_configs if "URLs" not in cfg["model"]]
        assert not missing, f"model definitions with no URLs key: {missing}"

    def test_url_values_are_lists_or_strings(self, default_model_configs):
        bad = []
        for path, cfg in default_model_configs:
            for key, value in _url_props(cfg["model"]):
                if not isinstance(value, (list, str)):
                    bad.append(f"{path.name}:{key} is {type(value).__name__}")
        assert not bad, f"URL entries must be a list or a string: {bad}"

    def test_url_lists_hold_supported_entry_types(self, default_model_configs):
        bad = []
        for path, cfg in default_model_configs:
            for key, value in _url_props(cfg["model"]):
                if not isinstance(value, list):
                    continue
                for entry in value:
                    if not isinstance(entry, (str, list, dict)):
                        bad.append(f"{path.name}:{key} holds a {type(entry).__name__}")
        assert not bad, f"malformed URL lists: {bad}"

    def test_weight_lists_are_not_empty(self, default_model_configs):
        """An empty ``URLs``/``URLs2`` makes get_model_filename() return "".

        Only the transformer weight lists are checked: get_model_recursive_prop()
        treats an empty auxiliary list exactly like a missing key, and
        defaults/hunyuan_t2v_accvideo.json does ship an inert ``"preload_URLs": []``.
        """

        empty = []
        for path, cfg in default_model_configs:
            for key, value in cfg["model"].items():
                if re.fullmatch(r"URLs\d*", key) and isinstance(value, list) and not value:
                    empty.append(f"{path.name}:{key}")
        assert not empty, f"models whose weight list is empty: {empty}"

    def test_string_urls_are_absolute_https(self, default_model_configs):
        # Checked as text only -- nothing here touches the network.
        bad = []
        for path, cfg in default_model_configs:
            for key, value in _url_props(cfg["model"]):
                if not isinstance(value, list):
                    continue
                for entry in value:
                    if isinstance(entry, str) and not entry.startswith("https://"):
                        bad.append(f"{path.name}:{key} -> {entry!r}")
        assert not bad, f"URLs that are not absolute https:// links: {bad}"

    def test_urls_are_downloadable_huggingface_links(self, default_model_configs):
        """download_file() only knows how to resolve `<repo>/resolve/main/<file>`.

        Anything else falls through to the raw-download branch, which is not what these
        weights expect. The URL must also end in a real filename.
        """

        bad = []
        for path, cfg in default_model_configs:
            for key, value in _url_props(cfg["model"]):
                for url in _flatten_urls(value):
                    # download_file() drops everything after the '|' before resolving.
                    head = url.split("|")[0]
                    filename = head.rsplit("/", 1)[-1]
                    if not head.startswith("https://huggingface.co/"):
                        bad.append(f"{path.name}:{key} -> {url!r} (not huggingface.co)")
                    elif "/resolve/main/" not in head:
                        bad.append(f"{path.name}:{key} -> {url!r} (no /resolve/main/)")
                    elif not filename:
                        bad.append(f"{path.name}:{key} -> {url!r} (no filename)")
        assert not bad, f"URLs the downloader cannot resolve: {bad}"

    def test_url_alternate_paths_are_safe_relative_paths(self, default_model_configs):
        """A URL may carry a `|<target dir>` suffix consumed by extract_alternate_path.

        That function raises for more than one '|', and clean_relative_path() rejects a
        target that could escape the checkpoints folder (absolute, or containing '..').
        """

        bad = []
        for path, cfg in default_model_configs:
            for key, value in _url_props(cfg["model"]):
                for url in _flatten_urls(value):
                    parts = url.split("|")
                    if len(parts) == 1:
                        continue
                    if len(parts) > 2:
                        bad.append(f"{path.name}:{key} -> {url!r} (more than one '|')")
                        continue
                    target = parts[1]
                    if target == "%lora_dir":
                        continue
                    escapes = (
                        not target
                        or target.startswith(("/", "\\"))
                        or re.match(r"^[A-Za-z]:", target)
                        or ".." in Path(target).parts
                    )
                    if escapes:
                        bad.append(f"{path.name}:{key} -> {url!r} (unsafe target {target!r})")
        assert not bad, f"malformed '|' alternate paths: {bad}"

    def test_no_url_is_listed_twice_in_the_same_property(self, default_model_configs):
        """A repeated entry means a quantization variant is missing from the list.

        get_model_filename() picks the first match, so the duplicate silently shadows
        whatever should have been there.
        """

        duplicates = []
        for path, cfg in default_model_configs:
            for key, value in _url_props(cfg["model"]):
                seen = set()
                for url in _flatten_urls(value):
                    if url in seen and (path.name, key, url) not in _KNOWN_DUPLICATE_URLS:
                        duplicates.append(f"{path.name}:{key} -> {url}")
                    seen.add(url)
        assert not duplicates, f"URLs listed twice in the same property: {duplicates}"

    def test_model_references_point_at_an_existing_definition(self, default_model_configs):
        known = {path.stem for path, _ in default_model_configs}
        dangling = []
        for path, cfg in default_model_configs:
            for key, value in _url_props(cfg["model"]):
                if isinstance(value, str) and value not in known:
                    dangling.append(f"{path.name}:{key} -> {value!r}")
        assert not dangling, (
            f"URL references to model definitions that do not exist: {dangling}"
        )

    def test_model_references_terminate(self, default_model_configs):
        """get_model_recursive_prop() raises past a depth of 10; cycles never resolve."""

        by_stem = {path.stem: cfg["model"] for path, cfg in default_model_configs}
        problems = []
        for path, cfg in default_model_configs:
            for key, value in _url_props(cfg["model"]):
                current, seen = value, [path.stem]
                while isinstance(current, str):
                    if current in seen:
                        problems.append(f"{path.name}:{key} cycles through {seen + [current]}")
                        break
                    seen.append(current)
                    if len(seen) > 10:
                        problems.append(f"{path.name}:{key} exceeds the depth guard: {seen}")
                        break
                    current = by_stem.get(current, {}).get(key)
        assert not problems, f"unresolvable URL references: {problems}"

    def test_lora_multipliers_line_up_with_the_lora_list(self, default_model_configs):
        """get_transformer_loras() pads with 1.0 and then truncates to len(loras).

        A list that is longer than "loras" therefore has entries silently dropped, and
        declaring multipliers with no loras at all drops the lot.
        """

        by_stem = {path.stem: cfg["model"] for path, cfg in default_model_configs}

        def resolve(stem, prop):
            value, depth = by_stem.get(stem, {}).get(prop), 0
            while isinstance(value, str) and depth <= 10:
                value, depth = by_stem.get(value, {}).get(prop), depth + 1
            return value if isinstance(value, list) else []

        mismatched = []
        for path, _ in default_model_configs:
            loras = resolve(path.stem, "loras")
            multipliers = resolve(path.stem, "loras_multipliers")
            if multipliers and len(multipliers) != len(loras):
                mismatched.append(f"{path.name}: {len(loras)} loras, {len(multipliers)} multipliers")
        assert not mismatched, f"loras_multipliers out of step with loras: {mismatched}"


class TestModelModules:
    """``modules`` is the odd one out: entries may themselves name another model.

    wgp.py resolves it in two steps::

        modules = get_model_recursive_prop(model_type, "modules", return_list=True)
        modules = [get_model_recursive_prop(module, "modules", sub_prop_name="_list", ...)
                   if isinstance(module, str) else module for module in modules]

    and the ``"_list"`` sub-property raises unless the referenced definition's
    ``modules`` is a list of *exactly one* element.
    """

    @staticmethod
    def _resolve(by_stem, stem, stack=()):
        value = by_stem.get(stem, {}).get("modules")
        while isinstance(value, str):
            if value in stack or len(stack) > 10:
                raise AssertionError(f"unresolvable modules reference from {stem!r}")
            stack = stack + (value,)
            value = by_stem.get(value, {}).get("modules")
        return value

    def test_modules_is_a_list_or_a_reference(self, default_model_configs):
        bad = [
            f"{p.name}: {type(cfg['model']['modules']).__name__}"
            for p, cfg in default_model_configs
            if "modules" in cfg["model"] and not isinstance(cfg["model"]["modules"], (list, str))
        ]
        assert not bad, f"'modules' must be a list or a model reference: {bad}"

    def test_module_references_resolve_to_a_single_element_list(self, default_model_configs):
        by_stem = {path.stem: cfg["model"] for path, cfg in default_model_configs}
        problems = []
        for path, cfg in default_model_configs:
            if "modules" not in cfg["model"]:
                continue
            try:
                modules = self._resolve(by_stem, path.stem)
            except AssertionError as exc:
                problems.append(f"{path.name}: {exc}")
                continue
            if not isinstance(modules, list):
                problems.append(f"{path.name}: resolves to {type(modules).__name__}")
                continue
            for module in modules:
                if not isinstance(module, str):
                    continue
                if module not in by_stem:
                    problems.append(f"{path.name}: module {module!r} has no definition")
                    continue
                try:
                    target = self._resolve(by_stem, module)
                except AssertionError as exc:
                    problems.append(f"{path.name}: {exc}")
                    continue
                if not isinstance(target, list) or len(target) != 1:
                    problems.append(
                        f"{path.name}: module {module!r} must expose exactly one entry, "
                        f"found {target!r}"
                    )
        assert not problems, "broken 'modules' references:\n" + "\n".join(problems)


class TestPluginCatalog:
    """``plugins.json`` is the catalogue PluginManager.load_catalog_entries() reads."""

    def test_catalog_is_a_non_empty_list_of_objects(self, plugin_catalog):
        assert isinstance(plugin_catalog, list) and plugin_catalog
        wrong = [i for i, entry in enumerate(plugin_catalog) if not isinstance(entry, dict)]
        assert not wrong, f"plugins.json entries that are not objects: {wrong}"

    def test_every_plugin_has_a_name_and_a_url(self, plugin_catalog):
        # _merge_catalog_entries() skips an entry whose url yields no plugin id, so a
        # missing url makes the plugin invisible rather than noisy.
        bad = []
        for index, plugin in enumerate(plugin_catalog):
            name, url = plugin.get("name"), plugin.get("url")
            if not isinstance(name, str) or not name.strip():
                bad.append(f"[{index}] name={name!r}")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                bad.append(f"[{index}] ({name!r}) url={url!r}")
        assert not bad, f"malformed plugins.json entries: {bad}"

    def test_plugin_names_are_unique(self, plugin_catalog):
        names = [p["name"] for p in plugin_catalog]
        assert len(names) == len(set(names)), f"duplicate plugin names in plugins.json: {names}"

    def test_plugin_ids_are_unique(self, plugin_catalog):
        """The catalogue is keyed by plugin_id_from_url(url) -- the GitHub repo name.

        Two entries resolving to the same id silently overwrite one another in
        _merge_catalog_entries().
        """

        def plugin_id(url: str) -> str:
            cleaned = url.split("?", 1)[0].split("#", 1)[0].rstrip("/")
            marker = "github.com/"
            index = cleaned.lower().find(marker)
            if index < 0:
                return cleaned.rsplit("/", 1)[-1]
            parts = [part for part in cleaned[index + len(marker):].split("/") if part]
            repo = parts[1] if len(parts) > 1 else ""
            return repo[:-4] if repo.endswith(".git") else repo

        ids: dict[str, list[str]] = {}
        for plugin in plugin_catalog:
            ids.setdefault(plugin_id(plugin["url"]), []).append(plugin["name"])
        clashes = {i: names for i, names in ids.items() if len(names) > 1}
        assert not clashes, f"plugins.json entries sharing a plugin id: {clashes}"

    def test_plugin_types_are_recognised(self, repo_root, plugin_catalog):
        """normalize_plugin_types() drops unknown types and falls back to ["app"].

        A typo here silently reclassifies the plugin, so the whitelist is read straight
        out of shared/utils/plugins.py rather than duplicated.
        """

        constants = _module_level_constants(_parse(repo_root / "shared" / "utils" / "plugins.py"))
        choices = constants.get("PLUGIN_TYPE_CHOICES")
        assert choices, "could not recover PLUGIN_TYPE_CHOICES from shared/utils/plugins.py"

        bad = []
        for plugin in plugin_catalog:
            declared = plugin.get("type")
            declared = [declared] if isinstance(declared, str) else declared or []
            if not declared:
                bad.append(f"{plugin.get('name')!r}: no type")
            for value in declared:
                if not isinstance(value, str) or value.strip().lower() not in choices:
                    bad.append(f"{plugin.get('name')!r}: {value!r}")
        assert not bad, f"unknown plugin types (allowed: {sorted(choices)}): {bad}"


class TestSetupConfig:
    """``setup_config.json`` drives setup.py's installer menus."""

    # setup.py indexes config['components'][<group>] for each of these, and reads the
    # recommendation for the detected GPU out of config['gpu_profiles'][key][<group>].
    COMPONENT_GROUPS = ("python", "torch", "triton", "sage", "sparge", "flash", "kernels")

    def test_top_level_sections_are_present(self, setup_config):
        assert isinstance(setup_config, dict)
        for section in ("components", "gpu_profiles"):
            assert section in setup_config, f"setup_config.json is missing {section!r}"
            assert isinstance(setup_config[section], dict), f"{section!r} must be an object"

    @pytest.mark.parametrize("group", COMPONENT_GROUPS)
    def test_component_group_exists_and_is_populated(self, setup_config, group):
        components = setup_config["components"]
        assert group in components, f"components is missing {group!r}"
        assert isinstance(components[group], dict) and components[group], (
            f"components[{group!r}] must be a non-empty object"
        )

    def test_every_component_entry_has_a_label(self, setup_config):
        # menu() prints options[k]['label'] for every key -- a missing label is a
        # KeyError in the middle of an interactive install.
        bad = []
        for group, entries in setup_config["components"].items():
            for key, entry in entries.items():
                if not isinstance(entry, dict):
                    bad.append(f"{group}.{key} is {type(entry).__name__}")
                elif not isinstance(entry.get("label"), str) or not entry["label"].strip():
                    bad.append(f"{group}.{key} label={entry.get('label')!r}")
        assert not bad, f"component entries without a usable label: {bad}"

    def test_python_entries_declare_a_version(self, setup_config):
        # config['components']['python'][py_k]['ver'] is read directly.
        bad = [
            f"python.{key}"
            for key, entry in setup_config["components"]["python"].items()
            if not isinstance(entry.get("ver"), str) or not entry["ver"].strip()
        ]
        assert not bad, f"python entries with no 'ver': {bad}"

    def test_installable_entries_declare_a_command(self, repo_root, setup_config):
        """Every non-python component is installed via resolve_cmd(entry['cmd']).

        The dict form is keyed by get_os_key(), so the keys must be OS names setup.py
        can actually produce -- anything else resolves to None and is skipped in silence.
        """

        os_keys = _returned_string_constants(_parse(repo_root / "setup.py"), "get_os_key")
        assert os_keys, "could not recover the OS keys from setup.py"

        bad = []
        for group, entries in setup_config["components"].items():
            if group == "python":
                continue
            for key, entry in entries.items():
                command = entry.get("cmd")
                if isinstance(command, str):
                    if not command.strip():
                        bad.append(f"{group}.{key} has an empty cmd")
                elif isinstance(command, dict):
                    unknown = set(command) - os_keys
                    if unknown:
                        bad.append(f"{group}.{key} has unknown OS keys {sorted(unknown)}")
                    if not command:
                        bad.append(f"{group}.{key} has an empty cmd mapping")
                else:
                    bad.append(f"{group}.{key} cmd is {type(command).__name__}")
        assert not bad, f"component entries setup.py cannot install: {bad}"

    def test_every_detectable_gpu_profile_exists(self, repo_root, setup_config):
        """get_profile_key() feeds config['gpu_profiles'][detected_key] directly.

        Any key it can return but the file does not define is a KeyError on that GPU.
        """

        keys = _returned_string_constants(_parse(repo_root / "setup.py"), "get_profile_key")
        assert keys, "could not recover the profile keys from setup.py"
        missing = sorted(keys - setup_config["gpu_profiles"].keys())
        assert not missing, f"gpu_profiles missing entries returned by get_profile_key: {missing}"

    def test_profiles_declare_every_component_they_are_asked_for(self, setup_config):
        # do_install_interactive() reads base['python'], base['torch'], base['triton'],
        # base['sage'] and base['flash'] with [] (sparge and kernels use .get).
        required = ("python", "torch", "triton", "sage", "flash")
        bad = []
        for key, profile in setup_config["gpu_profiles"].items():
            if not isinstance(profile, dict):
                bad.append(f"{key} is {type(profile).__name__}")
                continue
            for group in required:
                if group not in profile:
                    bad.append(f"{key} is missing {group!r}")
        assert not bad, f"incomplete gpu_profiles: {bad}"

    def test_profile_recommendations_name_real_components(self, setup_config):
        """A recommendation is a key into components[<group>]; a typo installs nothing."""

        components = setup_config["components"]
        dangling = []
        for key, profile in setup_config["gpu_profiles"].items():
            for group in ("python", "torch", "triton", "sage", "sparge", "flash"):
                value = profile.get(group)
                if value is None:
                    continue  # null means "skip this component"
                if value not in components.get(group, {}):
                    dangling.append(f"{key}.{group} -> {value!r}")
            for kernel in profile.get("kernels") or []:
                if kernel not in components.get("kernels", {}):
                    dangling.append(f"{key}.kernels -> {kernel!r}")
        assert not dangling, f"gpu_profiles referencing unknown components: {dangling}"
