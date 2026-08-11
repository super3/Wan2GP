"""Integrity checks for the JSON configuration bundled with the repository.

``defaults/`` holds ~212 model definitions. They are not documentation: ``wgp.py``
loads every one of them at startup, and ``refresh_model_defs`` re-raises on anything
it cannot parse or destructure::

    json_def = json.load(f)          # malformed JSON -> Exception for defaults/
    model_def = json_def.pop("model")  # missing "model" -> Exception for defaults/

So a single typo in this directory is a hard startup failure for every user, and the
kind of typo that is easy to make and invisible in review. These tests are cheap
insurance against that.

They are pure data checks: no project module is imported and no network request is
made. The architecture whitelist is recovered by parsing the handler modules with
``ast`` rather than importing them, since importing a handler would pull in torch.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Recovering the set of architectures the application actually supports
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
    tree = ast.parse(handler_path.read_text(encoding="utf-8"))
    constants = _module_level_constants(tree)
    found: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "query_supported_types"):
            continue
        for statement in ast.walk(node):
            if isinstance(statement, ast.Return) and statement.value is not None:
                found.update(
                    value
                    for value in _resolve_strings(statement.value, constants)
                    if isinstance(value, str)
                )
    return found


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
            except json.JSONDecodeError as exc:
                failures.append(f"{path.name}: {exc}")
        assert not failures, "malformed JSON in defaults/:\n" + "\n".join(failures)

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


class TestModelMetadata:
    def test_every_model_has_a_non_empty_name(self, default_model_configs):
        bad = [
            p.name
            for p, cfg in default_model_configs
            if not isinstance(cfg["model"].get("name"), str) or not cfg["model"]["name"].strip()
        ]
        assert not bad, f"files with a missing or blank model name: {bad}"

    def test_model_names_are_unique(self, default_model_configs):
        # Two models sharing a display name are indistinguishable in the UI dropdown.
        seen: dict[str, list[str]] = {}
        for path, cfg in default_model_configs:
            seen.setdefault(cfg["model"]["name"], []).append(path.name)
        duplicates = {name: files for name, files in seen.items() if len(files) > 1}
        assert not duplicates, f"duplicate model names: {duplicates}"

    def test_architecture_when_present_is_a_non_empty_string(self, default_model_configs):
        bad = []
        for path, cfg in default_model_configs:
            architecture = cfg["model"].get("architecture")
            if architecture is not None and (
                not isinstance(architecture, str) or not architecture.strip()
            ):
                bad.append(f"{path.name}: {architecture!r}")
        assert not bad, f"invalid architecture values: {bad}"

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


class TestModelUrls:
    """``URLs`` is either a list of sources or a string naming another model.

    get_model_recursive_prop() follows the string form recursively, so a dangling
    reference raises at download time and a cycle trips its depth guard.
    """

    @staticmethod
    def _url_items(cfg: dict):
        return [(k, v) for k, v in cfg["model"].items() if k.startswith("URLs")]

    def test_every_model_declares_urls(self, default_model_configs):
        missing = [p.name for p, cfg in default_model_configs if "URLs" not in cfg["model"]]
        assert not missing, f"model definitions with no URLs key: {missing}"

    def test_url_values_are_lists_or_strings(self, default_model_configs):
        bad = []
        for path, cfg in default_model_configs:
            for key, value in self._url_items(cfg):
                if not isinstance(value, (list, str)):
                    bad.append(f"{path.name}:{key} is {type(value).__name__}")
        assert not bad, f"URL entries must be a list or a string: {bad}"

    def test_url_lists_are_non_empty_and_hold_supported_entry_types(self, default_model_configs):
        bad = []
        for path, cfg in default_model_configs:
            for key, value in self._url_items(cfg):
                if not isinstance(value, list):
                    continue
                if not value:
                    bad.append(f"{path.name}:{key} is an empty list")
                for entry in value:
                    if not isinstance(entry, (str, list, dict)):
                        bad.append(f"{path.name}:{key} holds a {type(entry).__name__}")
        assert not bad, f"malformed URL lists: {bad}"

    def test_string_urls_are_absolute_https(self, default_model_configs):
        # Checked as text only -- nothing here touches the network.
        bad = []
        for path, cfg in default_model_configs:
            for key, value in self._url_items(cfg):
                if not isinstance(value, list):
                    continue
                for entry in value:
                    if isinstance(entry, str) and not entry.startswith("https://"):
                        bad.append(f"{path.name}:{key} -> {entry!r}")
        assert not bad, f"URLs that are not absolute https:// links: {bad}"

    def test_model_references_point_at_an_existing_definition(self, default_model_configs):
        known = {path.stem for path, _ in default_model_configs}
        dangling = []
        for path, cfg in default_model_configs:
            for key, value in self._url_items(cfg):
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
            for key, value in self._url_items(cfg):
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


class TestTopLevelConfigFiles:
    def test_plugins_json_is_a_list_of_described_plugins(self, repo_root):
        plugins = json.loads((repo_root / "plugins.json").read_text(encoding="utf-8"))
        assert isinstance(plugins, list) and plugins

        required = {"name", "url"}
        for index, plugin in enumerate(plugins):
            assert isinstance(plugin, dict), f"plugins.json[{index}] is not an object"
            missing = required - plugin.keys()
            assert not missing, f"plugins.json[{index}] ({plugin.get('name')!r}) missing {missing}"
            assert isinstance(plugin["name"], str) and plugin["name"].strip()
            assert isinstance(plugin["url"], str) and plugin["url"].startswith(("http://", "https://"))

    def test_plugin_names_are_unique(self, repo_root):
        plugins = json.loads((repo_root / "plugins.json").read_text(encoding="utf-8"))
        names = [p["name"] for p in plugins]
        assert len(names) == len(set(names)), f"duplicate plugin names in plugins.json: {names}"

    def test_setup_config_json_has_its_expected_sections(self, repo_root):
        config = json.loads((repo_root / "setup_config.json").read_text(encoding="utf-8"))
        assert isinstance(config, dict)
        for section in ("components", "gpu_profiles"):
            assert section in config, f"setup_config.json is missing {section!r}"
