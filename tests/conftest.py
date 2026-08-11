"""Shared pytest fixtures and import helpers for the Wan2GP test suite.

The tests in this directory are deliberately *dependency free*: they exercise the
pure-python logic in the project (prompt parsing, filename templating, LoRA
multiplier maths, frame scheduling, resolution handling, config files) without
importing torch, gradio, diffusers or any other heavyweight runtime dependency.
That keeps CI fast and makes the suite runnable on any machine with a plain
python install.

Some of that pure logic lives in packages whose ``__init__`` eagerly imports the
heavy stack -- ``shared/utils/__init__.py`` for instance pulls in torch and
diffusers via ``fm_solvers``.  Importing ``shared.utils.prompt_parser`` normally
executes that ``__init__`` first and fails.  ``import_pure_module`` below sidesteps
this by registering a lightweight stand-in for the parent package before the
submodule is imported, so the submodule itself is loaded normally (relative
imports included) while the heavy ``__init__`` is never executed.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_stub_package(package_name: str) -> None:
    """Register ``package_name`` as an empty package rooted at its real directory.

    ``importlib`` only executes a package ``__init__`` when the package is not
    already present in ``sys.modules``.  By pre-seeding a bare module object that
    carries the correct ``__path__`` we let submodule imports resolve exactly as
    usual while skipping the expensive ``__init__``.
    """

    if package_name in sys.modules:
        return

    parent, _, _ = package_name.rpartition(".")
    if parent:
        _install_stub_package(parent)

    package_dir = REPO_ROOT / Path(*package_name.split("."))
    if not package_dir.is_dir():
        raise RuntimeError(f"cannot stub {package_name!r}: {package_dir} is not a directory")

    stub = types.ModuleType(package_name)
    stub.__path__ = [str(package_dir)]
    stub.__file__ = None
    sys.modules[package_name] = stub

    if parent:
        setattr(sys.modules[parent], package_name.rpartition(".")[2], stub)


def import_pure_module(module_name: str):
    """Import ``module_name`` without executing its parent packages' ``__init__``.

    Use this for modules that are themselves stdlib-only but live inside a package
    whose ``__init__`` drags in the heavy runtime stack.
    """

    parent, _, _ = module_name.rpartition(".")
    if parent:
        _install_stub_package(parent)
    return importlib.import_module(module_name)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def default_model_configs() -> list[tuple[Path, dict]]:
    """Every ``defaults/*.json`` model definition, parsed once per session."""

    configs = []
    for path in sorted((REPO_ROOT / "defaults").glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            configs.append((path, json.load(handle)))
    return configs
