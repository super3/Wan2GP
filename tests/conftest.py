"""Shared pytest fixtures for the Wan2GP test suite.

The tests in this directory are deliberately *dependency free*: they exercise the
pure-python logic in the project (prompt parsing, filename templating, LoRA multiplier
maths, frame scheduling, resolution handling, config files) without importing torch,
gradio, diffusers or any other heavyweight runtime dependency. That keeps CI fast and
makes the suite runnable on any machine with a plain python install.

This only works because the packages those modules live in stay importable on their
own. ``tests/test_package_imports.py`` guards that property.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
