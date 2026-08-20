"""Tests for the pre-commit secret guard.

The guard's job is to fail closed on real credentials and stay quiet on the
placeholders and vendored bundles this repo is full of. A guard that cries wolf
gets bypassed with --no-verify and then protects nothing, so the false-positive
cases below matter as much as the true-positive ones.

No secrets live in this file: every "key" is assembled at runtime from a prefix
and filler, so the guard scanning its own repo does not flag its own tests.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GUARD = REPO / "runpod_worker" / "scripts" / "secret_guard.py"


def _load():
    spec = importlib.util.spec_from_file_location("secret_guard", GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load()


# Built, never written literally -- see the module docstring.
RUNPOD = "rpa_" + "A1b2C3d4E5" * 4
DOCKER = "dckr_pat_" + "Xy9Zw8Vu7T" * 2
AWS = "AKIA" + "ABCDEFGHIJKLMNOP"
GITHUB = "ghp_" + "a1B2c3D4e5" * 3 + "f6G7h8"
HFTOK = "hf_" + "qQwWeErRtT" * 4
ANTHROPIC = "sk-ant-" + "api03-" + "z9Y8x7W6v5" * 3


@pytest.mark.parametrize("secret,rule", [
    (RUNPOD, "RunPod API key"),
    (DOCKER, "Docker Hub token"),
    (AWS, "AWS access key id"),
    (GITHUB, "GitHub token"),
    (HFTOK, "Hugging Face token"),
    (ANTHROPIC, "Anthropic API key"),
])
def test_catches_prefixed_credentials(secret, rule):
    found = guard.scan_text(f'KEY = "{secret}"')
    assert [f[2] for f in found] == [rule], f"{rule} not caught"


def test_catches_private_key_block():
    assert guard.scan_text("-----BEGIN OPENSSH PRIVATE KEY-----")  # pragma: allowlist secret


def test_catches_unprefixed_credential_assignment():
    found = guard.scan_text('api_key = "8f3d92ab77c14e0fbb31d5a6e9c07421"')  # pragma: allowlist secret
    assert found and found[0][2] == "credential-shaped assignment"


def test_never_echoes_the_secret_back():
    """Findings are printed to terminals and CI logs, which are stored."""
    for _, _, _, evidence in guard.scan_text(f'KEY = "{RUNPOD}"'):
        assert RUNPOD not in evidence
        assert evidence.startswith("rpa_") and "chars" in evidence


@pytest.mark.parametrize("line", [
    'api_key = "your-api-key-here"',
    'password = "changeme-please-really"',
    'token = os.environ.get("WANGP_TOKEN", "")',
    'api_key = "<paste your key here>"',
    'client_secret = "${VAULT_CLIENT_SECRET}"',
    'api_key = "example-key-0123456789abcdef"',
])
def test_ignores_placeholders(line):
    assert guard.scan_text(line) == [], f"false positive on: {line}"


def test_allowlist_marker_suppresses_a_line():
    assert guard.scan_text(f'KEY = "{RUNPOD}"  # {guard.ALLOW_MARKER}') == []


def test_generic_rule_skips_long_minified_lines():
    """Vendored CSS/JS carries megabyte base64 blobs that hit loose rules by
    chance. The prefixed rules still apply; only the generic one is length-capped."""
    blob = "a1B2c3D4e5" * 200
    assert guard.scan_text(f'password = "{blob}"') == []
    assert len(f'password = "{blob}"') > guard.GENERIC_MAX_LINE


# ---------------------------------------------------------------------------
# Diff parsing: only ADDED lines, and never vendored trees
# ---------------------------------------------------------------------------

def _diff(path: str, added: str) -> str:
    return (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
            f"@@ -0,0 +1 @@\n+{added}\n")


def test_scan_diff_flags_added_line():
    findings = guard._scan_diff(_diff("runpod_worker/x.py", f'K = "{RUNPOD}"'))
    assert findings and findings[0][0] == "runpod_worker/x.py"


def test_scan_diff_ignores_removed_lines():
    """Deleting a leaked key must not block the commit that removes it."""
    body = (f"diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
            f"@@ -1 +0,0 @@\n-K = \"{RUNPOD}\"\n")
    assert guard._scan_diff(body) == []


@pytest.mark.parametrize("path", ["shared/gradio/bundle.css", "plugins/p/app.js"])
def test_scan_diff_skips_vendored_trees(path):
    assert guard._scan_diff(_diff(path, f'K = "{RUNPOD}"')) == []


def test_scan_diff_reports_correct_line_number():
    body = ("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
            f"@@ -0,0 +12,2 @@\n+harmless = 1\n+K = \"{RUNPOD}\"\n")
    findings = guard._scan_diff(body)
    assert findings and findings[0][1] == 13


# ---------------------------------------------------------------------------
# The guard must hold on this repo as it stands
# ---------------------------------------------------------------------------

def test_worker_tree_is_clean():
    """Every tracked file the worker owns, scanned as text."""
    findings = []
    for path in (REPO / "runpod_worker").rglob("*"):
        if not path.is_file() or path.suffix in (".whl", ".mp4", ".png", ".pyc"):
            continue
        findings += guard.scan_text(path.read_text(errors="replace"),
                                    str(path.relative_to(REPO)))
    assert findings == [], f"credential-shaped content in the worker tree: {findings}"


def test_installer_and_guard_are_executable():
    for name in ("secret_guard.py", "install_hooks.py"):
        path = REPO / "runpod_worker" / "scripts" / name
        assert path.is_file(), f"{name} missing"
        assert path.read_text().startswith("#!"), f"{name} needs a shebang"
