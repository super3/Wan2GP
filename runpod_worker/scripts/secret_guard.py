#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Refuse to commit credentials.

    python3 runpod_worker/scripts/secret_guard.py --staged     # pre-commit hook
    python3 runpod_worker/scripts/secret_guard.py --range A..B # CI, a PR's commits
    python3 runpod_worker/scripts/secret_guard.py FILE...      # ad hoc

Exit 0 clean, 1 on a finding. Pure stdlib; no torch, no network.

Why this exists: this repo has already lost files to silent tooling. ``.gitignore``
line 1 is a bare ``.*`` (it ate ``.dockerignore`` and the CI workflow), ``*.whl``
ate a build artifact, and ``*.html`` ate ``webdemo/index.html`` -- each time
``git add -A`` reported success. The same blindness runs the other way: a key
pasted into a file is committed just as quietly, and a published key stays
compromised no matter how fast the commit is reverted.

Scans only ADDED lines, so pre-existing content in a fork of an upstream tree
never blocks a commit about something else.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

#: Marker that declares a match intentional, e.g. a test fixture. Borrowed from
#: detect-secrets so the spelling is one people already know.
ALLOW_MARKER = "pragma: allowlist secret"

#: Vendored trees. Their minified CSS/JS carries megabyte-long base64 font blobs
#: that hit the loose prefix rules by chance; a real secret does not arrive by
#: way of an upstream bundle.
SKIP_PREFIXES = ("shared/gradio/", "plugins/", "assets/")

#: (name, regex). Prefixed rules are specific enough to run on any line length.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("RunPod API key", re.compile(r"\brpa_[A-Za-z0-9]{30,}")),
    ("Docker Hub token", re.compile(r"\bdckr_pat_[A-Za-z0-9_\-]{15,}")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b")),
    ("Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("OpenAI API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{34,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("private key block", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
)

#: Catches a credential with no recognizable prefix. Far looser, so it is held to
#: short lines -- a 4000-character minified bundle is not where someone pastes a
#: key, but it is exactly where random base64 mimics one.
GENERIC = re.compile(
    r"""(?ix)
    \b (?:api[_-]?key|auth[_-]?token|access[_-]?token|secret[_-]?key
         |password|passwd|client[_-]?secret|bearer)
    \b \s* [:=] \s* ["']([A-Za-z0-9_\-]{20,})["']
    """
)
GENERIC_MAX_LINE = 500

#: Placeholders people legitimately commit. Substring match, case-insensitive.
PLACEHOLDERS = ("example", "placeholder", "changeme", "your-", "your_", "xxxx",
                "dummy", "redacted", "<", "${", "os.environ", "getenv", "fake")


def _is_placeholder(value: str) -> bool:
    low = value.lower()
    return any(token in low for token in PLACEHOLDERS)


def scan_text(text: str, origin: str = "") -> list[tuple[str, int, str, str]]:
    """Return ``(origin, line_no, rule, evidence)`` for every match."""
    findings: list[tuple[str, int, str, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if ALLOW_MARKER in line:
            continue
        for name, pattern in PATTERNS:
            match = pattern.search(line)
            if match and not _is_placeholder(match.group(0)):
                findings.append((origin, number, name, _redact(match.group(0))))
        if len(line) <= GENERIC_MAX_LINE:
            match = GENERIC.search(line)
            if match and not _is_placeholder(match.group(1)):
                findings.append((origin, number, "credential-shaped assignment",
                                 _redact(match.group(1))))
    return findings


def _redact(value: str) -> str:
    """Never print the secret back -- terminal scrollback and CI logs are stored."""
    return f"{value[:6]}…{len(value)} chars" if len(value) > 10 else "…"


def _git(*args: str) -> str:
    return subprocess.run(("git", *args), capture_output=True, text=True,
                          check=False).stdout


def _added_lines(diff: str) -> list[tuple[str, int, str]]:
    """``(path, line_no, text)`` for added lines only, skipping vendored trees."""
    out: list[tuple[str, int, str]] = []
    path, line_no = "", 0
    for raw in diff.splitlines():
        if raw.startswith("+++ b/"):
            path, line_no = raw[6:], 0
        elif raw.startswith("@@"):
            match = re.search(r"\+(\d+)", raw)
            line_no = int(match.group(1)) - 1 if match else 0
        elif raw.startswith("+") and not raw.startswith("+++"):
            line_no += 1
            if not path.startswith(SKIP_PREFIXES):
                out.append((path, line_no, raw[1:]))
        elif not raw.startswith("-"):
            line_no += 1
    return out


def _scan_diff(diff: str) -> list[tuple[str, int, str, str]]:
    findings: list[tuple[str, int, str, str]] = []
    for path, number, text in _added_lines(diff):
        for _, _, rule, evidence in scan_text(text):
            findings.append((path, number, rule, evidence))
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--staged", action="store_true", help="scan the staged diff")
    ap.add_argument("--range", help="scan a commit range, e.g. origin/main..HEAD")
    ap.add_argument("paths", nargs="*", type=Path)
    args = ap.parse_args()

    if args.staged:
        findings = _scan_diff(_git("diff", "--cached", "--unified=0"))
    elif args.range:
        findings = _scan_diff(_git("diff", "--unified=0", args.range))
    elif args.paths:
        findings = []
        for path in args.paths:
            try:
                findings += scan_text(path.read_text(errors="replace"), str(path))
            except OSError as exc:
                print(f"secret-guard: cannot read {path}: {exc}", file=sys.stderr)
    else:
        ap.error("pass --staged, --range, or one or more paths")

    if not findings:
        return 0

    print("\nsecret-guard: refusing to continue -- credential-shaped content found\n",
          file=sys.stderr)
    for origin, number, rule, evidence in findings:
        print(f"  {origin}:{number}: {rule} ({evidence})", file=sys.stderr)
    print(
        "\nA committed key is compromised the moment it is pushed; reverting does not\n"
        "un-publish it. Move the value to an environment variable, then rotate it.\n"
        f"If this is a fixture, append a '{ALLOW_MARKER}' comment to the line.\n"
        "To bypass deliberately: git commit --no-verify\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
