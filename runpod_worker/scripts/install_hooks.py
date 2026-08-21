#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install the pre-commit secret guard into this clone.

    python3 runpod_worker/scripts/install_hooks.py            # install
    python3 runpod_worker/scripts/install_hooks.py --check    # report, change nothing

Git hooks live in .git/hooks, which is NOT version controlled, so a hook cannot
ship in a commit -- every clone has to opt in. That is also why CI runs the same
scanner (worker-ci.yml): the hook is the fast local guard, CI is the one that
actually holds for everyone.
"""

from __future__ import annotations

import argparse
import stat
import subprocess
import sys
from pathlib import Path

HOOK = """#!/bin/sh
# Installed by runpod_worker/scripts/install_hooks.py -- edits here are lost on
# reinstall. Bypass once with: git commit --no-verify
exec python3 "$(git rev-parse --show-toplevel)/runpod_worker/scripts/secret_guard.py" --staged
"""


def _git_dir() -> Path:
    out = subprocess.run(["git", "rev-parse", "--git-common-dir"],
                         capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise SystemExit("install_hooks: not inside a git repository")
    return Path(out.stdout.strip()).resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    hook_path = _git_dir() / "hooks" / "pre-commit"
    installed = hook_path.is_file() and "secret_guard.py" in hook_path.read_text()

    if args.check:
        print(f"pre-commit secret guard: {'installed' if installed else 'NOT installed'} "
              f"({hook_path})")
        return 0 if installed else 1

    if hook_path.is_file() and not installed:
        # Someone else's hook is here; refuse rather than silently destroy it.
        print(f"install_hooks: {hook_path} already exists and is not ours.\n"
              f"Add this line to it yourself:\n\n"
              f'  python3 "$(git rev-parse --show-toplevel)/runpod_worker/scripts/'
              f'secret_guard.py" --staged || exit 1\n', file=sys.stderr)
        return 1

    hook_path.parent.mkdir(parents=True, exist_ok=True)
    hook_path.write_text(HOOK)
    hook_path.chmod(hook_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"installed pre-commit secret guard -> {hook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
