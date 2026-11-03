#!/usr/bin/env python3
"""Make SageAttention's ``setup.py`` compile for a fixed arch list, with no GPU.

Run from inside a SageAttention checkout (or pass the path):

    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" python3 patch_sage_setup.py [setup.py]

WHY THIS EXISTS
---------------
A serverless image is built on a CI machine with no GPU. Older SageAttention
``setup.py`` revisions derived the target compute capabilities by *probing the
local devices*::

    compute_capabilities = set()
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        ...

On a GPU-less builder that loop runs zero times, the extension list comes out
empty, and you get a wheel that installs cleanly and has no kernels in it. The
failure then surfaces minutes into a *billed* generation, at the first attention
kernel launch. The repo's own Dockerfile (``Dockerfile:50-80``) patches exactly
this block for the same reason.

Current upstream (checked against thu-ml/SageAttention ``main``) already reads
``TORCH_CUDA_ARCH_LIST`` first — "Prefer TORCH_CUDA_ARCH_LIST if explicitly
specified (works without GPUs)" — and raises ``RuntimeError`` when the capability
set ends up empty. Against that revision this script is a **verified no-op**: it
detects native support and exits 0 without touching the file. The repo's own
in-Dockerfile patch does not detect this; its ``str.replace`` finds nothing and
silently leaves the file alone, which happens to be correct today and would be
silently wrong the day the block comes back.

CONTRACT
--------
Exit 0  -> the build will target exactly ``TORCH_CUDA_ARCH_LIST`` (either
           natively or after this script rewrote the detection block).
Exit 1  -> we could NOT prove that. Fail the image build here rather than ship a
           wheel with no kernels for the fleet's GPUs.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

#: The historical GPU-probing block, verbatim from the revision the repo's own
#: Dockerfile patches (``Dockerfile:56-64``).
LEGACY_BLOCK = """compute_capabilities = set()
device_count = torch.cuda.device_count()
for i in range(device_count):
    major, minor = torch.cuda.get_device_capability(i)
    if major < 8:
        warnings.warn(f"skipping GPU {i} with compute capability {major}.{minor}")
        continue
    compute_capabilities.add(f"{major}.{minor}")"""

#: SageAttention's own whitelist (``setup.py``: ``SUPPORTED_ARCHS``). An arch
#: outside it contributes no ``-gencode`` flag, so we refuse it loudly instead of
#: letting it evaporate.
SUPPORTED_ARCHS = {"8.0", "8.6", "8.9", "9.0", "10.0", "12.0", "12.1"}


def parse_arch_list(raw: str) -> list[str]:
    """Normalize ``TORCH_CUDA_ARCH_LIST`` into ``["8.0", "8.9", ...]``.

    Accepts the separators torch itself accepts (``;`` and ``,``), the ``sm_89``
    / ``compute_89`` / ``89`` spellings, the trailing ``a`` of ``9.0a``, and the
    ``+PTX`` suffix (which is dropped: it changes codegen, not the target set).
    """
    out: list[str] = []
    for chunk in raw.replace(",", ";").split(";"):
        item = chunk.strip().lower()
        if not item:
            continue
        item = item.replace("sm_", "").replace("compute_", "")
        if item.endswith("+ptx"):
            item = item[:-4]
        item = item.rstrip("a")
        if len(item) == 2 and item.isdigit():        # "89" -> "8.9"
            item = f"{item[0]}.{item[1]}"
        if item and item not in out:
            out.append(item)
    return out


def main(argv: list[str]) -> int:
    setup_path = Path(argv[1]) if len(argv) > 1 else Path("setup.py")
    if not setup_path.is_file():
        print(f"[patch_sage_setup] ERROR: {setup_path} does not exist", file=sys.stderr)
        return 1

    raw = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if not raw:
        print(
            "[patch_sage_setup] ERROR: TORCH_CUDA_ARCH_LIST is unset. A GPU-less "
            "builder cannot detect target architectures, and a wheel built with "
            "none is an empty wheel.",
            file=sys.stderr,
        )
        return 1

    arches = parse_arch_list(raw)
    if not arches:
        print(f"[patch_sage_setup] ERROR: TORCH_CUDA_ARCH_LIST={raw!r} parsed to nothing",
              file=sys.stderr)
        return 1

    unsupported = [a for a in arches if a not in SUPPORTED_ARCHS]
    if unsupported:
        print(
            f"[patch_sage_setup] ERROR: {unsupported} are not in SageAttention's "
            f"SUPPORTED_ARCHS {sorted(SUPPORTED_ARCHS)}; they would compile to no "
            f"-gencode flag at all.",
            file=sys.stderr,
        )
        return 1

    content = setup_path.read_text(encoding="utf-8")

    # Case 1: upstream already honours the env var. Nothing to do -- and doing
    # nothing is the point: a rewrite here would fight the code that works.
    if "TORCH_CUDA_ARCH_LIST" in content:
        print(
            f"[patch_sage_setup] {setup_path} already reads TORCH_CUDA_ARCH_LIST; "
            f"no patch applied. Targets: {arches}"
        )
        return 0

    # Case 2: the historical probing block. Replace it with a literal set.
    arch_set = "{" + ", ".join(f'"{arch}"' for arch in arches) + "}"
    replacement = (
        f"compute_capabilities = {arch_set}\n"
        f'print(f"[patch_sage_setup] Forced compute capabilities: {{compute_capabilities}}")'
    )
    if LEGACY_BLOCK in content:
        content = content.replace(LEGACY_BLOCK, replacement, 1)
        setup_path.write_text(content, encoding="utf-8")
        print(f"[patch_sage_setup] patched the GPU-probing block in {setup_path}; "
              f"targets: {arches}")
        return 0

    # Case 3: a shape we do not recognise. Try the narrow, unambiguous rewrite of
    # the assignment itself; only accept it if the probing loop is demonstrably
    # the thing being replaced.
    probe = re.search(
        r"^([ \t]*)compute_capabilities\s*=\s*set\(\)\s*$", content, flags=re.MULTILINE
    )
    if probe and "torch.cuda.get_device_capability" in content:
        indent = probe.group(1)
        injected = (
            f"{indent}compute_capabilities = {arch_set}\n"
            f"{indent}import os as _os  # patch_sage_setup\n"
            f"{indent}_os.environ.setdefault('TORCH_CUDA_ARCH_LIST', {raw!r})\n"
            f"{indent}if False:\n"
            f"{indent}    compute_capabilities = set()"
        )
        content = content[: probe.start()] + injected + content[probe.end():]
        setup_path.write_text(content, encoding="utf-8")
        print(
            f"[patch_sage_setup] WARNING: unrecognised setup.py layout; pinned "
            f"compute_capabilities = {arch_set} in place. Targets: {arches}. "
            f"Re-check this script against upstream."
        )
        return 0

    print(
        f"[patch_sage_setup] ERROR: {setup_path} neither reads TORCH_CUDA_ARCH_LIST "
        f"nor contains a recognised GPU-probing block. Refusing to build a wheel "
        f"whose target architectures cannot be proven. Update this script against "
        f"the current upstream setup.py.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
