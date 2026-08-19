"""Drift guard for models/minimax_h3/usp.py — text-only, no torch.

``usp._usp_forward`` is a rank-aware COPY of ``MiniMaxH3Model.forward``
(transformer.py). If upstream edits that forward, the copy silently diverges;
this test pins the mirrored region's hash so the divergence fails CI instead.
On failure: re-diff ``_usp_forward`` against the new upstream forward, port
the change, and update the hash below.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: sha256 of MiniMaxH3Model.forward's source at the revision usp.py mirrors.
MIRRORED_FORWARD_SHA256 = "a45ba9cf731416aa827b77d68f5d05223a653f97da1165dc13a3db25b3f40aeb"


def _forward_source() -> str:
    src = (REPO / "models" / "minimax_h3" / "transformer.py").read_text()
    match = re.search(r"    def forward\(self, video_x.*?(?=\n\n__all__|\nclass |\Z)", src, re.S)
    assert match, "MiniMaxH3Model.forward not found — the anchor regex needs updating"
    return match.group(0)


def test_upstream_forward_unchanged_since_usp_mirror():
    digest = hashlib.sha256(_forward_source().encode()).hexdigest()
    assert digest == MIRRORED_FORWARD_SHA256, (
        "MiniMaxH3Model.forward changed upstream since models/minimax_h3/usp.py mirrored it. "
        "Re-diff usp._usp_forward against transformer.py's forward, port the change, and "
        f"update MIRRORED_FORWARD_SHA256 to {digest}."
    )


def test_usp_module_stays_torch_lazy_for_this_suite():
    # The CPU suite must stay importable without torch; usp.py needs torch at
    # module scope, so nothing in runpod_worker may import it at module scope.
    for py in (REPO / "runpod_worker").rglob("*.py"):
        if py.name in ("usp_bench.py", "test_usp_gloo.py"):
            continue  # torch-dependent by design, never imported by the suite
        text = py.read_text()
        assert "minimax_h3.usp" not in text.replace("models/minimax_h3/usp", ""), (
            f"{py} references the torch-requiring usp module; keep the worker package torch-free"
        )
