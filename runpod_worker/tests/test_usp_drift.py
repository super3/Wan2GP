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

#: sha256 of MiniMaxH3Model.forward's source at the revision usp.py mirrors
#: (upstream dee6635).
MIRRORED_FORWARD_SHA256 = "c7ff3a3d9124eaa78dc79f739b8ad5720d7d8c41986527de42be59ec65bfd429"


def _forward_source() -> str:
    src = (REPO / "models" / "minimax_h3" / "transformer.py").read_text()
    # Search from the class, not the file start: upstream added
    # InputPreparation.forward(self, video_x, ...) above it, which a file-wide
    # search matches first -- the guard would then pin the wrong function.
    start = src.find("\nclass MiniMaxH3Model(")
    assert start >= 0, "class MiniMaxH3Model not found — the anchor needs updating"
    match = re.compile(r"    def forward\(self, video_x.*?(?=\n\n__all__|\nclass |\Z)", re.S).search(src, start)
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
    needle = "minimax_h3" + ".usp"  # split so this file does not match itself
    for py in (REPO / "runpod_worker").rglob("*.py"):
        if py.name in ("usp_bench.py", "test_usp_gloo.py", "test_usp_drift.py"):
            continue  # torch-dependent (or this guard itself), never imported by the suite
        text = py.read_text()
        assert needle not in text, (
            f"{py} references the torch-requiring usp module; keep the worker package torch-free"
        )


#: sha256 of AutoencoderKLMiniMaxH3._decode at the revision last_frame.py reasons about.
#: Re-verified at upstream dee6635 (runpod_worker/scripts/test_last_frame_cpu.py:
#: last frame bit-identical for 107-362 frames): the chunk loop is unchanged;
#: finalized frames now go to a CPU buffer via _prepare_decoded_chunk(uint8_rounding)
#: and a single latent frame takes a separate still-image path.
MIRRORED_DECODE_SHA256 = "57c6f669ed43592cd74d3a15e74f9aa7c1ee488f9dd25d9a75f38eb2d5760467"


def _decode_source() -> str:
    src = (REPO / "models" / "minimax_h3" / "components" / "video_autoencoder.py").read_text()
    match = re.search(r"    def _decode\(self, z.*?(?=\n    def )", src, re.S)
    assert match, "AutoencoderKLMiniMaxH3._decode not found — the anchor regex needs updating"
    return match.group(0)


def test_decode_chunking_unchanged_since_last_frame_patch():
    """models/minimax_h3/last_frame.py is only correct because _decode loops over
    INDEPENDENT chunks and the final frames come from the last chunk's unblended
    j==1 segment. If that loop changes, the tail slice may silently return frames
    from the wrong place — which looks like a plausible image, not an error."""
    digest = hashlib.sha256(_decode_source().encode()).hexdigest()
    assert digest == MIRRORED_DECODE_SHA256, (
        "AutoencoderKLMiniMaxH3._decode changed. Re-verify that the last chunk still "
        "reads exactly the trailing tokens_chunk_size + token_overlap latent frames and "
        "that nothing earlier contributes to the final frames, then update "
        f"MIRRORED_DECODE_SHA256 to {digest}."
    )
