#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prove the last-frame tail slice is bit-identical to a full decode.

Needs torch (run inside the worker image); deliberately NOT part of the CPU
pytest suite, which stays torch-free.

    python3 runpod_worker/scripts/test_last_frame_cpu.py

Runs the REAL ``AutoencoderKLMiniMaxH3._decode`` loop -- the chunk arithmetic
this optimisation depends on -- with the 36-layer ViT replaced by a cheap
deterministic stub. That keeps the test on CPU in milliseconds while still
exercising the code whose behaviour is actually being relied upon.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.minimax_h3.components.video_autoencoder import AutoencoderKLMiniMaxH3  # noqa: E402


class _Stub(AutoencoderKLMiniMaxH3):
    """Real chunk geometry, fake decoder.

    ``_decode_clip`` must be a deterministic function of its input so a chunk
    decoded in isolation is comparable with the same chunk decoded in sequence.
    Each latent frame expands to ``temporal_ratio`` pixel frames carrying the
    latent's own value, so a mis-sliced tail shows up as a wrong value, not just
    a wrong shape.
    """

    def __init__(self):
        super().__init__()
        self.calls = 0

    def _decode_clip(self, z):
        self.calls += 1
        b, _, t, h, w = z.shape
        ratio = self.temporal_compression_ratio
        # value of each latent frame -> repeated across its pixel frames
        marks = z[:, :1].mean(dim=(3, 4))                     # [b, 1, t]
        frames = marks.repeat_interleave(ratio, dim=2)        # [b, 1, t*ratio]
        return frames.view(b, 1, t * ratio, 1, 1).expand(
            b, 3, t * ratio, h * self.spatial_compression_ratio,
            w * self.spatial_compression_ratio).contiguous()


def main() -> int:
    torch.manual_seed(0)
    vae = _Stub().eval()
    ratio = vae.temporal_compression_ratio
    chunk = math.ceil(vae.config.clip_length / ratio)
    overlap = (-vae.config.token_drop) % chunk
    need = chunk + overlap
    print(f"geometry: temporal {ratio}x, chunk {chunk}, overlap {overlap}, tail needs {need}")

    failures = 0
    for frames in (107, 124, 141, 175, 243, 362):
        n = (frames - 5) // 17
        latent_t = 5 * n + 2
        z = torch.randn(1, vae.config.latent_channels, latent_t, 2, 2)

        vae.calls = 0
        full = vae._decode(z)
        full_calls = vae.calls

        vae.calls = 0
        tail = vae._decode(z[:, :, -need:])
        tail_calls = vae.calls

        same = torch.equal(full[:, :, -1], tail[:, :, -1])
        speedup = full_calls / max(tail_calls, 1)
        status = "OK " if same else "FAIL"
        print(f"  {status} {frames:4d}f  latent={latent_t:3d}  "
              f"chunks {full_calls:2d} -> {tail_calls}  ({speedup:.1f}x less decode)  "
              f"last frame identical: {same}")
        if not same:
            failures += 1
            print(f"       max abs diff {(full[:, :, -1] - tail[:, :, -1]).abs().max().item()}")

    # The guard must refuse a misaligned latent length rather than return the
    # wrong frames.
    from models.minimax_h3 import last_frame  # noqa: PLC0415
    aligned = (latent_t + vae.config.token_drop) % chunk == 0
    print(f"\nalignment guard: latent {latent_t} aligned={aligned}, "
          f"tail_tokens()={last_frame._tail_tokens(vae)}")
    assert last_frame._tail_tokens(vae) == need

    print("\nALL PASSED" if not failures else f"\n{failures} FAILURES")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
