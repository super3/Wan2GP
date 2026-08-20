# SPDX-License-Identifier: Apache-2.0
"""Decode only the final chunk of a MiniMax H3 video, for last-frame output.

Why this is exact rather than an approximation
----------------------------------------------
``AutoencoderKLMiniMaxH3._decode`` (components/video_autoencoder.py:895) is a
loop over INDEPENDENT temporal chunks::

    for i in range(num_chunks):
        clip = self._decode_clip(z[:, :, i * chunk : i * chunk + chunk + overlap])

The ViT decoder is non-causal *within* a chunk -- every latent voxel is a token
under full self-attention -- but nothing couples one chunk to the next except a
linear cross-fade over ``frame_overlap`` pixel frames.

Two properties make the tail slice give bit-identical output:

1. The geometry is fixed: ``17n + 5`` pixel frames map to ``5n + 2`` latent
   frames, so the latent length is ALWAYS ``2 (mod 5)``. Hence ``pad_tokens`` is
   always 0 and the final chunk always reads exactly the last
   ``tokens_chunk_size + token_overlap`` == 7 latent frames.
2. The frames that end the video come from that chunk's ``j == 1`` segment,
   which ``_decode`` copies in AFTER the loop without blending. No earlier chunk
   contributes to them.

So ``decode(z[..., -7:])`` reproduces the true final frames while running one
chunk instead of ``num_chunks`` -- 1/7 of the decode work for a 124-frame clip,
1/21 for a 362-frame one.

What this does NOT save: denoising. The DiT runs full 3D attention over the
whole sequence, so the final frame's latents depend on every other frame. The
floor for a 124-frame clip stays at the ~13.4 s denoise.

Usage (before generation)::

    from models.minimax_h3 import last_frame
    last_frame.activate()

``deactivate()`` restores the original decode.
"""

from __future__ import annotations

import math

_ORIGINAL_DECODE = None


def _tail_tokens(vae) -> int:
    """Latent frames the final chunk consumes, read from the model's own config."""
    temporal_ratio = int(vae.temporal_compression_ratio)
    clip_length = int(vae.config.clip_length)
    token_drop = int(vae.config.token_drop)
    tokens_chunk_size = math.ceil(clip_length / temporal_ratio)
    token_overlap = (-token_drop) % tokens_chunk_size
    return tokens_chunk_size + token_overlap


def activate() -> None:
    """Patch MiniMaxH3VideoVAE.decode to decode only the final chunk."""
    global _ORIGINAL_DECODE
    from .video_vae import MiniMaxH3VideoVAE  # noqa: PLC0415

    if _ORIGINAL_DECODE is not None:
        return
    _ORIGINAL_DECODE = MiniMaxH3VideoVAE.decode

    def decode(self, latents):
        need = _tail_tokens(self)
        # Guard the assumption rather than trusting it: a latent length that is
        # not 2 (mod 5) means the geometry changed upstream and the tail slice
        # would no longer align with the final chunk. Decode in full instead of
        # silently returning frames from the wrong place.
        available = latents.shape[2]
        tokens_chunk_size = math.ceil(int(self.config.clip_length)
                                      / int(self.temporal_compression_ratio))
        aligned = (available + int(self.config.token_drop)) % tokens_chunk_size == 0
        if available <= need or not aligned:
            return _ORIGINAL_DECODE(self, latents)
        return _ORIGINAL_DECODE(self, latents[:, :, -need:])

    MiniMaxH3VideoVAE.decode = decode


def deactivate() -> None:
    global _ORIGINAL_DECODE
    if _ORIGINAL_DECODE is None:
        return
    from .video_vae import MiniMaxH3VideoVAE  # noqa: PLC0415

    MiniMaxH3VideoVAE.decode = _ORIGINAL_DECODE
    _ORIGINAL_DECODE = None


def is_active() -> bool:
    return _ORIGINAL_DECODE is not None


__all__ = ["activate", "deactivate", "is_active"]
