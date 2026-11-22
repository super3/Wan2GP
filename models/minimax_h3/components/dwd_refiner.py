# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
# Follows LakonLab's SolH3DWDHead (Hansheng Chen, https://github.com/Lakonik/LakonLab v0.3.0, Apache License 2.0), run by WanGP's
# Sol-H3-VAE decoder (sol_decoder.py).
"""pi-H3's DWD pixel refiner: the last layers of Sol-H3-VAE's decoder, whose tokens are expanded from the transformer's DWD features
(models/minimax_h3/pi_h3_dwd.py) instead of embedded from latents. The windows, global tokens, rotary ids and the streaming of the
residual by native blocks are the Sol decoder's; a layer's window parity is its index in the full 24-layer decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..pi_h3_dwd import SOL_LAYERS, expand_tokens, unpatchify_pixels
from .sol_decoder import BLOCK, EMIT_BYTES, EMIT_TOKEN_BYTES, FRAMES_TO_TRIM, MiniMaxH3SolDecoder3d, _State


class MiniMaxH3DWDRefiner(MiniMaxH3SolDecoder3d):
    def __init__(self, in_channels, num_layers, expansion):
        super().__init__(num_layers=num_layers)
        del self.x_embedder  # the tokens come from feature_expand
        self.expansion = tuple(expansion)
        self.layer_offset = SOL_LAYERS - num_layers
        self.feature_expand = nn.Linear(in_channels, self.dim * self.expansion[0] * self.expansion[1])
        # the first latent of a native block gives one frame (the Sol decoder projects four and keeps the last)
        self.first_frame_proj_out = nn.Linear(self.dim, self.out_channels * (self.patch_size_t - FRAMES_TO_TRIM) * self.patch_size ** 2)
        self.interrupt = None

    @property
    def _interrupt(self):  # read by the Sol layers between steps
        return self.interrupt is not None and bool(self.interrupt())

    def _emit(self, x, lo, hi, state):
        """The final rows x of latent frames lo..hi, as the Sol decoder emits them, the first latent of each block by its own head."""
        p, H, W, HW = self.patch_size, state.H, state.W, state.HW
        band = max(1, EMIT_BYTES // (W * EMIT_TOKEN_BYTES))
        for i in range(hi - lo):
            head, frames = ((self.first_frame_proj_out, self.patch_size_t - FRAMES_TO_TRIM) if (lo + i) % BLOCK == 0 else
                            (self.proj_out, self.patch_size_t))
            for top in range(0, H, band):
                bottom = min(top + band, H)
                hidden = F.layer_norm(x[i * HW + top * W:i * HW + bottom * W].float(), (self.dim,), self.norm_out.weight.float(),
                                      self.norm_out.bias.float(), self.norm_out.eps)
                pixels = unpatchify_pixels(head(hidden.to(head.weight.dtype)), bottom - top, W, self.out_channels, frames, p)
                del hidden
                state.emit(pixels, top * p, bottom == H)
                del pixels

    def forward(self, features, token_grid, emit):
        """features: (t * h * w, in_channels) on a (t, h, w) token grid, its first latent starting a native block. emit receives the
        frames as the Sol decoder's forward gives them (pixels in Sol's ImageNet-normalized RGB)."""
        t, h, w = token_grid
        e_h, e_w = self.expansion
        state = _State(self, torch.empty((1, 0, t, h * e_h, w * e_w), dtype=features.dtype, device=features.device), emit)
        if self.layer_offset % 2:  # parities[0] is the unshifted windows of the full decoder's even layers
            state.parities.reverse()
        expand_tokens(features, token_grid, lambda rows: self.feature_expand(rows.to(self.feature_expand.weight.dtype)).to(state.dtype),
                      self.expansion, state.x[:state.N])
        state.x[state.N:] = torch.cat([self.register_tokens[0].to(state.dtype), state.x.new_zeros(1, self.dim)]).repeat(state.nb, 1)
        for layer, block in enumerate(self.transformer_blocks):
            block(state, layer)
