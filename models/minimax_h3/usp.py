# SPDX-License-Identifier: Apache-2.0
"""Ulysses sequence parallelism (USP) for MiniMax H3 — multi-GPU denoise.

Prototype, deliberately additive: nothing here runs unless :func:`activate`
is called (before the model loads). Pure ``torch.distributed`` — no xfuser,
no ray. The method mirrors komikndr/raylight's ComfyUI port of this model:

- ``MiniMaxH3Model.forward`` shards the packed text+video+audio token
  sequence contiguously across ranks right after it is assembled, and
  recomputes the AdaLN modulation segments for the shard (they are
  ``(start, stop, row)`` runs over global token indices);
- inside every DiT block's attention, an all-to-all trades the sequence
  axis for the head axis — each rank then holds the FULL sequence for
  ``heads / world_size`` heads — plain ``pay_attention`` runs unchanged,
  and a second all-to-all trades back (Ulysses attention). 56 heads and
  contiguous uneven shards mean no padding is ever needed:
  ``torch.distributed.all_to_all`` takes ragged tensor lists;
- after the block loop the shards are all-gathered, so the final layer and
  everything downstream is identical on every rank.

Every rank runs the ENTIRE pipeline (text encode, VAE, sampling loop)
redundantly and deterministically — initial latents come from a CPU
generator seeded by the task seed, so ranks stay in lockstep without a
command channel. Rank 0's outputs are the ones to keep.

Not supported under USP (assert loudly rather than silently corrupt):
- Sol sparse attention (its sink-token offsets are global-sequence-based);
  run ``--attention sdpa`` or ``sage2``.
- ``skip_steps_cache`` variants (Spectrum / FirstBlockCache): their
  full-sequence signatures diverge per shard.

Launch each rank as its own process (``torchrun --nproc-per-node=N`` or
manual env: RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT / LOCAL_RANK),
call ``activate()`` before the model loads, then generate as usual.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

__all__ = ["activate", "is_active", "rank", "world_size"]


class _State:
    enabled = False
    rank = 0
    world = 1
    bounds: list[int] | None = None  # per-forward shard bounds, len world+1
    checked_forward = False


_S = _State()


def is_active() -> bool:
    return _S.enabled and _S.world > 1


def rank() -> int:
    return _S.rank


def world_size() -> int:
    return _S.world


# ---------------------------------------------------------------------------
# Collectives. All operate on the shapes this model actually uses:
# hidden [seq, C]; q/k/v [1, seq, H, D].
# ---------------------------------------------------------------------------


def _shard_bounds(seq_len: int, world: int) -> list[int]:
    """Contiguous near-equal split; first ``seq_len % world`` shards get +1."""
    base, extra = divmod(seq_len, world)
    bounds = [0]
    for r in range(world):
        bounds.append(bounds[-1] + base + (1 if r < extra else 0))
    return bounds


def _shard_segments(segments, lo: int, hi: int):
    """Intersect the global (start, stop, row) runs with [lo, hi), shift by -lo."""
    out = []
    for start, stop, row in segments:
        s, e = max(start, lo), min(stop, hi)
        if s < e:
            out.append((s - lo, e - lo, row))
    return out


def _seq_to_heads(tensor: torch.Tensor, bounds: list[int]) -> torch.Tensor:
    """[1, shard, H, D] -> [1, S, H/world, D] (full sequence, head subset).

    ``all_to_all_single`` with explicit split sizes (alltoallv) — the list form
    of ``all_to_all`` requires uniform shapes on gloo, and shards are uneven."""
    world = _S.world
    t = tensor.squeeze(0)
    shard, heads, dim = t.shape
    hpr = heads // world
    # Destination-major send buffer: head-group g goes to rank g.
    send = t.view(shard, world, hpr, dim).permute(1, 0, 2, 3).contiguous().view(-1)
    out_splits = [(bounds[j + 1] - bounds[j]) * hpr * dim for j in range(world)]
    in_splits = [shard * hpr * dim] * world
    recv = torch.empty(sum(out_splits), dtype=t.dtype, device=t.device)
    dist.all_to_all_single(recv, send, output_split_sizes=out_splits, input_split_sizes=in_splits)
    pieces, offset = [], 0
    for j in range(world):
        rows = bounds[j + 1] - bounds[j]
        pieces.append(recv[offset:offset + rows * hpr * dim].view(rows, hpr, dim))
        offset += rows * hpr * dim
    return torch.cat(pieces, dim=0).unsqueeze(0)


def _heads_to_seq(tensor: torch.Tensor, bounds: list[int]) -> torch.Tensor:
    """[1, S, H/world, D] -> [1, shard, H, D] (own shard, all heads)."""
    world, r = _S.world, _S.rank
    t = tensor.squeeze(0)
    hpr, dim = t.shape[1], t.shape[2]
    shard = bounds[r + 1] - bounds[r]
    # Sequence-major already means rank j's rows are one contiguous slice.
    send = t.contiguous().view(-1)
    in_splits = [(bounds[j + 1] - bounds[j]) * hpr * dim for j in range(world)]
    out_splits = [shard * hpr * dim] * world
    recv = torch.empty(sum(out_splits), dtype=t.dtype, device=t.device)
    dist.all_to_all_single(recv, send, output_split_sizes=out_splits, input_split_sizes=in_splits)
    chunk = shard * hpr * dim
    groups = [recv[j * chunk:(j + 1) * chunk].view(shard, hpr, dim) for j in range(world)]
    return torch.cat(groups, dim=1).unsqueeze(0)


def _gather_seq(hidden: torch.Tensor, bounds: list[int]) -> torch.Tensor:
    """[shard, C] -> [S, C]. Shards are uneven (by at most one row), and
    ``all_gather`` needs uniform shapes — pad to the max shard, then trim."""
    world = _S.world
    sizes = [bounds[j + 1] - bounds[j] for j in range(world)]
    pad_to = max(sizes)
    local = hidden
    if local.shape[0] < pad_to:
        local = torch.cat((local, torch.zeros(pad_to - local.shape[0], hidden.shape[1],
                                              dtype=hidden.dtype, device=hidden.device)))
    recv = [torch.empty(pad_to, hidden.shape[1], dtype=hidden.dtype, device=hidden.device)
            for _ in range(world)]
    dist.all_gather(recv, local.contiguous())
    return torch.cat([recv[j][:sizes[j]] for j in range(world)], dim=0)


def _assert_ranks_agree(hidden: torch.Tensor) -> None:
    """One scalar collective proving every rank assembled the same sequence.

    Ranks compute encodes redundantly; if a nondeterministic kernel ever broke
    lockstep, the collectives would silently produce garbage. Checked once per
    process (the first forward), not per step."""
    if _S.checked_forward:
        return
    _S.checked_forward = True
    local = hidden.float().sum()
    checks = [torch.empty_like(local) for _ in range(_S.world)]
    dist.all_gather(checks, local)
    reference = checks[0].item()
    for r, value in enumerate(checks[1:], start=1):
        drift = abs(value.item() - reference) / (abs(reference) + 1e-6)
        if drift > 1e-3:
            raise RuntimeError(
                f"USP ranks diverged before the first block: rank 0 checksum {reference}, "
                f"rank {r} {value.item()} (relative drift {drift:.2e}). Encodes are not "
                f"deterministic across ranks on this host; USP needs identical inputs.")


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------


def _usp_attention_call(self, qkv_list, use_sol):
    """Class-level wrapper for MiniMaxH3SolAttention.__call__ (every DiT block
    routes through the model's singleton instance)."""
    from shared.attention import pay_attention

    if not is_active() or _S.bounds is None:
        return _ORIG_SOL_CALL(self, qkv_list, use_sol)
    if use_sol:
        raise RuntimeError("Sol sparse attention is global-sequence-based and unsupported "
                           "under USP; run --attention sdpa or sage2.")
    bounds = _S.bounds
    query, key, value = qkv_list
    qkv_list.clear()
    query = _seq_to_heads(query, bounds)
    key = _seq_to_heads(key, bounds)
    value = _seq_to_heads(value, bounds)
    output = pay_attention([query, key, value], recycle_q=True)
    del query, key, value
    return _heads_to_seq(output, bounds)


def _usp_forward(self, video_x, audio_x, sigma_video, sigma_audio, context, payload,
                 spectrum=None, first_block_cache=None):
    """Rank-aware copy of MiniMaxH3Model.forward (transformer.py:542-662).

    Kept as close to upstream as possible; the USP deltas are marked. Pinned to
    the upstream revision this branch tracks — re-diff on upstream bumps."""
    from . import transformer as T

    if not is_active():
        return _ORIG_FORWARD(self, video_x, audio_x, sigma_video, sigma_audio, context,
                             payload, spectrum=spectrum, first_block_cache=first_block_cache)
    if spectrum is not None or first_block_cache is not None:
        raise RuntimeError("skip_steps_cache (spectrum/first_block) is unsupported under USP; "
                           "disable the step cache.")

    device, dtype = video_x.device, self.dtype or next(self.blocks.parameters()).dtype
    video_dtype, audio_dtype = video_x.dtype, audio_x.dtype
    _, _, latent_t, latent_h, latent_w = video_x.shape
    audio_t = audio_x.shape[-1]
    text_tags = payload["text_token_tags"].view(-1).cpu()
    layout = self._layout(text_tags, latent_t, latent_h, latent_w, audio_t, payload)

    video_rows = T.patchify_video(video_x.to(torch.float32), self.patch_size)
    audio_rows = T.pack_audio(audio_x.to(torch.float32))
    cond_video, cond_audio = payload.get("cond_video_rows"), payload.get("cond_audio_rows")
    if cond_video is not None:
        video_rows = torch.cat((cond_video.to(device), video_rows))
    if cond_audio is not None:
        audio_rows = torch.cat((cond_audio.to(device), audio_rows))
    video_embeds = self.video_patch_proj(video_rows).to(dtype)
    del video_rows
    audio_embeds = self.audio_patch_proj(audio_rows).to(dtype)
    del audio_rows

    text_embeds = context[0]
    if text_embeds.shape[-1] != self.hidden_size:
        text_embeds = self.preprocess_text_embeds(context)[0]
    hidden = torch.empty(layout.sequence_length, self.hidden_size, dtype=dtype, device=device)
    hidden.index_copy_(0, layout.text_indices.to(device), text_embeds)
    hidden.index_copy_(0, layout.video_indices.to(device), video_embeds)
    hidden.index_copy_(0, layout.audio_indices.to(device), audio_embeds)
    del text_embeds, video_embeds, audio_embeds

    timestep, timestep_indices = T.build_row_timesteps(
        layout,
        float(1.0 - sigma_video.flatten()[0]),
        float(1.0 - sigma_audio.flatten()[0]),
        max(float(1.0 - sigma_video.flatten()[0]), T.VISUAL_COND_TIMESTEP),
        T.AUDIO_COND_TIMESTEP,
    )
    timestep, timestep_indices = timestep.to(device), timestep_indices.to(device)
    adaln_indices = timestep_indices * 3 + layout.token_tags.to(device).clamp_min(0)
    changes = torch.cat((torch.ones(1, dtype=torch.bool, device=device),
                         adaln_indices[1:] != adaln_indices[:-1],
                         torch.ones(1, dtype=torch.bool, device=device))).nonzero().flatten()
    segments = [(int(changes[index]), int(changes[index + 1]), int(adaln_indices[changes[index]]))
                for index in range(changes.numel() - 1)]
    temb = self._time_embedding(timestep)
    rope = payload.get("rope")
    if rope is None:
        positions = layout.position_ids.to(torch.float32)
        frequencies = positions.unsqueeze(-1) * self.rope.inv_freq.detach().cpu().view(1, 1, -1)
        rope = T._rope_table(torch.cat(frequencies.unbind(dim=1), dim=-1), dtype).to(device)
        payload["rope"] = rope
        del positions, frequencies
    del adaln_indices, changes
    target_video_rows = latent_t * (latent_h // self.patch_size[1]) * (latent_w // self.patch_size[2])
    target_audio_rows = audio_t * 2
    video_start = layout.sequence_length - target_video_rows
    audio_start = video_start - target_audio_rows
    # upstream 238e25f ("added sol attn 0.6.2") made tau a required positional:
    # sol_attention.py:23 begin_forward(self, layout, device, dtype, tau).
    self.sol_attention.begin_forward(layout, device, dtype, payload["attention_sparsity"])

    # ---- USP delta: shard the sequence across ranks --------------------
    _assert_ranks_agree(hidden)
    bounds = _shard_bounds(layout.sequence_length, _S.world)
    lo, hi = bounds[_S.rank], bounds[_S.rank + 1]
    hidden = hidden[lo:hi].contiguous()
    shard_segments = _shard_segments(segments, lo, hi)
    shard_rope = rope[:, lo:hi] if rope is not None else None
    _S.bounds = bounds
    try:
        for block in self.blocks:
            self._check_interrupt()
            h_list = [hidden]
            hidden = None
            hidden = block(h_list, temb, shard_segments, shard_rope)
    finally:
        _S.bounds = None
    hidden = _gather_seq(hidden, bounds)
    # ---- end USP delta: from here identical on every rank --------------

    video_row = int(timestep_indices[video_start])
    audio_row = int(timestep_indices[audio_start + min(layout.num_target_condition_audio_latents,
                                                       max(audio_t - 1, 0))])
    h_list = [hidden]
    hidden = None
    video, audio = self.final_layer(h_list, temb, (video_start, layout.sequence_length, video_row),
                                    (audio_start, video_start, audio_row))
    del temb, rope, timestep_indices
    video = T._to_dtype([video], video_dtype)
    audio = T._to_dtype([audio], audio_dtype)
    return (T.unpatchify_video_tokens(video, latent_t, latent_h, latent_w, self.latents_dim, self.patch_size),
            T.unpack_audio(audio))


_ORIG_FORWARD = None
_ORIG_SOL_CALL = None


def activate() -> int:
    """Initialize NCCL from torchrun-style env and patch the model classes.

    Call BEFORE the model loads (class-level patch — instances inherit it).
    Returns this process's rank. Safe to call when WORLD_SIZE is unset or 1:
    patches become passthroughs."""
    global _ORIG_FORWARD, _ORIG_SOL_CALL

    world = int(os.environ.get("WORLD_SIZE", "1"))
    _S.world = world
    _S.rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        if not dist.is_initialized():
            local = int(os.environ.get("LOCAL_RANK", _S.rank))
            torch.cuda.set_device(local % max(torch.cuda.device_count(), 1))
            dist.init_process_group("nccl")
        heads = 56  # MiniMaxH3 num_attention_heads; checked again at first a2a by shape math
        if heads % world:
            raise RuntimeError(f"world size {world} must divide {heads} attention heads")
    _S.enabled = world > 1

    from . import transformer as T
    from .sol_attention import MiniMaxH3SolAttention
    if _ORIG_FORWARD is None:
        _ORIG_FORWARD = T.MiniMaxH3Model.forward
        T.MiniMaxH3Model.forward = _usp_forward
    if _ORIG_SOL_CALL is None:
        _ORIG_SOL_CALL = MiniMaxH3SolAttention.__call__
        MiniMaxH3SolAttention.__call__ = _usp_attention_call
    return _S.rank
