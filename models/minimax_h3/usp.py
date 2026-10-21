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
  ``torch.distributed.all_to_all`` takes ragged tensor lists. A block's
  attention reaches ``pay_attention`` by one of two routes, and both are
  hooked: with split q/k/v projections (``qkv_splitting``, the default)
  through ``shared.attention_kit.qkv_attention``, which takes the Ulysses
  exchange as its ``attention_fn``; with the fused ``qkv_proj`` (the "Lower
  RAM" config, GGUF) through ``MiniMaxH3SolAttention.__call__``;
- after the block loop the shards are all-gathered, so the final layer and
  everything downstream is identical on every rank.

Every rank runs the ENTIRE pipeline (text encode, VAE, sampling loop)
redundantly and deterministically — initial latents come from a CPU
generator seeded by the task seed, so ranks stay in lockstep without a
command channel. Rank 0's outputs are the ones to keep.

Not supported under USP (assert loudly rather than silently corrupt):
- Sol, Veda and VSA sparse attention, and VDN hybrid attention: they plan
  their work over the global sequence / video grid; run ``--attention sdpa``
  or ``sage2`` on a standard H3 variant.
- ``skip_steps_cache`` variants (Spectrum / FirstBlockCache): their
  full-sequence signatures diverge per shard.
- ControlNet-Union control streams (``payload["control_rows"]``): the control
  stream and its skip row ranges are full-sequence.
- CUDA graphs of the DiT blocks: collectives run inside every block, so
  :func:`activate` registers an mmgp graph gate that keeps the blocks eager.

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


def _usp_heads_attention(qkv_list):
    """Ulysses attention of this rank's shard: q/k/v ``[1, shard, H, D]`` (handed
    off in ``qkv_list``) -> ``[1, shard, H, D]``. Between the two all-to-alls the
    rank holds the FULL sequence for ``H / world`` heads, so ``pay_attention``
    computes exact full-sequence attention for those heads."""
    from shared.attention import pay_attention

    bounds = _S.bounds
    query, key, value = qkv_list
    qkv_list.clear()
    query = _seq_to_heads(query, bounds)
    key = _seq_to_heads(key, bounds)
    value = _seq_to_heads(value, bounds)
    output = pay_attention([query, key, value], recycle_q=True)
    del query, key, value
    return _heads_to_seq(output, bounds)


def _usp_attention_call(self, qkv_list, use_sol):
    """Class-level wrapper for MiniMaxH3SolAttention.__call__: the DiT blocks'
    route to attention when their q/k/v come from the fused ``qkv_proj``."""
    if not is_active() or _S.bounds is None:
        return _ORIG_SOL_CALL(self, qkv_list, use_sol)
    if use_sol:
        raise RuntimeError("Sol sparse attention is global-sequence-based and unsupported "
                           "under USP; run --attention sdpa or sage2.")
    return _usp_heads_attention(qkv_list)


def _usp_qkv_attention(x_list, q_proj, k_proj, v_proj, heads, head_dim, norm_rope, extra_proj=None, **kwargs):
    """Wrapper for ``shared.attention_kit.qkv_attention``: the DiT blocks' route
    to attention when MMGP split ``qkv_proj`` into q/k/v (``qkv_splitting``, the
    MiniMax H3 default). That path projects, normalizes and rotates the shard's
    rows itself and never reaches MiniMaxH3SolAttention.__call__, so without
    this hook each rank would attend over its own shard only -- plausible-looking
    garbage, not an error. Outside a USP block loop (the TokenRefiner, any other
    model) it is a passthrough."""
    if not is_active() or _S.bounds is None:
        return _ORIG_QKV_ATTENTION(x_list, q_proj, k_proj, v_proj, heads, head_dim, norm_rope,
                                   extra_proj=extra_proj, **kwargs)
    if extra_proj is not None or kwargs.get("attention_fn") is not None:
        raise RuntimeError("Veda / VSA sparse attention is global-sequence-based and unsupported "
                           "under USP; run --attention sdpa or sage2 on a standard H3 variant.")
    # The all-to-all splits the heads across ranks itself and needs all of them at
    # once: an Attention Head Split group (e.g. 7 of 56 heads) would not divide by
    # the world size. head_split_level=0 keeps one group of all heads.
    kwargs["attention_fn"] = lambda qkv_list, group: _usp_heads_attention(qkv_list)
    kwargs["head_split_level"] = 0
    return _ORIG_QKV_ATTENTION(x_list, q_proj, k_proj, v_proj, heads, head_dim, norm_rope, **kwargs)


def _usp_graph_gate():
    """mmgp CUDA-graph gate (``cuda_graphs.register_gate``): the blocks run
    collectives, so they stay eager while USP is active."""
    return "USP sequence parallelism runs collectives inside the DiT blocks" if is_active() else None


def _refuse_global_attention(model):
    """Sparse attentions plan their work over the global sequence / video grid."""
    if model.sol_attention.enabled:
        raise RuntimeError("Sol sparse attention is global-sequence-based and unsupported "
                           "under USP; run --attention sdpa or sage2.")
    for name in ("veda_attention", "vsa_attention"):
        policy = getattr(model, name, None)
        if policy is not None and policy.enabled:
            raise RuntimeError(f"{name.split('_')[0].upper()} sparse attention is global-sequence-based "
                               "and unsupported under USP; run a standard H3 variant.")


def _usp_forward(self, video_x, audio_x, sigma_video, sigma_audio, context, payload,
                 spectrum=None, first_block_cache=None):
    """Rank-aware copy of MiniMaxH3Model.forward (transformer.py:753-948).

    Kept as close to upstream as possible; the USP deltas are marked. Pinned to
    the upstream revision this branch tracks (dee6635) — re-diff on upstream
    bumps; runpod_worker/tests/test_usp_drift.py fails when the original moves.
    Upstream paths this copy drops are refused up front instead of ported: the
    Spectrum forecast branch and the FirstBlockCache loop (step caches), the
    ControlNet-Union control stream, and VDN's per-block planning."""
    from mmgp import cuda_graphs
    from shared.kernels import ops as kernels

    from . import transformer as T

    if not is_active():
        return _ORIG_FORWARD(self, video_x, audio_x, sigma_video, sigma_audio, context,
                             payload, spectrum=spectrum, first_block_cache=first_block_cache)
    # ---- USP delta: refuse what this copy does not shard ----------------
    if spectrum is not None or first_block_cache is not None:
        raise RuntimeError("skip_steps_cache (spectrum/first_block) is unsupported under USP; "
                           "disable the step cache.")
    if payload.get("control_rows") is not None:
        raise RuntimeError("ControlNet-Union control streams are unsupported under USP: the control "
                           "stream and its skip row ranges are full-sequence.")
    if hasattr(self.blocks[0].attn, "vdn"):
        raise RuntimeError("VDN hybrid attention is global-sequence-based and unsupported under USP.")
    # ---- end USP delta ----------------------------------------------------

    device, dtype = video_x.device, self.dtype or next(self.blocks.parameters()).dtype
    video_dtype, audio_dtype = video_x.dtype, audio_x.dtype
    _, _, latent_t, latent_h, latent_w = video_x.shape
    sigma_video = sigma_video.flatten()
    if sigma_video.numel() not in (1, latent_t):
        raise ValueError(f"MiniMax H3 received {sigma_video.numel()} video sigmas for {latent_t} latent frames")
    if not self.control_layers and sigma_video.numel() > 1 and not bool((sigma_video == sigma_video[0]).all()):
        raise ValueError("MiniMax H3 requires one uniform video sigma")
    # A clean first-frame ControlNet anchor must not set the text/audio prefix's noise level.
    base_video_sigma = sigma_video.max() if self.control_layers else sigma_video[0]
    audio_t = audio_x.shape[-1]
    text_tags = payload["text_token_tags"].view(-1).cpu()
    layout = self._layout(text_tags, latent_t, latent_h, latent_w, audio_t, payload)
    target_video_order = payload.get("target_video_order")
    target_video_inverse_order = payload.get("target_video_inverse_order")
    target_video_fixed_rows = payload.get("target_video_fixed_rows", 0)
    target_video_mask_active = payload.get("target_video_mask_active", False)

    # (USP delta: no Spectrum forecast branch here -- spectrum is refused above.)
    hidden = self._prepare_graph_input(video_x, audio_x, context, payload, layout, dtype)
    if hidden is None:
        video_x = video_x.to(torch.float32)
        video_rows = T.patchify_video(video_x, self.patch_size)
        video_x= None
        if target_video_order is not None:
            video_rows = video_rows.index_select(0, target_video_order)
        audio_x = audio_x.to(torch.float32)
        audio_rows = T.pack_audio(audio_x)
        audio_x = None
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
    else:
        video_x = audio_x = None

    timestep, timestep_indices = T.build_row_timesteps(
        layout,
        float(1.0 - base_video_sigma),
        float(1.0 - sigma_audio.flatten()[0]),
        max(float(1.0 - base_video_sigma), T.VISUAL_COND_TIMESTEP),
        T.AUDIO_COND_TIMESTEP,
        keyframe_timesteps=T._keyframe_timesteps(payload.get("keyframes"), latent_h // self.patch_size[1] * (latent_w // self.patch_size[2])),
    )
    timestep, timestep_indices = timestep.to(device), timestep_indices.to(device)
    target_video_rows = latent_t * (latent_h // self.patch_size[1]) * (latent_w // self.patch_size[2])
    video_start = layout.sequence_length - target_video_rows
    video_head_row = int(timestep_indices[video_start])
    frame_rows = grouped_segments = None
    if target_video_order is not None:
        timestep, timestep_indices, video_head_row, grouped_segments = T._grouped_video_timestep_rows(
            timestep, timestep_indices, video_start, target_video_rows,
            target_video_fixed_rows, target_video_mask_active)
    elif sigma_video.numel() == latent_t:
        timestep, timestep_indices, video_head_row, frame_rows = T._frame_video_timestep_rows(timestep, timestep_indices, sigma_video, target_video_rows)
    adaln_indices = timestep_indices * 3 + layout.token_tags.to(device).clamp_min(0)
    changes = torch.cat((torch.ones(1, dtype=torch.bool, device=device), adaln_indices[1:] != adaln_indices[:-1],
                         torch.ones(1, dtype=torch.bool, device=device))).nonzero().flatten()
    segments = [(int(changes[index]), int(changes[index + 1]), int(adaln_indices[changes[index]]))
                for index in range(changes.numel() - 1)]
    if frame_rows is not None:
        segments = [segment for segment in segments if segment[0] < video_start]
        rows_per_frame = target_video_rows // latent_t
        segments.extend((video_start + index * rows_per_frame, video_start + (index + 1) * rows_per_frame,
                         int(row) * 3 + T.MINIMAX_H3_VIDEO_TAG) for index, row in enumerate(frame_rows))
    elif grouped_segments is not None:
        segments = [segment for segment in segments if segment[0] < video_start]
        segments.extend(grouped_segments)
    temb = self._time_embedding(timestep)
    ref2va_temb = self._time_embedding(timestep, self.ref2va_adaln_t_table) if self.hybrid_ref2va_blocks is not None else None
    rope = payload.get("rope")
    if rope is None:
        positions = layout.position_ids.to(torch.float32)
        frequencies = positions.unsqueeze(-1) * self.rope.inv_freq.detach().cpu().view(1, 1, -1)
        rope = T._rope_table(torch.cat(frequencies.unbind(dim=1), dim=-1), dtype).to(device)
        if target_video_order is not None:
            target_rope = rope[:, video_start:].index_select(1, target_video_order)
            rope[:, video_start:].copy_(target_rope)
            del target_rope
        payload["rope"] = rope
        del positions, frequencies
    del adaln_indices, changes
    target_audio_rows = audio_t * 2
    audio_start = video_start - target_audio_rows
    self.sol_attention.begin_forward(layout, device, dtype, payload["attention_sparsity"], target_video_order is not None)
    if hasattr(self, "veda_attention"):
        self.veda_attention.begin_forward(layout, (latent_t, latent_h // self.patch_size[1], latent_w // self.patch_size[2]), payload, device, dtype)
    if hasattr(self, "vsa_attention"):
        self.vsa_attention.begin_forward(layout, (latent_t, latent_h // self.patch_size[1], latent_w // self.patch_size[2]), payload, device, dtype)
    _refuse_global_attention(self)  # USP delta
    # VDN is refused above, so upstream's `if not hasattr(self.blocks[0].attn, "vdn")` always holds.
    rope = kernels.prepare_split_half_rope(rope)
    # (USP delta: no control stream setup -- control_rows is refused above.)

    # ---- USP delta: shard the sequence across ranks --------------------
    _assert_ranks_agree(hidden)
    bounds = _shard_bounds(layout.sequence_length, _S.world)
    lo, hi = bounds[_S.rank], bounds[_S.rank + 1]
    hidden = hidden[lo:hi].contiguous()
    shard_segments = _shard_segments(segments, lo, hi)
    # rope is [1, seq, 1, pairs, 2], or (that, its Kitchen-packed twin [1, seq, 1, pairs, 2, 2]).
    shard_rope = tuple(table[:, lo:hi] for table in rope) if isinstance(rope, tuple) else rope[:, lo:hi]
    _S.bounds = bounds
    try:
        with cuda_graphs.scope():  # _usp_graph_gate keeps every block eager
            for block_index in range(len(self.blocks)):
                self._check_interrupt()
                block = self.blocks[block_index]
                block_temb = ref2va_temb if ref2va_temb is not None and self.hybrid_ref2va_blocks[0] <= block_index <= self.hybrid_ref2va_blocks[1] else temb
                h_list = [hidden]
                hidden = None
                hidden = block(h_list, block_temb, shard_segments, shard_rope)
    finally:
        _S.bounds = None
    del shard_rope
    hidden = _gather_seq(hidden, bounds)
    # ---- end USP delta: from here identical on every rank --------------

    audio_row = int(timestep_indices[audio_start + min(layout.num_target_condition_audio_latents,
                                                       max(audio_t - 1, 0))])
    h_list = [hidden]
    hidden = None
    video, audio = self.final_layer(h_list, temb, (video_start, layout.sequence_length, video_head_row),
                                    (audio_start, video_start, audio_row))
    del temb, ref2va_temb, rope, timestep_indices
    if target_video_inverse_order is not None:
        video = video.index_select(0, target_video_inverse_order)
    video = T._to_dtype([video], video_dtype)
    audio = T._to_dtype([audio], audio_dtype)
    return (T.unpatchify_video_tokens(video, latent_t, latent_h, latent_w, self.latents_dim, self.patch_size),
            T.unpack_audio(audio))


_ORIG_FORWARD = None
_ORIG_SOL_CALL = None
_ORIG_QKV_ATTENTION = None


def activate() -> int:
    """Initialize NCCL from torchrun-style env and patch the model classes.

    Call BEFORE the model loads (class-level patch — instances inherit it).
    Returns this process's rank. Safe to call when WORLD_SIZE is unset or 1:
    patches become passthroughs."""
    global _ORIG_FORWARD, _ORIG_SOL_CALL, _ORIG_QKV_ATTENTION

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

    from mmgp import cuda_graphs
    from shared import attention_kit

    from . import transformer as T
    from .sol_attention import MiniMaxH3SolAttention
    if _ORIG_FORWARD is None:
        _ORIG_FORWARD = T.MiniMaxH3Model.forward
        T.MiniMaxH3Model.forward = _usp_forward
    if _ORIG_SOL_CALL is None:
        _ORIG_SOL_CALL = MiniMaxH3SolAttention.__call__
        MiniMaxH3SolAttention.__call__ = _usp_attention_call
    if _ORIG_QKV_ATTENTION is None:
        # transformer.py calls attention_kit.qkv_attention through the module, so
        # rebinding the module attribute reaches every block.
        _ORIG_QKV_ATTENTION = attention_kit.qkv_attention
        attention_kit.qkv_attention = _usp_qkv_attention
    cuda_graphs.register_gate(_usp_graph_gate)
    return _S.rank
