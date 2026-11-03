#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU correctness tests for FastH3's VSA under USP (models/minimax_h3/usp.py), using gloo and the Triton interpreter.

Needs torch and triton (any build: TRITON_INTERPRET=1, set below, runs WanGP's Triton kernels on CPU tensors) -- not part of the
torch-free pytest suite. No GPU, no weights:

    pip install triton            # only if the image lacks it
    python3 runpod_worker/scripts/test_usp_vsa_cpu.py

Proves:
  1. the REAL VSA kernels (vsa/attention.py, both fine paths: exact BF16 and the INT8-QK / FP16-PV one FastH3 runs by default)
     match a pure-torch reference of the same math (tile means, pooled scores, per-region top-k, block-sparse fine attention,
     gated coarse branch);
  2. VSA is head-separable: all heads in one call == any head subsets concatenated, bit for bit, on both paths;
  3. (2 gloo ranks) the USP VSA hook -- ragged sequence shard, all-to-all of q/k/v/gate, VSA on H/2 heads of the FULL sequence,
     all-to-all back -- reproduces single-process VSA bit for bit;
  4. (2 gloo ranks) a tiny MiniMaxH3Model with FastH3's VSA attention (Ref2VA layout: text, an image reference, a reference video
     region at the reference keep rate, audio, the generated video) gives the same forward under USP as the upstream unsharded
     forward, on both attention routes (split q/k/v through attention_kit, whose reference run attends by head groups, and the
     fused qkv_proj), on both fine paths.

Two test-only shims, neither touches a kernel: the interpreter stores bf16 as uint16 and its tl.dot multiplies those raw bit
patterns (it widens only fp8), so bf16 dot operands are widened to fp32 first -- exact, every bf16 value is an fp32 value; and
the Triton launch tuner, which times candidates with CUDA events, returns its static first candidate (what an untunable call
gets). VSAPolicy.begin_forward refuses non-CUDA devices, so part 4 plans the tile layout with a copy of it minus that guard.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import types
from pathlib import Path

os.environ["TRITON_INTERPRET"] = "1"  # before anything imports triton

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
WORLD = 2


def _setup():
    """Import shims (in every process): the repo on sys.path, models.minimax_h3 without its gradio-importing __init__, a CPU
    capability probe, the interpreter's bf16 tl.dot and the launch tuner."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import models  # noqa: F401  (plain package)

    if "models.minimax_h3" not in sys.modules:
        package = types.ModuleType("models.minimax_h3")
        package.__path__ = [str(REPO / "models" / "minimax_h3")]
        sys.modules["models.minimax_h3"] = package
    if not torch.cuda.is_available():
        torch.cuda.get_device_capability = lambda *args, **kwargs: (0, 0)

    import numpy as np
    import triton.language as tl
    from triton.runtime import interpreter

    if not getattr(interpreter.InterpreterBuilder.create_dot, "_bf16_widened", False):
        original = interpreter.InterpreterBuilder.create_dot

        def widen(handle):
            if handle.dtype.scalar == tl.bfloat16:
                return interpreter.TensorHandle((handle.data.astype(np.uint32) << 16).view(np.float32), tl.float32)
            return handle

        def create_dot(self, a, b, d, input_precision, max_num_imprecise_acc):
            return original(self, widen(a), widen(b), d, input_precision, max_num_imprecise_acc)

        create_dot._bf16_widened = True
        interpreter.InterpreterBuilder.create_dot = create_dot

    from shared.kernels import triton_tuner
    triton_tuner.LaunchTuner.config = lambda self, name, identity, device, candidates, prepare_scratch, tunable: candidates[0]


def _set_path(int8):
    """VSA's fine path: INT8-QK / FP16-PV ("fast", FastH3's default, the GPU log's "Triton INT8 QK / FP16 PV") or exact BF16."""
    from shared.kernels import kernel_policy
    kernel_policy.configure("fast" if int8 else "strict")


# ---------------------------------------------------------------------------
# 1. Pure-torch reference of the exact BF16 path
# ---------------------------------------------------------------------------


def vsa_reference(query, key, value, gate, layout):
    """[S, H, D] q/k/v/gate -> [S, H, D] fp32: VSAAttention._heads (int8_qk False) + kernel.attend + scatter_gated, one head at a
    time, from the tiling docstrings: tiles are `perm` slots (-1 = padding), every query attends the prefix tiles and, for a video
    query tile, the per-region top-k tiles of its pooled scores; the coarse branch is softmax(pooled scores) @ pooled v, added to
    each tile's rows scaled by the gate."""
    seq, heads, dim = query.shape
    perm, tile = layout.perm.cpu(), layout.tile
    real = perm >= 0
    source = perm.clamp_min(0)
    n_tiles, n_prefix = layout.n_tiles, layout.n_prefix_tiles
    valid = layout.valid_count.cpu().clamp_min(1).float()
    tile_of_slot = torch.arange(perm.numel()) // tile
    out = torch.zeros(seq, heads, dim)
    for head in range(heads):
        def tiled(x, head=head):
            rows = x[source, head].float()
            rows[~real] = 0
            return rows

        q, k, v = tiled(query), tiled(key), tiled(value)

        def pool(rows):
            return rows.view(n_tiles, tile, dim).sum(1) / valid[:, None]

        q_pooled, k_pooled, v_pooled = pool(q), pool(k), pool(v)
        scores = q_pooled @ k_pooled.T * dim ** -0.5
        allowed = torch.zeros(n_tiles, n_tiles, dtype=torch.bool)
        allowed[:n_prefix] = True
        allowed[n_prefix:, :n_prefix] = True
        for region in layout.regions:
            kept = torch.topk(scores[n_prefix:, region.start:region.stop], region.keep, dim=-1).indices + region.start
            allowed[torch.arange(n_prefix, n_tiles)[:, None], kept] = True
        mask = allowed[tile_of_slot][:, tile_of_slot] & real[None, :]
        probs = torch.softmax((q @ k.T * dim ** -0.5).masked_fill(~mask, float("-inf")), dim=-1)
        fine = probs.to(torch.bfloat16).float() @ v  # the kernel rounds P to BF16 for the PV product
        coarse = torch.softmax(scores, dim=-1) @ v_pooled
        rows = fine + coarse[tile_of_slot] * gate[source, head].float()
        out[source[real], head] = rows[real]
    return out


def _layouts():
    """FastH3-shaped tile layouts (dense text / image reference / audio spans, a reference video region at the reference keep
    rate, the generated video last), 128-token tiles as FastH3 Omni and 64-token ones, edge boxes partial. Odd lengths."""
    from models.minimax_h3.vsa import tiling

    spans = [(0, 37), (37, 85)]  # text, a 6x8 image reference
    reference_video = (85, (4, 4, 10))  # 160 rows
    audio = (245, 267)
    target_grid = (6, 8, 12)  # 576 rows
    seq = 267 + math.prod(target_grid)
    out = []
    for tile in (128, 64):
        layout = tiling.build_tile_layout(seq, spans + [audio], [reference_video, (seq - math.prod(target_grid), target_grid)],
                                          tile, 0.75, 0.5, "cpu")
        out.append(layout)
    return out


def _policy(layout, int8):
    from models.minimax_h3.vsa.attention import VSAPolicy

    policy = VSAPolicy({"vsa_sparsity": 0.75, "vsa_tile_size": layout.tile, "vsa_ref_keep_rate": 0.5})
    policy.layout, policy.int8_qk, policy.enabled = layout, int8, True
    return policy


def _qkvg(seq, heads, dim, seed):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(1, seq, heads, dim, generator=generator).to(torch.bfloat16) for _ in range(4)]


def _run_heads(attention, tensors, first, last, tiled_bytes=None):
    """VSA over heads [first, last) of disposable copies of q/k/v/gate [1, S, H, D]; tiled_bytes overrides VSA's head chunk bound."""
    from shared.attention_kit import HeadGroup
    from models.minimax_h3.vsa import attention as A

    saved = A._TILED_BYTES
    if tiled_bytes is not None:
        A._TILED_BYTES = tiled_bytes
    try:
        q, k, v, g = (t[:, :, first:last].clone() for t in tensors)
        return attention([q, k, v], HeadGroup(range(first, last), range(first, last), None), [g])
    finally:
        A._TILED_BYTES = saved


def part_single_process():
    from models.minimax_h3.vsa import attention as A

    heads, dim = 6, 32
    for layout in _layouts():
        target = layout.regions[-1]
        print(f"layout: {layout.tile}-token tiles, seq {layout.seq_len}, {layout.n_prefix_tiles} prefix tiles, regions "
              + ", ".join(f"{r.keep}/{r.stop - r.start}" for r in layout.regions)
              + f" (generated video {100 * (1 - target.keep / (target.stop - target.start)):.0f}% sparse)")
        tensors = _qkvg(layout.seq_len, heads, dim, seed=layout.tile)
        reference = vsa_reference(*(t[0] for t in tensors), layout)
        scale = reference.abs().max().item()
        for int8 in (False, True):
            attention = A.VSAAttention(_policy(layout, int8), 0)
            full = _run_heads(attention, tensors, 0, heads)
            halves = torch.cat((_run_heads(attention, tensors, 0, heads // 2), _run_heads(attention, tensors, heads // 2, heads)), dim=2)
            singles = torch.cat([_run_heads(attention, tensors, h, h + 1) for h in range(heads)], dim=2)
            chunked = _run_heads(attention, tensors, 0, heads, tiled_bytes=1)  # VSA's own head chunking: one head per chunk
            name = "INT8-QK/FP16-PV" if int8 else "exact BF16"
            for label, other in (("2 halves", halves), ("6 single heads", singles), ("1-head chunks", chunked)):
                assert torch.equal(full, other), f"{name} tile {layout.tile}: all heads != {label} (max {(full - other).abs().max()})"
            err = (full[0].float() - reference).abs().max().item()
            bound = (0.02 if not int8 else 0.06) * scale
            assert err <= bound, f"{name} tile {layout.tile}: kernel vs torch reference max err {err} > {bound}"
            print(f"  ok {name:15} all heads == halves == single heads == 1-head chunks (bitwise); "
                  f"kernel vs torch reference max|err| {err:.2e} (output scale {scale:.2f})")


# ---------------------------------------------------------------------------
# 3. + 4. Two gloo ranks
# ---------------------------------------------------------------------------


def _cpu_begin_forward(self, layout, grid, payload, device, dtype):
    """VSAPolicy.begin_forward (vsa/attention.py) minus its CUDA / SM80 / BF16 guard and banner."""
    from mmgp import offload
    from shared.kernels import kernel_policy
    from models.minimax_h3.vsa import tiling

    self.enabled = offload.shared_state.get("_attention") == "vsa"
    if not self.enabled:
        return
    if payload.get("target_video_order") is not None:
        raise ValueError("H3 VSA does not support grouped-row mask denoising")
    signature = (payload["layout_signature"], device)
    if signature == self.signature:
        return
    self.reset()
    self.signature = signature
    regions = [(start, region_grid) for start, _, region_grid in layout.reference_video_spans] + [(layout.sequence_length - math.prod(grid), grid)]
    self.layout = tiling.build_tile_layout(layout.sequence_length, layout.dense_spans, regions, self.tile, self.sparsity, self.ref_keep_rate, device)
    self.int8_qk = kernel_policy.allow_approximate()
    self.cuda_kind = None


def part_usp_attention(rank, usp):
    from shared.attention_kit import HeadGroup
    from models.minimax_h3.vsa import attention as A

    heads, dim = 6, 32
    for layout in _layouts():
        bounds = usp._shard_bounds(layout.seq_len, WORLD)
        lo, hi = bounds[rank], bounds[rank + 1]
        q, k, v, g = _qkvg(layout.seq_len, heads, dim, seed=7 + layout.tile)
        for int8 in (False, True):
            attention = A.VSAAttention(_policy(layout, int8), 0)
            group = HeadGroup(range(heads), range(heads), None)
            reference = usp._ORIG_VSA_CALL(attention, [q.clone(), k.clone(), v.clone()], group, [g.clone()])
            shard = [t[:, lo:hi].clone() for t in (q, k, v, g)]
            usp._S.bounds = bounds
            try:
                mine = attention(shard[:3], group, shard[3:])  # A.VSAAttention.__call__ is usp._usp_vsa_call
            finally:
                usp._S.bounds = None
            assert mine.shape == (1, hi - lo, heads, dim), mine.shape
            assert torch.equal(mine, reference[:, lo:hi]), \
                f"rank {rank} tile {layout.tile} int8={int8}: USP VSA != VSA (max {(mine - reference[:, lo:hi]).abs().max()})"
            gathered = usp._gather_seq(mine[0].reshape(hi - lo, -1).contiguous(), bounds).view(1, layout.seq_len, heads, dim)
            assert torch.equal(gathered, reference)
            if rank == 0:
                print(f"  ok USP VSA hook, {layout.tile}-token tiles, {'INT8-QK/FP16-PV' if int8 else 'exact BF16':15} shards "
                      f"{[bounds[j + 1] - bounds[j] for j in range(WORLD)]}: == single-process VSA (bitwise)")


def _build_model(split_qkv, dtype):
    from shared.kernels import ops as kernels
    import models.minimax_h3.transformer as T
    from models.minimax_h3.vsa.attention import VSAAttention, VSAPolicy

    kernels.mlp_chunk_rows = lambda tokens, row_bytes, device: tokens  # CPU: no L2 query
    torch.manual_seed(1234)
    model = T.MiniMaxH3Model(hidden_size=64, num_layers=2, token_refiner_num_layers=1, num_attention_heads=4,
                             attention_head_dim=16, ffn_hidden_size=96, latents_dim=24, audio_latents_dim=32,
                             patch_size=(1, 2, 2), text_dim=48, timestep_input_dim=32, time_embed_hidden_size=64,
                             time_embed_dim=32, rope_inv_freq_len=2, vsa=True, dtype=dtype, device="cpu")
    # the student's trained contract (models/minimax_h3/minimax_h3_main.py:189-194), 64-token tiles for a tiny grid
    model.vsa_attention = VSAPolicy({"vsa_sparsity": 0.75, "vsa_tile_size": 64, "vsa_ref_keep_rate": 0.5})
    for layer, block in enumerate(model.blocks):
        block.attn.vsa = VSAAttention(model.vsa_attention, layer)
    if split_qkv:
        from mmgp import offload
        offload.split_linear_modules(model, T.get_linear_split_map(model.attention_inner_size, 4, 16, "grouped"))
        assert hasattr(model.blocks[0].attn, "q_proj") and not hasattr(model.blocks[0].attn, "qkv_proj")
    return model.eval().requires_grad_(False)


def _model_inputs():
    """Ref2VA packing as FastH3 Omni sees it: text, one image reference (dense), one reference video (a VSA region at the
    reference keep rate), target audio, then the generated video (8x8x8 tokens: 8 tiles, keep 2)."""
    generator = torch.Generator().manual_seed(99)
    text_len, latent_t, latent_h, latent_w, audio_t = 13, 8, 16, 16, 5
    refs = [{"kind": "image", "latent_t": 1, "latent_h": 8, "latent_w": 8, "ref_audio_t": 0},
            {"kind": "video", "latent_t": 4, "latent_h": 8, "latent_w": 16, "ref_audio_t": 0}]
    cond_rows = 1 * 4 * 4 + 4 * 4 * 8
    video_x = torch.randn(1, 24, latent_t, latent_h, latent_w, generator=generator)
    audio_x = torch.randn(1, 32, 2, audio_t, generator=generator)
    context = torch.randn(1, text_len, 48, generator=generator)
    payload = {"text_token_tags": torch.ones(text_len, dtype=torch.long), "fps": 24, "keyframes": None,
               "audio_keyframes": None, "refs": refs, "cond_video_rows": torch.randn(cond_rows, 96, generator=generator),
               "cond_audio_rows": None, "target_audio_condition_latents": 0, "target_video_condition_frames": 0,
               "attention_sparsity": 0.0, "audio_time_offset": 0.0}
    return video_x, audio_x, torch.tensor([0.7]), torch.tensor([0.6]), context, payload


def part_usp_forward(rank, usp, dtype):
    from mmgp import offload
    from shared import attention_kit
    from models.minimax_h3.vsa.attention import VSAAttention, VSAPolicy

    VSAPolicy.begin_forward = _cpu_begin_forward
    offload.shared_state["_attention"] = "vsa"
    cases = [(split, int8) for split in (False, True) for int8 in (False, True)]
    calls = {"n": 0, "heads": []}
    original = usp._ORIG_VSA_CALL
    assert VSAAttention.__call__ is usp._usp_vsa_call

    def counting(self, qkv_list, group, gate_list):  # the unwrapped VSA call: sees the heads VSA actually attends
        calls["n"] += 1
        calls["heads"].append(qkv_list[0].shape[2])
        return original(self, qkv_list, group, gate_list)

    usp._ORIG_VSA_CALL = counting
    worst = 0.0
    try:
        for split, int8 in cases:
            _set_path(int8)
            video_x, audio_x, sv, sa, context, payload = _model_inputs()
            context = context.to(dtype)  # the text encoder hands the transformer its own dtype
            model = _build_model(split, dtype)
            # upstream forward, unsharded: USP's patches are passthroughs while _S.enabled is off; VSA's own
            # Attention Head Split level (transformer.VSA_HEAD_SPLIT_LEVEL) applies at any length here, as at 8192+ tokens
            usp._S.enabled = False
            min_split, attention_kit.MIN_SPLIT_TOKENS = attention_kit.MIN_SPLIT_TOKENS, 1
            calls.update(n=0, heads=[])
            try:
                ref_v, ref_a = model(video_x, audio_x, sv, sa, context, copy.copy(payload))
            finally:
                attention_kit.MIN_SPLIT_TOKENS = min_split
                usp._S.enabled = True
            reference_heads = list(calls["heads"])
            # the split route attends by VSA's head groups single-GPU (4 heads -> groups of 1), the fused one all heads at once
            assert max(reference_heads) == (1 if split else 4), reference_heads
            model = _build_model(split, dtype)
            calls.update(n=0, heads=[])
            out_v, out_a = model(video_x, audio_x, sv, sa, context, copy.copy(payload))
            assert calls["n"] == len(model.blocks) and set(calls["heads"]) == {4 // WORLD}, \
                f"rank {rank}: VSA ran {calls['n']}x on {calls['heads']} heads under USP, expected {len(model.blocks)}x on {4 // WORLD}"
            err = max((out_v - ref_v).abs().max().item(), (out_a - ref_a).abs().max().item())
            scale = max(ref_v.abs().max().item(), ref_a.abs().max().item())
            worst = max(worst, err / scale)
            assert err <= 2e-2 * scale, f"rank {rank} split={split} int8={int8}: max abs err {err} (scale {scale})"
            if rank == 0:
                layout = model.vsa_attention.layout
                print(f"  ok {str(dtype).split('.')[-1]:8} split_qkv={split!s:5} {'INT8-QK/FP16-PV' if int8 else 'exact BF16':15} "
                      f"seq {layout.seq_len}, regions " + ", ".join(f"{r.keep}/{r.stop - r.start}" for r in layout.regions)
                      + f": max|usp-ref| {err:.2e} (scale {scale:.2f}); reference VSA heads per call {sorted(set(reference_heads))}, "
                      f"USP {sorted(set(calls['heads']))}")
    finally:
        usp._ORIG_VSA_CALL = original
    return worst


def _worker(rank):
    _setup()
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=os.environ.get("USP_TEST_PORT", "29653"),
                      WORLD_SIZE=str(WORLD), RANK=str(rank))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    from models.minimax_h3 import usp

    usp.activate()  # gloo is already initialized: patches only
    assert usp.is_active() and usp._ORIG_VSA_CALL is not None
    part_usp_attention(rank, usp)
    # BF16 as on the GPU: the exact path's PV product takes BF16 operands only (tl.dot refuses bf16 x fp32)
    worst = part_usp_forward(rank, usp, torch.bfloat16)
    if rank == 0:
        print(f"  usp forward with VSA: worst relative error {worst:.1e}")
    dist.barrier()
    if rank == 0:
        print("usp vsa tests: ALL PASSED (kernel == torch reference, head separability, USP VSA hook, USP forward)")
    dist.destroy_process_group()


def main() -> int:
    _setup()
    print("1/2. VSA kernels (Triton interpreter) vs torch reference, and head separability")
    part_single_process()
    print("3/4. USP over 2 gloo ranks: the VSA hook, then a tiny H3 forward with VSA")
    mp.start_processes(_worker, nprocs=WORLD, start_method="spawn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
