#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU correctness tests for models/minimax_h3/usp.py, using gloo.

Needs torch (run inside the worker image); deliberately NOT part of the CPU
pytest suite, which stays torch-free. Loads usp.py by file path so no wgp /
model imports happen.

    python3 runpod_worker/scripts/test_usp_gloo.py

Proves, with 2 processes:
  1. shard bounds and AdaLN segment resharding match a single-process reference;
  2. the seq<->heads all-to-all round trip is lossless with uneven shards;
  3. USP attention (shard -> a2a -> SDPA on head subset -> a2a -> gather)
     is numerically identical to full-sequence SDPA — Ulysses is exact.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

REPO = Path(__file__).resolve().parents[2]
USP_PATH = REPO / "models" / "minimax_h3" / "usp.py"

WORLD = 2
SEQ = 1031          # deliberately odd: uneven shards (516 / 515)
HEADS = 8           # divisible by WORLD, tiny for CPU speed
DIM = 16
CHANNELS = 32


def _load_usp():
    spec = importlib.util.spec_from_file_location("h3_usp", USP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _worker(rank: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = os.environ.get("USP_TEST_PORT", "29617")
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    usp = _load_usp()
    usp._S.world, usp._S.rank, usp._S.enabled = WORLD, rank, True

    torch.manual_seed(7)  # identical tensors on both ranks
    bounds = usp._shard_bounds(SEQ, WORLD)
    assert bounds[0] == 0 and bounds[-1] == SEQ and len(bounds) == WORLD + 1
    assert max(bounds[i + 1] - bounds[i] for i in range(WORLD)) - \
           min(bounds[i + 1] - bounds[i] for i in range(WORLD)) <= 1
    lo, hi = bounds[rank], bounds[rank + 1]

    # --- 1. segment resharding matches the global modulate ------------------
    segments = [(0, 40, 0), (40, 900, 1), (900, SEQ, 2)]
    hidden = torch.randn(SEQ, CHANNELS)
    scale = torch.randn(3, CHANNELS)
    reference = hidden.clone()
    for start, stop, row in segments:
        reference[start:stop].mul_(1.0 + scale[row])
    shard = hidden[lo:hi].clone()
    for start, stop, row in usp._shard_segments(segments, lo, hi):
        shard[start:stop].mul_(1.0 + scale[row])
    assert torch.equal(shard, reference[lo:hi]), "segment resharding diverged"

    # --- 2. a2a round trip is lossless --------------------------------------
    full = torch.randn(1, SEQ, HEADS, DIM)
    my_shard = full[:, lo:hi].contiguous()
    heads_view = usp._seq_to_heads(my_shard, bounds)   # [1, SEQ, HEADS/WORLD, DIM]
    hpr = HEADS // WORLD
    assert torch.equal(heads_view, full[:, :, rank * hpr:(rank + 1) * hpr]), \
        "seq->heads produced the wrong head group"
    back = usp._heads_to_seq(heads_view, bounds)
    assert torch.equal(back, my_shard), "heads->seq did not invert seq->heads"

    # --- 3. USP attention == full attention ---------------------------------
    q = torch.randn(1, SEQ, HEADS, DIM)
    k = torch.randn(1, SEQ, HEADS, DIM)
    v = torch.randn(1, SEQ, HEADS, DIM)

    def sdpa(q_, k_, v_):
        out = torch.nn.functional.scaled_dot_product_attention(
            q_.transpose(1, 2), k_.transpose(1, 2), v_.transpose(1, 2))
        return out.transpose(1, 2)

    reference_attn = sdpa(q, k, v)[:, lo:hi]
    qs = usp._seq_to_heads(q[:, lo:hi].contiguous(), bounds)
    ks = usp._seq_to_heads(k[:, lo:hi].contiguous(), bounds)
    vs = usp._seq_to_heads(v[:, lo:hi].contiguous(), bounds)
    partial = sdpa(qs, ks, vs)
    mine = usp._heads_to_seq(partial, bounds)
    assert torch.allclose(mine, reference_attn, atol=1e-5), \
        f"USP attention diverged: max err {(mine - reference_attn).abs().max()}"

    # --- 4. gather restores the full sequence --------------------------------
    gathered = usp._gather_seq(hidden[lo:hi].contiguous(), bounds)
    assert torch.equal(gathered, hidden), "gather_seq diverged"

    dist.barrier()
    if rank == 0:
        print("usp gloo tests: ALL PASSED (bounds/segments, a2a round trip, "
              "exact attention equivalence, gather)")
    dist.destroy_process_group()


def main() -> int:
    mp.start_processes(_worker, nprocs=WORLD, start_method="spawn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
