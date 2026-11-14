"""A decode worker's VAE for the CPU tests: the tiny VAE of test_streaming.py, built the same way (same seed, same weights).

The worker process imports this module through its recipe ("worker_vae:tiny_vae", with this folder on sys.path). It registers the
same CPU stand-ins as test_streaming.py first, so the VAE's modules import on a CPU-only runner.
"""
import importlib
import sys
import types
from pathlib import Path

import torch

_ROOT = str(Path(__file__).resolve().parents[3])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
for name, rel in (("models", "models"), ("models.minimax_h3", "models/minimax_h3"), ("shared", "shared")):
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(_ROOT) / rel)]
        sys.modules[name] = package
if "shared.attention" not in sys.modules:
    import torch.nn.functional as F

    attention = types.ModuleType("shared.attention")

    def _sdpa(qkv_list, causal=False, **_kw):
        q, k, v = qkv_list
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal).transpose(1, 2)

    attention.pay_attention = _sdpa
    attention.sage_staged = lambda *_args, **_kw: None
    attention.sage_attention_mask = lambda *_args, **_kw: (_ for _ in ()).throw(NotImplementedError("no Sage masks on CPU"))
    sys.modules["shared.attention"] = attention
    sys.modules["shared"].attention = attention


def tiny_vae(fail: bool = False):
    if fail:
        raise RuntimeError("this recipe fails on purpose")
    AutoencoderKLMiniMaxH3 = importlib.import_module("models.minimax_h3.components.video_autoencoder").AutoencoderKLMiniMaxH3
    torch.manual_seed(0)
    return AutoencoderKLMiniMaxH3(latent_channels=4, block_out_channels=(8, 8, 8, 8, 8, 8), layers_per_block=1, norm_num_groups=4,
                                  decoder_num_layers=1, decoder_num_attention_heads=1, decoder_attention_head_dim=8,
                                  decoder_num_register_tokens=1, decoder_ffn_mult=1).eval()
