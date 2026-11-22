# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""CPU tests of pi-H3 DWD's head math and weight merge (models/minimax_h3/pi_h3_dwd.py) against LakonLab's released formulas
(https://github.com/Lakonik/LakonLab v0.3.0: MiniMaxH3DWDAdaLayerNormOut, SolH3DWDHead.forward / _project_pixels). The Sol layers
themselves are WanGP's Sol-H3-VAE decoder, which needs CUDA: their parity with LakonLab's refiner is checked on a GPU.

    python -m pytest models/minimax_h3/tests/test_pi_h3_dwd.py -q
"""

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]


def _load(name):
    """A models.minimax_h3 module without the package __init__ (which imports the whole handler)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    for package_name, folder in (("models", ROOT / "models"), ("models.minimax_h3", ROOT / "models" / "minimax_h3")):
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(folder)]
            sys.modules[package_name] = package
    return importlib.import_module(name)


pi_h3 = _load("models.minimax_h3.pi_h3")
dwd = _load("models.minimax_h3.pi_h3_dwd")


def test_merged_linear_names():
    assert dwd.merged_linear_name("transformer_blocks.46.attn.to_out.0.weight") == "blocks.46.attn.out_proj"
    assert dwd.merged_linear_name("transformer_blocks.49.ff.net.0.proj.weight") == "blocks.49.mlp.fc1"
    assert dwd.merged_linear_name("transformer_blocks.47.ff.net.2.weight") == "blocks.47.mlp.fc2"
    for other in ("transformer_blocks.46.attn.to_out.0.lora_A.weight", "transformer_blocks.46.adaln_proj.linear.bias",
                  "dwd_pixel_refiner.blocks.0.attn.to_out.weight", "transformer_blocks.46.attn.to_out.0.bias"):
        assert dwd.merged_linear_name(other) is None


def _lakonlab_norm_out(x, temb_diffusers, weight, bias, eps):
    """MiniMaxH3DWDAdaLayerNormOut (AdaLayerNormContinuous, affine-free LayerNorm): scale first, then shift."""
    scale, shift = F.linear(F.silu(temb_diffusers), weight, bias).chunk(2, dim=-1)
    return F.layer_norm(x, x.shape[-1:], eps=eps) * (1 + scale) + shift


def test_norm_out_matches_lakonlab():
    generator = torch.Generator().manual_seed(0)
    width, rows = 16, 40
    module = dwd.MiniMaxH3DWDNormOut(width, 8, 1e-5)
    with torch.no_grad():
        module.linear.weight.copy_(torch.randn(2 * width, 8, generator=generator))
        module.linear.bias.copy_(torch.randn(2 * width, generator=generator))
    hidden = torch.randn(rows, width, generator=generator) * 3 + 1
    temb = torch.randn(3, 8, generator=generator)  # three timesteps, WanGP's coordinates
    spans = [(0, 10, 2), (10, 35, 0), (35, 40, 1)]
    features = module(hidden, temb, spans)
    diffusers = temb * torch.tensor(pi_h3.ADALN_SIGNS)  # the port's coordinates: WanGP's with three axes negated
    for start, stop, row in spans:
        expected = _lakonlab_norm_out(hidden[start:stop], diffusers[row], module.linear.weight, module.linear.bias, 1e-5)
        torch.testing.assert_close(features[start:stop], expected, rtol=1e-5, atol=1e-5)
    # the signs matter: the SiLU does not let them fold into the weights
    assert not torch.allclose(features[:10], _lakonlab_norm_out(hidden[:10], temb[2], module.linear.weight, module.linear.bias, 1e-5), atol=1e-3)
    assert module(hidden.bfloat16(), temb, spans).dtype == torch.bfloat16


def test_expand_tokens_matches_lakonlab():
    generator = torch.Generator().manual_seed(1)
    t, h, w, width, dim, expansion = 7, 3, 5, 6, 4, (2, 2)
    expand = nn.Linear(width, dim * 4)
    features = torch.randn(t * h * w, width, generator=generator)
    out = torch.empty(t * h * 2 * w * 2, dim)
    with torch.no_grad():
        dwd.expand_tokens(features, (t, h, w), expand, expansion, out)
        # SolH3DWDHead.forward: x.view(B, t, h, w, e_h, e_w, D).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, t, h * e_h, w * e_w, D)
        expected = expand(features[None]).view(1, t, h, w, 2, 2, dim).permute(0, 1, 2, 4, 3, 5, 6).reshape(1, t, h * 2, w * 2, dim)
    torch.testing.assert_close(out, expected.reshape(-1, dim))


@pytest.mark.parametrize("frames", [1, 4])
def test_unpatchify_matches_lakonlab_projection(frames):
    """_project_pixels: (B, n, t, H, W, D) projections viewed (..., H, W, OUT_CH, frames, P, P), permuted to (frames, OUT_CH, H, P, W, P)."""
    generator = torch.Generator().manual_seed(2)
    rows, width, channels, patch = 3, 4, 3, 2
    projected = torch.randn(rows * width, channels * frames * patch * patch, generator=generator)
    pixels = dwd.unpatchify_pixels(projected, rows, width, channels, frames, patch)
    expected = projected.view(rows, width, channels, frames, patch, patch).permute(3, 2, 0, 4, 1, 5).reshape(frames, channels, rows * patch, width * patch)
    torch.testing.assert_close(pixels[0], expected.transpose(0, 1))  # WanGP's videos are (channels, frames, height, width)


def _head(layers=8):
    tensors = {"dwd_norm_out.linear.weight": torch.zeros(2 * 12, 8), "dwd_norm_out.linear.bias": torch.zeros(24),
               "dwd_pixel_refiner.feature_expand.weight": torch.zeros(16, 12), "dwd_pixel_refiner.register_tokens": torch.zeros(1, 4, 4)}
    for layer in range(layers):
        tensors[f"dwd_pixel_refiner.blocks.{layer}.attn.to_qkv.weight"] = torch.zeros(12, 4)
    return tensors


def test_head_tensors_and_shape():
    head = dwd.head_tensors({**_head(), "transformer_blocks.0.attn.to_out.0.lora_A.weight": torch.zeros(1)})
    assert "dwd_pixel_refiner.transformer_blocks.7.attn.to_qkv.weight" in head and not any(".blocks." in key for key in head)
    assert not any(key.startswith("transformer_blocks.") for key in head)
    assert dwd.refiner_shape(head) == (12, 8, 8)
    with pytest.raises(ValueError, match="linear pixel head"):
        dwd.head_tensors({**_head(), "dwd_linear_head.weight": torch.zeros(1)})


class FakeBlocks(nn.Module):
    """The linears a DWD adapter replaces, at toy widths: out_proj (in 256), fc1 (in 256, gate and up halves), fc2 (in 256)."""

    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        block = nn.Module()
        block.attn, block.mlp = nn.Module(), nn.Module()
        block.attn.out_proj = nn.Linear(256, 8, bias=False, dtype=dtype)
        block.mlp.fc1 = nn.Linear(256, 2 * 256, bias=False, dtype=dtype)
        block.mlp.fc2 = nn.Linear(256, 8, bias=False, dtype=dtype)
        self.blocks = nn.ModuleList([nn.Module(), block])


def _replaced(seed=4):
    generator = torch.Generator().manual_seed(seed)
    return {"transformer_blocks.1.attn.to_out.0.weight": torch.randn(8, 256, generator=generator).half(),
            "transformer_blocks.1.ff.net.0.proj.weight": torch.randn(512, 256, generator=generator).half(),
            "transformer_blocks.1.ff.net.2.weight": torch.randn(8, 256, generator=generator).half()}


def test_merge_into_a_bf16_transformer():
    model, tensors = FakeBlocks(), _replaced()
    fingerprints = dwd.merge_linears(model, dict(tensors))
    block = model.blocks[1]
    assert torch.equal(block.attn.out_proj.weight, tensors["transformer_blocks.1.attn.to_out.0.weight"].bfloat16())
    assert torch.equal(block.mlp.fc2.weight, tensors["transformer_blocks.1.ff.net.2.weight"].bfloat16())
    fc1 = tensors["transformer_blocks.1.ff.net.0.proj.weight"].bfloat16()
    assert torch.equal(block.mlp.fc1.weight, torch.cat((fc1[256:], fc1[:256])))  # diffusers' halves swapped, as for a LoRA's fc1
    assert all(parameter.dtype == torch.bfloat16 and not parameter.requires_grad for parameter in model.parameters())
    assert sorted(fingerprints) == ["blocks.1.attn.out_proj.weight", "blocks.1.mlp.fc1.weight", "blocks.1.mlp.fc2.weight"]
    # the LoRA conversion compares the adapter's tensors (WanGP names, unswapped) with these
    assert torch.equal(fingerprints["blocks.1.mlp.fc1.weight"], dwd.fingerprint(tensors["transformer_blocks.1.ff.net.0.proj.weight"]))
    with pytest.raises(ValueError, match="no linear"):
        dwd.merge_linears(model, {"transformer_blocks.0.attn.to_out.0.weight": torch.zeros(8, 256)})
    with pytest.raises(ValueError, match="shape"):
        dwd.merge_linears(model, {"transformer_blocks.1.attn.to_out.0.weight": torch.zeros(8, 128)})


def _convrot_linear(weight, group=256):
    from shared.convert.convrot import quantize_weight
    from shared.qtypes.int8_convrot import _QINT8_CONVROT_QTYPE, Int8ConvRotWeightTensor, QLinearInt8ConvRot

    module = QLinearInt8ConvRot(weight.shape[1], weight.shape[0], bias=False, dtype=torch.bfloat16, weights=_QINT8_CONVROT_QTYPE)
    data, scale = quantize_weight(weight, group, "cpu")
    module.weight = nn.Parameter(Int8ConvRotWeightTensor.create(_QINT8_CONVROT_QTYPE, 0, weight.size(), weight.stride(), data, scale,
                                                                compute_dtype=torch.bfloat16), requires_grad=False)
    module._convrot_group_size = group
    return module


def test_merge_into_an_int8_convrot_transformer():
    pytest.importorskip("optimum.quanto")
    from shared.convert.convrot import quantize_weight
    from shared.qtypes.int8_convrot import Int8ConvRotWeightTensor, _regular_hadamard

    generator = torch.Generator().manual_seed(5)
    module = _convrot_linear(torch.randn(8, 256, generator=generator))
    new = torch.randn(8, 256, generator=generator)
    dwd.replace_linear_weight(module, new.half(), device="cpu")
    weight = module.weight
    assert isinstance(weight, Int8ConvRotWeightTensor) and weight.dtype == torch.bfloat16 and module._convrot_group_size == 256
    data, scale = quantize_weight(new.half(), 256, "cpu")  # what WanGP's ConvRot converter writes for this weight
    assert torch.equal(weight._data, data) and torch.equal(weight._scale, scale) and weight._scale.dtype == torch.float32
    rotated_back = (weight._data.float() * weight._scale).reshape(8, 1, 256) @ _regular_hadamard(256, "cpu", torch.float32)
    torch.testing.assert_close(rotated_back.reshape(8, 256), new, rtol=0, atol=float(scale.max()))  # within one INT8 step


def test_other_formats_are_refused():
    quanto = pytest.importorskip("optimum.quanto")
    container = nn.Sequential(nn.Linear(256, 8, bias=False))  # quanto replaces child modules
    quanto.quantize(container, weights=quanto.qint8)
    quanto.freeze(container)
    assert getattr(container[0].weight, "qtype", None) == quanto.qint8
    with pytest.raises(ValueError, match="BF16 or INT8 ConvRot"):
        dwd.replace_linear_weight(container[0], torch.zeros(8, 256))
