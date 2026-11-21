# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""CPU tests of pi-H3's schedule and adapter conversion (models/minimax_h3/pi_h3.py): no GPU, no weights.

    python -m pytest models/minimax_h3/tests/test_pi_h3.py -q
"""

import importlib
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]


def _load_pi_h3():
    """models.minimax_h3.pi_h3 without the package __init__ (which imports the whole handler)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    for name, folder in (("models", ROOT / "models"), ("models.minimax_h3", ROOT / "models" / "minimax_h3")):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(folder)]
            sys.modules[name] = package
    return importlib.import_module("models.minimax_h3.pi_h3")


pi_h3 = _load_pi_h3()
from models.minimax_h3.lora_affine import _load_affine_package  # noqa: E402


def _lakonlab_grid(steps, max_raw_t, final_step_size_scale, shift, min_raw_t=0.0):
    """FlowMapSDEScheduler.set_timesteps (LakonLab, Apache-2.0), float32 as there: the shifted sigmas without the final 0."""
    raw = np.linspace(max_raw_t, (max_raw_t - min_raw_t) * final_step_size_scale / (steps - 1 + final_step_size_scale) + min_raw_t,
                      steps, dtype=np.float32)
    raw = torch.from_numpy(raw).clamp(min=0)
    return (shift * raw / (1 + (shift - 1) * raw)).tolist()


def _map_shifted_sigma(sigma, source_shift, target_shift):  # pipeline._map_shifted_sigma, which derives the audio grid
    base_sigma = sigma / (source_shift + sigma * (1.0 - source_shift))
    return target_shift * base_sigma / (1.0 + (target_shift - 1.0) * base_sigma)


@pytest.mark.parametrize("steps", [2, 4])
def test_schedule_matches_lakonlab(steps):
    max_raw_t = {2: 0.93, 4: 0.96}[steps]
    assert pi_h3.video_sigmas(steps) == pytest.approx(_lakonlab_grid(steps, max_raw_t, 0.6, 12.0), abs=1e-6)
    assert pi_h3.audio_sigmas(steps) == pytest.approx(_lakonlab_grid(steps, max_raw_t, 0.6, 3.0), abs=1e-6)
    raw = pi_h3.raw_times(steps)
    segments = [a - b for a, b in zip(raw, raw[1:] + [0.0], strict=True)]
    assert segments[-1] == pytest.approx(0.6 * segments[0])  # the last step is 0.6 of the others
    assert all(math.isclose(segment, segments[0]) for segment in segments[:-1])


def test_known_values():
    assert pi_h3.raw_times(4) == pytest.approx([0.96, 0.96 - 0.8 / 3, 0.96 - 1.6 / 3, 0.16])
    assert pi_h3.video_sigmas(4) == pytest.approx([0.996540, 0.964451, 0.899297, 0.695652], abs=1e-6)
    assert pi_h3.raw_times(2) == pytest.approx([0.93, 0.34875])
    assert pi_h3.video_sigmas(2) == pytest.approx([0.993767, 0.865340], abs=1e-6)


@pytest.mark.parametrize("steps", [2, 4])
def test_pipeline_audio_mapping_gives_the_audio_grid(steps):
    video = torch.tensor(pi_h3.video_sigmas(steps) + [0.0], dtype=torch.float64)
    audio = _map_shifted_sigma(video, 12.0, 3.0)
    assert audio[:-1].tolist() == pytest.approx(pi_h3.audio_sigmas(steps), abs=1e-9)
    assert float(audio[-1]) == 0.0


def test_steps_and_settings_checks():
    with pytest.raises(ValueError):
        pi_h3.raw_times(3)
    pi_h3.check_settings(4, "euler", 12.0, 1, None, None, [None])
    for args in ((3, "euler", 12.0, 1, None, None, None), (4, "res_multistep", 12.0, 1, None, None, None), (4, "euler", 6.0, 1, None, None, None),
                 (4, "euler", 12.0, 2, None, None, None), (4, "euler", 12.0, 1, 0.8, None, None), (4, "euler", 12.0, 1, None, object(), None),
                 (4, "euler", 12.0, 1, None, None, [[0.9, 0.5]])):
        with pytest.raises(ValueError):
            pi_h3.check_settings(*args)


def test_adapter_steps_from_path():
    assert pi_h3.adapter_steps_from_path("/loras/minimax_h3/pi_h3_4step_adapter/diffusion_pytorch_model.safetensors") == 4
    assert pi_h3.adapter_steps_from_path("pi_h3_2step_adapter/diffusion_pytorch_model.safetensors|x") == 2
    assert pi_h3.adapter_steps_from_path("/loras/minimax_h3/pi_h3_4step_dwd_adapter/diffusion_pytorch_model.safetensors") is None
    assert pi_h3.adapter_steps_from_path("/loras/minimax_h3/some_lora.safetensors") is None


HIDDEN, BLOCK_ROWS, VIDEO, AUDIO = 4, 18 * 4, 6, 3


class FakeH3(nn.Module):
    """The modules pi-H3 replaces, at toy widths, on the official FL2VA rank-8 table."""

    def __init__(self, table=None, pdd_head=False):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.use_adaln_curves, self.endpoint_gate = True, None
        self.register_buffer("adaln_t_table", _load_affine_package("fl2va", 8)[0].clone() if table is None else table)
        block = nn.Module()
        block.adaln_proj = nn.Module()
        block.adaln_proj.linear = nn.Linear(8, BLOCK_ROWS)
        self.blocks = nn.ModuleList([block])
        self.final_layer = nn.Module()
        self.final_layer.norm = nn.RMSNorm(HIDDEN, eps=1e-5)
        self.final_layer.adaln_proj = nn.Module()
        self.final_layer.adaln_proj.linear = nn.Linear(8, 2 * HIDDEN)
        self.final_layer.video_out = nn.Sequential(nn.Linear(HIDDEN, VIDEO)) if pdd_head else nn.Linear(HIDDEN, VIDEO)
        self.final_layer.audio_out = nn.Linear(HIDDEN, AUDIO)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5)
            self.final_layer.norm.weight.copy_(torch.rand(HIDDEN, generator=generator) + 0.5)


def _adapter(seed=1):
    """A pi-H3 adapter as preprocess_loras sees it: WanGP names, AdaLN weights in the diffusers port's coordinates."""
    generator = torch.Generator().manual_seed(seed)
    tensors = {
        "blocks.0.adaln_proj.linear.weight": torch.randn(BLOCK_ROWS, 8, generator=generator),
        "final_layer.adaln_proj.linear.weight": torch.randn(2 * HIDDEN, 8, generator=generator),
        "final_layer.adaln_proj.linear.bias": torch.randn(2 * HIDDEN, generator=generator),
        "final_layer.norm.weight": torch.rand(HIDDEN, generator=generator) + 0.5,
        "final_layer.video_out.weight": torch.randn(VIDEO, HIDDEN, generator=generator),
        "final_layer.video_out.bias": torch.randn(VIDEO, generator=generator),
        "final_layer.audio_out.weight": torch.randn(AUDIO, HIDDEN, generator=generator),
        "final_layer.audio_out.bias": torch.randn(AUDIO, generator=generator),
        "blocks.0.attn.out_proj.lora_A.weight": torch.randn(2, 7, generator=generator),
        "blocks.0.attn.out_proj.lora_B.weight": torch.randn(5, 2, generator=generator),
    }
    return {key: value.half() for key, value in tensors.items()}


def _apply(model, state_dict):
    """The weights mmgp's diffs give at multiplier 1."""
    modules = dict(model.named_modules())
    weights = {}
    for key, value in state_dict.items():
        if key.endswith(".diff"):
            module = key[:-len(".diff")]
            weights[module + ".weight"] = modules[module].weight.detach() + value
        elif key.endswith(".diff_b"):
            module = key[:-len(".diff_b")]
            weights[module + ".bias"] = modules[module].bias.detach() + value
    return weights


def _final(x, norm_weight, adaln_weight, adaln_bias, coordinates, out_weight, out_bias):
    """FinalLayer: out(norm(x) w (1 + scale) + shift), [shift; scale] = AdaLN(c)."""
    shift, scale = (coordinates @ adaln_weight.T + adaln_bias).chunk(2, dim=-1)
    hidden = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * norm_weight
    return (hidden * (1 + scale) + shift) @ out_weight.T + out_bias


def test_conversion_is_exact():
    model, adapter = FakeH3(), _adapter()
    reference = {key: value.float() for key, value in adapter.items()}
    state = dict(adapter)
    assert pi_h3.convert_full_tensors(model, state) == 1 + 3 + 4  # block AdaLN; final AdaLN weight, bias and norm fold; 2 heads x (weight, bias)
    assert sorted(key for key in state if not key.endswith((".diff", ".diff_b"))) == ["blocks.0.attn.out_proj.lora_A.weight", "blocks.0.attn.out_proj.lora_B.weight"]
    assert torch.equal(state["blocks.0.attn.out_proj.lora_A.weight"], adapter["blocks.0.attn.out_proj.lora_A.weight"])  # LoRA factors untouched
    weights = _apply(model, state)
    signs = torch.tensor(pi_h3.ADALN_SIGNS)
    torch.testing.assert_close(weights["blocks.0.adaln_proj.linear.weight"], reference["blocks.0.adaln_proj.linear.weight"] * signs)
    for head in ("video_out", "audio_out"):
        for name in ("weight", "bias"):
            torch.testing.assert_close(weights[f"final_layer.{head}.{name}"], reference[f"final_layer.{head}.{name}"])
    assert "final_layer.norm.diff" not in state  # folded: the kernels read the norm weight directly

    # the folded final layer (base norm weight) gives the adapter's final layer (its norm weight, AdaLN in WanGP's coordinates)
    generator = torch.Generator().manual_seed(2)
    x, coordinates = torch.randn(16, HIDDEN, generator=generator), torch.randn(16, 8, generator=generator)
    expected = _final(x, reference["final_layer.norm.weight"], reference["final_layer.adaln_proj.linear.weight"] * signs,
                      reference["final_layer.adaln_proj.linear.bias"], coordinates,
                      reference["final_layer.video_out.weight"], reference["final_layer.video_out.bias"])
    actual = _final(x, model.final_layer.norm.weight.detach(), weights["final_layer.adaln_proj.linear.weight"],
                    weights["final_layer.adaln_proj.linear.bias"], coordinates,
                    weights["final_layer.video_out.weight"], weights["final_layer.video_out.bias"])
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_signs_turn_the_diffusers_coordinates_into_wangp_ones():
    """A block reading the diffusers port's coordinates c (c = WanGP's coordinates times the signs, as their tables are) with the
    adapter's weights computes what WanGP computes with the converted weights."""
    model, adapter = FakeH3(), _adapter()
    state = dict(adapter)
    pi_h3.convert_full_tensors(model, state)
    weights = _apply(model, state)
    signs = torch.tensor(pi_h3.ADALN_SIGNS)
    wangp_coordinates = model.adaln_t_table[::64]
    diffusers_coordinates = wangp_coordinates * signs
    torch.testing.assert_close(diffusers_coordinates @ adapter["blocks.0.adaln_proj.linear.weight"].float().T,
                               wangp_coordinates @ weights["blocks.0.adaln_proj.linear.weight"].T)


def test_other_loras_are_left_alone():
    model = FakeH3()
    lora = {"blocks.0.attn.out_proj.lora_A.weight": torch.zeros(2, 7), "blocks.0.attn.out_proj.lora_B.weight": torch.zeros(5, 2),
            "blocks.0.adaln_proj.linear.lora_A.weight": torch.zeros(2, 8), "final_layer.video_out.diff_b": torch.zeros(VIDEO),
            "blocks.0.norm1.weight": torch.zeros(HIDDEN)}  # a stray full tensor of another module: not a pi-H3 adapter
    state = dict(lora)
    assert pi_h3.convert_full_tensors(model, state) == 0
    assert state.keys() == lora.keys()


def test_refusals():
    with pytest.raises(ValueError, match="DWD"):
        pi_h3.convert_full_tensors(FakeH3(), {**_adapter(), "dwd_pixel_refiner.proj_out.weight": torch.zeros(3, 3)})
    with pytest.raises(ValueError, match="AdaLN table"):
        pi_h3.convert_full_tensors(FakeH3(table=_load_affine_package("fl2va", 8)[0] * 1.01), _adapter())
    with pytest.raises(ValueError, match="FL2VA Pruned"):
        pi_h3.convert_full_tensors(FakeH3(pdd_head=True), _adapter())  # PDD / per-step heads are not Linear
    with pytest.raises(ValueError, match="unsupported full tensors"):
        pi_h3.convert_full_tensors(FakeH3(), {**_adapter(), "blocks.0.attn.out_proj.weight": torch.zeros(5, 7)})
    full = FakeH3()
    full.use_adaln_curves = False
    with pytest.raises(ValueError, match="FL2VA Pruned"):
        pi_h3.convert_full_tensors(full, _adapter())
    with torch.no_grad():
        zero_norm = FakeH3()
        zero_norm.final_layer.norm.weight[0] = 0
    with pytest.raises(ValueError, match="zero weight"):
        pi_h3.convert_full_tensors(zero_norm, _adapter())


def test_selected_adapter():
    adapter = "/loras/minimax_h3/pi_h3_4step_adapter/diffusion_pytorch_model.safetensors"
    other = "/loras/minimax_h3/some_lora.safetensors"
    assert pi_h3.selected_adapter([other, adapter], {"phase1": [0.7, 1.0]}, 4)
    assert pi_h3.selected_adapter([adapter], {"phase1": [[1.0, 1.0, 1.0, 1.0]]}, 4)  # one multiplier per step
    assert not pi_h3.selected_adapter([other], {"phase1": [1.0]}, 4)
    assert not pi_h3.selected_adapter([], None, 4)
    with pytest.raises(ValueError, match="multiplier 1"):
        pi_h3.selected_adapter([adapter], {"phase1": [0.8]}, 4)
    with pytest.raises(ValueError, match="multiplier 1"):
        pi_h3.selected_adapter([adapter], {"phase1": [[1.0, 1.0, 0.0, 1.0]]}, 4)
    with pytest.raises(ValueError, match="runs 4 steps"):
        pi_h3.selected_adapter([adapter], {"phase1": [1.0]}, 2)
