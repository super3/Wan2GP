# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
# The DWD head follows LakonLab's pi-H3 release (Hansheng Chen, https://github.com/Lakonik/LakonLab v0.3.0, Apache License 2.0):
# MiniMaxH3DWDAdaLayerNormOut, SolH3DWDHead and the DWD decode step of its MiniMax H3 pipeline.
"""pi-H3 DWD (Decoding Without a Decoder): Lakonik's DWD adapters predict the video's pixels at the last sampling step from the
transformer's final hidden states, so the video VAE only encodes the keyframes.

A DWD adapter is a pi-H3 adapter (pi_h3.py: LoRA factors, replaced AdaLN and final-layer tensors) plus:
- its last four blocks' out_proj, fc1 and fc2 replaced entirely: merged into the transformer when it loads (in INT8 ConvRot when the
  transformer is), so that no full-rank difference runs at every step;
- dwd_norm_out: an affine-free LayerNorm of the final hidden states of the target video rows, modulated by linear(silu(c)), c the
  rank-8 AdaLN coordinates of the diffusers port (WanGP's times ADALN_SIGNS: the SiLU keeps the signs out of the weights);
- dwd_pixel_refiner: the last 8 layers of Sol-H3-VAE's decoder (components/dwd_refiner.py, on WanGP's Sol decoder), whose tokens come
  from feature_expand (each 32 x 32-pixel DiT token -> 2 x 2 Sol tokens) instead of the latent embedding; the first latent of each
  native 5-latent block has its own one-frame projection. Its output is Sol's ImageNet-normalized RGB, decoded as the Sol VAE's.

The released pipeline's other changes (timesteps counted as noise levels, keyframe anchors noised by hand) only adapt H3's conventions to
LakonLab's scheduler: they match WanGP's pipeline as it is."""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pi_h3 import ADALN_SIGNS, DWD_PREFIXES

SOL_LAYERS = 24  # Sol-H3-VAE's decoder: the refiner is its last layers
SOL_PATCH = 16  # pixels of a Sol token
DIT_PIXEL_PATCH = 32  # pixels of a DiT video token (1 x 2 x 2 latents of the 16x VAE)
NORM_CHUNK_ROWS = 4096
FINGERPRINT_VALUES = 251
_MERGED = re.compile(r"^transformer_blocks\.(\d+)\.(attn\.to_out\.0|ff\.net\.0\.proj|ff\.net\.2)\.weight$")
_WANGP_LINEAR = {"attn.to_out.0": "attn.out_proj", "ff.net.0.proj": "mlp.fc1", "ff.net.2": "mlp.fc2"}


def merged_linear_name(key):
    """The WanGP module a DWD adapter's fully replaced linear (diffusers name) applies to, None for any other tensor."""
    match = _MERGED.match(key)
    return None if match is None else f"blocks.{match.group(1)}.{_WANGP_LINEAR[match.group(2)]}"


def fingerprint(tensor):
    """Evenly spaced values of a tensor: what the LoRA conversion compares the merged linears with."""
    flat = tensor.detach().reshape(-1)
    return flat[::max(1, flat.numel() // FINGERPRINT_VALUES)][:FINGERPRINT_VALUES].float().cpu()


class MiniMaxH3DWDNormOut(nn.Module):
    """LakonLab's MiniMaxH3DWDAdaLayerNormOut: LayerNorm(x) (1 + scale) + shift, [scale; shift] = linear(silu(c))."""

    def __init__(self, hidden, rank, eps):
        super().__init__()
        self.eps = eps
        self.linear = nn.Linear(rank, 2 * hidden)

    def modulation(self, temb):
        """(scale, shift), FP32, of each timestep's rank-8 coordinates temb (WanGP's)."""
        coordinates = temb.float() * torch.tensor(ADALN_SIGNS, dtype=torch.float32, device=temb.device)
        return self.linear(F.silu(coordinates).to(self.linear.weight.dtype)).float().chunk(2, dim=-1)

    def forward(self, hidden, temb, spans, dtype=None):
        """The features of the rows hidden (rows, width); spans: (start, stop, timestep row) covering them."""
        scale, shift = self.modulation(temb)
        features = torch.empty(hidden.shape, dtype=dtype or hidden.dtype, device=hidden.device)
        for start, stop, row in spans:
            for first in range(start, stop, NORM_CHUNK_ROWS):
                last = min(stop, first + NORM_CHUNK_ROWS)
                normalized = F.layer_norm(hidden[first:last].float(), (hidden.shape[-1],), eps=self.eps)
                features[first:last] = normalized.mul_(1.0 + scale[row]).add_(shift[row]).to(features.dtype)
        return features


def expand_tokens(features, token_grid, expand, expansion, out):
    """out (t * H * W, D), the Sol tokens in (t, y, x) order, = feature_expand of the (t * h * w, width) features of a (t, h, w) grid:
    each DiT token gives expansion[0] x expansion[1] Sol tokens, its output columns in (e_h, e_w, D) order."""
    t, h, w = token_grid
    e_h, e_w = expansion
    dim = out.shape[-1]
    frames = features.view(t, h * w, -1)
    per_call = max(1, 8192 // (h * w))
    for first in range(0, t, per_call):
        last = min(t, first + per_call)
        expanded = expand(frames[first:last])
        out[first * h * w * e_h * e_w:last * h * w * e_h * e_w] = (
            expanded.view(last - first, h, w, e_h, e_w, dim).permute(0, 1, 3, 2, 4, 5).reshape(-1, dim))
        del expanded
    return out


def unpatchify_pixels(pixels, rows, width, channels, frames, patch):
    """(rows * width, channels * frames * patch * patch) projections -> (1, channels, frames, rows * patch, width * patch)."""
    return (pixels.view(1, rows, width, channels, frames, patch, patch).permute(0, 3, 4, 1, 5, 2, 6)
            .reshape(1, channels, frames, rows * patch, width * patch))


def head_tensors(state_dict):
    """A DWD adapter's head (dwd_norm_out, dwd_pixel_refiner), named as WanGP's modules: the refiner's layers as the Sol decoder's."""
    head = {}
    for key, value in state_dict.items():
        if key.startswith(DWD_PREFIXES):
            head[key.replace("dwd_pixel_refiner.blocks.", "dwd_pixel_refiner.transformer_blocks.", 1)] = value
    if any(key.startswith("dwd_linear_head.") for key in head):
        raise ValueError("pi-H3 DWD: this adapter has a linear pixel head, which pi-H3's Sol refiner does not use")
    return head


def refiner_shape(head):
    """(DiT width, rank of the AdaLN coordinates, Sol layers) of a DWD head."""
    width = head["dwd_pixel_refiner.feature_expand.weight"].shape[1]
    rank = head["dwd_norm_out.linear.weight"].shape[1]
    layers = 1 + max(int(key.split(".")[2]) for key in head if key.startswith("dwd_pixel_refiner.transformer_blocks."))
    return width, rank, layers


def replace_linear_weight(module, weight, device=None):
    """module.weight = weight, kept in the module's format: BF16 / FP16, or INT8 ConvRot requantized as WanGP's ConvRot converter does."""
    current = module.weight
    if tuple(weight.shape) != tuple(current.shape):
        raise ValueError(f"pi-H3 DWD: a replaced linear has shape {tuple(weight.shape)}, the model's {tuple(current.shape)}")
    qtype = getattr(current, "qtype", None)
    if qtype is None and type(current) in (torch.Tensor, nn.Parameter) and current.dtype in (torch.bfloat16, torch.float16, torch.float32):
        module.weight = nn.Parameter(weight.to(current.dtype).contiguous(), requires_grad=False)
        return
    from shared.qtypes.int8_convrot import _QINT8_CONVROT_QTYPE, Int8ConvRotWeightTensor

    group = int(getattr(module, "_convrot_group_size", 0) or 0)
    if qtype != _QINT8_CONVROT_QTYPE or group <= 0:
        raise ValueError("pi-H3 DWD runs on the BF16 or INT8 ConvRot transformer of MiniMax H3 FL2VA Pruned: its last four blocks are "
                         f"replaced when it loads, which this format ({type(current).__name__}) does not allow")
    from shared.convert.convrot import quantize_weight

    data, scale = quantize_weight(weight, group, device or ("cuda" if torch.cuda.is_available() else "cpu"))
    module.weight = nn.Parameter(Int8ConvRotWeightTensor.create(qtype, current.axis, current.size(), current.stride(), data,
                                                                scale.reshape(current._scale.shape).to(current._scale.dtype),
                                                                activation_qtype=current.activation_qtype, compute_dtype=current.dtype),
                                 requires_grad=False)


def merge_linears(model, tensors, device=None):
    """Merge a DWD adapter's fully replaced linears (diffusers names) into the model. Returns their fingerprints by WanGP weight name."""
    modules, fingerprints = dict(model.named_modules()), {}
    for key, weight in tensors.items():
        name = merged_linear_name(key)
        module = modules.get(name)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"pi-H3 DWD adapters apply to MiniMax H3 FL2VA Pruned: this model has no linear {name}")
        fingerprints[name + ".weight"] = fingerprint(weight)
        if name.endswith(".mlp.fc1"):  # diffusers' fc1 rows are WanGP's halves swapped, as preprocess_loras swaps a LoRA's
            weight = torch.cat(weight.chunk(2, dim=0)[::-1])
        replace_linear_weight(module, weight, device)
    return fingerprints


def install(model, path, steps, device=None):
    """Make the transformer model a pi-H3 DWD one with the adapter at path: merge its replaced linears and attach its head
    (model.dwd_norm_out, model.dwd_pixel_refiner). The adapter's LoRA part is loaded later as a LoRA (pi_h3.convert_full_tensors)."""
    from safetensors import safe_open

    from .components.dwd_refiner import MiniMaxH3DWDRefiner

    with safe_open(path, framework="pt", device="cpu") as reader:
        keys = list(reader.keys())
        if not any(key.startswith(DWD_PREFIXES) for key in keys):
            raise ValueError(f"pi-H3 DWD: {path} is not a DWD adapter")
        fingerprints = merge_linears(model, {key: reader.get_tensor(key) for key in keys if merged_linear_name(key)}, device)
        head = head_tensors({key: reader.get_tensor(key) for key in keys if key.startswith(DWD_PREFIXES)})
    width, rank, layers = refiner_shape(head)
    if width != model.hidden_size or not getattr(model, "use_adaln_curves", False) or model.adaln_t_table.shape[1] != rank:
        raise ValueError("pi-H3 DWD adapters are trained on MiniMax H3 FL2VA Pruned (rank-8 AdaLN): select that model")
    dtype = model.dtype or torch.bfloat16
    norm_out = MiniMaxH3DWDNormOut(width, rank, model.final_layer.norm.eps)
    refiner = MiniMaxH3DWDRefiner(width, layers, (DIT_PIXEL_PATCH // SOL_PATCH,) * 2)
    fp32 = (norm_out, refiner.norm_out, refiner.proj_out, refiner.first_frame_proj_out)  # Sol's output head, and the AdaLN of 8 inputs
    kept = {id(parameter) for module in fp32 for parameter in module.parameters()}
    for parameter in refiner.parameters():  # parameters only: the rotary frequencies (a buffer) stay FP32
        if id(parameter) not in kept:
            parameter.data = parameter.data.to(dtype)
    for module in fp32:
        module._lock_dtype = torch.float32
    modules = {"dwd_norm_out": norm_out, "dwd_pixel_refiner": refiner}
    for prefix, module in modules.items():
        own = {key[len(prefix) + 1:]: value for key, value in head.items() if key.startswith(prefix + ".")}
        expected = module.state_dict()
        if own.keys() != expected.keys():
            raise ValueError(f"pi-H3 DWD: unexpected {prefix} tensors {sorted(own.keys() ^ expected.keys())[:6]}")
        module.load_state_dict({key: value.to(expected[key].dtype) for key, value in own.items()})
        module.requires_grad_(False).eval()
        setattr(model, prefix, module)
    model.pi_h3_dwd = {"steps": int(steps), "fingerprints": fingerprints}
    return len(fingerprints)


def decode(refiner, features, token_grid, uint8_rounding=None, interrupt=None):
    """The video (1, 3, frames, 16H, 16W) on the CPU of the DWD features (t * h * w, width) of a (t, h, w) token grid: in [-1, 1], or
    uint8 converted on the GPU as the Sol VAE converts its frames ("truncate" as WGP, "round")."""
    from mmgp import transfer

    from shared.utils.phase_progress import PhaseProgress

    from .video_vae import IMAGENET_MEAN, IMAGENET_STD

    t, h, w = token_grid
    e_h, e_w = refiner.expansion
    frames, height, width = refiner.output_frames(t), h * e_h * SOL_PATCH, w * e_w * SOL_PATCH
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=features.device).view(1, 3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=features.device).view(1, 3, 1, 1, 1)
    decoded = torch.empty(1, 3, frames, height, width, dtype=torch.float32 if uint8_rounding is None else torch.uint8, device="cpu")
    position = 0

    def emit(pixels, row, last):
        nonlocal position
        part = pixels.float().mul_(std).add_(mean).clamp_(0.0, 1.0).mul_(2.0).sub_(1.0)
        if uint8_rounding == "round":
            part = part.add_(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)
        elif uint8_rounding == "truncate":
            part = part.to(torch.float16).sub_(-1).mul_(127.5).clamp_(0, 255).to(torch.uint8)
        writer.write(decoded[:, :, position:position + part.shape[2], row:row + part.shape[-2]], part)
        if last:
            position += part.shape[2]

    with PhaseProgress(refiner.steps(t)) as progress, transfer.RamWriter("decoded video", depth=64, max_bytes=256 << 20) as writer:
        refiner.progress, refiner.interrupt = progress, interrupt
        try:
            refiner(features, token_grid, emit)
        finally:
            refiner.progress = refiner.interrupt = None
    if position != frames:
        raise RuntimeError(f"pi-H3 DWD decoded {position} frames, expected {frames}")
    return decoded
