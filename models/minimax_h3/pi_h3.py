# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
# The sampling grid follows LakonLab's FlowMapSDEScheduler.set_timesteps (Hansheng Chen, https://github.com/Lakonik/LakonLab,
# Apache License 2.0) with pi-H3's published settings (https://huggingface.co/Lakonik/pi-H3).
"""pi-H3: Lakonik's 2-step and 4-step pi-Flow v2 adapters of the pruned MiniMax H3 FL2VA model.

An adapter is a LoRA (rank 64 on every block's attention output and feed-forward linears) plus replaced tensors: every block's
pruned AdaLN projection and the final layer (AdaLN, norm, video and audio heads). WanGP loads it as a LoRA: preprocess_loras turns the
replaced tensors of a diffusers-format file into exact differences from the loaded base (`.diff` / `.diff_b`), so the adapter applies
at multiplier 1.

- AdaLN coordinates: the adapters were trained on the diffusers port of the pruned model (multimodalart/MiniMax-H3-Pruned), whose
  rank-8 timestep coordinates are WanGP's official FL2VA ones with three axes negated (its table and AdaLN weights times ADALN_SIGNS
  match WanGP's to float rounding).
- Final norm: the kernels read the final RMSNorm's weight directly, so the adapter's new weight w' is folded into the final AdaLN
  instead: norm(x) w' (1 + scale) + shift == norm(x) w (1 + r scale + r - 1) + shift with r = w' / w, a change of the scale rows only.
- Sampling: the student predicts the flow velocity of a one-point DX policy, which is a plain Euler step. Its grid is the raw times
  linspace(max_raw_t, max_raw_t * f / (steps - 1 + f), steps) (f = 0.6: the last step is 0.6 of the others), then 0, shifted by 12
  for the video and 3 for the audio; the noise starts at the first noise level.

Not supported yet, until LakonLab publishes pi-H3's inference code: the DWD adapters (pixels from the last step without the VAE
decoder: their refiner's attention layout is not in the weights) and the pipeline's latent variance calibrations."""

import math
import os

import torch

PI_H3_STEPS = (2, 4)
PI_H3_MAX_RAW_T = {2: 0.93, 4: 0.96}
PI_H3_FINAL_STEP_SIZE_SCALE = 0.6
PI_H3_VIDEO_SHIFT = 12.0
PI_H3_AUDIO_SHIFT = 3.0
ADALN_SIGNS = (1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0)  # multimodalart/MiniMax-H3-Pruned coordinates -> WanGP's official FL2VA ones
ADAPTER_FOLDERS = {2: "pi_h3_2step_adapter", 4: "pi_h3_4step_adapter"}
DWD_PREFIXES = ("dwd_pixel_refiner.", "dwd_norm_out.", "dwd_linear_head.")
_LORA_MARKERS = (".lora_", ".lora.", ".alpha", ".diff", ".diff_b", ".dora_scale", ".lokr_")
_BLOCK_ADALN = ".adaln_proj.linear.weight"
_FINAL = "final_layer."


def raw_times(steps):
    """The raw (unshifted) times of a pi-H3 schedule, without the final 0."""
    steps = int(steps)
    if steps not in PI_H3_MAX_RAW_T:
        raise ValueError(f"pi-H3 runs {' or '.join(map(str, PI_H3_STEPS))} steps, got {steps}")
    top, scale = PI_H3_MAX_RAW_T[steps], PI_H3_FINAL_STEP_SIZE_SCALE
    last = top * scale / (steps - 1 + scale)
    return [top + (last - top) * index / (steps - 1) for index in range(steps)]


def _shift(sigma, shift):
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def video_sigmas(steps):
    """The shifted video noise levels of a pi-H3 schedule, without the final 0 (the pipeline's explicit phase schedule)."""
    return [_shift(sigma, PI_H3_VIDEO_SHIFT) for sigma in raw_times(steps)]


def audio_sigmas(steps):
    return [_shift(sigma, PI_H3_AUDIO_SHIFT) for sigma in raw_times(steps)]


def is_full_tensor(key):
    return not any(marker in key for marker in _LORA_MARKERS)


def adapter_steps_from_path(path):
    """The step count a pi-H3 adapter's folder names (pi_h3_2step_adapter / pi_h3_4step_adapter), None when it names none."""
    folder = os.path.basename(os.path.dirname(str(path).split("|", 1)[0]))
    for steps, name in ADAPTER_FOLDERS.items():
        if folder.startswith(name):
            return steps
    return None


def selected_adapter(loras_selected, loras_slists, steps):
    """True when a pi-H3 adapter (its folder: pi_h3_2step_adapter / pi_h3_4step_adapter) is among the selected LoRAs; it must run at
    multiplier 1 (phase 1) with its own step count."""
    for index, lora in enumerate(loras_selected or ()):
        adapter_steps = adapter_steps_from_path(lora)
        if adapter_steps is None:
            continue
        multipliers = 1.0 if loras_slists is None else loras_slists["phase1"][index]
        if any(float(value) != 1.0 for value in (multipliers if isinstance(multipliers, list) else [multipliers])):
            raise ValueError("A pi-H3 adapter runs at multiplier 1 (its replaced tensors are exact differences from the base)")
        if adapter_steps != int(steps):
            raise ValueError(f"This pi-H3 adapter runs {adapter_steps} steps, not {steps}")
        return True
    return False


def is_adapter(state_dict):
    """A pi-H3 adapter (WanGP names): replaced AdaLN or final-layer tensors next to its LoRA factors, or a DWD head."""
    return any(key.startswith(DWD_PREFIXES) or is_full_tensor(key) and (key.startswith(_FINAL) or key.startswith("blocks.") and key.endswith(_BLOCK_ADALN))
               for key in state_dict)


def convert_full_tensors(model, state_dict):
    """Turn a pi-H3 adapter's replaced tensors (WanGP names, after the diffusers key conversion) into exact differences from `model`'s
    loaded weights, in place. Returns the number of tensors converted (0: not a pi-H3 adapter, left unchanged)."""
    if not is_adapter(state_dict):
        return 0
    full = {key: state_dict[key] for key in state_dict if is_full_tensor(key)}
    dwd = sorted(key for key in full if key.startswith(DWD_PREFIXES))
    if dwd:
        raise ValueError("This is a pi-H3 DWD adapter (it predicts pixels without the VAE decoder): WanGP does not support DWD yet, its "
                         "inference code is not published. Use pi_h3_2step_adapter or pi_h3_4step_adapter.")
    unexpected = sorted(key for key in full if not (key.startswith("blocks.") and key.endswith(_BLOCK_ADALN) or key.startswith(_FINAL)))
    if unexpected:
        raise ValueError(f"MiniMax H3 LoRA: unsupported full tensors {unexpected[:4]}{' ...' if len(unexpected) > 4 else ''}")
    if not getattr(model, "use_adaln_curves", False) or model.adaln_t_table.shape[1] != len(ADALN_SIGNS) or model.endpoint_gate is not None:
        raise ValueError("pi-H3 adapters are trained on MiniMax H3 FL2VA Pruned (rank-8 AdaLN): select that model")
    from .lora_affine import _load_affine_package

    official = _load_affine_package("fl2va", len(ADALN_SIGNS))[0]
    if not torch.allclose(model.adaln_t_table.detach().float().cpu(), official, rtol=0, atol=1e-6):
        raise ValueError("pi-H3 adapters need the official MiniMax H3 FL2VA Pruned checkpoint: this one has another AdaLN table")
    signs = torch.tensor(ADALN_SIGNS, dtype=torch.float32)
    modules = dict(model.named_modules())

    def current(module_name, name):
        module = modules.get(module_name)
        if not isinstance(module, (torch.nn.Linear, torch.nn.RMSNorm)):
            raise ValueError(f"pi-H3 adapters apply to MiniMax H3 FL2VA Pruned: this model's {module_name} is a {type(module).__name__}")
        tensor = getattr(module, name)
        if tensor is None or not tensor.dtype.is_floating_point:
            raise ValueError(f"pi-H3: {module_name}.{name} must be an unquantized floating-point tensor")
        return tensor.detach().float().cpu()

    def put_diff(module_name, name, new):
        old = current(module_name, name)
        if new.shape != old.shape:
            raise ValueError(f"pi-H3: {module_name}.{name} has shape {tuple(new.shape)}, the model {tuple(old.shape)}")
        state_dict[f"{module_name}.{'diff' if name == 'weight' else 'diff_b'}"] = (new.float() - old).contiguous()

    converted = 0
    for key in [key for key in full if key.startswith("blocks.")]:
        put_diff(key[:-len(".weight")], "weight", state_dict.pop(key).float() * signs)
        converted += 1
    final = {key[len(_FINAL):]: state_dict.pop(key) for key in [key for key in full if key.startswith(_FINAL)]}
    adaln = "final_layer.adaln_proj.linear"
    weight, bias = final.pop("adaln_proj.linear.weight", None), final.pop("adaln_proj.linear.bias", None)
    norm_weight = final.pop("norm.weight", None)
    if norm_weight is not None or weight is not None or bias is not None:
        weight = current(adaln, "weight") if weight is None else weight.float() * signs
        bias = current(adaln, "bias") if bias is None else bias.float()
        if norm_weight is not None:  # the final norm's new weight, folded into the AdaLN scale rows (the second half)
            ratio = norm_weight.float() / current("final_layer.norm", "weight")
            if not bool(torch.isfinite(ratio).all()):
                raise ValueError("pi-H3: the final norm cannot be folded into the AdaLN (a zero weight in the base norm)")
            hidden = ratio.numel()
            weight, bias = weight.clone(), bias.clone()
            weight[hidden:] *= ratio[:, None]
            bias[hidden:] = bias[hidden:] * ratio + ratio - 1.0
            converted += 1
        put_diff(adaln, "weight", weight)
        put_diff(adaln, "bias", bias)
        converted += 2
    for head in ("video_out", "audio_out"):
        for name in ("weight", "bias"):
            tensor = final.pop(f"{head}.{name}", None)
            if tensor is not None:
                put_diff(_FINAL + head, name, tensor)
                converted += 1
    if final:
        raise ValueError(f"MiniMax H3 LoRA: unsupported final-layer tensors {sorted(final)}")
    return converted


def check_settings(steps, sample_solver, shift, guidance_phases, starting_sigma, cache, custom_sigmas):
    if int(steps) not in PI_H3_STEPS:
        raise ValueError(f"pi-H3 runs {' or '.join(map(str, PI_H3_STEPS))} steps on its own schedule, got {steps}")
    if sample_solver != "euler" or not math.isclose(float(shift), PI_H3_VIDEO_SHIFT) or int(guidance_phases) != 1 or starting_sigma is not None or cache is not None:
        raise ValueError("pi-H3 samples one phase with the Euler sampler and flow shift 12, without partial-schedule refinement or step skipping")
    if custom_sigmas and any(sigmas is not None for sigmas in custom_sigmas):
        raise ValueError("pi-H3 runs its own schedule, without custom sigmas")
