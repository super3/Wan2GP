# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Keep a task's LoRAs loaded for the next task that selects the same ones (server_config "keep_loras_loaded").

WanGP otherwise unloads every LoRA when a task ends and loads them again when the next one starts: each file is read, preprocessed
(key conversion, adapter-specific conversions) and pinned again. For back-to-back generations with the same LoRAs, as a real-time
loop makes, that is per-clip work with nothing to show for it."""
import os

from mmgp import offload


def kept_lora_signature(trans_lora, trans2_lora, lora_paths, lora_multipliers, base_model_type, pinned):
    """What makes a task's LoRAs the ones already loaded (server_config "keep_loras_loaded"): the same model instances, files
    (path, size, modification time), multipliers and pinning."""
    try:
        files = tuple((os.path.abspath(path), os.path.getsize(path), os.path.getmtime(path)) for path in lora_paths)
    except OSError:
        return None
    return (id(trans_lora), id(trans2_lora), base_model_type, files, repr(lora_multipliers), bool(pinned))


def reuse_kept_loras(trans_lora, trans2_lora, signature, lora_multipliers):
    """With server_config "keep_loras_loaded", a task whose LoRAs match the ones the last task left loaded reuses them instead of
    reading, preprocessing and pinning the files again: the adapters are activated again with the task's multipliers, as
    offload.load_loras_into_model does at the end of a load. False when they do not match, or were unloaded since."""
    adapters = getattr(trans_lora, "_loras_adapters", None)
    if not adapters or getattr(trans_lora, "_kept_lora_signature", None) != signature or getattr(trans_lora, "_loras_errors", None):
        return False
    names = list(adapters)
    multipliers = [lora_multipliers[index] if index < len(lora_multipliers) else 1.0 for index in range(len(names))]
    offload.activate_loras(trans_lora, names, multipliers)
    if trans2_lora is not None:
        offload.sync_models_loras(trans_lora, trans2_lora)
    return True


def models_stay_loaded(enabled, aborted, more_windows, more_samples, next_model_type, model_type):
    """server_config "keep_model_loaded": whether a generation's models stay in VRAM for what runs next (no unload, no reload of their
    blocks, no full garbage collection in between). Only when this model runs again right after: another window or sample of this task,
    or the next queued task of the same model type; never after an abort, and the end of a queue unloads as before."""
    return bool(enabled) and not aborted and (bool(more_windows) or bool(more_samples) or next_model_type == model_type)
