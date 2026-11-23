# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""A stand-in for wgp.py, for the CPU tests of the Speedups hook: generate_media keeps, in the same order, the lines and locals of
WanGP v20.12's generate_media that the hook looks for (the LoRA load, the full garbage collections, the unload after a generation,
the "output" event, the LoRA unload at the end of the task)."""
# ruff: noqa  (its lines keep wgp.py's own layout)

import gc

offload = None  # a stand-in mmgp.offload, set by the tests
offloadobj = None
LOG = []
MODEL = None
MODEL2 = None  # a second transformer holding the LoRAs of the first (sync_models_loras), when set


def get_gen_info(state):
    return state.setdefault("gen", {})


def generate_media(task, send_cmd, state=None, model_type=None, activated_loras=(), loras_multipliers="", windows=1, crash=False):
    gen = get_gen_info(state)
    model_def = {}
    trans_lora, trans2_lora = MODEL, MODEL2
    loras_selected = list(activated_loras)
    if len(loras_selected) > 0:
        offload.load_loras_into_model(trans_lora, loras_selected, [1.0] * len(loras_selected), activate_all_loras=True, pinnedLora=True, maxReservedLoras=-1)
        if trans2_lora is not None:
            offload.sync_models_loras(trans_lora, trans2_lora)
    gc.collect()
    repeat_no, total_generation, window_no, total_windows = 1, 1, windows, windows  # the last window of the only sample
    LOG.append(("generate", model_type, getattr(trans_lora, "_loras_active_adapters", None)))
    if crash:
        try:
            raise RuntimeError("generation failed")
        except RuntimeError:
            offloadobj.unload_all()
            send_cmd("error", "generation failed")
            return False
    # a model whose generation never swaps models stays loaded (parked: other models and programs reclaim its VRAM as they need it)
    if not (model_def.get("no_model_swap", False) and offloadobj.park()):
        offloadobj.unload_all()
    gc.collect()
    LOG.append(("record", f"clip_{task['id']}.mp4"))
    send_cmd("output")
    offload.unload_loras_from_model(trans_lora)
    return True
