# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""CPU tests of shared/utils/kept_loras.py (server_config "keep_loras_loaded"):

    python -m pytest shared/utils/tests/test_kept_loras.py -q
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "kept_loras.py"


@pytest.fixture
def kept(monkeypatch):
    """The module, with a recording stand-in for mmgp.offload."""
    calls = []
    fake_offload = types.SimpleNamespace(activate_loras=lambda model, names, multipliers: calls.append(("activate", model, names, multipliers)),
                                         sync_models_loras=lambda model, model2: calls.append(("sync", model, model2)))
    monkeypatch.setitem(sys.modules, "mmgp", types.SimpleNamespace(offload=fake_offload))
    spec = importlib.util.spec_from_file_location("kept_loras_under_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.calls = calls
    return module


class Model:
    pass


def loaded(model, signature, adapters=None, errors=None):
    """A model as offload.load_loras_into_model leaves it, with the signature WanGP records after a load."""
    model._loras_adapters = {"0": "a.safetensors", "1": "b.safetensors"} if adapters is None else adapters
    model._loras_errors = [] if errors is None else errors
    model._kept_lora_signature = signature
    return model


@pytest.fixture
def files(tmp_path):
    paths = []
    for name in ("a.safetensors", "b.safetensors"):
        path = tmp_path / name
        path.write_bytes(b"lora")
        paths.append(str(path))
    return paths


def test_same_task_reuses_the_loaded_loras(kept, files):
    model = Model()
    multipliers = [1.0, [0.5, 1.0]]
    signature = kept.kept_lora_signature(model, None, files, multipliers, "minimax_h3_fl2va_pruned", True)
    loaded(model, signature)
    assert kept.reuse_kept_loras(model, None, kept.kept_lora_signature(model, None, files, multipliers, "minimax_h3_fl2va_pruned", True), multipliers)
    assert kept.calls == [("activate", model, ["0", "1"], [1.0, [0.5, 1.0]])]  # multipliers and step counters reset, as after a load


def test_second_model_is_synced(kept, files):
    model, model2 = Model(), Model()
    signature = kept.kept_lora_signature(model, model2, files, [1.0], "t", True)
    loaded(model, signature)
    assert kept.reuse_kept_loras(model, model2, signature, [1.0])
    assert kept.calls[0] == ("activate", model, ["0", "1"], [1.0, 1.0])  # a missing multiplier is 1, as in a load
    assert kept.calls[1] == ("sync", model, model2)


@pytest.mark.parametrize("change", ["multiplier", "file", "model", "pinning", "base model"])
def test_any_difference_reloads(kept, files, change):
    model = Model()
    signature = kept.kept_lora_signature(model, None, files, [1.0, 1.0], "t", True)
    loaded(model, signature)
    other_model = Model()
    if change == "file":
        os.utime(files[0], (1, 1))  # same path, another modification time
    new = {
        "multiplier": lambda: kept.kept_lora_signature(model, None, files, [1.0, 0.8], "t", True),
        "file": lambda: kept.kept_lora_signature(model, None, files, [1.0, 1.0], "t", True),
        "model": lambda: kept.kept_lora_signature(other_model, None, files, [1.0, 1.0], "t", True),
        "pinning": lambda: kept.kept_lora_signature(model, None, files, [1.0, 1.0], "t", False),
        "base model": lambda: kept.kept_lora_signature(model, None, files, [1.0, 1.0], "u", True),
    }[change]()
    assert new != signature
    assert not kept.reuse_kept_loras(model, None, new, [1.0, 1.0])
    assert kept.calls == []


def test_unloaded_or_failed_loras_are_not_reused(kept, files):
    model = Model()
    signature = kept.kept_lora_signature(model, None, files, [1.0], "t", True)
    loaded(model, signature)
    model._loras_adapters = None  # offload.unload_loras_from_model ran (end of task without the setting, error recovery)
    assert not kept.reuse_kept_loras(model, None, signature, [1.0])
    loaded(model, signature, errors=[("a.safetensors", "bad key")])
    assert not kept.reuse_kept_loras(model, None, signature, [1.0])
    never_loaded = Model()
    assert not kept.reuse_kept_loras(never_loaded, None, signature, [1.0])
    assert kept.calls == []


def test_missing_file_gives_no_signature(kept, files, tmp_path):
    assert kept.kept_lora_signature(Model(), None, [str(tmp_path / "gone.safetensors")], [1.0], "t", True) is None


@pytest.mark.parametrize("enabled, aborted, more_windows, more_samples, next_model, expected", [
    (1, False, False, False, "t", True),    # the next queued task runs the same model: nothing unloaded in between
    (0, False, False, False, "t", False),   # the setting off: unloaded after every generation, as before
    (1, True, False, False, "t", False),    # an abort unloads
    (1, False, False, False, None, False),  # the end of the queue unloads
    (1, False, False, False, "u", False),   # another model next: unloaded for it
    (1, False, True, False, None, True),    # another window of this task
    (1, False, False, True, "u", True),     # another sample of this task
])
def test_models_stay_loaded(kept, enabled, aborted, more_windows, more_samples, next_model, expected):
    assert kept.models_stay_loaded(enabled, aborted, more_windows, more_samples, next_model, "t") is expected
