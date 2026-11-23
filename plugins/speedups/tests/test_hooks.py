# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""CPU tests of the Speedups hook, on a stand-in wgp.py (fake_wgp.py) whose generate_media has the lines and locals of WanGP
v20.12's that the hook looks for, and a stand-in mmgp.offload:

    python -m pytest plugins/speedups/tests -q
"""

import gc
import importlib
import os
import sys
import types
from pathlib import Path

import pytest

PLUGINS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGINS))
sys.path.insert(0, str(PLUGINS.parent))


class FakeOffload:
    """mmgp.offload: LoRA loads, syncs and unloads, recorded."""

    def __init__(self):
        self.calls = []

    def load_loras_into_model(self, model, lora_path, lora_multi=None, activate_all_loras=True, check_only=False, ignore_model_variations=False,
                              pinnedLora=False, maxReservedLoras=-1, split_linear_modules_map=None, preprocess_sd=None, verboseLevel=-1):
        self.calls.append(("load", tuple(os.path.basename(p) for p in lora_path), check_only))
        if not check_only:
            model._loras_adapters = {str(index): path for index, path in enumerate(lora_path)}
            model._loras_active_adapters = list(model._loras_adapters)

    def unload_loras_from_model(self, model):
        self.calls.append(("unload",))
        model._loras_adapters = model._loras_active_adapters = None

    def sync_models_loras(self, model, model2):
        self.calls.append(("sync",))

    def profile(self, pipe, **kwargs):
        self.calls.append(("profile", dict(kwargs.get("coTenantsMap") or {})))
        return FakeOffloadObj()


class FakeOffloadObj:
    def __init__(self, log=None):
        self.unloads = 0
        self.refreshed = []
        self.log = log if log is not None else []

    def park(self):
        return False

    def unload_all(self):
        self.unloads += 1
        self.log.append(("unload_all",))

    def refresh_loras(self, model):
        self.refreshed.append(model)


class Model:
    pass


@pytest.fixture
def env(monkeypatch):
    """The stand-in wgp.py with the hook installed; run(tasks) runs a queue as WanGP's runners do (the running task stays first in
    gen["queue"])."""
    fake = importlib.import_module("speedups.tests.fake_wgp")
    hooks = importlib.import_module("speedups.hooks")
    offload = FakeOffload()
    fake.offload, fake.offloadobj, fake.MODEL, fake.MODEL2 = offload, FakeOffloadObj(fake.LOG), Model(), None
    fake.LOG.clear()
    assert hooks.install(fake.__dict__, hooks.Settings()) == []
    collects = []
    monkeypatch.setattr(gc, "collect", lambda *a, **k: collects.append(1) or 0)
    state, events = {}, []

    def call(task, **options):
        """One task, as a runner calls generate_media (the task stays in gen["queue"] until it ends)."""
        return fake.generate_media(task, lambda cmd, data=None: events.append((cmd, data)), state=state, **task["params"], **options)

    def queue(tasks):
        fake.get_gen_info(state)["queue"] = list(tasks)

    def run(tasks, **options):
        queue(tasks)
        results = []
        for task in tasks:
            results.append(call(task, **options))
            state["gen"]["queue"].remove(task)
        return results

    yield types.SimpleNamespace(wgp=fake, hooks=hooks, offload=offload, run=run, call=call, queue=queue, events=events, collects=collects,
                                state=state)
    hooks.uninstall()


def task(number, model_type="h3", loras=(), multipliers=""):
    return {"id": number, "params": {"model_type": model_type, "activated_loras": list(loras), "loras_multipliers": multipliers}}


# --- keep the model loaded --------------------------------------------------------------------------------------------------------

def test_model_kept_between_tasks_of_the_same_model(env):
    env.hooks.configure(env.hooks.Settings(keep_model_loaded=True))
    env.run([task(1), task(2), task(3, model_type="other")])
    # 1 -> 2 (same model): kept, with no full garbage collection after 1's generation nor before 2's; 2 -> 3 (another model) and
    # 3 (the end of the queue): unloaded right after their generation
    assert env.wgp.offloadobj.unloads == 2
    assert len(env.collects) == 4  # 6 in generate_media, 2 skipped
    # the last task's models leave right after its generation, before its output is recorded, as without the setting
    last = env.wgp.LOG[env.wgp.LOG.index(("generate", "other", None)):]
    assert last.index(("unload_all",)) < last.index(("record", "clip_3.mp4"))
    assert [cmd for cmd, _ in env.events] == ["output"] * 3


def test_a_changed_queue_unloads_at_the_end_of_the_task(env):
    env.hooks.configure(env.hooks.Settings(keep_model_loaded=True))
    first = task(1)
    env.queue([first, task(2)])
    env.call(first)
    assert env.wgp.offloadobj.unloads == 0 and env.hooks.STATE["kept_model_type"] == "h3"
    env.queue([])  # the queue changed: a task of another model runs instead, and the kept model is not its own
    env.call(task(9, model_type="other"))
    assert env.wgp.offloadobj.unloads == 1 and len(env.collects) == 3  # only 1's collection after its generation was skipped


def test_model_unloaded_after_an_abort_or_a_failure(env):
    env.hooks.configure(env.hooks.Settings(keep_model_loaded=True))
    env.state.setdefault("gen", {})["abort"] = True
    env.run([task(1), task(2)])
    assert env.wgp.offloadobj.unloads == 2
    env.state["gen"]["abort"] = False
    env.wgp.offloadobj.unloads = 0
    env.run([task(3), task(4)], crash=True)
    assert env.wgp.offloadobj.unloads == 2  # in the generation's except block: never deferred
    assert env.hooks.STATE["kept_model_type"] is None


def test_off_unloads_after_every_generation(env):
    env.run([task(1), task(2)])
    assert env.wgp.offloadobj.unloads == 2 and len(env.collects) == 4


def test_transformers_become_co_tenants_when_loaded_with_the_setting(env):
    env.hooks.configure(env.hooks.Settings(keep_model_loaded=True))
    env.wgp.offload.profile({"transformer": 1, "transformer2": 2, "vae": 3}, coTenantsMap={"tiny_vae": "*"})
    env.wgp.offload.profile({"text_encoder": 1}, coTenantsMap={})
    env.hooks.configure(env.hooks.Settings())
    env.wgp.offload.profile({"transformer": 1}, coTenantsMap={})
    assert [call[1] for call in env.offload.calls if call[0] == "profile"] == [{"tiny_vae": "*", "transformer": "*", "transformer2": "*"}, {}, {}]


def test_loras_loaded_into_a_kept_model_refresh_its_gpu_blocks(env, tmp_path):
    env.hooks.configure(env.hooks.Settings(keep_model_loaded=True))
    env.wgp.MODEL2 = Model()
    a, b = str(tmp_path / "a.safetensors"), str(tmp_path / "b.safetensors")
    env.run([task(1, loras=[a]), task(2, loras=[b]), task(3, loras=[b])])
    # task 1 loaded the model: nothing to refresh; tasks 2 and 3 load their LoRAs into the model the task before left in VRAM, and
    # into the second transformer that shares them
    assert env.wgp.offloadobj.refreshed == [env.wgp.MODEL, env.wgp.MODEL2] * 2


def test_a_check_only_load_or_an_unkept_model_refreshes_nothing(env, tmp_path):
    lora = str(tmp_path / "a.safetensors")
    env.run([task(1, loras=[lora]), task(2, loras=[lora])])  # the setting off: the model is loaded again for each task
    env.hooks.configure(env.hooks.Settings(keep_model_loaded=True))
    first = task(3)
    env.queue([first, task(4)])
    env.call(first)
    env.state["gen"]["queue"].remove(first)
    call = env.hooks.Call(task(4), env.state["gen"], "h3")
    call.model_kept = True
    env.hooks._local.call = call
    try:
        env.wgp.offload.load_loras_into_model(env.wgp.MODEL, [lora], check_only=True)
    finally:
        env.hooks._local.call = None
    assert env.wgp.offloadobj.refreshed == []


# --- install ------------------------------------------------------------------------------------------------------------------------

def test_uninstall_restores_wgp(env):
    originals = {name: env.hooks.STATE["originals"][name] for name in ("generate_media", "offload", "gc")}
    env.hooks.uninstall()
    for name, value in originals.items():
        assert env.wgp.__dict__[name] is value
    assert env.hooks.install(env.wgp.__dict__) == []


def test_settings_of_an_earlier_version_keep_only_the_model_setting():
    hooks = importlib.import_module("speedups.hooks")
    old = {"keep_model_loaded": True, "keep_loras_loaded": True, "background_save": True}
    assert hooks.Settings.from_dict(old) == hooks.Settings(keep_model_loaded=True)


def test_a_wgp_without_the_hooked_names_is_left_alone():
    hooks = importlib.import_module("speedups.hooks")
    if hooks.STATE["installed"]:
        hooks.uninstall()
    assert "generate_media" in hooks.install({"offload": object(), "gc": gc})
    assert not hooks.STATE["installed"]
