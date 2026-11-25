# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""CPU tests of the Speedups hooks, on a stand-in wgp.py (fake_wgp.py) whose generate_media has the lines and locals of WanGP
v20.10's that the hooks look for, and a stand-in mmgp.offload:

    python -m pytest plugins/speedups/tests -q
"""

import gc
import importlib
import os
import sys
import time
import types
from pathlib import Path

import pytest

PLUGINS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGINS))
sys.path.insert(0, str(PLUGINS.parent))


class FakeOffload:
    """mmgp.offload: LoRA loads, activations and unloads, recorded."""

    def __init__(self):
        self.calls = []

    def load_loras_into_model(self, model, lora_path, lora_multi=None, activate_all_loras=True, check_only=False, ignore_model_variations=False,
                              pinnedLora=False, maxReservedLoras=-1, split_linear_modules_map=None, preprocess_sd=None, verboseLevel=-1):
        self.calls.append(("load", tuple(os.path.basename(p) for p in lora_path)))
        model._loras_adapters = {str(index): path for index, path in enumerate(lora_path)}
        model._loras_active_adapters = list(model._loras_adapters)
        model._loras_errors = []

    def activate_loras(self, model, lora_nos, lora_multi=None):
        self.calls.append(("activate", tuple(lora_nos), tuple(lora_multi or ())))
        model._loras_active_adapters = list(lora_nos)

    def unload_loras_from_model(self, model):
        self.calls.append(("unload",))
        model._loras_adapters = model._loras_active_adapters = model._loras_errors = None

    def sync_models_loras(self, model, model2):
        pass

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
def env(tmp_path, monkeypatch):
    """The stand-in wgp.py with the hooks installed; returns (wgp, hooks, offload, run) where run(tasks) runs a queue as WanGP's
    runners do (the running task stays first in gen["queue"])."""
    fake = importlib.import_module("speedups.tests.fake_wgp")
    stub = types.ModuleType("shared.utils.video_continuation")
    stub.ContinuedVideo = fake.ContinuedVideo
    monkeypatch.setitem(sys.modules, "shared.utils.video_continuation", stub)
    hooks = importlib.import_module("speedups.hooks")
    offload = FakeOffload()
    fake.offload, fake.offloadobj, fake.MODEL = offload, FakeOffloadObj(fake.LOG), Model()
    fake.LOG.clear()
    hooks.saver.history.clear()
    fake.SETTINGS.update(encode_s=0.0, fail_encode=False)
    assert hooks.install(fake.__dict__, hooks.Settings()) == []
    collects = []
    monkeypatch.setattr(gc, "collect", lambda *a, **k: collects.append(1) or 0)
    state, events = {}, []

    def call(task, **options):
        """One task, as a runner calls generate_media (the task stays in gen["queue"] until it ends)."""
        return fake.generate_media(task, lambda cmd, data=None: events.append((cmd, data)), state=state, out_dir=str(tmp_path), **task["params"], **options)

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
                                state=state, dir=tmp_path)
    hooks.saver.wait()
    hooks.uninstall()


def task(number, model_type="h3", loras=(), multipliers=""):
    return {"id": number, "params": {"model_type": model_type, "activated_loras": list(loras), "loras_multipliers": multipliers}}


def leftovers(folder):
    return sorted(name for name in os.listdir(folder) if "speedups" in name or "_tmp" in name or name.endswith(".wav"))


# --- background save --------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sound", [True, False])
@pytest.mark.parametrize("continued", [True, False])
def test_background_save_overlaps_the_next_task(env, sound, continued):
    env.hooks.configure(env.hooks.Settings(background_save=True))
    env.wgp.SETTINGS["encode_s"] = 0.3
    first, second = task(1), task(2)
    env.queue([first, second])
    started = time.monotonic()
    env.call(first, sound=sound, continued=continued)
    assert time.monotonic() - started < 0.25  # returned before its file was written
    assert ("record", "clip_1.mp4") not in env.wgp.LOG and env.events == []
    assert os.path.exists(env.dir / "clip_1.mp4")  # the name is reserved
    env.state["gen"]["queue"].remove(first)
    env.call(second, sound=sound, continued=continued)
    # the last task of the queue returns once every file is written, in order, each announced after it is recorded
    assert [entry for entry in env.wgp.LOG if entry[0] == "record"] == [("record", "clip_1.mp4"), ("record", "clip_2.mp4")]
    expected = "video:1+sound:1" if sound else "video:1"
    assert (env.dir / "clip_1.mp4").read_text() == expected and (env.dir / "clip_2.mp4").read_text() == expected.replace("1", "2")
    assert [cmd for cmd, _ in env.events] == ["output", "output"]
    assert env.wgp.LOG.index(("record", "clip_1.mp4")) < env.wgp.LOG.index(("artifact", "clip_1.mp4"))
    assert leftovers(env.dir) == []  # temporary video, sound links and placeholders all removed


def test_off_by_default_and_waited_for_at_the_end_of_a_queue(env):
    env.run([task(1), task(2)])
    assert env.hooks.saver.history == [] and (env.dir / "clip_2.mp4").read_text() == "video:2+sound:2"
    env.hooks.configure(env.hooks.Settings(background_save=True))
    env.run([task(3)])  # alone in its queue: written before generate_media returns
    assert (env.dir / "clip_3.mp4").read_text() == "video:3+sound:3" and env.state["gen"]["file_list"][-1].endswith("clip_3.mp4")
    assert len(env.hooks.saver.history) == 1


@pytest.mark.parametrize("option, reason", [({"hdr": True}, "HDR"), ({"windows": 2}, "several windows")])
def test_anything_else_is_saved_in_the_foreground(env, capsys, option, reason):
    env.hooks.configure(env.hooks.Settings(background_save=True))
    env.wgp.SETTINGS["encode_s"] = 0.2
    first = task(1)
    env.queue([first, task(2)])
    env.call(first, **option)
    assert ("record", "clip_1.mp4") in env.wgp.LOG and env.events == [("output", None)]  # written before it returned
    assert reason in capsys.readouterr().out


def test_a_failed_save_is_reported_at_the_end_of_the_queue(env):
    env.hooks.configure(env.hooks.Settings(background_save=True))
    env.wgp.SETTINGS["fail_encode"] = True
    env.run([task(1), task(2)])
    errors = [data for cmd, data in env.events if cmd == "error"]
    assert errors == ["Saving clip_1.mp4 failed: encoder failed", "Saving clip_2.mp4 failed: encoder failed"]
    assert not os.path.exists(env.dir / "clip_1.mp4") and not any(entry[0] == "record" for entry in env.wgp.LOG)  # never published
    assert leftovers(env.dir) == []


# --- keep the model loaded --------------------------------------------------------------------------------------------------------

def test_model_kept_between_tasks_of_the_same_model(env):
    env.hooks.configure(env.hooks.Settings(keep_model_loaded=True))
    env.run([task(1), task(2), task(3, model_type="other")])
    # 1 -> 2 (same model): kept, with no full garbage collection after 1's generation nor before 2's; 2 -> 3 (another model) and
    # 3 (the end of the queue): unloaded right after their generation
    assert env.wgp.offloadobj.unloads == 2
    assert len(env.collects) == 4  # 6 in generate_media, 2 skipped
    # the last task's models leave right after its generation, before its file is written, as without the setting
    last = env.wgp.LOG[env.wgp.LOG.index(("generate", "other", None)):]
    assert last.index(("unload_all",)) < last.index(("record", "clip_3.mp4"))


def test_model_unloaded_after_an_abort_or_a_failure(env):
    env.hooks.configure(env.hooks.Settings(keep_model_loaded=True))
    env.state.setdefault("gen", {})["abort"] = True
    env.run([task(1), task(2)])
    assert env.wgp.offloadobj.unloads == 2
    env.state["gen"]["abort"] = False
    env.wgp.offloadobj.unloads = 0
    env.run([task(3), task(4)], crash=True)
    assert env.wgp.offloadobj.unloads == 2  # in the generation's except block: never deferred


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


# --- keep the LoRAs loaded --------------------------------------------------------------------------------------------------------

@pytest.fixture
def lora_files(tmp_path):
    files = []
    for name in ("a.safetensors", "b.safetensors"):
        (tmp_path / name).write_bytes(b"lora")
        files.append(str(tmp_path / name))
    return files


def test_loras_reused_by_the_next_task_selecting_them(env, lora_files):
    env.hooks.configure(env.hooks.Settings(keep_loras_loaded=True))
    a, b = lora_files
    env.run([task(1, loras=[a]), task(2, loras=[a]), task(3), task(4, loras=[a, b]), task(5, loras=[a, b], multipliers="0.5")])
    kinds = [call[0] if call[0] != "load" else ("load", call[1]) for call in env.offload.calls]
    # 1 loads; 2 reuses them (activated again); 3 selects none: unloaded at the end of 2 (and WanGP's own unload at the end of 3);
    # 4 loads two; 5 (other multipliers) reloads them
    assert kinds == [("load", ("a.safetensors",)), "activate", "unload", "unload", ("load", ("a.safetensors", "b.safetensors")), "unload",
                     ("load", ("a.safetensors", "b.safetensors")), "unload"]
    generated = [entry[2] for entry in env.wgp.LOG if entry[0] == "generate"]
    assert generated[2] is None  # task 3 ran without the LoRAs of task 2


def test_a_changed_queue_or_file_reloads(env, lora_files):
    env.hooks.configure(env.hooks.Settings(keep_loras_loaded=True))
    a, _ = lora_files
    first = task(1, loras=[a])
    env.queue([first, task(2, loras=[a])])
    env.call(first)
    env.queue([])  # the queue changed: another task runs instead
    env.call(task(9))
    assert [entry[2] for entry in env.wgp.LOG if entry[0] == "generate"][1] is None  # without the LoRAs kept for task 2
    third, fourth = task(3, loras=[a]), task(4, loras=[a])
    env.queue([third, fourth])
    env.call(third)
    env.state["gen"]["queue"].remove(third)
    os.utime(a, (1, 1))  # the file changed since it was loaded
    env.call(fourth)
    assert [call[0] for call in env.offload.calls].count("load") == 3


def test_loras_reloaded_into_a_kept_model_refresh_its_gpu_blocks(env, lora_files):
    env.hooks.configure(env.hooks.Settings(keep_loras_loaded=True, keep_model_loaded=True))
    a, b = lora_files
    env.run([task(1, loras=[a]), task(2, loras=[b])])
    assert env.wgp.offloadobj.refreshed == [env.wgp.MODEL]  # task 2's LoRAs, loaded into the model task 1 left in VRAM


# --- install ------------------------------------------------------------------------------------------------------------------------

def test_uninstall_restores_wgp(env):
    originals = {name: env.hooks.STATE["originals"][name] for name in ("generate_media", "save_video", "offload", "gc")}
    env.hooks.uninstall()
    for name, value in originals.items():
        assert env.wgp.__dict__[name] is value
    assert env.hooks.install(env.wgp.__dict__) == []


def test_a_wgp_without_the_hooked_names_is_left_alone():
    hooks = importlib.import_module("speedups.hooks")
    if hooks.STATE["installed"]:
        hooks.uninstall()
    assert "generate_media" in hooks.install({"offload": object(), "gc": gc})
    assert not hooks.STATE["installed"]
