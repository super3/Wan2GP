#!/usr/bin/env python3
"""MiniMax H3 model selection, one model per GPU (no USP): the 8 FL2VA combinations at 4 sizes, cold and warm.

On two GPUs it first runs a contention check, then shares the queue between the GPUs if running together costs <= 5%
(else GPU 0 runs everything). Needs plugins/chunk_streaming (branch feature/chunk-streaming-rebased) for the streaming
instrumentation; h3_bench_boot.sh with H3_BENCH=models builds the checkout.

    python runpod_worker/scripts/h3_bench_models.py --attention sage2 --profile 2 --verbose 1   # WanGP flags for every worker
    H3_BENCH_SINGLE=1 python runpod_worker/scripts/h3_bench_models.py ...                      # one GPU: no contention check
    H3_BENCH_FAKE=1 python runpod_worker/scripts/h3_bench_models.py                             # dry run of the scheduling (no GPU, no WanGP)

Environment: WANGP_ROOT (default: the checkout this script is in), H3_BENCH_OUT (default <WanGP>/outputs/h3_bench).

Every run streams (instrumented), so it yields the full render time and the earliest stall-free
playback start. Cold runs (the first run of each combination, 480p 5 s) are exclusive: the other
GPU is idle while one runs, so downloads and loads are timed cleanly and no file is downloaded twice
at once. Warm runs share the GPUs only if the contention check shows running together costs <= 5%.
Prints RESULT lines (h3_bench_tables.py turns them into tables); workers copy each warm run's video to <out>/videos/.
"""

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback

FAKE = os.environ.get("H3_BENCH_FAKE") == "1"
ROOT = os.path.abspath(os.environ.get("WANGP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.abspath(os.environ.get("H3_BENCH_OUT") or os.path.join(ROOT, "outputs", "h3_bench"))
WORKER_ARG = "--h3-bench-worker"  # first argument of a worker process; the remaining arguments are WanGP's flags
CLI = [arg for arg in sys.argv[1:] if arg != WORKER_ARG] or ["--attention", "sage2", "--profile", "2", "--verbose", "1"]
CELLS = [("832x480", 124), ("832x480", 243), ("1280x704", 124), ("1280x704", 243)]
THRESHOLD = 0.05


_SAY_LOCK = threading.Lock()


def say(tag, payload):
    line = "RESULT " + tag + " " + json.dumps(payload) + "\n"
    with _SAY_LOCK:  # two scheduler threads report at once: keep each line whole
        sys.stdout.write(line)
        sys.stdout.flush()


# ---------------------------------------------------------------------------- combinations

def combos():
    def profile(path, **overrides):
        if FAKE:
            return dict(overrides)
        with open(os.path.join(ROOT, "profiles", path)) as handle:
            settings = json.load(handle)
        settings.pop("profile_priority", None)
        settings.update(overrides)
        return settings

    no_lora = {"activated_loras": [], "loras_multipliers": "", "guidance_phases": 1, "guidance_scale": 1.0}
    larry = "minimax_h3/Larry v4 ema ckpt600 4 steps.json"
    l8 = "minimax_h3_fl2va/Lightx2v FL2V 8 Steps v1.0.json"
    return {
        "3": ("minimax_h3_fl2va_pruned", "Lightx2v 4 Steps v0.1", profile("minimax_h3_fl2va/Lightx2v FL2V 4 Steps v0.1.json")),
        "1b": ("minimax_h3_fl2va_pruned", "Larry v4 ckpt600", profile(larry, num_inference_steps=6)),
        "2b": ("minimax_h3_fl2va_pruned", "Lightx2v 8 Steps v1.0", profile(l8)),
        "1": ("minimax_h3_fl2va", "Larry v4 ckpt600", profile(larry, num_inference_steps=6)),
        "2": ("minimax_h3_fl2va", "Lightx2v 8 Steps v1.0", profile(l8)),
        "6": ("minimax_h3_fl2va", "None (28 steps)", dict(no_lora, num_inference_steps=28, sample_solver="euler", flow_shift=12.0)),
        "4": ("minimax_h3_vdn", "None (built in)", dict(no_lora, num_inference_steps=8)),
        "5": ("minimax_h3_fl2va_hyperflow", "None (built in)", dict(no_lora, num_inference_steps=6)),
    }


# Longest first, so the two queues finish close together; "3" runs first, in the contention check.
QUEUE_ORDER = ["6", "1", "2", "4", "2b", "1b", "5"]


# ---------------------------------------------------------------------------- worker

class DownloadWatch:
    def __init__(self, dirs):
        self.dirs, self.stop_flag = dirs, False
        self.start_bytes = self.last = self._size()
        self.t0, self.last_growth_s = time.monotonic(), 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _size(self):
        total = 0
        for d in self.dirs:
            for base, _, files in os.walk(d):
                for name in files:
                    try:
                        total += os.path.getsize(os.path.join(base, name))
                    except OSError:
                        pass
        return total

    def _run(self):
        while not self.stop_flag:
            size = self._size()
            if size > self.last:
                self.last, self.last_growth_s = size, time.monotonic() - self.t0
            time.sleep(1.0)

    def finish(self):
        self.stop_flag = True
        self.thread.join(5)
        self.last = max(self.last, self._size())
        gb = (self.last - self.start_bytes) / 1e9
        return round(gb, 2), (round(self.last_growth_s, 1) if gb > 0.001 else 0.0)


def worker():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    table = combos()
    if FAKE:
        def run(cmd):
            seconds = {"832x480": 1, "1280x704": 2}[cmd["resolution"]] * (2 if cmd["frames"] > 124 else 1) * 0.05
            time.sleep(seconds * (3 if cmd["tag"] == "cold" else 1))
            return {"ok": True, "wall_s": seconds, "ttff_s": seconds * 0.8}
    else:
        sys.path[:0] = [ROOT, os.path.join(ROOT, "plugins")]
        os.chdir(ROOT)
        from shared.api import init

        session = init(root=ROOT, output_dir=os.path.join(OUT, "out", f"gpu{gpu}"), cli_args=CLI, console_output=False)
        from chunk_streaming import streaming

        os.makedirs(os.path.join(OUT, "videos"), exist_ok=True)
        weight_dirs = [os.path.join(ROOT, "ckpts"), os.path.join(ROOT, "loras"), os.path.expanduser("~/.cache/huggingface")]
        prompt = json.load(open(os.path.join(ROOT, "defaults", "minimax_h3_fl2va.json")))["prompt"]

        def run(cmd):
            model_type, _, extra = table[cmd["combo"]]
            settings = {"model_type": model_type, "prompt": prompt, "resolution": cmd["resolution"],
                        "video_length": cmd["frames"], "seed": 42, "prompt_enhancer": "", "image_prompt_type": "",
                        "video_prompt_type": "", "audio_prompt_type": ""}
            settings.update(extra)
            watch = DownloadWatch(weight_dirs) if cmd["tag"] == "cold" else None
            rec = streaming.activate(store_frames=False, live_dir=os.path.join(OUT, "stream", f"gpu{gpu}", f"{cmd['combo']}-{cmd['tag']}"),
                                     audio_first=True)
            rec.reset()
            out = {"steps": settings.get("num_inference_steps")}
            try:
                started = time.monotonic()
                result = session.submit_task(settings).result()
                wall = time.monotonic() - started
                segments = rec.live.close() if rec.live is not None else []
                shift = 0.0 if rec.audio_first_effective else rec.audio_s
                ready = [s["t_done"] - started + shift for s in segments if s.get("t_done")]
                durations = [s["duration_s"] for s in segments if s.get("t_done")]
                out.update(ok=result.success, wall_s=round(wall, 2), errors=[str(e)[:300] for e in result.errors][:2],
                           audio_first=rec.audio_first_effective, segments=len(segments),
                           segments_with_audio=sum(1 for s in segments if s.get("has_audio")))
                if ready and len(ready) == len(durations):
                    out["ttff_s"] = round(streaming.no_buffer_start(ready, durations), 2)
                files = list(getattr(result, "generated_files", []) or [])
                if files and cmd.get("save"):
                    dest = os.path.join(OUT, "videos", f"{cmd['combo']}_{cmd['resolution']}_{cmd['frames']}f.mp4")
                    shutil.copy(files[0], dest)
                    out["video"] = os.path.basename(dest)
            except Exception as exc:
                traceback.print_exc()
                out["error"] = repr(exc)[:300]
            finally:
                streaming.deactivate()
                if watch is not None:
                    out["download_gb"], out["download_s"] = watch.finish()
            return out

    print("@@" + json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        cmd = json.loads(line)
        if cmd.get("quit"):
            break
        result = run(cmd)
        result.update({"combo": cmd["combo"], "resolution": cmd["resolution"], "frames": cmd["frames"],
                       "tag": cmd["tag"], "gpu": gpu, "profile": table[cmd["combo"]][1]})
        print("@@" + json.dumps(result), flush=True)


# ---------------------------------------------------------------------------- orchestrator

class Worker:
    def __init__(self, gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        self.gpu = gpu
        self.proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), WORKER_ARG, *CLI], env=env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.results = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        first = self.results.get()
        if not first.get("ready"):
            raise RuntimeError(f"worker {gpu} failed to start")

    def _read(self):
        for line in self.proc.stdout:
            if line.startswith("@@"):
                self.results.put(json.loads(line[2:]))
            elif "Traceback" in line or "Error" in line or "out of memory" in line:
                with _SAY_LOCK:
                    print(f"GPU{self.gpu}| {line.rstrip()[:400]}", flush=True)
        self.results.put({"dead": True})

    def run(self, combo, resolution, frames, tag, save=False):
        self.proc.stdin.write(json.dumps({"combo": combo, "resolution": resolution, "frames": frames, "tag": tag, "save": save}) + "\n")
        self.proc.stdin.flush()
        result = self.results.get()
        if result.get("dead"):
            raise RuntimeError(f"worker {self.gpu} died")
        return result

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"quit": True}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(60)
        except Exception:
            self.proc.kill()


class ColdLock:
    """Readers-writer lock: warm runs share the GPUs, a cold run has the machine to itself."""

    def __init__(self):
        self.cond = threading.Condition()
        self.active_warm = 0
        self.cold = False
        self.cold_waiting = 0

    def warm(self):
        lock = self

        class Ctx:
            def __enter__(self):
                with lock.cond:
                    while lock.cold or lock.cold_waiting:
                        lock.cond.wait()
                    lock.active_warm += 1

            def __exit__(self, *exc):
                with lock.cond:
                    lock.active_warm -= 1
                    lock.cond.notify_all()
        return Ctx()

    def exclusive(self):
        lock = self

        class Ctx:
            def __enter__(self):
                with lock.cond:
                    lock.cold_waiting += 1
                    while lock.cold or lock.active_warm:
                        lock.cond.wait()
                    lock.cold_waiting -= 1
                    lock.cold = True

            def __exit__(self, *exc):
                with lock.cond:
                    lock.cold = False
                    lock.cond.notify_all()
        return Ctx()


def run_unit(w, combo, lock, done_contention=None):
    """One combination on worker w: its cold run (exclusive), then the four warm cells."""
    warm_480p5 = None
    if done_contention is None:
        with lock.exclusive():
            cold = w.run(combo, "832x480", 124, "cold")
            if not cold.get("ok"):  # a transient download error: report it, then retry once (files already fetched stay)
                say("cold_failed_retrying", cold)
                cold = dict(w.run(combo, "832x480", 124, "cold"), retried=True)
        say("cold", cold)
    else:
        cold = done_contention
    for resolution, frames in CELLS:
        if done_contention is not None and (resolution, frames) == ("832x480", 124) and "warm_480p5" in done_contention:
            warm = done_contention["warm_480p5"]
        else:
            with lock.warm():
                warm = w.run(combo, resolution, frames, "warm", save=True)
        say("warm", warm)
        if (resolution, frames) == ("832x480", 124):
            warm_480p5 = warm.get("wall_s")
    if warm_480p5 and cold.get("wall_s") is not None:
        say("cold_split", {"combo": combo, "first_run_s": cold["wall_s"], "download_s": cold.get("download_s"),
                           "download_gb": cold.get("download_gb"), "warm_480p5_s": warm_480p5,
                           "load_and_warmup_s": round(cold["wall_s"] - (cold.get("download_s") or 0) - warm_480p5, 1)})


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def orchestrate():
    lock = ColdLock()
    w0 = Worker(0)
    if os.environ.get("H3_BENCH_SINGLE") == "1":  # one-GPU machine: no contention check, everything in the same order
        for combo in ["3"] + QUEUE_ORDER:
            run_unit(w0, combo, lock)
        w0.close()
        return
    # Contention check on combination 3 (FL2VA Pruned + Lightx2v 4-step v0.1), 480p 5 s.
    cold3 = w0.run("3", "832x480", 124, "cold")
    if not cold3.get("ok"):
        say("cold_failed_retrying", cold3)
        cold3 = dict(w0.run("3", "832x480", 124, "cold"), retried=True)
    say("cold", cold3)
    alone = [w0.run("3", "832x480", 124, "warm", save=(i == 0)) for i in range(2)]
    for r in alone:
        say("contention_alone_gpu0", r)
    w1 = Worker(1)
    w1_load = w1.run("3", "832x480", 124, "load")  # GPU 1 loads the model alone (weights already on disk)
    say("contention_gpu1_load", w1_load)
    w1_alone = w1.run("3", "832x480", 124, "warm")
    say("contention_alone_gpu1", w1_alone)
    together = []
    for _ in range(2):
        out = [None, None]
        threads = [threading.Thread(target=lambda i, w: out.__setitem__(i, w.run("3", "832x480", 124, "warm")), args=(i, w))
                   for i, w in enumerate((w0, w1))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        together.append(out)
        say("contention_together", {"gpu0": out[0], "gpu1": out[1]})
    a_wall, a_ttff = mean(r.get("wall_s") for r in alone), mean(r.get("ttff_s") for r in alone)
    t_wall = mean(o[0].get("wall_s") for o in together)
    t_ttff = mean(o[0].get("ttff_s") for o in together)
    t1_wall = mean(o[1].get("wall_s") for o in together)
    slow_wall = t_wall / a_wall - 1 if a_wall and t_wall else None
    slow_ttff = t_ttff / a_ttff - 1 if a_ttff and t_ttff else None
    parity = w1_alone.get("wall_s") / a_wall - 1 if a_wall and w1_alone.get("wall_s") else None
    slow_gpu1 = t1_wall / w1_alone["wall_s"] - 1 if t1_wall and w1_alone.get("wall_s") else None
    parallel = all(v is not None and abs(v) <= THRESHOLD for v in (slow_wall, slow_ttff, parity, slow_gpu1))
    say("contention_verdict", {"alone_wall_s": a_wall, "together_wall_s": t_wall, "slowdown_wall": slow_wall,
                               "alone_ttff_s": a_ttff, "together_ttff_s": t_ttff, "slowdown_ttff": slow_ttff,
                               "gpu1_vs_gpu0": parity, "slowdown_gpu1": slow_gpu1, "threshold": THRESHOLD,
                               "parallel": parallel})

    contention_unit = dict(cold3, warm_480p5=alone[0])
    if not parallel:
        w1.close()
        run_unit(w0, "3", lock, done_contention=contention_unit)
        for combo in QUEUE_ORDER:
            run_unit(w0, combo, lock)
        w0.close()
        return

    todo = queue.Queue()
    for combo in QUEUE_ORDER:
        todo.put(combo)

    def drain(w, first=None):
        try:
            if first is not None:
                run_unit(w, "3", lock, done_contention=first)
            while True:
                try:
                    combo = todo.get_nowait()
                except queue.Empty:
                    return
                run_unit(w, combo, lock)
        except Exception:
            traceback.print_exc()
            say("worker_failed", {"gpu": w.gpu})

    threads = [threading.Thread(target=drain, args=(w0, contention_unit)), threading.Thread(target=drain, args=(w1,))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    w0.close()
    w1.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == WORKER_ARG:
        worker()
        sys.exit(0)
    try:
        orchestrate()
    except BaseException:
        traceback.print_exc()
        print("RESULT FAILED", flush=True)
        raise
    print("RESULT DONE", flush=True)
