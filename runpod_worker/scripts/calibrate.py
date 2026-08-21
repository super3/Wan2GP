#!/usr/bin/env python3
"""Measure what a generation actually costs, and replace the plan's estimates.

``docs/RUNPOD_SERVERLESS.md`` is explicit that every second-level number in its
cold-start and cost tables is extrapolation:

    "Nothing in this repo or its README states an H3 generation wall-clock. ...
     scripts/calibrate.py exists to replace them; do not sign an SLA before
     running it."

This is that script. It runs a matrix of ``(resolution, video_length, steps)``
generations, records the wall clock of each and the phase boundaries inside it,
and prints a table plus the three numbers you need to configure the endpoint:
a recommended ``executionTimeout``, a recommended ``WANGP_DEFAULT_BUDGET_S`` /
``WANGP_MAX_BUDGET_S`` pair, and the cost per clip.

TWO MODES
---------
**Local (default).** Boots WanGP in this process and drives ``engine.run`` — the
same code path the handler uses, so ``phase_marks`` are real. Needs a GPU and the
weights. This is the Tier-2/Tier-3 measurement.

    python3 -m runpod_worker.scripts.calibrate \\
        --matrix steps=4,20 frames=124,362 resolution=832x480 --repeat 3 \\
        --profile "Turbo Lightx2v FL2V 4 Steps v1.0 768p" --gpu L40S

**Remote (``--endpoint``).** Drives a deployed endpoint over ``/run`` + ``/status``
and reads RunPod's own ``delayTime`` / ``executionTime`` plus the handler's
``output.metrics``. This is the Tier-4 measurement, and the only one that
measures queue delay and cold starts.

    RUNPOD_API_KEY=... python3 -m runpod_worker.scripts.calibrate \\
        --endpoint $ENDPOINT_ID --matrix steps=4,8,20 frames=124,209,362 --repeat 3

WHAT "COLD" MEANS HERE
----------------------
The first generation in a process also pays the 150-250 s weight read
(``wgp.py:6773`` -> ``load_models``). Percentiles are therefore computed over
*warm* runs, and cold runs are reported separately — mixing them produces a p99
that describes neither. Local mode detects cold via ``engine.is_warm()``; remote
mode uses ``output.metrics.jobs_served == 1``, which the handler reports.

Standard library only (``urllib.request`` for the REST calls) — no new
dependency, and the local mode's only heavy import is the worker's own engine.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INTERRUPTED = 130

#: $/s by GPU tier, from the plan's "GPU tier" table. RunPod changes these —
#: pass --price-per-second to override rather than trusting this dict.
GPU_PRICES: dict[str, float] = {
    "L40S": 0.00053,
    "L40": 0.00053,
    "RTX6000ADA": 0.00053,
    "A6000": 0.00034,
    "A40": 0.00034,
    "A100-80": 0.00076,
    "A100": 0.00076,
    "H100": 0.00155,
}

#: Axis name in --matrix -> WanGP settings key.
AXES: dict[str, str] = {
    "resolution": "resolution",
    "frames": "video_length",
    "video_length": "video_length",
    "steps": "num_inference_steps",
    "num_inference_steps": "num_inference_steps",
    "flow_shift": "flow_shift",
    "solver": "sample_solver",
    "sample_solver": "sample_solver",
    "window": "sliding_window_size",
    "overlap": "sliding_window_overlap",
}

#: Axes that are integers in WanGP's settings dict.
INT_AXES = {"video_length", "num_inference_steps", "sliding_window_size", "sliding_window_overlap"}
FLOAT_AXES = {"flow_shift"}

DEFAULT_MATRIX = ["steps=4,20", "frames=124", "resolution=832x480"]

#: A prompt short enough to read in a log and long enough to exercise the text
#: encoder. Overridable with --prompt / --prompt-file.
DEFAULT_PROMPT = (
    "integrated_multimodal_description: [Shot 1] A five-second cinematic single take of a "
    "lighthouse keeper climbing a spiral stair at dawn, warm lamp light sweeping past the "
    "window as the camera pushes in.\n"
    "overall_soundscape: Wind against glass, boots on iron stairs, a distant foghorn.\n"
    "non_diegetic_music: One slow rising string chord that fades at the end."
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate",
        description="Time a matrix of generations and print measured timeout/cost numbers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Axes accepted by --matrix: resolution, frames (= video_length), steps "
            "(= num_inference_steps), flow_shift, solver (= sample_solver), window, overlap.\n"
            "Example: --matrix steps=4,8,20 frames=124,209,362 resolution=832x480 --repeat 3\n"
            "Exit 0 when every run succeeded, 1 when any cell failed (the tables are "
            "printed either way)."
        ),
    )
    parser.add_argument(
        "--matrix",
        nargs="*",
        default=None,
        metavar="AXIS=V1,V2",
        help=f"matrix axes (default: {' '.join(DEFAULT_MATRIX)})",
    )
    parser.add_argument("--repeat", type=int, default=1, help="runs per cell (default 1)")
    parser.add_argument("--model-type", help="default $WANGP_MODEL_TYPE")
    parser.add_argument("--profile", help="accelerator profile to apply to every cell")
    parser.add_argument("--prompt", help="prompt text (default: a short built-in H3 prompt)")
    parser.add_argument("--prompt-file", help="read the prompt from this file")
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="fixed seed so cells are comparable; -1 for a fresh seed per run",
    )
    parser.add_argument(
        "--budget-s",
        type=float,
        help="per-run wall-clock budget (default: WANGP_MAX_BUDGET_S). In local mode "
        "this is passed straight to engine.run and is NOT clamped -- a 20-step 362-frame "
        "cell can outlast the endpoint's own ceiling and you still want the measurement.",
    )

    remote = parser.add_argument_group("remote mode")
    remote.add_argument("--endpoint", help="RunPod endpoint id; drives /run + /status instead")
    remote.add_argument("--api-key", help="default $RUNPOD_API_KEY")
    remote.add_argument(
        "--base-url",
        default=os.environ.get("RUNPOD_API_BASE", "https://api.runpod.ai/v2"),
        help="API base (default https://api.runpod.ai/v2)",
    )
    remote.add_argument("--poll-s", type=float, default=5.0, help="status poll interval")
    remote.add_argument(
        "--execution-timeout-ms",
        type=int,
        default=3600000,
        help="policy.executionTimeout sent with each job (default 3600000)",
    )
    remote.add_argument(
        "--timeout-s",
        type=float,
        default=3600.0,
        help="give up polling one job after this long (default 3600)",
    )

    cost = parser.add_argument_group("cost")
    cost.add_argument(
        "--gpu",
        default="L40S",
        help=f"GPU tier label for the cost column; known: {', '.join(sorted(GPU_PRICES))}",
    )
    cost.add_argument(
        "--price-per-second",
        type=float,
        help="$/s override (default: the --gpu tier's rate)",
    )
    cost.add_argument(
        "--cold-start-s",
        type=float,
        default=None,
        help="billed cold-start seconds to add to the cold-start cost line "
        "(default: measured, else 240)",
    )

    env = parser.add_argument_group("environment (local mode)")
    env.add_argument("--root", help="WanGP repo root (default $WANGP_ROOT or /opt/wangp)")
    env.add_argument("--config", help="path to wgp_config.json or the directory holding it")
    env.add_argument("--volume-root", help="network volume mount (default $WANGP_VOLUME_ROOT)")
    env.add_argument(
        "--keep-outputs",
        action="store_true",
        help="do not delete the generated videos (default: delete after probing)",
    )

    parser.add_argument("--json", dest="json_out", metavar="PATH", help="write raw results as JSON")
    parser.add_argument("--csv", dest="csv_out", metavar="PATH", help="write one row per run")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and exit without generating anything",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="only print the final tables")
    return parser


def _apply_env(args: argparse.Namespace) -> None:
    """CLI -> env, before runpod_worker.config is imported and freezes its paths."""
    if args.root:
        os.environ["WANGP_ROOT"] = str(Path(args.root).expanduser())
    if args.config:
        given = Path(args.config).expanduser()
        os.environ["WANGP_CONFIG_DIR"] = str(given.parent if given.suffix == ".json" else given)
    if args.volume_root:
        os.environ["WANGP_VOLUME_ROOT"] = str(Path(args.volume_root).expanduser())


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------


def parse_matrix(specs: Sequence[str] | None) -> dict[str, list[Any]]:
    """``["steps=4,8", "frames=124"]`` -> ``{"num_inference_steps": [4, 8], ...}``."""
    axes: dict[str, list[Any]] = {}
    for spec in specs if specs else DEFAULT_MATRIX:
        text = str(spec).strip()
        if not text:
            continue
        if "=" not in text:
            raise SystemExit(f"--matrix entry {text!r} is not AXIS=VALUE[,VALUE...]")
        name, _, raw = text.partition("=")
        key = AXES.get(name.strip().lower())
        if key is None:
            raise SystemExit(
                f"unknown matrix axis {name.strip()!r}; known: {', '.join(sorted(AXES))}"
            )
        values: list[Any] = []
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if key in INT_AXES:
                try:
                    values.append(int(chunk))
                except ValueError as exc:
                    raise SystemExit(f"{name}={chunk!r} is not an integer") from exc
            elif key in FLOAT_AXES:
                try:
                    values.append(float(chunk))
                except ValueError as exc:
                    raise SystemExit(f"{name}={chunk!r} is not a number") from exc
            else:
                values.append(chunk)
        if not values:
            raise SystemExit(f"--matrix entry {text!r} lists no values")
        # A repeated axis extends rather than replaces, so
        # "--matrix steps=4 steps=20" behaves like "steps=4,20".
        axes.setdefault(key, [])
        for value in values:
            if value not in axes[key]:
                axes[key].append(value)
    return axes


def expand_cells(axes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of the axes, in a stable order."""
    if not axes:
        return [{}]
    keys = list(axes)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(axes[key] for key in keys))]


def cell_label(cell: dict[str, Any]) -> str:
    short = {
        "resolution": "res",
        "video_length": "frames",
        "num_inference_steps": "steps",
        "sliding_window_size": "win",
        "sliding_window_overlap": "ovl",
        "sample_solver": "solver",
        "flow_shift": "shift",
    }
    return " ".join(f"{short.get(key, key)}={value}" for key, value in cell.items()) or "default"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile. Honest for the n<=5 samples calibration produces.

    ``statistics.quantiles`` interpolates, which invents a p99 from three samples
    that reads as more precise than the data supports.
    """
    data = sorted(float(value) for value in values)
    if not data:
        return None
    rank = max(1, math.ceil(fraction * len(data)))
    return data[min(rank, len(data)) - 1]


def summarize(values: Sequence[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    if not data:
        return {"n": 0}
    return {
        "n": len(data),
        "min": round(min(data), 2),
        "p50": round(percentile(data, 0.50) or 0.0, 2),
        "p90": round(percentile(data, 0.90) or 0.0, 2),
        "p99": round(percentile(data, 0.99) or 0.0, 2),
        "max": round(max(data), 2),
        "mean": round(statistics.fmean(data), 2),
    }


def phase_durations(marks: dict[str, float]) -> list[tuple[str, float]]:
    """Turn ``{phase: first_seen_at_s}`` into ``[(phase, duration_s)]``.

    ``engine.run`` records the elapsed time at which each phase was first
    observed (``phase_marks.setdefault(phase, now - t0)``), so the duration of a
    phase is the gap to the next one. The final phase's duration is unknown from
    the marks alone and is left to the caller (it is ``gen_s`` minus its start).
    """
    ordered = sorted(marks.items(), key=lambda item: item[1])
    out: list[tuple[str, float]] = []
    for index, (name, start) in enumerate(ordered):
        if index + 1 < len(ordered):
            out.append((name, round(ordered[index + 1][1] - start, 1)))
        else:
            out.append((name, -1.0))  # open-ended; filled in by the caller
    return out


# ---------------------------------------------------------------------------
# Local runner
# ---------------------------------------------------------------------------


class LocalRunner:
    """Drives ``engine.run`` in this process. Needs a GPU and the weights."""

    def __init__(self, args: argparse.Namespace, say) -> None:
        from runpod_worker import config as C  # noqa: PLC0415 - after _apply_env
        from runpod_worker import engine, media_out, schema

        self.C = C
        self.engine = engine
        self.schema = schema
        self.media_out = media_out
        self.args = args
        self.say = say

        started = time.monotonic()
        self.session = engine.boot()
        self.boot_s = round(time.monotonic() - started, 1)
        self.model_type = args.model_type or C.CONFIG.model_type
        say(f"wgp imported in {self.boot_s}s; model_type={self.model_type}")

        self.defaults = self.session.get_default_settings(self.model_type)
        self.model_def = self.session.get_model_def(self.model_type) or {}
        self.budget_s = float(
            args.budget_s if args.budget_s is not None else C.CONFIG.max_budget_s
        )

    @property
    def mode(self) -> str:
        return "local"

    def context(self) -> dict[str, Any]:
        return {
            "mode": "local",
            "model_type": self.model_type,
            "boot_s": self.boot_s,
            "attention_mode": getattr(
                self.session._ensure_runtime().module, "attention_mode", None
            ),
            "budget_s": self.budget_s,
        }

    def build_settings(self, cell: dict[str, Any], prompt: str, seed: int) -> Any:
        payload: dict[str, Any] = {
            "model_type": self.model_type,
            "settings": {"prompt": prompt, "seed": seed, **cell},
        }
        # Deliberately no runtime.timeout_s: WorkerConfig.budget_for clamps it to
        # WANGP_MAX_BUDGET_S, which would silently cut a cell short. The budget
        # this tool measures under is --budget-s, passed to engine.run directly.
        if self.args.profile:
            payload["profile"] = self.args.profile
        # Same validation path as the handler, so a cell that a client could not
        # submit cannot be measured either (frames are floored to the lattice
        # here, which is why the reported video_length may differ from the axis).
        return self.schema.parse(
            payload,
            model_type=self.model_type,
            allowed_settings=self.defaults,
            model_def=self.model_def,
            cfg=self.C.CONFIG,
            session=self.session,
        )

    def run_once(self, cell: dict[str, Any], prompt: str, seed: int) -> dict[str, Any]:
        from runpod_worker.errors import WorkerError

        record: dict[str, Any] = {"cell": dict(cell), "mode": "local"}
        try:
            request = self.build_settings(cell, prompt, seed)
        except WorkerError as exc:
            record.update(ok=False, error=f"{exc.code}: {exc.message}")
            return record

        record["settings"] = {
            key: request.settings.get(key)
            for key in (
                "resolution",
                "video_length",
                "num_inference_steps",
                "flow_shift",
                "guidance_scale",
                "sample_solver",
                "seed",
                "sliding_window_size",
                "sliding_window_overlap",
            )
        }
        record["cold"] = not self.engine.is_warm(self.model_type)

        wall_started = time.monotonic()
        try:
            outcome = self.engine.run(request.settings, budget_s=self.budget_s)
        except WorkerError as exc:
            record.update(
                ok=False,
                error=f"{exc.code}: {exc.message}",
                wall_s=round(time.monotonic() - wall_started, 2),
            )
            return record
        wall_s = round(time.monotonic() - wall_started, 2)

        record.update(
            ok=bool(outcome.success and outcome.files),
            wall_s=wall_s,
            gen_s=outcome.gen_s,
            timed_out=outcome.timed_out,
            cancelled=outcome.cancelled,
            phase_marks_s=dict(outcome.phase_marks),
            errors=list(outcome.errors)[:5],
        )
        if not record["ok"] and not record.get("error"):
            record["error"] = (
                "; ".join(outcome.errors[:3])
                or (
                    f"exceeded the {self.budget_s:.0f}s budget and was cancelled "
                    f"(raise --budget-s)"
                    if outcome.timed_out
                    else "no output file"
                )
            )

        video = outcome.video_path
        if video and os.path.isfile(video):
            record["output_bytes"] = os.path.getsize(video)
            probe = self.media_out.ffprobe(video)
            for key in (
                "duration_s", "fps", "width", "height", "video_codec",
                "has_audio", "audio_codec", "audio_sample_rate",
            ):
                if key in probe:
                    record[key] = probe[key]
            if record.get("duration_s") and record.get("gen_s"):
                record["realtime_factor"] = round(
                    float(record["gen_s"]) / float(record["duration_s"]), 2
                )
        if not self.args.keep_outputs:
            for path in outcome.files:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        return record


# ---------------------------------------------------------------------------
# Remote runner
# ---------------------------------------------------------------------------


class RemoteRunner:
    """Drives a deployed endpoint through ``/run`` + ``/status``.

    ``/runsync`` is deliberately not used: it waits 90 s by default and 300 s at
    most, which no video generation fits inside.
    """

    def __init__(self, args: argparse.Namespace, say) -> None:
        self.args = args
        self.say = say
        self.endpoint = str(args.endpoint)
        self.api_key = args.api_key or os.environ.get("RUNPOD_API_KEY", "")
        if not self.api_key:
            raise SystemExit("remote mode needs --api-key or $RUNPOD_API_KEY")
        self.base = f"{args.base_url.rstrip('/')}/{self.endpoint}"
        self.model_type = args.model_type or os.environ.get(
            "WANGP_MODEL_TYPE", "minimax_h3_fl2va_pruned"
        )
        self.budget_s = float(args.budget_s) if args.budget_s is not None else None

    @property
    def mode(self) -> str:
        return "remote"

    def context(self) -> dict[str, Any]:
        return {
            "mode": "remote",
            "endpoint": self.endpoint,
            "base_url": self.base,
            "model_type": self.model_type,
        }

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base}/{path.lstrip('/')}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            method="POST" if data is not None else "GET",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                body = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"{exc.code} {exc.reason} from {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"cannot reach {url}: {exc.reason}") from exc
        try:
            return json.loads(body or "{}")
        except ValueError as exc:
            raise RuntimeError(f"{path} returned non-JSON: {body[:200]}") from exc

    def run_once(self, cell: dict[str, Any], prompt: str, seed: int) -> dict[str, Any]:
        settings: dict[str, Any] = {"prompt": prompt, "seed": seed, **cell}
        job_input: dict[str, Any] = {"model_type": self.model_type, "settings": settings}
        if self.args.profile:
            job_input["profile"] = self.args.profile
        runtime: dict[str, Any] = {
            # A fresh idempotency key per run: the handler HEADs the derived
            # object key and returns the cached result for a repeat, which would
            # silently measure 0 GPU seconds (failure mode 23).
            "idempotency_key": f"calib-{uuid.uuid4().hex[:16]}"
        }
        if self.budget_s is not None:
            runtime["timeout_s"] = self.budget_s
        job_input["runtime"] = runtime

        payload = {
            "input": job_input,
            "policy": {"executionTimeout": int(self.args.execution_timeout_ms)},
        }

        record: dict[str, Any] = {"cell": dict(cell), "mode": "remote"}
        submitted = time.monotonic()
        try:
            queued = self._request("run", payload)
        except RuntimeError as exc:
            record.update(ok=False, error=str(exc))
            return record
        job_id = queued.get("id")
        record["job_id"] = job_id
        if not job_id:
            record.update(ok=False, error=f"/run returned no id: {queued}")
            return record

        deadline = submitted + float(self.args.timeout_s)
        status: dict[str, Any] = {}
        while True:
            if time.monotonic() > deadline:
                record.update(
                    ok=False,
                    error=f"job {job_id} still {status.get('status', 'IN_QUEUE')} after "
                    f"{self.args.timeout_s}s",
                    wall_s=round(time.monotonic() - submitted, 2),
                )
                return record
            time.sleep(max(0.5, float(self.args.poll_s)))
            try:
                status = self._request(f"status/{job_id}")
            except RuntimeError as exc:
                record.update(ok=False, error=str(exc))
                return record
            state = str(status.get("status", "")).upper()
            if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                break

        record["wall_s"] = round(time.monotonic() - submitted, 2)
        record["status"] = status.get("status")
        # RunPod reports both in milliseconds.
        if status.get("delayTime") is not None:
            record["delay_s"] = round(float(status["delayTime"]) / 1000.0, 2)
        if status.get("executionTime") is not None:
            record["execution_s"] = round(float(status["executionTime"]) / 1000.0, 2)

        output = status.get("output")
        output = output if isinstance(output, dict) else {}
        metrics = output.get("metrics") if isinstance(output.get("metrics"), dict) else {}
        record["metrics"] = metrics
        if metrics.get("generate_s") is not None:
            record["gen_s"] = float(metrics["generate_s"])
        elif record.get("execution_s") is not None:
            record["gen_s"] = record["execution_s"]
        if metrics.get("phase_marks_s"):
            record["phase_marks_s"] = dict(metrics["phase_marks_s"])
        # The handler reports jobs_served AFTER incrementing, so 1 == this job was
        # the first on that worker, i.e. it paid the weight load.
        served = metrics.get("jobs_served")
        record["cold"] = bool(served == 1) if served is not None else None
        if metrics.get("boot_ms"):
            record["boot_s"] = round(float(metrics["boot_ms"]) / 1000.0, 1)

        video = output.get("video") if isinstance(output.get("video"), dict) else {}
        for key, target in (
            ("size_bytes", "output_bytes"),
            ("duration_s", "duration_s"),
            ("fps", "fps"),
            ("width", "width"),
            ("height", "height"),
            ("video_codec", "video_codec"),
            ("has_audio", "has_audio"),
            ("audio_sample_rate", "audio_sample_rate"),
            ("transport", "transport"),
        ):
            if video.get(key) is not None:
                record[target] = video[key]
        if record.get("duration_s") and record.get("gen_s"):
            record["realtime_factor"] = round(
                float(record["gen_s"]) / float(record["duration_s"]), 2
            )

        resolved = output.get("resolved") if isinstance(output.get("resolved"), dict) else {}
        if resolved:
            record["settings"] = resolved

        if str(status.get("status", "")).upper() == "COMPLETED" and output.get("status") != "error":
            record["ok"] = True
        else:
            record["ok"] = False
            record["error"] = (
                status.get("error")
                or output.get("error")
                or output.get("error_code")
                or f"status={status.get('status')}"
            )
            if output.get("error_code"):
                record["error_code"] = output["error_code"]
        return record


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(value: Any, width: int, digits: int = 1) -> str:
    if value is None:
        return "-".rjust(width)
    if isinstance(value, float):
        return f"{value:,.{digits}f}".rjust(width)
    return str(value).rjust(width)


def render_table(rows: list[dict[str, Any]], headers: list[tuple[str, str, int]]) -> str:
    """``headers`` is ``[(key, label, width)]``; everything is right-aligned."""
    lines = [" ".join(label.rjust(width) for _key, label, width in headers)]
    lines.append(" ".join("-" * width for _key, _label, width in headers))
    for row in rows:
        cells = []
        for key, _label, width in headers:
            value = row.get(key)
            if isinstance(value, float):
                cells.append(_fmt(value, width))
            elif value is None:
                cells.append("-".rjust(width))
            else:
                text = str(value)
                cells.append(text[-width:].rjust(width) if len(text) > width else text.rjust(width))
        lines.append(" ".join(cells))
    return "\n".join(lines)


def aggregate(runs: list[dict[str, Any]], price: float) -> list[dict[str, Any]]:
    """One summary row per cell, warm runs only when there are any."""
    cells: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        cells.setdefault(cell_label(run["cell"]), []).append(run)

    summaries = []
    for label, group in cells.items():
        ok_runs = [run for run in group if run.get("ok")]
        warm = [run for run in ok_runs if not run.get("cold")]
        sample = warm or ok_runs
        seconds = [float(run.get("gen_s") or run.get("wall_s") or 0.0) for run in sample]
        stats = summarize(seconds)
        durations = [float(run["duration_s"]) for run in sample if run.get("duration_s")]
        sizes = [int(run["output_bytes"]) for run in sample if run.get("output_bytes")]
        summaries.append(
            {
                "cell": label,
                "n": len(group),
                "ok": len(ok_runs),
                "cold": sum(1 for run in group if run.get("cold")),
                "p50": stats.get("p50"),
                "p90": stats.get("p90"),
                "p99": stats.get("p99"),
                "min": stats.get("min"),
                "max": stats.get("max"),
                "clip_s": round(statistics.fmean(durations), 2) if durations else None,
                "rt_x": round(stats["p50"] / statistics.fmean(durations), 1)
                if durations and stats.get("p50")
                else None,
                "MB": round(statistics.fmean(sizes) / (1024 * 1024), 1) if sizes else None,
                "usd": round((stats.get("p50") or 0.0) * price, 4) if stats.get("n") else None,
                "usd_p90": round((stats.get("p90") or 0.0) * price, 4) if stats.get("n") else None,
                "_seconds": seconds,
            }
        )
    return summaries


def recommendations(
    runs: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    *,
    price: float,
    cold_start_s: float | None,
    cancel_grace_s: float,
) -> dict[str, Any]:
    """The three numbers this script exists to produce."""
    warm_seconds = [
        float(run.get("gen_s") or run.get("wall_s") or 0.0)
        for run in runs
        if run.get("ok") and not run.get("cold")
    ]
    all_seconds = [
        float(run.get("gen_s") or run.get("wall_s") or 0.0) for run in runs if run.get("ok")
    ]
    sample = warm_seconds or all_seconds
    if not sample:
        return {}

    overall_p99 = percentile(sample, 0.99) or 0.0
    slowest_cell_p99 = max((row.get("p99") or 0.0) for row in summaries) if summaries else 0.0

    # The plan's arithmetic: budget + cancel grace + ~60 s of probe/upload/cleanup
    # must stay under the endpoint's execution timeout, or RunPod's hard kill
    # wins the race against our cooperative cancel and the client gets a
    # platform error instead of a typed `timeout`.
    default_budget = int(math.ceil(overall_p99 * 1.3))
    max_budget = int(math.ceil(slowest_cell_p99 * 1.5)) or default_budget
    max_budget = max(max_budget, default_budget)
    overhead = 60
    execution_timeout_s = int(
        math.ceil((max_budget + cancel_grace_s + overhead) / 60.0) * 60
    )

    cold_runs = [
        float(run.get("gen_s") or run.get("wall_s") or 0.0) for run in runs
        if run.get("ok") and run.get("cold")
    ]
    measured_cold_adder = None
    if cold_runs and warm_seconds:
        measured_cold_adder = round(
            statistics.fmean(cold_runs) - statistics.fmean(warm_seconds), 1
        )
    adder = cold_start_s if cold_start_s is not None else (measured_cold_adder or 240.0)

    delays = [float(run["delay_s"]) for run in runs if run.get("delay_s") is not None]

    return {
        "warm_p50_s": round(percentile(sample, 0.50) or 0.0, 1),
        "warm_p90_s": round(percentile(sample, 0.90) or 0.0, 1),
        "warm_p99_s": round(overall_p99, 1),
        "slowest_cell_p99_s": round(slowest_cell_p99, 1),
        "cold_start_adder_s": round(float(adder), 1),
        "cold_start_measured": measured_cold_adder,
        "delay_p50_s": round(percentile(delays, 0.50) or 0.0, 1) if delays else None,
        "delay_p90_s": round(percentile(delays, 0.90) or 0.0, 1) if delays else None,
        "WANGP_DEFAULT_BUDGET_S": default_budget,
        "WANGP_MAX_BUDGET_S": max_budget,
        "WANGP_CANCEL_GRACE_S": int(cancel_grace_s),
        "executionTimeout_s": execution_timeout_s,
        "executionTimeout_ms": execution_timeout_s * 1000,
        "usd_per_clip_p50": round((percentile(sample, 0.50) or 0.0) * price, 4),
        "usd_per_clip_p90": round((percentile(sample, 0.90) or 0.0) * price, 4),
        "usd_cold_start_adder": round(float(adder) * price, 4),
    }


def write_csv(path: str, runs: list[dict[str, Any]]) -> None:
    import csv  # noqa: PLC0415 - only needed when --csv is passed

    columns = [
        "cell", "repeat", "ok", "cold", "wall_s", "gen_s", "execution_s", "delay_s",
        "duration_s", "realtime_factor", "output_bytes", "width", "height", "fps",
        "video_codec", "has_audio", "transport", "error",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            row = dict(run)
            row["cell"] = cell_label(run.get("cell") or {})
            writer.writerow(row)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _apply_env(args)

    axes = parse_matrix(args.matrix)
    cells = expand_cells(axes)
    repeat = max(1, int(args.repeat))

    prompt = DEFAULT_PROMPT
    if args.prompt_file:
        prompt = Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
    if args.prompt:
        prompt = args.prompt

    price = args.price_per_second
    if price is None:
        price = GPU_PRICES.get(str(args.gpu).upper().replace(" ", ""))
    if price is None:
        raise SystemExit(
            f"unknown --gpu {args.gpu!r}; pass --price-per-second or one of "
            f"{', '.join(sorted(GPU_PRICES))}"
        )

    def say(*parts: Any) -> None:
        if not args.quiet:
            print(*parts, flush=True)

    say("=" * 78)
    say(f"calibration plan: {len(cells)} cell(s) x {repeat} repeat(s) = "
        f"{len(cells) * repeat} generation(s)")
    say("=" * 78)
    for index, cell in enumerate(cells, 1):
        say(f"  {index:>2}. {cell_label(cell)}")
    say(f"  gpu={args.gpu} price=${price}/s  profile={args.profile or '(none)'}  "
        f"seed={args.seed}")
    say("")

    if args.dry_run:
        say("dry run: nothing was generated")
        return EXIT_OK

    try:
        runner: Any = (
            RemoteRunner(args, say) if args.endpoint else LocalRunner(args, say)
        )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED to start: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED

    cancel_grace_s = 150.0
    try:
        from runpod_worker import config as C  # noqa: PLC0415

        cancel_grace_s = float(C.CONFIG.cancel_grace_s)
    except Exception:  # noqa: BLE001 - remote mode may run outside the repo image
        pass

    runs: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, cell in enumerate(cells, 1):
        for attempt in range(1, repeat + 1):
            seed = args.seed if args.seed is not None and args.seed >= 0 else -1
            say(f"[{index}/{len(cells)} run {attempt}/{repeat}] {cell_label(cell)} ...")
            record = runner.run_once(cell, prompt, seed)
            record["repeat"] = attempt
            runs.append(record)
            if record.get("ok"):
                say(
                    f"    ok  wall={record.get('wall_s')}s gen={record.get('gen_s')}s"
                    f"{' (cold)' if record.get('cold') else ''}"
                    f" clip={record.get('duration_s')}s"
                    f" size={round((record.get('output_bytes') or 0) / 1048576, 1)}MB"
                )
                marks = record.get("phase_marks_s") or {}
                if marks:
                    total = float(record.get("gen_s") or record.get("wall_s") or 0.0)
                    parts = []
                    for name, duration in phase_durations(marks):
                        if duration < 0:
                            duration = round(max(0.0, total - marks[name]), 1)
                        parts.append(f"{name}={duration}s")
                    say("    phases: " + "  ".join(parts))
            else:
                say(f"    FAILED: {record.get('error')}")

    elapsed = round(time.monotonic() - started, 1)
    summaries = aggregate(runs, price)
    recs = recommendations(
        runs, summaries, price=price,
        cold_start_s=args.cold_start_s, cancel_grace_s=cancel_grace_s,
    )

    print("")
    print("=" * 78)
    print(f"RESULTS  ({sum(1 for run in runs if run.get('ok'))}/{len(runs)} ok, "
          f"{elapsed}s wall clock, {args.gpu} @ ${price}/s)")
    print("=" * 78)
    print(
        render_table(
            summaries,
            [
                ("cell", "cell", 34),
                ("n", "n", 3),
                ("ok", "ok", 3),
                ("cold", "cold", 4),
                ("p50", "p50 s", 8),
                ("p90", "p90 s", 8),
                ("p99", "p99 s", 8),
                ("clip_s", "clip s", 7),
                ("rt_x", "x RT", 6),
                ("MB", "MB", 6),
                ("usd", "$/clip", 8),
            ],
        )
    )
    print("")

    # Per-phase medians, across every successful run that reported marks.
    phase_totals: dict[str, list[float]] = {}
    for run in runs:
        marks = run.get("phase_marks_s") or {}
        if not (run.get("ok") and marks):
            continue
        total = float(run.get("gen_s") or run.get("wall_s") or 0.0)
        for name, duration in phase_durations(marks):
            if duration < 0:
                duration = max(0.0, total - float(marks[name]))
            phase_totals.setdefault(name, []).append(duration)
    if phase_totals:
        print("phase breakdown (median seconds across successful runs)")
        rows = [
            {"phase": name, "median_s": round(statistics.median(values), 1), "n": len(values)}
            for name, values in sorted(
                phase_totals.items(), key=lambda item: -statistics.median(item[1])
            )
        ]
        print(render_table(rows, [("phase", "phase", 24), ("median_s", "median s", 10),
                                  ("n", "n", 4)]))
        print("")

    failures = [run for run in runs if not run.get("ok")]
    if failures:
        print(f"failures ({len(failures)})")
        for run in failures[:10]:
            print(f"  {cell_label(run['cell'])}: {run.get('error')}")
        print("")

    if recs:
        print("=" * 78)
        print("RECOMMENDED CONFIGURATION (measured, not estimated)")
        print("=" * 78)
        print(f"  warm generation  p50 {recs['warm_p50_s']}s  p90 {recs['warm_p90_s']}s  "
              f"p99 {recs['warm_p99_s']}s")
        if recs.get("delay_p90_s") is not None:
            print(f"  queue delay      p50 {recs['delay_p50_s']}s  p90 {recs['delay_p90_s']}s")
        if recs.get("cold_start_measured") is not None:
            print(f"  cold-start adder {recs['cold_start_measured']}s (measured)")
        else:
            print(f"  cold-start adder {recs['cold_start_adder_s']}s "
                  f"(assumed; run with --repeat >= 2 to measure it)")
        print("")
        print(f"  WANGP_DEFAULT_BUDGET_S = {recs['WANGP_DEFAULT_BUDGET_S']}"
              f"   # p99 x 1.3")
        print(f"  WANGP_MAX_BUDGET_S     = {recs['WANGP_MAX_BUDGET_S']}"
              f"   # slowest cell p99 x 1.5")
        print(f"  WANGP_CANCEL_GRACE_S   = {recs['WANGP_CANCEL_GRACE_S']}")
        print(f"  endpoint executionTimeout = {recs['executionTimeout_s']}s "
              f"({recs['executionTimeout_ms']} ms)")
        print(f"      = max_budget {recs['WANGP_MAX_BUDGET_S']} + cancel grace "
              f"{recs['WANGP_CANCEL_GRACE_S']} + 60s probe/upload/cleanup, rounded up.")
        print("      Keep it strictly above the handler budget so the cooperative cancel")
        print("      always wins the race against RunPod's hard kill.")
        print("")
        print(f"  cost per clip: ${recs['usd_per_clip_p50']} (p50), "
              f"${recs['usd_per_clip_p90']} (p90) on {args.gpu} @ ${price}/s")
        print(f"  cold-start adder: +${recs['usd_cold_start_adder']} per cold worker")
        print("")

    payload = {
        "context": runner.context(),
        "gpu": args.gpu,
        "price_per_second": price,
        "matrix": {key: list(values) for key, values in axes.items()},
        "repeat": repeat,
        "profile": args.profile,
        "elapsed_s": elapsed,
        "runs": runs,
        "summaries": [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in summaries
        ],
        "recommendations": recs,
    }
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"raw results written to {args.json_out}")
    if args.csv_out:
        write_csv(args.csv_out, runs)
        print(f"per-run CSV written to {args.csv_out}")

    return EXIT_FAILED if failures else EXIT_OK


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(EXIT_INTERRUPTED) from None
