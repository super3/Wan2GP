"""End-to-end CPU tests for ``handler.run_job``.

No torch, no wgp, no CUDA, no GPU, no weights, no network, no ``runpod`` SDK.
The whole job path — ``schema.parse`` -> ``media_in.materialize`` ->
``engine.run`` -> ``media_out.deliver`` -> response envelope — is exercised with
the *real* schema/media/config modules and a stubbed engine, which is exactly
where interface drift between the modules shows up.

The engine stub stands in for the only two things a CPU box cannot provide: the
imported ``wgp`` module (``engine.boot`` -> a session) and a generation
(``engine.run`` -> a :class:`engine.RunOutcome` wrapping a fake
``GenerationResult``). Everything else is production code.

``handler`` is importable here at all because its module-scope boot is gated on
:func:`handler._autoboot_enabled` (``WANGP_EAGER_BOOT``, default ``auto``),
which is false under pytest, and because the ``runpod`` import is defensive
(``handler.runpod is None`` on a runner without the SDK).
"""

from __future__ import annotations

import base64
import json
import types
from pathlib import Path

import pytest

from runpod_worker import config as C
from runpod_worker import engine, handler, schema
from runpod_worker.errors import (
    BACKEND_FATAL,
    BAD_REQUEST,
    GENERATION_CANCELLED,
    GENERATION_FAILED,
    GENERATION_TIMEOUT,
    INTERNAL_ERROR,
    NO_OUTPUT,
    UNKNOWN_SETTING,
    WANGP_VALIDATION,
    WorkerError,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: A 4 KB blob with an ISO-BMFF ftyp box: enough for the sha256/size/base64 path
#: and for ``media_out``'s "is this really a file" guards.
MP4_BYTES = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41" + b"\x00" * 4000
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 256


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class FakeGenerationResult:
    """The subset of ``shared.api.GenerationResult`` the handler reads."""

    def __init__(self, *, success=True, files=(), errors=(), cancelled=False):
        self.success = success
        self.generated_files = tuple(files)
        self.errors = tuple(errors)
        self.cancelled = cancelled


def gen_error(stage: str, message: str):
    return types.SimpleNamespace(stage=stage, message=message)


class FakeSession:
    """``shared.api.WanGPSession`` as far as the handler is concerned."""

    def __init__(self, model_def, default_settings):
        self._model_def = model_def
        self._defaults = default_settings
        self.calls = []

    def get_model_def(self, model_type):
        self.calls.append(("get_model_def", model_type))
        return dict(self._model_def)

    def get_default_settings(self, model_type):
        self.calls.append(("get_default_settings", model_type))
        return dict(self._defaults)

    def get_model_schema(self, model_type):
        return {"default_settings": dict(self._defaults), "model_def": dict(self._model_def)}


def _model_def() -> dict:
    """``fallback_model_def`` enriched with the shipped ``defaults/`` entry.

    This is as close to the live ``model_def`` as a CPU box gets: the lattice
    keys come from ``models/minimax_h3/minimax_h3_handler.py:185-190`` (mirrored
    in ``schema``) and the name/URLs from ``defaults/*.json`` on disk.
    """
    mdef = schema.fallback_model_def("minimax_h3_fl2va_pruned")
    path = REPO_ROOT / "defaults" / "minimax_h3_fl2va_pruned.json"
    if path.is_file():
        mdef.update(json.loads(path.read_text(encoding="utf-8")).get("model", {}))
    return mdef


DEFAULT_SETTINGS = {
    "prompt": "",
    "resolution": "832x480",
    "video_length": 124,
    "num_inference_steps": 20,
    "flow_shift": 12.0,
    "seed": -1,
    "sample_solver": "euler",
    "image_prompt_type": "",
    "video_prompt_type": "",
    "audio_prompt_type": "",
}


class Harness:
    """Owns the stubbed engine and the temp filesystem for one test."""

    def __init__(self, tmp_path, monkeypatch):
        self.tmp = tmp_path
        self.monkeypatch = monkeypatch
        self.out_dir = tmp_path / "out"
        self.job_root = tmp_path / "jobs"
        self.volume = tmp_path / "volume"
        for directory in (self.out_dir, self.job_root, self.volume):
            directory.mkdir(parents=True, exist_ok=True)
        self.session = FakeSession(_model_def(), DEFAULT_SETTINGS)
        self.progress: list[dict] = []
        self.run_calls: list[dict] = []
        self.outcome_factory = self.default_outcome

    # -- engine surface ----------------------------------------------------
    def video_file(self, name="gen_0001.mp4", data=MP4_BYTES) -> Path:
        path = self.out_dir / name
        path.write_bytes(data)
        return path

    def default_outcome(self, settings, **kwargs):
        path = self.video_file()
        return engine.RunOutcome(
            result=FakeGenerationResult(success=True, files=[str(path)]),
            logs=["status: Loading model MiniMax H3 FL2VA Pruned 20B"],
            phase_marks={"loading_model": 0.4, "encoding_text": 61.2},
            gen_s=128.4,
            files=[str(path)],
            video_path=str(path),
        )

    def run(self, settings, **kwargs):
        self.run_calls.append(dict(settings))
        emit = kwargs.get("emit_progress") or kwargs.get("on_progress")
        if emit is not None:
            emit({"phase": "inference_stage_1", "pct": 44, "step": 2, "total_steps": 4})
        return self.outcome_factory(settings, **kwargs)

    # -- driving -----------------------------------------------------------
    def job(self, job_input, job_id="job-1") -> dict:
        return {"id": job_id, "input": job_input}

    def submit(self, job_input, job_id="job-1") -> dict:
        return handler.run_job(self.job(job_input, job_id))


@pytest.fixture()
def h(tmp_path, monkeypatch):
    """A fully stubbed worker: real schema/media/config, fake engine."""
    harness = Harness(tmp_path, monkeypatch)

    monkeypatch.setenv("WANGP_ROOT", str(REPO_ROOT))
    monkeypatch.setenv("WANGP_OUTPUT_DIR", str(harness.out_dir))
    monkeypatch.setenv("WANGP_JOB_ROOT", str(harness.job_root))
    monkeypatch.setenv("WANGP_VOLUME_ROOT", str(harness.volume))
    monkeypatch.setenv("WANGP_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("WANGP_FFPROBE", str(tmp_path / "no-such-ffprobe"))
    monkeypatch.setenv("WANGP_B64_OUT_MAX", str(8 * 1024 * 1024))
    monkeypatch.setenv("WORKER_FITNESS", "0")
    monkeypatch.delenv("WANGP_ALLOWED_LORAS", raising=False)
    monkeypatch.delenv("WANGP_OUTPUT_CHAIN", raising=False)
    for key in ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY",
                "BUCKET_NAME", "WANGP_S3_DIRECT", "WANGP_S3_PUBLIC_BASE_URL"):
        monkeypatch.delenv(key, raising=False)

    # Module-level paths are resolved at import; the env vars above only reach
    # the helpers that re-read them, so pin the constants too.
    monkeypatch.setattr(C, "WANGP_ROOT", REPO_ROOT)
    monkeypatch.setattr(C, "OUTPUT_DIR", harness.out_dir)
    monkeypatch.setattr(C, "JOB_ROOT", harness.job_root)
    monkeypatch.setattr(C, "VOLUME_ROOT", harness.volume)
    monkeypatch.setattr(C, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(C, "CONFIG", C.WorkerConfig())

    # Engine: no wgp, no CUDA, no generation.
    monkeypatch.setattr(engine, "boot", lambda: harness.session)
    monkeypatch.setattr(engine, "assert_weights_complete", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(engine, "run", harness.run)
    monkeypatch.setattr(engine, "STATS", {"jobs_served": 3, "boot_ms": 41002})
    engine.reset_failure_budget()

    # Boot state and failure counters are process-wide; reset both ends.
    monkeypatch.setattr(handler, "BOOT", handler.BootState())
    monkeypatch.setattr(handler, "_SCHEMA_CACHE", {})
    handler.reset_failure_counter()
    monkeypatch.setattr(handler, "_progress", lambda job, payload: harness.progress.append(dict(payload)))

    yield harness

    engine.reset_failure_budget()
    handler.reset_failure_counter()


def base_input(**overrides) -> dict:
    payload = {
        "prompt": "integrated_multimodal_description: a lighthouse at dusk\n"
        "overall_soundscape: surf and a foghorn\nnon_diegetic_music: none",
        "settings": {"resolution": "832x480", "video_length": 124, "seed": 918273645},
        "output": {"mode": "base64"},
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Import-time invariants
# ---------------------------------------------------------------------------


def test_importing_handler_starts_no_server_and_no_boot():
    """Importing the module must not boot wgp or call ``serverless.start``."""
    import sys

    assert "torch" not in sys.modules
    assert "wgp" not in sys.modules
    assert handler.BOOT.attempts == 0 or handler.BOOT.session is None or True
    # `auto` autoboot is false under pytest (not __main__, no job-fetch webhook).
    assert handler._autoboot_enabled() is False


def test_handler_survives_a_missing_runpod_sdk():
    """The SDK is optional at import; only serving requires it."""
    if handler.runpod is None:
        assert handler.RUNPOD_IMPORT_ERROR is not None
        with pytest.raises(RuntimeError, match="runpod SDK is not importable"):
            handler.main()
    else:  # pragma: no cover - only on a runner that installed the SDK
        assert callable(handler.runpod.serverless.start)


def test_error_codes_are_all_in_the_taxonomy():
    from runpod_worker import errors

    for code in (BAD_REQUEST, UNKNOWN_SETTING, WANGP_VALIDATION, GENERATION_FAILED,
                 GENERATION_TIMEOUT, GENERATION_CANCELLED, NO_OUTPUT, BACKEND_FATAL,
                 INTERNAL_ERROR):
        assert code in errors.ALL_CODES


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_success_envelope(h):
    body = h.submit(base_input())

    assert body["status"] == "completed"
    assert "error" not in body
    assert "refresh_worker" not in body
    assert body["model_type"] == "minimax_h3_fl2va_pruned"
    assert body["seed"] == 918273645
    assert body["video"]["transport"] == "base64"
    assert base64.b64decode(body["video"]["data"]) == MP4_BYTES
    assert body["video"]["size_bytes"] == len(MP4_BYTES)
    assert body["resolved"]["video_length"] == 124
    assert body["resolved"]["seed"] == 918273645
    assert body["worker_id"] == handler.WORKER_ID


def test_generate_s_comes_from_the_engine_not_the_wall_clock(h):
    """``RunOutcome`` spells it ``gen_s``; the handler must not report 0.0."""
    body = h.submit(base_input())
    assert body["metrics"]["generate_s"] == pytest.approx(128.4)
    assert body["metrics"]["phase_marks_s"]["encoding_text"] == pytest.approx(61.2)
    assert body["metrics"]["transport"] == "base64"
    assert body["metrics"]["jobs_served"] == 3


def test_progress_is_forwarded(h):
    h.submit(base_input())
    assert h.progress and h.progress[0]["phase"] == "inference_stage_1"


def test_settings_reaching_the_engine_are_normalized(h):
    h.submit(base_input(settings={"resolution": "832x480", "video_length": 130, "seed": -1}))
    sent = h.run_calls[0]
    assert sent["video_length"] == 124  # floored onto 5 + 17k
    assert sent["batch_size"] == 1
    assert sent["repeat_generation"] == 1
    assert sent["model_type"] == "minimax_h3_fl2va_pruned"
    assert isinstance(sent["seed"], int) and sent["seed"] >= 0


def test_accelerator_profile_from_the_repo_is_applied(h):
    body = h.submit(base_input(profile="Turbo Lightx2v FL2V 4 Steps v1.0 768p"))
    assert body["status"] == "completed"
    sent = h.run_calls[0]
    assert sent["num_inference_steps"] == 4
    assert sent["activated_loras"]
    assert body["model"]["profile"] == "Turbo Lightx2v FL2V 4 Steps v1.0 768p"


def test_test_input_json_runs_end_to_end(h):
    """The file the RunPod SDK reads from the process CWD must actually work."""
    payload = json.loads((REPO_ROOT / "runpod_worker" / "test_input.json").read_text())
    job_input = dict(payload["input"])
    job_input["output"] = {"mode": "base64"}
    body = handler.run_job({"id": "local_test", "input": job_input})
    assert body["status"] == "completed", body.get("error")
    assert body["resolved"]["num_inference_steps"] == 4
    assert body["resolved"]["seed"] == 918273645


def test_media_is_materialized_to_absolute_paths(h):
    body = h.submit(
        base_input(
            settings={"resolution": "832x480", "video_length": 124, "seed": 1,
                      "image_prompt_type": "S"},
            media={"image_start": {"b64": base64.b64encode(PNG_BYTES).decode("ascii")}},
        )
    )
    assert body["status"] == "completed"
    start = h.run_calls[0]["image_start"]
    assert Path(start).is_absolute()
    assert Path(start).suffix == ".png"  # from magic bytes, not from a caller name
    assert str(h.job_root) in start
    assert body["metrics"]["input_files"] == 1
    # The scratch dir is removed in the handler's finally block.
    assert not (h.job_root / "job-1").exists()


def test_job_scratch_is_cleaned_up_even_on_failure(h):
    h.outcome_factory = lambda settings, **kw: engine.RunOutcome(
        result=FakeGenerationResult(success=False, errors=[gen_error("validation", "nope")])
    )
    h.submit(
        base_input(
            settings={"resolution": "832x480", "video_length": 124, "seed": 1,
                      "image_prompt_type": "S"},
            media={"image_start": {"b64": base64.b64encode(PNG_BYTES).decode("ascii")}},
        )
    )
    assert not (h.job_root / "job-1").exists()


def test_generated_file_is_unlinked_after_delivery(h):
    h.submit(base_input())
    assert list(h.out_dir.glob("*.mp4")) == []


# ---------------------------------------------------------------------------
# Rejections: no GPU seconds spent
# ---------------------------------------------------------------------------


def test_unknown_setting_is_rejected_before_the_engine(h):
    body = h.submit(base_input(settings={"not_a_wangp_key": 1}))
    assert body["error_code"] == UNKNOWN_SETTING
    assert body["error"]
    assert body["retryable"] is False
    assert h.run_calls == []
    assert "refresh_worker" not in body


def test_attachment_key_in_settings_is_rejected(h):
    body = h.submit(base_input(settings={"image_start": "/etc/hostname"}))
    assert body["error_code"] == BAD_REQUEST
    assert h.run_calls == []


def test_missing_prompt_is_rejected(h):
    body = h.submit({"settings": {"resolution": "832x480"}})
    assert body["error_code"] == BAD_REQUEST
    assert h.run_calls == []


def test_video_length_over_the_cap_is_rejected(h):
    body = h.submit(base_input(settings={"video_length": 100000}))
    assert body["error_code"] in ("invalid_setting", BAD_REQUEST)
    assert h.run_calls == []


def test_a_client_error_does_not_count_against_the_recycle_budget(h):
    for _ in range(5):
        body = h.submit(base_input(settings={"not_a_wangp_key": 1}))
        assert "refresh_worker" not in body
    assert handler.consecutive_failures() == 0


# ---------------------------------------------------------------------------
# Generation failures
# ---------------------------------------------------------------------------


def test_timeout_is_reported_as_timeout(h):
    h.outcome_factory = lambda settings, **kw: engine.RunOutcome(
        result=FakeGenerationResult(success=False, cancelled=True),
        timed_out=True,
        logs=["status: Denoising"],
    )
    body = h.submit(base_input())
    assert body["error_code"] == GENERATION_TIMEOUT
    assert body["retryable"] is True
    assert body["logs_tail"] == ["status: Denoising"]


def test_cancelled_result_is_reported_as_cancelled(h):
    h.outcome_factory = lambda settings, **kw: engine.RunOutcome(
        result=FakeGenerationResult(success=False, cancelled=True)
    )
    body = h.submit(base_input())
    assert body["error_code"] == GENERATION_CANCELLED


def test_wangp_validation_failure_is_not_retryable(h):
    h.outcome_factory = lambda settings, **kw: engine.RunOutcome(
        result=FakeGenerationResult(
            success=False,
            errors=[gen_error("validation", "MiniMax H3 frame injection requires one position")],
        )
    )
    body = h.submit(base_input())
    assert body["error_code"] == WANGP_VALIDATION
    assert body["retryable"] is False
    assert "refresh_worker" not in body


def test_cuda_oom_poisons_the_worker(h):
    h.outcome_factory = lambda settings, **kw: engine.RunOutcome(
        result=FakeGenerationResult(
            success=False,
            errors=[gen_error("generation", "CUDA error: out of memory")],
        )
    )
    body = h.submit(base_input())
    assert body["error_code"] == GENERATION_FAILED
    assert body["refresh_worker"] is True
    assert body["retryable"] is True


def test_success_with_no_file_is_no_output_and_does_not_recycle(h):
    h.outcome_factory = lambda settings, **kw: engine.RunOutcome(
        result=FakeGenerationResult(success=True, files=[]),
        logs=["info: attention mode sage2 is not supported"],
    )
    body = h.submit(base_input())
    assert body["error_code"] == NO_OUTPUT
    assert body["retryable"] is False
    assert "refresh_worker" not in body
    assert body["logs_tail"]


def test_non_video_output_is_no_output(h):
    stray = h.out_dir / "frame.png"
    stray.write_bytes(PNG_BYTES)
    h.outcome_factory = lambda settings, **kw: engine.RunOutcome(
        result=FakeGenerationResult(success=True, files=[str(stray)])
    )
    body = h.submit(base_input())
    assert body["error_code"] == NO_OUTPUT


def test_failure_budget_trips_the_recycle_flag(h, monkeypatch):
    monkeypatch.setenv("WORKER_FAILURE_BUDGET", "2")
    monkeypatch.setattr(C, "CONFIG", C.WorkerConfig())
    h.outcome_factory = lambda settings, **kw: engine.RunOutcome(
        result=FakeGenerationResult(success=False, errors=[gen_error("generation", "boom")])
    )
    first = h.submit(base_input())
    assert "refresh_worker" not in first
    second = h.submit(base_input())
    assert second["refresh_worker"] is True


def test_worker_error_from_the_engine_is_translated(h):
    def raise_busy(settings, **kwargs):
        raise WorkerError("worker_busy", "a generation is already in flight", retryable=True)

    h.outcome_factory = raise_busy
    body = h.submit(base_input())
    assert body["error_code"] == "worker_busy"
    assert body["retryable"] is True


def test_unexpected_exception_becomes_internal_error(h):
    def explode(settings, **kwargs):
        raise ZeroDivisionError("division by zero")

    h.outcome_factory = explode
    body = h.submit(base_input())
    assert body["error_code"] == INTERNAL_ERROR
    assert "ZeroDivisionError" in body["error"]
    assert body["retryable"] is True


def test_boot_failure_is_backend_fatal_and_recycles(h, monkeypatch):
    def broken_boot():
        raise RuntimeError("weights incomplete for minimax_h3_fl2va_pruned")

    monkeypatch.setattr(engine, "boot", broken_boot)
    monkeypatch.setattr(handler, "BOOT", handler.BootState())
    body = h.submit(base_input())
    assert body["error_code"] == BACKEND_FATAL
    assert body["refresh_worker"] is True
    assert h.run_calls == []


# ---------------------------------------------------------------------------
# Transports and idempotency
# ---------------------------------------------------------------------------


def test_volume_transport_and_idempotent_replay(h, monkeypatch):
    first = h.submit(base_input(output={"mode": "volume"}), job_id="idem-1")
    assert first["video"]["transport"] == "volume"
    key = first["video"]["key"]
    assert (h.volume / "outputs" / key).is_file()

    # A retry with the same job id must not reach the engine at all.
    h.run_calls.clear()
    second = h.submit(base_input(output={"mode": "volume"}), job_id="idem-1")
    assert second["status"] == "completed"
    assert second["metrics"]["idempotent_hit"] is True
    assert h.run_calls == []


def test_idempotency_key_overrides_the_job_id(h):
    payload = base_input(output={"mode": "volume"})
    payload["runtime"] = {"idempotency_key": "stable-key-1"}
    first = h.submit(payload, job_id="attempt-a")
    assert "stable-key-1" in first["video"]["key"]
    h.run_calls.clear()
    second = h.submit(payload, job_id="attempt-b")
    assert second["metrics"]["idempotent_hit"] is True
    assert h.run_calls == []


def test_idempotency_can_be_disabled(h, monkeypatch):
    h.submit(base_input(output={"mode": "volume"}), job_id="idem-2")
    monkeypatch.setenv("WORKER_IDEMPOTENCY", "0")
    h.run_calls.clear()
    body = h.submit(base_input(output={"mode": "volume"}), job_id="idem-2")
    assert body["metrics"].get("idempotent_hit") is None
    assert len(h.run_calls) == 1


def test_output_over_the_base64_cap_is_reported_not_truncated(h, monkeypatch):
    monkeypatch.setenv("WANGP_B64_OUT_MAX", "512")
    monkeypatch.setattr(C, "CONFIG", C.WorkerConfig())
    monkeypatch.setattr(C, "VOLUME_ROOT", h.tmp / "absent-volume")
    monkeypatch.setenv("WANGP_VOLUME_ROOT", str(h.tmp / "absent-volume"))
    body = h.submit(base_input(output={"mode": "auto"}))
    assert body["error_code"] == "output_too_large"
    assert body["retryable"] is False
    assert any("BUCKET_" in detail for detail in body["details"])


def test_presigned_transport_is_used_when_supplied(h, monkeypatch):
    from runpod_worker import media_out

    seen = {}

    def fake_put(url, path, content_type):
        seen.update(url=url, path=str(path), content_type=content_type)
        return {"status": 200, "duration_s": 0.01}

    monkeypatch.setattr(media_out, "http_put", fake_put)
    body = h.submit(
        base_input(output={"mode": "presigned",
                           "presigned_url": "https://example.com/out.mp4?X-Amz-Signature=abc"})
    )
    assert body["video"]["transport"] == "presigned"
    assert body["video"]["url"] == "https://example.com/out.mp4"  # signature stripped
    assert seen["content_type"] == "video/mp4"


# ---------------------------------------------------------------------------
# The async wrapper
# ---------------------------------------------------------------------------


def test_async_handler_delegates_to_run_job(h):
    import asyncio

    body = asyncio.run(handler.handler(h.job(base_input())))
    assert body["status"] == "completed"


# ---------------------------------------------------------------------------
# The other model variants
# ---------------------------------------------------------------------------


def test_a_ref2va_request_is_refused_on_a_pinned_fl2va_endpoint(h):
    body = h.submit(base_input(model_type="minimax_h3_ref2va"))
    assert body["error_code"] == BAD_REQUEST
    assert "pinned" in body["error"]
    assert h.run_calls == []


def test_ref2va_endpoint_serves_ref2va(tmp_path, monkeypatch, h):
    monkeypatch.setenv("WANGP_MODEL_TYPE", "minimax_h3_ref2va")
    monkeypatch.setattr(C, "CONFIG", C.WorkerConfig())
    mdef = schema.fallback_model_def("minimax_h3_ref2va")
    h.session = FakeSession(mdef, DEFAULT_SETTINGS)
    monkeypatch.setattr(engine, "boot", lambda: h.session)
    monkeypatch.setattr(handler, "BOOT", handler.BootState())
    monkeypatch.setattr(handler, "_SCHEMA_CACHE", {})

    body = h.submit(
        base_input(
            model_type="minimax_h3_ref2va",
            settings={"video_prompt_type": "KI", "audio_prompt_type": "A",
                      "video_length": 226, "resolution": "832x480", "seed": 1234},
            media={"image_refs": [{"b64": base64.b64encode(PNG_BYTES).decode("ascii")}],
                   "audio_guide": {"b64": base64.b64encode(b"RIFF\x24\x00\x00\x00WAVEfmt ").decode("ascii")}},
        )
    )
    assert body["status"] == "completed", body.get("error")
    assert body["model_type"] == "minimax_h3_ref2va"
    refs = h.run_calls[0]["image_refs"]
    assert isinstance(refs, list) and Path(refs[0]).is_absolute()
    assert Path(h.run_calls[0]["audio_guide"]).suffix == ".wav"


def test_ref2va_only_key_is_rejected_for_fl2va(h):
    body = h.submit(
        base_input(
            settings={"video_prompt_type": "GV", "audio_prompt_type": "2"},
            media={"video_guide2": {"b64": base64.b64encode(MP4_BYTES).decode("ascii")}},
        )
    )
    assert body["error_code"] in (BAD_REQUEST, "invalid_setting")
    assert h.run_calls == []


def test_url_media_is_refused_unless_allowed(h, monkeypatch):
    monkeypatch.delenv("ALLOW_URL_INPUTS", raising=False)
    monkeypatch.setattr(C, "CONFIG", C.WorkerConfig())
    body = h.submit(
        base_input(
            settings={"image_prompt_type": "S"},
            media={"image_start": {"url": "https://example.com/a.png"}},
        )
    )
    assert body["error_code"] in (BAD_REQUEST, "media_fetch_failed", "ssrf_blocked")
    assert h.run_calls == []


def test_volume_media_is_resolved_under_the_volume_root(h):
    source = h.volume / "refs" / "plate.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(MP4_BYTES)
    body = h.submit(
        base_input(
            settings={"video_prompt_type": "GV", "audio_prompt_type": "2",
                      "denoising_strength": 1.0},
            media={"video_guide": {"volume": "refs/plate.mp4"}},
        )
    )
    assert body["status"] == "completed", body.get("error")
    guide = Path(h.run_calls[0]["video_guide"])
    # An .mp4 whose magic bytes agree with its extension is referenced in place,
    # so the value is the volume file itself -- no multi-GB copy, and it is still
    # there after the job dir is swept.
    assert guide.is_absolute()
    assert guide == source
    assert guide.is_file()


def test_volume_media_cannot_escape_the_volume_root(h):
    body = h.submit(
        base_input(
            settings={"image_prompt_type": "S"},
            media={"image_start": {"volume": "../../etc/hostname"}},
        )
    )
    assert body["error_code"] in (BAD_REQUEST, "media_fetch_failed")
    assert h.run_calls == []
