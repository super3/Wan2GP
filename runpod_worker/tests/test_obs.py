"""Tests for the observability additions: the log ring and the log shipper.

Motivated by an operational fact: RunPod exposes NO read API for serverless
worker container logs (the console view is the only reader, and
``/v2/{endpoint}/logs`` is a worker-key ingest route). The ring makes worker
history retrievable through the job status API; the shipper makes boot-time
deaths visible at an operator-supplied URL. Both must be pure stdlib and must
never raise into the worker.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from runpod_worker import handler, obs


@pytest.fixture(autouse=True)
def fresh_ring(monkeypatch):
    """Every test starts with an empty, default-sized ring and no shipper URL."""
    monkeypatch.delenv("WORKER_LOG_RING", raising=False)
    monkeypatch.delenv("LOG_SHIP_URL", raising=False)
    obs.reset_ring()
    yield
    obs.reset_ring()


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------


def test_ring_captures_emitted_records():
    obs.LOG.info("ring_probe", n=1)
    obs.LOG.warn("ring_probe", n=2)
    tail = obs.ring_tail()
    events = [json.loads(line)["n"] for line in tail if "ring_probe" in line]
    assert events[-2:] == [1, 2]


def test_ring_tail_limit_and_order():
    for i in range(5):
        obs.LOG.info("ordered", i=i)
    tail = [json.loads(line)["i"] for line in obs.ring_tail(3) if '"ordered"' in line]
    assert tail == [2, 3, 4]


def test_ring_respects_maxlen(monkeypatch):
    monkeypatch.setenv("WORKER_LOG_RING", "4")
    obs.reset_ring()
    for i in range(10):
        obs.LOG.info("overflow", i=i)
    tail = obs.ring_tail()
    assert len(tail) == 4
    assert [json.loads(line)["i"] for line in tail] == [6, 7, 8, 9]


def test_ring_disabled_with_zero(monkeypatch):
    monkeypatch.setenv("WORKER_LOG_RING", "0")
    obs.reset_ring()
    obs.LOG.info("dropped")
    assert obs.ring_tail() == []


def test_ring_skips_records_below_level(monkeypatch):
    monkeypatch.setenv("WORKER_LOG_LEVEL", "warn")
    obs.LOG.info("too_quiet")
    obs.LOG.error("loud")
    tail = "\n".join(obs.ring_tail())
    assert "too_quiet" not in tail
    assert "loud" in tail


# ---------------------------------------------------------------------------
# Response wiring
# ---------------------------------------------------------------------------


def test_error_response_carries_worker_logs():
    obs.LOG.info("pre_failure_context", detail="boot said something")
    body = handler.error_response("job-1", "internal_error", "boom")
    joined = "\n".join(body["worker_logs"])
    assert "pre_failure_context" in joined


def test_error_response_worker_logs_capped(monkeypatch):
    monkeypatch.setenv("WORKER_LOGS_TAIL", "2")
    for i in range(6):
        obs.LOG.info("filler", i=i)
    body = handler.error_response("job-1", "internal_error", "boom")
    assert len(body["worker_logs"]) == 2


def test_error_response_worker_logs_disable(monkeypatch):
    monkeypatch.setenv("WORKER_LOGS_TAIL", "0")
    obs.LOG.info("filler")
    body = handler.error_response("job-1", "internal_error", "boom")
    assert body["worker_logs"] == []


class _Req:
    model_type = "minimax_h3_fl2va_pruned"
    settings = {"seed": 7}
    resolved: dict = {}
    warnings: list = []
    profile = None

    def __init__(self, runtime):
        self.runtime = runtime


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, True),
        (1, True),
        ("true", True),
        ("YES", True),
        ("on", True),
        (False, False),
        (0, False),
        ("0", False),
        ("false", False),
        (None, False),
        ([], False),
    ],
)
def test_wants_debug_spellings(value, expected):
    assert handler._wants_debug(_Req({"debug": value})) is expected


def test_success_response_attaches_logs_only_on_debug():
    obs.LOG.info("visible_history")
    video = {"transport": "b64"}
    with_debug = handler._success_response(_Req({"debug": True}), video, {"total_s": 1.0})
    without = handler._success_response(_Req({}), video, {"total_s": 1.0})
    assert any("visible_history" in line for line in with_debug["worker_logs"])
    assert "worker_logs" not in without


# ---------------------------------------------------------------------------
# Shipper
# ---------------------------------------------------------------------------


class _Sink(BaseHTTPRequestHandler):
    received: list[dict] = []
    event = threading.Event()

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", 0))
        _Sink.received.append(json.loads(self.rfile.read(length)))
        _Sink.event.set()
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):  # silence the test log
        pass


@pytest.fixture()
def sink():
    _Sink.received = []
    _Sink.event = threading.Event()
    server = HTTPServer(("127.0.0.1", 0), _Sink)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/logs"
    server.shutdown()
    thread.join(timeout=5)


def test_shipper_posts_batches(monkeypatch, sink):
    monkeypatch.setenv("LOG_SHIP_URL", sink)
    monkeypatch.setenv("LOG_SHIP_INTERVAL_S", "0.2")
    obs.LOG.info("shipped_event", marker="abc123")
    assert _Sink.event.wait(timeout=10), "no batch arrived at the sink"
    records = [r for batch in _Sink.received for r in batch["records"]]
    assert any(r.get("event") == "shipped_event" for r in records)


def test_shipper_failure_never_raises(monkeypatch):
    # A port nothing listens on: enqueue + post must fail silently.
    monkeypatch.setenv("LOG_SHIP_URL", "http://127.0.0.1:9/logs")
    monkeypatch.setenv("LOG_SHIP_INTERVAL_S", "0.2")
    obs.LOG.info("doomed_event")  # must not raise, block, or kill the worker


def test_shipper_disabled_without_url():
    before = obs._SHIP_QUEUE.qsize()
    obs.LOG.info("not_shipped")
    assert obs._SHIP_QUEUE.qsize() == before


def test_stream_events_reach_ring_at_debug_level(monkeypatch):
    """The engine tees WanGP stream lines into LOG at debug level, so with
    WORKER_LOG_LEVEL=debug (+ LOG_SHIP_URL) mmgp's pinning prints become
    shippable. At the default level they must stay out of the ring."""
    monkeypatch.setenv("WORKER_LOG_LEVEL", "debug")
    obs.reset_ring()
    obs.LOG.debug("wangp_stream", line="stdout: Pinning data of 'text_encoder' to reserved RAM")
    assert any("Pinning data" in line for line in obs.ring_tail())
    monkeypatch.setenv("WORKER_LOG_LEVEL", "info")
    obs.reset_ring()
    obs.LOG.debug("wangp_stream", line="stdout: quiet")
    assert obs.ring_tail() == []
