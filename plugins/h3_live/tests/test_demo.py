# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The demo server in its dry-run mode (synthetic clips, no WanGP, no GPU; ffmpeg on the PATH):

    python -m pytest tests -q
"""

import importlib.util
import json
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h3_live_demo_under_test", ROOT / "demo" / "h3live_demo.py")
demo_module = importlib.util.module_from_spec(spec)
sys.modules["h3_live_demo_under_test"] = demo_module
spec.loader.exec_module(demo_module)


def fetch(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as reply:
        return reply.status, reply.headers, reply.read()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg on the PATH")
def test_the_demo_streams_an_endless_paced_session(tmp_path):
    args = demo_module.parse_args(["--dry-run", "--wangp", str(tmp_path), "--resolution", "64x48", "--frames", "12", "--encoder", "x264",
                                   "--lead", "1", "--window", "3", "--", "--profile", "1"])
    assert args.wangp_args == ["--profile", "1"]
    demo = demo_module.Demo(args)
    server = ThreadingHTTPServer(("127.0.0.1", 0), demo_module.make_handler(demo))
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    worker = threading.Thread(target=demo.run, daemon=True)
    worker.start()
    try:
        status, headers, body = fetch(port, "/")
        assert status == 200 and b"hls.js" in body and headers["Content-Type"].startswith("text/html")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            stats = json.loads(fetch(port, "/stats.json")[2])
            if stats["session"] and stats["session"]["clips"] >= 6:
                break
            time.sleep(0.2)
        assert stats["state"] == "live" and stats["session"]["clips"] >= 6 and stats["session"]["generation_x"] > 1
        seeds = [record["seed"] for record in stats["recent"]]
        assert len(set(seeds)) == len(seeds)  # a new seed per clip
        session = stats["session"]["id"]
        status, headers, playlist = fetch(port, f"/live/{session}/live.m3u8")
        assert headers["Cache-Control"] == "no-store" and playlist.decode().count("#EXTINF") == 3
        newest = playlist.decode().strip().splitlines()[-1]
        status, headers, segment = fetch(port, f"/live/{session}/{newest}")
        assert status == 200 and headers["Content-Type"] == "video/mp2t" and segment[:1] == b"\x47"  # an MPEG-TS packet
        for path in ("/live/../../etc/passwd", f"/live/{session}/session.json", f"/live/{session}/clip_00000.mp4", "/live/other/live.m3u8", "/nothing"):
            with pytest.raises(urllib.error.HTTPError) as refused:
                fetch(port, path)
            assert refused.value.code == 404
    finally:
        demo.request_stop()
        worker.join(30)
        server.shutdown()
    assert not worker.is_alive() and demo.state == "stopped"
