#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Serve index.html over http and open it.

Why this exists: opened as a file:// page, the tester works in Chrome but
Safari and Firefox refuse its cross-origin requests to api.runpod.ai outright
-- the button appears to do nothing. Served over http there is no file://
special case in any browser.

    python3 runpod_worker/webdemo/serve.py            # :8000, opens a browser
    python3 runpod_worker/webdemo/serve.py --port 9000 --no-open

RunPod answers preflight with access-control-allow-origin: *, so this is a
plain static server -- it never sees or proxies your API key.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true", help="do not launch a browser")
    args = ap.parse_args()

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(HERE))
    # allow_reuse_address: without it a restart inside TIME_WAIT dies on
    # "Address already in use", which reads as a broken script.
    socketserver.TCPServer.allow_reuse_address = True
    try:
        server = socketserver.TCPServer(("127.0.0.1", args.port), handler)
    except OSError as exc:
        print(f"could not bind port {args.port}: {exc}\ntry --port {args.port + 1}")
        return 1

    url = f"http://localhost:{args.port}/index.html"
    print(f"serving {HERE} at {url}\npress ctrl-c to stop")
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
