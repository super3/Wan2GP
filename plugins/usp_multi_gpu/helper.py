# SPDX-License-Identifier: Apache-2.0
"""A USP helper: one extra GPU for the WanGP process that started it (rank 0).

Started by the plugin (``launcher.start``), one per extra GPU, with
``CUDA_VISIBLE_DEVICES`` set to that GPU. It joins rank 0's NCCL group, opens a
headless WanGP session on the same WanGP folder and config, then runs each task
rank 0 sends, in lockstep (``sync.Follower``). Its own output files are thrown
away: rank 0 saves the result. For the same reason it skips the final video
decode of a task's last window (``sync.install_decode_skip``).

    python plugins/usp_multi_gpu/helper.py --root <WanGP> --address 127.0.0.1:PORT \\
        --rank 1 --world 2 --master-port PORT [--cli-args '<json list>']

The control connection's key comes from the ``USP_AUTHKEY`` environment variable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--address", required=True, help="host:port of rank 0's control listener")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--cli-args", default="[]", help="JSON list of WanGP command-line arguments")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    plugins_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in (root, plugins_dir):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(root)
    os.environ.update(RANK=str(args.rank), WORLD_SIZE=str(args.world), LOCAL_RANK="0",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(args.master_port))

    from multiprocessing.connection import Client

    host, port = args.address.rsplit(":", 1)
    conn = Client((host, int(port)), authkey=bytes.fromhex(os.environ.pop("USP_AUTHKEY")))

    import torch

    conn.send(("hello", args.rank, torch.cuda.get_device_name(0)))

    from usp_multi_gpu import sync, usp

    usp.activate(timeout_s=sync.NCCL_TIMEOUT_S)  # joins rank 0's NCCL group, patches the H3 classes
    output_dir = tempfile.mkdtemp(prefix=f"wangp_usp_rank{args.rank}_")
    try:
        from shared.api import init

        session = init(root=root, output_dir=output_dir, cli_args=json.loads(args.cli_args), console_output=False)
    except Exception:
        conn.send(("boot_failed", traceback.format_exc(limit=8)))
        return 1
    follower = sync.Follower(conn)
    sync.install(follower)
    # this helper's video is thrown away: its last decode would only compete with rank 0's decode for the GPU
    sync.install_decode_skip(lambda: sync.is_last_window(session._state.get("gen")))
    conn.send(("booted",))

    def run_task(settings):
        result = session.submit_task(settings).result()
        if result.success or result.cancelled:
            return None
        return "; ".join(str(error) for error in result.errors) or "the task failed"

    def after_task():
        shutil.rmtree(output_dir, ignore_errors=True)  # rank 0 saved the result
        os.makedirs(output_dir, exist_ok=True)

    try:
        sync.serve(follower, run_task, after_task)
    except (EOFError, OSError):
        pass  # rank 0 went away
    finally:
        sync.uninstall()
        try:
            usp.deactivate()
        except Exception:
            pass
        shutil.rmtree(output_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
