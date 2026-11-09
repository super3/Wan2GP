# SPDX-License-Identifier: Apache-2.0
"""Start and stop the USP helper processes from inside the WanGP process (rank 0)."""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Listener
from typing import Callable

#: The world size must divide MiniMax H3's 56 attention heads.
VALID_WORLD_SIZES = (2, 4, 7, 8)
#: WanGP flags a helper inherits from the UI process (everything else is UI- or server-only).
INHERITED_FLAGS = {
    "--config", "--loras", "--lora-config", "--profile", "--attention", "--compile", "--fp16", "--bf16",
    "--vae-config", "--perc-reserved-mem-max", "--vram-safety-coefficient", "--max-vram", "--max-ram",
    "--preload", "--offline", "--convrot", "--ram-allocator", "--vram-allocator", "--settings", "--verbose",
}
HELPER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "helper.py")
CONNECT_TIMEOUT_S = 180
BOOT_TIMEOUT_S = 900


@dataclass
class Running:
    world: int
    rank0_gpu: int
    helper_gpus: list[int]
    names: list[str]
    procs: list[subprocess.Popen]
    logs: list[str]
    leader: object
    saved_env: dict = field(default_factory=dict)


_RUNNING: Running | None = None
_LOCK = threading.Lock()


def running() -> Running | None:
    return _RUNNING


def leader():
    return _RUNNING.leader if _RUNNING is not None else None


# ---------------------------------------------------------------------------
# GPU inventory
# ---------------------------------------------------------------------------

def gpus() -> list[dict]:
    """The GPUs this process can see: index, name, memory, and P2P access from every other one."""
    import torch

    if not torch.cuda.is_available():
        return []
    count = torch.cuda.device_count()
    out = []
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        out.append({
            "index": index,
            "name": props.name,
            "memory_gb": round(props.total_memory / 2**30, 1),
            "p2p": {other: bool(torch.cuda.can_device_access_peer(index, other)) for other in range(count) if other != index},
        })
    return out


def physical_id(index: int) -> str:
    """The ``CUDA_VISIBLE_DEVICES`` entry for this process's device ``index``."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        return str(index)
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    return entries[index]


def inherited_cli_args(argv: list[str]) -> list[str]:
    """The UI process's WanGP flags a helper must share (config folder, profile, attention, ...)."""
    out, index = [], 0
    while index < len(argv):
        token = argv[index]
        flag = token.split("=", 1)[0]
        if flag in INHERITED_FLAGS:
            out.append(token)
            if "=" not in token and index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                out.append(argv[index + 1])
                index += 1
        index += 1
    return out


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _tail(path: str, lines: int = 25) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-lines:])
    except OSError:
        return ""


def _accept(listener: Listener, procs: list[subprocess.Popen], logs: list[str], count: int, timeout_s: float) -> list:
    """Accept ``count`` connections, failing fast when a helper process dies."""
    conns, errors = [], []

    def worker():
        try:
            for _ in range(count):
                conns.append(listener.accept())
        except Exception as exc:  # listener closed on timeout
            errors.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout_s
    while thread.is_alive() and time.monotonic() < deadline:
        for proc, log in zip(procs, logs, strict=True):
            if proc.poll() is not None:
                raise RuntimeError(f"GPU helper exited with code {proc.returncode} before connecting:\n{_tail(log)}")
        thread.join(0.5)
    if thread.is_alive() or len(conns) < count:
        raise RuntimeError(f"GPU helpers did not connect within {timeout_s:.0f} s")
    return conns


def _recv_with_deadline(conn, proc: subprocess.Popen, log: str, timeout_s: float, what: str):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if conn.poll(1.0):
            return conn.recv()
        if proc.poll() is not None:
            raise RuntimeError(f"GPU helper exited with code {proc.returncode} while {what}:\n{_tail(log)}")
    raise RuntimeError(f"GPU helper did not finish {what} within {timeout_s:.0f} s:\n{_tail(log)}")


# ---------------------------------------------------------------------------
# Start / stop
# ---------------------------------------------------------------------------

def start(world: int, rank0_gpu: int, helper_gpus: list[int], root: str, cli_args: list[str],
          status: Callable[[str], None] = lambda text: None) -> Running:
    """Spawn ``world - 1`` helpers, join them in one NCCL group, and gate H3 through them."""
    global _RUNNING

    from . import sync, usp

    with _LOCK:
        if _RUNNING is not None:
            raise RuntimeError("Multi-GPU is already running")
        if world not in VALID_WORLD_SIZES:
            raise ValueError(f"{world} GPUs: MiniMax H3's 56 attention heads split over {', '.join(map(str, VALID_WORLD_SIZES))} GPUs only")
        if len(helper_gpus) != world - 1 or rank0_gpu in helper_gpus or len(set(helper_gpus)) != len(helper_gpus):
            raise ValueError(f"need {world - 1} distinct helper GPUs other than GPU {rank0_gpu}")

        authkey = secrets.token_bytes(32)
        listener = Listener(("127.0.0.1", 0), authkey=authkey)
        master_port = _free_port()
        log_dir = tempfile.mkdtemp(prefix="wangp_usp_logs_")
        procs, logs, saved_env = [], [], {}
        try:
            for rank, gpu in enumerate(helper_gpus, start=1):
                log = os.path.join(log_dir, f"helper_rank{rank}_gpu{gpu}.log")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=physical_id(gpu), USP_AUTHKEY=authkey.hex(),
                           PYTHONUNBUFFERED="1")
                for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
                    env.pop(key, None)
                command = [sys.executable, HELPER_PATH, "--root", root,
                           "--address", f"127.0.0.1:{listener.address[1]}", "--rank", str(rank),
                           "--world", str(world), "--master-port", str(master_port),
                           "--cli-args", json.dumps(cli_args)]
                with open(log, "w", encoding="utf-8") as handle:
                    procs.append(subprocess.Popen(command, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT))
                logs.append(log)
            status(f"Started {len(procs)} helper process(es); waiting for them to connect…")

            conns = _accept(listener, procs, logs, len(procs), CONNECT_TIMEOUT_S)
            by_rank, names = {}, {}
            for conn in conns:
                hello = conn.recv()
                by_rank[hello[1]] = conn
                names[hello[1]] = hello[2]
            ordered = [by_rank[rank] for rank in range(1, world)]

            saved_env = {key: os.environ.get(key) for key in
                         ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", "TORCH_NCCL_ASYNC_ERROR_HANDLING")}
            os.environ.update(RANK="0", WORLD_SIZE=str(world), LOCAL_RANK=str(rank0_gpu),
                              MASTER_ADDR="127.0.0.1", MASTER_PORT=str(master_port))
            # A collective that times out raises instead of killing the WanGP process.
            os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "2")
            status("Joining the GPUs in one NCCL group…")
            usp.activate(timeout_s=sync.NCCL_TIMEOUT_S)

            status("Helpers are loading WanGP (about a minute)…")
            for rank, (conn, proc, log) in enumerate(zip(ordered, procs, logs, strict=True), start=1):
                reply = _recv_with_deadline(conn, proc, log, BOOT_TIMEOUT_S, "loading WanGP")
                if reply[0] != "booted":
                    raise RuntimeError(f"GPU helper {rank} failed to load WanGP:\n{reply[1] if len(reply) > 1 else reply}")

            lead = sync.Leader(ordered)
            sync.install(lead)
            import torch

            all_names = [torch.cuda.get_device_name(rank0_gpu)] + [names[rank] for rank in range(1, world)]
            _RUNNING = Running(world, rank0_gpu, list(helper_gpus), all_names, procs, logs, lead, saved_env)
            return _RUNNING
        except BaseException:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
            try:
                sync.uninstall()
                usp.deactivate()
            except Exception:
                pass
            _restore_env(saved_env)
            raise
        finally:
            listener.close()


def stop(timeout_s: float = 20.0) -> None:
    """Shut the helpers down and restore single-GPU H3."""
    global _RUNNING

    from . import sync, usp

    with _LOCK:
        current = _RUNNING
        if current is None:
            return
        _RUNNING = None
        current.leader.shutdown()
        deadline = time.monotonic() + timeout_s
        for proc in current.procs:
            try:
                proc.wait(max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                proc.kill()
        sync.uninstall()
        try:
            usp.deactivate()
        finally:
            _restore_env(current.saved_env)


def _restore_env(saved: dict) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
