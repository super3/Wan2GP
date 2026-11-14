# SPDX-License-Identifier: Apache-2.0
"""Decode worker: one process per extra decode GPU, holding only a VAE, for the multi-device decode of `streaming`.

The main process starts it with ``CUDA_VISIBLE_DEVICES`` set to one GPU (or none, for a CPU slot), so every kernel this process
launches runs on one device from one thread. Kernel libraries that keep per-process state (WGPKernels 1.4.3 configures its INT8
GEMM for the first GPU it meets only, and shares buffers between threads) then behave as they do in a single-GPU WanGP.

The worker builds its VAE from the recipe the main process recorded when it loaded its own (`streaming._record_recipe`): the same
loader and arguments, inside a headless WanGP session started with the main process's flags when the recipe says so, so the
weights, quantization and kernel backends match. It then decodes the clips it is sent, one at a time, and returns them.

Protocol (multiprocessing.connection, authenticated; tensors travel as raw bytes, never pickled):
  main -> worker  ("load", recipe, settings)                  worker -> main  ("ready", info) | ("error", summary, traceback)
  main -> worker  ("decode", meta, attrs, grad, inference, shared_state) + bytes   worker -> main  ("clip", meta) + bytes | ("error", summary, traceback)
  main -> worker  ("exit",)
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback

_DTYPES = {"float32": "float32", "float16": "float16", "bfloat16": "bfloat16", "uint8": "uint8", "int16": "int16"}


def pack(tensor):
    """(meta, buffer) of a tensor: its shape and dtype, and its bytes (a CPU copy, contiguous)."""
    import torch

    tensor = tensor.detach().to("cpu").contiguous()
    meta = {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}
    # one flat byte buffer: Connection.send_bytes frames a multi-dimensional buffer by its first dimension only
    return meta, tensor.reshape(-1).view(torch.uint8).numpy() if tensor.numel() else b""


def unpack(meta, buffer):
    """The tensor of pack()'s (meta, buffer), owning its memory."""
    import torch

    dtype = getattr(torch, _DTYPES[meta["dtype"]])
    if not meta["shape"] or 0 in meta["shape"]:
        return torch.empty(meta["shape"], dtype=dtype)
    return torch.frombuffer(bytearray(buffer), dtype=dtype).view(meta["shape"])


def failure(exc: BaseException):
    """("error", one-line summary, traceback): the summary leads, so a truncated report still names the exception."""
    return ("error", f"{type(exc).__name__}: {exc}"[:400], traceback.format_exc(limit=12))


def resolve(path: str):
    """`module.path:attribute` -> the attribute."""
    module_name, _, attribute = path.partition(":")
    value = importlib.import_module(module_name)
    for part in attribute.split("."):
        value = getattr(value, part)
    return value


def build_vae(recipe: dict, settings: dict):
    """The VAE of a recipe, in a headless WanGP session first when it asks for one (flags and kernel settings of the main process)."""
    if recipe.get("wangp"):
        import tempfile

        from shared.api import init

        session = init(root=settings["root"], output_dir=tempfile.mkdtemp(prefix="wangp_decode_worker_"),
                       cli_args=settings.get("cli_args", []), console_output=False)
        wgp = session._ensure_runtime().module
        for setter, key in (("apply_int8_kernel_setting", "int8_kernels"), ("apply_fp8_kernel_setting", "fp8_kernels")):
            if settings.get(key) is not None and hasattr(wgp, setter):
                getattr(wgp, setter)(settings[key])
    for path in reversed(recipe.get("sys_path", [])):
        if path not in sys.path:
            sys.path.insert(0, path)
    factory = resolve(recipe["factory"])
    vae = factory(*recipe.get("args", []), **recipe.get("kwargs", {}))
    return vae.eval()


def serve(conn, device_name: str) -> int:
    import torch

    message = conn.recv()
    if message[0] != "load":
        conn.send(("error", f"expected load, got {message[0]!r}"))
        return 1
    _, recipe, settings = message
    try:
        device = torch.device(device_name)
        vae = build_vae(recipe, settings)
        with torch.inference_mode(False):
            vae.to(device)
        name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        conn.send(("ready", {"device": str(device), "name": name, "pid": os.getpid()}))
    except Exception as exc:  # noqa: BLE001 - reported to the main process, which falls back to one device
        conn.send(failure(exc))
        return 1
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return 0
        if message[0] == "exit":
            return 0
        if message[0] != "decode":
            conn.send(("error", f"unknown request {message[0]!r}"))
            continue
        _, meta, attrs, grad_enabled, inference, shared_state = message
        try:
            if shared_state:  # the generation's attention settings (WanGP sets them in mmgp's shared state at each generation)
                from mmgp import offload

                offload.shared_state.update(shared_state)
            z = unpack(meta, conn.recv_bytes()).to(device)
            for key, value in attrs.items():
                vae.__dict__[key] = value
            vae.decoder.__dict__["_interrupt"] = False
            mode = torch.inference_mode() if inference else torch.set_grad_enabled(grad_enabled)
            with mode:
                clip = vae._decode_clip(z)
            meta, buffer = pack(clip)
            conn.send(("clip", meta))
            conn.send_bytes(buffer)
        except Exception as exc:  # noqa: BLE001
            conn.send(failure(exc))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--address", required=True, help="host:port of the main process's listener")
    parser.add_argument("--device", default="cuda:0", help="this process's device (cuda:0 is the GPU CUDA_VISIBLE_DEVICES selects)")
    parser.add_argument("--root", default=None, help="WanGP folder, put first on sys.path")
    args = parser.parse_args(argv)
    plugins_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in (args.root, plugins_dir):
        if path and path not in sys.path:
            sys.path.insert(0, path)
    if args.root:
        os.chdir(args.root)
    from multiprocessing.connection import Client

    host, port = args.address.rsplit(":", 1)
    conn = Client((host, int(port)), authkey=bytes.fromhex(os.environ.pop("DECODE_WORKER_AUTHKEY")))
    try:
        return serve(conn, args.device)
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
