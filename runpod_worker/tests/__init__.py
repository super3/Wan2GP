"""Tier-1 test suite for the RunPod worker: CPU only.

Nothing here imports torch, wgp, CUDA, gradio, numpy, boto3 or runpod, touches a
GPU, needs model weights, or makes an outbound network request. That is the whole
point of the ``schema`` / ``media_in`` / ``media_out`` / ``config`` / ``errors`` /
``obs`` split — it is what makes this suite runnable on a plain GitHub runner with
only ``pytest`` installed.

    pytest runpod_worker/tests -v

Run it scoped like that, not as a bare ``pytest`` from the repo root: WanGP ships
torch-importing files named ``test_*.py`` (``shared/mps/test_*.py``,
``models/wan/ovi/modules/mmaudio/test_vae.py``) that fail at collection time on a
CPU box.

What each file guards:

``test_wgp_config_drift.py``
    Re-derives, from ``wgp.py`` source, the ``server_config[...]`` keys that are
    read at import time without a guard, and asserts ``config.REQUIRED_WGP_KEYS``
    covers them. This is the regression test for ``KeyError: 'attention_mode'``,
    the one failure that stops the worker from booting at all.

``test_schema.py``
    Request validation: the four ``minimax_h3`` variants, the frame lattice
    (107 / 17 / 5), seed resolution, the letter-combination rules, the forbidden
    and unknown key guards, the LoRA guards, and ``ATTACHMENT_KEYS`` drift
    against ``wgp.py``.

``test_media.py``
    Magic-byte sniffing (extension comes from content, never from the caller),
    the byte caps, the volume path-traversal guard, and the output transport
    chain — including the ``rp_upload`` local-path fallback, which returns a
    filesystem path instead of raising.

``test_handler.py``
    The whole job path end to end -- ``schema.parse`` -> ``media_in.materialize``
    -> ``engine.run`` -> ``media_out.deliver`` -> response envelope -- with the
    real modules and a stubbed engine. Every error code, its ``retryable`` and
    ``refresh_worker`` pairing, the idempotent replay, and ``test_input.json``
    itself. This is the file that catches drift *between* modules.

``test_engine.py``
    ``engine.run``'s event drain loop against a fake ``SessionJob``: the
    termination condition, the cooperative cancel on budget overrun, the
    ``backend_fatal`` latch when a cancel never lands, the poison scan, and the
    between-jobs truncation of the lists WanGP appends to forever.

The presence of this file makes pytest resolve the modules as
``runpod_worker.tests.*``, which puts the repo root on ``sys.path`` so
``from runpod_worker import ...`` works with no conftest hack.
"""
