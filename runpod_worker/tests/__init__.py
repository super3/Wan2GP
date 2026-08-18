"""CPU-only test package for the RunPod worker.

Its presence makes pytest resolve these files as ``runpod_worker.tests.*``,
which puts the repo root on ``sys.path`` so ``from runpod_worker import ...``
works without a conftest hack.
"""
