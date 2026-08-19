# Temporary benchmark artifact — delete before merge

`sageattention-2.2.0-cp310-cp310-linux_x86_64.whl` is a SageAttention 2.2.0
wheel compiled for **sm_120 only** (Blackwell: RTX PRO 6000, RTX 5090) against
torch 2.13/cu13, python 3.10.

It exists only so a benchmark pod can `pip install` it without a 40-90 minute
on-pod compile. It is NOT part of the worker image (the image builds its own
via `--build-arg WITH_SAGE=1`) and must be deleted once the accelerator
benchmark is done. It will not load on Ampere/Ada hosts (A40, L40S, A100).
