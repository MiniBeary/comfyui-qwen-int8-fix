"""Surface the real CUDA error with CUDA_LAUNCH_BLOCKING=1.

Reconstructed from the debugging notes in Comfy-Org/ComfyUI issue #16443.
Without launch-blocking, the sticky illegal-memory-access surfaces one step
later at an unrelated allocation; with it, the failing call is reported
directly (on the affected setup: `cuBLAS error: 14` / INVALID_VALUE from
torch.ops.comfy_kitchen.int8_linear).

Run:  python repro_cublas.py
(or:  set CUDA_LAUNCH_BLOCKING=1 && python repro_int8.py)
"""
import os

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import torch

import repro_int8 as base

if __name__ == "__main__":
    for m, n, k, convrot in base.CASES:
        try:
            base.run_shape(m, n, k, convrot)
            print(f"ok: m={m} n={n} k={k} convrot={convrot}")
        except RuntimeError as e:
            print(f"FAILED at m={m}: {e}")
            raise
    print("PASS: no error surfaced")
