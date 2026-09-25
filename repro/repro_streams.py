"""Interleaved int8 kernels on two CUDA streams.

Reconstructed from the debugging notes in Comfy-Org/ComfyUI issue #16443.
Stream confusion was one suspect (ComfyUI stages weights across streams);
this test interleaves the same kernels on two streams and PASSES on
affected setups - another data point that the bug needs the full pipeline.

Run: python repro_streams.py
"""
import torch

import repro_int8 as base

if __name__ == "__main__":
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()

    for i, (m, n, k, convrot) in enumerate(base.CASES * 2):
        stream = s1 if i % 2 == 0 else s2
        with torch.cuda.stream(stream):
            base.run_shape(m, n, k, convrot)
        print(f"ok on stream {i % 2 + 1}: m={m} n={n}")

    torch.cuda.synchronize()
    print("PASS: two-stream interleaving OK")
