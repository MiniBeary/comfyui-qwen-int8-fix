"""Isolated smoke-test of comfy-kitchen's native int8 kernels.

Reconstructed from the debugging notes in Comfy-Org/ComfyUI issue #16443
(disk was cleaned before publishing). Call signatures target comfy-kitchen
0.2.35; if your version moved the API, adjust the calls accordingly.

The point of this script: on affected setups it PASSES while the full
ComfyUI pipeline still crashes - i.e. the kernels behave in isolation and
the failure is an interaction with ComfyUI's model management.

Run: python repro_int8.py
"""
import torch

try:
    import comfy_kitchen as ck
except ImportError:
    raise SystemExit("pip install comfy-kitchen first")

torch.manual_seed(0)
DEV = "cuda"


def run_shape(m: int, n: int, k: int, convrot: bool):
    # int8 weights with per-channel F32 scales, as shipped in
    # qwen_image_2.1_int8_convrot.safetensors (4096x4096 and 12288x4096)
    w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device=DEV)
    s = torch.rand(n, dtype=torch.float32, device=DEV) + 0.1
    x = torch.randn(m, k, dtype=torch.bfloat16, device=DEV)
    out = ck.int8_linear(x, w, s, convrot=convrot, convrot_groupsize=256)
    torch.cuda.synchronize()
    return out


CASES = [
    (1, 4096, 4096, True),      # single-token path
    (1024, 12288, 4096, True),  # convrot weight shape
    (4096, 4096, 4096, False),
]

if __name__ == "__main__":
    for m, n, k, convrot in CASES:
        run_shape(m, n, k, convrot)
        print(f"ok: m={m} n={n} k={k} convrot={convrot}")
    print("PASS: isolated int8 kernels OK")
