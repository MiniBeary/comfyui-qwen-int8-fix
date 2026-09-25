# ComfyUI native quant-kernel crash — diagnosis and workaround

**Fixes:** `Fatal Python error: Aborted` / `CUDA error: an illegal memory access was encountered` during model init or the first sampling steps, when running quantized diffusion models through comfy-kitchen's **native int8/fp8 kernels** — e.g. `qwen_image_2.1_int8_convrot` on an RTX 3080 Ti 12 GB.

**Upstream report with full root-cause analysis: [Comfy-Org/ComfyUI #16443](https://github.com/Comfy-Org/ComfyUI/issues/16443) (comment by @MiniBeary)**

## TL;DR

| | |
|---|---|
| Symptom | `Fatal Python error: Aborted` at `Model Initializing` or first sampler steps (log right before: `Model QwenImage21 ... Staged.`) |
| With `--disable-cuda-malloc` | clean `torch.AcceleratorError: CUDA error: an illegal memory access was encountered` |
| With `CUDA_LAUNCH_BLOCKING=1` | `cuBLAS error: 14` (INVALID_VALUE) at `torch.ops.comfy_kitchen.int8_linear` |
| Kernels in isolation | **pass** (see [`repro/`](repro/)) — same shapes, same streams, no failure |
| Full pipeline | crashes → interaction between native ops and ComfyUI's weight cast / offload / streaming machinery |
| Full disable of native backend | stable but ~20× slower (1–2 min/it) |
| **This patch** | **stable, ~1.2 s/it at 1024×1024 / 25 steps on a 3080 Ti** |

## The workaround

A single-file custom node that patches `comfy.ops.get_disabled_quant_formats` to add every native-kernel-backed quant format to the "disabled" set. ComfyUI then runs those layers through its **emulated** quant implementations, while attention (`sol_attn`), `rms_rope`, `rms_adaln` and the rest of the native backend stay untouched — most of the speed survives.

```
custom_nodes/
└── zz_hybrid_quant_patch.py   ← drop this file here
```

Restart ComfyUI. That's it.

### Tested

- ComfyUI 0.37.0, comfy-kitchen 0.2.35, torch 2.14.0+cu130, driver 596.21, RTX 3080 Ti (single GPU, sm_86)
- 3 consecutive full runs: 2× text-to-image + 1× image edit (with `QwenImage21Cache device=auto`) — no crashes, correct output
- Applies to int8 (`int8_tensorwise`, `convrot_w4a4`, `asym_w4a8_int8`) and fp8 (`float8_e4m3fn`, `float8_e5m2`, `nvfp4`, `mxfp8`) formats

## Repro scripts

[`repro/`](repro/) contains the isolation experiments that ruled the kernels themselves out:

- `repro_int8.py` — the exact weight shapes from the safetensors (4096×4096, 12288×4096, per-channel F32 scales, `convrot=True`, groupsize 256, m ∈ {1, 1024, 4096}) — passes
- `repro_cublas.py` — same calls with `CUDA_LAUNCH_BLOCKING=1` to surface the real error — passes in isolation
- `repro_streams.py` — interleaved calls on two CUDA streams — passes

Scripts were reconstructed from the debugging notes (the working disk was cleaned before this repo was published); they target comfy-kitchen 0.2.35 — adjust call signatures if your version differs.

## For maintainers

The per-format `disabled` mechanism (`get_disabled_quant_formats`) is a clean escape hatch. It might be worth honoring an env var (e.g. `COMFY_DISABLE_QUANT_FORMATS=int8_tensorwise,convrot_w4a4`) so affected users can drop specific native formats without disabling the whole backend or patching code.

## License

MIT
