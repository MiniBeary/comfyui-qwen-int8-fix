# zz_hybrid_quant_patch.py — force-emulate native quant kernels in ComfyUI
#
# Fixes the `illegal memory access` / `Fatal Python error: Aborted` crashes that
# occur when comfy-kitchen's native int8/fp8 kernels run inside ComfyUI's
# model-management pipeline on some RTX 30-series setups.
#
# Root cause write-up: https://github.com/Comfy-Org/ComfyUI/issues/16443
# (see the comment by @MiniBeary)
#
# Install: drop this file into ComfyUI/custom_nodes/ and restart.
# Nothing to configure. Safe to keep; if a future ComfyUI release fixes the
# native kernels, simply delete the file.
#
# The `zz_` prefix is intentional: custom nodes load alphabetically, and this
# one must patch AFTER any other custom node has already imported comfy.ops.

# Every quant format that dispatches to comfy-kitchen's native CUDA kernels.
# Layers using these formats fall back to ComfyUI's emulated implementations,
# while attention (sol_attn), rms_rope, rms_adaln etc. stay native.
FORMATS_TO_EMULATE = (
    "nvfp4",
    "mxfp8",
    "float8_e4m3fn",
    "float8_e5m2",
    "int8_tensorwise",
    "convrot_w4a4",
    "asym_w4a8_int8",
)


def _apply():
    import comfy.ops as ops

    if getattr(ops, "_zz_hybrid_quant_patch_applied", False):
        return

    _original = ops.get_disabled_quant_formats

    def get_disabled_quant_formats():
        disabled = set()
        try:
            disabled.update(_original() or ())
        except Exception:
            pass
        disabled.update(FORMATS_TO_EMULATE)
        return disabled

    ops.get_disabled_quant_formats = get_disabled_quant_formats
    ops._zz_hybrid_quant_patch_applied = True


_apply()
