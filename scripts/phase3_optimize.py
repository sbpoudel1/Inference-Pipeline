"""
scripts/phase3_optimize.py
--------------------------
Phase 3A: Produce two more variants of the model from the exported ONNX.

Inputs:
    models/detector.onnx                  (from phase2_export_onnx.py)

Outputs:
    models/detector_optimized.onnx        — ORT graph optimization applied
    models/detector_quantized.onnx        — dynamic INT8 quantization

What each transformation does:

GRAPH OPTIMIZATION (ORT_ENABLE_ALL):
    Pure graph rewrites that produce the SAME numerical output but with
    fewer operations. Examples:
      - constant folding: precompute parts of the graph that don't depend
        on input
      - operator fusion: collapse a Conv → BatchNorm → ReLU sequence into
        a single fused op
      - dead node elimination
    These are mathematically identical transformations. We bake them into
    a saved file so Phase 3 can benchmark "with optimization" vs "without".

DYNAMIC INT8 QUANTIZATION:
    Weights in the model are normally 32-bit floats. We cast them to 8-bit
    integers. The file shrinks ~4× (each weight is now 1 byte, not 4).
    Integer multiplication can be faster than float multiplication on many
    CPUs.

    "Dynamic" means: weights are pre-quantized once; activations (the
    intermediate tensor values) are quantized on the fly each time the
    model runs. The alternative ("static" quantization) requires running
    the model on calibration data ahead of time to learn good quantization
    ranges for activations too — better quality but more setup work.

    Side effect: some ops (NonZero, shape ops, etc.) can't be quantized at
    all, so the result is a hybrid INT8 + FP32 graph. The runtime inserts
    Quantize and Dequantize ops at the boundaries between INT8 and FP32
    regions. Those conversions cost time.
"""
from __future__ import annotations

from pathlib import Path

import onnxruntime as ort
# `quantize_dynamic` is the one-call API for dynamic quantization.
# `QuantType` enumerates the integer types you can quantize TO (QInt8 = signed,
# QUInt8 = unsigned). QUInt8 generally works well for weights.
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
SRC = MODEL_DIR / "detector.onnx"
OPT = MODEL_DIR / "detector_optimized.onnx"
QNT = MODEL_DIR / "detector_quantized.onnx"


def make_optimized() -> None:
    """
    Use ORT to load the model with full graph optimization enabled,
    then have ORT WRITE THE OPTIMIZED GRAPH back to disk.

    This is a slightly weird ORT feature: normally graph optimization
    happens silently at session-creation time. But by setting
    `optimized_model_filepath`, we ask ORT to also save the optimized
    version. Then we can reload that saved version at zero cost later.
    """
    print(f"[1/2] producing {OPT.name} via ORT_ENABLE_ALL graph optimization...")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.optimized_model_filepath = str(OPT)
    # Creating the session is the action — it loads, optimizes, and writes
    # the optimized graph to OPT in one go. We don't actually use the session.
    _ = ort.InferenceSession(str(SRC), opts, providers=["CPUExecutionProvider"])
    print(f"      → {OPT.relative_to(ROOT)}")


def make_quantized() -> None:
    """
    Apply dynamic quantization. This is a standalone tool — it reads the
    .onnx file, rewrites it with quantized weights, and writes the new file.
    No session is created here.
    """
    print(f"[2/2] producing {QNT.name} via dynamic INT8 quantization...")
    quantize_dynamic(
        model_input=str(SRC),    # input .onnx
        model_output=str(QNT),   # where to write the quantized version
        weight_type=QuantType.QUInt8,  # 8-bit unsigned integer weights
    )
    print(f"      → {QNT.relative_to(ROOT)}")


def main() -> None:
    if not SRC.exists():
        # Friendly error if Phase 2 hasn't been run yet.
        raise SystemExit(f"missing {SRC} — run phase2_export_onnx first")
    make_optimized()
    make_quantized()

    # Print all three file sizes side by side. The interesting comparison:
    # quantized is ~3× smaller than the original because each weight went
    # from 4 bytes (float32) to 1 byte (uint8).
    print("\nfile sizes:")
    for p in [SRC, OPT, QNT]:
        size = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name:35s} {size:6.1f} MB")


if __name__ == "__main__":
    main()
