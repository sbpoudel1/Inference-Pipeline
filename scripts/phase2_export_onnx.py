"""
scripts/phase2_export_onnx.py
-----------------------------
Phase 2A: Convert the pretrained PyTorch detector into an ONNX file.

What ONNX is, briefly:
    A neural network is a sequence of mathematical operations (matrix multiplies,
    additions, ReLUs, etc.) with a bunch of learned numbers (weights) attached.
    PyTorch stores this as Python objects in memory. ONNX is a standard file
    format — a binary file (.onnx) — that captures both the structure of the
    network AND the weights, in a way that's portable across languages and
    frameworks. Once exported, you can load and run the model from C++, Java,
    JavaScript, etc., with no PyTorch installed anywhere.

How export works:
    PyTorch's `torch.onnx.export()` works by TRACING — it runs the model
    once with a sample input, records every tensor operation that happens,
    and saves that recording as an ONNX graph. So we need to give it a
    sample input that has the right shape.

Output: models/detector.onnx (and models/coco_classes.json)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import torch
from torch import nn   # nn.Module is PyTorch's base class for "anything that has a forward pass"

from src.inference.model import load_detector

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
ONNX_PATH = MODEL_DIR / "detector.onnx"
CLASSES_PATH = MODEL_DIR / "coco_classes.json"


# ----- The export wrapper --------------------------------------------------
#
# Torchvision detection models have an awkward signature:
#     model(list_of_tensors) -> list_of_dicts
# That doesn't translate cleanly to ONNX, which prefers a single tensor in
# and a tuple of tensors out.
#
# So we define a thin wrapper that adapts the signature: takes one tensor,
# returns three tensors (boxes, scores, labels). The wrapper IS what we
# export, not the raw model.
#
# This is a standard pattern for ONNX export — anytime your real model has
# a weird input/output structure, wrap it.
class DetectorExportWrapper(nn.Module):
    """Single tensor in, three tensors out — ONNX-friendly signature."""

    def __init__(self, model: nn.Module):
        super().__init__()
        # Holding the original model as an attribute means PyTorch's tracing
        # will descend into it and capture all its internal operations.
        self.model = model

    def forward(self, image: torch.Tensor):
        # `image` has shape (C, H, W) and dtype float32 in [0, 1].
        # We wrap it in a single-element list because the underlying model
        # expects a list of images.
        outputs = self.model([image])
        # outputs is a list of length 1 (because we sent one image).
        # Each entry is a dict with "boxes", "scores", "labels" tensors.
        out = outputs[0]
        return out["boxes"], out["scores"], out["labels"]


def main() -> None:
    print("loading PyTorch detector...")
    model, _, class_names = load_detector()       # _ throws away the preprocess
    wrapper = DetectorExportWrapper(model).eval()  # eval() on the wrapper too

    # ----- The dummy input -------------------------------------------------
    # `torch.onnx.export` traces the model by actually running it once. We
    # give it a sample tensor with realistic shape.
    #
    # Shape (3, 480, 640) means: 3 channels, 480 rows, 640 columns.
    # The CONTENT doesn't matter — we use random numbers. What matters is
    # that the shape exercises the model's normal codepaths.
    #
    # Because we declared dynamic_axes below, the exported model will accept
    # any H, W at runtime — not just 480x640. The dummy shape is only the
    # example used during tracing.
    dummy = torch.randn(3, 480, 640)

    MODEL_DIR.mkdir(exist_ok=True)
    print(f"exporting → {ONNX_PATH}")
    print("(expect a wall of TracerWarnings — most are benign; we'll triage them)")
    print()

    # `warnings.catch_warnings()` lets us scope warning filter changes. We
    # set the simple filter to "default" so we SEE all warnings instead of
    # the default Python suppression of duplicates. This is intentional —
    # the warnings are informative.
    with warnings.catch_warnings():
        warnings.simplefilter("default")

        # ----- torch.onnx.export — the main event ----------------------
        # Parameters:
        #   wrapper         — the nn.Module to export
        #   (dummy,)        — sample inputs as a tuple (one per forward() arg)
        #   ONNX_PATH       — where to write the .onnx file
        #   input_names     — names for the graph's inputs (used by ORT later)
        #   output_names    — names for the graph's outputs
        #   dynamic_axes    — declares which tensor dimensions vary at runtime;
        #                     see below for the format
        #   opset_version=17 — which ONNX "instruction set" to target.
        #                     Higher = more ops available; the runtime must
        #                     understand the chosen opset. 17 is widely
        #                     supported by modern runtimes.
        #   do_constant_folding=True — at export time, fold constant subgraphs
        #                              into precomputed values. Smaller graph,
        #                              identical math.
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(ONNX_PATH),
            input_names=["image"],
            output_names=["boxes", "scores", "labels"],

            # dynamic_axes: a dict mapping tensor name → {dim_index: dim_name}.
            # We're saying:
            #   - "image" dim 1 (height) and dim 2 (width) can vary; we name them
            #   - "boxes" dim 0 (number of detections) can vary
            #   - same for "scores" and "labels"
            # The names ("height", "num_detections") are arbitrary; they only
            # show up in the graph's documented schema.
            dynamic_axes={
                "image": {1: "height", 2: "width"},
                "boxes": {0: "num_detections"},
                "scores": {0: "num_detections"},
                "labels": {0: "num_detections"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

    size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"\ndone. file size: {size_mb:.1f} MB")

    # ----- Sanity-check the exported file -----------------------------
    # The `onnx` Python package (separate from `onnxruntime`) can parse a
    # .onnx file as a protobuf and check it for structural validity.
    print("running onnx.checker.check_model...")
    import onnx
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)   # raises if the graph is malformed
    print("structural check passed")

    # Summarize the I/O shapes. dim_value is set when the dim is a fixed
    # integer; dim_param is set when the dim is a named dynamic axis.
    inputs = [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim])
              for i in onnx_model.graph.input]
    outputs = [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])
               for o in onnx_model.graph.output]
    print(f"  inputs:  {inputs}")
    print(f"  outputs: {outputs}")
    print(f"  nodes:   {len(onnx_model.graph.node)}")

    # ----- Save class names alongside the model -----------------------
    # The runtime container won't have torchvision installed, so it can't
    # query weights.meta["categories"]. We persist the class names as a
    # JSON file next to the .onnx file. predict_onnx.py reads it.
    CLASSES_PATH.write_text(json.dumps(class_names, indent=2))
    print(f"  wrote class names → {CLASSES_PATH.name}")


if __name__ == "__main__":
    main()
