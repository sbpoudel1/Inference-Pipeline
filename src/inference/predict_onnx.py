"""
src/inference/predict_onnx.py
-----------------------------
The ONNX-backed inference function.

Same `predict(image_path)` API as predict.py — but instead of running the
model through PyTorch, this version runs it through ONNX Runtime (which is
written in C++ and has no PyTorch dependency at all).

Why this file matters: it's what makes the Docker image small. The Dockerfile
ships ONLY this file (plus its tiny dependencies: numpy, pillow, onnxruntime).
PyTorch — which is huge — stays out of the runtime container entirely.

Notice: ZERO `import torch` lines in this file. Intentional.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import numpy as np                      # array library, replaces torch.Tensor here
import onnxruntime as ort               # the C++ runtime for .onnx files
from PIL import Image                   # same image library as before


# ----- Where to find the model + class names ------------------------------
#
# `Path(__file__).resolve().parents[2]` walks up:
#   __file__       = .../src/inference/predict_onnx.py
#   .parents[0]    = .../src/inference/
#   .parents[1]    = .../src/
#   .parents[2]    = .../ML Ops/             ← project root
ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = ROOT / "models" / "detector.onnx"
CLASSES_PATH = ROOT / "models" / "coco_classes.json"


class Detection(TypedDict):
    label: str
    score: float
    box: tuple[float, float, float, float]


# ----- Cached state -------------------------------------------------------
# Same lazy-loading pattern as predict.py, but the model object here is an
# `ort.InferenceSession` instead of a `torch.nn.Module`.
_session: ort.InferenceSession | None = None
_class_names: list[str] | None = None
_session_options: ort.SessionOptions | None = None


def configure_session(graph_optimization: bool = False) -> None:
    """
    Set up SessionOptions before the session is created.

    ONNX Runtime's SessionOptions object tells the runtime HOW to load and
    run the model. The most interesting knob for us is `graph_optimization_level`:
        ORT_DISABLE_ALL  — load the graph exactly as it was exported
        ORT_ENABLE_ALL   — run every available optimization pass (fusing
                            ops like Conv+BN+ReLU into one fused op, etc.)
    The Phase 3 benchmark calls this with graph_optimization=False or True
    to compare the two cases.
    """
    global _session_options, _session
    opts = ort.SessionOptions()
    if graph_optimization:
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _session_options = opts
    _session = None  # force the session to be re-created with the new options


def set_model_path(path: Path) -> None:
    """Point the backend at a different .onnx file (e.g. the quantized one)."""
    global ONNX_PATH, _session
    ONNX_PATH = path
    _session = None  # force re-load


def _ensure_loaded() -> None:
    """Load the ONNX session and class names on first call; reuse after that."""
    global _session, _class_names, _session_options
    if _session is None:
        opts = _session_options or ort.SessionOptions()
        # An InferenceSession is the runtime's loaded-and-ready model.
        # `providers` is the priority list of execution backends. CPUExecutionProvider
        # is the always-available CPU implementation. If we'd installed
        # onnxruntime-gpu we could prepend "CUDAExecutionProvider"; on macOS
        # there's also "CoreMLExecutionProvider" if you want to try it.
        _session = ort.InferenceSession(
            str(ONNX_PATH), opts, providers=["CPUExecutionProvider"]
        )
    if _class_names is None:
        # Read class names from JSON. We do this — rather than import torchvision
        # — so this file stays torch-free. The JSON was written during export.
        _class_names = json.loads(CLASSES_PATH.read_text())


def _preprocess(image: Image.Image) -> np.ndarray:
    """
    Turn a PIL image into the exact array shape the ONNX model expects.

    Two transformations matter here:

    1. Pixel values 0–255 (uint8) → 0.0–1.0 (float32).
       The model was trained on inputs in [0, 1].

    2. Axis order HWC → CHW.
       PIL/numpy use (height, width, channels) — natural for image data.
       PyTorch/ONNX use (channels, height, width) — natural for matrix math.
       This is a constant source of bugs; the fix is one transpose call.
    """
    arr = np.asarray(image, dtype=np.float32) / 255.0   # HWC, float32, [0,1]
    return arr.transpose(2, 0, 1)                       # → CHW


def predict(image_path: str | Path, score_threshold: float = 0.5) -> list[Detection]:
    """Same API as predict.py — pass in an image path, get detections back."""
    _ensure_loaded()
    assert _session is not None and _class_names is not None

    # Decode the image into a PIL object, then preprocess into a numpy CHW
    # float array. Same first two steps as the PyTorch version, just using
    # numpy instead of torchvision transforms.
    image = Image.open(image_path).convert("RGB")
    x = _preprocess(image)

    # session.run(output_names, input_feed) — when output_names is None,
    # ORT returns ALL outputs in the order they appear in the graph. We
    # exported the model with three outputs named "boxes", "scores", "labels",
    # in that order (see scripts/phase2_export_onnx.py).
    #
    # The input feed is a dict mapping input names to numpy arrays. Our model
    # has one input named "image".
    boxes, scores, labels = _session.run(None, {"image": x})

    # The rest is identical to predict.py — filter, format, sort. That
    # symmetry is the whole point: callers shouldn't be able to tell which
    # backend they're talking to.
    detections: list[Detection] = []
    for box, label_idx, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue
        detections.append(
            Detection(
                label=_class_names[int(label_idx)],
                score=float(score),
                box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            )
        )
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m src.inference.predict_onnx <image_path>")
        sys.exit(1)
    for d in predict(sys.argv[1]):
        print(f"  {d['label']:>15s}  {d['score']:.3f}  {d['box']}")
