"""
src/inference/predict.py
------------------------
The PyTorch-backed inference function.

This file exposes ONE thing to the rest of the code: a function called
`predict(image_path)` that takes a path to an image file and returns a list
of detections. Each detection is a plain Python dict.

The "predict_onnx.py" file in this folder exposes the SAME function signature
but uses the ONNX runtime instead. That's intentional — it means any code
that uses `predict()` can swap backends without changing.
"""
from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import torch
from PIL import Image  # Pillow — the standard Python image library

from src.inference.model import load_detector


# TypedDict is a way to declare the shape of a dictionary as a type. It does
# not enforce anything at runtime — it's purely for editors and type checkers
# to know that `Detection["label"]` is a string, `Detection["score"]` is a
# float, etc. Optional but makes the code easier to read and refactor.
class Detection(TypedDict):
    label: str
    score: float
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixel coordinates


# ----- Module-level cache --------------------------------------------------
#
# Loading the model takes ~1 second (the first time you also pay for downloading
# the weights, which is ~30s on a slow connection). We don't want to do that on
# every call to predict(). So we cache the model in module-level variables and
# load it lazily on the first call.
#
# A more elegant solution would be a class with these as attributes, but the
# function-with-globals pattern is dead simple and works fine for a single
# model. This is how real inference servers do it too.
_model = None
_preprocess = None
_class_names = None


def _ensure_loaded() -> None:
    """Load the model on the first call; reuse the cached version after that."""
    global _model, _preprocess, _class_names
    if _model is None:
        _model, _preprocess, _class_names = load_detector()


def predict(image_path: str | Path, score_threshold: float = 0.5) -> list[Detection]:
    """
    Run object detection on a single image.

    Args:
        image_path: path to a .jpg / .png / etc.
        score_threshold: detections with confidence below this are dropped.
                          0.5 is a common choice; lower = more detections,
                          some of them garbage.

    Returns:
        A list of detections, sorted by score (highest first). Boxes are
        in PIXEL COORDINATES of the original image (not normalized).
    """
    _ensure_loaded()
    # `assert` here is for the type-checker; if _ensure_loaded() ran, these
    # are all non-None. The runtime cost is negligible.
    assert _model is not None and _preprocess is not None and _class_names is not None

    # Step 1: open and decode the image file. .convert("RGB") guarantees we
    # have a 3-channel image even if the source was grayscale or RGBA.
    image = Image.open(image_path).convert("RGB")

    # Step 2: preprocess. This converts the PIL image into a PyTorch tensor
    # with shape (channels=3, height, width) and float values in [0, 1].
    # (For detection models, that's all the preprocess does — resize and
    # normalize happen inside the model.)
    input_tensor = _preprocess(image)

    # Step 3: actually run the model.
    #
    # `torch.inference_mode()` is a context manager that tells PyTorch:
    #   "I will not call .backward() on any of these results."
    # That lets PyTorch skip allocating the bookkeeping that gradient descent
    # needs. Same idea as the older torch.no_grad(), slightly faster.
    #
    # `_model([input_tensor])` — note the LIST around the tensor. Torchvision
    # detection models accept a batch of images, where each image can be a
    # different size. The list is the batch. Even for one image, we wrap it
    # in a list of length 1.
    with torch.inference_mode():
        outputs = _model([input_tensor])

    # `outputs` is also a list of length 1 (one entry per input image). Each
    # entry is a dict with three keys: "boxes", "labels", "scores".
    out = outputs[0]
    boxes = out["boxes"].tolist()    # convert PyTorch tensor → Python list
    labels = out["labels"].tolist()  # of integers (class indices)
    scores = out["scores"].tolist()  # of floats (confidence)

    # Step 4: filter by confidence and format into a clean list of dicts.
    # We deliberately return Python-native types here — no PyTorch tensors
    # leak out of this function. That matters because predict_onnx.py
    # returns the same Python types from a different backend.
    detections: list[Detection] = []
    for box, label_idx, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue  # skip low-confidence detections
        detections.append(
            Detection(
                label=_class_names[label_idx],  # "person", "car", etc.
                score=float(score),
                box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            )
        )

    # Sort highest-confidence first — makes the output easier to read and
    # makes "top-K" filtering trivial for callers.
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


# CLI test entrypoint: `python -m src.inference.predict path/to/image.jpg`
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m src.inference.predict <image_path>")
        sys.exit(1)
    for d in predict(sys.argv[1]):
        # %-style alignment so the output reads as a clean table
        print(f"  {d['label']:>15s}  {d['score']:.3f}  {d['box']}")
