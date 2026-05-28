"""
src/inference/model.py
----------------------
Loads the pretrained object detection model into memory.

Big picture: torchvision (PyTorch's computer-vision library) ships dozens of
pretrained models. We pick one — SSDLite320 MobileNetV3-Large — and return it
along with two helper objects: a preprocessing transform and the list of
class names the model can predict.

This file does NOT run inference. It just builds and hands back the model.
See predict.py for the actual "give me a photo, get detections back" code.
"""
from __future__ import annotations  # allows modern type hints like `list[str]` on Python 3.9+

# torchvision groups pretrained models by task. We're using object detection,
# so we import from torchvision.models.detection.
#
# Each pretrained model in torchvision comes as two things:
#   1. A *constructor function* (here: ssdlite320_mobilenet_v3_large) that
#      builds the model's architecture in memory.
#   2. A *weights enum* (here: SSDLite320_MobileNet_V3_Large_Weights) that
#      holds pointers to downloaded weight files and metadata.
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)


def load_detector():
    """
    Build the pretrained detector and return three things:
      1. model         — the PyTorch model object, ready for inference
      2. preprocess    — a callable that turns a PIL image into the tensor
                          format the model expects
      3. class_names   — list of 91 strings: "person", "bicycle", "car", ...

    On the first call, torchvision automatically downloads the weight file
    (~14 MB) from PyTorch's download server and caches it at
    ~/.cache/torch/hub/checkpoints/. Every subsequent call reuses the cache.
    """

    # `.COCO_V1` means "the weights trained on the COCO dataset, version 1".
    # COCO (Common Objects in Context) is a standard CV dataset with 80 object
    # categories — person, car, dog, chair, etc. Plus an implicit "background"
    # class, which is why the categories list has 91 entries instead of 80
    # (some indices are unused, that's a COCO quirk).
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1

    # Build the model's architecture and load the pretrained weights into it.
    # If we'd passed weights=None, we'd get a randomly-initialized model that
    # would output nonsense — exactly what you'd want if you were training
    # from scratch, exactly what we don't want here.
    model = ssdlite320_mobilenet_v3_large(weights=weights)

    # *** CRITICAL: switch the model to eval (inference) mode. ***
    # PyTorch models have two modes: train and eval. Layers like Dropout and
    # BatchNorm behave differently in each. If you forget .eval(), predictions
    # are silently degraded — not crash-wrong, just subtly worse. This is the
    # single most common PyTorch beginner footgun.
    model.eval()

    # `weights.transforms()` returns the EXACT preprocessing function that
    # was used during training. Using anything else means the model sees
    # inputs that don't match its training distribution, which gives bad
    # results. Bundling the preprocess with the weights is a torchvision
    # convenience that prevents a whole class of bugs.
    #
    # For detection models, this preprocessor is intentionally simple — it
    # just converts a PIL image to a float tensor in [0,1]. The resize and
    # normalize steps live INSIDE the model itself (in model.transform).
    preprocess = weights.transforms()

    # `weights.meta` is a dictionary that comes with every torchvision
    # weight bundle. It includes "categories" (the class names),
    # "_metrics" (achieved accuracy on the test set), etc.
    class_names = weights.meta["categories"]

    return model, preprocess, class_names


# This block only runs if you do `python -m src.inference.model` from the
# command line — it's a smoke test you can use to verify the file works.
# When this module is imported by other code, this block is skipped.
if __name__ == "__main__":
    model, preprocess, class_names = load_detector()

    # Count the model's parameters (a parameter = one trainable number, like
    # a weight or bias). 5 million is small as deep-learning models go.
    n_params = sum(p.numel() for p in model.parameters())

    print(f"loaded ssdlite320_mobilenet_v3_large")
    print(f"  params: {n_params:,}")
    print(f"  classes: {len(class_names)} (e.g. {class_names[1:6]})")
    print(f"  preprocess: {preprocess}")
