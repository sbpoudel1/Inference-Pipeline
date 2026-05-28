"""
scripts/phase2_parity.py
------------------------
Phase 2C: Verify the ONNX export didn't break the model.

After torch.onnx.export, the PyTorch and ONNX versions of the model should
produce numerically equivalent outputs. "Equivalent" doesn't mean identical
— float math can be reordered, NMS implementations can differ — so we check
that outputs match WITHIN A TOLERANCE.

Why this script matters: a successful export can still produce a subtly
wrong model. This is the bug-catching step that real MLOps engineers care
about, and the one that, if it fails, makes for the best interview story
("I traced an issue through these three places...").

What we check, per image:
  1. Same number of detections at a low score threshold (count must match)
  2. Same label order when sorted by score (label_mismatches must be 0)
  3. Score values agree to ~3 decimal places (max |Δscore| < 1e-3)
  4. Box coordinates agree to <1 pixel (max |Δbox| < 1.0)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# We import both backends here. By passing the same image through both and
# comparing outputs, we get an end-to-end parity test.
from src.inference import predict as pt_backend
from src.inference import predict_onnx as onnx_backend


ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "data" / "samples"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----- Tunable parameters --------------------------------------------------
#
# Lower threshold than the "real" 0.5 we use for end users. We want to
# compare ALL detections the model considers, not just the high-confidence
# ones. If both backends produce 200 raw detections with score >= 0.05 and
# all 200 match, the model is genuinely the same model.
SCORE_THRESHOLD = 0.05

# Tolerances. These are the values we picked after seeing what real drift
# looks like on this model + hardware. If the test fails, the first thing
# to check is whether these tolerances need adjusting.
BOX_TOL = 1.0       # boxes near image edges may drift by sub-pixel amounts
SCORE_TOL = 1e-3    # scores should agree to ~3 decimal places


def gather_images() -> list[Path]:
    return sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in IMG_EXTS)


def compare_one(img_path: Path) -> tuple[bool, str]:
    """
    Run both backends on one image, compare their outputs.

    Returns (passed?, human-readable message).
    """
    # Run each backend's predict() once. Both return list[Detection] sorted by
    # score descending — so element i in pt[] corresponds to element i in on[].
    pt = pt_backend.predict(img_path, score_threshold=SCORE_THRESHOLD)
    on = onnx_backend.predict(img_path, score_threshold=SCORE_THRESHOLD)

    # First sanity check: do both backends agree on how many detections
    # exist above threshold? If counts differ, the rest of the comparison
    # is meaningless.
    if len(pt) != len(on):
        return False, f"count mismatch: pytorch={len(pt)}, onnx={len(on)}"

    # Edge case: zero detections in both. Vacuously equal.
    if not pt:
        return True, "no detections in either backend (vacuously equal)"

    # Walk the two lists in lockstep and compute the worst disagreement on
    # each axis (label, score, box).
    label_mismatches = 0
    score_diffs = []
    box_diffs = []
    for p, o in zip(pt, on):
        if p["label"] != o["label"]:
            label_mismatches += 1
        score_diffs.append(abs(p["score"] - o["score"]))
        # max() over the 4 box coordinates — we care about the WORST corner.
        box_diffs.append(max(abs(a - b) for a, b in zip(p["box"], o["box"])))

    max_score_diff = max(score_diffs)
    max_box_diff = max(box_diffs)

    ok = (
        label_mismatches == 0
        and max_score_diff < SCORE_TOL
        and max_box_diff < BOX_TOL
    )
    return ok, (
        f"n={len(pt):3d}  "
        f"max|Δscore|={max_score_diff:.2e}  "  # 2.0e-06 etc.
        f"max|Δbox|={max_box_diff:.2f}px  "
        f"label_mismatches={label_mismatches}"
    )


def main() -> None:
    images = gather_images()
    if not images:
        sys.exit(f"no images in {SAMPLES_DIR}")

    print(f"parity check: PyTorch vs ONNX Runtime, threshold={SCORE_THRESHOLD}")
    print(f"tolerances: score < {SCORE_TOL}, box < {BOX_TOL}px\n")

    n_ok = 0
    for img in images:
        ok, msg = compare_one(img)
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] {img.name:30s}  {msg}")
        if ok:
            n_ok += 1

    print(f"\n{n_ok}/{len(images)} images passed")
    # Exit with code 1 on any failure — useful if you wire this into CI.
    if n_ok != len(images):
        sys.exit(1)


if __name__ == "__main__":
    main()
