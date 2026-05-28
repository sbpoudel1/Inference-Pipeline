"""
scripts/phase1_baseline.py
--------------------------
Phase 1 driver. The simplest possible "does this work" script.

What it does:
  1. Look for image files in data/samples/ (or take an image path as an arg).
  2. For each image: run predict(), print the detections, draw boxes on the
     image, save the result to outputs/.

How to run:
    python -m scripts.phase1_baseline                     # scan data/samples/
    python -m scripts.phase1_baseline path/to/my_pic.jpg  # one specific image

This is a SANITY CHECK, not a benchmark. The timing prints are useful but
include the very first warm-up run (slow), so don't take them as performance
numbers. Phase 3 does proper benchmarking with warmup discarded.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# matplotlib is a plotting library — overkill for drawing boxes, but it's
# already installed and the patches API is dead simple.
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

# Import the predict() function from our own module. This is why we run with
# `python -m scripts.phase1_baseline` (not `python scripts/phase1_baseline.py`):
# the -m form sets up the module search path so `src.inference.predict` works.
from src.inference.predict import predict


# Path constants. Computing them from __file__ means the script works no
# matter what directory you invoke it from.
ROOT = Path(__file__).resolve().parent.parent   # .../ML Ops/
SAMPLES_DIR = ROOT / "data" / "samples"
OUTPUTS_DIR = ROOT / "outputs"

# Image file extensions we'll consider. Lowercased; we match case-insensitively below.
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def visualize(image_path: Path, detections: list[dict], out_path: Path) -> None:
    """
    Draw bounding boxes + labels on the image and save the result as a PNG.

    Uses matplotlib because it gives us a nice axis-with-image plot and easy
    rectangle drawing. The "headless" `plt.close(fig)` at the end matters —
    without it, matplotlib accumulates figures in memory and eventually warns.
    """
    image = Image.open(image_path).convert("RGB")

    # fig, ax = "figure, axes". A matplotlib figure can contain multiple axes
    # (subplots); we just have one. figsize is in inches.
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)   # paint the image onto the axes
    ax.axis("off")     # hide the x/y tick marks — pure image, no chart frame

    for det in detections:
        # Each box is (x1, y1, x2, y2) — top-left corner and bottom-right corner.
        # matplotlib's Rectangle wants (x, y) of top-left + width + height,
        # so we convert.
        x1, y1, x2, y2 = det["box"]
        w, h = x2 - x1, y2 - y1

        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        ax.add_patch(rect)

        # Put a label tag just above the box. `max(y1 - 4, 0)` keeps it from
        # going above the top edge of the image.
        ax.text(
            x1, max(y1 - 4, 0),
            f"{det['label']} {det['score']:.2f}",
            color="black",
            fontsize=9,
            bbox=dict(facecolor="lime", alpha=0.7, pad=1, edgecolor="none"),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)  # free memory


def gather_images(arg_path: str | None) -> list[Path]:
    """Return the list of images to process — either a single path or everything in samples/."""
    if arg_path:
        p = Path(arg_path)
        if not p.exists():
            sys.exit(f"image not found: {p}")
        return [p]

    if not SAMPLES_DIR.exists():
        sys.exit(f"no samples dir at {SAMPLES_DIR} — drop some images there first")

    # iterdir() returns everything in the directory; we filter to image extensions.
    imgs = sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        sys.exit(
            f"no images found in {SAMPLES_DIR} — drop a .jpg/.png in there and re-run"
        )
    return imgs


def main() -> None:
    images = gather_images(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"running detector on {len(images)} image(s)\n")

    for img_path in images:
        # time.perf_counter() is the high-resolution monotonic clock — best
        # for measuring short durations. NEVER use time.time() for this, it
        # can jump backwards (NTP adjustments, daylight savings, etc.).
        t0 = time.perf_counter()
        detections = predict(img_path, score_threshold=0.5)
        dt = time.perf_counter() - t0  # elapsed seconds

        print(f"{img_path.name}  ({dt*1000:.0f} ms, {len(detections)} detections)")

        # Print up to 10 detections to keep the console readable on noisy images.
        for d in detections[:10]:
            print(f"  {d['label']:>15s}  {d['score']:.3f}")
        if len(detections) > 10:
            print(f"  ... +{len(detections) - 10} more")

        # Save the visualization. Filename stem = original filename minus extension.
        out_path = OUTPUTS_DIR / f"{img_path.stem}_phase1.png"
        visualize(img_path, detections, out_path)
        print(f"  -> {out_path.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()
