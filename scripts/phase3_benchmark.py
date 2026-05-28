"""
scripts/phase3_benchmark.py
---------------------------
Phase 3B: Measure latency of all four backends, fairly.

The four backends:
    1. PyTorch fp32                  — the original model running in PyTorch
    2. ONNX fp32 (no opt)            — same model, run via ONNX Runtime,
                                        graph optimization disabled
    3. ONNX fp32 (graph opt)         — same model, ORT_ENABLE_ALL
    4. ONNX int8 quantized           — the quantized variant + graph opt

Methodology (the part an interviewer might probe):

    TIMING SCOPE
        We measure ONLY the inference call. Image decoding and preprocessing
        happen once, before the timer starts. If we included file I/O, the
        numbers would mostly reflect the SSD speed, not the model.

    WARMUP
        The first few runs of a model are always slower — caches need to
        warm up, JIT compilers need to kick in, memory allocators need to
        page in pages, etc. We do 3 warmup runs and DISCARD those timings.

    MEASURE
        30 timed runs per backend. We report percentiles (median, p95, p99)
        instead of the mean — percentiles describe the distribution; the
        mean lies when there are outliers.

    SAME INPUT FOR ALL BACKENDS
        Every backend sees the same image, pre-loaded and preprocessed once
        per backend. (We can't share the preprocessed tensor across PyTorch
        and ONNX because they want different formats — torch tensor vs
        numpy array — but we use the same source image.)

Output:
    - Console table
    - JSON dump at outputs/phase3_benchmark.json (raw numbers for the README)
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from src.inference.model import load_detector

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "data" / "samples"
MODEL_DIR = ROOT / "models"
RESULTS_PATH = ROOT / "outputs" / "phase3_benchmark.json"

# Benchmark constants. Lower numbers = faster benchmark, higher variance.
# 3 + 30 is a reasonable trade for a per-image benchmark on CPU.
WARMUP = 3
MEASURE = 30
SCORE_THRESHOLD = 0.5


def pick_image() -> Path:
    """Use the first jpg/png in samples/ as the benchmark image."""
    return sorted(p for p in SAMPLES_DIR.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[0]


# ----- Building "run one inference" closures per backend ------------------
#
# Pattern: each `make_*_runfn` function preloads everything that's fixed
# (model, preprocessed input) and returns a zero-argument `run()` function
# that calls the model. `bench()` then times calls to run() in a tight loop.
# Closures are nice here — the preloaded state stays in the closure's scope.

def make_pytorch_runfn(img_path: Path):
    """Return a no-arg function that runs PyTorch inference on the preloaded image."""
    model, preprocess, _ = load_detector()
    pil = Image.open(img_path).convert("RGB")
    x = preprocess(pil)    # CHW float tensor, only computed once

    def run():
        # torch.inference_mode disables autograd bookkeeping for speed.
        with torch.inference_mode():
            return model([x])
    return run


def make_onnx_session(onnx_path: Path, graph_opt: bool) -> ort.InferenceSession:
    """Build an ORT session with optimizations on or off, deterministically."""
    opts = ort.SessionOptions()
    if graph_opt:
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    else:
        # IMPORTANT: explicitly disable. The default is ORT_ENABLE_ALL, so
        # if we didn't set this, the "no opt" run would secretly have opts on.
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])


def make_onnx_runfn(sess: ort.InferenceSession, img_path: Path):
    """Same idea as make_pytorch_runfn but for an ONNX session."""
    pil = Image.open(img_path).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    x = arr.transpose(2, 0, 1)  # HWC → CHW

    def run():
        return sess.run(None, {"image": x})
    return run


def bench(name: str, run_one) -> dict:
    """
    Time `run_one` repeatedly. Returns a dict of stats.

    Two loops:
      - First loop runs WARMUP times without timing — caches warm up.
      - Second loop runs MEASURE times, records each duration in ms.
    """
    # Warmup. We deliberately throw away the return value and the timings.
    for _ in range(WARMUP):
        run_one()

    # Measure. Multiply by 1000 to get milliseconds (perf_counter returns seconds).
    times_ms = []
    for _ in range(MEASURE):
        t0 = time.perf_counter()
        run_one()
        times_ms.append((time.perf_counter() - t0) * 1000)

    return {
        "name": name,
        # np.percentile gives us median (50th), tail latency (95th, 99th),
        # and the best run (min). p99 is "1% of requests are slower than this."
        "median_ms": float(np.percentile(times_ms, 50)),
        "p95_ms":    float(np.percentile(times_ms, 95)),
        "p99_ms":    float(np.percentile(times_ms, 99)),
        "min_ms":    float(min(times_ms)),
        "n_runs":    MEASURE,
    }


def count_detections(run_one, score_threshold: float) -> int:
    """
    Run the model once, count how many detections clear the threshold.

    This is a quick "did optimization break the model" check, not a real
    accuracy metric. The two backends return scores in different containers
    (PyTorch returns a list of dicts; ONNX returns a tuple of arrays), so
    we sniff which kind we got and unpack accordingly.
    """
    out = run_one()
    if isinstance(out, list) and out and isinstance(out[0], dict):
        # PyTorch shape: list of dicts
        scores = out[0]["scores"].cpu().numpy()
    else:
        # ONNX shape: tuple (boxes, scores, labels)
        scores = out[1]
    return int((scores >= score_threshold).sum())


def main() -> None:
    # ----- Header: environment info we want recorded with the results.
    img = pick_image()
    pil = Image.open(img)
    print(f"benchmark image: {img.name}  size={pil.size}")
    print(f"platform: {platform.platform()}")
    print(f"python:   {platform.python_version()}, torch {torch.__version__}, "
          f"onnxruntime {ort.__version__}")
    print(f"warmup={WARMUP}, measure={MEASURE}, score_threshold={SCORE_THRESHOLD}\n")

    # ----- Build the four run() callables, one per backend.
    pt_run = make_pytorch_runfn(img)
    onnx_vanilla = make_onnx_runfn(
        make_onnx_session(MODEL_DIR / "detector.onnx", graph_opt=False), img)
    onnx_opt = make_onnx_runfn(
        make_onnx_session(MODEL_DIR / "detector.onnx", graph_opt=True), img)
    onnx_q = make_onnx_runfn(
        make_onnx_session(MODEL_DIR / "detector_quantized.onnx", graph_opt=True), img)

    # ----- Cheap accuracy check before timing. If a backend has badly
    # broken outputs, we want to know before we trust its latency number.
    counts = {
        "pytorch fp32":          count_detections(pt_run, SCORE_THRESHOLD),
        "onnx fp32 (no opt)":    count_detections(onnx_vanilla, SCORE_THRESHOLD),
        "onnx fp32 (graph opt)": count_detections(onnx_opt, SCORE_THRESHOLD),
        "onnx int8 quantized":   count_detections(onnx_q, SCORE_THRESHOLD),
    }
    print("detection counts at threshold (sanity check; should be close):")
    for k, v in counts.items():
        print(f"  {k:28s}  {v} detections")
    print()

    # ----- Run the four benchmarks.
    results = [
        bench("pytorch fp32",          pt_run),
        bench("onnx fp32 (no opt)",    onnx_vanilla),
        bench("onnx fp32 (graph opt)", onnx_opt),
        bench("onnx int8 quantized",   onnx_q),
    ]

    # ----- Print the table.
    baseline = results[0]["median_ms"]
    print(f"{'backend':28s}  {'median':>10s}  {'p95':>10s}  {'p99':>10s}  {'min':>10s}  speedup")
    print("-" * 88)
    for r in results:
        speedup = baseline / r["median_ms"]   # >1.0 means faster than PyTorch
        print(f"{r['name']:28s}  "
              f"{r['median_ms']:>8.1f}ms  "
              f"{r['p95_ms']:>8.1f}ms  "
              f"{r['p99_ms']:>8.1f}ms  "
              f"{r['min_ms']:>8.1f}ms  "
              f"{speedup:>5.2f}x")

    # ----- Persist results as JSON so the README can quote real numbers.
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "image": img.name,
        "image_size": list(pil.size),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "warmup": WARMUP,
        "measure": MEASURE,
        "score_threshold": SCORE_THRESHOLD,
        "detection_counts": counts,
        "latency": results,
    }, indent=2))
    print(f"\nresults saved → {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
