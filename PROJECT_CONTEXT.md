# Computer Vision MLOps Side Project: Context for Claude Code

## TL;DR for the assistant

I am a bioinformatics analyst applying to a **New Grad Software Engineer, ML Ops** role at Symbotic (req R6028). My resume has real gaps for this role, specifically: computer vision, ONNX, Docker, edge deployment. I want to build a focused side project that closes those gaps in a way that is real (not resume padding) and that I can speak to confidently in an interview. Please help me build it end-to-end. I want to actually understand each step, not just copy/paste code.

---

## Who I am

- **Background:** Clinical Bioinformatics Analyst at Foundation Medicine since 2019 (~7 years).
- **Education:** Just completed MCIT at UPenn (May 2026). Prior MS in Bioinformatics from Johns Hopkins (2019).
- **What I know well:** Python (production scripts and pipelines), scikit-learn, pandas, numpy, SQL, Linux, Git. I've trained classification models on genomic data (multi-class mutation prediction, >95% accuracy). I've built API integrations, caching layers, and automation pipelines. I've used **pickle** to serialize sklearn models.
- **What I do NOT know yet:** PyTorch (only conceptually), computer vision, ONNX export/runtime, TensorRT, Docker beyond superficial use, GCP/Azure, C++, edge deployment. This project is partly to learn these.
- **Comfort level:** Senior in Python and data work, beginner in deep learning frameworks and deployment infrastructure. Please explain framework-specific concepts when they come up; do not assume I know PyTorch idioms.

## The target job (why this project matters)

Symbotic's MLOps team deploys computer vision models to a fleet of autonomous robots. Key responsibilities from the JD:

- Build CV solutions for robotic perception
- Deploy models to robotic hardware using **ONNX/TensorRT**
- Optimize edge computing infrastructure for real-time performance
- Structure **Docker compose** configurations for robotic hardware
- Build pipelines for dataset curation, training, evaluation

Required: Python or C++, cloud experience (GCP/Azure), complex data systems (sensor/camera data).

My resume has the Python and data-systems pieces. It's missing everything CV-specific and everything deployment-specific. This project closes those gaps.

## Project goal

Build a **computer vision inference pipeline** that demonstrates the model-handoff workflow this role is about: take a trained CV model, optimize it for production inference, containerize it, and benchmark it. The deliverable is a public GitHub repo I can link to on my resume.

The single resume bullet I want to be able to write at the end:

> Built an end-to-end computer vision inference pipeline: fine-tuned a PyTorch object detection model, exported to ONNX, optimized with ONNX Runtime (quantization, graph optimization), containerized with Docker, and benchmarked latency improvements (PyTorch vs. ONNX vs. quantized ONNX) on CPU.

That bullet hits CV, PyTorch, ONNX, ONNX Runtime, Docker, and benchmarking. Six gaps closed by one project.

## Project specification

### Phase 1: Baseline (get something working end-to-end)

1. Pick a pretrained vision model from `torchvision`. Start simple: image classification (ResNet50) or a small object detector (e.g., FasterRCNN or SSD with a MobileNet backbone). Object detection is closer to what robots actually do, but classification is faster to get running. **Recommend object detection** unless setup friction gets bad, then fall back to classification.
2. Run inference on a handful of test images. Confirm outputs look sane (visualize bounding boxes or top-5 predictions).
3. Wrap inference in a clean Python function: `predict(image_path) -> result`.

### Phase 2: ONNX export and parity check

1. Export the PyTorch model to ONNX using `torch.onnx.export`. Capture dynamic axes correctly so the model accepts variable batch sizes / image sizes.
2. Load the ONNX model with `onnxruntime` and run inference on the same test images.
3. **Critical step:** verify outputs match PyTorch outputs within a small numerical tolerance. This is the bug-catching moment that real MLOps engineers care about. If outputs diverge, debug before moving on.
4. Note any ops that didn't convert cleanly and how I resolved them.

### Phase 3: Optimization

1. Apply ONNX Runtime graph optimizations (`SessionOptions` with `graph_optimization_level = ORT_ENABLE_ALL`).
2. Apply dynamic quantization to the ONNX model (INT8 weights). Compare accuracy on the same test set.
3. Benchmark: latency (median, p95, p99) and throughput for:
   - Original PyTorch model
   - Vanilla ONNX
   - ONNX + graph optimization
   - Quantized ONNX
4. Produce a simple results table or chart. This is the most interview-able artifact: I want to be able to say "I saw a 2.3x speedup with quantization and a 0.4% accuracy drop" or whatever the actual numbers are.

### Phase 4: Containerization

1. Write a `Dockerfile` for the inference service. Multi-stage build, slim base image, only inference dependencies (no PyTorch in the final image, just `onnxruntime` and minimal Python stack). This mirrors the "no Python on the robot" lesson from our chat.
2. Wrap inference in a simple FastAPI server with a `/predict` endpoint that accepts an image and returns predictions.
3. Write a `docker-compose.yml` that brings up the service (overkill for one service, but the JD specifically mentions docker-compose, and it's good practice).
4. Confirm the container can be built and run, and that `curl` or a small Python client hits the endpoint successfully.

### Phase 5: Documentation

A real `README.md` covering:
- What the project does and why (one paragraph)
- Architecture diagram (even a hand-drawn one in mermaid is fine)
- How to run it (clone, build, predict)
- Benchmark results table
- A "what I learned" section, which is genuinely useful for an interview and also signals self-awareness

### Stretch goals (only if time allows)

- Try TensorRT optimization on a GPU machine (Colab works) and compare to ONNX Runtime
- Add a simple monitoring endpoint that logs inference latency over time
- Convert one of my sklearn pickle models from FMI work to ONNX using `skl2onnx` as a separate small example, this would let me speak to both the deep-learning ONNX path and the classical-ML ONNX path

## Success criteria

The project is done when:

1. The repo builds and runs cleanly on a fresh machine following only the README
2. Benchmark numbers are real, reproducible, and explained
3. I can verbally walk through the full pipeline without notes
4. I can answer "why ONNX instead of pickle" and "why a multi-stage Docker build" in interview terms (we already covered the first one in chat)
5. The bullet point in the resume bullet above is accurate, no embellishment

## Working preferences for the assistant

- **Explain as we go.** When introducing a new concept (PyTorch hooks, ONNX opsets, ORT execution providers, Docker layer caching), give me a couple of sentences of context. I'd rather understand than ship fast.
- **Show me failing states.** When something can go wrong (mismatched ONNX outputs, missing ops, version conflicts), let me hit those and debug them. They're the interview gold.
- **Be honest about gaps.** If you'd normally use TensorRT but we're on CPU, say so. If a step is hand-waving past something important, flag it.
- **No fake numbers.** Benchmark results must be real. If I say "2.3x speedup" in an interview and someone asks "on what hardware, what batch size, what input shape", I need to know.
- **Reasonable scope.** This should be a focused weekend project, not a month-long undertaking. If I'm scope-creeping, push back.

## Context from prior chat

A few things we already covered that the assistant should not need to re-explain:

- **Deep learning vs. TensorFlow vs. ONNX:** I understand these are different layers (field vs. framework vs. interchange format).
- **Pickle vs. ONNX:** I understand pickle serializes the Python object and requires Python to load; ONNX serializes the computation graph and is portable across languages and runtimes. I also know pickle is a security risk for untrusted input.
- **Why the robot doesn't run Python:** I get that the inference path on an edge device typically runs a compiled runtime, not a Python interpreter, which is why ONNX + a C++/optimized runtime is the standard production handoff.

## Starting state

Nothing built yet. Working from a fresh repo. Local machine; no GPU assumed (we'll note where GPU would help).

## First request for the assistant

When I start the Claude Code session, the first thing I want is:

1. A proposed directory structure for the repo
2. A `pyproject.toml` or `requirements.txt` with pinned versions
3. The Phase 1 baseline script (load a pretrained model, run inference on a sample image, print/visualize results)

Then we'll go phase by phase. Don't generate the whole project up front. Build incrementally and let me run each piece before moving on.
