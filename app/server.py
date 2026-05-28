"""
app/server.py
-------------
The HTTP-serving wrapper around the inference function.

This turns predict_onnx.py from a "Python library" into a "web service":
anyone who can send HTTP requests can now use the model, no Python knowledge
required.

What FastAPI is:
    A modern Python web framework. You declare endpoints by decorating
    functions. FastAPI handles request parsing, response serialization (it
    auto-converts dicts to JSON), validation, and async I/O. It's the
    de-facto standard for ML-serving microservices.

Endpoints exposed:
    GET  /         — health check, used by load balancers and curl
    POST /predict  — accepts a multipart file upload, returns JSON detections

How this gets launched:
    uvicorn app.server:app --host 0.0.0.0 --port 8000
    (uvicorn is the ASGI server that actually runs the FastAPI app object)

What runs inside the Docker container: this file, imported. The container
runs `uvicorn app.server:app` as its main process.
"""
from __future__ import annotations

import io
import logging
import tempfile
import time
from pathlib import Path

# FastAPI imports:
#   FastAPI      — the application class
#   File         — a marker for "this argument is a file upload"
#   UploadFile   — the type of the uploaded file
#   HTTPException — the standard way to return error responses
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

# Note: this server imports ONLY the ONNX backend. It must not import the
# PyTorch backend, because the Docker image doesn't ship torch.
from src.inference.predict_onnx import predict


# Standard Python logging setup. asctime is the timestamp, %message is whatever
# we pass to log.info(). This will print to stdout, which is what Docker captures.
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("server")


# The FastAPI application object. `title` and `version` show up in the
# auto-generated docs at http://localhost:8000/docs (a nice freebie).
app = FastAPI(title="CV Inference Service", version="0.1.0")


# ----- GET / -----------------------------------------------------------
#
# `@app.get("/")` registers this function as the handler for GET requests
# to the root path. Returning a dict — FastAPI converts it to JSON.
#
# Why have a health endpoint at all? In production, a load balancer or
# Kubernetes will poll it to check whether the service is alive. If you
# don't have one, the orchestrator can't tell live containers from dead
# ones.
@app.get("/")
def health() -> dict:
    return {"status": "ok", "service": "cv-inference"}


# ----- POST /predict ---------------------------------------------------
#
# Signature decoded:
#   `image: UploadFile = File(...)`
#       — the parameter `image` will be filled with the uploaded file.
#         `File(...)` is the FastAPI marker that says "this comes from a
#         multipart file field, not the JSON body." The `...` means
#         "required, no default."
#   `score_threshold: float = 0.5`
#       — FastAPI sees a primitive type with a default → treats it as a
#         query parameter. So the client can do
#         POST /predict?score_threshold=0.3
#
# `async def` is FastAPI's preferred form for endpoints that do I/O. The
# `await image.read()` below would block the whole event loop if this were
# a regular def, but `async def` lets other requests be served while we
# wait for the upload bytes.
@app.post("/predict")
async def predict_endpoint(
    image: UploadFile = File(...),
    score_threshold: float = 0.5,
) -> dict:
    # Step 1: read the entire upload into memory.
    # await: pause this coroutine until the I/O finishes, but let the
    # event loop service other requests in the meantime.
    raw = await image.read()
    if not raw:
        # HTTPException is FastAPI's way of returning a non-200 response.
        # The client gets a JSON body like {"detail": "empty file"} with
        # status 400 Bad Request.
        raise HTTPException(status_code=400, detail="empty file")

    # Step 2: validate the bytes are actually an image BEFORE we go anywhere
    # near the model. If someone POSTs an MP3, we don't want to find out
    # by way of a confusing tensor error 200ms later.
    #
    # PIL's verify() reads enough of the file to confirm structure. It's
    # cheap. We wrap it in BytesIO so PIL can treat the bytes like a file.
    try:
        Image.open(io.BytesIO(raw)).verify()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"not a valid image: {e}")

    # Step 3: run inference. predict() takes a file PATH, not bytes, so we
    # spool the upload to a temp file. The `delete=True` flag means the
    # temp file is deleted when the `with` block exits, even if an exception
    # is raised.
    #
    # Why not refactor predict() to accept bytes directly? It would be a
    # tiny change, but: by keeping predict() path-based, the CLI scripts
    # (phase1_baseline, etc.) and the server use the EXACT same API. Less
    # surface area to test.
    with tempfile.NamedTemporaryFile(suffix=".img", delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()  # ensure data is actually on disk before predict() reads it

        # Time just the inference call. Useful for server-side latency logs.
        t0 = time.perf_counter()
        detections = predict(tmp.name, score_threshold=score_threshold)
        latency_ms = (time.perf_counter() - t0) * 1000

    # Step 4: log + respond.
    # Structured-ish log line — easy to grep, easy to ship to a log aggregator.
    log.info(
        f"predict file={image.filename} bytes={len(raw)} "
        f"detections={len(detections)} latency_ms={latency_ms:.1f}"
    )

    # FastAPI converts the dict to JSON. Note we return latency_ms so the
    # client can see how fast the model ran (handy for debugging).
    return {
        "filename": image.filename,
        "n_detections": len(detections),
        "latency_ms": round(latency_ms, 1),
        "detections": detections,
    }
