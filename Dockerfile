# Multi-stage Dockerfile.
#
# Stage 1 (builder): install dependencies into a venv on a "fat" base image.
# Stage 2 (runtime): copy the venv + just the application code into a slim
# base image. The runtime image contains no pip cache, no build tools, and
# crucially NO PYTORCH. The training stack stays in the developer's lap;
# only the optimized .onnx model and ONNX Runtime ship to the device.
#
# Final image size: ~250 MB (mostly numpy + onnxruntime native libs).
# Compared to a naïve build with torch: ~2 GB.

# ----- Stage 1: builder -----
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
COPY requirements-inference.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements-inference.txt

# ----- Stage 2: runtime -----
FROM python:3.12-slim AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# venv carried over from the builder stage
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Only the code and model artifacts the runtime actually needs.
COPY src/ ./src/
COPY app/ ./app/
COPY models/detector.onnx models/coco_classes.json ./models/

EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
