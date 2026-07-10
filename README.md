---
title: Face Verification
emoji: 👤
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.34.0
app_file: app.py
pinned: false
---

# Face Verification Pipeline Prototype

This repository contains a production-ready prototype for a face verification pipeline, packaged as a Hugging Face Space using Gradio and managed with `uv`.

## 🚀 Objectives
The goal is to provide a complete workflow mimicking a production environment:
1. User uploads a reference profile image.
2. The system generates and caches its FaceNet512 embedding.
3. The user starts a webcam stream.
4. Each frame is analyzed for **Liveness** (Anti-Spoofing).
5. If live, the face is verified against the cached embedding using Cosine Similarity.

## 🧠 Models Used
- **MiniFASNet (ONNX)**: Used for liveness detection to prevent print or replay attacks. The ONNX format ensures maximum performance on CPUs and minimal dependencies.
- **FaceNet512 (via DeepFace)**: Generates highly accurate 512-dimensional facial embeddings for identity verification.

## 🏗️ Architecture
The project is built with clean architecture in mind to allow easy migration to a FastAPI backend:
- `src/app.py`: The Gradio frontend for Hugging Face Spaces.
- `src/services/liveness_service.py`: Encapsulates the MiniFASNet ONNX runtime logic.
- `src/services/embedding_service.py`: Wraps `deepface` to provide unified face extraction and FaceNet512 embeddings.
- `src/services/verification_service.py`: The core orchestrator managing the pipeline.
- `src/utils/`: Contains shared configuration, logging, and image conversion utilities (e.g. Base64 and PIL conversions).

## 🛠️ Local Setup
This project uses `uv` for lightning-fast dependency management.

1. Install `uv` if you haven't already.
2. Set up the environment and install dependencies:
```bash
uv sync
```
3. Run the Gradio application:
```bash
uv run python app.py
```

## 🔒 Configuration
You can find centralized configurable thresholds and model paths in `src/utils/config.py`.
- **Threshold**: Currently set to `0.4` for cosine similarity distance. (Distance < 0.4 = verified).

## ➡️ Future FastAPI Integration
The code is explicitly decoupled from Gradio. To integrate this into a FastAPI backend:
1. Expose an endpoint for `/register` that accepts a Base64 string, calls `verification_service.process_reference_image()` and saves the embedding array to your DB.
2. Expose a WebSocket endpoint or `/verify` endpoint that accepts frames (as Base64 strings or direct bytes), retrieves the cached embedding from memory, and calls `verification_service.process_webcam_frame()`.
