# Technology Stack and Architecture Overview

This document outlines the core technologies utilized in the Face Verification pipeline, their specific roles, and the rationale behind their selection.

## Core Technologies

### 1. Gradio (Frontend & Interface)
**Role**: Provides the web-based user interface. It handles image uploads, captures continuous streaming frames from the user's webcam, and displays real-time verification results.
**Why it was chosen**: Gradio is explicitly designed for demonstrating machine learning models. It abstracts away complex frontend web development (HTML/JS/WebRTC for webcams) and integrates flawlessly with hosting platforms like Hugging Face Spaces, allowing for rapid prototyping.

### 2. YuNet (Face Detection & Landmarks)
**Role**: Scans incoming images/frames to locate where a face is positioned and identifies five key facial landmarks (eyes, nose, mouth corners).
**Why it was chosen**: YuNet is an extremely lightweight, CNN-based face detector integrated into OpenCV. It provides high accuracy even under varying lighting or angles, and it runs exceptionally fast on standard CPUs, making it ideal for real-time streaming applications.

### 3. FaceNet512 via ONNX (Facial Embeddings)
**Role**: Takes the isolated face and converts it into a 512-dimensional mathematical vector (an embedding) that uniquely represents that person's facial features. 
**Why it was chosen**: FaceNet512 is an industry-standard model known for highly discriminative facial representations. Using the ONNX (Open Neural Network Exchange) format allows the system to run the model without the massive overhead of installing heavy deep-learning frameworks like TensorFlow or PyTorch.

### 4. MiniFASNet via ONNX (Face Anti-Spoofing / Liveness)
**Role**: Analyzes the texture and properties of the detected face to determine if it is a real, live 3D person or a 2D spoof attack (such as holding up a printed photo or a phone screen to the camera).
**Why it was chosen**: MiniFASNet is a highly efficient, state-of-the-art model for Silent Face Anti-Spoofing. It requires minimal computational power while providing robust defense against presentation attacks. 

### 5. Hugging Face Spaces
**Role**: The hosting and deployment platform for the application.
**Why it was chosen**: It provides a managed environment tailored for AI applications, automatically handling dependencies, server routing, and hardware allocation.

## How They Work Together (The Pipeline)

The entire system operates as a continuous, synchronized pipeline:

1. **Input Capture**: The user uploads a reference image and turns on their webcam via the interface.
2. **Detection & Alignment**: As webcam frames arrive, the face detector (YuNet) finds the face bounding box and landmarks. The system uses these mathematical coordinates to rotate and align the face, ensuring it is perfectly straight.
3. **Liveness Check**: Before any identity checking occurs, the cropped face is sent to the anti-spoofing model (MiniFASNet) to verify the presence of a live human. If a spoof is detected, the pipeline halts for that frame.
4. **Feature Extraction**: If the face is deemed "Live", it is passed to the recognition model (FaceNet512) to generate the facial embedding.
5. **Comparison**: The system calculates the distance between the live frame's embedding and the reference image's embedding to determine how mathematically similar they are.
6. **Temporal Smoothing**: Rather than making a harsh decision on a single frame, the pipeline aggregates the results (liveness votes and similarity scores) over a sliding window of the most recent frames. This creates a highly stable, final verification decision that is then pushed back to the user interface.
