import os
from pydantic import BaseModel

class VerificationConfig(BaseModel):
    # Threshold for FaceNet512 cosine similarity (0.3 - 0.4 is a common threshold for cosine metric)
    SIMILARITY_THRESHOLD: float = 0.4
    
    # Model configuration
    MINIFASNET_REPO_ID: str = "garciafido/minifasnet-v2-anti-spoofing-onnx"
    MINIFASNET_FILENAME: str = "minifasnet_v2.onnx"
    
    # Input sizes
    MINIFASNET_INPUT_SIZE: tuple[int, int] = (80, 80)
    
    # Face detection backend for DeepFace
    FACE_DETECTOR_BACKEND: str = "retinaface"
    
    # Recognition model
    FACE_RECOGNITION_MODEL: str = "Facenet512"

    class Config:
        frozen = True

# Global config instance
config = VerificationConfig()
