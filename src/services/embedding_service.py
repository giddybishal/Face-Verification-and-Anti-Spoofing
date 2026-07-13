import numpy as np
import time
import cv2
import os
import onnxruntime as ort
from typing import Tuple, List, Optional

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EmbeddingService:
    def __init__(self):
        """
        Initializes the EmbeddingService using pure ONNX and OpenCV YuNet.
        No TensorFlow dependencies.
        """
        logger.info("EmbeddingService instantiated. Models will be loaded lazily.")
        self.detector = None
        self.session = None
        self.input_name = None
        self.input_size = None
        self.is_nchw = None
        self.detector_path = "face_detection_yunet_2023mar.onnx"
        self.facenet_path = "facenet512.onnx"

    def _ensure_initialized(self):
        """
        Lazily initializes the ONNX FaceNet session and YuNet detector.
        """
        if self.session is not None and self.detector is not None:
            return
            
        logger.info("Initializing ONNX EmbeddingService models...")
        start_time = time.time()
        
        try:
            # Initialize YuNet Face Detector
            if not os.path.exists(self.detector_path):
                logger.error(f"YuNet model not found at {self.detector_path}")
                raise FileNotFoundError(self.detector_path)
                
            self.detector = cv2.FaceDetectorYN.create(
                model=self.detector_path,
                config="",
                input_size=(320, 320), # Will be updated dynamically per frame
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000
            )
            
            # Initialize FaceNet ONNX Session
            if not os.path.exists(self.facenet_path):
                logger.error(f"FaceNet ONNX model not found at {self.facenet_path}")
                raise FileNotFoundError(self.facenet_path)
                
            self.session = ort.InferenceSession(
                self.facenet_path,
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            
            # Usually FaceNet expects 160x160. We can infer from the model.
            input_shape = self.session.get_inputs()[0].shape
            self.input_size = (160, 160) if isinstance(input_shape[1], str) or input_shape[1] == -1 else (input_shape[1], input_shape[2])
            if input_shape[3] == 3: # NHWC
                self.input_size = (input_shape[1], input_shape[2])
                self.is_nchw = False
            else: # NCHW
                self.input_size = (input_shape[2], input_shape[3])
                self.is_nchw = True
                
            logger.info(f"ONNX EmbeddingService initialized in {time.time() - start_time:.2f}s. Expected shape: {input_shape}")
            
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingService: {e}")
            raise

    def align_face(self, img: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aligns the face using the 5 landmarks provided by YuNet.
        Landmarks: [right_eye, left_eye, nose_tip, right_mouth, left_mouth]
        """
        # Right eye and left eye (from the viewer's perspective, so subject's left/right)
        # In YuNet: 0,1 is right eye, 2,3 is left eye
        right_eye = (landmarks[0], landmarks[1])
        left_eye = (landmarks[2], landmarks[3])
        
        # Calculate angle
        dY = left_eye[1] - right_eye[1]
        dX = left_eye[0] - right_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Calculate center
        center = (int((right_eye[0] + left_eye[0]) / 2), int((right_eye[1] + left_eye[1]) / 2))
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply affine transform
        h, w = img.shape[:2]
        aligned = cv2.warpAffine(img, M, (w, h))
        return aligned, M

    def extract_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detects and extracts faces in the image using YuNet.
        Returns a list of dictionaries: 'face' (aligned crop), 'facial_area', 'confidence'.
        """
        try:
            self._ensure_initialized()
            h, w = image.shape[:2]
            self.detector.setInputSize((w, h))
            
            _, faces = self.detector.detect(image)
            
            valid_faces = []
            if faces is not None:
                for face in faces:
                    box = face[0:4].astype(int)
                    landmarks = face[4:14]
                    conf = face[14]
                    
                    x, y, box_w, box_h = box
                    # Ensure within bounds
                    x = max(0, x)
                    y = max(0, y)
                    box_w = min(w - x, box_w)
                    box_h = min(h - y, box_h)
                    
                    if box_w <= 0 or box_h <= 0:
                        continue
                        
                    # Align the entire image first based on landmarks
                    # (Standard approach to preserve crop quality)
                    # Alternatively, just crop the bounding box if alignment is skipped.
                    # For FaceNet, alignment is highly recommended.
                    aligned_img, M = self.align_face(image, landmarks)
                    
                    # Crop from the aligned image by transforming the bounding box center
                    cx = x + box_w / 2.0
                    cy = y + box_h / 2.0
                    
                    new_center = np.dot(M, np.array([cx, cy, 1.0]))
                    new_cx, new_cy = int(new_center[0]), int(new_center[1])
                    
                    new_x = max(0, int(new_cx - box_w / 2.0))
                    new_y = max(0, int(new_cy - box_h / 2.0))
                    
                    new_x_end = min(w, new_x + box_w)
                    new_y_end = min(h, new_y + box_h)
                    
                    face_crop = aligned_img[new_y:new_y_end, new_x:new_x_end]
                    
                    if face_crop.size == 0:
                        continue
                        
                    valid_faces.append({
                        'face': face_crop,
                        'facial_area': {'x': x, 'y': y, 'w': box_w, 'h': box_h},
                        'confidence': conf
                    })
                    
            return valid_faces
        except Exception as e:
            logger.error(f"Error during face extraction: {e}")
            return []

    def generate_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generates an embedding for a pre-cropped/aligned face image using ONNX.
        """
        try:
            self._ensure_initialized()
            start_time = time.time()
            
            # Preprocess for FaceNet
            face_img = cv2.resize(face_image, self.input_size)
            
            # Convert to RGB (FaceNet usually trained on RGB)
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1] or [-1, 1]
            # Standard FaceNet normalization is usually: img = (img - 127.5) / 128.0
            face_img = face_img.astype(np.float32)
            face_img = (face_img - 127.5) / 128.0
            
            if self.is_nchw:
                face_img = np.transpose(face_img, (2, 0, 1))
                
            input_tensor = np.expand_dims(face_img, axis=0)
            
            # Inference
            embedding = self.session.run(None, {self.input_name: input_tensor})[0][0]
            
            # L2 Normalize the embedding (Standard for FaceNet cosine similarity)
            embedding = embedding / np.linalg.norm(embedding)
            
            logger.debug(f"Generated embedding in {(time.time() - start_time)*1000:.2f}ms")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Computes cosine distance between two embeddings.
        Lower is more similar.
        """
        dot_product = np.dot(embedding1, embedding2)
        norm_a = np.linalg.norm(embedding1)
        norm_b = np.linalg.norm(embedding2)
        
        if norm_a == 0 or norm_b == 0:
            return 1.0 
            
        cosine_similarity = dot_product / (norm_a * norm_b)
        cosine_distance = 1 - cosine_similarity
        return float(cosine_distance)

# Singleton instance
embedding_service = EmbeddingService()
