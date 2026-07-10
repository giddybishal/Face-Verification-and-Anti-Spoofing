import numpy as np
import time
from typing import Dict, Any, Tuple

from src.utils.logger import get_logger
from src.utils.config import config
from src.services.liveness_service import liveness_service
from src.services.embedding_service import embedding_service

logger = get_logger(__name__)

class VerificationService:
    def process_reference_image(self, image: np.ndarray) -> Tuple[bool, str, np.ndarray]:
        """
        Processes the reference image uploaded by the user.
        - Detects faces.
        - Checks for exactly one face.
        - Generates embedding.
        
        Returns: (success, message, embedding)
        """
        logger.info("Processing reference image...")
        
        faces = embedding_service.extract_faces(image)
        
        if len(faces) == 0:
            return False, "No face detected in the reference image.", None
            
        if len(faces) > 1:
            return False, "Multiple faces detected. Please upload an image with only one face.", None
            
        face_data = faces[0]
        face_img = face_data['face'] # This is the aligned and extracted face image
        
        # DeepFace returns face crop in RGB float32 if not careful, but let's assume it handles it
        # Actually, if we pass the whole original image to represent it's safer, but extract_faces already gives the face.
        # Let's pass the face crop to generate_embedding.
        
        # Actually DeepFace.represent takes the whole image by default, but also accepts numpy arrays.
        # If detector_backend is 'skip', it expects the image to be just the face.
        # Since face_data['face'] is already the face, we use it.
        # wait, sometimes DeepFace normalizes it. We will just use the original image with DeepFace.represent 
        # but with enforce_detection=True to get the embedding.
        # Let's do it safely:
        try:
            # We will just call represent directly on the original image for the reference. 
            # We already know there's exactly 1 face (or at least 1). 
            result = embedding_service.generate_embedding(face_img)
            
            if result is not None:
                return True, "Reference image processed successfully.", result
            else:
                return False, "Failed to generate embedding from reference image.", None
                
        except Exception as e:
            logger.error(f"Error processing reference image: {e}")
            return False, f"Error: {e}", None

    def process_webcam_frame(self, frame: np.ndarray, reference_embedding: np.ndarray) -> Dict[str, Any]:
        """
        Processes a single webcam frame.
        - Detects face.
        - Checks liveness via MiniFASNet.
        - If live, generates embedding.
        - Computes similarity against reference embedding.
        
        Returns dictionary with results.
        """
        start_time = time.time()
        
        result = {
            "status": "Processing",
            "message": "",
            "liveness": "Unknown",
            "similarity": None,
            "threshold": config.SIMILARITY_THRESHOLD,
            "verified": False,
            "processing_time_ms": 0
        }
        
        if reference_embedding is None:
            result["status"] = "Error"
            result["message"] = "Waiting for reference image."
            return result
            
        try:
            # 1. Detect faces for bounding box
            # DeepFace extract_faces returns bounding box in facial_area
            faces = embedding_service.extract_faces(frame)
            
            if len(faces) == 0:
                result["status"] = "Warning"
                result["message"] = "No face detected."
                return result
                
            if len(faces) > 1:
                result["status"] = "Warning"
                result["message"] = "Multiple faces detected."
                return result
                
            face_data = faces[0]
            facial_area = face_data.get('facial_area', {})
            if not facial_area:
                result["status"] = "Error"
                result["message"] = "Failed to extract facial area."
                return result
                
            x = facial_area.get('x', 0)
            y = facial_area.get('y', 0)
            w = facial_area.get('w', 0)
            h = facial_area.get('h', 0)
            
            bbox = (x, y, w, h)
            
            # 2. Check Liveness
            is_live, liveness_conf, liveness_details = liveness_service.predict(frame, bbox)
            
            if not is_live:
                result["status"] = "Failed"
                result["message"] = "Spoof detected! Pipeline terminated."
                result["liveness"] = f"Spoof (Conf: {liveness_conf:.2f})"
                result["processing_time_ms"] = (time.time() - start_time) * 1000
                return result
                
            result["liveness"] = f"Live (Conf: {liveness_conf:.2f})"
            
            # 3. Generate Embedding for Live Face
            face_img = face_data['face']
            frame_embedding = embedding_service.generate_embedding(face_img)
            
            if frame_embedding is None:
                result["status"] = "Error"
                result["message"] = "Failed to generate embedding."
                return result
                
            # 4. Compute Similarity
            similarity = embedding_service.compute_similarity(reference_embedding, frame_embedding)
            result["similarity"] = similarity
            
            # 5. Verification Decision
            # Since distance is calculated (lower is better for distance), we check if it's below threshold
            # Wait, the threshold config says "cosine similarity" but usually threshold applies to distance for DeepFace.
            # DeepFace cosine threshold for FaceNet512 is 0.3. So distance < threshold means verified.
            is_verified = bool(similarity < config.SIMILARITY_THRESHOLD)
            
            result["verified"] = is_verified
            if is_verified:
                result["status"] = "Success"
                result["message"] = "Identity verified."
            else:
                result["status"] = "Failed"
                result["message"] = "Identity not verified."
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            result["status"] = "Error"
            result["message"] = str(e)
            
        result["processing_time_ms"] = (time.time() - start_time) * 1000
        return result

verification_service = VerificationService()
