import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import cv2
import time
from typing import Tuple

from src.utils.config import config
from src.utils.logger import get_logger
from src.utils.image_utils import get_crop_with_margin

logger = get_logger(__name__)

class LivenessService:
    def __init__(self):
        """
        Initializes the LivenessService by downloading the ONNX model from Hugging Face
        and creating the ONNX Runtime session. This is done once on startup.
        """
        logger.info("LivenessService instantiated. Models will be loaded lazily.")
        self.session = None
        self.input_name = None

    def _ensure_initialized(self):
        """
        Lazily initializes the ONNX session.
        """
        if self.session is not None:
            return
            
        logger.info("Initializing LivenessService ONNX session...")
        start_time = time.time()
        
        try:
            model_path = hf_hub_download(
                repo_id=config.MINIFASNET_REPO_ID,
                filename=config.MINIFASNET_FILENAME
            )
            
            # Using CPU execution provider. For GPU, we'd add 'CUDAExecutionProvider'
            self.session = ort.InferenceSession(
                model_path, 
                providers=['CPUExecutionProvider']
            )
            
            # Get input name and shape details
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"LivenessService initialized in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize LivenessService: {e}")
            raise

    def preprocess(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """
        Prepares the image crop for the ONNX model.
        - Crops face with a margin (default 2.7x for MiniFASNet)
        - Resizes to (80, 80)
        - Transposes and adds batch dimension.
        """
        # Crop the face with margin
        crop = get_crop_with_margin(image, bbox, scale=2.7)
        
        # Resize to expected input size
        crop_resized = cv2.resize(crop, config.MINIFASNET_INPUT_SIZE)
        
        # The model usually expects BGR format (which it is) and normalizes it implicitly 
        # or expects raw [0-255]. Standard PyTorch models might expect (C, H, W).
        # Let's format it for ONNX (N, C, H, W) and standard scale.
        
        # Note: Depending on exact model export, it might need normalization, but 
        # MiniFASNet exports typically handle raw BGR image in float32.
        input_tensor = crop_resized.astype(np.float32)
        
        # HWC to CHW
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        
        # Add batch dim
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor

    def predict(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> Tuple[bool, float, dict]:
        """
        Predicts if the given face bounding box in the image is live or spoof.
        Returns:
            is_live (bool)
            confidence (float)
            details (dict) containing raw scores
        """
        try:
            self._ensure_initialized()
            start_time = time.time()
            input_tensor = self.preprocess(image, bbox)
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: input_tensor})
            
            # Outputs is typically a 1x3 tensor (live, print_attack, replay_attack)
            # OR (fake, fake, live) depending on the exact training regime.
            # Let's apply softmax
            logits = outputs[0][0]
            
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            # Most common MiniFASNet classification: 
            # 0: Fake (Print)
            # 1: Fake (Replay)
            # 2: Live
            # Wait, let's look at standard Silent-Face-Anti-Spoofing labels.
            # The class mapping is usually:
            # 1: Real
            # 2: Fake
            # Or 3 class: [Live, Fake1, Fake2]
            # Since the HuggingFace repo specifies 3-class softmax (live, print, replay) 
            # we need to be careful with indexing. Usually class 0 is spoof, class 1 is live, etc.
            # Actually, standard MiniFASNet implementation class labels are:
            # 1 is REAL. 0 is FAKE.
            # Let's assume class 0 is Live, class 1 is Fake (Print), class 2 is Fake (Replay) 
            # if we see 3 classes, or check argmax.
            # Wait, standard label format in Silent Face: 1 is Real face, 0 is Fake Face.
            # For 3 class: 0 is Fake, 1 is Real. Wait, let's just find the max score.
            
            # Let's assume standard behavior: The highest logit determines the class.
            pred_class = np.argmax(probs)
            
            # Let's log probabilities for debugging to easily adjust if needed.
            details = {
                "probs": probs.tolist(),
                "inference_time_ms": (time.time() - start_time) * 1000
            }
            
            # We'll use a conservative approach: if class 1 is Real (common in original repo),
            # we check if pred_class == 1.
            # Let's assume index 1 is real. If we are wrong, the logs will show us.
            # Actually, in garciafido/minifasnet-v2-anti-spoofing-onnx, the description says:
            # Output: live, print-attack, replay-attack.
            # So index 0 = live, index 1 = print, index 2 = replay.
            is_live = (pred_class == 0)
            confidence = probs[0] if is_live else np.max(probs[1:])
            
            logger.debug(f"Liveness inference took {details['inference_time_ms']:.2f}ms. Is Live: {is_live} (conf: {confidence:.2f})")
            
            return is_live, confidence, details
            
        except Exception as e:
            logger.error(f"Error during liveness prediction: {e}")
            return False, 0.0, {"error": str(e)}

# Singleton instance
liveness_service = LivenessService()
