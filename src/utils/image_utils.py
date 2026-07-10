import base64
import cv2
import numpy as np
from typing import Union
from PIL import Image
import io

def base64_to_numpy(base64_str: str) -> np.ndarray:
    """
    Converts a base64 string to a numpy array (BGR format for OpenCV).
    """
    # Remove prefix if present (e.g., 'data:image/jpeg;base64,')
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
        
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np

def numpy_to_base64(img_np: np.ndarray, format: str = ".jpg") -> str:
    """
    Converts a numpy array (BGR format) to a base64 string.
    """
    _, buffer = cv2.imencode(format, img_np)
    base64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/{format[1:]};base64,{base64_str}"

def pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
    """
    Converts a PIL image to an OpenCV numpy array (BGR).
    """
    # Convert to RGB first to ensure consistency, then to BGR
    img_np = np.array(pil_img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_bgr

def standardize_image_input(image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Takes an image in various formats (base64, PIL, numpy) and standardizes 
    it to an OpenCV numpy array (BGR format).
    """
    if isinstance(image, str):
        return base64_to_numpy(image)
    elif isinstance(image, Image.Image):
        return pil_to_numpy(image)
    elif isinstance(image, np.ndarray):
        # Assume it's already in BGR format if it's a numpy array from cv2
        return image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

def get_crop_with_margin(image: np.ndarray, bbox: tuple[int, int, int, int], scale: float = 2.7) -> np.ndarray:
    """
    Crops a face from an image with a specific margin scale.
    This is often required for MiniFASNet which needs context around the face.
    
    bbox is (x, y, w, h)
    """
    h_img, w_img = image.shape[:2]
    x, y, w, h = bbox
    
    # Calculate center
    cx = x + w / 2
    cy = y + h / 2
    
    # Scale width and height
    w_new = w * scale
    h_new = h * scale
    
    # Calculate new bounding box
    x_new = max(0, int(cx - w_new / 2))
    y_new = max(0, int(cy - h_new / 2))
    x_new_end = min(w_img, int(cx + w_new / 2))
    y_new_end = min(h_img, int(cy + h_new / 2))
    
    # Crop
    crop = image[y_new:y_new_end, x_new:x_new_end]
    return crop
