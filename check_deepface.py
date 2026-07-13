import cv2
import numpy as np
from deepface import DeepFace

def main():
    print("Testing DeepFace...")
    img = cv2.imread("obama1.jpg")
    
    # 1. Look at how deepface represents the face
    try:
        results = DeepFace.represent(img_path="obama1.jpg", model_name="Facenet512", detector_backend="retinaface", enforce_detection=True)
        emb1 = results[0]["embedding"]
        print(f"DeepFace embedding norm: {np.linalg.norm(emb1)}")
        
        results2 = DeepFace.represent(img_path="obama2.jpg", model_name="Facenet512", detector_backend="retinaface", enforce_detection=True)
        emb2 = results2[0]["embedding"]
        
        results3 = DeepFace.represent(img_path="biden1.jpg", model_name="Facenet512", detector_backend="retinaface", enforce_detection=True)
        emb3 = results3[0]["embedding"]
        
        print("DeepFace Cosine distances:")
        def cos_dist(a, b):
            a = np.array(a)
            b = np.array(b)
            return 1 - np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
            
        print("obama1 vs obama2:", cos_dist(emb1, emb2))
        print("obama1 vs biden1:", cos_dist(emb1, emb3))
        
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
