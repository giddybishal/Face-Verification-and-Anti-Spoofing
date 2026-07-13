import os
import cv2
import numpy as np
import urllib.request
from src.services.embedding_service import embedding_service
from src.utils.config import config

def download_image(url, filename):
    if not os.path.exists(filename):
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
            out_file.write(response.read())

def main():
    print("Downloading test images...")
    # Obama
    download_image("https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg", "obama1.jpg")
    download_image("https://upload.wikimedia.org/wikipedia/commons/e/e9/Official_portrait_of_Barack_Obama.jpg", "obama2.jpg")
    # Biden
    download_image("https://upload.wikimedia.org/wikipedia/commons/6/68/Joe_Biden_presidential_portrait.jpg", "biden1.jpg")
    
    images = ["obama1.jpg", "obama2.jpg", "biden1.jpg"]
    
    # Initialize service
    embedding_service._ensure_initialized()
    
    embeddings = []
    
    for img_name in images:
        print(f"\nProcessing {img_name}")
        img = cv2.imread(img_name)
        if img is None:
            print(f"Failed to load {img_name}")
            continue
            
        faces = embedding_service.extract_faces(img)
        print(f"Detected {len(faces)} faces.")
        
        if len(faces) > 0:
            face_img = faces[0]['face']
            cv2.imwrite(f"crop_{img_name}", face_img)
            print(f"Saved crop_{img_name}. Shape: {face_img.shape}")
            
            emb = embedding_service.generate_embedding(face_img)
            print(f"Embedding shape: {emb.shape}, Norm: {np.linalg.norm(emb)}")
            embeddings.append((img_name, emb))
            
    print("\nPairwise Cosine Distances:")
    print("Threshold in config:", config.SIMILARITY_THRESHOLD)
    for i in range(len(embeddings)):
        for j in range(i, len(embeddings)):
            name1, emb1 = embeddings[i]
            name2, emb2 = embeddings[j]
            dist = embedding_service.compute_similarity(emb1, emb2)
            print(f"{name1} vs {name2}: {dist:.4f}")

if __name__ == "__main__":
    main()
