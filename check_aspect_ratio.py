import cv2
from src.services.embedding_service import embedding_service

def main():
    embedding_service._ensure_initialized()
    images = ["obama1.jpg", "obama2.jpg", "biden1.jpg"]
    for img_name in images:
        img = cv2.imread(img_name)
        if img is None:
            continue
        faces = embedding_service.extract_faces(img)
        if faces:
            face_img = faces[0]['face']
            h, w = face_img.shape[:2]
            print(f"{img_name} - crop size: {w}x{h}, aspect ratio: {w/h:.2f}")

if __name__ == "__main__":
    main()
