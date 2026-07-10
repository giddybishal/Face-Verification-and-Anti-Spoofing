import os
import tf2onnx
import tensorflow as tf
from deepface import DeepFace
import onnx

def export_facenet512():
    print("Building Facenet512 model via DeepFace...")
    model = DeepFace.build_model("Facenet512")
    
    # Save the keras model to a temp directory to convert it, or convert directly.
    # We can convert a tf.keras.Model directly using tf2onnx.convert.from_keras
    
    print("Exporting Keras model to ONNX...")
    spec = (tf.TensorSpec((None, 160, 160, 3), tf.float32, name="input_1"),)
    
    output_path = "facenet512.onnx"
    
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
    print(f"Successfully exported FaceNet512 to {output_path}")

if __name__ == "__main__":
    export_facenet512()
