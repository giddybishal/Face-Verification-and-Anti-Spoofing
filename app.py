import gradio as gr
import numpy as np

from src.services.verification_service import verification_service
from src.utils.logger import get_logger

logger = get_logger(__name__)

def process_reference(image: np.ndarray):
    """
    Handles the upload of the reference image.
    Returns: status message and the reference embedding state.
    """
    if image is None:
        return "Please upload an image.", None
        
    success, message, embedding = verification_service.process_reference_image(image)
    if success:
        return f"✅ {message}", embedding
    else:
        return f"❌ {message}", None

def process_frame(frame: np.ndarray, reference_embedding: np.ndarray):
    """
    Processes a frame from the webcam.
    Returns the HTML formatted status for the UI.
    """
    if frame is None:
        return "No frame detected."
        
    if reference_embedding is None:
        return "<h3>Waiting for Reference Image...</h3>"
        
    result = verification_service.process_webcam_frame(frame, reference_embedding)
    
    # Format the result as HTML for better presentation
    html = f"""
    <div style="padding: 15px; border-radius: 10px; border: 1px solid #ccc; background: #f9f9f9; color: black;">
        <h3 style="margin-top: 0;">Verification Status: {result['status']}</h3>
        <p><strong>Message:</strong> {result['message']}</p>
        <hr>
        <p><strong>Liveness:</strong> {result['liveness']}</p>
        <p><strong>Similarity Distance:</strong> {f"{result['similarity']:.4f}" if result['similarity'] is not None else "N/A"}</p>
        <p><strong>Threshold:</strong> {result['threshold']} (Lower is better)</p>
        <p><strong>Processing Time:</strong> {result['processing_time_ms']:.1f} ms</p>
    </div>
    """
    
    return html

def create_ui():
    """
    Creates and configures the Gradio Blocks UI.
    """
    with gr.Blocks(title="Face Verification Pipeline", theme=gr.themes.Soft()) as demo:
        # State to store the embedding
        reference_embedding = gr.State(None)
        
        gr.Markdown("# 🛡️ Face Verification Pipeline")
        gr.Markdown("Prototype for production-ready face verification using MiniFASNet for liveness and FaceNet512 for embeddings.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Upload Reference Image")
                reference_input = gr.Image(sources=["upload"], type="numpy", label="Profile Picture")
                ref_status_text = gr.Markdown("Waiting for image...")
                
            with gr.Column(scale=2):
                gr.Markdown("### 2. Live Verification")
                webcam_input = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam")
                
            with gr.Column(scale=1):
                gr.Markdown("### 3. Verification Results")
                results_output = gr.HTML("Waiting for stream...")
                
        # Event Listeners
        # When reference image is uploaded
        reference_input.upload(
            fn=process_reference,
            inputs=[reference_input],
            outputs=[ref_status_text, reference_embedding]
        )
        
        reference_input.clear(
            fn=lambda: ("Waiting for image...", None),
            inputs=[],
            outputs=[ref_status_text, reference_embedding]
        )
        
        # When webcam streams
        webcam_input.stream(
            fn=process_frame,
            inputs=[webcam_input, reference_embedding],
            outputs=[results_output],
            time_limit=15 # Adjust if it buffers too much
        )
        
    return demo

if __name__ == "__main__":
    logger.info("Starting Gradio Application...")
    demo = create_ui()
    # Share=False for security, but allow it to run on any network interface for Spaces
    demo.launch(server_name="0.0.0.0", server_port=7860)
