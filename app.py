import gradio as gr
import numpy as np
import spaces

from src.services.verification_service import verification_service
from src.utils.logger import get_logger

logger = get_logger(__name__)

@spaces.GPU
def process_reference(image: np.ndarray):
    """
    Handles the upload of the reference image.
    Returns: status message, the reference embedding state, and an empty history list.
    """
    if image is None:
        return "Please upload an image.", None, []
        
    success, message, embedding = verification_service.process_reference_image(image)
    if success:
        return f"✅ {message}", embedding, []
    else:
        return f"❌ {message}", None, []

@spaces.GPU
def process_frame(frame: np.ndarray, reference_embedding: np.ndarray, history: list):
    """
    Processes a frame from the webcam.
    Returns the HTML formatted status for the UI and updated history.
    """
    if history is None:
        history = []
        
    if frame is None:
        return "No frame detected.", history
        
    if reference_embedding is None:
        return "<h3>Waiting for Reference Image...</h3>", history
        
    result = verification_service.process_webcam_frame(frame, reference_embedding)
    
    # Add to history if a face was processed
    if result["status"] not in ["Warning", "Error"]:
        history.append({
            "similarity": result.get("similarity"),
            "is_live": "Live" in result.get("liveness", ""),
            "verified": result.get("verified", False)
        })
        
    # Maintain a sliding window of max 10 frames
    MAX_FRAMES = 10
    if len(history) > MAX_FRAMES:
        history.pop(0)
        
    # If we don't have enough history yet, or no valid frames
    if len(history) == 0:
        html = f"""
        <div style="padding: 15px; border-radius: 10px; border: 1px solid var(--border-color-primary, #ccc); background: var(--background-fill-secondary, #f9f9f9); color: var(--body-text-color, black);">
            <h3 style="margin-top: 0;">Verification Status: {result['status']}</h3>
            <p><strong>Message:</strong> {result['message']}</p>
        </div>
        """
        return html, history
        
    # Aggregate
    valid_sims = [h["similarity"] for h in history if h["similarity"] is not None]
    avg_similarity = sum(valid_sims) / len(valid_sims) if valid_sims else None
    
    live_count = sum(1 for h in history if h["is_live"])
    is_live_aggregated = live_count > (len(history) / 2)
    
    threshold = result['threshold']
    is_verified_aggregated = (avg_similarity is not None and avg_similarity < threshold)
    
    if is_verified_aggregated and is_live_aggregated:
        status = "Success"
        message = "Identity verified (Stable)."
    elif not is_live_aggregated:
        status = "Failed"
        message = "Spoof detected (Stable)!"
    else:
        status = "Failed"
        message = "Identity not verified (Stable)."
    
    # Format the result as HTML for better presentation
    html = f"""
    <div style="padding: 15px; border-radius: 10px; border: 1px solid var(--border-color-primary, #ccc); background: var(--background-fill-secondary, #f9f9f9); color: var(--body-text-color, black);">
        <h3 style="margin-top: 0;">Verification Status: {status}</h3>
        <p><strong>Message:</strong> {message}</p>
        <hr>
        <p><strong>Liveness (Stable):</strong> {"Live" if is_live_aggregated else "Spoof"} ({live_count}/{len(history)} frames)</p>
        <p><strong>Similarity Distance (Avg over {len(history)} frames):</strong> {f"{avg_similarity:.4f}" if avg_similarity is not None else "N/A"}</p>
        <p><strong>Threshold:</strong> {threshold} (Lower is better)</p>
        <p><strong>Current Frame Processing Time:</strong> {result['processing_time_ms']:.1f} ms</p>
    </div>
    """
    
    return html, history

def create_ui():
    """
    Creates and configures the Gradio Blocks UI.
    """
    with gr.Blocks(title="Face Verification Pipeline", theme=gr.themes.Soft()) as demo:
        # State to store the embedding
        reference_embedding = gr.State(None)
        history = gr.State([])
        
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
            outputs=[ref_status_text, reference_embedding, history]
        )
        
        reference_input.clear(
            fn=lambda: ("Waiting for image...", None, []),
            inputs=[],
            outputs=[ref_status_text, reference_embedding, history]
        )
        
        # When webcam streams
        webcam_input.stream(
            fn=process_frame,
            inputs=[webcam_input, reference_embedding, history],
            outputs=[results_output, history],
            time_limit=15 # Adjust if it buffers too much
        )
        
    return demo

if __name__ == "__main__":
    logger.info("Starting Gradio Application...")
    demo = create_ui()
    # Share=False for security, but allow it to run on any network interface for Spaces
    demo.launch(server_name="0.0.0.0", server_port=7860)
