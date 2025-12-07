#Project.py

import streamlit as st
import tempfile
import cv2
import os
from PIL import Image
from yolov8_detector import YOLOv8Detector

# --- Configuration ---
MODEL_WEIGHTS = 'yolov8n.pt' # Change this to your actual custom-trained model file (e.g., 'yolov8_safety.pt')

# --- Initialize Detector (Caching for performance) ---
@st.cache_resource
def load_detector():
    """Load the YOLOv8 model once and cache it."""
    detector = YOLOv8Detector(model_path=MODEL_WEIGHTS)
    return detector

detector = load_detector()

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="Community Vision: AI Hazard Detection Prototype",
    layout="wide"
)

st.title("🚦 Community Vision: AI-Powered Hazard Detection")
st.markdown("Prototype application for real-time urban pedestrian safety monitoring using **YOLOv8**.")

# File Uploader
st.sidebar.header("Upload File")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image or video file...",
    type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov']
)

# Confidence Slider
st.sidebar.header("Model Settings")
confidence = st.sidebar.slider(
    'Confidence Threshold', 0.0, 1.0, 0.25, 0.05
)

if uploaded_file is not None and detector.model is not None:
    file_type = uploaded_file.type.split('/')[0]

    st.subheader(f"Processing Uploaded {file_type.capitalize()}")
    
    # Create two columns for side-by-side comparison
    col1, col2 = st.columns(2)
    
    # --- Image Processing ---
    if file_type == 'image':
        
        # Display Original Image
        original_image = Image.open(uploaded_file)
        with col1:
            st.markdown("**Original Image**")
            st.image(original_image, caption="Uploaded Image", use_column_width=True)

        # Process and Display Detected Image
        try:
            # Convert PIL image to OpenCV format (numpy array)
            frame = np.array(original_image.convert('RGB'))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Manually set confidence for the single-run inference
            detector.model.conf = confidence
            detected_frame = detector.detect_and_draw(frame)
            
            # Convert back to RGB for Streamlit display
            detected_image = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.markdown("**AI Hazard Detection Result**")
                st.image(detected_image, caption="Detected Hazards", use_column_width=True)
                
            st.success("Image processing complete!")
            
        except Exception as e:
            st.error(f"Error processing image: {e}")

    # --- Video Processing ---
    elif file_type == 'video':
        
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name

        st.info("Processing video. This may take a moment...")
        
        # Use a placeholder for real-time video output
        video_placeholder = st.empty()
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup temporary video writer for the output video
        output_file_name = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
        out = cv2.VideoWriter(output_file_name, fourcc, fps, (width, height))
        
        frame_count = 0
        
        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Manually set confidence for the single-run inference
            detector.model.conf = confidence
            detected_frame = detector.detect_and_draw(frame)
            
            # Write the processed frame to the output video file
            out.write(detected_frame)

            # Display a progress update
            if frame_count % int(fps) == 0: # Update every second
                cv_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(cv_rgb, caption="Real-time Detection", use_column_width=True, channels="RGB")
            
            frame_count += 1

        cap.release()
        out.release()
        os.unlink(video_path) # Clean up the temporary input file
        
        st.success("Video processing complete!")

        # Provide a download link for the processed video
        with open(output_file_name, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name=output_file_name,
                mime="video/mp4"
            )

        # Clean up the output video file
        if os.path.exists(output_file_name):
            os.unlink(output_file_name)


elif detector.model is None:
    st.error("🚨 **Error:** The YOLOv8 model failed to load. Please ensure your model file is in the correct path or check your installation.")

else:
    st.info("👆 Please upload an **image** or **video** file to begin hazard detection.")
