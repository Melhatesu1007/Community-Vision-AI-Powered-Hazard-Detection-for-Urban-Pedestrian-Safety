# yolov8_detector.py

import cv2
import numpy as np
from ultralytics import YOLO

class YOLOv8Detector:
    """
    A class to handle the loading and inference of a YOLOv8 model.
    """
    def __init__(self, model_path='yolov8n.pt'):
        # NOTE: If your custom model is very large (e.g., > 50MB), 
        # you may need to host it outside of GitHub (e.g., Google Drive) 
        # and download it at runtime. 'yolov8n.pt' will download automatically 
        # if not found locally, which is generally acceptable for cloud deployment.
        try:
            self.model = YOLO(model_path)
            print(f"YOLOv8 model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            
    def detect_and_draw(self, frame):
        """
        Runs inference on a single frame (image) and draws bounding boxes.
        
        Args:
            frame (np.ndarray): The image frame in BGR format.
            
        Returns:
            np.ndarray: The frame with detection bounding boxes drawn on it.
        """
        if self.model is None:
            return frame

        # Run inference on the frame
        # Model confidence will be dynamically set by the Streamlit app.py
        results = self.model(frame, verbose=False)
        
        # Draw bounding boxes and labels onto the image
        annotated_frame = results[0].plot()
        
        return annotated_frame
