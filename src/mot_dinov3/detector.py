# src/mot_dinov3/detector.py (Refactored)
from typing import Tuple
import numpy as np
from ultralytics import YOLO

class Detector:
    """
    A simplified wrapper for the Ultralytics YOLO detector.
    
    This class initializes a YOLO model and provides a clean interface
    to get detection results (boxes, confidences, classes). It lets the
    underlying YOLO object manage its own device (CPU/GPU).
    """
    def __init__(self, model_path: str = "yolov8n.pt", imgsz: int = 960):
        """
        Initializes the YOLO detector.
        
        Args:
            model_path (str): Path to the YOLO model file (e.g., 'yolov8n.pt').
            imgsz (int): The image size for inference.
        """
        self.model = YOLO(model_path)
        self.imgsz = imgsz

    def detect(self, frame_bgr: np.ndarray, conf_thres: float = 0.3
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs object detection on a single frame.

        Args:
            frame_bgr (np.ndarray): The input image in BGR format.
            conf_thres (float): The confidence threshold for detections.

        Returns:
            A tuple containing:
            - boxes_xyxy (np.ndarray): (N, 4) array of bounding boxes.
            - conf (np.ndarray): (N,) array of confidences.
            - cls (np.ndarray): (N,) array of class IDs.
        """
        results = self.model.predict(
            source=frame_bgr, 
            imgsz=self.imgsz, 
            conf=conf_thres,
            verbose=False,
            # device=None (default) uses the device the model is already on
        )
        
        # The result for the first image
        r = results[0]
        boxes_obj = r.boxes

        if boxes_obj is None or len(boxes_obj) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32)
            )

        # Extract and convert to NumPy arrays
        boxes = boxes_obj.xyxy.cpu().numpy().astype(np.float32)
        conf = boxes_obj.conf.cpu().numpy().astype(np.float32)
        cls = boxes_obj.cls.cpu().numpy().astype(np.int32)
        
        return boxes, conf, cls