# src/mot_dinov3/detector.py (Enhanced with Class Mapping and Printing)
from typing import Tuple, Dict, Optional, List
import numpy as np
from ultralytics import YOLO

# Default COCO class remapping for vehicle consolidation
DEFAULT_VEHICLE_REMAPPING = {
    7: 2,  # truck -> car
    5: 2,  # bus -> car
    3: 2,  # motorcycle -> car
    1: 2   # bicycle -> car
}

class Detector:
    """
    A wrapper for the Ultralytics YOLO detector with class remapping and summary printing.
    """
    def __init__(self, model_path: str = "yolov8n.pt", imgsz: int = 960, 
                 remap_classes: Optional[Dict[int, int]] = DEFAULT_VEHICLE_REMAPPING,
                 classes_to_track: Optional[List[int]] = None):
        """
        Initializes the YOLO detector and its configurations.
        
        Args:
            model_path (str): Path to the YOLO model file.
            imgsz (int): The image size for inference.
            remap_classes (dict, optional): Dictionary to remap class IDs.
            classes_to_track (list, optional): List of specific class IDs to detect.
        """
        print(f"💡 Initializing YOLO detector with model: {model_path}")
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.class_remap = remap_classes if remap_classes else {}

        # Print a compact summary of the model's names for the requested classes
        if classes_to_track and hasattr(self.model, 'names'):
            class_summary = []
            for cid in sorted(classes_to_track):
                name = self.model.names.get(cid, 'Unknown')
                class_summary.append(f"'{name}' ({cid})")
            print(f"   Tracking the following classes: {', '.join(class_summary)}")
        
        # Print the active class remapping for confirmation
        if self.class_remap and hasattr(self.model, 'names'):
            print("   Applying class remapping:")
            remap_summary = []
            for src_id, tgt_id in self.class_remap.items():
                src_name = self.model.names.get(src_id, f'ID {src_id}')
                tgt_name = self.model.names.get(tgt_id, f'ID {tgt_id}')
                remap_summary.append(f"'{src_name}' -> '{tgt_name}'")
            print(f"     - {', '.join(remap_summary)}")


    def detect(self, frame_bgr: np.ndarray, conf_thres: float = 0.3,
               classes_to_track: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs object detection and applies class remapping.
        """
        results = self.model.predict(
            source=frame_bgr, 
            imgsz=self.imgsz, 
            conf=conf_thres,
            classes=classes_to_track, # Pass classes directly for efficient filtering
            verbose=False,
        )
        
        r = results[0]
        boxes_obj = r.boxes

        if boxes_obj is None or len(boxes_obj) == 0:
            return (np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,)))

        boxes = boxes_obj.xyxy.cpu().numpy().astype(np.float32)
        conf = boxes_obj.conf.cpu().numpy().astype(np.float32)
        cls = boxes_obj.cls.cpu().numpy().astype(np.int32)
        
        # Apply class remapping after detection
        if self.class_remap:
            cls_remapped = cls.copy()
            for original_id, target_id in self.class_remap.items():
                cls_remapped[cls == original_id] = target_id
            return boxes, conf, cls_remapped
        
        return boxes, conf, cls