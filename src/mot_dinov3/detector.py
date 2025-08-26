from typing import Tuple
import numpy as np
from ultralytics import YOLO

class Detector:
    """
    Ultralytics YOLO detector wrapper.
    detect(frame_bgr) -> boxes_xyxy (N,4), conf (N,), cls (N,)
    """
    def __init__(self, model: str = "yolov8n.pt", device: str | None = None, imgsz: int = 960):
        self.model = YOLO(model)
        self.device = device or ("cuda" if self.model.device.type != "cpu" else "cpu")
        self.imgsz = imgsz

    def detect(self, frame_bgr: np.ndarray, conf_thres: float = 0.3
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        results = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=conf_thres,
                                     verbose=False, device=self.device)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32))
        b = r.boxes
        boxes = b.xyxy.detach().cpu().numpy().astype(np.float32)
        conf = b.conf.detach().cpu().numpy().astype(np.float32)
        cls  = b.cls.detach().cpu().numpy().astype(np.int32)
        return boxes, conf, cls

