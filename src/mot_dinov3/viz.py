import cv2
import numpy as np
from .tracker import Track

def draw_tracks(frame: np.ndarray, tracks: list[Track], color=(0, 255, 0)):
    for t in tracks:
        x1, y1, x2, y2 = t.box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {t.tid}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

