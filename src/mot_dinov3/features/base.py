# src/mot_dinov3/features/base.py
from typing import Protocol, Optional, Any
import numpy as np

class FeatureExtractor(Protocol):
    name: str
    dim: int
    def embed_crops(self,
                    frame_bgr: np.ndarray,
                    boxes_xyxy: np.ndarray,
                    pad_ratio: Optional[float] = None,
                    square: Optional[bool] = None,
                    **kwargs: Any) -> np.ndarray: ...
