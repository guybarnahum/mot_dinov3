# src/mot_dinov3/features/dino.py
from __future__ import annotations
from typing import Optional, Any
import numpy as np
from ..embedder import DinoV3Embedder

class DINOExtractor:
    """
    Thin adapter around DinoV3Embedder so the tracker pipeline can treat it as a generic
    feature extractor. You can set default crop behavior at construction time, and/or
    override per-call via pad_ratio/square kwargs.
    """
    def __init__(self, model_id: str, device: str, autocast: bool,
                 pad: float = 0.12, square: bool = True):
        self._e = DinoV3Embedder(model_id, device=device, use_autocast=autocast)
        self.name = f"dino:{model_id}"
        self.dim = getattr(self._e, "emb_dim", 768)
        self._pad_default = float(pad)
        self._square_default = bool(square)

    def embed_crops(self,
                    frame_bgr: np.ndarray,
                    boxes_xyxy: np.ndarray,
                    pad_ratio: Optional[float] = None,
                    square: Optional[bool] = None,
                    **_: Any) -> np.ndarray:
        """
        Extract embeddings for the given boxes.

        Parameters (overrides are optional)
        - pad_ratio: float | None  -> padding around box (defaults to ctor pad)
        - square: bool | None      -> make crop square (defaults to ctor square)
        - **_: Any                 -> ignore unknown kwargs for forward-compat

        Returns
        - np.ndarray of shape (N, D), L2-normalized
        """
        pad = self._pad_default if pad_ratio is None else float(pad_ratio)
        sq = self._square_default if square is None else bool(square)
        return self._e.embed_crops(frame_bgr, boxes_xyxy, pad_ratio=pad, square=sq)
