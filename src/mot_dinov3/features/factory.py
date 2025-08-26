# src/mot_dinov3/features/factory.py
from __future__ import annotations
from .dino import DINOExtractor

def create_extractor(kind: str, model_id: str, device: str, autocast: bool, pad: float, square: bool):
    kind = (kind or "dino").lower()
    if kind in ("dino", "dinov3", "dinov2"):
        return DINOExtractor(model_id, device, autocast, pad, square)
    raise ValueError(f"Unknown embedder kind: {kind}")

