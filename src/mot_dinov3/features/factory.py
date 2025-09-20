# src/mot_dinov3/features/factory.py
from __future__ import annotations
from .dino import DINOExtractor
from .transreid import TransReIDExtractor
from .osnet import OSNetExtractor

def create_extractor(kind: str, model_id: str, device: str, autocast: bool, pad: float, square: bool, **kwargs):
    k = (kind or "dino").lower()
    if k in ("dino","dinov3","dinov2"):
        return DINOExtractor(model_id, device, autocast, pad, square, **kwargs)
    if k in ("transreid","reid","trans-reid"):
        return TransReIDExtractor(model_id, device, autocast, pad, square, **kwargs)
    if k in ("osnet","os-net"):
        return OSNetExtractor(model_id, device, autocast, pad, square, **kwargs)
    raise ValueError(f"Unknown embedder kind: {kind}")
