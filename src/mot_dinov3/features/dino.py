# src/mot_dinov3/features/dino.py
from __future__ import annotations
from ..embedders import create_embedder
from .base import GenericExtractor

class DINOExtractor(GenericExtractor):
    def __init__(self, model_id: str, device: str, autocast: bool,
                 pad: float=0.12, square: bool=True, **kwargs):
        e = create_embedder("dino", model_name=model_id, device=device, use_autocast=False)
        super().__init__(e, name=f"dino:{model_id}", device=device,
                         pad=pad, square=square, autocast=autocast)
