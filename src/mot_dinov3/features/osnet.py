# src/mot_dinov3/features/osnet.py
from __future__ import annotations
from ..embedders import create_embedder
from .base import GenericExtractor

class TransReIDExtractor(GenericExtractor):
    def __init__(self, model_id: str, device: str, autocast: bool,
                 pad: float=0.08, square: bool=False, image_size=(256,128), **kwargs):
        e = create_embedder("osnet", model_name=model_id, device=device,
                            use_autocast=False, image_size=image_size)
        super().__init__(e, name=f"osnet:{model_id}", device=device,
                         pad=pad, square=square, autocast=autocast)


