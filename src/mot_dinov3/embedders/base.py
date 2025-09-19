# src/mot_dinov3/embedders/base.py
from __future__ import annotations
from typing import Protocol, Dict, List, Optional, Tuple
import os, torch, numpy as np
from PIL import Image

def get_hf_token() -> Optional[str]:
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.getenv(k): return os.getenv(k)
    return None

class BaseEmbedder(Protocol):
    model_name: str
    device: str
    emb_dim: int
    processor: Optional[object]

    def _prep_inputs(self, pil_images: List[Image.Image]) -> Dict[str, torch.Tensor]: ...
    def _extract_feat(self, out) -> torch.Tensor: ...
    def embed_crops(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray,
                    pad_ratio: float = 0.1, square: bool = False) -> np.ndarray: ...

def pick_amp_dtype() -> torch.dtype:
    try:
        return torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
    except Exception:
        return torch.float16

