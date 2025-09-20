# src/mot_dinov3/embedders/base.py
from __future__ import annotations
from typing import Protocol, Dict, List, Optional, Tuple
import os, torch, numpy as np
from PIL import Image

class GatedModelAccessError(RuntimeError):
    """Raised when a gated HF repo is requested without granted access / auth."""

def get_hf_token() -> Optional[str]:
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.getenv(k): return os.getenv(k)
    return None

def load_with_token(factory, repo_id: str, **extra):
    """Call HF .from_pretrained with a token if present; supports old/new kw names."""
    tok = get_hf_token()
    try:
        return factory(repo_id, token=tok, **extra) if tok else factory(repo_id, **extra)
    except TypeError:
        return factory(repo_id, use_auth_token=tok, **extra) if tok else factory(repo_id, **extra)

def pick_amp_dtype() -> torch.dtype:
    try:
        return torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
    except Exception:
        return torch.float16

class BaseEmbedder(Protocol):
    model_name: str
    device: str
    emb_dim: int
    processor: Optional[object]
    def _prep_inputs(self, pil_images: List[Image.Image]) -> Dict[str, torch.Tensor]: ...
    def _extract_feat(self, out) -> torch.Tensor: ...
    def embed_crops(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray,
                    pad_ratio: float = 0.1, square: bool = False) -> np.ndarray: ...
