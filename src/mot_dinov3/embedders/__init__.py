# src/mot_dinov3/embedders/__init__.py
from .dino import DinoV3Embedder
try:
    from .transreid import TransReIDEmbedder
except Exception:
    TransReIDEmbedder = None

try:
    from .osnet import OSNetEmbedder           
except Exception:
    OSNetEmbedder = None

from .base import GatedModelAccessError, parse_hf_spec 

_REG = {
    "dino": DinoV3Embedder, "dinov3": DinoV3Embedder, "dinov2": DinoV3Embedder,
    "transreid": TransReIDEmbedder, "reid": TransReIDEmbedder, "trans-reid": TransReIDEmbedder,
}

def create_embedder(kind: str, **kwargs):
    k = (kind or "dino").lower()
    cls = _REG.get(k)
    if cls is None:
        raise ValueError(f"Unknown or unavailable embedder kind: {kind}")
    return cls(**kwargs)
