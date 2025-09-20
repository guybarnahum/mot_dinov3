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

def parse_hf_spec(spec: Optional[str]):
    """
    Parse a compact Hugging Face spec and return (repo_id, file, revision, repo_type).

    ACCEPTED FORMS
      - "org/repo"
      - "org/repo@rev"
      - "org/repo#sub/dir/file.pth"
      - "org/repo#sub/dir/file.pth@rev"
      - "hf:org/repo[:sub/dir/file.pth][@rev]"
      - "spaces/owner/name#path@rev"      -> repo_type='space'
      - "datasets/owner/name#path@rev"    -> repo_type='dataset'
      - "models/owner/name#path@rev"      -> repo_type=None (model; default)

    Returns:
      (repo_id, file, revision, repo_type)  # repo_type in {None, 'space', 'dataset'}
    """
    if not spec or "/" not in spec:
        return (None, None, None, None)

    s = spec.strip()

    # Strip optional "hf:" prefix
    if s.startswith("hf:"):
        s = s[3:]

    repo_type = None
    # Normalize known type prefixes and strip them from the repo id
    for prefix, rtype in (
        ("spaces/", "space"), ("space/", "space"),
        ("datasets/", "dataset"), ("dataset/", "dataset"),
        ("models/", None), ("model/", None),
    ):
        if s.startswith(prefix):
            s = s[len(prefix):]
            repo_type = rtype
            break

    # Split file part: prefer '#', but also accept a single ':' (hf:org/repo:sub/file)
    repo_part, file_part = s, None
    if "#" in s:
        repo_part, file_part = s.split("#", 1)
    elif ":" in s and s.count(":") == 1:
        repo_part, file_part = s.split(":", 1)

    # Extract revision: can be after file or after repo
    revision = None
    if file_part and "@" in file_part:
        file_part, revision = file_part.rsplit("@", 1)
    elif "@" in repo_part:
        repo_part, revision = repo_part.rsplit("@", 1)

    repo_id = repo_part.strip() if repo_part else None
    file_path = file_part.strip() if file_part else None
    rev = revision.strip() if revision else None

    return (repo_id or None, file_path or None, rev or None, repo_type)
  
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

