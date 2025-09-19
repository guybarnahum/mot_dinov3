# src/mot_dinov3/embedders/dino.py
from __future__ import annotations
import os
import numpy as np
import torch, cv2

try:
    from PIL import Image
    Resample = Image.Resampling
except Exception:
    Resample = Image  # Pillow < 10
# then use Resample.BICUBIC

from typing import List, Tuple, Optional, Any, Dict
from contextlib import nullcontext

# Apply shims early (torch compiler / NumPy, and HF token bridge)
from .. import compat as _compat
_compat.apply(strict_numpy=False, quiet=True)

from transformers import AutoImageProcessor, AutoModel
from .base import GatedModelAccessError, load_with_token 

def _to_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


class DinoV3Embedder:
    """
    Robust DINO(v3/v2) embedder.

    Load order:
      1) AutoImageProcessor (trust_remote_code=True, prefer fast)
      2) If unavailable/unrecognized, build a manual preprocessor from model config:
         - image_size from config (or 224)
         - mean/std from config (or ImageNet)
    Produces L2-normalized global embeddings from pooled output or patch-token mean.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        device: Optional[str] = None,
        use_autocast: bool = True,
        verbose: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_autocast = bool(use_autocast and (self.device == "cuda") and torch.cuda.is_available())
        self.model_name = model_name
        self.verbose = verbose

        # Grab token from env (supports HF_TOKEN and legacy names via compat bridge)
        tok = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )

        # --- Load model first (we can still run manual preprocessing) ---
        try:
            self.model = load_with_token(
                AutoModel.from_pretrained, model_name, trust_remote_code=True
            ).to(self.device).eval()
        except Exception as e:
            msg = str(e).lower()
            if ("gated repo" in msg) or ("401" in msg and "huggingface" in msg) or ("access to model" in msg):
                tips = (
                    f"Model '{model_name}' is gated on Hugging Face.\n"
                    "To use it:\n"
                    "  1) Visit the model page while logged in and click “Agree/Request access”.\n"
                    "  2) Provide an access token (recommended via .env):\n"
                    "       HF_TOKEN=hf_xxx  # then `source .venv/bin/activate` and run again\n"
                    "     Or log in once: `huggingface-cli login`.\n"
                    "  3) Or choose an open fallback, e.g.: --dinov3 facebook/dinov2-base"
                )
                if os.getenv("HF_HUB_DISABLE_IMPLICIT_TOKEN") in {"1", "ON", "YES", "TRUE"}:
                    tips += "\nNote: HF_HUB_DISABLE_IMPLICIT_TOKEN=1 is set; unset it or pass the token explicitly."
                raise GatedModelAccessError(tips) from e
            raise

        # --- Try to load an image processor (preferred path) ---
        self.processor = None
        try:
            # Prefer fast processor (DINOv3 often only ships a *fast* class)
            self.processor = _load_with_token(
                AutoImageProcessor.from_pretrained,
                model_name,
                use_fast=True,
                trust_remote_code=True,
            )
            if self.verbose:
                print(f"[embedder] Using AutoImageProcessor (fast) for {model_name}")
        except Exception as e:
            # Fall back to manual preprocessing if processor isn't available
            if self.verbose:
                print(f"[embedder] AutoImageProcessor unavailable for {model_name} → "
                      f"falling back to manual preprocessing ({e})")
            self._build_manual_preproc()

        # --- Probe output dims (pooled vs tokens) ---
        with torch.inference_mode():
            dummy = Image.new("RGB", (getattr(self, "image_size", 224), getattr(self, "image_size", 224)))
            inputs = self._prep_inputs([dummy])  # dict with pixel_values
            out = self.model(**inputs)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                self.emb_dim = int(out.pooler_output.shape[-1])
                self._use_pooled = True
            else:
                toks = out.last_hidden_state
                self.emb_dim = int((toks[:, 1:, :].mean(dim=1) if toks.shape[1] > 1 else toks[:, 0, :]).shape[-1])
                self._use_pooled = False

        # --- Choose autocast dtype (bf16 if supported, else fp16) ---
        if self.use_autocast:
            try:
                bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            except Exception:
                bf16_ok = False
            self._amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
        else:
            self._amp_dtype = torch.float32

        if (self.processor is None) and self.verbose:
            # REFACTOR: Use a more readable f-string for output
            mean_str = [f"{x:.3f}" for x in self.mean.flatten()]
            std_str = [f"{x:.3f}" for x in self.std.flatten()]
            print(f"[embedder] Manual preprocess: size={self.image_size}, mean={mean_str}, std={std_str}")

    # ---------- Preprocessing paths ----------

    def _build_manual_preproc(self):
        """Derive image_size / mean / std from model config, or use sensible defaults."""
        cfg = getattr(self.model, "config", None)
        # image_size
        img_size = 224
        for key in ("image_size",):
            if cfg is not None and hasattr(cfg, key):
                v = getattr(cfg, key)
                img_size = int(v if isinstance(v, int) else _to_list(v)[0])
                break
        if cfg is not None and hasattr(cfg, "vision_config"):
            vc = cfg.vision_config
            if hasattr(vc, "image_size"):
                v = vc.image_size
                img_size = int(v if isinstance(v, int) else _to_list(v)[0])

        # normalization
        mean = std = None
        for holder in (cfg, getattr(cfg, "vision_config", None)):
            if holder is None:
                continue
            if getattr(holder, "image_mean", None) and getattr(holder, "image_std", None):
                mean = [float(x) for x in holder.image_mean]
                std = [float(x) for x in holder.image_std]
                break
        if mean is None or std is None:
            # Fall back to ImageNet (Dinov2/DINO commonly use these)
            mean = [0.485, 0.456, 0.406]
            std  = [0.229, 0.224, 0.225]

        self.image_size = int(img_size)
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array(std,  dtype=np.float32).reshape(1, 1, 3)
        self._manual = True
        if self.verbose:
            print(f"[embedder] Manual preprocess: size={self.image_size}, mean={mean}, std={std}")

    def _prep_inputs(self, pil_images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        """Return a dict suitable for self.model(**inputs)."""
        if self.processor is not None:
            # Preferred path: delegate to HF processor
            batch = self.processor(images=pil_images, return_tensors="pt")
            return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        # Manual path: resize→toTensor→normalize to model’s expected HxW
        arrs = []
        for im in pil_images:
            im = im.resize((self.image_size, self.image_size), resample=Resample.BICUBIC)
            a = np.asarray(im, dtype=np.float32) / 255.0  # HWC in [0,1]
            a = (a - self.mean) / self.std
            a = a.transpose(2, 0, 1)  # CHW
            arrs.append(a)
        x = torch.from_numpy(np.stack(arrs, axis=0)).to(self.device, non_blocking=True)  # [B,3,H,W]
        return {"pixel_values": x}

    def _maybe_autocast(self):
        if self.use_autocast:
            return torch.autocast(device_type="cuda", dtype=self._amp_dtype)
        return nullcontext()