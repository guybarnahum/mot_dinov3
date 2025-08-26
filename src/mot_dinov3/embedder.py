# src/mot_dinov3/embedder.py
from __future__ import annotations
import os, math
import numpy as np
import torch, cv2
from PIL import Image
from typing import List, Tuple, Optional, Any, Dict

# Apply shims early (torch compiler / NumPy, and HF token bridge)
from . import compat as _compat
_compat.apply(strict_numpy=False, quiet=True)

from transformers import AutoImageProcessor, AutoModel


class GatedModelAccessError(RuntimeError):
    """Raised when a gated HF repo is requested without granted access / auth."""


def _to_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


class DinoV3Embedder:
    """
    Robust DINO(v3/v2) embedder.

    Load order:
      1) AutoImageProcessor (trust_remote_code=True, use_fast=False)
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
        self.use_autocast = use_autocast and (self.device == "cuda")
        self.model_name = model_name
        self.verbose = verbose

        # Grab token from env (supports HF_TOKEN and legacy names via compat bridge)
        tok = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )

        def _load_with_token(factory, repo_id, **extra):
            # Newer transformers uses token=, older uses use_auth_token=
            try:
                return factory(repo_id, token=tok, **extra) if tok else factory(repo_id, **extra)
            except TypeError:
                return factory(repo_id, use_auth_token=tok, **extra) if tok else factory(repo_id, **extra)

        # --- Load model first (we can still run manual preprocessing) ---
        try:
            self.model = _load_with_token(AutoModel.from_pretrained, model_name, trust_remote_code=True).to(self.device).eval()
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
            # use_fast=False prevents the “use_fast True but no fast class” warning
            self.processor = _load_with_token(
                AutoImageProcessor.from_pretrained,
                model_name,
                use_fast=False,
                trust_remote_code=True,
            )
            if self.verbose:
                print(f"[embedder] Using AutoImageProcessor for {model_name}")
        except Exception as e:
            if self.verbose:
                print(f"[embedder] AutoImageProcessor unavailable for {model_name} → falling back to manual preprocessing ({e})")
            self._build_manual_preproc()

        # --- Probe output dims (pooled vs tokens) ---
        with torch.inference_mode():
            # Build a minimal dummy input once we know how to preprocess
            dummy = Image.new("RGB", (self.image_size, self.image_size))
            inputs = self._prep_inputs([dummy])  # dict with pixel_values
            out = self.model(**inputs)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                self.emb_dim = int(out.pooler_output.shape[-1])
                self._use_pooled = True
            else:
                toks = out.last_hidden_state
                self.emb_dim = int((toks[:, 1:, :].mean(dim=1) if toks.shape[1] > 1 else toks[:, 0, :]).shape[-1])
                self._use_pooled = False

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
            return {k: v.to(self.device) for k, v in batch.items()}

        # Manual path: resize→toTensor→normalize to model’s expected HxW
        arrs = []
        for im in pil_images:
            im = im.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
            a = np.asarray(im, dtype=np.float32) / 255.0  # HWC in [0,1]
            a = (a - self.mean) / self.std
            a = a.transpose(2, 0, 1)  # CHW
            arrs.append(a)
        x = torch.from_numpy(np.stack(arrs, axis=0)).to(self.device)  # [B,3,H,W]
        return {"pixel_values": x}

    # ---------- Public API ----------

    @torch.inference_mode()
    def embed_crops(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """
        frame_bgr: HxWx3 uint8 (OpenCV BGR)
        boxes_xyxy: (N,4) float/ints in absolute pixel coords
        returns: (N, D) float32 (L2-normalized)
        """
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            # emb_dim is probed in __init__
            return np.zeros((0, getattr(self, "emb_dim", 768)), dtype=np.float32)

        h, w = frame_bgr.shape[:2]
        pil_batch: List[Image.Image] = []
        for (x1, y1, x2, y2) in boxes_xyxy.astype(int):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                x2, y2 = min(w - 1, x1 + 2), min(h - 1, y1 + 2)
            crop = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            pil_batch.append(Image.fromarray(crop))

        inputs = self._prep_inputs(pil_batch)

        if self.use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.model(**inputs)
        else:
            out = self.model(**inputs)

        if getattr(self, "_use_pooled", False) and hasattr(out, "pooler_output") and out.pooler_output is not None:
            emb = out.pooler_output
        else:
            toks = out.last_hidden_state
            emb = toks[:, 1:, :].mean(dim=1) if toks.shape[1] > 1 else toks[:, 0, :]

        emb = torch.nn.functional.normalize(emb, dim=1)
        return emb.detach().cpu().float().numpy()
