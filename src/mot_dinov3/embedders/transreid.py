# src/mot_dinov3/embedders/transreid.py
from __future__ import annotations
import numpy as np, torch
from typing import List, Dict, Optional, Tuple
try:
    from PIL import Image
    Resample = Image.Resampling
except Exception:
    Resample = Image  # Pillow < 10
# then use Resample.BICUBIC

from transformers import AutoModel, AutoImageProcessor
from .base import BaseEmbedder, pick_amp_dtype

class TransReIDEmbedder:  # implements BaseEmbedder
    def __init__(self, model_name: str="your-hf-user/transreid-vit-s",
                 device: Optional[str]=None, use_autocast: bool=True,
                 image_size: Tuple[int,int]=(256,128), verbose: bool=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name, self.verbose = model_name, verbose
        self.use_autocast = use_autocast and (self.device == "cuda") and torch.cuda.is_available()
        self.image_size = image_size  # (H,W)

        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device).eval()
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        except Exception as e:
            if verbose: print(f"[transreid] no processor: {e} → manual")
            self.processor = None
            self.mean = np.array([0.485,0.456,0.406],dtype=np.float32).reshape(1,1,3)
            self.std  = np.array([0.229,0.224,0.225],dtype=np.float32).reshape(1,1,3)

        self._amp_dtype = pick_amp_dtype() if self.use_autocast else torch.float32

        with torch.inference_mode():
            H,W = self.image_size
            dummy = Image.new("RGB", (W, H))
            out = self.model(**self._prep_inputs([dummy]))
            feat = self._extract_feat(out)
            self.emb_dim = int(feat.shape[-1])

    def _prep_inputs(self, pil_images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        if self.processor is not None:
            batch = self.processor(images=pil_images, return_tensors="pt")
            return {k:v.to(self.device, non_blocking=True) for k,v in batch.items()}
        H,W = self.image_size; arrs=[]
        for im in pil_images:
            im = im.resize((W, H), Resample.BICUBIC)
            a = np.asarray(im, dtype=np.float32)/255.0
            a = (a - self.mean)/self.std
            arrs.append(a.transpose(2,0,1))
        x = torch.from_numpy(np.stack(arrs)).to(self.device, non_blocking=True)
        return {"pixel_values": x}

    def _extract_feat(self, out) -> torch.Tensor:
        for k in ("feat","features","global_feat","pooler_output"):
            if isinstance(out, dict) and k in out and out[k] is not None: return out[k]
            if hasattr(out, k) and getattr(out, k) is not None: return getattr(out, k)
        toks = getattr(out, "last_hidden_state", None)
        if toks is None: raise RuntimeError("TransReID: no recognizable feature in forward output.")
        return toks[:,1:,:].mean(1) if toks.shape[1]>1 else toks[:,0,:]

