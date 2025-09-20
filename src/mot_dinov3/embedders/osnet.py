# src/mot_dinov3/embedders/osnet.py
from __future__ import annotations
import os, re, importlib
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from PIL import Image
    Resample = Image.Resampling          # Pillow >= 10
except Exception:
    from PIL import Image
    Resample = Image                     # Pillow < 10

# --- Minimal stub if tensorboard is missing (for torchreid import) ---
try:
    from torch.utils import tensorboard as _tb  # requires 'tensorboard' package
except Exception:
    import sys, types
    class _NullWriter:
        def __init__(self, *a, **k): pass
        def add_scalar(self, *a, **k): pass
        def add_histogram(self, *a, **k): pass
        def add_image(self, *a, **k): pass
        def close(self): pass
    sys.modules["torch.utils.tensorboard"] = types.SimpleNamespace(SummaryWriter=_NullWriter)

from huggingface_hub import HfApi, hf_hub_download
from .base import pick_amp_dtype, get_hf_token, parse_hf_spec

def _import_torchreid_modules():
    """
    Import only the submodules we need so we don't pull in optional dataset deps (e.g. gdown).
    """
    m_models = importlib.import_module("torchreid.reid.models")
    m_utils  = importlib.import_module("torchreid.reid.utils")
    return m_models, m_utils


class OSNetEmbedder:
    """
    OSNet feature embedder with Hugging Face weight fetching.

    Args:
      model_name: torchreid OSNet arch (e.g. 'osnet_x1_0', 'osnet_ain_x1_0', ...)
                  OR a Hugging Face spec:
                    'org/repo'
                    'org/repo@rev'
                    'org/repo#sub/dir/file.pth'
                    'org/repo#sub/dir/file.pth@rev'
                    'hf:org/repo[:sub/dir/file.pth][@rev]'
      image_size: (H, W), OSNet typically uses (256, 128)
      device: 'cuda' or 'cpu'
      use_autocast: AMP on CUDA
      weights: local path or URL to .pt/.pth (optional)
      hf_repo: Hugging Face repo id to fetch weights (e.g. 'kadirnar/osnet_x1_0_imagenet',
               or 'spaces/rachana219/MODT2')
      hf_file: filename inside the repo (e.g. 'osnet_x1_0_imagenet.pt' or
               'trackers/strongsort/deep/checkpoint/osnet_x1_0_msmt17.pth')
      revision: optional HF revision/commit to pin
      verbose: print hints
    """
    # --- replace your current __init__ signature & amp dtype setup with this ---

class OSNetEmbedder:
    def __init__(self,
                 model_name: str = "osnet_x1_0",
                 image_size: Tuple[int, int] = (256, 128),
                 device: Optional[str] = None,
                 use_autocast: bool = True,
                 weights: Optional[str] = None,
                 hf_repo: Optional[str] = None,
                 hf_file: Optional[str] = None,
                 revision: Optional[str] = None,
                 verbose: bool = True,
                 amp_dtype: Optional[object] = None,   # <-- NEW: accept amp_dtype
                 **kwargs):                              # <-- NEW: swallow any future extras

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name, self.verbose = model_name, verbose
        self.image_size = image_size
        self.use_autocast = bool(use_autocast and self.device == "cuda")

        # accept string or torch.dtype; fall back to picker
        if amp_dtype is None:
            self._amp_dtype = pick_amp_dtype() if self.use_autocast else torch.float32
        else:
            if isinstance(amp_dtype, str):
                amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp_dtype.lower(), torch.float32)
            self._amp_dtype = amp_dtype

        # If --embed-model looks like an HF spec, parse it and fill hf_repo/file/revision.
        repo_auto, file_auto, rev_auto = parse_hf_spec(self.model_name)
        if repo_auto:
            hf_repo = hf_repo or repo_auto
            hf_file = hf_file or file_auto
            revision = revision or rev_auto

            # If model_name was a repo spec (not a bare arch), set a sensible default arch,
            # then optionally infer arch from the filename.
            if ("/" in self.model_name) or self.model_name.startswith("hf:"):
                self.model_name = "osnet_x1_0"
            if hf_file:
                for arch in ("osnet_ain_x1_0", "osnet_ibn_x1_0", "osnet_x1_0", "osnet_x0_75", "osnet_x0_5", "osnet_x0_25"):
                    if arch in hf_file:
                        self.model_name = arch
                        break

        # Normalization (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # Import only what we need from torchreid to avoid optional deps
        treid_models, treid_utils = _import_torchreid_modules()

        # 1) Build backbone (ImageNet init True; we’ll load ReID weights next)
        self.model = treid_models.build_model(
            name=self.model_name, num_classes=1000, loss="softmax", pretrained=True
        )
        if hasattr(self.model, "classifier"):
            self.model.classifier = nn.Identity()
        self.model = self.model.to(self.device).eval()
        if self.device == "cuda":
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
            except Exception:
                pass

        # 2) Decide the weight source
        weight_path = None
        if weights and os.path.isfile(weights):
            weight_path = weights
        elif weights and weights.startswith(("http://", "https://")):
            # Let torchreid handle URLs (Google Drive, etc.)
            try:
                fpath = treid_utils.download_url(weights, root="~/.cache/torchreid")
                weight_path = fpath
            except Exception as e:
                if verbose:
                    print(f"[osnet] Could not fetch URL weights '{weights}': {e}")

        if weight_path is None and hf_repo:
            # If hf_file not given, try to auto-pick a .pt/.pth that mentions 'osnet'
            if hf_file is None:
                try:
                    files = HfApi().list_repo_files(repo_id=hf_repo, revision=revision)
                    cands = [f for f in files if f.lower().endswith((".pt", ".pth")) and "osnet" in f.lower()]
                    # Prefer ReID dataset–trained weights if present
                    priority = ["msmt", "market", "duke", "imagenet"]
                    cands.sort(key=lambda f: next((i for i, kw in enumerate(priority) if kw in f.lower()), 999))
                    if cands:
                        hf_file = cands[0]
                        if verbose:
                            print(f"[osnet] auto-selected HF file: {hf_file}")
                except Exception as e:
                    if verbose:
                        print(f"[osnet] Could not list files for {hf_repo}: {e}")

            if hf_file:
                try:
                    weight_path = hf_hub_download(
                        repo_id=hf_repo, filename=hf_file, revision=revision, token=get_hf_token()
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to download {hf_repo}/{hf_file} from Hugging Face: {e}")

        # 3) Load weights if found (okay if None -> ImageNet init)
        if weight_path:
            try:
                treid_utils.load_pretrained_weights(self.model, weight_path)
                if verbose:
                    print(f"[osnet] loaded weights: {weight_path}")
            except Exception as e:
                if verbose:
                    print(f"[osnet] Failed to load weights '{weight_path}': {e} (using ImageNet init)")

        # 4) Probe embedding dim
        with torch.inference_mode():
            H, W = self.image_size
            dummy = torch.zeros(1, 3, H, W, device=self.device, dtype=torch.float32)
            z = self.model(dummy)
            if isinstance(z, torch.Tensor):
                if z.ndim == 2:
                    self.emb_dim = int(z.shape[-1])
                elif z.ndim == 4:
                    self.emb_dim = int(torch.mean(z, dim=(2, 3)).shape[1])
                else:
                    # Fallback: flatten last dim
                    self.emb_dim = int(z.view(z.size(0), -1).shape[1])
            elif isinstance(z, (tuple, list)) and z and isinstance(z[0], torch.Tensor):
                zz = z[0]
                self.emb_dim = int(zz.shape[-1]) if zz.ndim == 2 else int(torch.mean(zz, dim=(2, 3)).shape[1])
            else:
                raise RuntimeError("OSNet: unexpected forward() output during probing")

    # ---------- I/O expected by GenericExtractor ----------

    def _prep_inputs(self, pil_images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        H, W = self.image_size
        arrs = []
        for im in pil_images:
            im = im.resize((W, H), resample=Resample.BICUBIC)
            a = np.asarray(im, dtype=np.float32) / 255.0
            a = (a - self.mean) / self.std
            arrs.append(a.transpose(2, 0, 1))
        x = torch.from_numpy(np.stack(arrs, axis=0)).to(dtype=torch.float32)
        if self.device == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
        # GenericExtractor calls self.model(**inputs), and torchreid models expect 'x'
        return {"x": x}

    def _extract_feat(self, out) -> torch.Tensor:
        if isinstance(out, torch.Tensor):
            if out.ndim == 2:         # [N, D]
                return out
            if out.ndim == 4:         # [N, C, H, W] → GAP
                return torch.mean(out, dim=(2, 3))
        if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
            z = out[0]
            return z if z.ndim == 2 else torch.mean(z, dim=(2, 3))
        raise RuntimeError("OSNet: unexpected forward() output")

    # Allow GenericExtractor to call self.model(**inputs) directly.
    # (No need to override __call__; GenericExtractor uses self.model)
