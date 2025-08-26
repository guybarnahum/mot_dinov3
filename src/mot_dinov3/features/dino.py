# src/mot_dinov3/features/dino.py
from __future__ import annotations
from typing import Optional, Any, List
import numpy as np
import cv2
import torch
from PIL import Image

from ..embedder import DinoV3Embedder


class DINOExtractor:
    """
    Thin adapter around DinoV3Embedder that can either:
      - use the embedder's AutoImageProcessor/manual preproc (default), or
      - use a fast manual preprocessor (OpenCV + NumPy) with pinned-memory copies
        and channels_last, which can shave a few ms per frame.

    Args:
      model_id: HF repo/model id
      device:  "cuda" or "cpu"
      autocast: enable AMP
      pad:     crop padding ratio (default 0.12)
      square:  square crops (default True)
      prefer_manual_preproc: if True, bypass AutoImageProcessor and use manual path
      amp_dtype: "fp16"/"bf16"/torch dtype or None → dtype for autocast

    Notes:
      - On CUDA, we put the model in channels_last, and we produce channels_last inputs.
      - Manual preproc uses mean/std from the embedder config (or ImageNet fallback).
    """

    def __init__(self, model_id: str, device: str, autocast: bool,
                 pad: float = 0.12, square: bool = True,
                 prefer_manual_preproc: bool = False,
                 amp_dtype: Optional[object] = None):

        # Load model/processor via embedder (handles HF auth & gating)
        self._e = DinoV3Embedder(model_id, device=device, use_autocast=False)
        self.name = f"dino:{model_id}"
        self.dim = int(getattr(self._e, "emb_dim", 768))
        self.device = device

        # crop defaults
        self._pad_default = float(pad)
        self._square_default = bool(square)

        # set autocast policy
        self._use_autocast = bool(autocast) and (device == "cuda")
        if self._use_autocast:
            if isinstance(amp_dtype, str):
                amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp_dtype.lower())
            if amp_dtype is None:
                try:
                    major, _ = torch.cuda.get_device_capability()
                except Exception:
                    major = 0
                amp_dtype = torch.bfloat16 if major >= 8 else torch.float16  # Ampere+: bf16, T4/Turing: fp16
            self._amp_dtype = amp_dtype
        else:
            self._amp_dtype = None

        # --- channels_last on model (safe even if small/no gain for ViTs)
        if self.device == "cuda":
            try:
                self._e.model = self._e.model.to(memory_format=torch.channels_last)
            except Exception:
                pass

        # --- preproc selection
        self._prefer_manual = bool(prefer_manual_preproc)
        # Resolve size/mean/std for manual path (read from processor/config if available)
        self._resolve_preproc_defaults()

    # -------------------- preproc helpers --------------------

    def _resolve_preproc_defaults(self):
        """Derive image_size / mean / std from embedder/processor or fallback to ImageNet."""
        # try embedder manual values first (present if embedder built manual preproc)
        img_size = getattr(self._e, "image_size", None)
        mean_arr = getattr(self._e, "mean", None)  # shape (1,1,3) if present
        std_arr  = getattr(self._e, "std", None)

        # otherwise use processor hints
        if img_size is None:
            proc = getattr(self._e, "processor", None)
            if proc is not None:
                # Typical keys: size, crop_size; many processors expose .image_mean/.image_std
                size = getattr(proc, "size", None) or getattr(proc, "crop_size", None)
                if isinstance(size, dict):
                    # choose something sensible (e.g., 'shortest_edge' or 'height' or 'width')
                    img_size = int(size.get("shortest_edge") or size.get("height") or size.get("width") or 224)
                elif isinstance(size, int):
                    img_size = size
                else:
                    img_size = 224
                if hasattr(proc, "image_mean"):
                    mean_arr = np.array(proc.image_mean, dtype=np.float32).reshape(1,1,3)
                if hasattr(proc, "image_std"):
                    std_arr = np.array(proc.image_std, dtype=np.float32).reshape(1,1,3)

        # final fallbacks
        if img_size is None:
            img_size = 224
        if mean_arr is None or std_arr is None:
            mean_arr = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1,1,3)
            std_arr  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1,1,3)

        self._img_size = int(img_size)
        self._mean = mean_arr.astype(np.float32)   # (1,1,3)
        self._std  = std_arr.astype(np.float32)    # (1,1,3)

    @staticmethod
    def _expand_and_clip(x1, y1, x2, y2, w, h, pad_ratio: float, square: bool):
        bw, bh = x2 - x1, y2 - y1
        if square:
            s = max(bw, bh)
            cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
            x1, x2 = int(round(cx - s / 2.0)), int(round(cx + s / 2.0))
            y1, y2 = int(round(cy - s / 2.0)), int(round(cy + s / 2.0))
            bw, bh = x2 - x1, y2 - y1
        if pad_ratio > 0:
            px, py = int(round(bw * pad_ratio)), int(round(bh * pad_ratio))
            x1, y1, x2, y2 = x1 - px, y1 - py, x2 + px, y2 + py
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1: x2 = min(w - 1, x1 + 2)
        if y2 <= y1: y2 = min(h - 1, y1 + 2)
        return x1, y1, x2, y2

    def _make_pil_batch(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray,
                        pad_ratio: float, square: bool) -> List[Image.Image]:
        h, w = frame_bgr.shape[:2]
        pil_batch: List[Image.Image] = []
        for (x1, y1, x2, y2) in boxes_xyxy.astype(int):
            x1, y1, x2, y2 = self._expand_and_clip(x1, y1, x2, y2, w, h, pad_ratio, square)
            crop_rgb = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            pil_batch.append(Image.fromarray(crop_rgb))
        return pil_batch

    def _prep_manual_tensors(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray,
                             pad_ratio: float, square: bool) -> dict[str, torch.Tensor]:
        """
        Fast manual preprocessor:
          - OpenCV resize to self._img_size
          - normalize with self._mean/self._std
          - produce NCHW tensor with channels_last memory format
          - use pinned memory and non_blocking transfer to GPU
        """
        h, w = frame_bgr.shape[:2]
        arrs = []
        for (x1, y1, x2, y2) in boxes_xyxy.astype(int):
            x1, y1, x2, y2 = self._expand_and_clip(x1, y1, x2, y2, w, h, pad_ratio, square)
            crop = frame_bgr[y1:y2, x1:x2]                              # BGR uint8
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)                # RGB
            crop = cv2.resize(crop, (self._img_size, self._img_size), interpolation=cv2.INTER_CUBIC)
            a = crop.astype(np.float32) / 255.0                         # HWC [0,1]
            a = (a - self._mean) / self._std                            # normalize
            arrs.append(a)
        nhwc = np.stack(arrs, axis=0)                                   # [N,H,W,3] float32
        x = torch.from_numpy(nhwc)                                      # CPU tensor shares memory with NumPy (no copy)

        # NCHW shape but channels_last memory format (works well with many CUDA kernels)
        x = x.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)

        if self.device == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)

        return {"pixel_values": x}

    # -------------------- public API --------------------

    @torch.inference_mode()
    def embed_crops(self,
                    frame_bgr: np.ndarray,
                    boxes_xyxy: np.ndarray,
                    pad_ratio: Optional[float] = None,
                    square: Optional[bool] = None,
                    **_: Any) -> np.ndarray:

        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        pad = self._pad_default if pad_ratio is None else float(pad_ratio)
        sq = self._square_default if square is None else bool(square)

        # Choose preprocessor
        if self._prefer_manual:
            inputs = self._prep_manual_tensors(frame_bgr, boxes_xyxy, pad, sq)
        else:
            # Use embedder’s processor/manual (whatever it has)
            pil_batch = self._make_pil_batch(frame_bgr, boxes_xyxy, pad, sq)
            inputs = self._e._prep_inputs(pil_batch)

        # Forward pass with AMP if configured
        if self._use_autocast and self._amp_dtype is not None and self.device == "cuda":
            with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                out = self._e.model(**inputs)
        else:
            out = self._e.model(**inputs)

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            z = out.pooler_output
        else:
            toks = out.last_hidden_state
            z = toks[:, 1:, :].mean(dim=1) if toks.shape[1] > 1 else toks[:, 0, :]

        z = torch.nn.functional.normalize(z, dim=1)
        return z.detach().cpu().float().numpy()
