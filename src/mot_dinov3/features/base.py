# src/mot_dinov3/features/base.py
from __future__ import annotations
import numpy as np, cv2, torch
from typing import Optional, List, Tuple
from PIL import Image

class GenericExtractor:
    """
    Wraps any embedder with _prep_inputs() and _extract_feat(), handling crop/pad/resize/normalize once.
    """
    def __init__(self, embedder, name: str, device: str,
                 pad: float, square: bool, autocast: bool, amp_dtype: Optional[object] = None):
        self._e = embedder
        self.name = name
        self.dim = int(getattr(embedder, "emb_dim", 768))
        self.device = device
        self._pad = float(pad)
        self._square = bool(square)
        self._use_autocast = bool(autocast and device == "cuda")
        self._amp_dtype = amp_dtype or (torch.bfloat16 if torch.cuda.is_available() and
                                        getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16)

    @staticmethod
    def _expand_and_clip(x1, y1, x2, y2, w, h, pad_ratio, square):
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

        # Half-open clamping for slicing [y1:y2, x1:x2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Mark invalid if degenerate after clamp
        valid = (x2 - x1) >= 2 and (y2 - y1) >= 2
        return x1, y1, x2, y2, valid

    @torch.inference_mode()
    def embed_crops(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray,
                    pad_ratio: Optional[float]=None, square: Optional[bool]=None, **_) -> np.ndarray:
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        pad = self._pad if pad_ratio is None else float(pad_ratio)
        sq  = self._square if square is None else bool(square)

        h, w = frame_bgr.shape[:2]
        pil_batch, valid_idx = [], []
        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy.astype(int)):
            x1, y1, x2, y2, ok = self._expand_and_clip(x1, y1, x2, y2, w, h, pad, sq)
            if not ok:
                continue
            crop_rgb = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            pil_batch.append(Image.fromarray(crop_rgb))
            valid_idx.append(i)

        # If none valid, return all-zeros
        out = np.zeros((len(boxes_xyxy), self.dim), dtype=np.float32)
        if not pil_batch:
            return out

        inputs = self._e._prep_inputs(pil_batch)
        if self._use_autocast and self.device == "cuda":
            with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                raw = self._e.model(**inputs)
        else:
            raw = self._e.model(**inputs)

        z = torch.nn.functional.normalize(self._e._extract_feat(raw), dim=1)
        z = z.detach().cpu().float().numpy()

        # Scatter back into the full array; zero vectors for invalid crops
        out[np.array(valid_idx, dtype=int)] = z
        return out