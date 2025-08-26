# src/mot_dinov3/scheduler.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix a:(M,4) vs b:(N,4) in xyxy."""
    M, N = len(a), len(b)
    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=np.float32)
    a = a.astype(np.float32); b = b.astype(np.float32)
    x11, y11, x12, y12 = a[:,0][:,None], a[:,1][:,None], a[:,2][:,None], a[:,3][:,None]
    x21, y21, x22, y22 = b[:,0][None,:], b[:,1][None,:], b[:,2][None,:], b[:,3][None,:]
    inter_w = np.maximum(0, np.minimum(x12, x22) - np.maximum(x11, x21))
    inter_h = np.maximum(0, np.minimum(y12, y22) - np.maximum(y11, y21))
    inter = inter_w * inter_h
    area_a = (x12 - x11) * (y12 - y11)
    area_b = (x22 - x21) * (y22 - y21)
    union = area_a + area_b - inter + 1e-6
    return (inter / union).astype(np.float32)

@dataclass
class SchedulerConfig:
    overlap_thr: float = 0.2   # det-det IoU > thr => crowded → compute real embedding
    iou_gate: float = 0.6      # IoU with prior track ≥ gate => candidate for reuse
    refresh_every: int = 5     # refresh real embedding every N frames per track

class EmbeddingScheduler:
    """
    Decides which detections need real embeddings *now* and which can reuse a track's cached
    embedding (EMA). Maintains last refresh frame per track id for periodic refresh.
    """
    def __init__(self, cfg: SchedulerConfig | None = None):
        self.cfg = cfg or SchedulerConfig()
        self.last_embed_frame: dict[int, int] = {}  # tid -> frame_idx
        self.stat_real = 0
        self.stat_reuse = 0

    def reset_stats(self):
        self.stat_real = 0
        self.stat_reuse = 0

    def plan(self, tracker, boxes: np.ndarray, frame_idx: int) -> tuple[np.ndarray, list[Optional[int]], list]:
        """
        Returns:
          - need_mask: (N,) bool — True => compute real embedding for detection j
          - reuse_tid: list[Optional[int]] length N — track id to reuse from, or None
          - active_snapshot: list of ACTIVE tracks used for the decision
        """
        N = len(boxes)
        need_mask = np.ones(N, dtype=bool)
        reuse_tid: list[Optional[int]] = [None] * N

        # Snapshot ACTIVE tracks once per frame
        active = [t for t in getattr(tracker, "tracks", []) if getattr(t, "state", "active") == "active"]
        if N == 0 or len(active) == 0:
            return need_mask, reuse_tid, active

        # Mark crowded detections (det-det overlaps)
        crowded = np.zeros(N, dtype=bool)
        if N > 1:
            I_det = _iou_xyxy(boxes, boxes)
            np.fill_diagonal(I_det, 0.0)
            crowded = (I_det > self.cfg.overlap_thr).any(axis=1)

        # Best IoU continuity with ACTIVE tracks
        A = np.stack([t.box for t in active], axis=0).astype(np.float32)
        I = _iou_xyxy(A, boxes)          # (T,N)
        best_idx = I.argmax(axis=0)      # index into active
        best_iou = I.max(axis=0)

        for j in range(N):
            if best_iou[j] < self.cfg.iou_gate:
                continue  # weak continuity → compute
            if crowded[j]:
                continue  # crowded → compute
            tid = int(active[best_idx[j]].tid)
            last = self.last_embed_frame.get(tid, -10**9)
            if (frame_idx - last) >= self.cfg.refresh_every:
                continue  # due for refresh → compute
            # safe to reuse
            need_mask[j] = False
            reuse_tid[j] = tid
        return need_mask, reuse_tid, active

    def mark_refreshed(self, active_snapshot: list, boxes: np.ndarray, idx_need: np.ndarray, frame_idx: int):
        """
        After computing real embeddings for detections idx_need (on the same frame),
        assign those refreshes to the most plausible ACTIVE track by IoU to its *current* box.
        """
        if len(idx_need) == 0 or len(active_snapshot) == 0:
            return
        A = np.stack([t.box for t in active_snapshot], axis=0).astype(np.float32)
        I = _iou_xyxy(A, boxes)  # (T,N)
        best_idx = I.argmax(axis=0)
        for j in idx_need:
            tid = int(active_snapshot[best_idx[j]].tid)
            self.last_embed_frame[tid] = frame_idx

    def summary(self, frames: int) -> str:
        if frames <= 0:
            return "Embeddings computed: 0 total (0.00/frame), reused: 0"
        per_frame = self.stat_real / frames
        return f"Embeddings computed: {self.stat_real} total ({per_frame:.2f}/frame), reused: {self.stat_reuse}"
