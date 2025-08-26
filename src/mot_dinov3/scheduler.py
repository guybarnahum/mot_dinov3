# src/mot_dinov3/scheduler.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterable
import numpy as np

def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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
    overlap_thr: float = 0.2   # det-det IoU > thr => crowded
    iou_gate: float = 0.6      # IoU with prior track ≥ gate => stable candidate
    refresh_every: int = 5     # periodic refresh every N frames
    ema_alpha: float = 0.35    # EMA for ms-per-crop

class EmbeddingScheduler:
    """
    Decides which detections need embeddings now and which can reuse,
    and separates 'critical' (must compute) from 'refresh' (nice-to-have).
    Also keeps a small backlog of refresh TIDs across frames.
    """
    def __init__(self, cfg: SchedulerConfig | None = None):
        self.cfg = cfg or SchedulerConfig()
        self.last_embed_frame: dict[int, int] = {}  # tid -> frame_idx last *real* embed
        self.pending_refresh: list[int] = []        # FIFO of tids to refresh when time allows
        self.stat_real = 0
        self.stat_reuse = 0
        self.ms_per_crop_ema: float = 5.0           # rough starting guess (GPU); updated per batch

    def reset_stats(self):
        self.stat_real = 0
        self.stat_reuse = 0

    def _push_pending(self, tids: Iterable[int]):
        seen = set(self.pending_refresh)
        for tid in tids:
            if tid is None:
                continue
            if tid not in seen:
                self.pending_refresh.append(int(tid))
                seen.add(int(tid))

    def _pop_pending(self, k: int) -> list[int]:
        if k <= 0:
            return []
        k = min(k, len(self.pending_refresh))
        out = self.pending_refresh[:k]
        self.pending_refresh = self.pending_refresh[k:]
        return out

    def _remove_pending(self, tids: Iterable[int]):
        if not tids:
            return
        s = set(int(t) for t in tids)
        self.pending_refresh = [t for t in self.pending_refresh if t not in s]

    def plan(self, tracker, boxes: np.ndarray, frame_idx: int) -> tuple[
        np.ndarray,           # need_mask: (N,) True => compute embedding
        list[Optional[int]],  # reuse_tid: per det, tid to reuse from (if reusing)
        list,                 # active_snapshot: ACTIVE tracks list
        np.ndarray,           # due_refresh_mask: (N,) True => this need is "refresh only"
        list[Optional[int]],  # match_tid: per det, best-matching ACTIVE tid (if any)
    ]:
        N = len(boxes)
        need_mask = np.ones(N, dtype=bool)
        reuse_tid: list[Optional[int]] = [None] * N
        due_refresh = np.zeros(N, dtype=bool)
        match_tid: list[Optional[int]] = [None] * N

        active = [t for t in getattr(tracker, "tracks", []) if getattr(t, "state", "active") == "active"]
        if N == 0:
            return need_mask, reuse_tid, active, due_refresh, match_tid
        if len(active) == 0:
            # first frames: everything is critical (no reuse, no refresh)
            return need_mask, reuse_tid, active, due_refresh, match_tid

        # crowded detections
        crowded = np.zeros(N, dtype=bool)
        if N > 1:
            I_det = _iou_xyxy(boxes, boxes)
            np.fill_diagonal(I_det, 0.0)
            crowded = (I_det > self.cfg.overlap_thr).any(axis=1)

        # continuity to ACTIVE tracks
        A = np.stack([t.box for t in active], axis=0).astype(np.float32)
        I = _iou_xyxy(A, boxes)          # (T,N)
        best_idx = I.argmax(axis=0)      # index into active
        best_iou = I.max(axis=0)

        for j in range(N):
            # no good match → critical (e.g., new object)
            if best_iou[j] < self.cfg.iou_gate:
                continue
            # crowded → critical (needs appearance now)
            if crowded[j]:
                continue
            # at this point: stable candidate
            tid = int(active[best_idx[j]].tid)
            match_tid[j] = tid
            last = self.last_embed_frame.get(tid, -10**9)
            if (frame_idx - last) >= self.cfg.refresh_every:
                # refresh-only need (optional; can be deferred under budget)
                due_refresh[j] = True
            else:
                # safe to reuse
                need_mask[j] = False
                reuse_tid[j] = tid

        # stash refresh TIDs as backlog candidates (we may do only some this frame)
        due_tids = [match_tid[j] for j in range(N) if due_refresh[j] and match_tid[j] is not None]
        self._push_pending(due_tids)
        return need_mask, reuse_tid, active, due_refresh, match_tid

    def mark_refreshed(self, active_snapshot: list, boxes: np.ndarray, idx_need: np.ndarray, frame_idx: int) -> set[int]:
        """
        After computing real embeddings for detections idx_need (this frame),
        assign those refreshes to the most plausible ACTIVE track by IoU.
        Returns set of refreshed tids.
        """
        refreshed: set[int] = set()
        if len(idx_need) == 0 or len(active_snapshot) == 0:
            return refreshed
        A = np.stack([t.box for t in active_snapshot], axis=0).astype(np.float32)
        # IoU(A, boxes)
        x11, y11, x12, y12 = A[:,0][:,None], A[:,1][:,None], A[:,2][:,None], A[:,3][:,None]
        B = boxes.astype(np.float32)
        x21, y21, x22, y22 = B[:,0][None,:], B[:,1][None,:], B[:,2][None,:], B[:,3][None,:]
        inter_w = np.maximum(0, np.minimum(x12, x22) - np.maximum(x11, x21))
        inter_h = np.maximum(0, np.minimum(y12, y22) - np.maximum(y11, y21))
        inter = inter_w * inter_h
        area_a = (x12 - x11) * (y12 - y11)
        area_b = (x22 - x21) * (y22 - y21)
        I = inter / (area_a + area_b - inter + 1e-6)  # (T,N)

        best_idx = I.argmax(axis=0)
        for j in idx_need:
            tid = int(active_snapshot[best_idx[j]].tid)
            self.last_embed_frame[tid] = frame_idx
            refreshed.add(tid)
        # remove these tids from backlog if present
        self._remove_pending(refreshed)
        return refreshed

    def update_ema_ms_per_crop(self, batch_ms: float, count: int):
        if count <= 0:
            return
        per = batch_ms / max(1, count)
        a = float(self.cfg.ema_alpha)
        self.ms_per_crop_ema = (1 - a) * self.ms_per_crop_ema + a * per

    def summary(self, frames: int) -> str:
        if frames <= 0:
            return "Embeddings computed: 0 total (0.00/frame), reused: 0"
        per_frame = self.stat_real / frames
        return f"Embeddings computed: {self.stat_real} total ({per_frame:.2f}/frame), reused: {self.stat_reuse}"
