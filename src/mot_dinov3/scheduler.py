# src/mot_dinov3/scheduler.py (Refactored)
from __future__ import annotations
from collections import deque # REFACTOR: Use deque for an efficient queue
from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional, Tuple, Iterable, Set

# Import shared helpers from the new utils module
from . import utils

import numpy as np
import torch

@dataclass
class SchedulerConfig:
    overlap_thr: float = 0.2
    iou_gate: float = 0.6
    refresh_every: int = 5
    ema_alpha: float = 0.35

class EmbeddingScheduler:
    """
    Decides which detections need embeddings and which can reuse cached ones,
    separating 'critical' from budget-limited 'refresh' computations.
    """
    def __init__(self, cfg: SchedulerConfig | None = None):
        self.cfg = cfg or SchedulerConfig()
        self.last_embed_frame: dict[int, int] = {}
        # REFACTOR: Use deque for an O(1) FIFO queue
        self.pending_refresh: deque[int] = deque()
        self.stat_real = 0
        self.stat_reuse = 0
        self.ms_per_crop_ema: float = 5.0

    def reset_stats(self):
        self.stat_real = 0
        self.stat_reuse = 0

    def _push_pending(self, tids: Iterable[int]):
        # REFACTOR: Use a set for faster `in` check
        seen = set(self.pending_refresh)
        for tid in tids:
            if tid is not None and tid not in seen:
                self.pending_refresh.append(int(tid))

    def _pop_pending(self, k: int) -> list[int]:
        if k <= 0: return []
        popped = []
        for _ in range(min(k, len(self.pending_refresh))):
            popped.append(self.pending_refresh.popleft())
        return popped

    def _remove_pending(self, tids: Iterable[int]):
        if not tids: return
        # REFACTOR: Rebuilding is efficient for deque when removing multiple items
        s = set(int(t) for t in tids)
        self.pending_refresh = deque(t for t in self.pending_refresh if t not in s)

    @staticmethod
    def _embedder_device_type(embedder) -> str:
        """Heuristically determines the device ('cuda' or 'cpu') of the embedder."""
        dev = getattr(embedder, "device", None)
        # Handle cases where the actual model is wrapped
        if dev is None and hasattr(embedder, "_e"):
            dev = getattr(embedder._e, "device", None)
        if isinstance(dev, torch.device):
            return dev.type
        return str(dev or "cpu")
        
    def plan(self, tracker, boxes: np.ndarray, frame_idx: int):
        N = len(boxes)
        active = [t for t in getattr(tracker, "tracks", []) if getattr(t, "state", "active") == "active"]

        if N == 0 or len(active) == 0:
            # No tracks to reuse from or no detections to process
            iou_active_det = np.zeros((len(active), N), dtype=np.float32)
            return (
                np.ones(N, dtype=bool), [None] * N, active, 
                np.zeros(N, dtype=bool), [None] * N, iou_active_det
            )

        # REFACTOR: Vectorized planning logic
        # 1. Identify crowded detections
        iou_det_det = utils.iou_matrix(boxes, boxes)
        np.fill_diagonal(iou_det_det, 0.0)
        is_crowded = (iou_det_det > self.cfg.overlap_thr).any(axis=1)

        # 2. Find best-matching active track for each detection
        active_boxes = np.stack([t.box for t in active]).astype(np.float32)
        iou_active_det = utils.iou_matrix(active_boxes, boxes) # (T, N)
        
        best_track_idx = iou_active_det.argmax(axis=0) # (N,)
        best_iou = iou_active_det.max(axis=0)          # (N,)
        
        match_tid = [int(active[best_track_idx[j]].tid) for j in range(N)]

        # 3. Determine who can reuse embeddings
        can_reuse = (best_iou >= self.cfg.iou_gate) & ~is_crowded
        
        # 4. From reusable candidates, see who is due for a refresh
        is_due_refresh = np.zeros(N, dtype=bool)
        if np.any(can_reuse):
            reusable_indices = np.where(can_reuse)[0]
            for j in reusable_indices:
                tid = match_tid[j]
                last = self.last_embed_frame.get(tid, -1)
                if (frame_idx - last) >= self.cfg.refresh_every:
                    is_due_refresh[j] = True
        
        # 5. Finalize masks and lists
        # Need embedding if it's not a safe reuse (i.e., critical or due for refresh)
        need_mask = ~can_reuse | is_due_refresh
        
        # Can only reuse if it's a stable candidate NOT due for refresh
        can_truly_reuse = can_reuse & ~is_due_refresh
        reuse_tid = [match_tid[j] if can_truly_reuse[j] else None for j in range(N)]
        
        # TIDs due for refresh are candidates for the backlog
        due_tids = [match_tid[j] for j in np.where(is_due_refresh)[0]]
        self._push_pending(due_tids)

        # Make match_tid None where IoU is too low
        final_match_tid = [match_tid[j] if best_iou[j] >= self.cfg.iou_gate else None for j in range(N)]

        return need_mask, reuse_tid, active, is_due_refresh, final_match_tid, iou_active_det

    # REFACTOR: Pass IoU matrix to avoid re-computation
    def mark_refreshed(self, active_snapshot: list, boxes: np.ndarray, idx_need: np.ndarray, frame_idx: int, iou_matrix_val: np.ndarray) -> set[int]:
        refreshed: set[int] = set()
        if len(idx_need) == 0 or len(active_snapshot) == 0:
            return refreshed
        
        best_track_indices = iou_matrix_val[:, idx_need].argmax(axis=0)
        
        for i, det_idx in enumerate(idx_need):
            track_idx = best_track_indices[i]
            tid = int(active_snapshot[track_idx].tid)
            self.last_embed_frame[tid] = frame_idx
            refreshed.add(tid)
            
        self._remove_pending(refreshed)
        return refreshed

    def update_ema_ms_per_crop(self, batch_ms: float, count: int):
        if count <= 0: return
        per = batch_ms / count
        a = self.cfg.ema_alpha
        self.ms_per_crop_ema = (1 - a) * self.ms_per_crop_ema + a * per

    @torch.inference_mode()
    def run(self,
            frame: np.ndarray,
            boxes: np.ndarray,
            embedder,
            tracker,
            frame_idx: int,
            budget_ms: float = 0.0) -> Tuple[np.ndarray, float, Set[int]]:
        on_cuda = self._embedder_device_type(embedder) == "cuda"
        if on_cuda: torch.cuda.synchronize()
        t0 = perf_counter()

        N = len(boxes)
        D = int(getattr(embedder, "dim", 768))
        if N == 0:
            return np.zeros((0, D), dtype=np.float32), 0.0, set()

        # REFACTOR: plan now returns the IoU matrix to be reused
        need_mask, reuse_tids, active, due_mask, match_tid, iou_active_det = self.plan(tracker, boxes, frame_idx)

        embs = np.zeros((N, D), dtype=np.float32)
        refreshed_tids: set[int] = set()

        # Handle reuse
        if len(active) > 0:
            tid2emb = {int(t.tid): t.emb for t in active}
            for j, tid in enumerate(reuse_tids):
                if tid is not None and tid in tid2emb:
                    embs[j] = tid2emb[tid]
                    self.stat_reuse += 1
        
        # --- Compute Critical Embeddings (not budgeted) ---
        idx_crit = np.where(need_mask & ~due_mask)[0]
        if len(idx_crit) > 0:
            embs_crit, _ = self._compute_embeddings(frame, boxes, idx_crit, embedder, on_cuda)
            embs[idx_crit] = embs_crit
            refreshed_tids |= self.mark_refreshed(active, boxes, idx_crit, frame_idx, iou_active_det)

        # --- Compute Refresh Embeddings (budgeted) ---
        if on_cuda: torch.cuda.synchronize()
        used_ms = (perf_counter() - t0) * 1000.0
        remaining_ms = max(0.0, budget_ms - used_ms) if budget_ms > 0 else float('inf')

        if remaining_ms > 0 and len(active) > 0:
            idx_refresh = self._get_budgeted_refresh_indices(
                active, boxes, match_tid, due_mask, idx_crit, frame_idx,
                iou_active_det, t0, budget_ms
            )
            if len(idx_refresh) > 0:
                embs_ref, _ = self._compute_embeddings(frame, boxes, idx_refresh, embedder, on_cuda)
                embs[idx_refresh] = embs_ref
                refreshed_this_batch = self.mark_refreshed(active, boxes, idx_refresh, frame_idx, iou_active_det)
                refreshed_tids |= refreshed_this_batch
        
        if on_cuda: torch.cuda.synchronize()
        return embs, perf_counter() - t0, refreshed_tids

    def _compute_embeddings(self, frame, boxes, indices, embedder, on_cuda):
        """Helper to run embedder and update EMA time."""
        t1 = perf_counter()
        embeddings = embedder.embed_crops(frame, boxes[indices])
        if on_cuda: torch.cuda.synchronize()
        batch_ms = (perf_counter() - t1) * 1000.0
        
        self.update_ema_ms_per_crop(batch_ms, len(indices))
        self.stat_real += len(indices)
        return embeddings, batch_ms

    def _get_budgeted_refresh_indices(self, active, boxes, match_tid, due_mask, idx_crit, frame_idx, iou_active_det, t0, budget_ms):
        """Determines which tracks to refresh within the time budget."""
        # Prioritize candidates from the backlog, then this frame's due items
        due_tids_this = {match_tid[j] for j in np.where(due_mask)[0] if match_tid[j] is not None}
        candidates = list(self.pending_refresh) + [t for t in sorted(due_tids_this, reverse=True) if t not in self.pending_refresh]
        
        # Map each track to its best-matching detection this frame
        tid_to_det_map = self._map_tids_to_best_det(active, iou_active_det)

        idx_refresh = []
        processed_tids = set()

        for tid in candidates:
            if budget_ms > 0:
                if (perf_counter() - t0) * 1000.0 + self.ms_per_crop_ema > budget_ms:
                    break # Predicted to go over budget
            
            det_idx = tid_to_det_map.get(tid)
            if det_idx is not None and det_idx not in idx_crit and det_idx not in idx_refresh:
                idx_refresh.append(det_idx)
                processed_tids.add(tid)

        self._remove_pending(processed_tids)
        return np.array(idx_refresh, dtype=int)

    # REFACTOR: Simplified, vectorized, and uses pre-computed IoU
    def _map_tids_to_best_det(self, active_snapshot, iou_active_det: np.ndarray) -> dict[int, int]:
        """For each active track, find the detection index with the highest IoU."""
        if not active_snapshot or iou_active_det.shape[1] == 0:
            return {}
        
        best_det_indices = iou_active_det.argmax(axis=1) # (T,)
        return {
            int(track.tid): best_det_indices[i]
            for i, track in enumerate(active_snapshot)
        }

    def summary(self, frames: int) -> str:
        if frames <= 0: return "Embeddings: 0 computed, 0 reused"
        per_frame = self.stat_real / frames
        return f"Embeddings computed: {self.stat_real} total ({per_frame:.2f}/frame), reused: {self.stat_reuse}"