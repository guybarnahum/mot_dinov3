# src/mot_dinov3/scheduler.py (Refactored with Optimized Visualization Logic)
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np
import torch

# Assuming utils.iou_matrix exists in the same package
from . import utils

@dataclass
class SchedulerConfig:
    """Configuration for the EmbeddingScheduler."""
    overlap_thr: float = 0.2
    iou_gate: float = 0.6
    refresh_every: int = 5
    ema_alpha: float = 0.35
    force_compute_all: bool = False

class EmbeddingScheduler:
    """
    Decides which detections need embeddings and which can reuse cached ones,
    separating 'critical' from budget-limited 'refresh' computations.
    """
    def __init__(self, cfg: SchedulerConfig | None = None):
        self.cfg = cfg or SchedulerConfig()
        self.last_embed_frame: Dict[int, int] = {}
        self.pending_refresh: deque[int] = deque()
        self.stat_real = 0
        self.stat_reuse = 0
        self.ms_per_crop_ema: float = 5.0

    def reset_stats(self):
        """Resets the computation/reuse statistics."""
        self.stat_real = 0
        self.stat_reuse = 0

    def _push_pending(self, tids: Iterable[int]):
        """Adds track IDs due for a refresh to the backlog queue."""
        seen = set(self.pending_refresh)
        for tid in tids:
            if tid is not None and tid not in seen:
                self.pending_refresh.append(int(tid))

    def _remove_pending(self, tids: Iterable[int]):
        """Removes track IDs from the backlog queue after they've been processed."""
        if not tids:
            return
        s = set(int(t) for t in tids)
        self.pending_refresh = deque(t for t in self.pending_refresh if t not in s)

    @staticmethod
    def _embedder_device_type(embedder) -> str:
        """Heuristically determines the device ('cuda' or 'cpu') of the embedder."""
        dev = getattr(embedder, "device", None)
        if dev is None and hasattr(embedder, "_e"):
            dev = getattr(embedder._e, "device", None)
        if isinstance(dev, torch.device):
            return dev.type
        return str(dev or "cpu")

    def plan(self, tracker, boxes: np.ndarray, frame_idx: int):
        """
        Analyzes detections and active tracks to create a computation plan.
        """
        N = len(boxes)
        active = [t for t in getattr(tracker, "tracks", []) if getattr(t, "state", "active") == "active"]

        if self.cfg.force_compute_all:
            # If true, bypass all logic and mark every detection for computation.
            iou_active_det = np.zeros((len(active), N), dtype=np.float32)
            return (
                np.ones(N, dtype=bool), [None] * N, active,
                np.zeros(N, dtype=bool), np.zeros(N, dtype=bool), [None] * N, iou_active_det
            )

        if N == 0 or not active:
            iou_active_det = np.zeros((len(active), N), dtype=np.float32)
            return (
                np.ones(N, dtype=bool), [None] * N, active,
                np.zeros(N, dtype=bool), np.zeros(N, dtype=bool), [None] * N, iou_active_det
            )

        iou_det_det = utils.iou_matrix(boxes, boxes)
        np.fill_diagonal(iou_det_det, 0.0)
        is_crowded = (iou_det_det > self.cfg.overlap_thr).any(axis=1)

        active_boxes = np.stack([t.box for t in active]).astype(np.float32)
        iou_active_det = utils.iou_matrix(active_boxes, boxes)
        best_track_idx = iou_active_det.argmax(axis=0)
        best_iou = iou_active_det.max(axis=0)
        match_tid = [int(active[best_track_idx[j]].tid) for j in range(N)]

        can_reuse = (best_iou >= self.cfg.iou_gate) & ~is_crowded

        is_due_refresh = np.zeros(N, dtype=bool)
        if np.any(can_reuse):
            reusable_indices = np.where(can_reuse)[0]
            for j in reusable_indices:
                tid = match_tid[j]
                last = self.last_embed_frame.get(tid, -1)
                if (frame_idx - last) >= self.cfg.refresh_every:
                    is_due_refresh[j] = True

        need_mask = ~can_reuse | is_due_refresh
        can_truly_reuse = can_reuse & ~is_due_refresh
        reuse_tid = [match_tid[j] if can_truly_reuse[j] else None for j in range(N)]
        
        due_tids = [match_tid[j] for j in np.where(is_due_refresh)[0]]
        self._push_pending(due_tids)

        final_match_tid = [match_tid[j] if best_iou[j] >= self.cfg.iou_gate else None for j in range(N)]

        return need_mask, reuse_tid, active, is_due_refresh, is_crowded, final_match_tid, iou_active_det

    def _mark_computed(self, active_snapshot: list, idx_computed: np.ndarray, frame_idx: int,
                       iou_matrix_val: np.ndarray, viz_info: dict | None, reason_map: dict) -> set[int]:
        """Updates internal state for tracks that received a new embedding."""
        computed_tids: set[int] = set()
        if len(idx_computed) == 0 or not active_snapshot:
            return computed_tids

        best_track_indices = iou_matrix_val[:, idx_computed].argmax(axis=0)

        for i, det_idx in enumerate(idx_computed):
            track_idx = best_track_indices[i]
            if iou_matrix_val[track_idx, det_idx] >= self.cfg.iou_gate:
                tid = int(active_snapshot[track_idx].tid)
                self.last_embed_frame[tid] = frame_idx
                computed_tids.add(tid)
                
                # **VIZ LOGIC**: Update the viz_info dict, only if it's provided.
                if viz_info is not None:
                    if (reason := reason_map.get(det_idx)):
                        viz_info[tid] = reason

        self._remove_pending(computed_tids)
        return computed_tids

    def update_ema_ms_per_crop(self, batch_ms: float, count: int):
        """Updates the exponential moving average of embedding latency per crop."""
        if count <= 0: return
        per = batch_ms / count
        a = self.cfg.ema_alpha
        self.ms_per_crop_ema = (1 - a) * self.ms_per_crop_ema + a * per

    @torch.inference_mode()
    def run(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        embedder,
        tracker,
        frame_idx: int,
        budget_ms: float = 0.0,
        viz_info: dict[int, str] | None = None
    ) -> Tuple[np.ndarray, float, Set[int]]:
        on_cuda = self._embedder_device_type(embedder) == "cuda"
        if on_cuda: torch.cuda.synchronize()
        t0 = perf_counter()

        N = len(boxes)
        D = int(getattr(embedder, "dim", 768))
        if N == 0:
            return np.zeros((0, D), dtype=np.float32), 0.0, set()

        plan_outputs = self.plan(tracker, boxes, frame_idx)
        need_mask, reuse_tids, active, due_mask, crowd_mask, match_tid, iou_active_det = plan_outputs

        embs = np.zeros((N, D), dtype=np.float32)
        computed_tids: set[int] = set()
        reason_map: Dict[int, str] = {} # Will stay empty if viz_info is None
        reason_map: Dict[int, str] = {}

        # Handle reuse
        if active:
            tid2emb = {int(t.tid): t.emb for t in active}
            for j, tid in enumerate(reuse_tids):
                if tid is not None and tid in tid2emb:
                    embs[j] = tid2emb[tid]
                    self.stat_reuse += 1

        # Compute Critical Embeddings (not budgeted)
        idx_crit = np.where(need_mask & ~due_mask)[0]
        if len(idx_crit) > 0:
            if viz_info is not None:
                for j in idx_crit:
                    reason_map[j] = 'C-crowd' if crowd_mask[j] else 'C-new'
            
            embs_crit, _ = self._compute_embeddings(frame, boxes, idx_crit, embedder, on_cuda)
            embs[idx_crit] = embs_crit
            computed_tids |= self._mark_computed(active, idx_crit, frame_idx, iou_active_det, viz_info, reason_map)

        # Compute Refresh Embeddings (budgeted)
        if on_cuda: torch.cuda.synchronize()
        used_ms = (perf_counter() - t0) * 1000.0
        remaining_ms = max(0.0, budget_ms - used_ms) if budget_ms > 0 else float('inf')

        if remaining_ms > 0:
            # --- MODIFIED: Identify refresh candidates directly by their detection index ---
            refresh_candidate_indices = np.where(due_mask)[0]
            
            idx_refresh = self._get_budgeted_refresh_indices(
                refresh_candidate_indices, match_tid, t0, budget_ms
            )
            
            if len(idx_refresh) > 0:
                if viz_info is not None:
                    for j in idx_refresh:
                        reason_map[j] = 'R-refresh'

                embs_ref, _ = self._compute_embeddings(frame, boxes, idx_refresh, embedder, on_cuda)
                embs[idx_refresh] = embs_ref
                computed_tids |= self._mark_computed(active, idx_refresh, frame_idx, iou_active_det, viz_info, reason_map)

        if on_cuda: torch.cuda.synchronize()
        return embs, perf_counter() - t0, computed_tids

    # In scheduler.py, replace the _get_budgeted_refresh_indices() method:
    def _get_budgeted_refresh_indices(self, candidate_indices: np.ndarray,
                                    match_tid: list[Optional[int]],
                                    t0: float, budget_ms: float) -> np.ndarray:
        """Determines which detection indices to refresh within the time budget."""
        if len(candidate_indices) == 0:
            return np.array([], dtype=int)
        
        # Prioritize items that are already in the pending backlog
        backlog_indices = [idx for idx in candidate_indices if match_tid[idx] in self.pending_refresh]
        due_indices = [idx for idx in candidate_indices if match_tid[idx] not in self.pending_refresh]
        
        # Process backlog first, then newly due items
        prioritized_indices = backlog_indices + due_indices
        
        idx_to_process = []
        for det_idx in prioritized_indices:
            # Predict if computing one more embedding will exceed the budget
            if budget_ms > 0 and (perf_counter() - t0) * 1000.0 + self.ms_per_crop_ema > budget_ms:
                break
            idx_to_process.append(det_idx)
                
        return np.array(idx_to_process, dtype=int)

    def _compute_embeddings(self, frame, boxes, indices, embedder, on_cuda):
        """Helper to run embedder, time it, and update stats."""
        t1 = perf_counter()
        embeddings = embedder.embed_crops(frame, boxes[indices])
        if on_cuda: torch.cuda.synchronize()
        batch_ms = (perf_counter() - t1) * 1000.0
        
        self.update_ema_ms_per_crop(batch_ms, len(indices))
        self.stat_real += len(indices)
        return embeddings, batch_ms

    def _map_tids_to_best_det(self, active_snapshot, iou_active_det: np.ndarray) -> dict[int, int]:
        """For each active track, find the detection index with the highest IoU."""
        if not active_snapshot or iou_active_det.shape[1] == 0:
            return {}
        
        best_det_indices = iou_active_det.argmax(axis=1)
        return {int(track.tid): best_det_indices[i] for i, track in enumerate(active_snapshot)}

    def summary(self, frames: int) -> str:
        """Returns a string summary of scheduler performance."""
        if frames <= 0: return "Embeddings: 0 computed, 0 reused"
        per_frame = self.stat_real / frames
        return f"Embeddings computed: {self.stat_real} total ({per_frame:.2f}/frame), reused: {self.stat_reuse}"