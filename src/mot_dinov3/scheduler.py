# src/mot_dinov3/scheduler.py
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional, Tuple, Iterable, Set

import numpy as np
import torch

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

    @staticmethod
    def _embedder_device_type(embedder) -> str:
        """
        Return 'cuda' or 'cpu' given either a DINOExtractor or a raw embedder.
        Handles the adapter case where the real embedder is at ._e.
        """
        dev = getattr(embedder, "device", None)
        if dev is None and hasattr(embedder, "_e"):  # DINOExtractor → DinoV3Embedder
            dev = getattr(embedder._e, "device", None)
        if isinstance(dev, torch.device):
            return dev.type
        return str(dev or "cpu")
        
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

    @torch.inference_mode()
    def run(self,
            frame: np.ndarray,
            boxes: np.ndarray,
            embedder,
            tracker,
            frame_idx: int,
            budget_ms: float = 0.0) -> Tuple[np.ndarray, float, Set[int]]:
        """
        Orchestrate embeddings for this frame:
        1) Reuse cached vectors where allowed (for stable tracks).
        2) Compute 'critical' embeddings immediately (new/crowded/weak IoU).
        3) Spend remaining per-frame budget on 'refresh' embeddings for stale tracks.

        Returns:
        embs           : (N,D) float32
        t_emb_seconds  : wall time spent here
        refreshed_tids : set of track IDs refreshed now (for UI '*')
        """
        on_cuda = (self._embedder_device_type(embedder) == "cuda")
        if on_cuda:
            torch.cuda.synchronize()
        t0 = perf_counter()

        N = int(len(boxes))
        D = int(getattr(embedder, "dim", getattr(embedder, "emb_dim", 768)))
        if N == 0:
            return np.zeros((0, D), dtype=np.float32), 0.0, set()

        need_mask, reuse_tids, active_snapshot, due_mask, match_tid = self.plan(tracker, boxes, frame_idx)

        embs = np.zeros((N, D), dtype=np.float32)
        refreshed_tids: set[int] = set()

        # ---- Reuse cached
        if len(active_snapshot):
            tid2emb = {int(t.tid): t.emb for t in active_snapshot}
            for j, tid in enumerate(reuse_tids):
                if tid is not None and tid in tid2emb:
                    embs[j] = tid2emb[tid]
                    self.stat_reuse += 1

        # ---- 1) Critical set (not budgeted)
        idx_crit = np.nonzero(need_mask & ~due_mask)[0]
        if len(idx_crit) > 0:
            t1 = perf_counter()
            embs_crit = embedder.embed_crops(frame, boxes[idx_crit])
            if on_cuda:
                torch.cuda.synchronize()
            batch_ms = (perf_counter() - t1) * 1000.0
            embs[idx_crit] = embs_crit
            self.update_ema_ms_per_crop(batch_ms, len(idx_crit))
            self.stat_real += len(idx_crit)
            refreshed_tids |= self.mark_refreshed(active_snapshot, boxes, idx_crit, frame_idx)

        # Remaining budget (ms)
        if budget_ms <= 0:
            remaining_ms = float("inf")
        else:
            if on_cuda:
                torch.cuda.synchronize()
            used_ms = (perf_counter() - t0) * 1000.0
            remaining_ms = max(0.0, budget_ms - used_ms)

        # ---- 2) Refresh (strictly budgeted)
        if remaining_ms > 0 and len(active_snapshot) > 0:
            # rank stale first
            def _stale(tid: int) -> int:
                return frame_idx - int(self.last_embed_frame.get(int(tid), -10**9))

            due_tids_this = [tid for j, tid in enumerate(match_tid) if due_mask[j] and tid is not None]
            due_order = sorted(set(due_tids_this), key=_stale, reverse=True)
            candidates = list(self.pending_refresh) + [t for t in due_order if t not in self.pending_refresh]

            # map TID → best det index via IoU with ACTIVE snapshot
            best_det_for_tid = self._map_best_det_for_tids(active_snapshot, boxes)

            chosen_now: list[int] = []
            for tid in candidates:
                if budget_ms <= 0:  # unlimited
                    chosen_now.append(int(tid)); continue
                if on_cuda:
                    torch.cuda.synchronize()
                used_ms = (perf_counter() - t0) * 1000.0
                remaining_ms = max(0.0, budget_ms - used_ms)
                est = max(0.5, self.ms_per_crop_ema)
                if remaining_ms < est:
                    break
                chosen_now.append(int(tid))

            idx_refresh: list[int] = []
            for tid in chosen_now:
                j = best_det_for_tid.get(int(tid))
                if j is None or j in idx_crit or j in idx_refresh:
                    continue
                idx_refresh.append(int(j))

            if len(idx_refresh) > 0:
                t2 = perf_counter()
                embs_ref = embedder.embed_crops(frame, boxes[np.array(idx_refresh, dtype=int)])
                if on_cuda:
                    torch.cuda.synchronize()
                batch_ms = (perf_counter() - t2) * 1000.0
                embs[np.array(idx_refresh, dtype=int)] = embs_ref
                self.update_ema_ms_per_crop(batch_ms, len(idx_refresh))
                self.stat_real += len(idx_refresh)
                refreshed_tids |= self.mark_refreshed(active_snapshot, boxes, np.array(idx_refresh, dtype=int), frame_idx)
                self._remove_pending(chosen_now)

        if on_cuda:
            torch.cuda.synchronize()
        return embs, (perf_counter() - t0), refreshed_tids

    def _map_best_det_for_tids(self, active_snapshot, boxes: np.ndarray) -> dict[int, int]:
        """For each ACTIVE tid, pick the detection index with max IoU this frame."""
        if len(active_snapshot) == 0 or len(boxes) == 0:
            return {}
        A = np.stack([t.box for t in active_snapshot], axis=0).astype(np.float32)
        B = boxes.astype(np.float32)
        x11, y11, x12, y12 = A[:,0][:,None], A[:,1][:,None], A[:,2][:,None], A[:,3][:,None]
        x21, y21, x22, y22 = B[:,0][None,:], B[:,1][None,:], B[:,2][None,:], B[:,3][None,:]
        inter_w = np.maximum(0, np.minimum(x12, x22) - np.maximum(x11, x21))
        inter_h = np.maximum(0, np.minimum(y12, y22) - np.maximum(y11, y21))
        inter = inter_w * inter_h
        area_a = (x12 - x11) * (y12 - y11)
        area_b = (x22 - x21) * (y22 - y21)
        I = inter / (area_a + area_b - inter + 1e-6)  # (T,N)

        best: dict[int, int] = {}
        for det_j in range(len(B)):
            trk_i = int(I[:, det_j].argmax())
            tid = int(active_snapshot[trk_i].tid)
            if (tid not in best) or (I[trk_i, det_j] > I[trk_i, best[tid]]):
                best[tid] = det_j
        return best

    def summary(self, frames: int) -> str:
        if frames <= 0:
            return "Embeddings computed: 0 total (0.00/frame), reused: 0"
        per_frame = self.stat_real / frames
        return f"Embeddings computed: {self.stat_real} total ({per_frame:.2f}/frame), reused: {self.stat_reuse}"
