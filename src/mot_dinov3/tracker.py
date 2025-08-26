# src/mot_dinov3/tracker.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

# Optional Hungarian; we fall back to greedy if SciPy isn't present.
try:
    from scipy.optimize import linear_sum_assignment
    HAS_HUNGARIAN = True
except Exception:
    HAS_HUNGARIAN = False


# ----------------------------- helpers -----------------------------

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    IoU matrix for axis-aligned boxes in XYXY format.
    a: (M,4), b: (N,4) -> (M,N)
    """
    M, N = len(a), len(b)
    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    x11, y11, x12, y12 = a[:, 0][:, None], a[:, 1][:, None], a[:, 2][:, None], a[:, 3][:, None]
    x21, y21, x22, y22 = b[:, 0][None, :], b[:, 1][None, :], b[:, 2][None, :], b[:, 3][None, :]
    inter_w = np.maximum(0.0, np.minimum(x12, x22) - np.maximum(x11, x21))
    inter_h = np.maximum(0.0, np.minimum(y12, y22) - np.maximum(y11, y21))
    inter = inter_w * inter_h
    area_a = np.maximum(0.0, (x12 - x11)) * np.maximum(0.0, (y12 - y11))
    area_b = np.maximum(0.0, (x22 - x21)) * np.maximum(0.0, (y22 - y21))
    union = area_a + area_b - inter + 1e-6
    return (inter / union).astype(np.float32)


def cosine_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Returns 1 - cosine_similarity for L2-normalized embeddings.
    A: (M,D), B: (N,D) -> (M,N), lower is better.
    """
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)), dtype=np.float32)
    return (1.0 - A @ B.T).astype(np.float32)


def centers_xyxy(b: np.ndarray) -> np.ndarray:
    """Centers of boxes in XYXY; returns (N,2)."""
    if len(b) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    b = b.astype(np.float32)
    cx = (b[:, 0] + b[:, 2]) * 0.5
    cy = (b[:, 1] + b[:, 3]) * 0.5
    return np.stack([cx, cy], axis=1).astype(np.float32)


# ------------------------------ core -------------------------------

@dataclass
class Track:
    tid: int
    box: np.ndarray                 # (4,) xyxy
    emb: np.ndarray                 # (D,) L2-normalized
    cls: Optional[int] = None
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    state: str = "active"           # "active" | "lost" | "removed"
    alpha: float = 0.9              # EMA factor for embeddings
    gallery: List[np.ndarray] = field(default_factory=list)
    gallery_max: int = 10

    def update(self, box: np.ndarray, emb: np.ndarray):
        """Update track with a new detection + embedding; EMA + bounded gallery."""
        self.box = box.astype(np.float32)
        # EMA on embedding (keep normalized)
        x = self.alpha * self.emb + (1.0 - self.alpha) * emb
        n = np.linalg.norm(x) + 1e-12
        self.emb = (x / n).astype(np.float32)
        # gallery (append raw embedding)
        self.gallery.append(emb.astype(np.float32).copy())
        if len(self.gallery) > self.gallery_max:
            self.gallery.pop(0)
        self.hits += 1
        self.time_since_update = 0
        self.age += 1
        self.state = "active"

    def mark_lost(self):
        self.time_since_update += 1
        self.age += 1
        if self.state == "active":
            self.state = "lost"

    def mark_removed(self):
        self.state = "removed"

    def best_sim(self, emb: np.ndarray) -> float:
        """
        Max dot product vs current EMA embedding and last few gallery items.
        Embeddings are assumed L2-normalized.
        """
        best = float(self.emb @ emb)
        # compare against last up to 5 gallery entries
        for g in self.gallery[-5:]:
            s = float(g @ emb)
            if s > best:
                best = s
        return best


class SimpleTracker:
    """
    Re-ID capable tracker:
      - Stage 1: ACTIVE ↔ detections, two passes (high-conf then low-conf),
                 cost = w_iou * (1 - IoU) + w_app * (1 - cosine_sim)
      - Stage 2: LOST ↔ remaining detections using appearance-only + loose center gate
      - Tracks have states: ACTIVE / LOST / REMOVED
      - Per-track embedding is EMA + small gallery for robustness
      - Optional class-consistent matching
      - Hungarian or greedy association
    """

    def __init__(
        self,
        iou_weight: float = 0.3,
        app_weight: float = 0.7,
        iou_thresh: float = 0.3,          # IoU gate for high-conf pass
        iou_thresh_low: float = 0.2,      # IoU gate for low-conf pass
        reid_sim_thresh: float = 0.6,     # dot(sim) threshold to revive LOST
        max_age: int = 30,                # frames before ACTIVE becomes LOST
        reid_max_age: int = 60,           # frames to keep LOST before removal
        ema_alpha: float = 0.9,
        gallery_size: int = 10,
        use_hungarian: bool = True,
        class_consistent: bool = True,
        conf_high: float = 0.5,
        conf_low: float = 0.1,
        center_gate_base: float = 50.0,   # px allowance for re-ID when just lost
        center_gate_slope: float = 10.0,  # grows per frame lost
    ):
        self.iou_w = float(iou_weight)
        self.app_w = float(app_weight)
        self.iou_thresh = float(iou_thresh)
        self.iou_thresh_low = float(iou_thresh_low)
        self.reid_sim_thresh = float(reid_sim_thresh)
        self.max_age = int(max_age)
        self.reid_max = int(reid_max_age)
        self.ema_alpha = float(ema_alpha)
        self.gallery_size = int(gallery_size)
        self.use_hungarian = (use_hungarian and HAS_HUNGARIAN)
        self.class_consistent = bool(class_consistent)
        self.conf_high = float(conf_high)
        self.conf_low = float(conf_low)
        self.center_gate_base = float(center_gate_base)
        self.center_gate_slope = float(center_gate_slope)

        self.tracks: List[Track] = []
        self._next_id = 1

    # ----------------------- internal utilities -----------------------

    def _new_track(self, box: np.ndarray, emb: np.ndarray, cls: Optional[int]) -> Track:
        t = Track(
            tid=self._next_id,
            box=box.astype(np.float32),
            emb=emb.astype(np.float32),
            cls=cls,
            alpha=self.ema_alpha,
            gallery_max=self.gallery_size,
        )
        self._next_id += 1
        self.tracks.append(t)
        return t

    def _active_tracks(self) -> List[int]:
        return [i for i, t in enumerate(self.tracks) if t.state == "active"]

    def _lost_tracks(self) -> List[int]:
        return [i for i, t in enumerate(self.tracks) if (t.state == "lost" and t.time_since_update <= self.reid_max)]

    def _prune_removed(self):
        keep = []
        for t in self.tracks:
            if t.state == "removed":
                continue
            if t.state == "lost" and t.time_since_update > self.reid_max:
                continue
            keep.append(t)
        self.tracks = keep

    def _apply_class_mask(self, C: np.ndarray, act_idx: List[int], det_ids: List[int], clses: Optional[np.ndarray]) -> np.ndarray:
        """Set a very large cost when track.cls and det class disagree."""
        if clses is None or not self.class_consistent:
            return C
        BIG = 1e6
        for r, ti in enumerate(act_idx):
            tcls = self.tracks[ti].cls
            if tcls is None:
                continue
            for c, j in enumerate(det_ids):
                dcls = int(clses[j]) if j < len(clses) else None
                if dcls is not None and dcls != tcls:
                    C[r, c] = BIG
        return C

    def _associate_active(
        self,
        act_idx: List[int],
        det_ids: List[int],
        boxes: np.ndarray,
        embs: np.ndarray,
        clses: Optional[np.ndarray],
        iou_gate: float,
    ) -> Tuple[set[int], set[int]]:
        """
        Associate ACTIVE tracks with a subset of detections (det_ids).
        Returns (unmatched_det_ids, unmatched_track_indices).
        """
        unmatched_dets = set(det_ids)
        unmatched_tracks = set(act_idx)
        if len(act_idx) == 0 or len(det_ids) == 0:
            return unmatched_dets, unmatched_tracks

        A = np.stack([self.tracks[i].box for i in act_idx], axis=0).astype(np.float32)
        B = boxes[det_ids].astype(np.float32)
        I = iou_xyxy(A, B)                         # (T,N)
        Aemb = np.stack([self.tracks[i].emb for i in act_idx], axis=0).astype(np.float32)
        C_app = cosine_dist(Aemb, embs[det_ids])   # (T,N)
        C = self.iou_w * (1.0 - I) + self.app_w * C_app
        C = self._apply_class_mask(C, act_idx, det_ids, clses)

        # Hard IoU gating for temporal continuity
        # (Optionally, we could set a floor; here we leave cost as-is, relying on combined score.)
        # To *enforce* the gate strictly, uncomment the next line:
        # C[I < iou_gate] = 1e6

        if self.use_hungarian:
            rows, cols = linear_sum_assignment(C)
            for r, c in zip(rows, cols):
                ti = act_idx[r]
                j = det_ids[c]
                # Basic acceptance; you can add additional gates here if needed
                self.tracks[ti].update(B[c], embs[j])
                unmatched_dets.discard(j)
                unmatched_tracks.discard(ti)
        else:
            # Greedy: lowest cost first
            order = np.argsort(C, axis=None)
            used_r, used_c = set(), set()
            for flat in order:
                r, c = divmod(flat, C.shape[1])
                if r in used_r or c in used_c:
                    continue
                ti = act_idx[r]
                j = det_ids[c]
                self.tracks[ti].update(B[c], embs[j])
                used_r.add(r); used_c.add(c)
                unmatched_dets.discard(j)
                unmatched_tracks.discard(ti)

        return unmatched_dets, unmatched_tracks

    # --------------------------- public API ---------------------------

    def update(
        self,
        det_boxes: np.ndarray,
        det_embs: np.ndarray,
        confs: Optional[np.ndarray] = None,
        clses: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """
        Update tracker with detections of the current frame.

        det_boxes: (N,4) XYXY
        det_embs : (N,D) L2-normalized embeddings
        confs    : (N,) optional detector confidences
        clses    : (N,) optional detector class ids

        Returns ACTIVE tracks (draw these). LOST tracks are kept internally for re-ID.
        """
        det_boxes = det_boxes.astype(np.float32) if det_boxes is not None else np.zeros((0, 4), np.float32)
        det_embs = det_embs.astype(np.float32) if det_embs is not None else np.zeros((0, 128), np.float32)

        N = len(det_boxes)
        if confs is None:
            hi_ids = list(range(N))
            lo_ids: List[int] = []
        else:
            hi_ids = [i for i, c in enumerate(confs) if c >= self.conf_high]
            lo_ids = [i for i, c in enumerate(confs) if self.conf_low <= c < self.conf_high]

        # Stage 1: ACTIVE ↔ high-confidence detections
        act_idx = self._active_tracks()
        unmatched_hi, unmatched_tracks = self._associate_active(
            act_idx, hi_ids, det_boxes, det_embs, clses, self.iou_thresh
        )

        # Stage 1b: ACTIVE ↔ low-confidence detections
        unmatched_lo, unmatched_tracks = self._associate_active(
            list(unmatched_tracks), lo_ids, det_boxes, det_embs, clses, self.iou_thresh_low
        )

        unmatched_dets = set(unmatched_hi) | set(unmatched_lo)

        # Tracks without matches become LOST (or remain LOST, aging)
        for ti in list(unmatched_tracks):
            t = self.tracks[ti]
            t.mark_lost()
            if t.time_since_update > self.max_age:
                t.state = "lost"  # explicit; pruning happens later

        # Stage 2: Re-ID LOST ↔ remaining detections (appearance-only + loose center gate)
        det_left = sorted(list(unmatched_dets))
        lost_idx = self._lost_tracks()
        if len(lost_idx) and len(det_left):
            # Build cost = 1 - max_sim
            T = len(lost_idx)
            D = len(det_left)
            C = np.ones((T, D), dtype=np.float32)

            # centers & gating radius
            lost_centers = centers_xyxy(np.stack([self.tracks[i].box for i in lost_idx], axis=0))
            det_centers = centers_xyxy(det_boxes[det_left])

            for r, ti in enumerate(lost_idx):
                tr = self.tracks[ti]
                # allowed center jump grows with time since update
                allow = self.center_gate_base + self.center_gate_slope * float(tr.time_since_update)
                for jj, j in enumerate(det_left):
                    # center gate
                    if len(lost_centers) and len(det_centers):
                        if np.linalg.norm(lost_centers[r] - det_centers[jj]) > allow:
                            C[r, jj] = 1e6  # gated out
                            continue
                    sim = tr.best_sim(det_embs[j])  # dot in [-1,1], higher=better
                    C[r, jj] = 1.0 - sim

            # Solve assignment
            if self.use_hungarian:
                rows, cols = linear_sum_assignment(C)
                taken_dets: set[int] = set()
                for r, c in zip(rows, cols):
                    sim = 1.0 - C[r, c]
                    if sim < self.reid_sim_thresh:
                        continue
                    ti = lost_idx[r]
                    j = det_left[c]
                    if j in taken_dets:
                        continue
                    self.tracks[ti].update(det_boxes[j], det_embs[j])
                    taken_dets.add(j)
                    unmatched_dets.discard(j)
            else:
                # Greedy on appearance cost
                order = np.argsort(C, axis=None)
                used_r, used_c = set(), set()
                for flat in order:
                    r, c = divmod(flat, C.shape[1])
                    if r in used_r or c in used_c:
                        continue
                    sim = 1.0 - C[r, c]
                    if sim < self.reid_sim_thresh:
                        continue
                    ti = lost_idx[r]
                    j = det_left[c]
                    self.tracks[ti].update(det_boxes[j], det_embs[j])
                    used_r.add(r); used_c.add(c)
                    unmatched_dets.discard(j)

        # New tracks for any unmatched detections
        for j in sorted(list(unmatched_dets)):
            cls = int(clses[j]) if clses is not None and j < len(clses) else None
            self._new_track(det_boxes[j], det_embs[j], cls)

        # Prune very old LOST
        self._prune_removed()

        # Return ACTIVE tracks only (to draw). LOST are kept internally for future re-ID.
        return [t for t in self.tracks if t.state == "active"]
