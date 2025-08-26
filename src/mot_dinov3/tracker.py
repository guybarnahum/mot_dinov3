# src/mot_dinov3/tracker.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set
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
    cls: Optional[int] = None       # stable/majority class (for display)
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    state: str = "active"           # "active" | "lost" | "removed"
    alpha: float = 0.9              # EMA factor base for embeddings
    gallery: List[np.ndarray] = field(default_factory=list)
    gallery_max: int = 10

    # New: class stability & bookkeeping
    cls_hist: Dict[int, float] = field(default_factory=dict)  # exponential votes
    last_conf: float = 0.0

    def update(self, box: np.ndarray, emb: np.ndarray,
               det_conf: float = 1.0, det_cls: Optional[int] = None,
               conf_min_update: float = 0.3, conf_update_weight: float = 0.5):
        """
        Update track with a new detection + embedding.
        - Embedding EMA is confidence-weighted: stronger update when det_conf is high.
        - Class histogram gets confidence-weighted votes; self.cls becomes majority class.
        """
        self.box = box.astype(np.float32)

        # Confidence-weighted EMA on embedding (keep normalized)
        # Effective beta in [0, 1]: larger when more confident.
        det_conf = float(np.clip(det_conf, 0.0, 1.0))
        if det_conf >= conf_min_update:
            beta = (1.0 - self.alpha)
            beta *= (conf_update_weight * det_conf + (1.0 - conf_update_weight))
            x = (1.0 - beta) * self.emb + beta * emb
            n = np.linalg.norm(x) + 1e-12
            self.emb = (x / n).astype(np.float32)
            # gallery (append raw embedding)
            self.gallery.append(emb.astype(np.float32).copy())
            if len(self.gallery) > self.gallery_max:
                self.gallery.pop(0)

            # Class histogram (only if class provided)
            if det_cls is not None and det_cls >= 0:
                w = det_conf * 0.6  # internal smoothing; tuned with cls votes
                self.cls_hist[det_cls] = self.cls_hist.get(det_cls, 0.0) * (1.0 - w) + w
                # light decay on others so the majority can change
                for k in list(self.cls_hist.keys()):
                    if k != det_cls:
                        self.cls_hist[k] *= (1.0 - 0.05 * w)
                        if self.cls_hist[k] < 1e-4:
                            del self.cls_hist[k]
                if self.cls_hist:
                    self.cls = max(self.cls_hist.items(), key=lambda kv: kv[1])[0]

        self.hits += 1
        self.time_since_update = 0
        self.age += 1
        self.state = "active"
        self.last_conf = det_conf

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
                 (optional IoU-only on low-conf to avoid drifting)
      - Stage 2: LOST ↔ remaining detections using appearance-only + loose center gate
      - Tracks have states: ACTIVE / LOST / REMOVED
      - Per-track embedding is confidence-weighted EMA + small gallery for robustness
      - Soft class penalty to reduce ID switches on detector class flips
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

        # Class handling:
        class_consistent: bool = True,    # apply soft class penalty if True
        class_penalty: float = 0.15,      # additive cost when det class != track.cls

        # Confidence-aware updates:
        conf_high: float = 0.5,           # high-conf threshold
        conf_low: float = 0.1,            # low-conf threshold
        conf_min_update: float = 0.3,     # only update emb/class if det_conf >= this
        conf_update_weight: float = 0.5,  # scales EMA strength with det_conf

        # Low-conf behavior:
        low_conf_iou_only: bool = True,   # if True, Stage 1b uses IoU-only cost for low-conf
                                          # and (implicitly) updates will be suppressed by conf_min_update

        # Re-ID gating:
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

        # class handling
        self.class_consistent = bool(class_consistent)
        self.class_penalty = float(class_penalty)

        # confidence-aware updates
        self.conf_high = float(conf_high)
        self.conf_low = float(conf_low)
        self.conf_min_update = float(conf_min_update)
        self.conf_update_weight = float(conf_update_weight)

        # low-conf behavior
        self.low_conf_iou_only = bool(low_conf_iou_only)

        # re-id gating
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
            cls=cls if (cls is not None and cls >= 0) else None,  # initial displayed class
            alpha=self.ema_alpha,
            gallery_max=self.gallery_size,
        )
        if t.cls is not None:
            t.cls_hist[t.cls] = 1.0
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

    def _add_soft_class_penalty(self, C: np.ndarray, act_idx: List[int], det_ids: List[int], clses: Optional[np.ndarray]) -> np.ndarray:
        """
        Add a soft penalty (additive cost) when detection class != track's stable class (t.cls).
        This reduces ID switches from detector class flicker without hard-blocking matches.
        """
        if clses is None or not self.class_consistent or self.class_penalty <= 0.0:
            return C
        if len(act_idx) == 0 or len(det_ids) == 0:
            return C

        P = np.zeros_like(C, dtype=np.float32)
        for r, ti in enumerate(act_idx):
            tcls = self.tracks[ti].cls  # stabilized label
            if tcls is None:
                continue
            for c, j in enumerate(det_ids):
                dcls = int(clses[j]) if j < len(clses) else None
                if dcls is not None and dcls != tcls:
                    P[r, c] = 1.0
        return C + self.class_penalty * P

    def _associate_active(
        self,
        act_idx: List[int],
        det_ids: List[int],
        boxes: np.ndarray,
        embs: np.ndarray,
        clses: Optional[np.ndarray],
        confs: Optional[np.ndarray],
        iou_gate: float,
        use_iou_only: bool = False,
    ) -> Tuple[Set[int], Set[int]]:
        """
        Associate ACTIVE tracks with a subset of detections (det_ids).
        Returns (unmatched_det_ids, unmatched_track_indices).
        """
        unmatched_dets: Set[int] = set(det_ids)
        unmatched_tracks: Set[int] = set(act_idx)
        if len(act_idx) == 0 or len(det_ids) == 0:
            return unmatched_dets, unmatched_tracks

        A = np.stack([self.tracks[i].box for i in act_idx], axis=0).astype(np.float32)
        B = boxes[det_ids].astype(np.float32)
        I = iou_xyxy(A, B)  # (T,N)

        if use_iou_only:
            C = (1.0 - I)
        else:
            Aemb = np.stack([self.tracks[i].emb for i in act_idx], axis=0).astype(np.float32)
            C_app = cosine_dist(Aemb, embs[det_ids])   # (T,N)
            C = self.iou_w * (1.0 - I) + self.app_w * C_app

        # Soft class penalty (if enabled)
        C = self._add_soft_class_penalty(C, act_idx, det_ids, clses)

        if self.use_hungarian:
            rows, cols = linear_sum_assignment(C)
            for r, c in zip(rows, cols):
                ti = act_idx[r]
                j = det_ids[c]
                det_conf = float(confs[j]) if confs is not None and j < len(confs) else 1.0
                det_cls  = int(clses[j]) if clses is not None and j < len(clses) else None
                # Update; internal logic will suppress embed/class updates if det_conf is low
                self.tracks[ti].update(
                    B[c], embs[j],
                    det_conf=det_conf,
                    det_cls=det_cls,
                    conf_min_update=self.conf_min_update,
                    conf_update_weight=self.conf_update_weight,
                )
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
                det_conf = float(confs[j]) if confs is not None and j < len(confs) else 1.0
                det_cls  = int(clses[j]) if clses is not None and j < len(clses) else None
                self.tracks[ti].update(
                    B[c], embs[j],
                    det_conf=det_conf,
                    det_cls=det_cls,
                    conf_min_update=self.conf_min_update,
                    conf_update_weight=self.conf_update_weight,
                )
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
        confs    : (N,) optional detector confidences in [0,1]
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

        # Stage 1: ACTIVE ↔ high-confidence detections (IoU+appearance)
        act_idx = self._active_tracks()
        unmatched_hi, unmatched_tracks = self._associate_active(
            act_idx, hi_ids, det_boxes, det_embs, clses, confs, self.iou_thresh, use_iou_only=False
        )

        # Stage 1b: ACTIVE ↔ low-confidence detections
        # Optionally IoU-only (to avoid drifting embeddings/classes on weak boxes)
        unmatched_lo, unmatched_tracks = self._associate_active(
            list(unmatched_tracks), lo_ids, det_boxes, det_embs, clses, confs, self.iou_thresh_low,
            use_iou_only=self.low_conf_iou_only
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
            T = len(lost_idx)
            D = len(det_left)
            C = np.ones((T, D), dtype=np.float32)

            lost_centers = centers_xyxy(np.stack([self.tracks[i].box for i in lost_idx], axis=0))
            det_centers = centers_xyxy(det_boxes[det_left])

            for r, ti in enumerate(lost_idx):
                tr = self.tracks[ti]
                allow = self.center_gate_base + self.center_gate_slope * float(tr.time_since_update)
                for jj, j in enumerate(det_left):
                    # center gate
                    if len(lost_centers) and len(det_centers):
                        if np.linalg.norm(lost_centers[r] - det_centers[jj]) > allow:
                            C[r, jj] = 1e6
                            continue
                    sim = tr.best_sim(det_embs[j])  # dot in [-1,1], higher=better
                    C[r, jj] = 1.0 - sim

            if self.use_hungarian:
                rows, cols = linear_sum_assignment(C)
                taken_dets: Set[int] = set()
                for r, c in zip(rows, cols):
                    sim = 1.0 - C[r, c]
                    if sim < self.reid_sim_thresh:
                        continue
                    ti = lost_idx[r]
                    j = det_left[c]
                    if j in taken_dets:
                        continue
                    det_conf = float(confs[j]) if confs is not None and j < len(confs) else 1.0
                    det_cls  = int(clses[j]) if clses is not None and j < len(clses) else None
                    self.tracks[ti].update(
                        det_boxes[j], det_embs[j],
                        det_conf=det_conf,
                        det_cls=det_cls,
                        conf_min_update=self.conf_min_update,
                        conf_update_weight=self.conf_update_weight,
                    )
                    taken_dets.add(j)
                    unmatched_dets.discard(j)
            else:
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
                    det_conf = float(confs[j]) if confs is not None and j < len(confs) else 1.0
                    det_cls  = int(clses[j]) if clses is not None and j < len(clses) else None
                    self.tracks[ti].update(
                        det_boxes[j], det_embs[j],
                        det_conf=det_conf,
                        det_cls=det_cls,
                        conf_min_update=self.conf_min_update,
                        conf_update_weight=self.conf_update_weight,
                    )
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
