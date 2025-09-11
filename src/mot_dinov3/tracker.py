# src/mot_dinov3/tracker.py (Full Replacement)
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set
import numpy as np

# Import shared helpers from the new utils module
from . import utils

# Optional Hungarian; we fall back to greedy if SciPy isn't present.
try:
    from scipy.optimize import linear_sum_assignment
    HAS_HUNGARIAN = True
except ImportError:
    HAS_HUNGARIAN = False

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
    cls_hist: Dict[int, float] = field(default_factory=dict)
    last_conf: float = 0.0

    def update(self, box: np.ndarray, emb: np.ndarray,
               det_conf: float = 1.0, det_cls: Optional[int] = None,
               conf_min_update: float = 0.3, conf_update_weight: float = 0.5,
               class_vote_smoothing: float = 0.6, class_decay_factor: float = 0.05):
        self.box = box.astype(np.float32)

        det_conf = float(np.clip(det_conf, 0.0, 1.0))
        if det_conf >= conf_min_update:
            beta = (1.0 - self.alpha) * (conf_update_weight * det_conf + (1.0 - conf_update_weight))
            x = (1.0 - beta) * self.emb + beta * emb
            self.emb = (x / (np.linalg.norm(x) + 1e-12)).astype(np.float32)
            
            self.gallery.append(emb.astype(np.float32).copy())
            if len(self.gallery) > self.gallery_max: self.gallery.pop(0)

            if det_cls is not None and det_cls >= 0:
                self._update_class_hist(det_cls, det_conf, class_vote_smoothing, class_decay_factor)

        self.hits += 1
        self.time_since_update = 0
        self.age += 1
        self.state = "active"
        self.last_conf = det_conf

    def _update_class_hist(self, det_cls: int, det_conf: float, smoothing: float, decay: float):
        """Helper to manage the class histogram update logic."""
        w = det_conf * smoothing
        self.cls_hist[det_cls] = self.cls_hist.get(det_cls, 0.0) * (1.0 - w) + w
        
        for k in list(self.cls_hist.keys()):
            if k != det_cls:
                self.cls_hist[k] *= (1.0 - decay * w)
                if self.cls_hist[k] < 1e-4: del self.cls_hist[k]
        
        if self.cls_hist:
            self.cls = max(self.cls_hist.items(), key=lambda kv: kv[1])[0]

    def mark_lost(self):
        self.time_since_update += 1
        self.age += 1
        if self.state == "active":
            self.state = "lost"

    def mark_removed(self):
        self.state = "removed"

    def best_sim(self, emb: np.ndarray) -> float:
        """Max dot product vs current EMA embedding and recent gallery items."""
        best = float(self.emb @ emb)
        for g_emb in self.gallery[-5:]:
            sim = float(g_emb @ emb)
            if sim > best: best = sim
        return best


class SimpleTracker:
    def __init__(
        self,
        iou_weight: float = 0.3,
        app_weight: float = 0.7,
        iou_thresh: float = 0.3,
        iou_thresh_low: float = 0.2,
        reid_sim_thresh: float = 0.6,
        max_age: int = 30,
        reid_max_age: int = 60,
        ema_alpha: float = 0.9,
        gallery_size: int = 10,
        use_hungarian: bool = True,
        class_consistent: bool = True,
        class_penalty: float = 0.15,
        conf_high: float = 0.5,
        conf_low: float = 0.1,
        conf_min_update: float = 0.3,
        conf_update_weight: float = 0.5,
        low_conf_iou_only: bool = True,
        center_gate_base: float = 50.0,
        center_gate_slope: float = 10.0,
        # REFACTOR: Expose "magic numbers" as configurable parameters
        class_vote_smoothing: float = 0.6,
        class_decay_factor: float = 0.05,
    ):
        self.iou_w, self.app_w = float(iou_weight), float(app_weight)
        self.iou_thresh, self.iou_thresh_low = float(iou_thresh), float(iou_thresh_low)
        self.reid_sim_thresh = float(reid_sim_thresh)
        self.max_age, self.reid_max = int(max_age), int(reid_max_age)
        self.ema_alpha, self.gallery_size = float(ema_alpha), int(gallery_size)
        self.use_hungarian = (use_hungarian and HAS_HUNGARIAN)
        self.class_consistent, self.class_penalty = bool(class_consistent), float(class_penalty)
        self.conf_high, self.conf_low = float(conf_high), float(conf_low)
        self.conf_min_update, self.conf_update_weight = float(conf_min_update), float(conf_update_weight)
        self.low_conf_iou_only = bool(low_conf_iou_only)
        self.center_gate_base, self.center_gate_slope = float(center_gate_base), float(center_gate_slope)
        self.class_vote_smoothing, self.class_decay_factor = float(class_vote_smoothing), float(class_decay_factor)

        self.tracks: List[Track] = []
        self._next_id = 1

    def _new_track(self, box: np.ndarray, emb: np.ndarray, cls: Optional[int]) -> Track:
        t = Track(
            tid=self._next_id, box=box, emb=emb,
            cls=cls if (cls is not None and cls >= 0) else None,
            alpha=self.ema_alpha, gallery_max=self.gallery_size,
        )
        if t.cls is not None: t.cls_hist[t.cls] = 1.0
        self._next_id += 1
        self.tracks.append(t)
        return t

    def _prune_removed(self):
        self.tracks = [t for t in self.tracks if t.state != "removed" and 
                       not (t.state == "lost" and t.time_since_update > self.reid_max)]

    # REFACTOR: Vectorized implementation for massive speedup
    def _add_soft_class_penalty(self, C: np.ndarray, act_idx: List[int], det_ids: List[int], clses: Optional[np.ndarray]) -> np.ndarray:
        if clses is None or not self.class_consistent or self.class_penalty <= 0.0 or not act_idx or not det_ids:
            return C

        track_classes = np.array([self.tracks[ti].cls for ti in act_idx], dtype=object)
        det_classes = clses[det_ids]

        is_valid_track = (track_classes != None)[:, None]
        is_mismatched = track_classes[:, None] != det_classes[None, :]
        penalty_mask = np.logical_and(is_valid_track, is_mismatched)

        return C + self.class_penalty * penalty_mask.astype(np.float32)

    def _associate(self, cost_matrix: np.ndarray, unmatched_dets: Set[int], unmatched_tracks: Set[int],
                   act_idx: List[int], det_ids: List[int], boxes: np.ndarray, embs: np.ndarray,
                   clses: Optional[np.ndarray], confs: Optional[np.ndarray]):
        
        if self.use_hungarian:
            rows, cols = linear_sum_assignment(cost_matrix)
        else: # Greedy assignment
            rows, cols = [], []
            used_r, used_c = set(), set()
            for r, c in np.unravel_index(np.argsort(cost_matrix, axis=None), cost_matrix.shape):
                if r not in used_r and c not in used_c:
                    rows.append(r); cols.append(c)
                    used_r.add(r); used_c.add(c)
        
        for r, c in zip(rows, cols):
            ti, j = act_idx[r], det_ids[c]
            det_conf = confs[j] if confs is not None else 1.0
            det_cls = int(clses[j]) if clses is not None else None

            self.tracks[ti].update(
                boxes[j], embs[j], det_conf=det_conf, det_cls=det_cls,
                conf_min_update=self.conf_min_update, conf_update_weight=self.conf_update_weight,
                class_vote_smoothing=self.class_vote_smoothing, class_decay_factor=self.class_decay_factor
            )
            unmatched_dets.discard(j)
            unmatched_tracks.discard(ti)

    def update(self, det_boxes: np.ndarray, det_embs: np.ndarray,
               confs: Optional[np.ndarray] = None, clses: Optional[np.ndarray] = None) -> List[Track]:
        
        N = len(det_boxes)
        hi_ids = [i for i, c in enumerate(confs)] if confs is not None else list(range(N))
        lo_ids = [i for i, c in enumerate(confs) if self.conf_low <= c < self.conf_high] if confs is not None else []
        if confs is not None: hi_ids = [i for i in hi_ids if i not in set(lo_ids)]
        
        # --- Stage 1: Associate ACTIVE tracks ---
        act_idx = [i for i, t in enumerate(self.tracks) if t.state == "active"]
        unmatched_tracks = set(act_idx)
        
        # Pass 1a: High-confidence detections
        unmatched_hi, unmatched_tracks = self._match_active(act_idx, hi_ids, det_boxes, det_embs, clses, confs, use_iou_only=False)
        
        # Pass 1b: Low-confidence detections
        unmatched_lo, unmatched_tracks = self._match_active(list(unmatched_tracks), lo_ids, det_boxes, det_embs, clses, confs, use_iou_only=self.low_conf_iou_only)
        
        for ti in unmatched_tracks: self.tracks[ti].mark_lost()
        
        # --- Stage 2: Re-ID LOST tracks ---
        unmatched_dets = unmatched_hi.union(unmatched_lo)
        self._reid_lost(sorted(list(unmatched_dets)), det_boxes, det_embs, clses, confs)

        # --- Stage 3: Create new tracks ---
        for j in sorted(list(unmatched_dets)):
            cls = int(clses[j]) if clses is not None else None
            self._new_track(det_boxes[j], det_embs[j], cls)

        self._prune_removed()
        return [t for t in self.tracks if t.state == "active"]

    def _match_active(self, act_idx, det_ids, boxes, embs, clses, confs, use_iou_only=False):
        unmatched_dets, unmatched_tracks = set(det_ids), set(act_idx)
        if not act_idx or not det_ids: return unmatched_dets, unmatched_tracks

        track_boxes = np.stack([self.tracks[i].box for i in act_idx])
        iou_val = utils.iou_matrix(track_boxes, boxes[det_ids])
        cost_iou = 1.0 - iou_val
        
        if use_iou_only:
            cost_matrix = cost_iou
        else:
            track_embs = np.stack([self.tracks[i].emb for i in act_idx])
            cost_app = utils.cosine_cost_matrix(track_embs, embs[det_ids])
            cost_matrix = self.iou_w * cost_iou + self.app_w * cost_app
        
        cost_matrix = self._add_soft_class_penalty(cost_matrix, act_idx, det_ids, clses)
        
        self._associate(cost_matrix, unmatched_dets, unmatched_tracks, act_idx, det_ids, boxes, embs, clses, confs)
        return unmatched_dets, unmatched_tracks

    def _reid_lost(self, det_left, boxes, embs, clses, confs):
        lost_idx = [i for i, t in enumerate(self.tracks) if t.state == "lost"]
        if not lost_idx or not det_left: return

        lost_embs = np.stack([self.tracks[i].emb for i in lost_idx])
        cost_matrix = utils.cosine_cost_matrix(lost_embs, embs[det_left])
        
        # Gating
        lost_centers = utils.centers_xyxy(np.stack([self.tracks[i].box for i in lost_idx]))
        det_centers = utils.centers_xyxy(boxes[det_left])
        dist = np.linalg.norm(lost_centers[:, None, :] - det_centers[None, :, :], axis=2)
        allowance = np.array([self.center_gate_base + self.center_gate_slope * t.time_since_update for t in [self.tracks[i] for i in lost_idx]])
        cost_matrix[dist > allowance[:, None]] = 1e6 # Gate out distant matches

        self._associate(cost_matrix, set(det_left), set(lost_idx), lost_idx, det_left, boxes, embs, clses, confs)