# src/mot_dinov3/tracker.py (Enhanced with Re-ID Events and Motion Heuristics)
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import deque

import numpy as np

from . import utils

try:
    from scipy.optimize import linear_sum_assignment
    HAS_HUNGARIAN = True
except ImportError:
    HAS_HUNGARIAN = False

@dataclass
class Track:
    tid: int
    box: np.ndarray
    emb: np.ndarray
    cls: Optional[int] = None
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    state: str = "active"
    alpha: float = 0.9
    gallery: List[np.ndarray] = field(default_factory=list, repr=False)
    gallery_max: int = 10
    cls_hist: Dict[int, float] = field(default_factory=dict, repr=False)
    last_conf: float = 0.0
    center: np.ndarray = field(init=False)
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]), repr=False)
    # --- ENHANCED: Add position history for drawing trails ---
    center_history: deque = field(default_factory=lambda: deque(maxlen=25), repr=False)


    def __post_init__(self):
        self.center = utils.centers_xyxy(self.box[np.newaxis, :])[0]
        self.center_history.append(self.center.astype(int))

    def update(self, box: np.ndarray, emb: np.ndarray,
               det_conf: float = 1.0, det_cls: Optional[int] = None,
               conf_min_update: float = 0.3, conf_update_weight: float = 0.5,
               class_vote_smoothing: float = 0.6, class_decay_factor: float = 0.05):
        
        old_center = self.center
        self.center = utils.centers_xyxy(box[np.newaxis, :])[0]
        velocity = self.center - old_center
        self.velocity = 0.7 * self.velocity + 0.3 * velocity 
        
        # --- ENHANCED: Append new position to the history ---
        self.center_history.append(self.center.astype(int))

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
        w = det_conf * smoothing
        self.cls_hist[det_cls] = self.cls_hist.get(det_cls, 0.0) * (1.0 - w) + w
        for k in list(self.cls_hist.keys()):
            if k != det_cls:
                self.cls_hist[k] *= (1.0 - decay * w)
                if self.cls_hist[k] < 1e-4: del self.cls_hist[k]
        if self.cls_hist: self.cls = max(self.cls_hist.items(), key=lambda kv: kv[1])[0]

    def mark_lost(self):
        self.time_since_update += 1
        self.age += 1
        if self.state == "active": self.state = "lost"

    def best_sim(self, emb: np.ndarray) -> float:
        best = float(self.emb @ emb)
        for g_emb in self.gallery[-5:]:
            sim = float(g_emb @ emb)
            if sim > best: best = sim
        return best


class SimpleTracker:
    def __init__(self, **kwargs):
        self.iou_w = kwargs.get('iou_weight', 0.3)
        self.app_w = kwargs.get('app_weight', 0.7)
        self.iou_thresh = kwargs.get('iou_thresh', 0.3)
        self.iou_thresh_low = kwargs.get('iou_thresh_low', 0.2)
        self.reid_sim_thresh = kwargs.get('reid_sim_thresh', 0.6)
        self.max_age = kwargs.get('max_age', 30)
        self.reid_max = kwargs.get('reid_max_age', 60)
        self.ema_alpha = kwargs.get('ema_alpha', 0.9)
        self.gallery_size = kwargs.get('gallery_size', 10)
        self.use_hungarian = kwargs.get('use_hungarian', True) and HAS_HUNGARIAN
        self.class_consistent = kwargs.get('class_consistent', True)
        self.class_penalty = kwargs.get('class_penalty', 0.15)
        self.conf_high = kwargs.get('conf_high', 0.5)
        self.conf_low = kwargs.get('conf_low', 0.1)
        self.conf_min_update = kwargs.get('conf_min_update', 0.3)
        self.conf_update_weight = kwargs.get('conf_update_weight', 0.5)
        self.low_conf_iou_only = kwargs.get('low_conf_iou_only', True)
        self.center_gate_base = kwargs.get('center_gate_base', 50.0)
        self.center_gate_slope = kwargs.get('center_gate_slope', 10.0)
        self.class_vote_smoothing = kwargs.get('class_vote_smoothing', 0.6)
        self.class_decay_factor = kwargs.get('class_decay_factor', 0.05)
        self.motion_gate = kwargs.get('motion_gate', True)
        self.extrapolation_window = kwargs.get('extrapolation_window', 30)
        
        self.tracks: List[Track] = []
        self._next_id = 1
        self.reid_events_this_frame: List[Dict] = []

    def _new_track(self, box: np.ndarray, emb: np.ndarray, cls: Optional[int]) -> Track:
        t = Track(tid=self._next_id, box=box, emb=emb, cls=cls if (cls is not None and cls >= 0) else None,
                  alpha=self.ema_alpha, gallery_max=self.gallery_size)
        if t.cls is not None: t.cls_hist[t.cls] = 1.0
        self._next_id += 1
        self.tracks.append(t)
        return t

    def _prune_removed(self):
        self.tracks = [t for t in self.tracks if t.state != "removed" and 
                       not (t.state == "lost" and t.time_since_update > self.reid_max)]

    def _add_soft_class_penalty(self, C: np.ndarray, act_idx: List[int], det_ids: List[int], clses: Optional[np.ndarray]) -> np.ndarray:
        if clses is None or not self.class_consistent or self.class_penalty <= 0.0 or not act_idx or not det_ids: return C
        track_classes = np.array([self.tracks[ti].cls for ti in act_idx], dtype=object)
        det_classes = clses[det_ids]
        is_valid_track = (track_classes != None)[:, None]
        is_mismatched = track_classes[:, None] != det_classes[None, :]
        penalty_mask = np.logical_and(is_valid_track, is_mismatched)
        return C + self.class_penalty * penalty_mask.astype(np.float32)

    def _associate(self, cost_matrix: np.ndarray, unmatched_dets: Set[int], unmatched_tracks: Set[int],
                   act_idx: List[int], det_ids: List[int], boxes: np.ndarray, embs: np.ndarray,
                   clses: Optional[np.ndarray], confs: Optional[np.ndarray], is_reid: bool = False):
        if self.use_hungarian: rows, cols = linear_sum_assignment(cost_matrix)
        else:
            rows, cols, used_r, used_c = [], [], set(), set()
            for idx in np.argsort(cost_matrix.flatten()):
                r, c = np.unravel_index(idx, cost_matrix.shape)
                if r not in used_r and c not in used_c:
                    rows.append(r); cols.append(c); used_r.add(r); used_c.add(c)
        
        for r, c in zip(rows, cols):
            if cost_matrix[r, c] > 0.99: continue
            ti, j = act_idx[r], det_ids[c]
            track = self.tracks[ti]

            if is_reid:
                event = {"tid": track.tid, "old_box": track.box.copy(), "new_box": boxes[j], "score": track.best_sim(embs[j])}
                self.reid_events_this_frame.append(event)
            
            track.update(boxes[j], embs[j], det_conf=(confs[j] if confs is not None else 1.0), det_cls=(int(clses[j]) if clses is not None else None),
                         conf_min_update=self.conf_min_update, conf_update_weight=self.conf_update_weight,
                         class_vote_smoothing=self.class_vote_smoothing, class_decay_factor=self.class_decay_factor)
            unmatched_dets.discard(j)
            unmatched_tracks.discard(ti)

    def update(self, det_boxes: np.ndarray, det_embs: np.ndarray,
               confs: Optional[np.ndarray] = None, clses: Optional[np.ndarray] = None) -> Tuple[List[Track], List[Dict]]:
        self.reid_events_this_frame.clear()
        N = len(det_boxes)
        # --- Stage 1: Associate ACTIVE tracks ---
        if confs is not None:
            hi_ids, lo_ids = [i for i, c in enumerate(confs) if c >= self.conf_high], [i for i, c in enumerate(confs) if self.conf_low <= c < self.conf_high]
        else: hi_ids, lo_ids = list(range(N)), []
        act_idx = [i for i, t in enumerate(self.tracks) if t.state == "active"]
        unmatched_tracks = set(act_idx)
        unmatched_hi, unmatched_tracks = self._match_active(list(unmatched_tracks), hi_ids, det_boxes, det_embs, clses, confs, use_iou_only=False)
        unmatched_lo, unmatched_tracks = self._match_active(list(unmatched_tracks), lo_ids, det_boxes, det_embs, clses, confs, use_iou_only=self.low_conf_iou_only)
        for ti in unmatched_tracks: self.tracks[ti].mark_lost()
        
        # --- Stage 2: Re-ID LOST tracks ---
        unmatched_dets = unmatched_hi.union(unmatched_lo)
        self._reid_lost(unmatched_dets, det_boxes, det_embs, clses, confs)

        # --- Stage 3: Create new tracks ---
        for j in sorted(list(unmatched_dets)):
            cls = int(clses[j]) if clses is not None else None
            self._new_track(det_boxes[j], det_embs[j], cls)
        self._prune_removed()
        return self.tracks, self.reid_events_this_frame

    def _match_active(self, act_idx, det_ids, boxes, embs, clses, confs, use_iou_only=False):
        unmatched_dets, unmatched_tracks = set(det_ids), set(act_idx)
        if not act_idx or not det_ids: return unmatched_dets, unmatched_tracks
        track_boxes = np.stack([self.tracks[i].box for i in act_idx])
        cost_iou = 1.0 - utils.iou_matrix(track_boxes, boxes[det_ids])
        if use_iou_only:
            cost_matrix = cost_iou; cost_matrix[cost_iou > (1.0 - self.iou_thresh_low)] = 1e6
        else:
            track_embs = np.stack([self.tracks[i].emb for i in act_idx])
            cost_app = utils.cosine_cost_matrix(track_embs, embs[det_ids])
            cost_matrix = self.iou_w * cost_iou + self.app_w * cost_app
            cost_matrix[cost_iou > (1.0 - self.iou_thresh)] = 1e6
        cost_matrix = self._add_soft_class_penalty(cost_matrix, act_idx, det_ids, clses)
        self._associate(cost_matrix, unmatched_dets, unmatched_tracks, act_idx, det_ids, boxes, embs, clses, confs)
        return unmatched_dets, unmatched_tracks

    def _reid_lost(self, unmatched_dets: Set[int], boxes: np.ndarray, embs: np.ndarray,
                   clses: Optional[np.ndarray], confs: Optional[np.ndarray]):
        # --- ENHANCED: Split lost tracks into two groups for staged search ---
        lost_idx = [i for i, t in enumerate(self.tracks) if t.state == "lost"]
        lost_tracks_all = [self.tracks[i] for i in lost_idx]
        lost_gated_idx = [i for i, t in zip(lost_idx, lost_tracks_all) if t.time_since_update <= self.extrapolation_window]
        lost_global_idx = [i for i, t in zip(lost_idx, lost_tracks_all) if t.time_since_update > self.extrapolation_window]
        
        # --- Phase 2a: Gated Re-ID for recently lost tracks ---
        det_left_ids = sorted(list(unmatched_dets))
        if lost_gated_idx and det_left_ids:
            lost_tracks = [self.tracks[i] for i in lost_gated_idx]
            det_boxes, det_embs = boxes[det_left_ids], embs[det_left_ids]
            cost_app = utils.cosine_cost_matrix(np.stack([t.emb for t in lost_tracks]), det_embs)
            cost_matrix = cost_app.copy()
            cost_matrix[cost_app > (1.0 - self.reid_sim_thresh)] = 1e6
            if self.motion_gate:
                time_since = np.array([t.time_since_update for t in lost_tracks])
                pred_centers = np.stack([t.center for t in lost_tracks]) + np.stack([t.velocity for t in lost_tracks]) * time_since[:, np.newaxis]
                det_centers = utils.centers_xyxy(det_boxes)
                dist = np.linalg.norm(pred_centers[:, np.newaxis, :] - det_centers[np.newaxis, :, :], axis=2)
                allowance = self.center_gate_base + self.center_gate_slope * time_since
                cost_matrix[dist > allowance[:, np.newaxis]] = 1e6
            unmatched_lost_gated = set(range(len(lost_gated_idx)))
            self._associate(cost_matrix, unmatched_dets, unmatched_lost_gated, lost_gated_idx, det_left_ids, boxes, embs, clses, confs, is_reid=True)
        
        # --- Phase 2b: Global Re-ID for long-lost tracks ---
        det_left_ids = sorted(list(unmatched_dets))
        if lost_global_idx and det_left_ids:
            lost_tracks = [self.tracks[i] for i in lost_global_idx]
            det_embs = embs[det_left_ids]
            cost_app = utils.cosine_cost_matrix(np.stack([t.emb for t in lost_tracks]), det_embs)
            cost_matrix = cost_app.copy()
            cost_matrix[cost_app > (1.0 - self.reid_sim_thresh)] = 1e6
            unmatched_lost_global = set(range(len(lost_global_idx)))
            self._associate(cost_matrix, unmatched_dets, unmatched_lost_global, lost_global_idx, det_left_ids, boxes, embs, clses, confs, is_reid=True)
