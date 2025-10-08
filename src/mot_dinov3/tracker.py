# src/mot_dinov3/tracker.py (Final Polished Version)
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from collections import deque

import numpy as np

from . import utils

try:
    from scipy.optimize import linear_sum_assignment
    HAS_HUNGARIAN = True
except ImportError:
    HAS_HUNGARIAN = False

LARGE_COST = 1e9

class KalmanFilter:
    """A simple Kalman filter for tracking objects in 2D space with constant velocity."""
    def __init__(self, dt: float = 1.0, std_acc: float = 1.0, x_std_meas: float = 1.0, y_std_meas: float = 1.0):
        self.dt = dt
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.array([[(dt**4)/4, 0, (dt**3)/2, 0], [0, (dt**4)/4, 0, (dt**3)/2],
                           [(dt**3)/2, 0, dt**2, 0], [0, (dt**3)/2, 0, dt**2]]) * std_acc**2
        self.R = np.array([[x_std_meas**2, 0], [0, y_std_meas**2]])
        self.x = np.zeros(4)
        self.P = np.eye(4) * 500.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z: np.ndarray):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S.T, (self.H @ self.P).T).T
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T

@dataclass(slots=True)
class Track:
    tid: int; box: np.ndarray; emb: np.ndarray; cls: Optional[int]; hits: int; age: int
    time_since_update: int; state: str; alpha: float; gallery: deque; cls_hist: Dict[int, float]
    last_conf: float; kf: KalmanFilter; center_history: deque; last_known_crop: Optional[np.ndarray]
    search_radius: float

    @property
    def center(self) -> np.ndarray: return self.kf.x[:2]
    @property
    def velocity(self) -> np.ndarray: return self.kf.x[2:]
    @property
    def is_static(self) -> bool: return np.linalg.norm(self.velocity) < 1.5

    def update(self, box: np.ndarray, emb: np.ndarray, **kwargs):
        self.box = box.astype(np.float32)
        new_center_measurement = utils.centers_xyxy(box[np.newaxis, :])[0]
        self.kf.update(new_center_measurement)
        self.center_history.append(self.center.astype(int))

        det_conf = float(np.clip(kwargs.get('det_conf', 1.0), 0.0, 1.0))
        if det_conf >= kwargs.get('conf_min_update', 0.3):
            beta = (1.0 - self.alpha) * (kwargs.get('conf_update_weight', 0.5) * det_conf + (1.0 - kwargs.get('conf_update_weight', 0.5)))
            x = (1.0 - beta) * self.emb + beta * emb
            self.emb = (x / (np.linalg.norm(x) + 1e-12)).astype(np.float32)
            self.gallery.append(self.emb.copy())
            
            det_cls = kwargs.get('det_cls', None)
            if det_cls is not None and det_cls >= 0:
                self._update_class_hist(det_cls, det_conf, kwargs.get('class_vote_smoothing', 0.6), kwargs.get('class_decay_factor', 0.05))

        self.hits += 1
        self.time_since_update = 0
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
        if self.state == "active":
            self.state = "lost"
            self.time_since_update = 0

class SimpleTracker:
    def __init__(self, **kwargs):
        self.iou_w: float = kwargs.get('iou_weight', 0.3)
        self.app_w: float = kwargs.get('app_weight', 0.7)
        self.iou_thresh: float = kwargs.get('iou_thresh', 0.3)
        self.iou_thresh_low: float = kwargs.get('iou_thresh_low', 0.2)
        self.reid_sim_thresh: float = kwargs.get('reid_sim_thresh', 0.6)
        self.reid_max_age: int = kwargs.get('reid_max_age', 60)
        self.ema_alpha: float = kwargs.get('ema_alpha', 0.9)
        self.gallery_size: int = kwargs.get('gallery_size', 10)
        self.use_hungarian: bool = kwargs.get('use_hungarian', True) and HAS_HUNGARIAN
        self.motion_gate: bool = kwargs.get('motion_gate', True)
        self.extrapolation_window: int = kwargs.get('extrapolation_window', 30)
        self.reid_debug_k: int = kwargs.get('reid_debug_k', 3)
        self.conf_high: float = kwargs.get('conf_high', 0.5)
        self.conf_min_update: float = kwargs.get('conf_min_update', 0.3)
        self.conf_update_weight: float = kwargs.get('conf_update_weight', 0.5)
        self.motion_gate_base: float = kwargs.get('motion_gate_base', 25.0)
        self.motion_gate_vel_factor: float = kwargs.get('motion_gate_vel_factor', 2.0)
        self.motion_gate_min_growth: float = kwargs.get('motion_gate_min_growth', 2.0)
        self.class_consistent: bool = kwargs.get('class_consistent', True)
        self.class_penalty: float = kwargs.get('class_penalty', 0.15)
        self.class_vote_smoothing: float = kwargs.get('class_vote_smoothing', 0.6)
        self.class_decay_factor: float = kwargs.get('class_decay_factor', 0.05)
        
        self.tracks: List[Track] = []
        self._next_id: int = 1
        self.reid_events_this_frame: List[Dict] = []
        self.reid_debug_info: Dict = {}

    def _new_track(self, box: np.ndarray, emb: np.ndarray, cls: Optional[int], frame: np.ndarray, conf: float) -> Track:
        emb_norm = emb.astype(np.float32) / (np.linalg.norm(emb) + 1e-12)
        kf = KalmanFilter()
        kf.x[:2] = utils.centers_xyxy(box[np.newaxis,:])[0].reshape(2)
        
        t = Track(
            tid=self._next_id, box=box.astype(np.float32), emb=emb_norm, cls=cls, hits=1, age=1,
            time_since_update=0, state="active", alpha=self.ema_alpha,
            gallery=deque([emb_norm], maxlen=self.gallery_size), cls_hist={}, last_conf=conf, kf=kf,
            center_history=deque([kf.x[:2].astype(int)], maxlen=25),
            last_known_crop=utils.get_crop(frame, box), search_radius=0.0
        )
        if t.cls is not None: t.cls_hist[t.cls] = 1.0
        self._next_id += 1
        self.tracks.append(t)
        return t

    def _prune_removed(self):
        self.tracks = [t for t in self.tracks if not (t.state == "lost" and t.time_since_update > self.reid_max_age)]

    def _associate(self, cost_matrix: np.ndarray, unmatched_dets: Set[int], unmatched_tracks: Set[int],
                   act_idx: List[int], det_ids: List[int], boxes: np.ndarray, embs: np.ndarray,
                   frame: np.ndarray, clses: Optional[np.ndarray], confs: Optional[np.ndarray], is_reid: bool = False):
        if not act_idx or not det_ids: return
        
        if self.use_hungarian: rows, cols = linear_sum_assignment(cost_matrix)
        else:
            rows, cols, used_r, used_c = [], [], set(), set()
            for idx in np.argsort(cost_matrix.flatten()):
                r, c = np.unravel_index(idx, cost_matrix.shape)
                if r not in used_r and c not in used_c: rows.append(r); cols.append(c); used_r.add(r); used_c.add(c)
        
        for r, c in zip(rows, cols):
            if cost_matrix[r, c] >= LARGE_COST: continue
            ti, j = act_idx[r], det_ids[c]
            track = self.tracks[ti]

            if is_reid:
                event = {"tid": track.tid, "old_box": track.box.copy(), "new_box": boxes[j], "score": 1.0 - cost_matrix[r,c]}
                self.reid_events_this_frame.append(event)
            
            track.update(boxes[j], embs[j], det_conf=(confs[j] if confs is not None else 1.0), 
                         det_cls=(int(clses[j]) if clses is not None else None),
                         conf_min_update=self.conf_min_update, conf_update_weight=self.conf_update_weight,
                         class_vote_smoothing=self.class_vote_smoothing, class_decay_factor=self.class_decay_factor)
            track.last_known_crop = utils.get_crop(frame, boxes[j])
            
            unmatched_dets.discard(j)
            unmatched_tracks.discard(ti)
    
    def _predicted_boxes(self, act_idx: List[int]) -> np.ndarray:
        if not act_idx: return np.empty((0, 4), dtype=np.float32)
        out = []
        for i in act_idx:
            t = self.tracks[i]; last_c = utils.centers_xyxy(t.box[np.newaxis,:])[0]
            delta  = t.center - last_c; x1,y1,x2,y2 = t.box
            out.append(np.array([x1+delta[0], y1+delta[1], x2+delta[0], y2+delta[1]]))
        return np.stack(out).astype(np.float32)

    def update(self, det_boxes: np.ndarray, det_embs: np.ndarray, frame: np.ndarray,
               confs: Optional[np.ndarray] = None, clses: Optional[np.ndarray] = None) -> Tuple[List[Track], List[Dict], Dict]:
        # --- CORRECTED: Use a multi-line loop to correctly update all tracks ---
        for t in self.tracks:
            t.kf.predict()
            t.age += 1
            if t.state == 'lost':
                t.time_since_update += 1
                
        self.reid_events_this_frame.clear(); self.reid_debug_info.clear()

        det_embs_norm = det_embs.astype(np.float32) / (np.linalg.norm(det_embs, axis=1, keepdims=True) + 1e-12)
        act_idx = [i for i, t in enumerate(self.tracks) if t.state == "active"]
        det_ids = list(range(len(det_boxes)))
        
        unmatched_dets, unmatched_tracks = self._match_active(act_idx, det_ids, det_boxes, det_embs_norm, frame, clses, confs)
        for ti in unmatched_tracks: self.tracks[ti].mark_lost()
        
        self._reid_lost(unmatched_dets, det_boxes, det_embs_norm, frame, clses, confs)
        for j in sorted(list(unmatched_dets)):
            self._new_track(det_boxes[j], det_embs_norm[j], (int(clses[j]) if clses is not None else None), frame, (confs[j] if confs is not None else 0.0))
            
        self._prune_removed()
        return self.tracks, self.reid_events_this_frame, self.reid_debug_info

    def _match_active(self, act_idx: List[int], det_ids: List[int], boxes: np.ndarray, embs: np.ndarray, frame: np.ndarray, 
                      clses: Optional[np.ndarray], confs: Optional[np.ndarray]):
        unmatched_dets, unmatched_tracks = set(det_ids), set(act_idx)
        if not act_idx or not det_ids: return unmatched_dets, unmatched_tracks
        
        track_boxes_pred = self._predicted_boxes(act_idx)
        cost_iou = 1.0 - utils.iou_matrix(track_boxes_pred, boxes[det_ids])
        
        track_embs = np.stack([self.tracks[i].emb for i in act_idx])
        cost_app = 1.0 - (track_embs @ embs[det_ids].T)
        cost_matrix = self.iou_w * cost_iou + self.app_w * cost_app

        if self.class_consistent and clses is not None:
            track_clses = np.array([self.tracks[i].cls for i in act_idx])
            det_clses = clses[det_ids]
            is_known = track_clses != None
            if np.any(is_known):
                mismatch = (track_clses[is_known, None] != det_clses[None, :])
                cost_matrix[is_known] += self.class_penalty * mismatch

        if confs is not None:
            det_confs = confs[det_ids]
            is_low = det_confs < self.conf_high; low_cols, high_cols = np.where(is_low)[0], np.where(~is_low)[0]
            if low_cols.size:
                mask = np.zeros_like(cost_matrix, dtype=bool); mask[:, low_cols] = cost_iou[:, low_cols] > (1.0 - self.iou_thresh_low); cost_matrix[mask] = LARGE_COST
            if high_cols.size:
                mask = np.zeros_like(cost_matrix, dtype=bool); mask[:, high_cols] = cost_iou[:, high_cols] > (1.0 - self.iou_thresh); cost_matrix[mask] = LARGE_COST
        else: cost_matrix[cost_iou > (1.0 - self.iou_thresh)] = LARGE_COST
            
        self._associate(cost_matrix, unmatched_dets, unmatched_tracks, act_idx, det_ids, boxes, embs, frame, clses, confs)
        return unmatched_dets, unmatched_tracks

    def _reid_lost(self, unmatched_dets: Set[int], boxes: np.ndarray, embs: np.ndarray, frame: np.ndarray,
                   clses: Optional[np.ndarray], confs: Optional[np.ndarray]):
        if not unmatched_dets: return
        lost_idx = [i for i, t in enumerate(self.tracks) if t.state == "lost"]
        if not lost_idx: return
        
        lost_tracks_all = [self.tracks[i] for i in lost_idx]
        lost_gated_idx = [i for i,t in zip(lost_idx,lost_tracks_all) if t.time_since_update <= self.extrapolation_window]
        lost_global_idx = [i for i,t in zip(lost_idx,lost_tracks_all) if t.time_since_update > self.extrapolation_window]
        
        det_left = sorted(list(unmatched_dets))
        if lost_gated_idx and det_left: self._process_reid_group(lost_gated_idx, det_left, unmatched_dets, boxes, embs, frame, clses, confs, True)
        det_left = sorted(list(unmatched_dets))
        if lost_global_idx and det_left: self._process_reid_group(lost_global_idx, det_left, unmatched_dets, boxes, embs, frame, clses, confs, False)

    def _process_reid_group(self, lost_idx_group: List[int], det_ids: List[int], unmatched_dets: Set[int],
                            boxes: np.ndarray, embs: np.ndarray, frame: np.ndarray, clses: Optional[np.ndarray],
                            confs: Optional[np.ndarray], use_motion_gating: bool):
        lost_tracks = [self.tracks[i] for i in lost_idx_group]
        track_embs = np.stack([t.emb for t in lost_tracks])
        det_embs_subset = embs[det_ids]
        cost_app = 1.0 - (track_embs @ det_embs_subset.T)

        if self.reid_debug_k > 0:
            for i, track in enumerate(lost_tracks):
                if track.tid in self.reid_debug_info: continue
                sim_scores = 1.0 - cost_app[i, :]
                top_k_indices = np.argsort(sim_scores)[::-1][:self.reid_debug_k]
                candidates = [{'box': boxes[det_ids[j]], 'score': sim_scores[j], 'crop': utils.get_crop(frame, boxes[det_ids[j]])} 
                              for j in top_k_indices if sim_scores[j] >= self.reid_sim_thresh * 0.5]
                if candidates: self.reid_debug_info[track.tid] = {'query_crop': track.last_known_crop, 'candidates': candidates}

        cost_matrix = cost_app.copy()
        cost_matrix[cost_app > (1.0 - self.reid_sim_thresh)] = LARGE_COST
        
        if use_motion_gating and self.motion_gate:
            pred_centers = np.stack([t.center for t in lost_tracks])
            det_centers = utils.centers_xyxy(boxes[det_ids])
            dist = np.linalg.norm(pred_centers[:, np.newaxis, :] - det_centers[np.newaxis, :, :], axis=2)
            
            time_since = np.array([t.time_since_update for t in lost_tracks])
            velocities = np.stack([t.velocity for t in lost_tracks])
            velocity_magnitudes = np.linalg.norm(velocities, axis=1)
            growth_rate_slope = (velocity_magnitudes * self.motion_gate_vel_factor) + self.motion_gate_min_growth
            allowance = self.motion_gate_base + growth_rate_slope * time_since
            allowance = np.clip(allowance, 10.0, 1000.0)

            for i, track in enumerate(lost_tracks): track.search_radius = allowance[i]
            cost_matrix[dist > allowance[:, np.newaxis]] = LARGE_COST

        self._associate(cost_matrix, unmatched_dets, set(lost_idx_group), lost_idx_group, det_ids, boxes, embs, frame, clses, confs, is_reid=True)