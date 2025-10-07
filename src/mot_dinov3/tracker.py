# src/mot_dinov3/tracker.py (Refactored with Kalman Filter)
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

# --- NEW: A self-contained Linear Kalman Filter for motion tracking ---
class KalmanFilter:
    """
    A simple Kalman filter for tracking objects in 2D space with constant velocity.
    The state is [x, y, vx, vy].
    The measurement is [x, y].
    """
    def __init__(self, dt: float = 1.0, std_acc: float = 1.0, x_std_meas: float = 1.0, y_std_meas: float = 1.0):
        self.dt = dt
        # State transition matrix
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        # Process noise covariance (models acceleration uncertainty)
        self.Q = np.array([[(dt**4)/4, 0, (dt**3)/2, 0],
                           [0, (dt**4)/4, 0, (dt**3)/2],
                           [(dt**3)/2, 0, dt**2, 0],
                           [0, (dt**3)/2, 0, dt**2]]) * std_acc**2
        # Measurement noise covariance (models sensor uncertainty)
        self.R = np.array([[x_std_meas**2, 0],
                           [0, y_std_meas**2]])
        # Initial state estimate [x, y, vx, vy] and covariance
        self.x = np.zeros(4)
        self.P = np.eye(4) * 500.0

    def predict(self):
        """Predicts the next state."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z: np.ndarray):
        """Updates the state with a new measurement z."""
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(4)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

@dataclass
class Track:
    """A class to represent a single tracked object, now using a Kalman Filter."""
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
    center_history: deque = field(default_factory=lambda: deque(maxlen=25), repr=False)
    last_known_crop: Optional[np.ndarray] = field(default=None, repr=False)
    
    # --- MODIFIED: Kalman Filter now manages the track's motion state ---
    kf: KalmanFilter = field(default_factory=KalmanFilter, repr=False)

    def __post_init__(self):
        """Initialize the Kalman Filter with the first detection."""
        initial_center = utils.centers_xyxy(self.box[np.newaxis, :])[0]
        self.kf.x[:2] = initial_center.reshape(2)
        self.center_history.append(initial_center.astype(int))

    # --- MODIFIED: Center and velocity are now properties derived from the Kalman Filter's state ---
    @property
    def center(self) -> np.ndarray:
        return self.kf.x[:2]
    
    @property
    def velocity(self) -> np.ndarray:
        return self.kf.x[2:]

    def update(self, box: np.ndarray, emb: np.ndarray,
               det_conf: float = 1.0, det_cls: Optional[int] = None,
               conf_min_update: float = 0.3, conf_update_weight: float = 0.5,
               class_vote_smoothing: float = 0.6, class_decay_factor: float = 0.05):
        """Updates the track's state with a new detection."""
        self.box = box.astype(np.float32)
        
        # Feed the new bounding box center measurement to the Kalman Filter
        new_center_measurement = utils.centers_xyxy(box[np.newaxis, :])[0]
        self.kf.update(new_center_measurement)
        
        # The center property will now reflect the smoothed state
        self.center_history.append(self.center.astype(int))

        # Update appearance embedding with EMA
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
        """Marks the track as lost."""
        if self.state == "active":
            self.state = "lost"
            self.time_since_update = 1

    def best_sim(self, emb: np.ndarray) -> float:
        best = float(self.emb @ emb)
        for g_emb in self.gallery[-5:]:
            sim = float(g_emb @ emb)
            if sim > best: best = sim
        return best

    @property
    def is_static(self) -> bool:
        """Returns True if the track's velocity is below a threshold."""
        STATIC_VELOCITY_THRESHOLD = 1.5 
        return np.linalg.norm(self.velocity) < STATIC_VELOCITY_THRESHOLD

class SimpleTracker:
    def __init__(self, **kwargs):
        self.iou_w = kwargs.get('iou_weight', 0.3)
        self.app_w = kwargs.get('app_weight', 0.7)
        self.iou_thresh = kwargs.get('iou_thresh', 0.3)
        self.iou_thresh_low = kwargs.get('iou_thresh_low', 0.2)
        self.reid_sim_thresh = kwargs.get('reid_sim_thresh', 0.6)
        self.reid_max_age = kwargs.get('reid_max_age', 60)
        self.ema_alpha = kwargs.get('ema_alpha', 0.9)
        self.gallery_size = kwargs.get('gallery_size', 10)
        self.use_hungarian = kwargs.get('use_hungarian', True) and HAS_HUNGARIAN
        self.class_consistent = kwargs.get('class_consistent', True)
        self.class_penalty = kwargs.get('class_penalty', 0.15)
        self.conf_high = kwargs.get('conf_high', 0.5)
        self.conf_min_update = kwargs.get('conf_min_update', 0.3)
        self.conf_update_weight = kwargs.get('conf_update_weight', 0.5)
        self.low_conf_iou_only = kwargs.get('low_conf_iou_only', True)
        self.center_gate_base = kwargs.get('center_gate_base', 50.0)
        self.center_gate_slope = kwargs.get('center_gate_slope', 10.0)
        self.class_vote_smoothing = kwargs.get('class_vote_smoothing', 0.6)
        self.class_decay_factor = kwargs.get('class_decay_factor', 0.05)
        self.motion_gate = kwargs.get('motion_gate', True)
        self.extrapolation_window = kwargs.get('extrapolation_window', 30)
        self.reid_debug_k = kwargs.get('reid_debug_k', 3)
        
        self.tracks: List[Track] = []
        self._next_id = 1
        self.reid_events_this_frame: List[Dict] = []
        self.reid_debug_info: Dict = {}

    def _new_track(self, box: np.ndarray, emb: np.ndarray, cls: Optional[int], frame: np.ndarray) -> Track:
        t = Track(tid=self._next_id, box=box, emb=emb, cls=cls if (cls is not None and cls >= 0) else None,
                  alpha=self.ema_alpha, gallery_max=self.gallery_size)
        t.last_known_crop = utils.get_crop(frame, box)
        if t.cls is not None: t.cls_hist[t.cls] = 1.0
        self._next_id += 1
        self.tracks.append(t)
        return t

    def _prune_removed(self):
        self.tracks = [t for t in self.tracks if not (t.state == "lost" and t.time_since_update > self.reid_max_age)]

    def _associate(self, cost_matrix: np.ndarray, unmatched_dets: Set[int], unmatched_tracks: Set[int],
                   act_idx: List[int], det_ids: List[int], boxes: np.ndarray, embs: np.ndarray,
                   frame: np.ndarray, clses: Optional[np.ndarray], confs: Optional[np.ndarray], is_reid: bool = False):
        if self.use_hungarian:
            rows, cols = linear_sum_assignment(cost_matrix)
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
            
            track.update(boxes[j], embs[j], det_conf=(confs[j] if confs is not None else 1.0), 
                         det_cls=(int(clses[j]) if clses is not None else None),
                         conf_min_update=self.conf_min_update, conf_update_weight=self.conf_update_weight,
                         class_vote_smoothing=self.class_vote_smoothing, class_decay_factor=self.class_decay_factor)
            track.last_known_crop = utils.get_crop(frame, boxes[j])
            
            unmatched_dets.discard(j)
            unmatched_tracks.discard(ti)

    def update(self, det_boxes: np.ndarray, det_embs: np.ndarray, frame: np.ndarray,
               confs: Optional[np.ndarray] = None, clses: Optional[np.ndarray] = None) -> Tuple[List[Track], List[Dict], Dict]:
        """The main entry point for updating the tracker state with a new frame."""
        # --- MODIFIED: The "predict" step now updates the Kalman Filter state for each track ---
        for t in self.tracks:
            t.kf.predict() # Predict next state based on the KF's motion model
            t.age += 1
            if t.state == 'lost':
                t.time_since_update += 1

        self.reid_events_this_frame.clear()
        self.reid_debug_info.clear()
        N = len(det_boxes)

        act_idx = [i for i, t in enumerate(self.tracks) if t.state == "active"]
        det_ids = list(range(N))
        
        unmatched_dets, unmatched_tracks = self._match_active(
            act_idx, det_ids, det_boxes, det_embs, frame, clses, confs
        )
        
        for ti in unmatched_tracks:
            self.tracks[ti].mark_lost()
        
        self._reid_lost(unmatched_dets, det_boxes, det_embs, frame, clses, confs)

        for j in sorted(list(unmatched_dets)):
            cls = int(clses[j]) if clses is not None else None
            self._new_track(det_boxes[j], det_embs[j], cls, frame)
            
        self._prune_removed()
        return self.tracks, self.reid_events_this_frame, self.reid_debug_info

    def _match_active(self, act_idx: List[int], det_ids: List[int], boxes: np.ndarray, embs: np.ndarray, frame: np.ndarray, 
                      clses: Optional[np.ndarray], confs: Optional[np.ndarray]):
        unmatched_dets, unmatched_tracks = set(det_ids), set(act_idx)
        if not act_idx or not det_ids:
            return unmatched_dets, unmatched_tracks
        
        # --- MODIFIED: Get predicted boxes from Kalman Filter state ---
        track_boxes = np.stack([self.tracks[i].box for i in act_idx]) # IoU is still based on last known box
        cost_iou = 1.0 - utils.iou_matrix(track_boxes, boxes[det_ids])
        
        track_embs = np.stack([self.tracks[i].emb for i in act_idx])
        cost_app = utils.cosine_cost_matrix(track_embs, embs[det_ids])
        
        cost_matrix = self.iou_w * cost_iou + self.app_w * cost_app

        if confs is not None:
            det_confs = confs[det_ids]
            is_low_conf = det_confs < self.conf_high
            low_conf_iou_mask = cost_iou[:, is_low_conf] > (1.0 - self.iou_thresh_low)
            cost_matrix[:, is_low_conf][low_conf_iou_mask] = 1e6
            high_conf_iou_mask = cost_iou[:, ~is_low_conf] > (1.0 - self.iou_thresh)
            cost_matrix[:, ~is_low_conf][high_conf_iou_mask] = 1e6
        else:
            cost_matrix[cost_iou > (1.0 - self.iou_thresh)] = 1e6
            
        self._associate(cost_matrix, unmatched_dets, unmatched_tracks, act_idx, det_ids, boxes, embs, frame, clses, confs)
        return unmatched_dets, unmatched_tracks

    def _reid_lost(self, unmatched_dets: Set[int], boxes: np.ndarray, embs: np.ndarray, frame: np.ndarray,
                   clses: Optional[np.ndarray], confs: Optional[np.ndarray]):
        if not unmatched_dets: return
        lost_idx = [i for i, t in enumerate(self.tracks) if t.state == "lost"]
        if not lost_idx: return
        
        lost_tracks_all = [self.tracks[i] for i in lost_idx]
        lost_gated_idx = [i for i, t in zip(lost_idx, lost_tracks_all) if t.time_since_update <= self.extrapolation_window]
        lost_global_idx = [i for i, t in zip(lost_idx, lost_tracks_all) if t.time_since_update > self.extrapolation_window]
        
        det_left_ids = sorted(list(unmatched_dets))
        if lost_gated_idx and det_left_ids:
            self._process_reid_group(lost_gated_idx, det_left_ids, unmatched_dets, boxes, embs, frame, clses, confs, use_motion_gating=True)

        det_left_ids = sorted(list(unmatched_dets))
        if lost_global_idx and det_left_ids:
            self._process_reid_group(lost_global_idx, det_left_ids, unmatched_dets, boxes, embs, frame, clses, confs, use_motion_gating=False)

    def _process_reid_group(self, lost_idx_group: List[int], det_ids: List[int], unmatched_dets: Set[int],
                            boxes: np.ndarray, embs: np.ndarray, frame: np.ndarray, clses: Optional[np.ndarray],
                            confs: Optional[np.ndarray], use_motion_gating: bool):
        lost_tracks = [self.tracks[i] for i in lost_idx_group]
        det_embs_subset = embs[det_ids]
        cost_app = utils.cosine_cost_matrix(np.stack([t.emb for t in lost_tracks]), det_embs_subset)
        
        if self.reid_debug_k > 0:
            for i, track in enumerate(lost_tracks):
                if track.tid in self.reid_debug_info: continue
                sim_scores = 1.0 - cost_app[i, :]
                top_k_indices = np.argsort(sim_scores)[::-1][:self.reid_debug_k]
                
                candidates = []
                for det_idx_in_subset in top_k_indices:
                    if sim_scores[det_idx_in_subset] < self.reid_sim_thresh * 0.5: continue
                    original_det_idx = det_ids[det_idx_in_subset]
                    candidate_box = boxes[original_det_idx]
                    candidates.append({
                        'box': candidate_box, 
                        'score': sim_scores[det_idx_in_subset],
                        'crop': utils.get_crop(frame, candidate_box)
                    })
                
                if candidates:
                    self.reid_debug_info[track.tid] = {
                        'query_crop': track.last_known_crop,
                        'candidates': candidates
                    }

        cost_matrix = cost_app.copy()
        cost_matrix[cost_app > (1.0 - self.reid_sim_thresh)] = 1e6
        
        if use_motion_gating and self.motion_gate:
            # --- MODIFIED: Predictions now come directly from the Kalman Filter's predicted state ---
            # The 'center' property of a track now refers to the KF's predicted position for this frame.
            pred_centers = np.stack([t.center for t in lost_tracks])
            det_centers = utils.centers_xyxy(boxes[det_ids])
            dist = np.linalg.norm(pred_centers[:, np.newaxis, :] - det_centers[np.newaxis, :, :], axis=2)
            
            time_since = np.array([t.time_since_update for t in lost_tracks])
            allowance = self.center_gate_base + self.center_gate_slope * time_since
            cost_matrix[dist > allowance[:, np.newaxis]] = 1e6

        unmatched_lost = set(range(len(lost_idx_group)))
        self._associate(cost_matrix, unmatched_dets, unmatched_lost, lost_idx_group, det_ids, boxes, embs, frame, clses, confs, is_reid=True)