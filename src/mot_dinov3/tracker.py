from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from .utils import iou_matrix, cosine_sim_matrix

try:
    from scipy.optimize import linear_sum_assignment
    HAS_HUNGARIAN = True
except Exception:
    HAS_HUNGARIAN = False

@dataclass
class Track:
    tid: int
    box: np.ndarray    # (4,) xyxy
    emb: np.ndarray    # (D,)
    age: int = 0
    misses: int = 0

class SimpleTracker:
    """
    IoU + appearance tracker with EMA embedding update.
    Use greedy or Hungarian assignment with combined cost.
    """
    def __init__(self, iou_w=0.5, app_w=0.5, iou_thresh=0.1, app_thresh=0.2,
                 max_misses=30, emb_momentum=0.9, use_hungarian: bool = True):
        self.tracks: List[Track] = []
        self.next_id = 1
        self.iou_w = iou_w
        self.app_w = app_w
        self.iou_thresh = iou_thresh
        self.app_thresh = app_thresh
        self.max_misses = max_misses
        self.emb_momentum = emb_momentum
        self.use_hungarian = use_hungarian and HAS_HUNGARIAN

    def _assign(self, IoU: np.ndarray, COS: np.ndarray) -> List[Tuple[int,int]]:
        T, D = IoU.shape
        if T == 0 or D == 0:
            return []
        cost = (1.0 - IoU) * self.iou_w + (1.0 - COS) * self.app_w  # lower is better

        if self.use_hungarian:
            r, c = linear_sum_assignment(cost)
            pairs = []
            for tr, de in zip(r, c):
                if IoU[tr, de] >= self.iou_thresh or COS[tr, de] >= self.app_thresh:
                    pairs.append((tr, de))
            return pairs

        # Greedy fallback
        pairs = []
        flat = [(cost[t, d], t, d) for t in range(T) for d in range(D)]
        flat.sort(key=lambda x: x[0])
        used_t, used_d = set(), set()
        for _, t, d in flat:
            if t in used_t or d in used_d:
                continue
            if IoU[t, d] < self.iou_thresh and COS[t, d] < self.app_thresh:
                continue
            pairs.append((t, d))
            used_t.add(t); used_d.add(d)
        return pairs

    def update(self, det_boxes: np.ndarray, det_embs: np.ndarray):
        T = len(self.tracks)
        D = len(det_boxes)

        assigned_tracks = set()
        assigned_dets = set()

        if T > 0 and D > 0:
            track_boxes = np.stack([t.box for t in self.tracks], axis=0)
            track_embs  = np.stack([t.emb for t in self.tracks], axis=0)
            IoU = iou_matrix(track_boxes, det_boxes)
            COS = cosine_sim_matrix(track_embs, det_embs)

            for t_idx, d_idx in self._assign(IoU, COS):
                assigned_tracks.add(t_idx)
                assigned_dets.add(d_idx)
                t = self.tracks[t_idx]
                t.box = det_boxes[d_idx]
                # EMA on embedding
                e = self.emb_momentum * t.emb + (1 - self.emb_momentum) * det_embs[d_idx]
                t.emb = e / (np.linalg.norm(e) + 1e-8)
                t.age += 1
                t.misses = 0

        # new tracks
        for d_idx in range(D):
            if d_idx not in assigned_dets:
                self.tracks.append(Track(tid=self.next_id, box=det_boxes[d_idx],
                                         emb=det_embs[d_idx], age=1, misses=0))
                self.next_id += 1

        # age & prune
        for i, t in enumerate(self.tracks):
            if i not in assigned_tracks:
                t.misses += 1
                t.age += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        return self.tracks

