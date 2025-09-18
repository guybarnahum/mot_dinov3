# src/mot_dinov3/viz.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from collections import deque

import cv2
import numpy as np

# ---------- Color helpers ----------

def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    """h,s,v in [0,1] -> r,g,b in [0,1]"""
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1.0 - s), v * (1.0 - f * s), v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0: r, g, b = v, t, p
    elif i == 1: r, g, b = q, v, p
    elif i == 2: r, g, b = p, v, t
    elif i == 3: r, g, b = p, q, v
    elif i == 4: r, g, b = t, p, v
    else: r, g, b = v, p, q
    return r, g, b

def _id_to_color(tid: int, desaturate: bool = False) -> Tuple[int, int, int]:
    """Deterministic vivid color from an integer track id. Returns BGR."""
    phi = 0.618033988749895
    h = (tid * phi) % 1.0
    s, v = (0.4, 0.95) if desaturate else (0.9, 1.0)
    r, g, b = _hsv_to_rgb(h, s, v)
    return int(b * 255), int(g * 255), int(r * 255)

# ---------- Drawing helpers ----------

def _draw_label(img: np.ndarray, text: str, pos: Tuple[int, int], color: Tuple[int, int, int],
                font_scale: float = 0.5, thickness: int = 1,
                bg_color: Optional[Tuple[int, int, int]] = None) -> None:
    """Draw a filled label box with text."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    pad = 3
    bg_color = bg_color if bg_color is not None else color
    
    cv2.rectangle(img, (x, y - th - 2 * pad), (x + tw + 2 * pad, y), bg_color, -1)
    cv2.putText(img, text, (x + pad, y - pad), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def _draw_dashed_rect(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                      color: Tuple[int, int, int], thickness: int = 1, gap: int = 10):
    """Draws a dashed rectangle."""
    x1, y1 = pt1; x2, y2 = pt2
    for x in range(x1, x2, gap * 2):
        cv2.line(img, (x, y1), (min(x + gap, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + gap, x2), y2), color, thickness)
    for y in range(y1, y2, gap * 2):
        cv2.line(img, (x1, y), (x1, min(y + gap, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + gap, y2)), color, thickness)

# ---------- Public API ----------

def draw_tracks(
    frame: np.ndarray,
    tracks: list,
    draw_lost: bool = False,
    thickness: int = 2,
    font_scale: float = 0.5,
    viz_info: Optional[Dict[int, str]] = None,
    viz_trails: bool = False,
    viz_prediction: bool = False
) -> None:
    """Draws track boxes, labels, and optional trails/predictions."""
    if not tracks: return
    H, W = frame.shape[:2]

    for t in tracks:
        state = getattr(t, "state", "active")
        if state != "active" and not draw_lost: continue

        x1, y1, x2, y2 = np.clip(t.box, [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1]).astype(int)
        tid = int(t.tid)
        is_lost = state != "active"
        color = _id_to_color(tid, desaturate=is_lost)

        # --- ENHANCED: Draw trails for active tracks ---
        if viz_trails and not is_lost and hasattr(t, 'center_history') and len(t.center_history) > 1:
            points = np.array(list(t.center_history), dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=thickness // 2, lineType=cv2.LINE_AA)

        if is_lost:
            _draw_dashed_rect(frame, (x1, y1), (x2, y2), color, thickness)
            # --- ENHANCED: Draw predicted path for lost tracks ---
            if viz_prediction and hasattr(t, 'velocity'):
                pred_center = (t.center + t.velocity * 5).astype(int) # Predict 5 frames ahead
                cv2.arrowedLine(frame, tuple(t.center.astype(int)), tuple(pred_center),
                                color, thickness, tipLength=0.3)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        label_parts = [f"ID {tid}"]
        if viz_info and (reason := viz_info.get(tid)):
            label_parts[0] += f" [{reason}]"
        if hasattr(t, "cls") and t.cls is not None:
            label_parts.append(f"c{int(t.cls)}")
        if is_lost:
            label_parts.append(f"({state})")
        
        label = " | ".join(label_parts)
        _draw_label(frame, label, (x1, y1), color, font_scale=font_scale, thickness=1)

def draw_hud(frame: np.ndarray, stats: Dict):
    """Draws a Heads-Up Display with system statistics."""
    font, font_scale, thickness, color = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, (255, 255, 255)
    x, y, line_h = 10, 20, 18
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    def draw_text(text, val=""):
        nonlocal y
        full_text = f"{text}: {val}" if val else text
        cv2.putText(frame, full_text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_h

    draw_text("Frame", stats.get('Frame', 'N/A'))
    draw_text("FPS", f"{stats.get('FPS', 0):.1f}")
    if 'Scheduler' in stats:
        y += 5; draw_text("-- Scheduler --")
        draw_text("  Budget", stats['Scheduler'].get('Budget', 'N/A'))
        draw_text("  Backlog", stats['Scheduler'].get('Backlog', 'N/A'))
        draw_text("  Actions", stats['Scheduler'].get('Actions', 'N/A'))
    if 'Tracker' in stats:
        y += 5; draw_text("-- Tracker --")
        draw_text("  Active", stats['Tracker'].get('Active', 'N/A'))
        draw_text("  Lost", stats['Tracker'].get('Lost', 'N/A'))

def draw_reid_links(frame: np.ndarray, reid_events: List[Dict]):
    """Draws enhanced visual links for Re-ID events."""
    for event in reid_events:
        new_box, old_box = event['new_box'].astype(int), event['old_box'].astype(int)
        tid = int(event['tid'])
        c_new = (int(new_box[0] + new_box[2]) // 2, int(new_box[1] + new_box[3]) // 2)
        c_old = (int(old_box[0] + old_box[2]) // 2, int(old_box[1] + old_box[3]) // 2)

        color = _id_to_color(tid, desaturate=False)
        # --- ENHANCED: Draw a thicker line and a "flash" circle ---
        cv2.line(frame, c_old, c_new, color, 3, cv2.LINE_AA)
        cv2.circle(frame, c_new, 8, (255, 255, 255), -1, cv2.LINE_AA) # White flash
        cv2.circle(frame, c_new, 6, color, -1, cv2.LINE_AA)

        score = event['score']
        label = f"ID {tid} [Re-ID: {score:.2f}]"
        # --- ENHANCED: Use a bright, distinct background for the Re-ID label ---
        _draw_label(frame, label, (new_box[0], new_box[1]), color, font_scale=0.5, bg_color=(255, 255, 255))
