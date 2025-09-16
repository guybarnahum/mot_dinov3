# src/mot_dinov3/viz.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------- Color helpers ----------

def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    """h,s,v in [0,1] -> r,g,b in [0,1]  (small, fast, no dependency)"""
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0: r, g, b = v, t, p
    elif i == 1: r, g, b = q, v, p
    elif i == 2: r, g, b = p, v, t
    elif i == 3: r, g, b = p, q, v
    elif i == 4: r, g, b = t, p, v
    else: r, g, b = v, p, q
    return r, g, b

def _id_to_color(tid: int, desaturate: bool = False) -> Tuple[int, int, int]:
    """
    Deterministic vivid color from an integer track id.
    Uses golden-ratio hue hop for well-spaced colors. Returns BGR for OpenCV.
    """
    phi = 0.618033988749895  # golden ratio conjugate
    h = (tid * phi) % 1.0
    s, v = (0.4, 0.95) if desaturate else (0.9, 1.0) # Desaturate for lost tracks
    r, g, b = _hsv_to_rgb(h, s, v)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR

# ---------- Drawing helpers ----------

def _draw_label(img: np.ndarray, x: int, y: int, text: str,
                color_bgr: Tuple[int, int, int],
                font_scale: float = 0.5, thickness: int = 1) -> None:
    """Draw a filled label box with text at the top-left corner (x,y) of a bbox."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x1, y1 = int(x), int(y)
    pad = 3
    # Background rectangle clamped to image bounds
    y_bg_top = max(0, y1 - th - 2 * pad)
    y_bg_bot = max(0, y1)
    x_bg_left = max(0, x1)
    x_bg_right = min(img.shape[1], x1 + tw + 2 * pad)
    cv2.rectangle(img, (x_bg_left, y_bg_top), (x_bg_right, y_bg_bot), color_bgr, -1)
    # White text on colored bg is often more readable
    cv2.putText(img, text, (x_bg_left + pad, y_bg_bot - pad),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def _draw_dashed_rect(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                      color: Tuple[int, int, int], thickness: int = 1, gap: int = 10):
    """Draws a dashed rectangle, as OpenCV doesn't have a native function."""
    x1, y1 = pt1
    x2, y2 = pt2
    # Top and bottom edges
    for x in range(x1, x2, gap * 2):
        cv2.line(img, (x, y1), (min(x + gap, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + gap, x2), y2), color, thickness)
    # Left and right edges
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
    viz_info: Optional[Dict[int, str]] = None
) -> None:
    """
    Draws track boxes and labels directly onto the frame (in-place).

    Args:
        frame: The image frame to draw on (modified in-place).
        tracks: A list of track objects to visualize.
        draw_lost: If True, also draws tracks marked as 'LOST'.
        thickness: Line thickness for the bounding box.
        font_scale: Font scale for the label text.
        viz_info: Dict mapping track ID to a scheduler reason code (e.g., 'C-crowd').
    """
    if not tracks:
        return

    H, W = frame.shape[:2]

    for t in tracks:
        state = getattr(t, "state", "active")
        if state != "active" and not draw_lost:
            continue

        x1, y1, x2, y2 = np.clip(t.box, [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1]).astype(int)
        tid = int(t.tid)
        
        is_lost = state != "active"
        color = _id_to_color(tid, desaturate=is_lost)

        if is_lost:
            _draw_dashed_rect(frame, (x1, y1), (x2, y2), color, thickness)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        # Build the label string, incorporating the scheduler's reason code
        label_parts = [f"ID {tid}"]
        if viz_info and (reason := viz_info.get(tid)):
            label_parts[0] += f" [{reason}]" # e.g., "ID 42 [C-crowd]"
        
        if hasattr(t, "cls") and t.cls is not None:
            label_parts.append(f"c{int(t.cls)}")
        if is_lost:
            label_parts.append(f"({state})")
        
        label = " | ".join(label_parts)
        _draw_label(frame, x1, y1, label, color, font_scale=font_scale, thickness=1)


def draw_hud(frame: np.ndarray, stats: Dict):
    """
    Draws a Heads-Up Display with system statistics on the frame.

    Args:
        frame: The image frame to draw on (modified in-place).
        stats: A dictionary containing statistics to display.
               Example: {'Frame': 125, 'FPS': 28.5, 
                         'Scheduler': {'Budget': '15/20ms', 'Backlog': 7}}
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_color = (255, 255, 255)
    x, y, line_h = 10, 20, 18
    
    # Create a semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    def draw_text(text, val=""):
        nonlocal y
        full_text = f"{text}: {val}" if val else text
        cv2.putText(frame, full_text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        y += line_h

    draw_text("Frame", stats.get('Frame', 'N/A'))
    draw_text("FPS", f"{stats.get('FPS', 0):.1f}")
    y += 5 # Add a small gap

    if 'Scheduler' in stats:
        draw_text("-- Scheduler --")
        draw_text("  Budget", stats['Scheduler'].get('Budget', 'N/A'))
        draw_text("  Backlog", stats['Scheduler'].get('Backlog', 'N/A'))
        draw_text("  Actions", stats['Scheduler'].get('Actions', 'N/A'))
    
    if 'Tracker' in stats:
        y+= 5
        draw_text("-- Tracker --")
        draw_text("  Active", stats['Tracker'].get('Active', 'N/A'))
        draw_text("  Lost", stats['Tracker'].get('Lost', 'N/A'))


def draw_reid_links(frame: np.ndarray, reid_events: List[Dict]):
    """
    Draws visual links for Re-ID events for a single frame.

    Args:
        frame: The image frame to draw on.
        reid_events: List of dicts, where each dict contains info about a
                     re-identification, e.g., {'tid', 'old_box', 'new_box', 'score'}.
    """
    for event in reid_events:
        new_box = event['new_box'].astype(int)
        old_box = event['old_box'].astype(int)
        tid = int(event['tid'])
        
        c_new = (int(new_box[0] + new_box[2]) // 2, int(new_box[1] + new_box[3]) // 2)
        c_old = (int(old_box[0] + old_box[2]) // 2, int(old_box[1] + old_box[3]) // 2)

        # Draw a bright line connecting the old and new positions
        color = _id_to_color(tid, desaturate=False)
        cv2.line(frame, c_old, c_new, color, 2, cv2.LINE_AA)
        cv2.circle(frame, c_new, 5, color, -1)

        # Draw a special label for the Re-ID event
        score = event['score']
        label = f"ID {tid} [Re-ID: {score:.2f}]"
        _draw_label(frame, new_box[0], new_box[1], label, color, font_scale=0.5)