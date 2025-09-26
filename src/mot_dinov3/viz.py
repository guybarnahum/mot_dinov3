# src/mot_dinov3/viz.py (Enhanced with Debug Panels)
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
    """Fallback deterministic color from an integer track id. Returns BGR."""
    phi = 0.618033988749895
    h = (tid * phi) % 1.0
    s, v = (0.4, 0.95) if desaturate else (0.9, 1.0)
    r, g, b = _hsv_to_rgb(h, s, v)
    return int(b * 255), int(g * 255), int(r * 255)

def _state_to_color(track) -> Tuple[int, int, int]:
    """Returns a BGR color based on the track's state."""
    RECENT_LOSS_THRESHOLD = 30  # Should align with tracker's extrapolation_window
    if track.state == "active":
        return (200, 120, 0) if track.is_static else (0, 200, 50)  # Blue for Static, Green for Dynamic
    elif track.state == "lost":
        if track.time_since_update <= RECENT_LOSS_THRESHOLD:
            return (0, 165, 255)  # Orange for Recently Lost
        else:
            return (0, 0, 220)  # Red for Long-Term Lost
    return (255, 255, 255) # White for any other state

# ---------- Drawing helpers ----------

def _draw_label(img: np.ndarray, text: str, pos: Tuple[int, int], color: Tuple[int, int, int],
                font_scale: float = 0.5, thickness: int = 1,
                bg_color: Optional[Tuple[int, int, int]] = None) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    pad = 3
    bg_color = bg_color if bg_color is not None else color
    cv2.rectangle(img, (x, y - th - 2 * pad), (x + tw + 2 * pad, y), bg_color, -1)
    cv2.putText(img, text, (x + pad, y - pad), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def _resize_crop(crop: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resizes a crop to a target size, preserving aspect ratio by padding."""
    if crop is None or crop.size == 0:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    target_w, target_h = size
    h, w, _ = crop.shape
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas

# ---------- Panel Drawing Functions ----------

def _draw_lost_panel(canvas: np.ndarray, tracks: list, y_start: int, panel_h: int):
    """Draws the 'Hall of the Lost' panel for long-lost tracks."""
    cv2.rectangle(canvas, (0, y_start), (canvas.shape[1], y_start + panel_h), (20, 20, 20), -1)
    _draw_label(canvas, "Long-Term Lost Tracks", (10, y_start + 20), (150, 150, 150))
    
    RECENT_LOSS_THRESHOLD = 30
    long_lost_tracks = [t for t in tracks if t.state == "lost" and t.time_since_update > RECENT_LOSS_THRESHOLD]
    
    thumb_w, thumb_h = 80, 80
    x_offset = 10
    
    for t in long_lost_tracks:
        if t.last_known_crop is None: continue
        thumbnail = _resize_crop(t.last_known_crop, (thumb_w, thumb_h))
        
        y_pos = y_start + 30
        canvas[y_pos:y_pos + thumb_h, x_offset:x_offset + thumb_w] = thumbnail
        
        label = f"ID {t.tid} ({t.time_since_update}f)"
        _draw_label(canvas, label, (x_offset, y_pos + thumb_h + 15), _state_to_color(t))
        x_offset += thumb_w + 10
        if x_offset + thumb_w > canvas.shape[1]: break

def _draw_reid_debug_panel(canvas: np.ndarray, reid_debug_info: dict, y_start: int, panel_h: int):
    """Draws the Re-ID candidate comparison panel."""
    cv2.rectangle(canvas, (0, y_start), (canvas.shape[1], y_start + panel_h), (20, 20, 20), -1)
    _draw_label(canvas, "Re-ID Candidate Matching", (10, y_start + 20), (150, 150, 150))

    if not reid_debug_info: return

    # Visualize the first lost track in the debug info
    query_tid = next(iter(reid_debug_info))
    info = reid_debug_info[query_tid]
    
    thumb_w, thumb_h = 80, 80
    x_offset = 10
    y_pos = y_start + 30
    
    # Draw Query
    if info['query_crop'] is not None:
        query_thumb = _resize_crop(info['query_crop'], (thumb_w, thumb_h))
        canvas[y_pos:y_pos + thumb_h, x_offset:x_offset + thumb_w] = query_thumb
        _draw_label(canvas, f"Query ID {query_tid}", (x_offset, y_pos + thumb_h + 15), (255, 255, 0))
    
    x_offset += thumb_w + 20
    cv2.line(canvas, (x_offset, y_pos), (x_offset, y_pos + thumb_h), (100, 100, 100), 2)
    x_offset += 10

    # Draw Candidates
    for cand in info['candidates']:
        cand_thumb = _resize_crop(cand['crop'], (thumb_w, thumb_h))
        canvas[y_pos:y_pos + thumb_h, x_offset:x_offset + thumb_w] = cand_thumb
        
        score = cand['score']
        color = (0, 255, 0) if score > 0.6 else (0, 215, 255) # Green for high score, Yellow for low
        _draw_label(canvas, f"Score: {score:.2f}", (x_offset, y_pos + thumb_h + 15), color)
        x_offset += thumb_w + 10
        if x_offset + thumb_w > canvas.shape[1]: break

# ---------- Public API ----------

def draw_tracks(frame: np.ndarray, tracks: list, tracker_config: dict):
    """Draws tracks, trails, and predicted search areas onto the frame."""
    H, W = frame.shape[:2]
    RECENT_LOSS_THRESHOLD = tracker_config.get('extrapolation_window', 30)

    for t in tracks:
        color = _state_to_color(t)
        x1, y1, x2, y2 = np.clip(t.box, [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1]).astype(int)
        
        # Draw Trail
        if t.state == "active" and len(t.center_history) > 1:
            points = np.array(list(t.center_history), dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
        
        # Draw Box and Search Area
        if t.state == "lost" and t.time_since_update <= RECENT_LOSS_THRESHOLD:
            # Draw predicted search area for recently lost tracks
            pred_center = (t.center + t.velocity * t.time_since_update).astype(int)
            allowance = tracker_config.get('center_gate_base', 50.0) + \
                        tracker_config.get('center_gate_slope', 10.0) * t.time_since_update
            cv2.circle(frame, tuple(pred_center), int(allowance), color, 1, cv2.LINE_AA)
            cv2.line(frame, tuple(t.center.astype(int)), tuple(pred_center), color, 1, cv2.LINE_AA)
        
        # Draw bounding box for active tracks
        if t.state == "active":
             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        label = f"ID {t.tid}"
        _draw_label(frame, label, (x1, y1), color)

def draw_hud(frame: np.ndarray, stats: Dict):
    """Draws a Heads-Up Display with system statistics."""
    font, font_scale, thickness, color = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, (255, 255, 255)
    x, y, line_h = 10, 20, 18
    
    # Create a semi-transparent background for the HUD
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


def draw_reid_links(frame: np.ndarray, reid_events: List[Dict], tracks: list):
    """Draws visual links for Re-ID events, now using state-based color."""
    tid_to_track = {t.tid: t for t in tracks}
    for event in reid_events:
        tid = int(event['tid'])
        track = tid_to_track.get(tid)
        color = _state_to_color(track) if track else _id_to_color(tid)

        new_box, old_box = event['new_box'].astype(int), event['old_box'].astype(int)
        c_new = (int(new_box[0] + new_box[2]) // 2, int(new_box[1] + new_box[3]) // 2)
        c_old = (int(old_box[0] + old_box[2]) // 2, int(old_box[1] + old_box[3]) // 2)

        cv2.line(frame, c_old, c_new, color, 3, cv2.LINE_AA)
        cv2.circle(frame, c_new, 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, c_new, 6, color, -1, cv2.LINE_AA)
        _draw_label(frame, f"Re-ID: {event['score']:.2f}", c_new, color)


def create_enhanced_frame(
    frame: np.ndarray, 
    tracks: list,
    reid_events: List[Dict],
    reid_debug_info: dict,
    tracker_config: dict
) -> np.ndarray:
    """
    Creates a single large frame with the main view and debug panels.
    This is the new primary function to call for visualization.
    """
    # --- CORRECTED: Correctly unpack frame height and width ---
    frame_h, frame_w = frame.shape[:2]
    panel_h = 150  # Height for each debug panel

    # --- CORRECTED: Use the correct frame_w for the canvas width ---
    canvas = np.zeros((frame_h + panel_h * 2, frame_w, 3), dtype=np.uint8)
    canvas[:frame_h, :, :] = frame
    
    # 1. Draw main tracks, search areas, and re-id links onto the top frame part
    draw_tracks(canvas, tracks, tracker_config)
    draw_reid_links(canvas, reid_events, tracks)

    # 2. Draw the "Hall of the Lost" panel
    lost_panel_y_start = frame_h
    _draw_lost_panel(canvas, tracks, lost_panel_y_start, panel_h)

    # 3. Draw the Re-ID Debug panel
    reid_panel_y_start = frame_h + panel_h
    _draw_reid_debug_panel(canvas, reid_debug_info, reid_panel_y_start, panel_h)
    
    # 4. Draw the HUD on top of everything
    draw_hud(canvas, hud_stats) 

    return canvas