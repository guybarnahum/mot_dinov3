# src/mot_dinov3/viz.py (Enhanced with Arrows from Lost Panel)
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# --- Public constants for easy tuning of the debug view ---
DEBUG_PANEL_HEIGHT = 256
DEBUG_THUMBNAIL_SIZE = (180, 180)

# ---------- Color helpers ----------

def _state_to_color(track) -> Tuple[int, int, int]:
    """Returns a BGR color based on the track's state."""
    RECENT_LOSS_THRESHOLD = 30
    if track.state == "active":
        return (200, 120, 0) if track.is_static else (0, 200, 50)  # Blue for Static, Green for Dynamic
    elif track.state == "lost":
        if track.time_since_update <= RECENT_LOSS_THRESHOLD:
            return (0, 165, 255)  # Orange for Recently Lost
        else:
            return (0, 0, 220)  # Red for Long-Term Lost
    return (255, 255, 255)

# ---------- Drawing helpers ----------

def _draw_label(img: np.ndarray, text: str, pos: Tuple[int, int], color: Tuple[int, int, int],
                font_scale: float = 0.5, thickness: int = 1,
                bg_color: Optional[Tuple[int, int, int]] = None) -> None:
    """Draw a filled label box with auto-contrasting text."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    pad = 3
    bg_color = bg_color if bg_color is not None else color
    
    luminance = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]
    text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

    cv2.rectangle(img, (x, y - th - 2 * pad), (x + tw + 2 * pad, y), bg_color, -1)
    cv2.putText(img, text, (x + pad, y - pad), font, font_scale, text_color, thickness, cv2.LINE_AA)

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

# ---------- Panel and Legend Drawing Functions ----------

def _draw_lost_panel(canvas: np.ndarray, tracks: list, y_start: int, panel_h: int):
    """Draws the 'Hall of the Lost' panel for long-lost tracks."""
    cv2.rectangle(canvas, (0, y_start), (canvas.shape[1], y_start + panel_h), (20, 20, 20), -1)
    _draw_label(canvas, "Long-Term Lost Tracks", (10, y_start + 25), (150, 150, 150), font_scale=0.6)
    
    RECENT_LOSS_THRESHOLD = 30
    long_lost_tracks = [t for t in tracks if t.state == "lost" and t.time_since_update > RECENT_LOSS_THRESHOLD]
    
    thumb_w, thumb_h = DEBUG_THUMBNAIL_SIZE
    x_offset = 10
    
    for t in long_lost_tracks:
        if t.last_known_crop is None: continue
        thumbnail = _resize_crop(t.last_known_crop, (thumb_w, thumb_h))
        
        y_pos = y_start + 40
        canvas[y_pos:y_pos + thumb_h, x_offset:x_offset + thumb_w] = thumbnail
        
        color = _state_to_color(t)
        label = f"ID {t.tid} ({t.time_since_update}f)"
        _draw_label(canvas, label, (x_offset, y_pos + thumb_h + 20), color)
        
        # --- NEW: Draw an arrow from the thumbnail to the last known position ---
        arrow_start_pt = (x_offset + thumb_w // 2, y_pos + thumb_h // 2)
        arrow_end_pt = tuple(t.center.astype(int))
        cv2.arrowedLine(canvas, arrow_start_pt, arrow_end_pt, color, 2, tipLength=0.2)
        
        x_offset += thumb_w + 10
        if x_offset + thumb_w > canvas.shape[1]: break

def _draw_reid_debug_panel(canvas: np.ndarray, reid_debug_info: dict, y_start: int, panel_h: int):
    """Draws the Re-ID candidate comparison panel."""
    cv2.rectangle(canvas, (0, y_start), (canvas.shape[1], y_start + panel_h), (20, 20, 20), -1)
    _draw_label(canvas, "Re-ID Candidate Matching", (10, y_start + 25), (150, 150, 150), font_scale=0.6)

    if not reid_debug_info: return

    query_tid = next(iter(reid_debug_info))
    info = reid_debug_info[query_tid]
    
    thumb_w, thumb_h = DEBUG_THUMBNAIL_SIZE
    x_offset = 10
    y_pos = y_start + 40
    
    if info['query_crop'] is not None:
        query_thumb = _resize_crop(info['query_crop'], (thumb_w, thumb_h))
        canvas[y_pos:y_pos + thumb_h, x_offset:x_offset + thumb_w] = query_thumb
        _draw_label(canvas, f"Query ID {query_tid}", (x_offset, y_pos + thumb_h + 20), (255, 255, 0))
    
    x_offset += thumb_w + 20
    cv2.line(canvas, (x_offset, y_pos), (x_offset, y_pos + thumb_h), (100, 100, 100), 2)
    x_offset += 10

    for cand in info['candidates']:
        cand_thumb = _resize_crop(cand['crop'], (thumb_w, thumb_h))
        canvas[y_pos:y_pos + thumb_h, x_offset:x_offset + thumb_w] = cand_thumb
        score = cand['score']
        color = (0, 255, 0) if score > 0.6 else (0, 215, 255)
        _draw_label(canvas, f"Score: {score:.2f}", (x_offset, y_pos + thumb_h + 20), color)
        x_offset += thumb_w + 10
        if x_offset + thumb_w > canvas.shape[1]: break

def draw_legend(frame: np.ndarray):
    """Draws a legend for track colors and shapes in the top-right corner."""
    legend_items = {
        "Dynamic": (0, 200, 50),
        "Static": (200, 120, 0),
        "Recent Loss": (0, 165, 255),
        "Search Area": (0, 165, 255),
    }
    x, y, line_h, pad = frame.shape[1] - 180, 20, 22, 5
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - pad, y - pad), (x + 150 + pad, y + len(legend_items) * line_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, (label, color) in enumerate(legend_items.items()):
        y_pos = y + i * line_h
        if label == "Search Area":
            cv2.circle(frame, (x + 10, y_pos + 7), 8, color, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (x, y_pos), (x + 20, y_pos + 15), color, -1)
        
        cv2.putText(frame, label, (x + 30, y_pos + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# ---------- Public API ----------

def draw_tracks(frame: np.ndarray, tracks: list, tracker_config: dict):
    """Draws tracks, trails, and predicted search areas onto the frame."""
    H, W = frame.shape[:2]
    RECENT_LOSS_THRESHOLD = tracker_config.get('extrapolation_window', 30)

    for t in tracks:
        color = _state_to_color(t)
        x1, y1, x2, y2 = np.clip(t.box, [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1]).astype(int)
        
        if t.state == "active" and len(t.center_history) > 1:
            points = np.array(list(t.center_history), dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
        
        if t.state == "lost" and t.time_since_update <= RECENT_LOSS_THRESHOLD:
            pred_center = (t.center + t.velocity * t.time_since_update).astype(int)
            allowance = tracker_config.get('center_gate_base', 50.0) + \
                        tracker_config.get('center_gate_slope', 10.0) * t.time_since_update
            cv2.circle(frame, tuple(pred_center), int(allowance), color, 1, cv2.LINE_AA)
            cv2.line(frame, tuple(t.center.astype(int)), tuple(pred_center), color, 1, cv2.LINE_AA)
        
        if t.state == "active":
             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        _draw_label(frame, f"ID {t.tid}", (x1, y1), color)

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

def draw_reid_links(frame: np.ndarray, reid_events: List[Dict], tracks: list):
    """Draws visual links for Re-ID events, using state-based color."""
    tid_to_track = {t.tid: t for t in tracks}
    for event in reid_events:
        tid = int(event['tid'])
        track = tid_to_track.get(tid)
        color = _state_to_color(track) if track else (200,200,200)

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
    tracker_config: dict,
    hud_stats: dict
) -> np.ndarray:
    """Creates a single large frame with the main view and debug panels."""
    frame_h, frame_w = frame.shape[:2]
    panel_h = DEBUG_PANEL_HEIGHT

    canvas = np.zeros((frame_h + panel_h * 2, frame_w, 3), dtype=np.uint8)
    canvas[:frame_h, :, :] = frame
    
    draw_tracks(canvas, tracks, tracker_config)
    draw_reid_links(canvas, reid_events, tracks)

    lost_panel_y_start = frame_h
    _draw_lost_panel(canvas, tracks, lost_panel_y_start, panel_h)

    reid_panel_y_start = frame_h + panel_h
    _draw_reid_debug_panel(canvas, reid_debug_info, reid_panel_y_start, panel_h)
    
    draw_legend(canvas)

    if hud_stats:
        draw_hud(canvas, hud_stats)

    return canvas