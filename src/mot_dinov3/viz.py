# src/mot_dinov3/viz.py (Final Polished Version)
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import utils

# --- Public constants for easy tuning of the debug view ---
DEBUG_PANEL_HEIGHT = 200
DEBUG_THUMBNAIL_SIZE = (120, 120)
ARROW_TIP_PIXELS = 10

# ---------- Color helpers ----------

from typing import Tuple

# Named Color Constants (BGR format)
COLOR_DYNAMIC_ACTIVE = (0, 200, 50)    # Green
COLOR_STATIC_ACTIVE  = (200, 120, 0)   # Blue
COLOR_RECENT_LOST    = (0, 165, 255)   # Orange
COLOR_LONG_TERM_LOST = (0, 0, 220)     # Red
COLOR_DEFAULT        = (255, 255, 255) # White

def _state_to_color(track, recent_loss_threshold: int) -> Tuple[int, int, int]:
    """Returns a BGR color based on the track's state."""
    if track.state == "active":
        return COLOR_STATIC_ACTIVE if track.is_static else COLOR_DYNAMIC_ACTIVE
    elif track.state == "lost":
        if track.time_since_update <= recent_loss_threshold:
            return COLOR_RECENT_LOST
        else:
            return COLOR_LONG_TERM_LOST
    return COLOR_DEFAULT

# ---------- Drawing helpers ----------

def _draw_arrow(canvas: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], 
                thickness: int = 2, absolute_tip_pixels: int = ARROW_TIP_PIXELS):
    """Draws an arrow with a constant pixel-sized arrowhead."""
    arrow_length = np.linalg.norm(np.array(pt2) - np.array(pt1))
    if arrow_length <= 0: return
    
    # Use a direct calculation to convert absolute pixel size to a relative factor
    relative_tip_length = absolute_tip_pixels / arrow_length
    
    cv2.arrowedLine(canvas, pt1, pt2, color, thickness, 
                    line_type=cv2.LINE_AA, tipLength=relative_tip_length)

def _draw_label(img: np.ndarray, text: str, pos: Tuple[int, int], color: Tuple[int, int, int],
                font_scale: float = 0.5, thickness: int = 1,
                bg_color: Optional[Tuple[int, int, int]] = None) -> None:
    """Draw a filled label box with auto-contrasting text and safe placement."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    pad = 3
    
    safe_y = max(th + pad * 2, y)
    safe_x = min(max(0, x), img.shape[1] - (tw + 2 * pad))
    
    bg_color = bg_color if bg_color is not None else color
    luminance = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]
    text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

    cv2.rectangle(img, (safe_x, safe_y - th - 2 * pad), (safe_x + tw + 2 * pad, safe_y), bg_color, -1)
    cv2.putText(img, text, (safe_x + pad, safe_y - pad), font, font_scale, text_color, thickness, cv2.LINE_AA)

def _resize_crop(crop: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resizes a crop to a target size, preserving aspect ratio by padding."""
    if crop is None or crop.size == 0: return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    tw, th = size; h, w, _ = crop.shape; s = min(tw/w, th/h); nw, nh = int(w*s), int(h*s)
    resized = cv2.resize(crop, (nw,nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8); xo, yo = (tw-nw)//2, (th-nh)//2
    canvas[yo:yo+nh, xo:xo+nw] = resized
    return canvas

# ---------- Panel and Legend Drawing Functions ----------

def _draw_track_list_in_panel(canvas: np.ndarray, tracks_to_draw: list, title: str,
                              x_start: int, x_end: int, y_start: int,
                              recent_loss_threshold: int):
    """A generic helper to draw a titled list of track thumbnails in a panel section."""
    _draw_label(canvas, title, (x_start, y_start + 25), (150, 150, 150), font_scale=0.6)
    
    thumb_w, thumb_h = DEBUG_THUMBNAIL_SIZE
    x_offset = x_start
    y_pos = y_start + 40
    
    for t in tracks_to_draw:
        if t.last_known_crop is None: continue
        if x_offset + thumb_w > x_end: break
        
        thumbnail = _resize_crop(t.last_known_crop, (thumb_w, thumb_h))
        canvas[y_pos:y_pos + thumb_h, x_offset:x_offset + thumb_w] = thumbnail
        
        color = _state_to_color(t, recent_loss_threshold)
        
        id_label = f"ID {t.tid} ({t.time_since_update}f)"
        _draw_label(canvas, id_label, (x_offset, y_pos + thumb_h + 20), color)
        
        if t.time_since_update <= recent_loss_threshold:
            vx, vy = t.velocity
            v_label = f"v=[{vx:.1f},{vy:.1f}]"
            _draw_label(canvas, v_label, (x_offset, y_pos + thumb_h + 38), color, font_scale=0.4)
        
        arrow_start_pt = (x_offset + thumb_w // 2, y_pos + thumb_h // 2)
        arrow_end_pt = tuple(utils.centers_xyxy(t.box[np.newaxis, :])[0].astype(int))
        
        # Draw the arrow directly onto the main canvas
        _draw_arrow(canvas, arrow_start_pt, arrow_end_pt, color, absolute_tip_pixels=ARROW_TIP_PIXELS)
        
        x_offset += thumb_w + 10

def _draw_all_lost_tracks_panel(canvas: np.ndarray, tracks: list, y_start: int, panel_h: int, tracker_config: dict):
    """Draws a panel with two sorted lists: recently lost and long-term lost tracks."""
    cv2.rectangle(canvas, (0, y_start), (canvas.shape[1], y_start + panel_h), (20, 20, 20), -1)
    
    lost_tracks = sorted([t for t in tracks if t.state == "lost"], key=lambda t: t.time_since_update)
    recent_loss_threshold = tracker_config.get('extrapolation_window', 30)
    
    recent_lost = [t for t in lost_tracks if t.time_since_update <= recent_loss_threshold]
    long_term_lost = [t for t in lost_tracks if t.time_since_update > recent_loss_threshold]
    
    panel_midpoint = canvas.shape[1] // 2
    cv2.line(canvas, (panel_midpoint, y_start), (panel_midpoint, y_start + panel_h), (80, 80, 80), 1)

    # Call the helper for each section (no overlay is needed)
    _draw_track_list_in_panel(canvas, recent_lost, "Recently Lost (Gated Search)",
                              10, panel_midpoint, y_start, recent_loss_threshold)
    _draw_track_list_in_panel(canvas, long_term_lost, "Long-Term Lost (Global Search)",
                              panel_midpoint + 10, canvas.shape[1], y_start, recent_loss_threshold)

# In src/mot_dinov3/viz.py

def _draw_reid_debug_panel(canvas: np.ndarray, reid_debug_info: dict, y_start: int, panel_h: int, frame_idx: int):
    """Draws the Re-ID candidate comparison panel, cycling through available tracks."""
    cv2.rectangle(canvas, (0, y_start), (canvas.shape[1], y_start + panel_h), (20, 20, 20), -1)
    _draw_label(canvas, "Re-ID Candidate Matching", (10, y_start + 25), (150, 150, 150), font_scale=0.6)

    if not reid_debug_info: return

    # --- MODIFIED: Cycle through all available lost tracks using the frame index ---
    lost_tids = sorted(list(reid_debug_info.keys()))
    if not lost_tids: return

    # Pick a track to focus on for this frame
    focus_idx = frame_idx % len(lost_tids)
    query_tid = lost_tids[focus_idx]
    info = reid_debug_info[query_tid]
    
    thumb_w, thumb_h = DEBUG_THUMBNAIL_SIZE
    x_offset, y_pos = 10, y_start + 40
    
    if info['query_crop'] is not None:
        query_thumb = _resize_crop(info['query_crop'], (thumb_w, thumb_h))
        canvas[y_pos:y_pos + thumb_h, x_offset:x_offset + thumb_w] = query_thumb
        # Display which track is being focused on
        _draw_label(canvas, f"Query ID {query_tid}", (x_offset, y_pos + thumb_h + 20), (255, 255, 0))
    
    x_offset += thumb_w + 20
    cv2.line(canvas, (x_offset, y_pos), (x_offset, y_pos + thumb_h), (100, 100, 100), 2)
    x_offset += 10

    for cand in info['candidates']:
        if x_offset + thumb_w > canvas.shape[1]: break
        cand_thumb = _resize_crop(cand['crop'], (thumb_w, thumb_h))
        canvas[y_pos:y_pos + thumb_h, x_offset:x_offset + thumb_w] = cand_thumb
        score = cand['score']
        color = (0, 255, 0) if score > 0.6 else (0, 215, 255)
        _draw_label(canvas, f"Score: {score:.2f}", (x_offset, y_pos + thumb_h + 20), color)
        x_offset += thumb_w + 10

def draw_legend(frame: np.ndarray):
    """Draws a legend for track colors and shapes in the top-right corner."""
    legend_items = {
        "Dynamic": (0, 200, 50),
        "Static": (200, 120, 0),
        "Recent Loss": (0, 165, 255),
        "Search Area": (0, 255, 255)
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

def draw_hud(frame: np.ndarray, stats: Dict):
    """Draws a Heads-Up Display with system statistics."""
    font, scale, thick, color = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, (255, 255, 255)
    x, y, line_h = 10, 20, 18
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    def draw_text(text, val=""):
        nonlocal y; full_text = f"{text}: {val}" if val else text
        cv2.putText(frame, full_text, (x, y), font, scale, color, thick, cv2.LINE_AA); y += line_h
    
    draw_text("Frame", stats.get('Frame', 'N/A'))
    draw_text("FPS", f"{stats.get('FPS', 0):.1f}")

# ---------- Public API ----------

def draw_tracks(frame: np.ndarray, tracks: list, tracker_config: dict):
    """Draws tracks, trails, and predicted search areas onto the frame."""
    H, W = frame.shape[:2]
    recent_loss_threshold = tracker_config.get('extrapolation_window', 30)
    debug_color = (0, 255, 255) # Yellow
    search_area_color = (0, 255, 255) # Yellow, consistent with legend

    for t in tracks:
        color = _state_to_color(t, recent_loss_threshold)
        
        if t.state == "active":
            x1, y1, x2, y2 = np.clip(t.box, [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1]).astype(int)
            if len(t.center_history) > 1:
                cv2.polylines(frame, [np.array(list(t.center_history), dtype=np.int32).reshape((-1,1,2))], False, color, 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            _draw_label(frame, f"ID {t.tid}", (x1, y1), color)

        elif t.state == "lost" and t.time_since_update <= recent_loss_threshold:
            x1, y1, x2, y2 = np.clip(t.box, [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1]).astype(int)
            last_seen_center = tuple(utils.centers_xyxy(np.array([[x1, y1, x2, y2]]))[0].astype(int))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), debug_color, 1)
            cv2.line(frame, (last_seen_center[0] - 7, last_seen_center[1]), (last_seen_center[0] + 7, last_seen_center[1]), debug_color, 1)
            cv2.line(frame, (last_seen_center[0], last_seen_center[1] - 7), (last_seen_center[0], last_seen_center[1] + 7), debug_color, 1)
            
            pred_center = tuple(t.center.astype(int))
            allowance = t.search_radius
            
            cv2.circle(frame, pred_center, max(1, int(allowance)), search_area_color, 1, cv2.LINE_AA)
            
            if t.time_since_update > 0:
                cv2.line(frame, last_seen_center, pred_center, search_area_color, 1, cv2.LINE_AA)

def draw_reid_links(frame: np.ndarray, reid_events: List[Dict], tracks: list, tracker_config: dict):
    """Draws visual links for Re-ID events with a distinct color."""
    reid_color = (255, 0, 255)  # Magenta
    for event in reid_events:
        new_box, old_box = event['new_box'].astype(int), event['old_box'].astype(int)
        
        c_new = tuple(utils.centers_xyxy(new_box[np.newaxis, :])[0].astype(int))
        c_old = tuple(utils.centers_xyxy(old_box[np.newaxis, :])[0].astype(int))

        _draw_arrow(frame, c_old, c_new, reid_color, thickness=3, absolute_tip_pixels=ARROW_TIP_PIXELS)
        cv2.circle(frame, c_new, 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, c_new, 6, reid_color, -1, cv2.LINE_AA)
        _draw_label(frame, f"Re-ID: {event['score']:.2f}", c_new, reid_color)


def create_enhanced_frame(frame: np.ndarray, tracks: list, reid_events: List[Dict],
                        reid_debug_info: dict, tracker_config: dict, hud_stats: dict,
                        frame_idx: int) -> np.ndarray: 
    """Creates a single large frame with the main view and debug panels."""
    frame_h, frame_w = frame.shape[:2]
    panel_h = DEBUG_PANEL_HEIGHT
    canvas = np.zeros((frame_h + panel_h * 2, frame_w, 3), dtype=np.uint8)
    canvas[:frame_h, :, :] = frame

    draw_tracks(canvas, tracks, tracker_config)
    draw_reid_links(canvas, reid_events, tracks, tracker_config)

    lost_panel_y_start = frame_h
    _draw_all_lost_tracks_panel(canvas, tracks, lost_panel_y_start, panel_h, tracker_config)

    reid_panel_y_start = frame_h + panel_h
    # --- MODIFIED: Pass frame_idx to the debug panel ---
    _draw_reid_debug_panel(canvas, reid_debug_info, reid_panel_y_start, panel_h, frame_idx)

    draw_legend(canvas)
    if hud_stats:
        draw_hud(canvas, hud_stats)

    return canvas