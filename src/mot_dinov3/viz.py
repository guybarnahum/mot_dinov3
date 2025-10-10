# src/mot_dinov3/viz.py (Final Polished Version)
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import utils

# --- Public constants for easy tuning of the debug view ---
DEBUG_PANEL_HEIGHT = 200
DEBUG_THUMBNAIL_SIZE = (80, 80)
ARROW_TIP_PIXELS = 15

# ---------- Color Constants (BGR format) ----------
COLOR_TENTATIVE      = (255, 255, 255)  # White
COLOR_DYNAMIC_ACTIVE = (180, 70, 0)      # Dark Blue
COLOR_STATIC_ACTIVE  = (200, 120, 0)    # Light Blue
COLOR_RECENT_LOST    = (0, 165, 255)    # Orange
COLOR_LONG_TERM_LOST = (0, 0, 220)       # Red

COLOR_BLACK          = (0, 0, 0)        # Black
COLOR_PANEL_BG       = (20, 20, 20)     # Dark Gray
COLOR_PANEL_TEXT     = (150, 150, 150)  # Light Gray
COLOR_SEPARATOR_DARK = (80, 80, 80)     # Gray
COLOR_DEBUG_YELLOW   = (0, 255, 255)    # Yellow
COLOR_REID           = (255, 0, 255)    # Magenta
COLOR_QUERY          = (255, 255, 0)    # Cyan
COLOR_WINNER         = (0, 255, 0)        # Green
COLOR_CANDIDATE      = (0, 215, 255)    # Yellow-Orange


# ---------- Color helpers ----------
def _state_to_color(track, recent_loss_threshold: int) -> Tuple[int, int, int]:
    """Returns a BGR color based on the track's state."""
    if track.state == "tentative":
        return COLOR_TENTATIVE
    elif track.state == "active":
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
    if arrow_length > 0:
        relative_tip_length = absolute_tip_pixels / arrow_length
        cv2.arrowedLine(canvas, pt1, pt2, color, thickness, line_type=cv2.LINE_AA, tipLength=relative_tip_length)

def _draw_label(img: np.ndarray, text: str, pos: Tuple[int, int], color: Tuple[int, int, int],
                font_scale: float = 0.5, thickness: int = 1, bg_color: Optional[Tuple[int, int, int]] = None) -> None:
    """Draw a filled label box with auto-contrasting text and safe placement."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    pad = 3
    safe_y = max(th + pad * 2, y)
    safe_x = min(max(0, x), img.shape[1] - (tw + 2 * pad))
    bg_color = bg_color if bg_color is not None else color
    luminance = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]
    text_color = COLOR_BLACK if luminance > 128 else COLOR_DEFAULT
    cv2.rectangle(img, (safe_x, safe_y - th - 2*pad), (safe_x + tw + 2*pad, safe_y), bg_color, -1)
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
def _draw_track_list_in_panel(canvas: np.ndarray, tracks_to_draw: list, title: str, x_start: int, x_end: int, y_start: int, recent_loss_threshold: int):
    """A generic helper to draw a titled list of track thumbnails in a panel section."""
    _draw_label(canvas, title, (x_start, y_start + 25), COLOR_PANEL_TEXT, font_scale=0.6)
    thumb_w, thumb_h = DEBUG_THUMBNAIL_SIZE
    x_offset, y_pos = x_start, y_start + 40
    for t in tracks_to_draw:
        if t.last_known_crop is None or x_offset + thumb_w > x_end: continue
        canvas[y_pos:y_pos + thumb_h, x_offset:x_offset + thumb_w] = _resize_crop(t.last_known_crop, (thumb_w, thumb_h))
        color = _state_to_color(t, recent_loss_threshold)
        id_label = f"ID {t.tid} ({t.time_since_update}f)"
        _draw_label(canvas, id_label, (x_offset, y_pos + thumb_h + 20), color)
        if t.time_since_update <= recent_loss_threshold:
            vx, vy = t.velocity; v_label = f"v=[{vx:.1f},{vy:.1f}]"
            _draw_label(canvas, v_label, (x_offset, y_pos + thumb_h + 38), color, font_scale=0.4)
        arrow_start_pt = (x_offset + thumb_w//2, y_pos + thumb_h//2)
        arrow_end_pt = tuple(utils.centers_xyxy(t.box[np.newaxis,:])[0].astype(int))
        _draw_arrow(canvas, arrow_start_pt, arrow_end_pt, color)
        x_offset += thumb_w + 10

def _draw_all_lost_tracks_panel(canvas: np.ndarray, tracks: list, y_start: int, panel_h: int, tracker_config: dict):
    """Draws a panel with two sorted lists: recently lost and long-term lost tracks."""
    cv2.rectangle(canvas, (0, y_start), (canvas.shape[1], y_start + panel_h), COLOR_PANEL_BG, -1)
    lost_tracks = sorted([t for t in tracks if t.state == "lost"], key=lambda t: t.time_since_update)
    thresh = tracker_config.get('extrapolation_window', 30)
    recent_lost = [t for t in lost_tracks if t.time_since_update <= thresh]
    long_term_lost = [t for t in lost_tracks if t.time_since_update > thresh]
    midpoint = canvas.shape[1] // 2
    cv2.line(canvas, (midpoint, y_start), (midpoint, y_start + panel_h), COLOR_SEPARATOR_DARK, 1)
    _draw_track_list_in_panel(canvas, recent_lost, "Recently Lost (Gated Search)", 10, midpoint, y_start, thresh)
    _draw_track_list_in_panel(canvas, long_term_lost, "Long-Term Lost (Global Search)", midpoint + 10, canvas.shape[1], y_start, thresh)

def _draw_reid_debug_panel(canvas: np.ndarray, reid_debug_info: dict, reid_events: list, y_start: int, panel_h: int, frame_idx: int):
    """Draws a multi-row panel showing a rotating selection of Re-ID candidates."""
    cv2.rectangle(canvas, (0, y_start), (canvas.shape[1], y_start + panel_h), COLOR_PANEL_BG, -1)
    lost_tids = sorted(list(reid_debug_info.keys())); num_lost = len(lost_tids)
    _draw_label(canvas, "Re-ID Candidates for Lost Tracks", (10, y_start + 25), COLOR_PANEL_TEXT, font_scale=0.6)
    if not lost_tids: return
    max_queries = 3; start_idx = (frame_idx//5)%num_lost; tids_to_show = [lost_tids[(start_idx+i)%num_lost] for i in range(min(max_queries, num_lost))]
    _draw_label(canvas, f"(Showing {len(tids_to_show)} of {num_lost})", (270, y_start + 25), COLOR_PANEL_TEXT, font_scale=0.6)
    winners = {e['tid']: e['new_box'] for e in reid_events}; thumb_w, thumb_h = DEBUG_THUMBNAIL_SIZE
    x, y = 10, y_start + 30; row_h = thumb_h + 35
    for tid in tids_to_show:
        if y + row_h > y_start + panel_h: break
        info = reid_debug_info[tid]
        if info['query_crop'] is not None:
            canvas[y:y+thumb_h, x:x+thumb_w] = _resize_crop(info['query_crop'], (thumb_w, thumb_h))
            _draw_label(canvas, f"Query ID {tid}", (x, y + thumb_h + 20), COLOR_QUERY)
        cand_x = x + thumb_w + 20; cv2.line(canvas, (cand_x - 10, y), (cand_x - 10, y + thumb_h), COLOR_DEFAULT, 1)
        for cand in info['candidates']:
            if cand_x + thumb_w > canvas.shape[1]: break
            canvas[y:y+thumb_h, cand_x:cand_x+thumb_w] = _resize_crop(cand['crop'], (thumb_w, thumb_h))
            winner = tid in winners and np.array_equal(cand['box'], winners[tid])
            color = COLOR_WINNER if winner else COLOR_CANDIDATE
            _draw_label(canvas, f"Score: {cand['score']:.2f}", (cand_x, y + thumb_h + 20), color)
            if winner: cv2.rectangle(canvas, (cand_x, y), (cand_x + thumb_w, y + thumb_h), COLOR_WINNER, 2)
            cand_x += thumb_w + 10
        y += row_h

def draw_legend(frame: np.ndarray):
    """Draws a legend for track colors and shapes in the top-right corner."""
    legend_items = {
        "Tentative": COLOR_TENTATIVE,
        "Dynamic": COLOR_DYNAMIC_ACTIVE,
        "Static": COLOR_STATIC_ACTIVE,
        "Recent Loss": COLOR_RECENT_LOST,
        "Search Area": COLOR_DEBUG_YELLOW
    }
    x, y, line_h, pad = frame.shape[1] - 180, 20, 22, 5
    overlay = frame.copy(); cv2.rectangle(overlay, (x-pad, y-pad), (x+170+pad, y+len(legend_items)*line_h), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, (label, color) in enumerate(legend_items.items()):
        y_pos = y + i * line_h
        if label == "Search Area": cv2.circle(frame, (x+10, y_pos+7), 8, color, 1, cv2.LINE_AA)
        else: cv2.rectangle(frame, (x, y_pos), (x + 20, y_pos + 15), color, -1)
        cv2.putText(frame, label, (x+30, y_pos+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_DEFAULT, 1, cv2.LINE_AA)

def draw_hud(frame: np.ndarray, stats: Dict):
    """Draws a Heads-Up Display with system statistics."""
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    x, y, line_h = 10, 20, 18
    overlay = frame.copy(); cv2.rectangle(overlay, (0,0), (220,120), COLOR_BLACK, -1); cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    def draw(text, val=""): nonlocal y; cv2.putText(frame, f"{text}: {val}" if val else text, (x, y), font, scale, COLOR_DEFAULT, thick, cv2.LINE_AA); y+=line_h
    draw("Frame", stats.get('Frame','N/A')); draw("FPS", f"{stats.get('FPS', 0):.1f}")

# ---------- Public API ----------
def draw_tracks(frame: np.ndarray, tracks: list, tracker_config: dict):
    """Draws tracks, trails, and predicted search areas onto the frame."""
    H, W = frame.shape[:2]
    thresh = tracker_config.get('extrapolation_window', 30)
    probation_period = tracker_config.get('probation_period', 5)
    for t in tracks:
        # Pass both thresholds to the color function
        color = _state_to_color(t, thresh, probation_period)
        
        x1, y1, x2, y2 = np.clip(t.box, [0,0,0,0], [W-1,H-1,W-1,H-1]).astype(int)

        if t.state in ("active", "tentative"):
            if len(t.center_history) > 1: cv2.polylines(frame, [np.array(list(t.center_history),dtype=np.int32).reshape((-1,1,2))], False, color, 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label = f"ID {t.tid}" + (" (T)" if t.state == "tentative" else "")
            _draw_label(frame, label, (x1, y1), color)
        elif t.state == "lost" and t.time_since_update <= thresh:
            last_c = tuple(utils.centers_xyxy(np.array([[x1,y1,x2,y2]]))[0].astype(int))
            cv2.rectangle(frame, (x1,y1), (x2,y2), COLOR_DEBUG_YELLOW, 1)
            cv2.line(frame, (last_c[0]-7,last_c[1]), (last_c[0]+7,last_c[1]), COLOR_DEBUG_YELLOW, 1)
            cv2.line(frame, (last_c[0],last_c[1]-7), (last_c[0],last_c[1]+7), COLOR_DEBUG_YELLOW, 1)
            pred_c = tuple(t.center.astype(int)); allowance = t.search_radius
            cv2.circle(frame, pred_c, max(1,int(allowance)), COLOR_DEBUG_YELLOW, 1, cv2.LINE_AA)
            if t.time_since_update > 0: cv2.line(frame, last_c, pred_c, COLOR_DEBUG_YELLOW, 1, cv2.LINE_AA)

def draw_reid_links(frame: np.ndarray, reid_events: List[Dict], tracks: list, tracker_config: dict):
    """Draws visual links for Re-ID events with a distinct color."""
    for event in reid_events:
        new_b, old_b = event['new_box'].astype(int), event['old_box'].astype(int)
        c_new = tuple(utils.centers_xyxy(new_b[np.newaxis,:])[0].astype(int))
        c_old = tuple(utils.centers_xyxy(old_b[np.newaxis,:])[0].astype(int))
        _draw_arrow(frame, c_old, c_new, COLOR_REID, 3)
        cv2.circle(frame, c_new, 8, COLOR_DEFAULT, -1, cv2.LINE_AA)
        cv2.circle(frame, c_new, 6, COLOR_REID, -1, cv2.LINE_AA)
        _draw_label(frame, f"Re-ID: {event['score']:.2f}", c_new, COLOR_REID)

def create_enhanced_frame(frame: np.ndarray, tracks: list, reid_events: List[Dict],
                          reid_debug_info: dict, tracker_config: dict, hud_stats: dict, frame_idx: int) -> np.ndarray:
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
    _draw_reid_debug_panel(canvas, reid_debug_info, reid_events, reid_panel_y_start, panel_h, frame_idx)
    draw_legend(canvas)
    if hud_stats: draw_hud(canvas, hud_stats)
    return canvas