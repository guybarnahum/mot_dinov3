# src/mot_dinov3/viz.py
from __future__ import annotations

import cv2
import numpy as np
from typing import Iterable, Optional

# ---------- Color helpers ----------

def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
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

def _id_to_color(tid: int) -> tuple[int, int, int]:
    """
    Deterministic vivid color from an integer track id.
    Uses golden-ratio hue hop for well-spaced colors.
    Returns BGR for OpenCV.
    """
    phi = 0.618033988749895  # golden ratio conjugate
    h = (tid * phi) % 1.0
    s, v = 0.9, 1.0
    r, g, b = _hsv_to_rgb(h, s, v)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR

# ---------- Drawing helpers ----------

def _draw_label(img: np.ndarray, x: int, y: int, text: str,
                color_bgr: tuple[int, int, int],
                font_scale: float = 0.5, thickness: int = 1) -> None:
    """
    Draw a filled label box with text at the top-left corner (x,y) of a bbox.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x1, y1 = int(x), int(y)
    pad = 3
    # Background rectangle above the top-left corner (clamped to image)
    y_bg_top = max(0, y1 - th - 2 * pad)
    y_bg_bot = max(0, y1)
    x_bg_left = max(0, x1)
    x_bg_right = max(0, x1 + tw + 2 * pad)
    cv2.rectangle(img, (x_bg_left, y_bg_top), (x_bg_right, y_bg_bot), color_bgr, thickness=-1)
    # Black text on colored bg
    cv2.putText(img, text, (x_bg_left + pad, y_bg_bot - pad),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

# ---------- Public API ----------

def draw_tracks(frame: np.ndarray,
                tracks,
                draw_lost: bool = False,
                thickness: int = 2,
                font_scale: float = 0.5,
                mark_tids: set[int] | None = None) -> np.ndarray:
    """
    Draw colored boxes + 'ID <tid>' labels for tracks.

    mark_tids:
        Optional set of track IDs. If a track's ID is in this set, a '*' is appended
        to the label to indicate the embedding was computed this frame.
    """
    if tracks is None:
        return frame

    H, W = frame.shape[:2]
    mark_tids = mark_tids or set()

    for t in tracks:
        state = getattr(t, "state", "active")
        if not draw_lost and state != "active":
            continue

        x1, y1, x2, y2 = [int(v) for v in t.box]
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))

        color = _id_to_color(int(t.tid))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        label = f"ID {t.tid}"
        if getattr(t, "cls", None) is not None:
            label += f" (c{int(t.cls)})"   # or map id -> class name if you have a list

        # if you already mark embed-computed tracks with a star:
        if star_ids is not None and t.tid in star_ids:
            label += " *"

        if hasattr(t, "cls") and t.cls is not None:
            label += f" | c{int(t.cls)}"
        if state != "active":
            label += " (lost)"

        _draw_label(frame, x1, y1, label, color, font_scale=font_scale, thickness=1)

    return frame

