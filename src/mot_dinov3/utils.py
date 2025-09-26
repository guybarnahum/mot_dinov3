# src/mot_dinov3/utils.py
import numpy as np

def iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculates a vectorized IoU matrix between two sets of boxes.
    - boxes1: (M, 4) array of XYXY boxes
    - boxes2: (N, 4) array of XYXY boxes
    - Returns: (M, N) matrix of IoU scores
    """
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    # Vectorized intersection calculation
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    xA = np.maximum(x11, x21.T)
    yA = np.maximum(y11, y21.T)
    xB = np.minimum(x12, x22.T)
    yB = np.minimum(y12, y22.T)

    inter_area = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)

    # Vectorized area and union calculation
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    union_area = area1 + area2.T - inter_area

    return (inter_area / (union_area + 1e-6)).astype(np.float32)


def cosine_cost_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Calculates the cosine distance (1 - similarity) between two sets of
    L2-normalized embeddings. Lower values mean more similar.
    - A: (M, D) array of embeddings
    - B: (N, D) array of embeddings
    - Returns: (M, N) matrix of cosine distances
    """
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    
    # Assumes A and B are already L2-normalized, so similarity is a simple dot product
    similarity = A @ B.T
    
    # Cost is 1 - similarity
    return (1.0 - similarity).astype(np.float32)


def centers_xyxy(b: np.ndarray) -> np.ndarray:
    """Calculates the centers of boxes in XYXY format."""
    if len(b) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    cx = (b[:, 0] + b[:, 2]) * 0.5
    cy = (b[:, 1] + b[:, 2]) * 0.5
    return np.stack([cx, cy], axis=1).astype(np.float32)


def get_crop(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Crops a region from the frame based on the bounding box,
    ensuring the coordinates are within the frame's boundaries.
    """
    x1, y1, x2, y2 = box.astype(int)
    h, w = frame.shape[:2]
    
    # Clamp coordinates to be within the frame dimensions
    x1_c = max(0, x1)
    y1_c = max(0, y1)
    x2_c = min(w, x2)
    y2_c = min(h, y2)
    
    # Return an empty array if the box is invalid or out of bounds
    if x1_c >= x2_c or y1_c >= y2_c:
        return np.array([[]], dtype=frame.dtype)
        
    return frame[y1_c:y2_c, x1_c:x2_c]