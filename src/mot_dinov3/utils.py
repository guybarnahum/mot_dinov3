import numpy as np

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    M, N = len(a), len(b)
    ious = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        ax1, ay1, ax2, ay2 = a[i]
        a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        for j in range(N):
            bx1, by1, bx2, by2 = b[j]
            b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            union = a_area + b_area - inter + 1e-6
            ious[i, j] = inter / union
    return ious

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)), dtype=np.float32)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A @ B.T

