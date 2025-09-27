import numpy as np
import cv2
from typing import Dict, Any, List, Tuple


def seg_to_centerlines(prob: np.ndarray, thr: float = 0.5) -> List[np.ndarray]:
    # prob: HxW in [0,1]
    binm = (prob > thr).astype(np.uint8) * 255
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    # Skeletonize via thinning (fallback: use contours + approx)
    contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in contours:
        if cv2.arcLength(c, False) < 50:
            continue
        lines.append(c.squeeze(1))
    return lines


def row_to_points(row_pred: Dict[str, Any], anchors: Dict[str, Any]) -> List[np.ndarray]:
    # Convert row-anchor x indices to polylines at given Y rows
    rows = anchors['rows']
    x0, x1 = anchors['x_range']
    x_bins = row_pred['row_x_logits'].shape[-1]
    B, L, R, X = row_pred['row_x_logits'].shape
    out = []
    probs = row_pred['row_x_logits'].softmax(dim=-1).detach().cpu().numpy()
    exist = row_pred['row_exist_logits'].sigmoid().detach().cpu().numpy()
    for b in range(B):
        lanes = []
        for l in range(L):
            pts = []
            for r_idx, y in enumerate(rows):
                if exist[b,l,r_idx] < 0.5:
                    continue
                x_idx = probs[b,l,r_idx].argmax()
                x = int(x0 + (x1 - x0) * (x_idx / (x_bins - 1)))
                pts.append((x, int(y)))
            if len(pts) > 2:
                lanes.append(np.array(pts, dtype=np.int32))
        out.append(lanes)
    return out


def fit_polylines(points: List[np.ndarray]) -> List[np.poly1d]:
    fits = []
    for pts in points:
        if len(pts) < 3:
            continue
        xs = pts[:,0]
        ys = pts[:,1]
        # Fit x=f(y) 3rd-order
        coeffs = np.polyfit(ys, xs, 3)
        fits.append(np.poly1d(coeffs))
    return fits
