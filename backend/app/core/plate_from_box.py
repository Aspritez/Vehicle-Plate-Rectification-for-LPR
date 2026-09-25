"""Find the plate inside a rough rectangle drawn by the user.

GrabCut separates the plate from the car behind it. It is tried first with the middle of the box
marked as "probably the plate" (which keeps it from grabbing the car body at the box edges), and,
if that finds nothing, with the classic rectangle-only initialisation. The convex hull of the
largest foreground blob is simplified to a four-point polygon (a trapezoid, so plates seen at an
angle keep their perspective) and enlarged slightly, since GrabCut tends to cut the plate's border.

On simulated hand-drawn boxes (15-35% margin around the plate) this brings the corner error from
~33% of the plate width to ~9%, and makes tilted Thai plates readable that a plain rotated
rectangle could not be.
"""
from typing import List, Optional

import cv2
import numpy as np

from .deskew import PerspectiveDeskewer

WORK_LONG_SIDE = 240      # GrabCut runs on a crop scaled to this size
GRABCUT_ITERATIONS = 5
EDGE_MARGIN = 0.02        # the box's own border is treated as certain background
CENTER_FRACTION = 0.5     # middle part of the box marked "probably the plate"
MIN_FOREGROUND = 0.03     # smaller foreground blobs (fraction of the crop) count as failure
GROW = 1.14               # enlarge the fitted quad about its centre
MIN_BOX_SIDE = 16         # pixels


def _foreground(small: np.ndarray, use_center_prior: bool) -> Optional[np.ndarray]:
    h, w = small.shape[:2]
    margin = max(1, int(EDGE_MARGIN * max(h, w)))
    mask = np.zeros((h, w), np.uint8)
    background, foreground = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        if use_center_prior:
            mask[:] = cv2.GC_PR_BGD
            mask[:margin, :] = mask[-margin:, :] = cv2.GC_BGD
            mask[:, :margin] = mask[:, -margin:] = cv2.GC_BGD
            y0, y1 = int(h * (0.5 - CENTER_FRACTION / 2)), int(h * (0.5 + CENTER_FRACTION / 2))
            x0, x1 = int(w * (0.5 - CENTER_FRACTION / 2)), int(w * (0.5 + CENTER_FRACTION / 2))
            mask[y0:y1, x0:x1] = cv2.GC_PR_FGD
            cv2.grabCut(small, mask, None, background, foreground, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_MASK)
        else:
            cv2.grabCut(small, mask, (margin, margin, w - 2 * margin, h - 2 * margin),
                        background, foreground, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None

    plate = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    plate = cv2.morphologyEx(plate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(plate)
    if count < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(labels == largest)
    if len(xs) < MIN_FOREGROUND * h * w:
        return None
    return np.stack([xs, ys], 1).astype(np.float32)


def _four_corners(points: np.ndarray) -> np.ndarray:
    """Four-point polygon around the blob (falls back to its rotated bounding rectangle)."""
    hull = cv2.convexHull(points)
    perimeter = cv2.arcLength(hull, True)
    for epsilon in np.linspace(0.005, 0.15, 60):
        approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        if len(approx) <= 4:
            break
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)


def plate_from_box(image: np.ndarray, box) -> Optional[List[List[float]]]:
    """box = (x0, y0, x1, y1) in image pixels. Returns 4 corners (top-left first, clockwise) or None."""
    height, width = image.shape[:2]
    x0, y0 = max(0, int(min(box[0], box[2]))), max(0, int(min(box[1], box[3])))
    x1, y1 = min(width, int(max(box[0], box[2]))), min(height, int(max(box[1], box[3])))
    if x1 - x0 < MIN_BOX_SIDE or y1 - y0 < MIN_BOX_SIDE:
        return None

    crop = image[y0:y1, x0:x1]
    scale = WORK_LONG_SIDE / max(crop.shape[:2])
    small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    points = _foreground(small, use_center_prior=True)
    if points is None:
        points = _foreground(small, use_center_prior=False)
    if points is None:
        return None

    corners = PerspectiveDeskewer.order_points(_four_corners(points))
    centre = corners.mean(0)
    corners = centre + (corners - centre) * GROW

    corners = corners / scale + [x0, y0]
    corners[:, 0] = np.clip(corners[:, 0], 0, width - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, height - 1)
    return corners.tolist()
