"""Snap a rough plate quadrilateral onto the plate's real edges.

The locator (or a hand-placed box) is typically several percent of the plate width away from the
plate outline. For each side of the quad this looks along the side's normal for strong edges,
fits a line through the best candidates, and then picks the combination of four lines that forms
the most plausible plate: strong edges, near-parallel opposite sides, a plate-like aspect ratio,
and little movement away from the starting box.
"""
import itertools
from typing import List, Optional, Tuple

import cv2
import numpy as np

WINDOW = 0.15          # how far a side may move, as a fraction of the plate's other dimension
TOP_CANDIDATES = 3     # edge candidates considered per side
MIN_REL_STRENGTH = 0.15
PROXIMITY_SIGMA = 0.7  # prefer edges close to the starting side
LAMBDA_SHIFT = 0.6     # penalty for moving away from the starting box
LAMBDA_PARALLEL = 1.0  # penalty for non-parallel opposite sides
LAMBDA_ASPECT = 0.6    # penalty for an implausible width:height ratio
ITERATIONS = 2
OUTPUT_MARGIN = 1.1    # enlarge the fitted quad slightly: edges snap to the inner frame, and cutting a character costs more than a margin
BLUR_SIGMA = 1.2


def _intersect(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> Optional[np.ndarray]:
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    return p1 + np.linalg.solve(A, p2 - p1)[0] * d1


def _fit_line(points: np.ndarray, iterations: int = 60, tolerance: float = 1.5):
    """Robust line through points (RANSAC, then least squares on the inliers): (centre, unit direction)."""
    rng = np.random.default_rng(0)
    best = None
    for _ in range(iterations):
        i, j = rng.choice(len(points), 2, replace=False)
        d = points[j] - points[i]
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            continue
        d = d / norm
        distance = np.abs((points - points[i]) @ np.array([-d[1], d[0]]))
        inliers = distance < tolerance
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < 4:
        return None
    inlier_points = points[best]
    centre = inlier_points.mean(0)
    return centre, np.linalg.svd(inlier_points - centre)[2][0]


def _side_candidates(gx, gy, a, b, other):
    """Candidate edge lines for the side a->b as (point, direction, relative strength, relative offset)."""
    d = b - a
    t = d / np.linalg.norm(d)
    normal = np.array([t[1], -t[0]], np.float32)  # outward for a clockwise quad (image y points down)
    reach = int(max(6, WINDOW * other))
    samples = np.linspace(0.06, 0.94, 48)
    offsets = np.arange(-reach, reach + 1, 1.0)

    base = a[None, :] + samples[:, None] * d[None, :]
    px = (base[:, 0:1] + offsets[None, :] * normal[0]).astype(np.float32)
    py = (base[:, 1:2] + offsets[None, :] * normal[1]).astype(np.float32)
    gn = np.abs(cv2.remap(gx, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) * normal[0]
                + cv2.remap(gy, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) * normal[1])

    profile = cv2.GaussianBlur(gn.mean(0).reshape(1, -1), (0, 0), 1.5).ravel()
    peak_value = profile.max()
    if peak_value <= 1e-6:
        return []

    peaks = [j for j in range(1, len(profile) - 1)
             if profile[j] >= profile[j - 1] and profile[j] > profile[j + 1] and profile[j] >= 0.25 * peak_value]
    peaks.sort(key=lambda j: -profile[j] * np.exp(-(offsets[j] / (PROXIMITY_SIGMA * reach)) ** 2))

    candidates = []
    for j in peaks[:TOP_CANDIDATES]:
        window = slice(max(0, j - 4), min(len(offsets), j + 5))
        position = offsets[window][np.argmax(gn[:, window], axis=1)]
        strong = gn[:, window].max(1) >= MIN_REL_STRENGTH * peak_value
        line = None
        if strong.sum() >= 6:
            line = _fit_line(base[strong] + position[strong, None] * normal[None, :])
        if line is None:
            line = (a + offsets[j] * normal, t)
        centre, direction = line
        if direction @ t < 0:
            direction = -direction
        candidates.append((centre, direction, profile[j] / peak_value, offsets[j] / reach))
    return candidates


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(u @ v), 0, 1))))


def refine_quad(gray: np.ndarray, quad) -> Tuple[List[List[float]], bool]:
    """Refine a clockwise quad (top-left first) in a grayscale image. Returns (corners, refined?);
    when no trustworthy result is found the input corners are returned unchanged."""
    start = np.asarray(quad, np.float32).reshape(4, 2)
    side = float(np.mean([np.linalg.norm(start[(i + 1) % 4] - start[i]) for i in range(4)]))
    scale = min(max(250.0 / max(side, 1e-3), 0.5), 4.0)  # work on a crop where a side is ~250px

    low, high = start.min(0) - 0.7 * side, start.max(0) + 0.7 * side
    x0, y0 = int(max(0, np.floor(low[0]))), int(max(0, np.floor(low[1])))
    x1, y1 = int(min(gray.shape[1], np.ceil(high[0]))), int(min(gray.shape[0], np.ceil(high[1])))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return start.tolist(), False

    crop = cv2.resize(gray[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    crop = cv2.GaussianBlur(crop, (0, 0), BLUR_SIGMA).astype(np.float32)
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)

    quad_c = (start - [x0, y0]) * scale
    for _ in range(ITERATIONS):
        per_side = []
        for i in range(4):
            a, b = quad_c[i], quad_c[(i + 1) % 4]
            if np.linalg.norm(b - a) < 10:
                return start.tolist(), False
            other = 0.5 * (np.linalg.norm(quad_c[(i + 3) % 4] - a) + np.linalg.norm(quad_c[(i + 2) % 4] - b))
            candidates = _side_candidates(gx, gy, a, b, other)
            per_side.append(candidates or [(a, (b - a) / np.linalg.norm(b - a), 0.0, 0.0)])

        best = None
        for combo in itertools.product(*per_side):
            corners = []
            for i in range(4):
                p = _intersect(combo[(i + 3) % 4][0], combo[(i + 3) % 4][1], combo[i][0], combo[i][1])
                if p is None:
                    break
                corners.append(p)
            if len(corners) < 4:
                continue

            c = np.array(corners, np.float32)
            edges = np.roll(c, -1, 0) - c
            turn = edges[:, 0] * np.roll(edges, -1, 0)[:, 1] - edges[:, 1] * np.roll(edges, -1, 0)[:, 0]
            width = 0.5 * (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3]))
            height = 0.5 * (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1]))
            if not np.all(turn > 0) or height < 1 or not (1.3 <= width / height <= 6.0):
                continue

            ratio = width / height
            aspect_penalty = 0.0 if 1.8 <= ratio <= 5.0 else min(abs(np.log(ratio / (1.8 if ratio < 1.8 else 5.0))), 1.0)
            strength = np.mean([k[2] for k in combo])
            shift = np.mean([abs(k[3]) for k in combo])
            parallel = (_angle(combo[0][1], combo[2][1]) + _angle(combo[1][1], combo[3][1])) / 90.0
            score = strength - LAMBDA_SHIFT * shift - LAMBDA_PARALLEL * parallel - LAMBDA_ASPECT * aspect_penalty
            if best is None or score > best[0]:
                best = (score, c)

        if best is None:
            return start.tolist(), False
        quad_c = best[1]

    refined = quad_c / scale + [x0, y0]
    centre = refined.mean(0)
    refined = centre + (refined - centre) * OUTPUT_MARGIN
    if np.linalg.norm(centre - start.mean(0)) > 0.6 * side:
        return start.tolist(), False
    return refined.astype(np.float32).tolist(), True
