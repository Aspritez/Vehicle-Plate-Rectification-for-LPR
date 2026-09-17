"""Explicit corner rectification and geometric validation."""

import cv2
import numpy as np

MAX_OUTPUT_SIDE = 2000


def rectangle(width: int, height: int) -> np.ndarray:
    return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                     [0, height - 1]], dtype=np.float32)


def ordered_quad(points: np.ndarray, image_shape=None) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("ต้องมีมุมป้าย 4 จุดที่เป็นพิกัดถูกต้อง")
    hull = cv2.convexHull(points).reshape(-1, 2)
    if len(hull) != 4 or abs(cv2.contourArea(hull)) < 25:
        raise ValueError("มุมต้องเป็นสี่เหลี่ยมนูน มีพื้นที่พอ และไม่มีจุดซ้ำ")
    center = hull.mean(axis=0)
    order = np.argsort(np.arctan2(hull[:, 1] - center[1], hull[:, 0] - center[0]))
    quad = hull[order]
    quad = np.roll(quad, -int(np.argmin(quad.sum(axis=1))), axis=0)
    if cv2.contourArea(quad, oriented=True) < 0:
        quad = quad[[0, 3, 2, 1]]
    if np.min(np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)) < 3:
        raise ValueError("มุมป้ายอยู่ใกล้กันเกินไป")
    if image_shape is not None:
        h, w = image_shape[:2]
        if (quad < 0).any() or (quad[:, 0] > w - 1).any() or (quad[:, 1] > h - 1).any():
            raise ValueError("มุมป้ายต้องอยู่ภายในภาพ")
    return quad.astype(np.float32)


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([np.asarray(points).reshape(-1, 2), np.ones(len(points))])
    mapped = homogeneous @ homography.T
    if not np.isfinite(mapped).all() or np.any(np.abs(mapped[:, 2]) < 1e-8):
        raise ValueError("การแปลงมีจุดที่ไม่สามารถคำนวณได้")
    return mapped[:, :2] / mapped[:, 2, None]


def rectify_corners(image: np.ndarray, points: np.ndarray, width: int = 640,
                    aspect: float = 2.0, rotate: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 160 <= width <= MAX_OUTPUT_SIDE or not 0.75 <= aspect <= 5:
        raise ValueError("ขนาดหรือสัดส่วนปลายทางอยู่นอกช่วงที่รองรับ")
    height = round(width / aspect)
    if height > MAX_OUTPUT_SIDE:
        raise ValueError("ภาพปลายทางสูงเกิน 2000 พิกเซล ลดความกว้างหรือปรับสัดส่วน")
    quad = ordered_quad(points, image.shape)
    if rotate:
        quad = np.roll(quad, -1, axis=0)
    h = cv2.getPerspectiveTransform(quad, rectangle(width, height))
    if not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
        raise ValueError("คำนวณการแปลงไม่ได้ กรุณาแก้มุมป้าย")
    warped = cv2.warpPerspective(image, h, (width, height), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    return warped, h, quad
