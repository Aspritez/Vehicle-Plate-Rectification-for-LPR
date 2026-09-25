import cv2
import numpy as np
from typing import Tuple, List
from .config import PLATE_STANDARD_RATIO, DESKEWED_PLATE_WIDTH, DESKEWED_PLATE_HEIGHT

class PerspectiveDeskewer:
    TARGET_WIDTH = DESKEWED_PLATE_WIDTH
    TARGET_HEIGHT = DESKEWED_PLATE_HEIGHT

    @staticmethod
    def order_points(pts) -> np.ndarray:
        """Order corner points clockwise starting from the top-left.

        Sorting by angle around the centroid always yields 4 distinct points, unlike the
        classic sum/difference trick which duplicates a corner (and drops another) when
        the plate is strongly skewed.
        """
        pts = np.asarray(pts, dtype="float32").reshape(4, 2)
        centre = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
        clockwise = pts[np.argsort(angles)]  # image y points down, so ascending angle = clockwise
        start = int(np.argmin(clockwise.sum(axis=1)))
        return np.roll(clockwise, -start, axis=0)

    @staticmethod
    def arrange_corners(pts) -> np.ndarray:
        """Keep the given corner order (top-left, top-right, bottom-right, bottom-left) when it
        already forms a valid convex quadrilateral, e.g. corners placed by the user or the
        locator; otherwise fall back to sorting them."""
        pts = np.asarray(pts, dtype="float32").reshape(4, 2)
        edges = np.roll(pts, -1, axis=0) - pts
        cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        if np.all(cross > 0):
            return pts                      # convex and clockwise on screen: trust the given start corner
        return PerspectiveDeskewer.order_points(pts)  # mirrored or crossing: the start corner is unknown

    @staticmethod
    def deskew_plate(image: np.ndarray, corners: List[List[float]]) -> Tuple[np.ndarray, bool]:
        """Apply perspective transformation to deskew plate."""
        if not corners or len(corners) != 4:
            return image, False

        try:
            src_pts = np.array(corners, dtype="float32")
            src_pts = PerspectiveDeskewer.arrange_corners(src_pts)

            dst_pts = np.array([
                [0, 0],
                [PerspectiveDeskewer.TARGET_WIDTH, 0],
                [PerspectiveDeskewer.TARGET_WIDTH, PerspectiveDeskewer.TARGET_HEIGHT],
                [0, PerspectiveDeskewer.TARGET_HEIGHT]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(
                image, M,
                (PerspectiveDeskewer.TARGET_WIDTH, PerspectiveDeskewer.TARGET_HEIGHT)
            )

            return warped, True
        except Exception:
            return image, False

    @staticmethod
    def _ink_mask(gray: np.ndarray) -> np.ndarray:
        """Text mask: the minority side of an Otsu split (text covers less area than the plate)."""
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        if (mask > 0).mean() > 0.5:
            mask = 255 - mask
        h, w = mask.shape
        margin_y, margin_x = int(0.04 * h), int(0.03 * w)      # ignore the plate edge / border
        mask[:margin_y] = 0
        mask[h - margin_y:] = 0
        mask[:, :margin_x] = 0
        mask[:, w - margin_x:] = 0
        return mask.astype(np.float32) / 255

    @staticmethod
    def _shear_matrix(width: int, height: int, degrees: float) -> np.ndarray:
        """Horizontal shear about the image centre (3x3)."""
        k = np.tan(np.radians(degrees))
        return np.array([[1, k, -k * height / 2], [0, 1, 0], [0, 0, 1]], dtype=np.float64)

    @staticmethod
    def _rotation_matrix(width: int, height: int, degrees: float) -> np.ndarray:
        return np.vstack([cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0), [0, 0, 1]])

    @staticmethod
    def straighten_text(image: np.ndarray, max_rotation: float = 12, max_shear: float = 15) -> np.ndarray:
        """Remove residual tilt from a warped plate using the text itself.

        The rotation that makes the text lines sharpest in a row-sum profile, and then the shear that
        makes the characters' vertical strokes sharpest in a column-sum profile, are applied together as
        one affine warp. A change is only made when it clearly sharpens the profile.
        """
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ink = PerspectiveDeskewer._ink_mask(gray)
        h, w = ink.shape
        if ink.sum() < 0.02 * h * w:
            return image

        def sharpness(profile):
            return float(np.var(np.diff(profile)))

        def rotated(angle):
            return cv2.warpAffine(ink, cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1), (w, h))

        def search(candidates, score):
            values = [score(c) for c in candidates]
            return candidates[int(np.argmax(values))], max(values)

        rotation_score = lambda a: sharpness(rotated(a).sum(axis=1))
        angle, _ = search(np.arange(-max_rotation, max_rotation + 0.01, 1.0), rotation_score)
        angle, best = search(np.arange(angle - 1, angle + 1.01, 0.2), rotation_score)
        angle = angle if best > 1.05 * rotation_score(0.0) else 0.0

        base = rotated(angle)
        shear_score = lambda s: sharpness(cv2.warpAffine(
            base, PerspectiveDeskewer._shear_matrix(w, h, s)[:2], (w, h)).sum(axis=0))
        shear, _ = search(np.arange(-max_shear, max_shear + 0.01, 2.0), shear_score)
        shear, best = search(np.arange(shear - 2, shear + 2.01, 0.5), shear_score)
        shear = shear if best > 1.05 * shear_score(0.0) else 0.0

        if abs(angle) < 0.3 and abs(shear) < 1.0:
            return image
        transform = (PerspectiveDeskewer._shear_matrix(w, h, shear)
                     @ PerspectiveDeskewer._rotation_matrix(w, h, angle))[:2]
        return cv2.warpAffine(image, transform, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
