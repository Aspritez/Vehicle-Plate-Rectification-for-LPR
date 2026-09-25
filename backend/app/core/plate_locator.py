"""SIFT-based license plate locator trained from labeled images.

Instead of matching SIFT features against a single template (which cannot
generalise across plate designs), a classifier learns which SIFT keypoints
(RootSIFT descriptor + scale/response/orientation) lie inside a plate. At
inference the keypoints are scored, the densest cluster of plate-like keypoints
is grouped, and its rotated bounding box is returned as the 4 plate corners.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .deskew import PerspectiveDeskewer

TARGET_LONG_SIDE = 480      # images are resized to this before feature extraction
MIN_CLUSTER_KEYPOINTS = 4   # fewer plate-like keypoints than this -> no detection


class SIFTPlateLocator:
    def __init__(self, sigma_frac: float = 0.04, keep: float = 0.5,
                 radius_frac: float = 0.12, grow: float = 1.3):
        self.sift = cv2.SIFT_create(contrastThreshold=0.02)
        self.classifier = None
        self.sigma_frac = sigma_frac      # heat-map blur, as a fraction of the long side
        self.keep = keep                  # keep keypoints scoring above keep * max score
        self.radius_frac = radius_frac    # cluster radius around the heat-map peak
        self.grow = grow                  # enlarge the cluster box (keypoints sit inside the plate edge)

    # ---------- features ----------
    def _resize(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        scale = TARGET_LONG_SIDE / max(h, w)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interp), scale

    def extract(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return keypoint coordinates (resized image), features and the resize scale."""
        small, scale = self._resize(image)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        kps, desc = self.sift.detectAndCompute(gray, None)
        if desc is None or len(kps) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 132), np.float32), scale

        desc = np.sqrt(desc / (np.abs(desc).sum(1, keepdims=True) + 1e-7))  # RootSIFT
        xy = np.array([k.pt for k in kps], np.float32)
        extra = np.array([[k.size / TARGET_LONG_SIDE * 20, k.response * 100,
                           np.cos(np.deg2rad(k.angle)), np.sin(np.deg2rad(k.angle))]
                          for k in kps], np.float32)
        return xy, np.hstack([desc, extra]).astype(np.float32), scale

    # ---------- training ----------
    def fit(self, images: List[np.ndarray], polygons: List[np.ndarray], trees: int = 200) -> Dict:
        from sklearn.ensemble import ExtraTreesClassifier

        X, Y = [], []
        for img, poly in zip(images, polygons):
            xy, feats, scale = self.extract(img)
            if len(xy) == 0:
                continue
            poly_small = (np.asarray(poly, np.float32) * scale).reshape(-1, 1, 2)
            inside = [cv2.pointPolygonTest(poly_small, (float(x), float(y)), False) >= 0 for x, y in xy]
            X.append(feats)
            Y.append(np.array(inside, np.int32))

        X, Y = np.vstack(X), np.concatenate(Y)
        self.classifier = ExtraTreesClassifier(
            n_estimators=trees, min_samples_leaf=3, class_weight="balanced",
            n_jobs=-1, random_state=0).fit(X, Y)
        return {"keypoints": int(len(Y)), "plate_keypoints": int(Y.sum())}

    # ---------- inference ----------
    def locate(self, image: np.ndarray) -> Tuple[Optional[List[List[float]]], Dict]:
        """Return (corners TL,TR,BR,BL in original-image pixels, info) or (None, info)."""
        if self.classifier is None:
            return None, {"error": "Plate locator model is not loaded"}

        xy, feats, scale = self.extract(image)
        if len(xy) == 0:
            return None, {"error": "No SIFT keypoints found in the image"}

        p = self.classifier.predict_proba(feats)[:, 1]
        box = self._group(xy, p, image.shape[:2], scale)
        if box is None:
            return None, {"error": "No plate-like region found", "keypoints": int(len(xy))}

        corners = (PerspectiveDeskewer.order_points(box) / scale).tolist()
        return corners, {"keypoints": int(len(xy))}

    def _group(self, xy: np.ndarray, p: np.ndarray, orig_shape, scale: float) -> Optional[np.ndarray]:
        h, w = int(round(orig_shape[0] * scale)), int(round(orig_shape[1] * scale))
        heat = np.zeros((h, w), np.float32)
        xi = np.clip(xy[:, 0].astype(int), 0, w - 1)
        yi = np.clip(xy[:, 1].astype(int), 0, h - 1)
        np.add.at(heat, (yi, xi), p)
        heat = cv2.GaussianBlur(heat, (0, 0), self.sigma_frac * max(h, w))

        cy, cx = np.unravel_index(heat.argmax(), heat.shape)
        dist = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
        chosen = (dist < self.radius_frac * max(h, w)) & (p > self.keep * p.max())
        if chosen.sum() < MIN_CLUSTER_KEYPOINTS:
            return None

        box = cv2.boxPoints(cv2.minAreaRect(xy[chosen].astype(np.float32)))
        centre = box.mean(0)
        return centre + (box - centre) * self.grow

    # ---------- persistence ----------
    def save(self, path: Path) -> None:
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"classifier": self.classifier, "sigma_frac": self.sigma_frac, "keep": self.keep,
                     "radius_frac": self.radius_frac, "grow": self.grow}, path)

    @classmethod
    def load(cls, path: Path) -> "SIFTPlateLocator":
        import joblib
        data = joblib.load(path)
        locator = cls(data["sigma_frac"], data["keep"], data["radius_frac"], data["grow"])
        locator.classifier = data["classifier"]
        return locator
