import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from .config import *

class SIFTDetector:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2),
            dict(checks=50)
        )
        self.template = None
        self.template_kp = None
        self.template_des = None

    def load_template(self, template_path: str) -> bool:
        """Load and process template image."""
        self.template = cv2.imread(template_path)
        if self.template is None:
            return False

        self.template_kp, self.template_des = self.sift.detectAndCompute(
            cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY), None
        )
        return len(self.template_kp) > 0

    def detect_plate(self, image: np.ndarray) -> Tuple[Optional[List[List[float]]], Dict]:
        """Detect license plate location using SIFT matching."""
        if self.template_des is None:
            return None, {"error": "Template not loaded"}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)

        if des is None or len(kp) < SIFT_KEYPOINT_THRESHOLD:
            return None, {"error": "Not enough keypoints found"}

        matches = self.matcher.knnMatch(self.template_des, des, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < SIFT_RATIO_TEST * n.distance:
                    good_matches.append(m)

        if len(good_matches) < FEATURE_MATCH_THRESHOLD:
            return None, {"error": f"Not enough good matches: {len(good_matches)}"}

        src_pts = np.float32([self.template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)

        if H is None:
            return None, {"error": "Homography computation failed"}

        h, w = self.template.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)

        plate_corners = [[float(pt[0][0]), float(pt[0][1])] for pt in transformed_corners]

        return plate_corners, {
            "matches": len(good_matches),
            "keypoints": len(kp)
        }

    def draw_matches(self, image: np.ndarray, corners: List[List[float]]) -> np.ndarray:
        """Draw detected plate region on image."""
        img_copy = image.copy()
        if corners:
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(img_copy, [pts], True, (0, 255, 0), 2)
        return img_copy
