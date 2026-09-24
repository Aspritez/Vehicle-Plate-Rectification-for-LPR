"""
Vehicle License Plate Rectification – Computer Vision Pipeline
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Optional, Tuple


def order_points(pts: np.ndarray) -> np.ndarray:
    """Sort four 2-D points into (TL, TR, BR, BL) order."""
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR
    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


def validate_quadrilateral(
    corners: List[List[int]],
    image_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[bool, str]:
    if corners is None or len(corners) != 4:
        return False, "Exactly 4 corner points are required."

    pts = np.array(corners, dtype="float32")
    ordered = order_points(pts)
    tl, tr, br, bl = ordered

    contour = ordered.reshape(4, 1, 2).astype(np.int32)
    if not cv2.isContourConvex(contour):
        return False, "The quadrilateral is not convex (edges cross)."

    w_top    = float(np.linalg.norm(tr - tl))
    w_bottom = float(np.linalg.norm(br - bl))
    h_left   = float(np.linalg.norm(bl - tl))
    h_right  = float(np.linalg.norm(br - tr))
    avg_w = (w_top + w_bottom) / 2.0
    avg_h = (h_left + h_right) / 2.0

    if avg_h < 1:
        return False, "Height is essentially zero."

    aspect = avg_w / avg_h
    # Extremely forgiving aspect ratio to accommodate heavy skew
    if not (0.8 <= aspect <= 10.0):
        return False, f"Aspect ratio {aspect:.2f} doesn't look like a plate."

    if image_shape is not None:
        img_h, img_w = image_shape[:2]
        area = cv2.contourArea(contour)
        img_area = img_w * img_h
        if area < img_area * 0.00005:
            return False, "Selected region is too small."
        if area > img_area * 0.99:
            return False, "Selected region covers almost the entire image."

    return True, "Valid quadrilateral."


def _generate_synthetic_template() -> np.ndarray:
    """Generate a generic reference plate for SIFT feature matching."""
    template = np.ones((130, 400), dtype=np.uint8) * 255
    cv2.rectangle(template, (4, 4), (396, 126), 0, 4)
    # Add high-frequency text to generate rich SIFT descriptors
    cv2.putText(template, "XYZ 9876", (30, 90), cv2.FONT_HERSHEY_DUPLEX, 2.2, 0, 5)
    return template


def _get_morphology_mask(gray: np.ndarray) -> np.ndarray:
    """Generate a binary mask of text-heavy/plate-like regions to filter SIFT."""
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect_kernel)
    
    sobelx = cv2.Sobel(tophat, cv2.CV_32F, 1, 0, ksize=3)
    sobelx = np.absolute(sobelx)
    minVal, maxVal = np.min(sobelx), np.max(sobelx)
    if maxVal > 0:
        sobelx = (255 * ((sobelx - minVal) / (maxVal - minVal))).astype("uint8")
    else:
        sobelx = sobelx.astype("uint8")
        
    sobelx = cv2.GaussianBlur(sobelx, (5, 5), 0)
    _, thresh = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sq_kernel)
    closed = cv2.erode(closed, None, iterations=1)
    # Dilate heavily to ensure the mask covers the whole plate area for SIFT
    closed = cv2.dilate(closed, None, iterations=4)
    return closed


def _detect_via_sift_matching(gray: np.ndarray, image_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    """
    Implements the full SIFT -> KNN Matching -> Lowe's Ratio -> RANSAC Homography pipeline.
    """
    template = _generate_synthetic_template()
    sift = cv2.SIFT_create()
    
    # 1. Feature Extraction on Template
    kp1, des1 = sift.detectAndCompute(template, None)
    
    # IMPROVEMENT: Use the morphology mask so SIFT ONLY looks at plate-like textures
    # This prevents it from matching against trees, cars, and fences!
    mask = _get_morphology_mask(gray)
    kp2, des2 = sift.detectAndCompute(gray, mask)
    
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return None
        
    # 2. Feature Matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # 3. Descriptor Ratio Testing (Lowe's Test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    # 4. Robust Geometric Estimation (RANSAC)
    if len(good_matches) >= 5: # Need at least 4 for Homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            # 5. Transform template corners to image space
            h, w = template.shape
            pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            corners = dst.reshape(4, 2).tolist()
            
            ok, _ = validate_quadrilateral(corners, image_shape)
            if ok:
                return order_points(np.array(corners))
                
    return None


def _detect_via_contours(gray: np.ndarray, image_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    """Fallback 1: Contour detection"""
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
        if len(approx) == 4:
            corners = approx.reshape(4, 2).tolist()
            ok, _ = validate_quadrilateral(corners, image_shape)
            if ok:
                return order_points(np.array(corners))
    return None


def detect_plate_corners(image: np.ndarray) -> Optional[List[List[int]]]:
    """Return four plate-corner points [[x,y], …] or None."""
    try:
        if image is None: return None
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Primary: Advanced SIFT Matching with ROI Masking
        result = _detect_via_sift_matching(gray, image.shape)
        if result is not None: return result.astype(int).tolist()

        # 2. Fallback: Edge/Contour based detection
        result = _detect_via_contours(gray, image.shape)
        if result is not None: return result.astype(int).tolist()

        # 3. Final Fallback: centred rectangle
        cx, cy = w // 2, h // 2
        rw, rh = int(w * 0.3), int(h * 0.3)
        return [
            [cx - rw, cy - rh], [cx + rw, cy - rh],
            [cx + rw, cy + rh], [cx - rw, cy + rh],
        ]
    except Exception as exc:
        print(f"[detect_plate_corners] {exc}")
        return None


def rectify_plate(
    image: np.ndarray,
    corners: List[List[int]],
    output_width: int = 400,
    output_height: int = 130,
) -> np.ndarray:
    """Perspective-correct the plate using cv2.getPerspectiveTransform."""
    try:
        src = order_points(np.array(corners, dtype="float32"))
        dst = np.array([
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ], dtype="float32")
        
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(
            image, M, (output_width, output_height),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
    except Exception as exc:
        print(f"[rectify_plate] {exc}")
        return image.copy()
