"""Descriptor matching and robust image-to-reference homography."""

from dataclasses import dataclass, field
from time import perf_counter
import cv2
import numpy as np

from src.geometry import project_points, rectangle
from src.imaging import resized


@dataclass
class MatchResult:
    accepted: bool = False
    reason: str = ""
    homography: np.ndarray | None = None
    warped: np.ndarray | None = None
    visualization: np.ndarray | None = None
    source_quad: np.ndarray | None = None
    metrics: dict = field(default_factory=dict)


def estimate_correspondences(src: np.ndarray, dst: np.ndarray, threshold: float = 3.0):
    """Estimate with redundant observations; four-point rectification is separate."""
    if len(src) < 8 or len(src) != len(dst):
        raise ValueError("ต้องมีคู่จุดอย่างน้อย 8 คู่สำหรับการประมาณแบบ robust")
    h, mask = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC,
                                threshold, maxIters=3000, confidence=0.995)
    if h is None or mask is None or not np.isfinite(h).all():
        raise ValueError("RANSAC ไม่พบ Homography ที่เหมาะสม")
    return h, mask.ravel().astype(bool)


def match_rectify(source: np.ndarray, reference: np.ndarray, method: str = "SIFT",
                  ratio: float = 0.75, ransac_px: float = 3.0,
                  min_inliers: int = 8, min_ratio: float = 0.45,
                  min_coverage: float = 0.03) -> MatchResult:
    start = perf_counter()
    result = MatchResult()

    def finish(reason: str):
        result.reason = reason
        result.metrics["elapsed_ms"] = round((perf_counter() - start) * 1000, 2)
        return result

    if method not in ("SIFT", "ORB") or not 0 < ratio < 1 or ransac_px <= 0:
        return finish("ค่าการจับคู่ไม่ถูกต้อง")
    if min(*source.shape[:2], *reference.shape[:2]) < 16:
        return finish("บริเวณป้ายเล็กเกินไป ต้องกว้างและสูงอย่างน้อย 16 พิกเซล")
    src, s_source = resized(source, 1200)
    ref, s_reference = resized(reference, 1200)
    gray_src, gray_ref = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in (src, ref))
    extractor = cv2.SIFT_create(nfeatures=2500) if method == "SIFT" else cv2.ORB_create(nfeatures=3000, fastThreshold=12)
    kp_src, desc_src = extractor.detectAndCompute(gray_src, None)
    kp_ref, desc_ref = extractor.detectAndCompute(gray_ref, None)
    result.metrics.update(method=method, source_keypoints=len(kp_src), reference_keypoints=len(kp_ref),
                          ratio_test=ratio, ransac_threshold_px=ransac_px,
                          matching_size_reference=[ref.shape[1], ref.shape[0]])
    if desc_src is None or desc_ref is None or min(len(kp_src), len(kp_ref)) < 2:
        return finish("จุดเด่นไม่เพียงพอ กรุณาเลือกบริเวณที่เห็นตัวอักษรชัด หรือเปลี่ยนภาพอ้างอิง")
    norm = cv2.NORM_L2 if method == "SIFT" else cv2.NORM_HAMMING
    pairs = cv2.BFMatcher(norm).knnMatch(desc_src, desc_ref, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    # A reference keypoint must not be counted as independent evidence multiple times.
    unique = {}
    for match in sorted(good, key=lambda m: m.distance):
        unique.setdefault(match.trainIdx, match)
    good = list(unique.values())
    result.metrics.update(knn_pairs=len(pairs), good_matches=len(good))
    if len(good) < max(8, min_inliers):
        result.visualization = cv2.drawMatches(src, kp_src, ref, kp_ref, good[:150], None,
                                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return finish("คู่จุดหลังกรองไม่เพียงพอ ลองเปลี่ยนบริเวณป้ายหรือใช้มุม 4 จุด")
    src_points = np.float32([kp_src[m.queryIdx].pt for m in good])
    ref_points = np.float32([kp_ref[m.trainIdx].pt for m in good])
    try:
        h_small, inlier = estimate_correspondences(src_points, ref_points, ransac_px)
    except ValueError as exc:
        return finish(str(exc))
    result.visualization = cv2.drawMatches(src, kp_src, ref, kp_ref, good[:150], None,
                                           matchColor=(37, 190, 145), matchesMask=inlier[:150].astype(int).tolist(),
                                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    count = int(inlier.sum())
    fraction = count / len(good)
    result.metrics.update(inliers=count, inlier_ratio=round(fraction, 4))
    if count < min_inliers or fraction < min_ratio:
        return finish("คู่ที่สอดคล้องกับ Homography มีน้อยเกินไป ไม่ยอมรับการแปลงนี้")
    coverage_src = cv2.contourArea(cv2.convexHull(src_points[inlier])) / (src.shape[0] * src.shape[1])
    coverage_ref = cv2.contourArea(cv2.convexHull(ref_points[inlier])) / (ref.shape[0] * ref.shape[1])
    errors = np.linalg.norm(project_points(src_points[inlier], h_small) - ref_points[inlier], axis=1)
    result.metrics.update(source_coverage=round(coverage_src, 4), reference_coverage=round(coverage_ref, 4),
                          median_reprojection_px=round(float(np.median(errors)), 3),
                          p95_reprojection_px=round(float(np.percentile(errors, 95)), 3))
    if min(coverage_src, coverage_ref) < min_coverage:
        return finish("จุดที่จับคู่กระจุกอยู่บริเวณเล็กเกินไป กรุณาครอปให้ใกล้ป้ายหรือเปลี่ยนภาพ")
    if np.median(errors) > ransac_px:
        return finish("ความคลาดเคลื่อนการจับคู่สูงเกินไป")
    try:
        if abs(np.linalg.det(h_small)) < 1e-12:
            return finish("Homography เสื่อม ไม่สามารถปรับมุมได้")
        ref_corners = rectangle(ref.shape[1], ref.shape[0])
        inverse = np.linalg.inv(h_small)
        plane = np.column_stack([ref_corners, np.ones(4)]) @ inverse.T
        if np.min(plane[:, 2]) * np.max(plane[:, 2]) <= 0:
            return finish("การแปลงตัดผ่านจุดอนันต์ กรุณาเปลี่ยนคู่ภาพ")
        quad_small = project_points(ref_corners, inverse).astype(np.float32)
        area = cv2.contourArea(quad_small, oriented=True)
        if not cv2.isContourConvex(quad_small) or area <= 25:
            return finish("รูปร่างที่แปลงผิดปกติหรือกลับด้าน")
        sh, sw = src.shape[:2]
        if (quad_small < [-0.12 * sw, -0.12 * sh]).any() or (quad_small > [1.12 * sw, 1.12 * sh]).any():
            return finish("ภาพอ้างอิงมีพื้นที่ที่อยู่นอกภาพป้ายมากเกินไป กรุณาครอปให้ตรงบริเวณ")
        h_original = np.linalg.inv(s_reference) @ h_small @ s_source
        out, s_output = resized(reference, 1600)
        h_output = s_output @ h_original
        validity = cv2.warpPerspective(np.full(source.shape[:2], 255, np.uint8), h_output,
                                       (out.shape[1], out.shape[0]), flags=cv2.INTER_NEAREST)
        valid_fraction = float(np.count_nonzero(validity) / validity.size)
        result.metrics["valid_output_fraction"] = round(valid_fraction, 4)
        if valid_fraction < 0.90:
            return finish("ภาพต้นฉบับครอบคลุมพื้นที่ปลายทางไม่พอ กรุณาปรับบริเวณที่เลือก")
        result.warped = cv2.warpPerspective(source, h_output, (out.shape[1], out.shape[0]),
                                            flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
        result.homography = h_output
        result.source_quad = project_points(quad_small, np.linalg.inv(s_source))
        result.accepted = True
        return finish("ผ่านเกณฑ์การจับคู่และรูปทรง — โปรดตรวจภาพผลลัพธ์ด้วยสายตา")
    except (np.linalg.LinAlgError, ValueError, cv2.error):
        return finish("การแปลงไม่เสถียร กรุณาเปลี่ยน reference หรือระบุมุม 4 จุด")
