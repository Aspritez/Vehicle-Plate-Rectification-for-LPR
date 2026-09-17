"""Bounded plate proposals and geometric/content validation, without training.

Scores are heuristics, not probabilities. Coordinates use the full input image.
"""
from dataclasses import dataclass, field
from time import perf_counter
import cv2
import numpy as np
from src.geometry import ordered_quad, project_points, rectangle, rectify_corners
from src.imaging import resized


@dataclass(frozen=True)
class DetectorConfig:
    low: int = 40
    high: int = 140
    max_candidates: int = 12
    max_refinement_side: int = 2000
    max_refinement_quads: int = 16
    min_text_score: float = 0.46
    min_components: int = 4
    min_edge_support: float = 0.55
    min_score: float = 0.64
    ambiguity_margin: float = 0.045
    content_proposals: bool = False
    max_proposal_reviews: int = 200
    min_aspect: float = 0.7
    canonical_text_aspect: bool = False
    neutral_proposals: bool = False
    appearance_weight: float = 0.0
    min_contrast: float = 0.0
    min_neutral: float = 0.0
    min_main_height: float = 0.0


def web_detector_config(low=40, high=140):
    """Frozen Thai-plate tuning profile used by the single-image web workflow."""
    return DetectorConfig(
        low=low, high=high, content_proposals=True, max_candidates=24,
        max_proposal_reviews=250, min_aspect=.25, canonical_text_aspect=True,
        neutral_proposals=True, appearance_weight=.25, min_contrast=55,
        min_neutral=.25, min_main_height=.22, min_score=.90,
    )


@dataclass
class Candidate:
    box: tuple[int, int, int, int]
    corners: np.ndarray | None
    score: float
    sources: set[str] = field(default_factory=set)
    metrics: dict = field(default_factory=dict)
    accepted: bool = False
    reason: str = "ยังไม่ได้ตรวจละเอียด"


@dataclass
class DetectionResult:
    candidates: list[Candidate]
    diagnostics: dict[str, np.ndarray]
    selected: Candidate | None = None
    reason: str = ""
    elapsed_ms: float = 0.0


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return intersection / max(1, aw * ah + bw * bh - intersection)


def _quad(contour):
    perimeter = cv2.arcLength(contour, True)
    for epsilon in (0.012, 0.02, 0.03, 0.045):
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(approx) == 4:
            try:
                return ordered_quad(approx.reshape(4, 2))
            except ValueError:
                pass
    return None


def _shape(contour):
    area = abs(cv2.contourArea(contour))
    rot = cv2.minAreaRect(contour)
    rectangularity = min(1., area / max(1., rot[1][0] * rot[1][1]))
    solidity = area / max(1., cv2.contourArea(cv2.convexHull(contour)))
    # minAreaRect is ONLY a scoring feature, never a source of perspective corners.
    aspect = max(rot[1]) / max(1., min(rot[1]))
    aspect_score = float(np.exp(-.5 * (np.log(aspect / 2.) / .8) ** 2))
    return .45 * rectangularity + .25 * solidity + .30 * aspect_score


def _merge_candidates(candidates, limit):
    merged = []
    for item in sorted(candidates, key=lambda c: c.score, reverse=True):
        duplicate = next((old for old in merged if box_iou(item.box, old.box) > .58), None)
        if duplicate is None:
            merged.append(item)
        else:
            duplicate.sources |= item.sources
            if duplicate.corners is None and item.corners is not None:
                duplicate.corners = item.corners
    merged.sort(key=lambda c: c.score + min(.06, .015 * len(c.sources)), reverse=True)
    return merged[:limit]


def propose_candidates(image, config):
    candidates, diagnostics = [], {}
    for max_side, use_clahe in ((1000, False), (1400, True)):
        small, scale = resized(image, max_side)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        if use_clahe:
            gray = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, config.low, config.high)
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
        masks = [("edges", edges), ("closing", closing), ("threshold", threshold)]
        if config.neutral_proposals:
            hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
            neutral = cv2.inRange(hsv, (0, 0, 90), (179, 100, 255))
            masks.append(("neutral", cv2.morphologyEx(neutral, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))))
        h, w = gray.shape
        for factor in (.018, .035):
            kw = max(9, int(w * factor) | 1)
            kh = max(3, int(kw / 3) | 1)
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                                       cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)))
            mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            groups = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, kw // 2), 3)))
            masks.append((f"text-{kw}", groups))
            if factor == .035:
                diagnostics[f"{max_side} / Black-hat groups"] = groups
        diagnostics[f"{max_side} / Canny"] = edges
        for name, mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:600]:
                area = abs(cv2.contourArea(contour))
                x, y, bw, bh = cv2.boundingRect(contour)
                if not max(65, h * w * .00008) < area < h * w * .92:
                    continue
                if min(bw, bh) < 16 or not config.min_aspect < bw / bh < 7:
                    continue
                density = np.count_nonzero(edges[y:y + bh, x:x + bw]) / (bw * bh)
                if not .012 < density < .60:
                    continue
                shape = _shape(contour)
                if shape < .5:
                    continue
                # Text-group boxes are search seeds; their corners are not plate corners.
                corners = None if name.startswith("text") else _quad(contour)
                x0, y0 = int(x / scale[0, 0]), int(y / scale[1, 1])
                x1 = min(image.shape[1], int(np.ceil((x + bw) / scale[0, 0])))
                y1 = min(image.shape[0], int(np.ceil((y + bh) / scale[1, 1])))
                if corners is not None:
                    corners = project_points(corners, np.linalg.inv(scale)).astype(np.float32)
                candidates.append(Candidate((x0, y0, x1 - x0, y1 - y0), corners,
                                             .75 * shape + .25 * min(density / .13, 1), {f"{max_side}:{name}"}))
    if config.content_proposals:
        pool = _merge_candidates(candidates, config.max_proposal_reviews)
        for candidate in pool:
            x, y, w, h = candidate.box
            if candidate.corners is not None:
                # Normalize text layout for skewed/anisotropically resized input.
                h_small = cv2.getPerspectiveTransform(candidate.corners, rectangle(320, 160))
                preview = cv2.warpPerspective(image, h_small, (320, 160))
            else:
                preview = cv2.resize(image[y:y+h, x:x+w], (320, 160))
            content = text_evidence(preview)
            appearance = appearance_evidence(preview)
            # Actual row structure has more weight than repeated pavement edges.
            base = .35 * candidate.score + .65 * content['text_score']
            candidate.score = (1-config.appearance_weight) * base + config.appearance_weight * appearance['appearance_score']
            candidate.metrics['proposal_text_score'] = content['text_score']
        pool.sort(key=lambda c: c.score, reverse=True)
        return pool[:config.max_candidates], diagnostics
    return _merge_candidates(candidates, config.max_candidates), diagnostics


def text_evidence(image):
    """Aligned main components, not OCR; small marks need not join main rows."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    gray, _ = resized(gray, 600)
    h, w = gray.shape
    best = {"text_score": 0., "components": 0, "rows": 0, "text_span": 0., "main_height": 0.}
    for threshold in (cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                      cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)):
        for binary in (threshold, 255 - threshold):
            _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            components = []
            for x, y, bw, bh, area in stats[1:]:
                if x < .025 * w or y < .025 * h or x + bw > .975 * w or y + bh > .975 * h:
                    continue
                if not .11 * h < bh < .80 * h or not .008 * w < bw < .24 * w:
                    continue
                if not .07 < bw / bh < 1.5 or not .08 < area / (bw * bh) < .95:
                    continue
                components.append((float(x), float(y), float(bw), float(bh)))
            if not 3 <= len(components) <= 100:
                continue
            remaining, rows = list(components), []
            for _ in range(2):
                if not remaining:
                    break
                groups = []
                for _, ay, _, ah in remaining:
                    groups.append([c for c in remaining if .50 < c[3] / ah < 1.9
                                   and abs((c[1] + c[3] / 2) - (ay + ah / 2)) < .40 * max(ah, c[3])])
                row = max(groups, key=len)
                if len(row) < 3:
                    break
                rows.append(row)
                remaining = [c for c in remaining if c not in row]
            if not rows:
                continue
            chosen = [c for row in rows for c in row]
            spans, regularity = [], []
            for row in rows:
                spans.append((max(c[0] + c[2] for c in row) - min(c[0] for c in row)) / w)
                heights = np.array([c[3] for c in row])
                regularity.append(max(0., 1 - np.std(heights) / max(1, np.mean(heights))))
            span = max(spans)
            score = (.45 * min(len(chosen) / 6, 1) + .30 * min(span / .65, 1)
                     + .25 * float(np.mean(regularity))) * min(span / .35, 1)
            if score > best["text_score"]:
                best = {"text_score": round(float(score), 4), "components": len(chosen),
                        "rows": len(rows), "text_span": round(float(span), 4),
                        "main_height": round(max(float(np.median([c[3] for c in row])) / h for row in rows), 4)}
    return best


def appearance_evidence(image):
    """Soft appearance cue for mostly neutral plate backgrounds, not a color rule."""
    h, w = image.shape[:2]
    inner = image[max(1, h//20):h-max(1,h//20), max(1,w//20):w-max(1,w//20)]
    gray = cv2.cvtColor(inner, cv2.COLOR_RGB2GRAY)
    saturation = cv2.cvtColor(inner, cv2.COLOR_RGB2HSV)[:, :, 1]
    contrast = float(np.percentile(gray, 90) - np.percentile(gray, 10))
    neutral = float(np.mean(saturation < 60))
    return {"contrast": round(contrast, 3), "neutral_fraction": round(neutral, 4),
            "appearance_score": round(.6 * min(contrast / 150, 1) + .4 * neutral, 4)}


def edge_support(edges, corners, distance=None):
    if distance is None:
        distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    sides = []
    tolerance = max(1.8, min(edges.shape) * .006)
    for start, end in zip(corners, np.roll(corners, -1, axis=0)):
        samples = np.linspace(start, end, 90)
        xs = np.clip(np.rint(samples[:, 0]).astype(int), 0, edges.shape[1] - 1)
        ys = np.clip(np.rint(samples[:, 1]).astype(int), 0, edges.shape[0] - 1)
        sides.append(float(np.mean(distance[ys, xs] <= tolerance)))
    return float(np.mean(sides)), min(sides)


def fit_border_lines(edges, corners, cloud=None):
    """Refine contour edges using nearby observed edge pixels, never an invented box."""
    if cloud is None:
        ys, xs = np.nonzero(edges)
        cloud = np.column_stack([xs, ys]).astype(np.float32)
    lines = []
    band = max(2., min(edges.shape) * .009)
    for start, end in zip(corners, np.roll(corners, -1, axis=0)):
        vector = end - start
        length = np.linalg.norm(vector)
        direction = vector / max(1, length)
        relative = cloud - start
        along = relative @ direction
        normal = np.abs(relative[:, 0] * direction[1] - relative[:, 1] * direction[0])
        pixels = cloud[(normal < band) & (along > .06 * length) & (along < .94 * length)]
        if len(pixels) < 15:
            return corners
        vx, vy, px, py = cv2.fitLine(pixels, cv2.DIST_HUBER, 0, .01, .01).ravel()
        lines.append(np.array([-vy, vx, vy * px - vx * py]))
    refined = []
    for i in range(4):
        point = np.cross(lines[i - 1], lines[i])
        if abs(point[2]) < 1e-6:
            return corners
        refined.append(point[:2] / point[2])
    refined = np.float32(refined)
    if np.max(np.linalg.norm(refined - corners, axis=1)) > 3 * band:
        return corners
    try:
        return ordered_quad(refined, edges.shape)
    except ValueError:
        return corners


def refine_candidate(image, candidate, config):
    x, y, w, h = candidate.box
    px, py = max(8, round(w * .22)), max(8, round(h * .55))
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(image.shape[1], x + w + px), min(image.shape[0], y + h + py)
    crop, scale = resized(image[y0:y1, x0:x1], config.max_refinement_side)
    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    edges = cv2.Canny(gray, config.low, config.high)
    distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    ey, ex = np.nonzero(edges)
    edge_cloud = np.column_stack([ex, ey]).astype(np.float32)
    masks = [edges, cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)),
             cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]]
    quads = []
    if candidate.corners is not None:
        quads.append(project_points(candidate.corners - [x0, y0], scale).astype(np.float32))
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:100]:
            quad = _quad(contour)
            if quad is not None:
                quads.append(quad)
    best, seen = None, []
    for quad in quads:
        area = cv2.contourArea(quad)
        target_area = w * h * scale[0, 0] * scale[1, 1]
        if not .28 * target_area < area < 3.7 * target_area:
            continue
        if any(np.mean(np.linalg.norm(quad - old, axis=1)) < 3 for old in seen):
            continue
        full_quad = project_points(quad, np.linalg.inv(scale)) + [x0, y0]
        if box_iou(cv2.boundingRect(full_quad.astype(np.float32)), candidate.box) < .25:
            continue
        if len(seen) >= config.max_refinement_quads:
            break
        seen.append(quad)
        support, weakest = edge_support(edges, quad, distance)
        if support < .4 or weakest < .2:
            continue
        fitted = fit_border_lines(edges, quad, edge_cloud)
        fitted_support, fitted_weakest = edge_support(edges, fitted, distance)
        if fitted_support >= support and fitted_weakest >= weakest:
            quad, support, weakest = fitted, fitted_support, fitted_weakest
        if support < .4 or weakest < .2:
            continue
        lengths = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
        aspect = (lengths[0] + lengths[2]) / max(1, lengths[1] + lengths[3])
        if not config.min_aspect < aspect < 6:
            continue
        # Preliminary warp checks content; its apparent aspect is not the final ratio.
        ph = 240 if config.canonical_text_aspect else max(60, min(360, round(480 / aspect)))
        transform = cv2.getPerspectiveTransform(quad, rectangle(480, ph))
        trial = cv2.warpPerspective(crop, transform, (480, ph))
        evidence = text_evidence(trial)
        appearance = appearance_evidence(trial)
        shape = _shape(quad)
        score = .22 * shape + .40 * evidence["text_score"] + .28 * support + .10 * min(len(candidate.sources) / 4, 1)
        score = (1-config.appearance_weight) * score + config.appearance_weight * appearance['appearance_score']
        metrics = {**evidence, **appearance, "shape_score": round(shape, 4), "edge_support": round(support, 4),
                   "weakest_edge_support": round(weakest, 4), "agreement": len(candidate.sources),
                   "refinement_size": [crop.shape[1], crop.shape[0]]}
        accepted = (evidence["components"] >= config.min_components
                    and evidence["text_score"] >= config.min_text_score
                    and support >= config.min_edge_support and weakest >= .30 and score >= config.min_score
                    and appearance['contrast'] >= config.min_contrast and appearance['neutral_fraction'] >= config.min_neutral
                    and evidence['main_height'] >= config.min_main_height)
        if evidence["components"] < config.min_components or evidence["text_score"] < config.min_text_score:
            reason = "หลักฐานกลุ่มตัวอักษรไม่เพียงพอ"
        elif support < config.min_edge_support or weakest < .30:
            reason = "ขอบป้ายบางด้านไม่ชัด"
        elif score < config.min_score:
            reason = "คะแนนรวมยังไม่ผ่านเกณฑ์"
        elif appearance['contrast'] < config.min_contrast or appearance['neutral_fraction'] < config.min_neutral or evidence['main_height'] < config.min_main_height:
            reason = "ลักษณะพื้นหลังหรือขนาดกลุ่มตัวอักษรไม่ผ่านเกณฑ์"
        else:
            reason = "ผ่านเกณฑ์รูปทรง ขอบ และกลุ่มตัวอักษร"
        if best is None or (accepted, score) > (best.accepted, best.score):
            try:
                full_quad = ordered_quad(project_points(quad, np.linalg.inv(scale)) + [x0, y0], image.shape)
            except ValueError:
                continue
            best = Candidate(cv2.boundingRect(full_quad), full_quad, float(score), candidate.sources.copy(), metrics, accepted, reason)
    return best or Candidate(candidate.box, None, 0., candidate.sources.copy(), {}, False, "ไม่พบมุมป้ายที่มีขอบจริงรองรับครบ")


def select_automatic(candidates, config):
    eligible = sorted((c for c in candidates if c.accepted and c.corners is not None), key=lambda c: c.score, reverse=True)
    if not eligible:
        return None, "ยังไม่มีบริเวณที่ผ่านเกณฑ์อัตโนมัติ กรุณาระบุมุมป้ายบนภาพเต็ม"
    if len(eligible) > 1 and eligible[0].score - eligible[1].score < config.ambiguity_margin:
        return None, "พบหลายบริเวณที่มีคะแนนใกล้กัน กรุณาเลือกป้ายโดยระบุมุมบนภาพเต็ม"
    return eligible[0], "ปรับป้ายอัตโนมัติแล้ว โปรดตรวจว่าเลือกป้ายและสัดส่วนถูกต้อง"


def detect_plate(image, config=None):
    config = config or DetectorConfig()
    start = perf_counter()
    proposals, diagnostics = propose_candidates(image, config)
    refined = [refine_candidate(image, candidate, config) for candidate in proposals]
    refined.sort(key=lambda c: (c.accepted, c.score), reverse=True)
    final = []
    for candidate in refined:
        duplicate = next((old for old in final if box_iou(candidate.box, old.box) > .55), None)
        if duplicate is None:
            final.append(candidate)
        else:
            duplicate.sources |= candidate.sources
    selected, reason = select_automatic(final, config)
    return DetectionResult(final, diagnostics, selected, reason, round((perf_counter() - start) * 1000, 2))


def detect_candidates(image, low=40, high=140):
    """Compatibility for reference ROI selection; candidate does not mean accepted."""
    result = detect_plate(image, DetectorConfig(low=low, high=high))
    return result.candidates, result.diagnostics["1400 / Canny"]


def auto_rectify(image, width=640, aspect=2., rotate=False, config=None):
    """One call: detection, rejection/selection, full-source image warp."""
    detection = detect_plate(image, config)
    if detection.selected is None:
        return detection, None, None, None
    warped, h, quad = rectify_corners(image, detection.selected.corners, width, aspect, rotate)
    return detection, warped, h, quad
