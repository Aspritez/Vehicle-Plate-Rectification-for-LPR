"""Procedural geometry/content checks; not a real plate evaluation dataset."""
import cv2
import numpy as np

from src.detection import (Candidate, DetectorConfig, auto_rectify, detect_plate,
                           refine_candidate, select_automatic, text_evidence)
from src.geometry import project_points, rectangle


def panel(text=True):
    image = np.full((180, 440, 3), 245, np.uint8)
    cv2.rectangle(image, (4, 4), (435, 175), (20, 20, 20), 3)
    if text:
        cv2.putText(image, "AB 12345", (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(image, "TEST", (158, 146), cv2.FONT_HERSHEY_SIMPLEX, .8, (20, 20, 20), 2, cv2.LINE_AA)
    return image


def perspective_scene():
    image = np.full((600, 1000, 3), 100, np.uint8)
    corners = np.float32([[330, 280], [820, 305], [790, 495], [315, 470]])
    h = cv2.getPerspectiveTransform(rectangle(440, 180), corners)
    warped = cv2.warpPerspective(panel(), h, (1000, 600))
    mask = cv2.warpPerspective(np.full((180, 440), 255, np.uint8), h, (1000, 600))
    image[mask > 0] = warped[mask > 0]
    # Larger distractor with a border but no characters.
    cv2.rectangle(image, (30, 25), (690, 200), (240, 240, 240), -1)
    cv2.rectangle(image, (36, 31), (684, 194), (20, 20, 20), 3)
    return image, corners


def test_text_evidence_distinguishes_empty_frame_and_main_glyphs():
    assert text_evidence(panel(False))["components"] == 0
    evidence = text_evidence(panel())
    assert evidence["components"] >= 6
    assert evidence["text_score"] > .65
    assert evidence["rows"] in (1, 2)


def test_blank_frame_is_rejected_instead_of_warped():
    image = np.full((400, 800, 3), 100, np.uint8)
    image[100:280, 200:640] = panel(False)
    detection, warped, h, corners = auto_rectify(image)
    assert detection.selected is None
    assert warped is h is corners is None
    assert all(not c.accepted for c in detection.candidates)


def test_one_click_finds_text_panel_and_preserves_global_geometry():
    image, true_corners = perspective_scene()
    detection, warped, h, corners = auto_rectify(image)
    assert detection.selected is not None, [(c.reason, c.score, c.metrics) for c in detection.candidates]
    assert detection.selected.metrics["components"] >= 4
    assert np.max(np.linalg.norm(corners - true_corners, axis=1)) < 16
    assert warped.shape == (320, 640, 3)
    assert np.allclose(project_points(corners, h), rectangle(640, 320), atol=.01)
    assert any("text" in source for candidate in detection.candidates for source in candidate.sources)


def test_ambiguous_candidates_are_not_arbitrarily_selected():
    first = Candidate((0, 0, 400, 180), rectangle(400, 180), .80, accepted=True)
    second = Candidate((500, 0, 400, 180), rectangle(400, 180) + [500, 0], .78, accepted=True)
    selected, reason = select_automatic([first, second], DetectorConfig())
    assert selected is None
    assert "หลายบริเวณ" in reason
    first.accepted = False
    selected, _ = select_automatic([first, second], DetectorConfig())
    assert selected is second


def test_text_seed_without_plate_border_cannot_invent_a_quad():
    image = np.full((300, 640, 3), 245, np.uint8)
    cv2.putText(image, "TEST 123", (130, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3)
    seed = Candidate((120, 140, 230, 50), None, .9, {"1000:text-21"})
    refined = refine_candidate(image, seed, DetectorConfig())
    assert not refined.accepted


def test_two_equally_supported_panels_require_manual_selection():
    image = np.full((600, 1100, 3), 100, np.uint8)
    image[100:280, 50:490] = panel()
    image[330:510, 610:1050] = panel()
    result = detect_plate(image)
    assert result.selected is None
    assert "หลายบริเวณ" in result.reason
