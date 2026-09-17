"""Small procedural checks of code correctness, not a plate accuracy dataset."""

from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image

from src.detection import detect_candidates
from src.geometry import ordered_quad, project_points, rectangle, rectify_corners
from src.imaging import decode_image, png_bytes, prepare_ocr, resized
from src.matching import estimate_correspondences, match_rectify


def test_decode_rejects_non_images_and_unsupported_formats():
    with pytest.raises(ValueError):
        decode_image(b"not an image")
    stream = BytesIO()
    Image.new("RGB", (32, 32)).save(stream, "GIF")
    with pytest.raises(ValueError, match="JPG"):
        decode_image(stream.getvalue())


def test_exif_orientation_and_alpha_are_normalized():
    stream = BytesIO()
    im = Image.new("RGB", (48, 24), "red")
    exif = Image.Exif()
    exif[274] = 6
    im.save(stream, "JPEG", exif=exif)
    assert decode_image(stream.getvalue()).shape == (48, 24, 3)
    stream = BytesIO()
    Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(stream, "PNG")
    assert np.all(decode_image(stream.getvalue()) == 255)


def test_pixel_limit_before_decoding(monkeypatch):
    monkeypatch.setattr("src.imaging.MAX_PIXELS", 400)
    with pytest.raises(ValueError, match="24"):
        decode_image(png_bytes(np.zeros((21, 21, 3), np.uint8)))


def test_resize_uses_exact_independent_axis_scales():
    original = np.zeros((301, 1003, 3), np.uint8)
    small, scale = resized(original, 700)
    points = np.array([[1002, 300], [100, 80]], float)
    assert small.shape[1] == 700
    assert np.allclose(project_points(project_points(points, scale), np.linalg.inv(scale)), points)
    assert scale[1, 1] == small.shape[0] / 301


@pytest.mark.parametrize("points", [
    [[0, 0], [0, 0], [50, 50], [0, 50]],
    [[0, 0], [10, 0], [20, 0], [30, 0]],
    [[0, 0], [100, 0], [10, 10], [0, 100]],
    [[0, 0], [20, 0], [20, float("nan")], [0, 20]],
])
def test_invalid_quads_rejected(points):
    with pytest.raises(ValueError):
        ordered_quad(np.array(points))


def test_manual_homography_maps_known_corners():
    image = np.zeros((300, 600, 3), np.uint8)
    corners = np.float32([[75, 50], [520, 80], [480, 240], [45, 210]])
    output, h, ordered = rectify_corners(image, corners[[2, 0, 3, 1]], width=640, aspect=2)
    assert output.shape == (320, 640, 3)
    assert np.allclose(project_points(ordered, h), rectangle(640, 320), atol=1e-3)


def test_outside_corners_and_excessive_output_rejected():
    image = np.zeros((100, 100, 3), np.uint8)
    with pytest.raises(ValueError):
        rectify_corners(image, rectangle(110, 100))
    with pytest.raises(ValueError):
        rectify_corners(image, rectangle(100, 100), width=2000, aspect=.8)


def test_ransac_recovers_geometry_with_outliers():
    rng = np.random.default_rng(17)
    source = rng.uniform([0, 0], [600, 300], size=(80, 2))
    truth = np.array([[1.1, .12, 24], [-.08, .92, 18], [.0003, -.0002, 1]])
    target = project_points(source, truth) + rng.normal(0, .12, (80, 2))
    target[-20:] = rng.uniform([0, 0], [600, 300], size=(20, 2))
    cv2.setRNGSeed(17)
    found, inliers = estimate_correspondences(source, target, threshold=1.5)
    error = np.linalg.norm(project_points(source[:60], found) - project_points(source[:60], truth), axis=1)
    assert np.median(error) < .25
    assert inliers[:60].sum() >= 55
    assert inliers[-20:].sum() <= 2


@pytest.mark.parametrize("method", ["SIFT", "ORB"])
def test_matching_blank_images_fails_cleanly(method):
    blank = np.full((120, 300, 3), 255, np.uint8)
    result = match_rectify(blank, blank, method=method)
    assert not result.accepted
    assert result.homography is None and result.warped is None
    assert result.metrics["source_keypoints"] == 0


def procedural_texture(seed=42):
    """In-memory generic geometric pattern, not a saved image or plate fixture."""
    rng = np.random.default_rng(seed)
    image = np.full((260, 600, 3), 230, np.uint8)
    for _ in range(160):
        x, y = rng.integers([10, 10], [580, 240])
        color = tuple(int(v) for v in rng.integers(10, 180, 3))
        cv2.circle(image, (int(x), int(y)), int(rng.integers(2, 9)), color, -1)
    return image


@pytest.mark.parametrize("method", ["SIFT", "ORB"])
def test_feature_pipeline_estimates_actual_perspective(method):
    reference = procedural_texture()
    destination = np.float32([[35, 25], [565, 10], [585, 235], [15, 245]])
    forward = cv2.getPerspectiveTransform(rectangle(600, 260), destination)
    source = cv2.warpPerspective(reference, forward, (600, 260), borderValue=(230, 230, 230))
    result = match_rectify(source, reference, method=method)
    assert result.accepted, result.reason
    recovered = project_points(destination, result.homography)
    assert np.max(np.linalg.norm(recovered - rectangle(600, 260), axis=1)) < 6
    assert result.warped.shape == reference.shape


def test_unrelated_patterns_do_not_produce_accepted_output():
    result = match_rectify(procedural_texture(42), procedural_texture(99))
    assert not result.accepted
    assert result.warped is None


def test_detector_handles_blank_input_without_guessing_corners():
    found, edges = detect_candidates(np.full((200, 400, 3), 255, np.uint8))
    assert found == []
    assert edges.shape == (200, 400)


def test_ocr_export_preserves_dimensions_and_binary_values():
    image = procedural_texture()
    original = image.copy()
    output = prepare_ocr(image, "ขาวดำ (Adaptive threshold)", True, True)
    assert output.shape == image.shape[:2]
    assert set(np.unique(output)) <= {0, 255}
    assert np.array_equal(image, original)
    assert Image.open(BytesIO(png_bytes(output))).size == (600, 260)
