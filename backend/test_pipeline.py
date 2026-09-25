#!/usr/bin/env python3
"""
Phase 1: Proof of Concept Test Script
Tests SIFT Matching, Perspective Correction, and OCR Pipeline
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.core.sift_detector import SIFTDetector
from app.core.deskew import PerspectiveDeskewer
from app.core.preprocessor import ImagePreprocessor
from app.core.ocr_engine import OCREngine

def test_sift_detection():
    """Test SIFT-based plate detection."""
    print("\n=== Phase 1: SIFT Detection Test ===")

    detector = SIFTDetector()
    template_path = "app/templates/thai_plate_standard.png"

    if not Path(template_path).exists():
        print(f"Template not found at {template_path}")
        print("Creating a sample template placeholder...")
        sample_plate = np.ones((150, 340, 3), dtype=np.uint8) * 255
        cv2.rectangle(sample_plate, (10, 10), (330, 140), (0, 0, 0), 2)
        cv2.putText(sample_plate, "1KH9999", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        cv2.imwrite(template_path, sample_plate)
        print(f"Sample template created at {template_path}")

    success = detector.load_template(template_path)
    print(f"Template loaded: {success}")

    if not success:
        print("Could not load template")
        return None

    test_image_path = "app/test_sample/sample_car.jpg"
    if not Path(test_image_path).exists():
        print(f"Test image not found at {test_image_path}")
        print("Creating a synthetic test image...")
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        template = cv2.imread(template_path)
        h, w = template.shape[:2]
        test_image[150:150+h, 200:200+w] = template
        cv2.imwrite(test_image_path, test_image)
        print(f"Test image created at {test_image_path}")

    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print(f"Could not read test image from {test_image_path}")
        return None

    downscaled, scale = ImagePreprocessor.downscale_for_detection(test_image, max_width=1024)

    corners, info = detector.detect_plate(downscaled)
    print(f"Detection info: {info}")

    if corners:
        corners = ImagePreprocessor.upscale_corners(corners, scale)
        print(f"Plate detected! Corners: {corners}")
        return test_image, corners
    else:
        print("Plate not detected")
        return test_image, None

def test_deskewing(image, corners):
    """Test perspective correction."""
    print("\n=== Phase 2: Perspective Deskewing Test ===")

    if corners is None:
        print("No corners to deskew")
        return None

    deskewed, success = PerspectiveDeskewer.deskew_plate(image, corners)
    print(f"Deskewing success: {success}")

    if success:
        print(f"Deskewed image shape: {deskewed.shape}")
        cv2.imwrite("output_deskewed.jpg", deskewed)
        print("Deskewed image saved to output_deskewed.jpg")
        return deskewed
    return None

def test_preprocessing(image):
    """Test image enhancement."""
    print("\n=== Phase 3: Image Enhancement Test ===")

    enhanced = ImagePreprocessor.enhance_for_ocr(image)
    print(f"Enhanced image shape: {enhanced.shape}")
    cv2.imwrite("output_enhanced.jpg", enhanced)
    print("Enhanced image saved to output_enhanced.jpg")
    return enhanced

def test_ocr(image):
    """Test OCR recognition."""
    print("\n=== Phase 4: OCR Recognition Test ===")

    ocr = OCREngine()
    result = ocr.recognize_plate(image)

    print(f"OCR Result:")
    print(f"  Text: {result.get('text', 'N/A')}")
    print(f"  Confidence: {result.get('confidence', 0):.2%}")
    print(f"  Province: {result.get('province', 'N/A')}")
    print(f"  Valid: {result.get('is_valid', False)}")

    return result

def main():
    """Run complete pipeline test."""
    print("=" * 50)
    print("License Plate Recognition - Phase 1 POC")
    print("=" * 50)

    image, corners = test_sift_detection()

    if image is None:
        print("\nTest failed: Could not load test image")
        return

    if corners:
        deskewed = test_deskewing(image, corners)
        if deskewed is not None:
            enhanced = test_preprocessing(deskewed)
            result = test_ocr(enhanced)
    else:
        print("\nTesting with original image...")
        enhanced = test_preprocessing(image)
        result = test_ocr(enhanced)

    print("\n" + "=" * 50)
    print("Phase 1 POC Test Complete")
    print("=" * 50)

if __name__ == "__main__":
    main()
