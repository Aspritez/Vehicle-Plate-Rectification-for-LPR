import cv2
import numpy as np
from typing import Tuple
from .config import CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE, BILATERAL_FILTER_D, BILATERAL_FILTER_SIGMA, DEFAULT_DOWNSCALE_WIDTH

class ImagePreprocessor:
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit: float = None, tile_size: int = None) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        if clip_limit is None:
            clip_limit = CLAHE_CLIP_LIMIT
        if tile_size is None:
            tile_size = CLAHE_TILE_SIZE
        gray = ImagePreprocessor.to_grayscale(image)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(gray)

    @staticmethod
    def denoise(image: np.ndarray, d: int = None, sigma_color: int = None, sigma_space: int = None) -> np.ndarray:
        """Apply bilateral filter for noise reduction while preserving edges."""
        if d is None:
            d = BILATERAL_FILTER_D
        if sigma_color is None:
            sigma_color = BILATERAL_FILTER_SIGMA
        if sigma_space is None:
            sigma_space = BILATERAL_FILTER_SIGMA
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    @staticmethod
    def apply_otsu_threshold(image: np.ndarray) -> Tuple[np.ndarray, int]:
        """Apply Otsu's thresholding."""
        gray = ImagePreprocessor.to_grayscale(image)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary, _

    @staticmethod
    def apply_adaptive_threshold(image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding."""
        gray = ImagePreprocessor.to_grayscale(image)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    @staticmethod
    def enhance_for_ocr(image: np.ndarray, scale: int = 2) -> np.ndarray:
        """Grayscale, enlarge and lightly equalise a deskewed plate for OCR.

        The plate is deliberately not binarised: on low-resolution CCTV crops thresholding
        merges or drops strokes, and reading the grayscale image gave clearly better results.
        """
        gray = ImagePreprocessor.to_grayscale(image)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)

    @staticmethod
    def downscale_for_detection(image: np.ndarray, max_width: int = None) -> Tuple[np.ndarray, float]:
        """Downscale image for faster SIFT detection."""
        if max_width is None:
            max_width = DEFAULT_DOWNSCALE_WIDTH
        h, w = image.shape[:2]
        if w <= max_width:
            return image, 1.0

        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)

        downscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return downscaled, scale

    @staticmethod
    def upscale_corners(corners, scale: float):
        """Upscale corner coordinates back to original image size."""
        if corners is None or scale == 1.0:
            return corners
        return [[pt[0] / scale, pt[1] / scale] for pt in corners]
