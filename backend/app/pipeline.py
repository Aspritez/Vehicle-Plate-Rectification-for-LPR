"""The plate-reading pipeline, independent of any web framework.

Used by the FastAPI routes (backend/app/api/routes.py) and by the Streamlit app (streamlit_app.py).
All results are plain dicts; images are numpy arrays under "images".
"""
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .core.config import PLATE_STANDARD_RATIO
from .core.deskew import PerspectiveDeskewer
from .core.ocr_engine import OCREngine
from .core.plate_from_box import plate_from_box
from .core.plate_locator import SIFTPlateLocator
from .core.plate_refiner import refine_quad
from .core.preprocessor import ImagePreprocessor
from .core.super_resolution import SCALE as SR_SCALE, SuperResolver

MODEL_PATH = Path(__file__).resolve().parent / "models" / "plate_locator.joblib"

MIN_ROTATED_CONFIDENCE = 0.4  # OCR confidence needed to prefer another plate orientation
SR_MIN_GAIN = 0.15            # how much better (OCR quality score) the super-resolved reading must be to be used
SR_SKIP_CONFIDENCE = 0.9      # a valid reading at least this confident, with a province, is not second-guessed
SR_REGION_MARGIN = 0.25       # context kept around the plate when cropping it for super-resolution


def default_corners(width: int, height: int) -> List[List[float]]:
    """Centered box with a standard plate aspect ratio, a starting point for manual adjustment."""
    box_w = width * 0.4
    box_h = min(box_w / PLATE_STANDARD_RATIO, height * 0.4)
    box_w = box_h * PLATE_STANDARD_RATIO
    x0, y0 = (width - box_w) / 2, (height - box_h) / 2
    return [[x0, y0], [x0 + box_w, y0], [x0 + box_w, y0 + box_h], [x0, y0 + box_h]]


class PlateReader:
    """Loads the models once and reads plates. Requests are processed one at a time: EasyOCR is not
    documented as thread-safe and the work is CPU heavy."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.locator: Optional[SIFTPlateLocator] = None
        self.locator_error: Optional[str] = None
        try:
            self.locator = SIFTPlateLocator.load(model_path)
        except Exception as e:
            self.locator_error = (f"Could not load plate locator model from {model_path} ({e}). "
                                  f"Run 'python train_locator.py' in the backend folder.")
            print(f"Warning: {self.locator_error}")

        self.ocr_engine = OCREngine()
        self.super_resolver = SuperResolver()
        self._lock = threading.Lock()

    # ---------- public API ----------
    def recognize(self, image: np.ndarray) -> Dict:
        """Locate the plate automatically, then deskew and read it."""
        start = time.time()
        height, width = image.shape[:2]
        if self.locator is None:
            return self._not_found(self.locator_error, width, height, start)

        with self._lock:
            corners, info = self.locator.locate(image)
            if corners is None:
                return self._not_found(info.get("error", "No plate found"), width, height, start)
            return self._run(image, self._snap(image, corners), start)

    def recognize_box(self, image: np.ndarray, box) -> Dict:
        """Find the plate inside a rough rectangle [x0, y0, x1, y1] drawn by the user, then read it."""
        start = time.time()
        x0, y0, x1, y1 = box
        with self._lock:
            corners = plate_from_box(image, box)
            if corners is None:
                return {
                    "status": "no_plate_detected",
                    "plate_found": False,
                    "message": "Could not find a plate inside the box. Draw it closer around the plate, or drag the corners",
                    "suggested_corners": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                    "processing_time_ms": int((time.time() - start) * 1000),
                }
            return self._run(image, corners, start)

    def recognize_corners(self, image: np.ndarray, corners: List[List[float]], snap: bool = False) -> Dict:
        """Read the plate at corners placed by the user (optionally snapped onto the plate's edges)."""
        start = time.time()
        with self._lock:
            return self._run(image, self._snap(image, corners) if snap else corners, start)

    # ---------- internals ----------
    def _not_found(self, message: str, width: int, height: int, start: float) -> Dict:
        return {
            "status": "no_plate_detected",
            "plate_found": False,
            "message": message,
            "suggested_corners": default_corners(width, height),
            "processing_time_ms": int((time.time() - start) * 1000),
        }

    @staticmethod
    def _snap(image: np.ndarray, corners: List[List[float]]) -> List[List[float]]:
        """Move rough corners onto the plate's real edges (unchanged if no trustworthy fit is found)."""
        ordered = PerspectiveDeskewer.arrange_corners(corners)
        refined, _ = refine_quad(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), ordered)
        return refined

    def _read_corners(self, image: np.ndarray, corners: List[List[float]], use_sr: bool = False):
        """Deskew the plate at these corners, remove residual tilt, prepare it for OCR and read it.

        With use_sr the plate region is first enlarged 4x by super-resolution (at native resolution,
        before the perspective warp). Returns (deskewed, enhanced, ocr_result) or None."""
        source, points = image, np.asarray(PerspectiveDeskewer.arrange_corners(corners), np.float32)
        if use_sr:
            side = float(np.linalg.norm(points[1] - points[0]))
            low = np.maximum(points.min(0) - SR_REGION_MARGIN * side, 0).astype(int)
            high = np.minimum(points.max(0) + SR_REGION_MARGIN * side, [image.shape[1], image.shape[0]]).astype(int)
            crop = image[low[1]:high[1], low[0]:high[0]]
            if crop.size == 0 or not self.super_resolver.can_upscale(crop):
                return None
            source, points = self.super_resolver.upscale(crop), (points - low) * SR_SCALE

        deskewed, ok = PerspectiveDeskewer.deskew_plate(source, points.tolist())
        if not ok:
            return None
        deskewed = PerspectiveDeskewer.straighten_text(deskewed)
        enhanced = ImagePreprocessor.enhance_for_ocr(deskewed)
        return deskewed, enhanced, self.ocr_engine.recognize_plate(enhanced)

    def _run(self, image: np.ndarray, corners: List[List[float]], start: float) -> Dict:
        """Deskew the plate at the given corners, enhance it and run OCR."""
        ocr = self.ocr_engine
        visualization = image.copy()
        cv2.polylines(visualization, [np.array(corners, dtype=np.int32)], True, (0, 255, 0), 2)

        reading = self._read_corners(image, corners)
        if reading is None:
            return {
                "status": "deskew_failed",
                "plate_found": True,
                "corners": corners,
                "message": "Could not straighten the plate from these corners. Adjust the corners and try again.",
                "processing_time_ms": int((time.time() - start) * 1000),
            }
        deskewed_plate, enhanced_plate, ocr_result = reading

        if not ocr_result["is_valid"]:
            # The start corner may be wrong (plate came out sideways or upside down): try the other three.
            # A different orientation is only adopted when it reads as a real Thai plate with good
            # confidence; otherwise (e.g. a foreign plate the Thai reader cannot parse) it is kept as is.
            ordered = PerspectiveDeskewer.arrange_corners(corners)
            best = None
            for shift in (1, 2, 3):
                rolled = np.roll(ordered, -shift, axis=0).tolist()
                candidate = self._read_corners(image, rolled)
                if candidate is None:
                    continue
                result = candidate[2]
                if (result["is_valid"] and result["confidence"] >= max(MIN_ROTATED_CONFIDENCE, ocr_result["confidence"] + 0.15)
                        and (best is None or ocr.quality(result) > ocr.quality(best[3]))):
                    best = (rolled, candidate[0], candidate[1], result)
            if best is not None:
                corners, deskewed_plate, enhanced_plate, ocr_result = best

        # Small blurry plates: try again on a super-resolved copy, keep it only if it reads clearly better.
        super_resolution_used = False
        confident = (ocr_result["is_valid"] and ocr_result["confidence"] >= SR_SKIP_CONFIDENCE and ocr_result["province"])
        if not confident and not ocr_result.get("error") and self.super_resolver.available:
            enlarged = self._read_corners(image, corners, use_sr=True)
            if enlarged is not None and ocr.quality(enlarged[2]) > ocr.quality(ocr_result) + SR_MIN_GAIN:
                deskewed_plate, enhanced_plate, ocr_result = enlarged
                super_resolution_used = True

        if ocr_result.get("error"):
            return {
                "status": "ocr_failed",
                "plate_found": True,
                "corners": corners,
                "message": f"OCR failed: {ocr_result['error']}",
                "processing_time_ms": int((time.time() - start) * 1000),
            }

        return {
            "status": "success",
            "processing_time_ms": int((time.time() - start) * 1000),
            "plate_found": True,
            "corners": corners,
            "super_resolution_used": super_resolution_used,
            "images": {"deskewed": deskewed_plate, "enhanced": enhanced_plate, "visualization": visualization},
            "ocr_result": {
                "text": ocr_result["text"],
                "province": ocr_result["province"],
                "confidence": ocr_result["confidence"],
                "is_valid": ocr_result["is_valid"],
            },
        }
