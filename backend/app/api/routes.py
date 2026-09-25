import cv2
import base64
import time
from pathlib import Path
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from typing import List, Optional

from ..core.plate_locator import SIFTPlateLocator
from ..core.plate_refiner import refine_quad
from ..core.plate_from_box import plate_from_box
from ..core.super_resolution import SuperResolver, SCALE as SR_SCALE
from ..core.deskew import PerspectiveDeskewer
from ..core.preprocessor import ImagePreprocessor
from ..core.ocr_engine import OCREngine
from ..core.config import PLATE_STANDARD_RATIO

router = APIRouter(prefix="/api/v1", tags=["recognition"])

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "plate_locator.joblib"

locator: Optional[SIFTPlateLocator] = None
locator_error: Optional[str] = None
try:
    locator = SIFTPlateLocator.load(MODEL_PATH)
except Exception as e:
    locator_error = (f"Could not load plate locator model from {MODEL_PATH} ({e}). "
                     f"Run 'python train_locator.py' in the backend folder.")
    print(f"Warning: {locator_error}")

ocr_engine = OCREngine()
super_resolver = SuperResolver()

MIN_ROTATED_CONFIDENCE = 0.4  # OCR confidence needed to prefer another plate orientation
SR_MIN_GAIN = 0.15            # how much better (OCR quality score) the super-resolved reading must be to be used
SR_SKIP_CONFIDENCE = 0.9      # a valid reading at least this confident, with a province, is not second-guessed
SR_REGION_MARGIN = 0.25       # context kept around the plate when cropping it for super-resolution


class DeskewRequest(BaseModel):
    image_base64: str
    corners: List[List[float]]
    snap: bool = False  # snap the corners onto the plate's edges before recognising


class BoxRequest(BaseModel):
    image_base64: str
    box: List[float]  # [x0, y0, x1, y1]: a rough rectangle drawn around the plate


def image_to_base64(image: np.ndarray) -> str:
    """Convert image to base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"


def decode_image(data: bytes) -> Optional[np.ndarray]:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def default_corners(width: int, height: int) -> List[List[float]]:
    """Centered box with a standard plate aspect ratio, offered as a starting point for manual adjustment."""
    box_w = width * 0.4
    box_h = min(box_w / PLATE_STANDARD_RATIO, height * 0.4)
    box_w = box_h * PLATE_STANDARD_RATIO
    x0, y0 = (width - box_w) / 2, (height - box_h) / 2
    return [[x0, y0], [x0 + box_w, y0], [x0 + box_w, y0 + box_h], [x0, y0 + box_h]]


def read_corners(image: np.ndarray, corners: List[List[float]], use_sr: bool = False):
    """Deskew the plate at these corners, remove residual tilt, prepare it for OCR and read it.

    With use_sr the plate region is first enlarged 4x by super-resolution (at native resolution, before
    the perspective warp). Returns (deskewed, enhanced, ocr_result), or None when it cannot be done.
    """
    source, points = image, np.asarray(PerspectiveDeskewer.arrange_corners(corners), np.float32)
    if use_sr:
        side = float(np.linalg.norm(points[1] - points[0]))
        low = np.maximum(points.min(0) - SR_REGION_MARGIN * side, 0).astype(int)
        high = np.minimum(points.max(0) + SR_REGION_MARGIN * side, [image.shape[1], image.shape[0]]).astype(int)
        crop = image[low[1]:high[1], low[0]:high[0]]
        if crop.size == 0 or not super_resolver.can_upscale(crop):
            return None
        source, points = super_resolver.upscale(crop), (points - low) * SR_SCALE

    deskewed, ok = PerspectiveDeskewer.deskew_plate(source, points.tolist())
    if not ok:
        return None
    deskewed = PerspectiveDeskewer.straighten_text(deskewed)
    enhanced = ImagePreprocessor.enhance_for_ocr(deskewed)
    return deskewed, enhanced, ocr_engine.recognize_plate(enhanced)


def snap_corners(image: np.ndarray, corners: List[List[float]]) -> List[List[float]]:
    """Move rough corners onto the plate's real edges (unchanged if no trustworthy fit is found)."""
    ordered = PerspectiveDeskewer.arrange_corners(corners)
    refined, _ = refine_quad(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), ordered)
    return refined


def run_recognition(image: np.ndarray, corners: List[List[float]], start_time: float) -> dict:
    """Deskew the plate at the given corners, enhance it and run OCR."""
    visualization = image.copy()
    cv2.polylines(visualization, [np.array(corners, dtype=np.int32)], True, (0, 255, 0), 2)

    reading = read_corners(image, corners)
    if reading is None:
        return {
            "status": "deskew_failed",
            "plate_found": True,
            "corners": corners,
            "message": "Could not straighten the plate from these corners. Adjust the corners and try again.",
            "processing_time_ms": int((time.time() - start_time) * 1000),
        }
    deskewed_plate, enhanced_plate, ocr_result = reading

    if not ocr_result["is_valid"]:
        # The start corner may be wrong (plate came out sideways or upside down): try the other three.
        # A different orientation is only adopted when it reads as a real Thai plate with good confidence;
        # otherwise (e.g. a foreign plate the Thai reader cannot parse) the original orientation is kept.
        ordered = PerspectiveDeskewer.arrange_corners(corners)
        best = None
        for shift in (1, 2, 3):
            rolled = np.roll(ordered, -shift, axis=0).tolist()
            candidate = read_corners(image, rolled)
            if candidate is None:
                continue
            result = candidate[2]
            if (result["is_valid"] and result["confidence"] >= max(MIN_ROTATED_CONFIDENCE, ocr_result["confidence"] + 0.15)
                    and (best is None or ocr_engine.quality(result) > ocr_engine.quality(best[3]))):
                best = (rolled, candidate[0], candidate[1], result)
        if best is not None:
            corners, deskewed_plate, enhanced_plate, ocr_result = best

    # Small blurry plates: try again on a super-resolved copy and keep it only if it reads clearly better.
    super_resolution_used = False
    confident = (ocr_result["is_valid"] and ocr_result["confidence"] >= SR_SKIP_CONFIDENCE and ocr_result["province"])
    if not confident and not ocr_result.get("error") and super_resolver.available:
        enlarged = read_corners(image, corners, use_sr=True)
        if enlarged is not None and ocr_engine.quality(enlarged[2]) > ocr_engine.quality(ocr_result) + SR_MIN_GAIN:
            deskewed_plate, enhanced_plate, ocr_result = enlarged
            super_resolution_used = True

    if ocr_result.get("error"):
        return {
            "status": "ocr_failed",
            "plate_found": True,
            "corners": corners,
            "message": f"OCR failed: {ocr_result['error']}",
            "processing_time_ms": int((time.time() - start_time) * 1000),
        }

    return {
        "status": "success",
        "processing_time_ms": int((time.time() - start_time) * 1000),
        "plate_found": True,
        "corners": corners,
        "super_resolution_used": super_resolution_used,
        "deskewed_plate_base64": image_to_base64(deskewed_plate),
        "enhanced_plate_base64": image_to_base64(enhanced_plate),
        "visualization_base64": image_to_base64(visualization),
        "ocr_result": {
            "text": ocr_result["text"],
            "province": ocr_result["province"],
            "confidence": ocr_result["confidence"],
            "is_valid": ocr_result["is_valid"],
        },
    }


@router.post("/recognize")
async def recognize_plate(file: UploadFile = File(...)):
    """Locate the plate with SIFT, then deskew and OCR it."""
    start_time = time.time()

    try:
        image = decode_image(await file.read())
        if image is None:
            return JSONResponse({"status": "error", "message": "Invalid image file"}, status_code=400)

        height, width = image.shape[:2]

        if locator is None:
            return JSONResponse({
                "status": "no_plate_detected",
                "plate_found": False,
                "message": locator_error,
                "suggested_corners": default_corners(width, height),
                "processing_time_ms": int((time.time() - start_time) * 1000),
            })

        corners, info = locator.locate(image)
        if corners is None:
            return JSONResponse({
                "status": "no_plate_detected",
                "plate_found": False,
                "message": info.get("error", "No plate found"),
                "suggested_corners": default_corners(width, height),
                "processing_time_ms": int((time.time() - start_time) * 1000),
            })

        return JSONResponse(run_recognition(image, snap_corners(image, corners), start_time))

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.post("/interactive-deskew")
async def interactive_deskew(request: DeskewRequest):
    """Re-run recognition with corner points adjusted by the user."""
    start_time = time.time()
    try:
        encoded = request.image_base64.split(",", 1)[-1]
        image = decode_image(base64.b64decode(encoded))
        if image is None:
            return JSONResponse({"status": "error", "message": "Invalid image data"}, status_code=400)

        if len(request.corners) != 4:
            return JSONResponse({"status": "error", "message": "Exactly 4 corners are required"}, status_code=400)

        corners = snap_corners(image, request.corners) if request.snap else request.corners
        return JSONResponse(run_recognition(image, corners, start_time))

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.post("/recognize-box")
async def recognize_box(request: BoxRequest):
    """Find the plate inside a rough rectangle drawn by the user, then deskew and OCR it."""
    start_time = time.time()
    try:
        encoded = request.image_base64.split(",", 1)[-1]
        image = decode_image(base64.b64decode(encoded))
        if image is None:
            return JSONResponse({"status": "error", "message": "Invalid image data"}, status_code=400)
        if len(request.box) != 4:
            return JSONResponse({"status": "error", "message": "box must be [x0, y0, x1, y1]"}, status_code=400)

        x0, y0, x1, y1 = request.box
        rectangle = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        corners = plate_from_box(image, request.box)
        if corners is None:
            return JSONResponse({
                "status": "no_plate_detected",
                "plate_found": False,
                "message": "Could not find a plate inside the box. Draw it closer around the plate, or drag the corners",
                "suggested_corners": rectangle,
                "processing_time_ms": int((time.time() - start_time) * 1000),
            })

        return JSONResponse(run_recognition(image, corners, start_time))

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "locator_loaded": locator is not None,
        "super_resolution_available": super_resolver.available,
        "locator_error": locator_error,
    })
