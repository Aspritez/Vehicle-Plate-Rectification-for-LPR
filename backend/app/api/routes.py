import base64
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..core.config import MAX_IMAGE_SIZE_MB
from ..pipeline import PlateReader

router = APIRouter(prefix="/api/v1", tags=["recognition"])

# Models are loaded once. PlateReader processes one request at a time (see pipeline.py). The endpoints are
# plain `def`, so waiting requests do not block the event loop (health checks and the page stay responsive).
reader = PlateReader()

MAX_UPLOAD_BYTES = int(MAX_IMAGE_SIZE_MB * 1024 * 1024)
MAX_PIXELS = 40_000_000       # refuse huge images (memory / time)


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


def load_image(data: bytes):
    """Decode an upload. Returns (image, None, None) or (None, message, http_status)."""
    if len(data) > MAX_UPLOAD_BYTES:
        return None, f"Image is larger than {MAX_IMAGE_SIZE_MB} MB", 413
    image = decode_image(data)
    if image is None:
        return None, "Invalid image file", 400
    if image.shape[0] * image.shape[1] > MAX_PIXELS:
        return None, f"Image has more than {MAX_PIXELS // 1_000_000} megapixels", 413
    return image, None, None


def load_base64_image(data_url: str):
    """Same as load_image for a base64 data URL (size is checked before decoding)."""
    encoded = data_url.split(",", 1)[-1]
    if len(encoded) > MAX_UPLOAD_BYTES * 4 // 3 + 16:
        return None, f"Image is larger than {MAX_IMAGE_SIZE_MB} MB", 413
    try:
        return load_image(base64.b64decode(encoded))
    except Exception:
        return None, "Invalid image data", 400


def to_response(result: dict) -> dict:
    """Turn the pipeline result (numpy images) into the JSON the web page expects (base64 images)."""
    images = result.pop("images", None)
    if images:
        result["deskewed_plate_base64"] = image_to_base64(images["deskewed"])
        result["enhanced_plate_base64"] = image_to_base64(images["enhanced"])
        result["visualization_base64"] = image_to_base64(images["visualization"])
    return result


def error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"status": "error", "message": message}, status_code=status)


@router.post("/recognize")
def recognize_plate(file: UploadFile = File(...)):
    """Locate the plate with SIFT, then deskew and OCR it."""
    try:
        image, message, status = load_image(file.file.read(MAX_UPLOAD_BYTES + 1))
        if message:
            return error(message, status)
        return JSONResponse(to_response(reader.recognize(image)))
    except Exception as e:
        return error(str(e), 500)


@router.post("/interactive-deskew")
def interactive_deskew(request: DeskewRequest):
    """Re-run recognition with corner points adjusted by the user."""
    try:
        image, message, status = load_base64_image(request.image_base64)
        if message:
            return error(message, status)
        if len(request.corners) != 4:
            return error("Exactly 4 corners are required", 400)
        return JSONResponse(to_response(reader.recognize_corners(image, request.corners, request.snap)))
    except Exception as e:
        return error(str(e), 500)


@router.post("/recognize-box")
def recognize_box(request: BoxRequest):
    """Find the plate inside a rough rectangle drawn by the user, then deskew and OCR it."""
    try:
        image, message, status = load_base64_image(request.image_base64)
        if message:
            return error(message, status)
        if len(request.box) != 4:
            return error("box must be [x0, y0, x1, y1]", 400)
        return JSONResponse(to_response(reader.recognize_box(image, request.box)))
    except Exception as e:
        return error(str(e), 500)


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "locator_loaded": reader.locator is not None,
        "super_resolution_available": reader.super_resolver.available,
        "locator_error": reader.locator_error,
    })
