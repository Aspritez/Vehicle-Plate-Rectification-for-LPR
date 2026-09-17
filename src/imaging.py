"""Image decoding, display scaling, and OCR preparation. All arrays are RGB."""

from io import BytesIO
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 24_000_000


def decode_image(data: bytes) -> np.ndarray:
    if not data or len(data) > MAX_BYTES:
        raise ValueError("ไฟล์ต้องมีข้อมูลและมีขนาดไม่เกิน 10 MB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as im:
                if im.format not in ("JPEG", "PNG"):
                    raise ValueError("รองรับเฉพาะภาพ JPG และ PNG")
                if im.width * im.height > MAX_PIXELS:
                    raise ValueError("ภาพมีขนาดเกิน 24 ล้านพิกเซล กรุณาย่อภาพก่อน")
                if min(im.size) < 16:
                    raise ValueError("ภาพต้องมีขนาดอย่างน้อย 16 × 16 พิกเซล")
                im = ImageOps.exif_transpose(im)
                if "A" in im.getbands() or "transparency" in im.info:
                    rgba = im.convert("RGBA")
                    bg = Image.new("RGBA", rgba.size, "white")
                    im = Image.alpha_composite(bg, rgba)
                return np.asarray(im.convert("RGB")).copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombWarning,
            Image.DecompressionBombError) as exc:
        raise ValueError("อ่านภาพไม่ได้: ไฟล์เสียหรือขนาดภาพไม่เหมาะสม") from exc


def resized(image: np.ndarray, max_side: int) -> tuple[np.ndarray, np.ndarray]:
    """Return image and exact original-to-resized homogeneous coordinates."""
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    output = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA) if scale < 1 else image
    return output, np.diag([nw / w, nh / h, 1.0])


def png_bytes(image: np.ndarray) -> bytes:
    out = BytesIO()
    Image.fromarray(image).save(out, format="PNG")
    return out.getvalue()


def prepare_ocr(image: np.ndarray, mode: str, contrast: bool, denoise: bool) -> np.ndarray:
    output = image.copy()
    if denoise:
        output = cv2.bilateralFilter(output, 5, 30, 30)
    if contrast:
        lab = cv2.cvtColor(output, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        output = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if mode == "สี":
        return output
    gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
    if mode == "ขาวดำ (Adaptive threshold)":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 9)
    return gray
