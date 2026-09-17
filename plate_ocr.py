"""Thai license-plate OCR post-processing.

The OCR engine is injected so this module stays testable without downloading
model weights.  The Streamlit app supplies a cached EasyOCR reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
import re
from typing import Any, Protocol

import cv2
import numpy as np


THAI_PROVINCES = (
    "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร",
    "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท", "ชัยภูมิ",
    "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง", "ตราด", "ตาก", "นครนายก",
    "นครปฐม", "นครพนม", "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์",
    "นนทบุรี", "นราธิวาส", "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี",
    "ประจวบคีรีขันธ์", "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา",
    "พังงา", "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์",
    "แพร่", "พะเยา", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน",
    "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง", "ราชบุรี", "ลพบุรี",
    "ลำปาง", "ลำพูน", "เลย", "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล",
    "สมุทรปราการ", "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี",
    "สิงห์บุรี", "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์",
    "หนองคาย", "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี",
    "อุตรดิตถ์", "อุทัยธานี", "อุบลราชธานี",
)


class OCRReader(Protocol):
    def readtext(self, image: np.ndarray, **kwargs: Any) -> list[Any]: ...


@dataclass(frozen=True)
class PlateOCRResult:
    registration_raw: str
    registration: str
    registration_confidence: float
    province_raw: str
    province: str
    province_confidence: float
    province_similarity: float
    province_status: str


def prepare_plate_regions(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upscale and split a two-line Thai plate into number/province regions."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("ภาพป้ายทะเบียนว่างเปล่า")

    enlarged = cv2.resize(image_bgr, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
    enhanced = cv2.addWeighted(enhanced, 1.7, blurred, -0.7, 0)

    height = enhanced.shape[0]
    number_region = enhanced[:max(1, round(height * 0.70)), :]
    province_region = enhanced[min(height - 1, round(height * 0.52)):, :]
    return enhanced, number_region, province_region


def combine_ocr_results(results: list[Any]) -> tuple[str, float]:
    """Join OCR fragments from left to right and return their mean confidence."""
    if not results:
        return "", 0.0

    ordered = sorted(results, key=lambda result: min(point[0] for point in result[0]))
    text = " ".join(str(result[1]) for result in ordered).strip()
    confidence = float(np.mean([float(result[2]) for result in ordered]))
    return text, confidence


def format_registration(raw_text: str) -> str:
    """Normalize whitespace/punctuation and format Thai letters plus digits."""
    cleaned = re.sub(r"[^ก-๙A-Za-z0-9]", "", raw_text)
    match = re.fullmatch(r"([ก-ฮ]{1,3})([0-9๐-๙]{1,4})", cleaned)
    return f"{match.group(1)} {match.group(2)}" if match else cleaned


def correct_province(raw_text: str, cutoff: float = 0.35) -> tuple[str, float, str]:
    """Map noisy OCR text to the closest valid province, retaining uncertainty."""
    cleaned = re.sub(r"[^ก-๙]", "", raw_text)
    if not cleaned:
        return "", 0.0, "human_review"

    matches = get_close_matches(cleaned, THAI_PROVINCES, n=1, cutoff=cutoff)
    if not matches:
        return "", 0.0, "human_review"

    province = matches[0]
    similarity = SequenceMatcher(None, cleaned, province).ratio()
    status = "exact" if cleaned == province else "dictionary_correction"
    return province, similarity, status


def _read_region(reader: OCRReader, region: np.ndarray) -> list[Any]:
    return reader.readtext(
        region,
        detail=1,
        paragraph=False,
        decoder="beamsearch",
        text_threshold=0.25,
        low_text=0.15,
        link_threshold=0.25,
    )


def read_plate(image_bgr: np.ndarray, reader: OCRReader) -> tuple[PlateOCRResult, np.ndarray, np.ndarray, np.ndarray]:
    """Read registration number and province from an already rectified plate."""
    enhanced, number_region, province_region = prepare_plate_regions(image_bgr)
    registration_raw, registration_confidence = combine_ocr_results(
        _read_region(reader, number_region)
    )
    province_raw, province_confidence = combine_ocr_results(
        _read_region(reader, province_region)
    )

    province, province_similarity, province_status = correct_province(province_raw)
    if province_status == "exact" and province_confidence < 0.50:
        province_status = "low_confidence"

    result = PlateOCRResult(
        registration_raw=registration_raw,
        registration=format_registration(registration_raw),
        registration_confidence=registration_confidence,
        province_raw=province_raw,
        province=province,
        province_confidence=province_confidence,
        province_similarity=province_similarity,
        province_status=province_status,
    )
    return result, enhanced, number_region, province_region
