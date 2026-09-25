import difflib
import re
from typing import Dict, List, Tuple

import cv2
import easyocr
import numpy as np

from .thai_provinces import CONSONANTS, PROVINCES

DIGITS = "0123456789"
PREFIX_CHARS = "123456789" + CONSONANTS   # letters part: no 0, so a letter is never read as a zero
# Thai letters, vowels and tone marks: the alphabet of the province line.
THAI_CHARS = ("".join(chr(c) for c in range(0x0E01, 0x0E3B))
              + "".join(chr(c) for c in range(0x0E40, 0x0E4F)))

# Registration: optional leading digit (1-9), one or two consonants, then 1-4 digits.
PLATE_PATTERN = re.compile(rf"^([1-9]?[{CONSONANTS}]{{1,2}})([0-9]{{1,4}})$")

ROW_GAP_RATIO = 0.16        # vertical gap (fraction of image height) that separates the two plate lines
FALLBACK_SPLIT = 0.62       # top-line / province-line boundary when text detection finds nothing
MIN_PROVINCE_SCORE = 0.6    # below this similarity the province line is left blank (0.6 gave no wrong provinces on synthetic tests)
MIN_NUMBER_GAP = 0.05       # horizontal gap (fraction of width) that separates the letters from the number


def _clean(text: str) -> str:
    return re.sub(r"\s+", "", text)


class OCREngine:
    def __init__(self):
        self.reader = easyocr.Reader(['th', 'en'], gpu=False)

    def recognize_plate(self, image: np.ndarray) -> Dict:
        """Read a deskewed Thai plate: registration on the top line, province on the second.

        The two lines are located first and read separately with restricted alphabets
        (consonants + digits for the registration, Thai characters for the province), which
        removes Latin look-alikes and stops the province text from mixing into the number.
        """
        try:
            gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            top_boxes, province_boxes = self._locate_rows(gray)

            top_text, top_conf = self._read_registration(gray, top_boxes)
            province_raw, _ = self._read(gray, province_boxes, THAI_CHARS)

            plate_text, is_valid = self._format_plate(_clean(top_text))
            province, province_score = self._match_province(province_raw)

            return {
                "text": plate_text,
                "raw_text": f"{top_text} {province_raw}".strip(),
                "confidence": top_conf,
                "province": province,
                "province_score": province_score,
                "is_valid": is_valid,
            }
        except Exception as e:
            return {"text": "", "confidence": 0, "province": "", "province_score": 0.0,
                    "is_valid": False, "error": str(e)}

    @staticmethod
    def quality(result: Dict) -> float:
        """Score a result so alternative readings of the same plate can be compared."""
        return (1.0 if result.get("is_valid") else 0.0) + result.get("province_score", 0.0) \
            + 0.5 * result.get("confidence", 0.0)

    # ---------- line location ----------
    def _locate_rows(self, gray: np.ndarray) -> Tuple[List[List[int]], List[List[int]]]:
        height, width = gray.shape[:2]
        horizontal, _ = self.reader.detect(gray, min_size=10, text_threshold=0.5, low_text=0.3,
                                           link_threshold=0.4, width_ths=0.3, add_margin=0.1)
        boxes = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in horizontal[0]]

        if not boxes:
            split = int(height * FALLBACK_SPLIT)
            return [[0, width, 0, split]], [[0, width, split, height]]

        # Two lines: cut at the largest vertical gap between box centres.
        boxes.sort(key=lambda b: (b[2] + b[3]) / 2)
        centres = [(b[2] + b[3]) / 2 for b in boxes]
        gaps = [centres[i + 1] - centres[i] for i in range(len(centres) - 1)]
        if gaps and max(gaps) > ROW_GAP_RATIO * height:
            cut = int(np.argmax(gaps)) + 1
            return boxes[:cut], boxes[cut:]
        return boxes, []

    def _read_registration(self, gray: np.ndarray, boxes: List[List[int]]) -> Tuple[str, float]:
        """Read the top line. When the letters and the number are separate text blocks they are read
        with separate alphabets: letters (+ leading digit) on the left, digits only on the right."""
        boxes = sorted(boxes, key=lambda b: b[0])
        gaps = [boxes[i + 1][0] - boxes[i][1] for i in range(len(boxes) - 1)]
        if gaps and max(gaps) >= MIN_NUMBER_GAP * gray.shape[1]:
            cut = int(np.argmax(gaps)) + 1
            left, right = boxes[:cut], boxes[cut:]
        else:
            # The detector merged everything into one block: look for the blank gap in the ink.
            merged = [min(b[0] for b in boxes), max(b[1] for b in boxes),
                      min(b[2] for b in boxes), max(b[3] for b in boxes)]
            left, right = self._split_at_gap(gray, merged)

        if left and right:
            prefix, prefix_conf = self._read(gray, left, PREFIX_CHARS)
            number, number_conf = self._read(gray, right, DIGITS)
            return prefix + number, (prefix_conf + number_conf) / 2
        return self._read(gray, boxes, DIGITS + CONSONANTS)

    @staticmethod
    def _split_at_gap(gray: np.ndarray, box: List[int]) -> Tuple[List[List[int]], List[List[int]]]:
        """Split one text block into letters | number at the widest blank column run in its middle."""
        height, width = gray.shape[:2]
        x0, x1 = max(0, box[0]), min(width, box[1])
        y0, y1 = max(0, box[2]), min(height, box[3])
        if x1 - x0 < 20 or y1 - y0 < 10:
            return [], []

        ink = cv2.threshold(gray[y0:y1, x0:x1], 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        columns = np.convolve(ink.sum(axis=0) / 255.0, np.ones(5) / 5, mode="same")
        baseline = np.percentile(columns, 5)     # ink that every column has (plate edges, screws, ...)
        blank = columns <= baseline + 0.15 * (columns.max() - baseline)

        lo, hi = int(0.15 * len(blank)), int(0.85 * len(blank))
        best_start, best_len, run_start = 0, 0, None
        for i in range(lo, hi + 1):
            if i < hi and blank[i]:
                run_start = i if run_start is None else run_start
            elif run_start is not None:
                if i - run_start > best_len:
                    best_start, best_len = run_start, i - run_start
                run_start = None
        if best_len < 0.06 * (y1 - y0):      # gap must be a real character space, not noise
            return [], []

        cut_left, cut_right = x0 + best_start, x0 + best_start + best_len
        return [[x0, cut_left, y0, y1]], [[cut_right, x1, y0, y1]]

    def _read(self, gray: np.ndarray, boxes: List[List[int]], allowlist: str) -> Tuple[str, float]:
        if not boxes:
            return "", 0.0
        boxes = sorted(boxes, key=lambda b: b[0])
        results = self.reader.recognize(gray, horizontal_list=boxes, free_list=[],
                                        allowlist=allowlist, detail=1)
        if not results:
            return "", 0.0
        text = "".join(t for _, t, _ in results)
        return text, float(np.mean([c for _, _, c in results]))

    # ---------- post-processing ----------
    @staticmethod
    def _format_plate(text: str) -> Tuple[str, bool]:
        match = PLATE_PATTERN.match(text)
        if match:
            return f"{match.group(1)} {match.group(2)}", True
        return text, False

    @staticmethod
    def _match_province(text: str) -> Tuple[str, float]:
        """Snap the OCR'd province line to the closest of the 77 official names."""
        text = _clean(text)
        if not text:
            return "", 0.0
        best = max(PROVINCES, key=lambda p: difflib.SequenceMatcher(None, text, p).ratio())
        score = difflib.SequenceMatcher(None, text, best).ratio()
        return (best, score) if score >= MIN_PROVINCE_SCORE else ("", score)
