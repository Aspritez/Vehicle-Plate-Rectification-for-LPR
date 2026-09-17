import unittest

import cv2
import numpy as np

from plate_ocr import (
    combine_ocr_results,
    correct_province,
    format_registration,
    prepare_plate_regions,
)


class PlateOCRTests(unittest.TestCase):
    def test_registration_formatting(self):
        self.assertEqual(format_registration("กต 3789"), "กต 3789")
        self.assertEqual(format_registration("กต-3789"), "กต 3789")
        self.assertEqual(format_registration("AB 1234"), "AB1234")

    def test_province_dictionary_corrects_noisy_text(self):
        province, similarity, status = correct_province("เทชรบธี")
        self.assertEqual(province, "เพชรบุรี")
        self.assertGreater(similarity, 0.60)
        self.assertEqual(status, "dictionary_correction")

    def test_empty_province_requires_review(self):
        self.assertEqual(correct_province(""), ("", 0.0, "human_review"))

    def test_ocr_fragments_are_sorted_and_averaged(self):
        results = [
            ([[50, 0], [70, 0], [70, 20], [50, 20]], "3789", 0.8),
            ([[0, 0], [40, 0], [40, 20], [0, 20]], "กต", 0.6),
        ]
        text, confidence = combine_ocr_results(results)
        self.assertEqual(text, "กต 3789")
        self.assertAlmostEqual(confidence, 0.7)

    def test_preprocessing_splits_two_line_plate(self):
        plate = np.full((100, 240, 3), 235, dtype=np.uint8)
        cv2.putText(plate, "AB 1234", (15, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (10, 10, 10), 2, cv2.LINE_AA)
        enhanced, number_region, province_region = prepare_plate_regions(plate)
        self.assertEqual(enhanced.shape, (400, 960))
        self.assertEqual(number_region.shape[1], enhanced.shape[1])
        self.assertEqual(province_region.shape[1], enhanced.shape[1])
        self.assertGreater(number_region.shape[0], province_region.shape[0])


if __name__ == "__main__":
    unittest.main()
