"""
Unit tests for DownstreamHelper (Stage S1 - S8).
"""

import unittest
from paddleocr_vl.downstream_helpers import DownstreamHelper
from paddleocr_vl.layout_parser import PageOCRResult, WordToken, BoundingBox


class TestDownstreamHelper(unittest.TestCase):

    def setUp(self):
        # Create a mock PageOCRResult
        words = [
            WordToken("Patient", BoundingBox.from_pixels(10, 10, 50, 30, 500, 500), 0.95, 1, 1),
            WordToken("denies", BoundingBox.from_pixels(60, 10, 100, 30, 500, 500), 0.95, 1, 1),
            WordToken("neck", BoundingBox.from_pixels(110, 10, 150, 30, 500, 500), 0.95, 1, 1),
            WordToken("pain", BoundingBox.from_pixels(160, 10, 200, 30, 500, 500), 0.95, 1, 1),
        ]
        lines = [
            "Facility: Mercy General Hospital Phone: (555) 123-4567 ZIP: 90210",
            "Page 1 of 5 MRN: A1234567 NPI: 1234567890",
            "Line before quote",
            "Patient denies neck pain or cervical tenderness",
            "Line after quote",
            "Discharge instructions and follow-up on 10/15/2025",
        ]
        self.page = PageOCRResult(
            fp_id="fp_1",
            page_number=1,
            image_path="output/images/fp_1.png",
            width=500,
            height=500,
            dpi=150,
            markdown_text="\n".join(lines),
            raw_text="\n".join(lines),
            blocks=[],
            words=words,
            lines=lines,
            low_confidence_spans=[],
            mean_confidence=0.95,
            low_confidence_ratio=0.0,
            needs_vision_review=False,
            has_handwriting=False,
        )

    def test_signal_extraction(self):
        signals = DownstreamHelper.extract_page_signals(self.page)
        self.assertEqual(signals["page_x_of_y"], {"current": 1, "total": 5})
        self.assertIn("A1234567", signals["mrns"])
        self.assertIn("1234567890", signals["npi_numbers"])
        self.assertIn("(555) 123-4567", signals["phone_numbers"])
        self.assertIn("90210", signals["zip_codes"])

    def test_quote_surrounding_context(self):
        ctx = DownstreamHelper.get_quote_with_surrounding_context(
            self.page, "neck pain", window=1
        )
        self.assertIn("[READ-ONLY PAGE CONTEXT — EXCLUDE IF PRIOR HISTORY OR RULED OUT]", ctx)
        self.assertIn("Line before quote", ctx)
        self.assertIn(">>> Patient denies neck pain or cervical tenderness", ctx)
        self.assertIn("Line after quote", ctx)

    def test_evidence_bbox_finder(self):
        bbox = DownstreamHelper.find_evidence_bbox(self.page, "neck pain")
        self.assertIsNotNone(bbox)
        self.assertEqual(bbox["fp_id"], "fp_1")
        self.assertEqual(bbox["pixel_box"], [110, 10, 200, 30])


if __name__ == "__main__":
    unittest.main()

