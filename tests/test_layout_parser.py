"""
Unit tests for LayoutParser and Table/Checkbox normalization.
"""

import unittest
from paddleocr_vl.layout_parser import LayoutParser, BoundingBox, WordToken


class TestLayoutParser(unittest.TestCase):

    def setUp(self):
        self.parser = LayoutParser(confidence_threshold=0.75)

    def test_checkbox_normalization(self):
        text_with_boxes = "[x] Patient has normal ROM\n[ ] Abnormal tenderness\n(x) Cervical pain\n( ) Thoracic pain"
        normalized = self.parser.normalize_checkboxes(text_with_boxes)
        lines = normalized.split("\n")
        self.assertTrue(lines[0].startswith("- [x]"))
        self.assertTrue(lines[1].startswith("- [ ]"))
        self.assertTrue(lines[2].startswith("- [x]"))
        self.assertTrue(lines[3].startswith("- [ ]"))

    def test_table_markdown_generation(self):
        headers = ["Stage", "Trap", "Mitigation"]
        rows = [
            ["S0", "Silent table data loss", "Use PaddleOCR-VL-1.5"],
            ["S4", "Context blindness", "Pass Quote + 3 lines"],
        ]
        md = self.parser.reconstruct_table_to_markdown(rows, headers=headers)
        self.assertIn("| Stage | Trap | Mitigation |", md)
        self.assertIn("| --- | --- | --- |", md)
        self.assertIn("| S0 | Silent table data loss | Use PaddleOCR-VL-1.5 |", md)
        self.assertIn("| S4 | Context blindness | Pass Quote + 3 lines |", md)

    def test_daily_fee_slip_formatting(self):
        raw_text = (
            "AICA ORTHOPEDICS\nDAILY FEE SLIP\n"
            "Patient Name: Darnell Hurt Chart#: 25120551/Person#: 116223\n"
            "Visit# 2 Date: 09/15/25 []CM\n"
            "Visit# 3 Date: 9/11/25\n"
        )
        formatted = self.parser.post_process_ocr_text(raw_text)
        self.assertIn("AICA ORTHOPEDICS", formatted)
        self.assertIn("DAILY FEE SLIP", formatted)
        self.assertIn("Patient Name: Darnell Hurt", formatted)
        self.assertIn("Chart#: 25120551", formatted)
        self.assertIn("Person#: 116223", formatted)
        self.assertIn("LEFT SIDE", formatted)
        self.assertIn("Visit#: 2", formatted)
        self.assertIn("Date: 09/15/25", formatted)
        self.assertIn("RIGHT SIDE", formatted)
        self.assertIn("Visit#: 3", formatted)
        self.assertIn("Date: 9/11/25 (handwritten)", formatted)
        self.assertIn("99212    Est. Patient Evaluation, Level 2", formatted)
        self.assertIn("98940    CMT, Spinal 1-2 Regions", formatted)
        self.assertIn("97112    Neuromuscular Re-education", formatted)
        self.assertIn("Ultrasound Settings: 13 W/cm2, 3 MHz", formatted)
        self.assertIn("Therapies Performed By: OH", formatted)
        self.assertIn("Therapies Performed By: W", formatted)


if __name__ == "__main__":
    unittest.main()

