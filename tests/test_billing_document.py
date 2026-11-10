"""
Test end-to-end OCR and layout extraction on the scanned billing document:
Billing-Concord Radiology_J. Williams.pdf
"""

import unittest
from pathlib import Path

from paddleocr_vl.config import PipelineConfig
from paddleocr_vl.downstream_helpers import DownstreamHelper
from paddleocr_vl.pipeline import OCRPipeline

BILLING_PDF_PATH = "/home/neural/vishal/DL/test_data/docs/Billing-Concord Radiology_J. Williams.pdf"


class TestBillingDocumentIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pdf_file = Path(BILLING_PDF_PATH)
        cls.output_dir = Path("./output_test_billing")
        cls.config = PipelineConfig(
            target_dpi=150,
            confidence_threshold=0.75,
            output_dir=cls.output_dir,
            save_json=True,
            save_markdown=True,
        )
        cls.pipeline = OCRPipeline(cls.config)
        cls.result = cls.pipeline.process(cls.pdf_file)

    def test_total_pages(self):
        self.assertEqual(self.result.total_pages, 3)

    def test_page_1_affidavit(self):
        p1 = self.result.get_by_fp_id("fp_1")
        self.assertIsNotNone(p1)
        self.assertIn("AFFIDAVIT OF BILLING RECORDS CUSTODIAN", p1.markdown_text)
        self.assertIn("STATE OF CALIFORNIA", p1.markdown_text)
        self.assertIn("COUNTY OF RIVERSIDE", p1.markdown_text)
        self.assertIn("I, WILFREDO TRINIDAD", p1.markdown_text)
        self.assertIn("CONCORD RADIOLOGY", p1.markdown_text)
        self.assertIn("JARMELIA WILLIAMS", p1.markdown_text)
        self.assertIn("Signature of Records Custodian: Signed", p1.markdown_text)
        self.assertIn("Notary Public: See attached (handwritten)", p1.markdown_text)
        self.assertIn("My Commission Expires: Blank", p1.markdown_text)

    def test_page_2_california_jurat(self):
        p2 = self.result.get_by_fp_id("fp_2")
        self.assertIsNotNone(p2)
        self.assertIn("CALIFORNIA JURAT", p2.markdown_text)
        self.assertIn("GOVERNMENT CODE § 8202", p2.markdown_text)
        self.assertIn("State of California", p2.markdown_text)
        self.assertIn("County of Riverside", p2.markdown_text)
        self.assertIn("Date: 3", p2.markdown_text)
        self.assertIn("Month: Sept", p2.markdown_text)
        self.assertIn("Year: 20 16", p2.markdown_text)
        self.assertIn("Wilfredo Trinidad", p2.markdown_text)
        self.assertIn("Rosanne Hernandez", p2.markdown_text)
        self.assertIn("Orange County", p2.markdown_text)
        self.assertIn("2512809", p2.markdown_text)

    def test_page_3_concord_radiology_billing(self):
        p3 = self.result.get_by_fp_id("fp_3")
        self.assertIsNotNone(p3)
        self.assertIn("CONCORD RADIOLOGY", p3.markdown_text)
        self.assertIn("PO BOX 4897 DEPT 313", p3.markdown_text)
        self.assertIn("JARMELIA WILLIAMS", p3.markdown_text)
        self.assertIn("Responsible Party", p3.markdown_text)
        self.assertIn("2969607-CMGR", p3.markdown_text)
        self.assertIn("09-02-2026", p3.markdown_text)
        self.assertIn("1227-Stevens MD, Jeffrey", p3.markdown_text)
        self.assertIn("72125 - CT NECK SPINE WO DYE", p3.markdown_text)
        self.assertIn("187.00", p3.markdown_text)
        self.assertIn("146.81", p3.markdown_text)
        self.assertIn("149.00", p3.markdown_text)
        self.assertIn("Account Balance: $0.00", p3.markdown_text)
        self.assertIn("(866) 202-9467", p3.markdown_text)

    def test_page_3_signals(self):
        p3 = self.result.get_by_fp_id("fp_3")
        signals = DownstreamHelper.extract_page_signals(p3)
        self.assertIn("(866) 202-9467", signals["phone_numbers"])
        self.assertIn("2969607-CMGR", signals["mrns"])


if __name__ == "__main__":
    unittest.main()

