"""
Integration test running on the actual user PDF:
MRS AI Pointer Automation: Best Approaches & Development Guide.
"""

import unittest
from pathlib import Path

from paddleocr_vl.config import PipelineConfig
from paddleocr_vl.pipeline import OCRPipeline

PDF_PATH = "/home/neural/.gemini/antigravity/brain/e05075f9-60e9-4524-97aa-f783d5e39604/.user_uploaded/media_1793937630172_8fd26b9f.pdf"


class TestUserPDFIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pdf_file = Path(PDF_PATH)
        cls.output_dir = Path("./output_test_pdf_integration")
        cls.config = PipelineConfig(
            target_dpi=150,
            confidence_threshold=0.75,
            output_dir=cls.output_dir,
            save_json=True,
            save_markdown=True,
        )
        cls.pipeline = OCRPipeline(cls.config)
        cls.result = cls.pipeline.process(cls.pdf_file)

    def test_page_count(self):
        self.assertEqual(self.result.total_pages, 7)

    def test_page_4_table_reconstruction(self):
        p4 = self.result.get_by_fp_id("fp_4")
        self.assertIsNotNone(p4)
        # Check that table markdown was produced
        self.assertIn("| Stage S4 Trap | Why It Happens | Best Approach to Fix It |", p4.markdown_text)
        self.assertIn("| 1. Context-Blind Summarization |", p4.markdown_text)
        self.assertIn("| 2. Forced Pick-List Guessing |", p4.markdown_text)

    def test_page_2_tickbox_preservation(self):
        p2 = self.result.get_by_fp_id("fp_2")
        self.assertIsNotNone(p2)
        # Check tickbox states preserved
        self.assertIn("[x]", p2.markdown_text)
        self.assertIn("[ ]", p2.markdown_text)

    def test_stage_s4_context_retrieval(self):
        ctx = self.result.get_quote_context("fp_4", "L4-L5 disc protrusion", window=2)
        self.assertIn("[READ-ONLY PAGE CONTEXT — EXCLUDE IF PRIOR HISTORY OR RULED OUT]", ctx)
        self.assertIn(">>>", ctx)

    def test_stage_s6_evidence_bbox(self):
        bbox = self.result.find_evidence_bbox("fp_4", "L4-L5 disc protrusion")
        self.assertIsNotNone(bbox)
        self.assertEqual(bbox["fp_id"], "fp_4")
        self.assertEqual(len(bbox["pixel_box"]), 4)
        self.assertEqual(len(bbox["normalized_box"]), 4)


if __name__ == "__main__":
    unittest.main()

