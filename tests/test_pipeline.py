"""
End-to-end integration tests for OCRPipeline.
"""

import unittest
from pathlib import Path
from PIL import Image, ImageDraw

from paddleocr_vl.config import PipelineConfig
from paddleocr_vl.pipeline import OCRPipeline


class TestOCRPipeline(unittest.TestCase):

    def setUp(self):
        self.output_dir = Path("./output_unit_pipeline")
        self.config = PipelineConfig(
            target_dpi=150,
            confidence_threshold=0.75,
            output_dir=self.output_dir,
            save_json=True,
            save_markdown=True,
        )
        self.pipeline = OCRPipeline(self.config)

    def test_pipeline_on_image(self):
        # Create synthetic image with readable text
        img_path = self.output_dir / "synthetic_page.png"
        img = Image.new("RGB", (600, 300), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "Hospital: Mercy Medical Center", fill="black")
        draw.text((20, 60), "Patient: John Doe DOB: 01/01/1980", fill="black")
        draw.text((20, 100), "[x] Cervical spine range of motion normal", fill="black")
        draw.text((20, 140), "MRN: M987654 Phone: (555) 987-6543", fill="black")
        img.save(img_path)

        res = self.pipeline.process(img_path)
        self.assertEqual(res.total_pages, 1)
        page = res.pages[0]
        self.assertEqual(page.fp_id, "fp_1")
        self.assertTrue(Path(page.image_path).exists())
        self.assertTrue((self.output_dir / "json" / "fp_1.json").exists())
        self.assertTrue((self.output_dir / "markdown" / "fp_1.md").exists())
        self.assertTrue((self.output_dir / "document_summary.json").exists())


if __name__ == "__main__":
    unittest.main()

