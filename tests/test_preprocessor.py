"""
Unit tests for DocumentPreprocessor.
"""

import unittest
from pathlib import Path
from PIL import Image

from paddleocr_vl.config import PipelineConfig
from paddleocr_vl.preprocessor import DocumentPreprocessor


class TestDocumentPreprocessor(unittest.TestCase):

    def setUp(self):
        self.config = PipelineConfig(target_dpi=150, output_dir=Path("./output_test_unit"))
        self.preprocessor = DocumentPreprocessor(self.config)

    def test_dpi_normalization(self):
        # Create image with explicit 72 DPI in info
        img = Image.new("RGB", (72, 72), color="white")
        img.info["dpi"] = (72, 72)
        normalized = self.preprocessor.normalize_dpi(img)
        # Should be scaled up to 150 DPI dimensions
        self.assertGreater(normalized.width, 100)
        self.assertGreater(normalized.height, 100)

    def test_clean_borders(self):
        # Create image with black borders
        img = Image.new("RGB", (200, 200), color="white")
        pixels = img.load()
        # Add black border on left
        for y in range(200):
            for x in range(10):
                pixels[x, y] = (0, 0, 0)

        cleaned = self.preprocessor.clean_borders(img, margin_ratio=0.06)
        cleaned_px = cleaned.load()
        # Border should be whitened
        self.assertEqual(cleaned_px[2, 50], (255, 255, 255))

    def test_process_single_page(self):
        img = Image.new("RGB", (300, 400), color="white")
        page = self.preprocessor.process_single_page(img, page_number=1, fp_id="test_p1")
        self.assertEqual(page.page_number, 1)
        self.assertEqual(page.fp_id, "test_p1")
        self.assertTrue(Path(page.image_path).exists())


if __name__ == "__main__":
    unittest.main()
