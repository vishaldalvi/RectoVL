"""
Unit tests for model selection: PaddleOCR-VL-1.5 vs PaddleOCR-VL-1.6 vs PP-OCRv6.
"""

import unittest
from pathlib import Path
from PIL import Image

from paddleocr_vl.config import PipelineConfig, normalize_model_name, SUPPORTED_MODELS
from paddleocr_vl.pipeline import OCRPipeline


class TestModelSelection(unittest.TestCase):

    def test_alias_normalization(self):
        # VL-1.5 aliases
        for alias in ["1.5", "vl-1.5", "vl1.5", "paddleocr-vl-1.5", "paddleocr-vl1.5"]:
            self.assertEqual(normalize_model_name(alias), "paddleocr-vl-1.5")

        # VL-1.6 aliases
        for alias in ["1.6", "vl-1.6", "vl1.6", "paddleocr-vl-1.6", "paddleocr-vl1.6"]:
            self.assertEqual(normalize_model_name(alias), "paddleocr-vl-1.6")

        # PP-OCRv6 aliases
        for alias in ["v6", "ocrv6", "ppocrv6", "pp-ocrv6", "pp-ocr-v6"]:
            self.assertEqual(normalize_model_name(alias), "pp-ocrv6")

        # PP-OCRv5 aliases
        for alias in ["v5", "ocrv5", "ppocrv5", "pp-ocrv5"]:
            self.assertEqual(normalize_model_name(alias), "pp-ocrv5")

    def test_config_auto_assignment(self):
        cfg_15 = PipelineConfig(model_type="1.5")
        self.assertEqual(cfg_15.model_type, "paddleocr-vl-1.5")
        self.assertEqual(cfg_15.pipeline_version, "v1.5")
        self.assertEqual(cfg_15.model_name_or_path, "PaddlePaddle/PaddleOCR-VL-1.5")

        cfg_16 = PipelineConfig(model_type="1.6")
        self.assertEqual(cfg_16.model_type, "paddleocr-vl-1.6")
        self.assertEqual(cfg_16.pipeline_version, "v1.6")
        self.assertEqual(cfg_16.model_name_or_path, "PaddlePaddle/PaddleOCR-VL-1.6")

        cfg_v6 = PipelineConfig(model_type="v6")
        self.assertEqual(cfg_v6.model_type, "pp-ocrv6")
        self.assertEqual(cfg_v6.ocr_version, "PP-OCRv6")
        self.assertEqual(cfg_v6.model_name_or_path, "PaddlePaddle/PP-OCRv6")

    def test_pipeline_execution_with_different_models(self):
        tmp_img = Path("./output_model_test_img.png")
        img = Image.new("RGB", (200, 100), color="white")
        img.save(tmp_img)

        for model in ["paddleocr-vl-1.5", "paddleocr-vl-1.6", "pp-ocrv6"]:
            cfg = PipelineConfig(
                model_type=model,
                output_dir=Path(f"./output_test_{model.replace('.', '_')}"),
                save_json=False,
                save_markdown=False,
            )
            pipeline = OCRPipeline(cfg)
            result = pipeline.process(tmp_img)
            self.assertEqual(result.total_pages, 1)
            self.assertEqual(result.pages[0].metadata["model_type"], model)

        if tmp_img.exists():
            tmp_img.unlink()


if __name__ == "__main__":
    unittest.main()

