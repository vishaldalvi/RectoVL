"""
Configuration options for PaddleOCR pipeline.
Supports interchangeable models:
1. PaddleOCR-VL-1.5 (Default Stage S0 Layout-Aware 0.9B VLM)
2. PaddleOCR-VL-1.6 (Updated 0.9B VLM with PP-DocLayoutV3)
3. PP-OCRv6 (High-Speed Universal OCR with PPLCNetV4 backbone)
and vice versa.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any


SUPPORTED_MODELS = {
    "paddleocr-vl-1.5": {
        "family": "vlm",
        "description": "PaddleOCR-VL-1.5 (0.9B Vision-Language Model, Layout-Aware)",
        "hf_model": "PaddlePaddle/PaddleOCR-VL-1.5",
        "pipeline_version": "v1.5",
    },
    "paddleocr-vl-1.6": {
        "family": "vlm",
        "description": "PaddleOCR-VL-1.6 (0.9B VLM, PP-DocLayoutV3, Polygonal Localization)",
        "hf_model": "PaddlePaddle/PaddleOCR-VL-1.6",
        "pipeline_version": "v1.6",
    },
    "pp-ocrv6": {
        "family": "ppocr",
        "description": "PP-OCRv6 (Lightweight Universal OCR, PPLCNetV4, 34.5M Server/Medium)",
        "hf_model": "PaddlePaddle/PP-OCRv6",
        "ocr_version": "PP-OCRv6",
    },
    "pp-ocrv5": {
        "family": "ppocr",
        "description": "PP-OCRv5 (Legacy Universal Text OCR)",
        "hf_model": "PaddlePaddle/PP-OCRv5",
        "ocr_version": "PP-OCRv5",
    },
}


def normalize_model_name(name: str) -> str:
    """Normalize model string or alias to canonical model ID."""
    clean = name.strip().lower().replace("_", "-").replace(" ", "")
    if clean in ("1.5", "vl-1.5", "vl1.5", "paddleocr-vl-1.5", "paddleocr-vl1.5", "paddleocr1.5"):
        return "paddleocr-vl-1.5"
    if clean in ("1.6", "vl-1.6", "vl1.6", "paddleocr-vl-1.6", "paddleocr-vl1.6", "paddleocr1.6"):
        return "paddleocr-vl-1.6"
    if clean in ("v6", "ocrv6", "ppocrv6", "pp-ocrv6", "pp-ocr-v6", "pp-ocr-6"):
        return "pp-ocrv6"
    if clean in ("v5", "ocrv5", "ppocrv5", "pp-ocrv5", "pp-ocr-v5", "pp-ocr-5"):
        return "pp-ocrv5"
    if clean in SUPPORTED_MODELS:
        return clean
    return name.strip()


@dataclass
class PipelineConfig:
    """Configuration settings for PaddleOCR document processing."""

    # Model Selection: "paddleocr-vl-1.5", "paddleocr-vl-1.6", "pp-ocrv6", "pp-ocrv5"
    model_type: str = "paddleocr-vl-1.5"

    # Image extraction & normalization
    target_dpi: int = 150      # Display DPI (saved to images/{fp_id}.png for Qwen Vision & Reviewer UI)
    ocr_dpi: int = 300         # Scan & OCR DPI (300 DPI gives maximum character accuracy on scans)
    max_image_dimension: int = 3500  # Cap ultra-large images to prevent DecompressionBomb & memory blowup
    deskew_enabled: bool = True
    orientation_correction: bool = True
    enhance_contrast: bool = True
    clean_borders: bool = True
    denoise: bool = True
    enable_post_processing: bool = True  # Contextual OCR error correction for legal/medical scans

    # OCR Model configuration
    model_name_or_path: Optional[str] = None
    pipeline_version: str = "v1.5"
    ocr_version: str = "PP-OCRv6"
    backend: str = "auto"  # "auto", "paddle", "transformers", "fallback"
    device: str = "auto"  # "cuda", "cpu", "auto"
    gpu_memory_fraction: float = 0.40  # Strict memory cap for GPU isolation (Stage S0)

    # Downstream Routing & Verification (Stage S0)
    confidence_threshold: float = 0.75  # Mark words below 0.75 for Qwen3.5-27B Vision
    handwriting_review_threshold: float = 0.20  # If >20% words are low-conf, flag must-review
    preserve_checkboxes: bool = True  # Preserve [x] vs [ ] tick-box states
    preserve_markdown_tables: bool = True  # Keep markdown table syntax (TEDS >= 0.95)

    # Output options
    save_150dpi_images: bool = True  # Required for Qwen Vision & Reviewer UI evidence
    save_json: bool = True
    save_markdown: bool = True
    output_dir: Path = Path("./output")

    # Document filtering
    start_page: int = 1
    max_pages: Optional[int] = None
    page_ids_prefix: str = "fp"  # e.g., fp_1, fp_2 matching mrs_free_pointers

    def __post_init__(self):
        self.model_type = normalize_model_name(self.model_type)
        model_meta = SUPPORTED_MODELS.get(self.model_type, {})

        if self.model_name_or_path is None:
            self.model_name_or_path = model_meta.get("hf_model", f"PaddlePaddle/{self.model_type}")

        if "pipeline_version" in model_meta:
            self.pipeline_version = model_meta["pipeline_version"]
        if "ocr_version" in model_meta:
            self.ocr_version = model_meta["ocr_version"]

    def to_dict(self) -> dict:
        """Convert config to dictionary representation."""
        data = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                data[k] = str(v)
            else:
                data[k] = v
        return data
