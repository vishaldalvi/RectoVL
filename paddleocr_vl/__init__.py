"""
PaddleOCR-VL-1.5 Layout-Aware Document OCR Toolkit.
Engineered for medical record pointer automation (Stages S0 - S8).
"""

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Allow processing ultra-high-resolution scans and photos without decompression bomb errors

from .config import PipelineConfig, SUPPORTED_MODELS, normalize_model_name
from .downstream_helpers import DownstreamHelper
from .layout_parser import (
    BoundingBox,
    LayoutBlock,
    LayoutParser,
    PageOCRResult,
    WordToken,
)
from .paddleocr_vl_engine import PaddleOCRVLEngine
from .pipeline import DocumentOCRResult, OCRPipeline
from .preprocessor import DocumentPreprocessor, PreprocessedPage
from .routing import VisionRouter, VisionRoutingPayload

__all__ = [
    "PipelineConfig",
    "SUPPORTED_MODELS",
    "normalize_model_name",
    "OCRPipeline",
    "DocumentPreprocessor",
    "PreprocessedPage",
    "PaddleOCRVLEngine",
    "LayoutParser",
    "BoundingBox",
    "WordToken",
    "LayoutBlock",
    "PageOCRResult",
    "DocumentOCRResult",
    "VisionRouter",
    "VisionRoutingPayload",
    "DownstreamHelper",
]

