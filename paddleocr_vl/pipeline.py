"""
Master PaddleOCR-VL-1.5 OCR Pipeline.
Coordinates end-to-end processing:
Input PDF / Image -> 150 DPI Preprocessing -> Layout-Aware OCR ->
Confidence Filtering (<0.75) -> Downstream Signal & Context Extraction ->
Structured JSON / Markdown / 150 DPI Page Exports.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from .config import PipelineConfig
from .downstream_helpers import DownstreamHelper
from .layout_parser import PageOCRResult
from .paddleocr_vl_engine import PaddleOCRVLEngine
from .preprocessor import DocumentPreprocessor, PreprocessedPage
from .routing import VisionRouter, VisionRoutingPayload

logger = logging.getLogger("PaddleOCR-VL-Pipeline")


@dataclass
class DocumentOCRResult:
    """Complete document-level extraction result across all pages."""
    file_path: str
    total_pages: int
    pages: List[PageOCRResult]
    vision_payloads: List[VisionRoutingPayload]
    config: Dict[str, Any]
    output_directory: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    elapsed_seconds: float = 0.0

    def get_page(self, page_number: int) -> Optional[PageOCRResult]:
        """Lookup page by 1-indexed page number."""
        for p in self.pages:
            if p.page_number == page_number:
                return p
        return None

    def get_by_fp_id(self, fp_id: str) -> Optional[PageOCRResult]:
        """Lookup page by File Page ID (e.g., 'fp_1')."""
        for p in self.pages:
            if p.fp_id == fp_id:
                return p
        return None

    def get_quote_context(self, fp_id: str, quote: str, window: int = 3) -> str:
        """Stage S4 context retrieval for a specific page."""
        page = self.get_by_fp_id(fp_id)
        if not page:
            return f"[ERROR: Page {fp_id} not found]\n{quote}"
        return DownstreamHelper.get_quote_with_surrounding_context(page, quote, window)

    def find_evidence_bbox(self, fp_id: str, quote: str) -> Optional[Dict[str, Any]]:
        """Stage S6-S8 reviewer UI evidence highlight coordinates."""
        page = self.get_by_fp_id(fp_id)
        if not page:
            return None
        return DownstreamHelper.find_evidence_bbox(page, quote)

    def to_dict(self) -> dict:
        """Convert entire document OCR result to structured dictionary."""
        sec_per_page = round(self.elapsed_seconds / max(self.total_pages, 1), 2) if self.total_pages > 0 else 0.0
        return {
            "file_path": self.file_path,
            "total_pages": self.total_pages,
            "output_directory": self.output_directory,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "seconds_per_page": sec_per_page,
            "pages": [p.to_dict() for p in self.pages],
            "vision_routing": [v.to_dict() for v in self.vision_payloads],
            "config": self.config,
        }


class OCRPipeline:
    """Master pipeline for PaddleOCR-VL-1.5 document extraction."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.images_dir = self.output_dir / "images"
        self.json_dir = self.output_dir / "json"
        self.markdown_dir = self.output_dir / "markdown"

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)

        # Initialize subcomponents
        self.preprocessor = DocumentPreprocessor(self.config)
        self.engine = PaddleOCRVLEngine(self.config)
        self.router = VisionRouter(confidence_threshold=self.config.confidence_threshold)

    def process(
        self,
        file_path: Union[str, Path],
        start_page: Optional[int] = None,
        max_pages: Optional[int] = None,
    ) -> DocumentOCRResult:
        """
        Main entry point: Process PDF or Image file through complete pipeline.
        """
        import datetime
        import time

        start_dt = datetime.datetime.now()
        start_time_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        t0 = time.time()

        input_path = Path(file_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input document does not exist: {input_path}")

        logger.info("Starting PaddleOCR processing for: %s at %s", input_path.name, start_time_str)

        # 1. Page Extraction & Preprocessing (150 DPI, auto-orientation, deskew, contrast)
        preprocessed_pages = self.preprocessor.process_document(
            input_path, start_page=start_page, max_pages=max_pages
        )

        page_results: List[PageOCRResult] = []
        vision_payloads: List[VisionRoutingPayload] = []

        total_pages_count = len(preprocessed_pages)

        # 2. OCR Execution, Layout Analysis, and Signal Extraction per page
        for idx, ppage in enumerate(preprocessed_pages, 1):
            t_page_start = time.time()
            logger.info("Running Layout-Aware OCR on %s (page %d/%d)", ppage.fp_id, idx, total_pages_count)
            page_res = self.engine.predict_page(
                image_path=ppage.image_path,
                ocr_image_path=getattr(ppage, "ocr_image_path", ppage.image_path),
                page_number=ppage.page_number,
                fp_id=ppage.fp_id,
            )

            # Extract signals for Multi-Split Boundary Window (Stages S1 & S2)
            DownstreamHelper.extract_page_signals(page_res)

            # Build Vision Routing Payload for Stage S0 (<0.75 confidence spans)
            vision_payload = self.router.build_routing_payload(page_res)

            # 3. Save exports
            if self.config.save_markdown:
                md_path = self.markdown_dir / f"{page_res.fp_id}.md"
                md_path.write_text(page_res.markdown_text, encoding="utf-8")

            if self.config.save_json:
                json_path = self.json_dir / f"{page_res.fp_id}.json"
                json_path.write_text(
                    json.dumps(page_res.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            page_results.append(page_res)
            vision_payloads.append(vision_payload)
            page_elapsed = time.time() - t_page_start
            logger.info("Page %s completed in %.2fs", ppage.fp_id, page_elapsed)

        # 4. Save document-level manifest with timing info
        end_dt = datetime.datetime.now()
        end_time_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - t0

        doc_result = DocumentOCRResult(
            file_path=str(input_path),
            total_pages=len(page_results),
            pages=page_results,
            vision_payloads=vision_payloads,
            config=self.config.to_dict(),
            output_directory=str(self.output_dir),
            start_time=start_time_str,
            end_time=end_time_str,
            elapsed_seconds=elapsed,
        )

        summary_file = self.output_dir / "document_summary.json"
        summary_file.write_text(
            json.dumps(doc_result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "Successfully processed %d pages from %s to %s (total: %.2fs, %.2fs/page). Summary saved to %s",
            len(page_results),
            start_time_str,
            end_time_str,
            elapsed,
            elapsed / max(len(page_results), 1),
            summary_file,
        )

        return doc_result

