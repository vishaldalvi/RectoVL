"""
Vision Routing & Quality Verification for Stage S0.
Packages low-confidence OCR spans (<0.75) alongside the 150 DPI page image
for targeted verification by Qwen3.5-27B Vision.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .layout_parser import PageOCRResult, WordToken


@dataclass
class VisionRoutingPayload:
    """Payload sent to Qwen3.5-27B Vision model for verifying low-confidence or handwritten areas."""
    fp_id: str
    page_number: int
    image_path: str  # 150 DPI normalized image path
    needs_vision_review: bool
    has_handwriting: bool
    mean_confidence: float
    low_confidence_count: int
    low_confidence_spans: List[Dict[str, Any]]
    targeted_vision_prompt: str

    def to_dict(self) -> dict:
        return {
            "fp_id": self.fp_id,
            "page_number": self.page_number,
            "image_path": self.image_path,
            "needs_vision_review": self.needs_vision_review,
            "has_handwriting": self.has_handwriting,
            "mean_confidence": round(self.mean_confidence, 4),
            "low_confidence_count": self.low_confidence_count,
            "low_confidence_spans": self.low_confidence_spans,
            "targeted_vision_prompt": self.targeted_vision_prompt,
        }


class VisionRouter:
    """Routes low-confidence document regions to Qwen3.5-27B Vision."""

    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold

    def build_routing_payload(self, page_result: PageOCRResult) -> VisionRoutingPayload:
        """
        Builds the exact targeted vision payload required by Stage S0.
        Marks words with OCR confidence below 0.75 with their coordinates and context.
        """
        low_spans = []
        for word in page_result.low_confidence_spans:
            # Find surrounding line context
            line_idx = word.line_number - 1
            line_context = ""
            if 0 <= line_idx < len(page_result.lines):
                line_context = page_result.lines[line_idx]

            low_spans.append({
                "word": word.text,
                "confidence": round(word.confidence, 4),
                "line_number": word.line_number,
                "line_context": line_context,
                "bbox_normalized": [
                    word.bbox.x_min,
                    word.bbox.y_min,
                    word.bbox.x_max,
                    word.bbox.y_max,
                ],
                "bbox_pixels": [
                    word.bbox.pixel_x_min,
                    word.bbox.pixel_y_min,
                    word.bbox.pixel_x_max,
                    word.bbox.pixel_y_max,
                ],
            })

        # Construct targeted instruction for Qwen3.5-27B Vision
        prompt_parts = [
            f"Page {page_result.page_number} ({page_result.fp_id}) contains {len(low_spans)} low-confidence OCR tokens (confidence < {self.confidence_threshold}).",
            "Do not transcribe the entire page from scratch. Focus verification strictly on the following suspect coordinates:",
        ]
        for idx, span in enumerate(low_spans[:25], start=1):
            prompt_parts.append(
                f"{idx}. Word '{span['word']}' (conf: {span['confidence']}) at pixel box {span['bbox_pixels']} on line: \"{span['line_context']}\""
            )
        if len(low_spans) > 25:
            prompt_parts.append(f"... and {len(low_spans) - 25} additional low-confidence spans.")

        prompt_parts.append(
            "Verify the exact text, clinical dosages, dates, and whether this region represents handwriting or diagram annotations."
        )

        return VisionRoutingPayload(
            fp_id=page_result.fp_id,
            page_number=page_result.page_number,
            image_path=page_result.image_path,
            needs_vision_review=page_result.needs_vision_review,
            has_handwriting=page_result.has_handwriting,
            mean_confidence=page_result.mean_confidence,
            low_confidence_count=len(low_spans),
            low_confidence_spans=low_spans,
            targeted_vision_prompt="\n".join(prompt_parts),
        )

