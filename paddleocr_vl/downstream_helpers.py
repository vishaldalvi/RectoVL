"""
Downstream Integration Helpers for MRS AI Stages S1 - S8.
Provides:
1. Header/Footer & Signal Extractor (Page X of Y, Fax stamps, MRN, NPI, Phone, ZIP) for Stages S1 & S2.
2. Context Window Retriever (Quote ± 3 surrounding lines) for Stage S4.
3. Evidence Bounding Box Finder for Stage S6 - S8 Reviewer UI.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from .layout_parser import PageOCRResult


class DownstreamHelper:
    """Helper methods connecting Stage S0 OCR outputs to downstream MRS pipeline stages."""

    # Regex patterns for clinical and administrative signals (Stages S1 & S2)
    PAGE_X_OF_Y_PATTERN = re.compile(r"\bpage\s*(\d+)\s*(?:of|/)\s*(\d+)\b", re.IGNORECASE)
    FAX_TIMESTAMP_PATTERN = re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b)(?:\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)?",
        re.IGNORECASE,
    )
    MRN_PATTERN = re.compile(
        r"\b(?:MRN|MR#|Medical Record #|Account #|Account Number|Account)[:\s\|]*([A-Z0-9-]+)\b",
        re.IGNORECASE,
    )
    PHONE_PATTERN = re.compile(
        r"(?:(?:\+?1[-.\s]?)?(?:\(\s*\d{3}\s*\)|\b\d{3}\b)[-.\s]?\d{3}[-.\s]?\d{4}\b)"
    )
    ZIP_PATTERN = re.compile(r"\b\d{5}(?:-\d{4})?\b")
    NPI_PATTERN = re.compile(r"\b(?<!\d)\d{10}(?!\d)\b")  # 10-digit National Provider Identifier
    DATE_PATTERN = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b")

    @classmethod
    def extract_page_signals(cls, page_result: PageOCRResult, header_lines: int = 6, footer_lines: int = 6) -> Dict[str, Any]:
        """
        Extract boundary and routing signals for Multi-Split Boundary Window (Stage S1 & S2).
        Returns top headers, bottom footers, and detected regex signals.
        """
        lines = page_result.lines
        total_lines = len(lines)

        top_headers = lines[: min(header_lines, total_lines)]
        bottom_footers = lines[max(0, total_lines - footer_lines) :] if total_lines > header_lines else []

        full_text = "\n".join(lines)

        # Match Page X of Y
        page_x_of_y = None
        pxy_match = cls.PAGE_X_OF_Y_PATTERN.search(full_text)
        if pxy_match:
            page_x_of_y = {"current": int(pxy_match.group(1)), "total": int(pxy_match.group(2))}

        # Match Fax Timestamps
        fax_stamps = cls.FAX_TIMESTAMP_PATTERN.findall("\n".join(top_headers + bottom_footers))

        # Match MRNs
        mrns = cls.MRN_PATTERN.findall(full_text)

        # Match Phones
        phones = cls.PHONE_PATTERN.findall(full_text)

        # Match ZIP codes
        zips = cls.ZIP_PATTERN.findall(full_text)

        # Match NPI numbers (10 digits)
        npis = cls.NPI_PATTERN.findall(full_text)

        # Dates found
        dates = cls.DATE_PATTERN.findall(full_text)

        signals = {
            "fp_id": page_result.fp_id,
            "page_number": page_result.page_number,
            "top_headers": top_headers,
            "bottom_footers": bottom_footers,
            "page_x_of_y": page_x_of_y,
            "fax_timestamps": list(set(fax_stamps))[:5],
            "mrns": list(set(mrns))[:5],
            "phone_numbers": list(set(phones))[:5],
            "zip_codes": list(set(zips))[:5],
            "npi_numbers": list(set(npis))[:5],
            "all_dates_found": list(set(dates))[:10],
        }

        page_result.signals = signals
        return signals

    @classmethod
    def get_quote_with_surrounding_context(
        cls,
        page_result: PageOCRResult,
        quote: str,
        window: int = 3,
    ) -> str:
        """
        Retrieves verbatim quote + ±3 surrounding lines for Stage S4 Summarization.
        Label: [READ-ONLY PAGE CONTEXT — EXCLUDE IF PRIOR HISTORY OR RULED OUT]
        """
        clean_quote = quote.strip().lower()
        lines = page_result.lines
        match_idx = -1

        # Search for exact or substring match in lines
        for idx, line in enumerate(lines):
            if clean_quote in line.lower():
                match_idx = idx
                break

        # Fallback to word-level overlap
        if match_idx == -1:
            quote_words = set(clean_quote.split())
            best_overlap = 0
            for idx, line in enumerate(lines):
                line_words = set(line.lower().split())
                overlap = len(quote_words & line_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    match_idx = idx

        if match_idx == -1:
            return f"[READ-ONLY PAGE CONTEXT — EXCLUDE IF PRIOR HISTORY OR RULED OUT]\n{quote}"

        start_idx = max(0, match_idx - window)
        end_idx = min(len(lines), match_idx + window + 1)

        context_lines = []
        context_lines.append("[READ-ONLY PAGE CONTEXT — EXCLUDE IF PRIOR HISTORY OR RULED OUT]")
        for i in range(start_idx, end_idx):
            prefix = ">>> " if i == match_idx else "    "
            context_lines.append(f"{prefix}{lines[i]}")

        return "\n".join(context_lines)

    @classmethod
    def find_evidence_bbox(
        cls,
        page_result: PageOCRResult,
        quote: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Finds pixel coordinates and normalized bbox for drawing the highlight box
        on the 150 DPI page image in Stage S6-S8 Reviewer UI.
        """
        quote_tokens = [q.lower().strip(",.()[]") for q in quote.split() if q.strip(",.()[]")]
        if not quote_tokens or not page_result.words:
            return None

        # Slide over word tokens to find matching contiguous sequence
        matched_words = []
        for i in range(len(page_result.words) - len(quote_tokens) + 1):
            window_words = page_result.words[i : i + len(quote_tokens)]
            window_texts = [w.text.lower().strip(",.()[]") for w in window_words]
            if window_texts == quote_tokens:
                matched_words = window_words
                break

        # Fallback to single word / best overlap
        if not matched_words:
            for w in page_result.words:
                clean_text = w.text.lower().strip(",.()[]")
                if clean_text in quote_tokens:
                    matched_words.append(w)

        if not matched_words:
            return None

        min_x = min(w.bbox.pixel_x_min for w in matched_words)
        min_y = min(w.bbox.pixel_y_min for w in matched_words)
        max_x = max(w.bbox.pixel_x_max for w in matched_words)
        max_y = max(w.bbox.pixel_y_max for w in matched_words)

        pw = max(page_result.width, 1)
        ph = max(page_result.height, 1)

        return {
            "fp_id": page_result.fp_id,
            "page_number": page_result.page_number,
            "image_path": page_result.image_path,
            "quote": quote,
            "pixel_box": [min_x, min_y, max_x, max_y],
            "normalized_box": [
                round(min_x / pw, 5),
                round(min_y / ph, 5),
                round(max_x / pw, 5),
                round(max_y / ph, 5),
            ],
        }
