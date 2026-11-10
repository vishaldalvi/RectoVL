"""
Command-line interface for PaddleOCR-VL-1.5 OCR pipeline.
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from .config import PipelineConfig
from .pipeline import OCRPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="PaddleOCR-VL-1.5: Layout-Aware Document OCR with Preprocessing and Downstream Routing."
    )
    parser.add_argument("input", type=str, help="Path to input PDF or image file.")
    parser.add_argument(
        "-m", "--model", "--ocr-model",
        dest="model",
        type=str,
        default="paddleocr-vl-1.5",
        help="OCR model to use: 'paddleocr-vl-1.5', 'paddleocr-vl-1.6', 'pp-ocrv6' (aliases: '1.5', '1.6', 'v6'). Default: 'paddleocr-vl-1.5'."
    )
    parser.add_argument("-o", "--output-dir", type=str, default="./output", help="Directory for generated files.")
    parser.add_argument("--dpi", type=int, default=150, help="Target DPI for PDF rasterization and image normalization (default: 150).")
    parser.add_argument("--conf-threshold", type=float, default=0.75, help="Confidence threshold below which tokens are flagged for Qwen Vision review (default: 0.75).")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "paddle", "transformers", "fallback"], help="Inference backend to use.")
    parser.add_argument("--start-page", type=int, default=1, help="Starting page number (1-indexed, default: 1).")
    parser.add_argument("--max-pages", type=int, default=None, help="Maximum number of pages to process.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Hardware device.")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.40, help="Per-process GPU memory limit fraction (default: 0.40).")
    parser.add_argument("--no-deskew", action="store_true", help="Disable automatic deskewing.")
    parser.add_argument("--no-orient", action="store_true", help="Disable automatic rotation/orientation correction.")
    parser.add_argument("--no-contrast", action="store_true", help="Disable contrast enhancement.")
    parser.add_argument("--print-markdown", action="store_true", help="Print extracted Markdown to stdout.")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    config = PipelineConfig(
        model_type=args.model,
        target_dpi=args.dpi,
        confidence_threshold=args.conf_threshold,
        backend=args.backend,
        device=args.device,
        gpu_memory_fraction=args.gpu_memory_fraction,
        deskew_enabled=not args.no_deskew,
        orientation_correction=not args.no_orient,
        enhance_contrast=not args.no_contrast,
        output_dir=Path(args.output_dir),
        start_page=args.start_page,
        max_pages=args.max_pages,
    )

    start_dt = datetime.datetime.now()
    start_time_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()

    print("=" * 70)
    print("PaddleOCR Document Pipeline")
    print(f"Model:        {config.model_type} ({config.model_name_or_path})")
    print(f"Input:        {input_path.name}")
    print(f"Target DPI:   {config.target_dpi}")
    print(f"Conf Cutoff:  {config.confidence_threshold}")
    print(f"Output Dir:   {config.output_dir}")
    print(f"Start Time:   {start_time_str}")
    print("=" * 70)

    pipeline = OCRPipeline(config)
    result = pipeline.process(input_path, start_page=args.start_page, max_pages=args.max_pages)

    end_dt = datetime.datetime.now()
    end_time_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    elapsed = time.time() - t0
    sec_per_page = (elapsed / result.total_pages) if result.total_pages > 0 else 0.0

    print("\nProcessing complete:")
    print(f"Total Pages Processed: {result.total_pages}")
    print(f"Start Time:            {start_time_str}")
    print(f"End Time:              {end_time_str}")
    if elapsed >= 60:
        mins = int(elapsed // 60)
        secs = elapsed % 60
        print(f"Total Time Taken:      {elapsed:.2f}s ({mins}m {secs:.1f}s)")
    else:
        print(f"Total Time Taken:      {elapsed:.2f}s")
    print(f"Average Speed:         {sec_per_page:.2f}s per page")
    print("-" * 70)
    for page in result.pages:
        print(
            f"  - [{page.fp_id}] Page {page.page_number}: {len(page.words)} words, "
            f"Mean Conf: {page.mean_confidence:.2f}, "
            f"Low-Conf Spans: {len(page.low_confidence_spans)} "
            f"({'NEEDS VISION REVIEW' if page.needs_vision_review else 'OK'})"
        )

    if args.print_markdown:
        print("\n" + "=" * 70)
        print("Extracted Markdown:")
        print("=" * 70)
        for page in result.pages:
            print(f"\n--- {page.fp_id} (Page {page.page_number}) ---")
            print(page.markdown_text)


if __name__ == "__main__":
    main()

