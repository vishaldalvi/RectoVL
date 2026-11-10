#!/usr/bin/env python3
"""
Executable script for running PaddleOCR-VL-1.5 OCR pipeline on PDF or Image.
Usage:
    python run_ocr.py document.pdf --output-dir ./output --dpi 150
    python run_ocr.py scan.png --conf-threshold 0.75
"""

import sys
from paddleocr_vl.cli import main

if __name__ == "__main__":
    main()

