# PaddleOCR-VL-1.5 Layout-Aware OCR & Preprocessing Pipeline

A high-performance, layout-aware OCR extraction pipeline implemented from scratch using **PaddleOCR-VL-1.5 (0.9B)**. Built specifically for medical records pointer automation and multi-stage downstream pipelines (Stages S0 to S8 in the MRS AI architecture).

---

## Key Features & Stage S0 Alignment

| Capability | Implementation in Pipeline | Downstream Alignment |
|---|---|---|
| **Layout-Aware Parsing** | Preserves Markdown tables (`\| col1 \| col2 \|`) with high TEDS fidelity (>= 0.9543) | Stage S0 & S3: Range of Motion (ROM), lab tables, and clinical grids |
| **Tick-Box State Tracking** | Normalizes checkboxes to `[x]` (checked) and `[ ]` (unchecked) | Physical Examination (27) and clinical symptom checklists |
| **Standardized 150 DPI Output** | Renders & normalizes pages at 150 DPI saved as `images/{fp_id}.png` | Stage S0 & S6: Sent to Qwen3.5-27B Vision & Reviewer UI evidence highlight |
| **Confidence Routing (< 0.75)** | Flags every token where OCR confidence < 0.75; generates targeted vision prompt | Stage S0: Targeted verification without transcribing entire page from scratch |
| **GPU Memory Isolation** | Configurable per-process GPU memory limits (`gpu_memory_fraction=0.40`) | Stage S0: Prevents OCR memory pressure from thrashing concurrent vLLM batches |
| **Multi-Split Boundary Signals** | Extracts top headers, bottom footers, `Page X of Y`, MRN, Phone, NPI, Fax stamps | Stages S1 & S2: Multi-Split Boundary Window & Facility Clustering |
| **Quote Context Retrieval** | `get_quote_context(fp_id, quote, window=3)` with `[READ-ONLY PAGE CONTEXT]` label | Stage S4: Prevents context-blind summarization (excludes prior history) |
| **Evidence Bounding Boxes** | `find_evidence_bbox(fp_id, quote)` returns pixel and normalized box coordinates | Stages S6 - S8: Reviewer UI evidence highlighting on the 150 DPI page image |

---

## Architecture Overview

```
                      Input PDF or Image File
                                │
                                ▼
                 ┌───────────────────────────────┐
                 │    DocumentPreprocessor       │
                 │  - 150 DPI Rasterization      │
                 │  - Auto-Orientation (0-270°)  │
                 │  - Deskewing (±5° Projection) │
                 │  - Dark Scanner Border Trim   │
                 │  - Contrast / CLAHE Normalize │
                 └──────────────┬────────────────┘
                                │
                                ▼
                 ┌───────────────────────────────┐
                 │     PaddleOCR-VL-1.5 Engine   │
                 │  - Native PaddleOCRVL (v1.5)  │
                 │  - HF Transformers / vLLM     │
                 │  - Layout Fallback Mode       │
                 │  - GPU Memory Limit (40%)     │
                 └──────────────┬────────────────┘
                                │
                                ▼
                 ┌───────────────────────────────┐
                 │     Layout & Signal Parser    │
                 │  - Markdown Tables (TEDS 0.95)│
                 │  - Checkbox Normalizer [x]/[ ]│
                 │  - Word BBoxes & Confidences  │
                 │  - Low-Conf Spans (< 0.75)    │
                 │  - Signals: MRN, NPI, Page X/Y│
                 └──────────────┬────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
 150 DPI Images           Structured JSON           Markdown Text
`images/{fp_id}.png`    `json/{fp_id}.json`     `markdown/{fp_id}.md`
```

---

## Model Selection: PaddleOCR-VL-1.5 vs PaddleOCR-VL-1.6 vs PP-OCRv6

The pipeline allows seamlessly switching between **PaddleOCR-VL-1.5**, **PaddleOCR-VL-1.6**, and **PP-OCRv6** (or vice versa) via CLI arguments or Python configuration:

| Model | Architecture | Size | Strengths & Best Use Cases | CLI Flag |
|---|---|---|---|---|
| **PaddleOCR-VL-1.5** *(Default)* | NaViT Visual Encoder + ERNIE-4.5-0.3B | 0.9B | SOTA layout-aware parsing, Markdown table preservation (TEDS >= 0.95), tickbox states `[x]` vs `[ ]` | `--model paddleocr-vl-1.5`<br>*(or `--model 1.5`)* |
| **PaddleOCR-VL-1.6** | NaViT Dynamic-Res + PP-DocLayoutV3 | 0.9B | Enhanced polygonal localization for warped/skewed scans, seal recognition, improved table parsing | `--model paddleocr-vl-1.6`<br>*(or `--model 1.6`)* |
| **PP-OCRv6** | PPLCNetV4 Backbone + SVTR | 34.5M *(Server)* / 7.7M *(Mobile)* | Ultra-fast throughput (up to 20x faster than VLMs), low GPU memory footprint, 50-language universal text OCR | `--model pp-ocrv6`<br>*(or `--model v6`)* |

---

## Installation

### 1. Minimal / Local Execution (CPU / Test Environments)
```bash
pip install -r requirements.txt
# Ensure poppler-utils and tesseract are available:
# Ubuntu/Debian: sudo apt-get install poppler-utils tesseract-ocr
```

### 2. Production GPU Execution (PaddleOCR-VL-1.5 Native)
```bash
# Install PaddlePaddle GPU matching your CUDA environment (e.g. cu126)
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -U "paddleocr[doc-parser]"
```

### 3. Alternative Hugging Face Transformers / vLLM Backend
```bash
pip install "transformers>=4.40.0" torch torchvision
```

---

## CLI Usage

Run the OCR pipeline directly from the command line:

```bash
# Process a PDF file using default PaddleOCR-VL-1.5 at 150 DPI
python run_ocr.py medical_record.pdf --output-dir ./output --dpi 150

# Switch to PaddleOCR-VL-1.6 (enhanced PP-DocLayoutV3 layout parsing)
python run_ocr.py medical_record.pdf --model paddleocr-vl-1.6 --output-dir ./output

# Switch to PP-OCRv6 (high-speed universal text OCR)
python run_ocr.py medical_record.pdf --model pp-ocrv6 --output-dir ./output

# Process a single scanned image with custom confidence threshold
python run_ocr.py clinic_note.png --model 1.6 --conf-threshold 0.75

# Process specific pages and print extracted Markdown to console
python run_ocr.py large_packet.pdf --model v6 --start-page 1 --max-pages 5 --print-markdown
```

### CLI Arguments
- `input`: Path to PDF or image file (PNG, JPG, TIFF, etc.).
- `-m`, `--model`, `--ocr-model`: OCR model to use: `'paddleocr-vl-1.5'`, `'paddleocr-vl-1.6'`, `'pp-ocrv6'` (aliases: `'1.5'`, `'1.6'`, `'v6'`). Default: `'paddleocr-vl-1.5'`.
- `-o`, `--output-dir`: Output directory (default: `./output`).
- `--dpi`: Target DPI for rendering & normalization (default: `150`).
- `--conf-threshold`: Confidence cutoff for flagging low-confidence words (default: `0.75`).
- `--backend`: Inference backend: `auto`, `paddle`, `transformers`, or `fallback` (default: `auto`).
- `--start-page`: First page to process (default: `1`).
- `--max-pages`: Maximum pages to process.
- `--gpu-memory-fraction`: Per-process GPU memory allocation cap (default: `0.40`).
- `--no-deskew`: Disable fine deskewing.
- `--no-orient`: Disable 90°/180°/270° orientation detection.
- `--no-contrast`: Disable contrast enhancement.
- `--print-markdown`: Output extracted Markdown to stdout.

---

## Using in Another Process (Python API)

You can import and integrate this pipeline directly into downstream stages (vLLM workers, Celery tasks, FastAPI microservices, or S1-S8 scripts):

```python
from paddleocr_vl import OCRPipeline, PipelineConfig

# 1. Initialize pipeline with choice of model
# Choose between: "paddleocr-vl-1.5", "paddleocr-vl-1.6", or "pp-ocrv6"
config = PipelineConfig(
    model_type="paddleocr-vl-1.6", # Or "paddleocr-vl-1.5", "pp-ocrv6"
    target_dpi=150,                # Preserves 150 DPI for Qwen Vision and Reviewer UI
    confidence_threshold=0.75,     # Marks tokens below 0.75 for targeted review
    gpu_memory_fraction=0.40,      # Strict memory cap for GPU isolation from vLLM
    output_dir="./ocr_results",
    save_json=True,
    save_markdown=True,
)
pipeline = OCRPipeline(config)

# 2. Process document (PDF or Image)
doc_result = pipeline.process("medical_packet.pdf")
print(f"Processed {doc_result.total_pages} pages.")

# 3. Access per-page structured data
for page in doc_result.pages:
    print(f"Page ID: {page.fp_id} (Page {page.page_number})")
    print(f"Mean Confidence: {page.mean_confidence:.3f}")
    print(f"Needs Vision Review: {page.needs_vision_review}")
    print(f"150 DPI Image: {page.image_path}")

    # Access Markdown with preserved tables and tickboxes
    print(page.markdown_text)

# 4. Stage S1 & S2: Access Multi-Split Boundary signals
for page in doc_result.pages:
    signals = page.signals
    print("Page X of Y:", signals.get("page_x_of_y"))
    print("MRNs found:", signals.get("mrns"))
    print("Phone numbers:", signals.get("phone_numbers"))
    print("NPI numbers:", signals.get("npi_numbers"))

# 5. Stage S4: Retrieve quote with ±3 surrounding lines for safe summarization
context = doc_result.get_quote_context(
    fp_id="fp_4",
    quote="L4-L5 disc protrusion",
    window=3
)
print(context)
# Outputs:
# [READ-ONLY PAGE CONTEXT — EXCLUDE IF PRIOR HISTORY OR RULED OUT]
#     Section: Physical Examination
#     Past Medical History: 2019 MVA resolved
# >>> Patient demonstrates L4-L5 disc protrusion with radicular symptoms
#     Lumbar flexion limited to 40 degrees
#     DTRs intact bilaterally

# 6. Stage S6-S8: Get bounding box coordinates for Reviewer UI evidence highlighting
evidence_box = doc_result.find_evidence_bbox("fp_4", "L4-L5 disc protrusion")
print(evidence_box["pixel_box"])      # [x_min, y_min, x_max, y_max] on 150 DPI image
print(evidence_box["normalized_box"]) # [0.455, 0.123, 0.484, 0.132]
```

---

## Output Structure

Running the pipeline creates the following directory structure:

```
output_dir/
├── document_summary.json       # Document-level summary and all page manifests
├── images/
│   ├── fp_1.png                # Normalized 150 DPI image for Page 1
│   ├── fp_2.png                # Normalized 150 DPI image for Page 2
│   └── ...
├── markdown/
│   ├── fp_1.md                 # Layout-aware Markdown with tables & tickboxes
│   ├── fp_2.md
│   └── ...
└── json/
    ├── fp_1.json               # Word bounding boxes, confidences, blocks, signals
    ├── fp_2.json
    └── ...
```

### JSON Page Schema Example (`json/{fp_id}.json`):
```json
{
  "fp_id": "fp_1",
  "page_number": 1,
  "image_path": "output/images/fp_1.png",
  "dimensions": { "width": 1240, "height": 1754, "dpi": 150 },
  "mean_confidence": 0.942,
  "low_confidence_ratio": 0.028,
  "needs_vision_review": false,
  "has_handwriting": false,
  "total_words": 396,
  "low_confidence_count": 11,
  "signals": {
    "page_x_of_y": { "current": 1, "total": 7 },
    "fax_timestamps": [],
    "mrns": ["A102938"],
    "phone_numbers": ["(555) 123-4567"],
    "zip_codes": ["90210"],
    "npi_numbers": ["1234567890"]
  },
  "markdown_text": "## MRS AI Pointer Automation...\n\n| Stage | Trap | ... |",
  "blocks": [...],
  "low_confidence_spans": [
    {
      "text": "b1ur",
      "bbox": {
        "normalized": [0.12, 0.34, 0.18, 0.36],
        "pixels": [148, 596, 223, 631]
      },
      "confidence": 0.62,
      "line_number": 14,
      "is_low_confidence": true
    }
  ]
}
```

---

## Running Test Suite

```bash
python3 -m unittest discover -s tests
```
All unit tests and user PDF end-to-end integration tests execute within seconds.

