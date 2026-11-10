"""
Document and Page Preprocessor.
Handles PDF rasterization at 300 DPI (for OCR accuracy) and 150 DPI (for Qwen Vision / UI),
auto-orientation, deskewing, border cleanup, and contrast normalization.
"""

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
Image.MAX_IMAGE_PIXELS = None  # Allow processing ultra-high-resolution scans and photos without decompression bomb errors

from .config import PipelineConfig


@dataclass
class PreprocessedPage:
    """Represents a single preprocessed page ready for OCR and downstream routing."""
    page_number: int  # 1-indexed
    fp_id: str  # Unique File Page ID, e.g. "fp_1"
    image_path: Path  # Path to saved 150 DPI page image (for Qwen Vision & Reviewer UI)
    ocr_image_path: Path  # Path to 300 DPI preprocessed image (for high-accuracy OCR)
    width: int  # 150 DPI width
    height: int  # 150 DPI height
    ocr_width: int  # 300 DPI width
    ocr_height: int  # 300 DPI height
    dpi: int = 150
    ocr_dpi: int = 300
    skew_angle: float = 0.0
    rotation_applied: int = 0  # 0, 90, 180, 270
    metadata: dict = field(default_factory=dict)


class DocumentPreprocessor:
    """Preprocesses PDF or Image documents for PaddleOCR."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.output_dir = Path(self.config.output_dir)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def is_pdf(self, file_path: Union[str, Path]) -> bool:
        """Check if file is a PDF either by extension or magic header."""
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            return True
        try:
            with open(path, "rb") as f:
                header = f.read(5)
                return header.startswith(b"%PDF-")
        except Exception:
            return False

    def get_pdf_page_count(self, pdf_path: Union[str, Path]) -> int:
        """Determine total number of pages in a PDF."""
        path = Path(pdf_path)
        try:
            proc = subprocess.run(
                ["pdfinfo", str(path)],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if proc.returncode == 0:
                match = re.search(r"Pages:\s*(\d+)", proc.stdout)
                if match:
                    return int(match.group(1))
        except Exception:
            pass

        try:
            import pikepdf
            with pikepdf.open(path) as doc:
                return len(doc.pages)
        except Exception:
            pass

        try:
            with open(path, "rb") as f:
                content = f.read()
                matches = re.findall(rb"/Type\s*/Page\b", content)
                if matches:
                    return len(matches)
        except Exception:
            pass

        return 1

    def extract_pdf_pages_to_images(
        self,
        pdf_path: Union[str, Path],
        start_page: int = 1,
        max_pages: Optional[int] = None,
    ) -> List[Tuple[int, Path]]:
        """
        Rasterize PDF pages to high-resolution PNGs at ocr_dpi (300 DPI).
        Returns list of (page_number, temp_image_path).
        """
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        total_pages = self.get_pdf_page_count(pdf_path)
        end_page = total_pages
        if max_pages is not None:
            end_page = min(total_pages, start_page + max_pages - 1)

        extracted_pages: List[Tuple[int, Path]] = []
        temp_dir = Path(tempfile.mkdtemp(prefix="paddleocr_vl_pdf_"))

        # Rasterize at ocr_dpi (default 300 DPI) for maximum character accuracy
        raster_dpi = getattr(self.config, "ocr_dpi", 300)

        if shutil.which("pdftoppm"):
            prefix = temp_dir / "page"
            cmd = [
                "pdftoppm",
                "-png",
                "-r", str(raster_dpi),
                "-f", str(start_page),
                "-l", str(end_page),
                str(pdf_path),
                str(prefix),
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            rendered_files = sorted(temp_dir.glob("page-*.png"))
            for idx, file in enumerate(rendered_files):
                page_num = start_page + idx
                extracted_pages.append((page_num, file))
        else:
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(
                    pdf_path,
                    dpi=raster_dpi,
                    first_page=start_page,
                    last_page=end_page,
                )
                for idx, img in enumerate(images):
                    page_num = start_page + idx
                    out_path = temp_dir / f"page-{page_num:04d}.png"
                    img.save(out_path, format="PNG")
                    extracted_pages.append((page_num, out_path))
            except Exception as e:
                raise RuntimeError(
                    f"Unable to rasterize PDF. Neither pdftoppm nor pdf2image succeeded: {e}"
                )

        return extracted_pages

    def detect_orientation(self, image_path: Path) -> int:
        """Detect rotation (0, 90, 180, 270 degrees) using Tesseract OSD."""
        if not self.config.orientation_correction:
            return 0

        if shutil.which("tesseract"):
            try:
                proc = subprocess.run(
                    ["tesseract", str(image_path), "stdout", "--psm", "0"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                match = re.search(r"Rotate:\s*(\d+)", proc.stdout)
                if match:
                    deg = int(match.group(1))
                    if deg in (90, 180, 270):
                        return deg
            except Exception:
                pass
        return 0

    def detect_skew_angle(self, image: Image.Image, search_range: float = 5.0, step: float = 0.5) -> float:
        """Detect skew angle using horizontal projection profile variance."""
        if not self.config.deskew_enabled:
            return 0.0

        try:
            thumb = image.convert("L")
            thumb.thumbnail((600, 600))
            threshold = 200
            bw = thumb.point(lambda p: 255 if p > threshold else 0)
            w, h = bw.size

            best_angle = 0.0
            max_variance = -1.0
            num_steps = int(2 * search_range / step) + 1

            for i in range(num_steps):
                angle = -search_range + i * step
                rotated = bw.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=255)
                pixels = list(rotated.getdata())
                row_counts = [0] * h
                for y in range(h):
                    row_counts[y] = pixels[y * w : (y + 1) * w].count(0)

                mean = sum(row_counts) / len(row_counts)
                variance = sum((x - mean) ** 2 for x in row_counts) / len(row_counts)

                if variance > max_variance:
                    max_variance = variance
                    best_angle = angle

            return best_angle
        except Exception:
            return 0.0

    def clean_borders(self, image: Image.Image, margin_ratio: float = 0.04) -> Image.Image:
        """Whiten scanner black borders and photocopier edge shadows."""
        if not self.config.clean_borders:
            return image

        try:
            img = image.convert("RGB")
            w, h = img.size
            margin_x = int(w * margin_ratio)
            margin_y = int(h * margin_ratio)

            gray = img.convert("L")
            pixels = gray.load()

            def is_dark_strip(x1, y1, x2, y2, threshold=60, dark_ratio=0.6):
                count = 0
                total = (x2 - x1) * (y2 - y1)
                if total <= 0:
                    return False
                for y in range(y1, y2):
                    for x in range(x1, x2):
                        if pixels[x, y] < threshold:
                            count += 1
                return (count / total) >= dark_ratio

            cleaned = img.copy()
            clean_px = cleaned.load()

            if is_dark_strip(0, 0, margin_x, h):
                for y in range(h):
                    for x in range(margin_x):
                        clean_px[x, y] = (255, 255, 255)

            if is_dark_strip(w - margin_x, 0, w, h):
                for y in range(h):
                    for x in range(w - margin_x, w):
                        clean_px[x, y] = (255, 255, 255)

            if is_dark_strip(0, 0, w, margin_y):
                for y in range(margin_y):
                    for x in range(w):
                        clean_px[x, y] = (255, 255, 255)

            if is_dark_strip(0, h - margin_y, w, h):
                for y in range(h - margin_y, h):
                    for x in range(w):
                        clean_px[x, y] = (255, 255, 255)

            return cleaned
        except Exception:
            return image

    def enhance_contrast(self, image: Image.Image) -> Image.Image:
        """Enhance readability of faint medical records, carbon copies, and fax notes."""
        if not self.config.enhance_contrast:
            return image

        try:
            img = ImageOps.autocontrast(image, cutoff=1)
            img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.15)
            return img
        except Exception:
            return image

    def normalize_dpi(self, image: Image.Image, target_dpi: int = 150) -> Image.Image:
        """Resample image to target DPI while maintaining aspect ratio."""
        info = image.info
        current_dpi = info.get("dpi", (target_dpi, target_dpi))
        if isinstance(current_dpi, (tuple, list)) and len(current_dpi) >= 2:
            xdpi, ydpi = current_dpi[0], current_dpi[1]
        else:
            xdpi, ydpi = target_dpi, target_dpi

        if abs(xdpi - target_dpi) > 10 or abs(ydpi - target_dpi) > 10:
            scale_x = target_dpi / max(xdpi, 1.0)
            scale_y = target_dpi / max(ydpi, 1.0)
            new_w = max(int(image.width * scale_x), 1)
            new_h = max(int(image.height * scale_y), 1)
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

        return image

    def process_single_page(
        self,
        image_source: Union[Image.Image, Path, str],
        page_number: int = 1,
        fp_id: Optional[str] = None,
    ) -> PreprocessedPage:
        """
        Apply complete preprocessing pipeline to a single page image.
        Outputs:
        1. 300 DPI preprocessed image (used by OCR engine for max character accuracy).
        2. 150 DPI normalized image saved to `output/images/{fp_id}.png` (used by Qwen Vision & Reviewer UI).
        """
        if fp_id is None:
            fp_id = f"{self.config.page_ids_prefix}_{page_number}"

        if isinstance(image_source, (str, Path)):
            img_path = Path(image_source)
            pil_image = Image.open(img_path).convert("RGB")
        else:
            pil_image = image_source.convert("RGB")

        # Downscale ultra-high-resolution pages (e.g. photos embedded at 1pt=1px or poster scans > 25MP)
        # Standard 300 DPI letter is 2550 x 3300 (~8.4 MP). Capping max dimension to max_image_dimension (default 3500)
        # preserves full 300 DPI OCR fidelity while preventing DecompressionBomb and memory exhaustion.
        max_dim = getattr(self.config, "max_image_dimension", 3500)
        if max(pil_image.width, pil_image.height) > max_dim:
            scale = max_dim / max(pil_image.width, pil_image.height)
            new_w = max(int(pil_image.width * scale), 1)
            new_h = max(int(pil_image.height * scale), 1)
            pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        orig_w, orig_h = pil_image.size

        # 1. Auto-Orientation (0, 90, 180, 270)
        rotation_applied = 0
        if self.config.orientation_correction:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                pil_image.save(tmp.name, format="PNG")
                rot_deg = self.detect_orientation(Path(tmp.name))
                if rot_deg > 0:
                    pil_image = pil_image.rotate(-rot_deg, expand=True)
                    rotation_applied = rot_deg
                os.unlink(tmp.name)

        # 2. Deskewing
        skew_angle = 0.0
        if self.config.deskew_enabled:
            skew_angle = self.detect_skew_angle(pil_image)
            if abs(skew_angle) >= 0.2:
                pil_image = pil_image.rotate(
                    skew_angle,
                    resample=Image.Resampling.BICUBIC,
                    expand=False,
                    fillcolor=(255, 255, 255),
                )

        # 3. Clean Scanner Borders
        pil_image = self.clean_borders(pil_image)

        # 4. Enhance Contrast
        ocr_enhanced = self.enhance_contrast(pil_image)

        ocr_w, ocr_h = ocr_enhanced.size

        # 5. Save 300 DPI preprocessed image for high-accuracy OCR engine
        temp_dir = Path(tempfile.gettempdir()) / "paddleocr_vl_ocr_pages"
        temp_dir.mkdir(parents=True, exist_ok=True)
        ocr_temp_path = temp_dir / f"{fp_id}_ocr300.png"
        ocr_enhanced.save(ocr_temp_path, format="PNG")

        # 6. Generate and save 150 DPI display/routing image (Stage S0 requirement)
        target_dpi = self.config.target_dpi  # 150
        scale = target_dpi / getattr(self.config, "ocr_dpi", 300)
        disp_w = max(int(ocr_w * scale), 1)
        disp_h = max(int(ocr_h * scale), 1)
        display_img = ocr_enhanced.resize((disp_w, disp_h), Image.Resampling.LANCZOS)

        target_path = self.images_dir / f"{fp_id}.png"
        display_img.save(target_path, format="PNG", dpi=(target_dpi, target_dpi))

        return PreprocessedPage(
            page_number=page_number,
            fp_id=fp_id,
            image_path=target_path,
            ocr_image_path=ocr_temp_path,
            width=disp_w,
            height=disp_h,
            ocr_width=ocr_w,
            ocr_height=ocr_h,
            dpi=target_dpi,
            ocr_dpi=getattr(self.config, "ocr_dpi", 300),
            skew_angle=skew_angle,
            rotation_applied=rotation_applied,
            metadata={
                "original_width": orig_w,
                "original_height": orig_h,
                "display_image_path": str(target_path),
                "ocr_image_path": str(ocr_temp_path),
            },
        )

    def process_document(
        self,
        document_path: Union[str, Path],
        start_page: Optional[int] = None,
        max_pages: Optional[int] = None,
    ) -> List[PreprocessedPage]:
        """Process any PDF or Image file, extracting and preprocessing all requested pages."""
        doc_path = Path(document_path).resolve()
        start = start_page or self.config.start_page
        limit = max_pages or self.config.max_pages

        preprocessed_pages: List[PreprocessedPage] = []

        if self.is_pdf(doc_path):
            extracted = self.extract_pdf_pages_to_images(
                doc_path, start_page=start, max_pages=limit
            )
            for page_num, img_path in extracted:
                page_fp_id = f"{self.config.page_ids_prefix}_{page_num}"
                page_obj = self.process_single_page(
                    image_source=img_path,
                    page_number=page_num,
                    fp_id=page_fp_id,
                )
                preprocessed_pages.append(page_obj)
                # Cleanup raw extracted page to prevent filling up /tmp on large binders
                try:
                    if Path(img_path).exists():
                        Path(img_path).unlink()
                except Exception:
                    pass

            if extracted:
                try:
                    extracted[0][1].parent.rmdir()
                except Exception:
                    pass
        else:
            page_fp_id = f"{self.config.page_ids_prefix}_{start}"
            page_obj = self.process_single_page(
                image_source=doc_path,
                page_number=start,
                fp_id=page_fp_id,
            )
            preprocessed_pages.append(page_obj)

        return preprocessed_pages
