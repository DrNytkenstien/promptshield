from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Tuple, Union, Optional, Dict, Any

try:
    import pymupdf as fitz
    HAS_FITZ = True
except ImportError:
    try:
        import fitz
        HAS_FITZ = True
    except ImportError:
        HAS_FITZ = False

try:
    import PIL
    from PIL import Image, ImageEnhance, ExifTags
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pytesseract
    # Check for common Windows installation paths if tesseract is not on PATH
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
    ]
    if not shutil.which("tesseract"):
        for path in tesseract_paths:
            if Path(path).exists():
                pytesseract.pytesseract.tesseract_cmd = path
                break

    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False


def extract_image_metadata(img: Any) -> str:
    metadata_chunks = []
    try:
        exif_data = img._getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                metadata_chunks.append(f"EXIF {tag}: {value}")
    except Exception:
        pass

    if hasattr(img, "info") and isinstance(img.info, dict):
        for k, v in img.info.items():
            if isinstance(v, (str, bytes)):
                metadata_chunks.append(f"Metadata {k}: {v}")

    return "\n".join(metadata_chunks)


def preprocess_image_for_ocr(img: Any) -> Any:
    gray = img.convert("L")
    enhancer = ImageEnhance.Contrast(gray)
    return enhancer.enhance(2.5)


def process_image(image_input: Union[str, Path, bytes, Any], enhance_contrast: bool = True) -> Tuple[str, str]:
    if not HAS_PIL:
        raise ImportError("Pillow is required for image processing.")

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    elif hasattr(image_input, "convert"):
        img = image_input
    else:
        raise ValueError("Unsupported image input type.")

    metadata_text = extract_image_metadata(img)
    proc_img = preprocess_image_for_ocr(img) if enhance_contrast else img

    ocr_text = ""
    if HAS_TESSERACT:
        try:
            ocr_text = pytesseract.image_to_string(proc_img)
        except Exception:
            ocr_text = "[OCR Engine Error: Tesseract binary not installed on system]"

    return ocr_text.strip(), metadata_text.strip()


def process_pdf(pdf_input: Union[str, Path, bytes], min_text_len: int = 20) -> str:
    if not HAS_FITZ:
        raise ImportError("PyMuPDF (pymupdf) is required for PDF processing.")

    if isinstance(pdf_input, (str, Path)):
        doc = fitz.open(pdf_input)
    elif isinstance(pdf_input, bytes):
        doc = fitz.open(stream=pdf_input, filetype="pdf")
    else:
        raise ValueError("Unsupported PDF input type.")

    extracted_pages = []

    for page_num, page in enumerate(doc):
        page_text = page.get_text().strip()

        if len(page_text) < min_text_len and HAS_PIL:
            try:
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text, meta_text = process_image(img)
                combined = f"{page_text}\n{ocr_text}\n{meta_text}".strip()
            except Exception:
                combined = page_text

            content = combined if combined else page_text
            extracted_pages.append(f"--- Page {page_num + 1} ---\n{content}".strip())
        else:
            extracted_pages.append(f"--- Page {page_num + 1} ---\n{page_text}".strip())

    doc.close()
    return "\n\n".join(extracted_pages).strip()


def process_audio(audio_path: Union[str, Path], model_name: str = "base") -> str:
    if not HAS_WHISPER:
        raise ImportError("openai-whisper is required for audio transcription.")

    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path))
    return result.get("text", "").strip()


def ingest_media(content: Any, media_type: str = "auto") -> Tuple[str, str]:
    if isinstance(content, (str, Path)) and str(content).lower().endswith(".pdf"):
        media_type = "pdf"

    if media_type == "pdf":
        return process_pdf(content), "pdf_document"

    if media_type == "audio" or (isinstance(content, (str, Path)) and str(content).lower().endswith((".mp3", ".wav", ".m4a"))):
        return process_audio(content), "audio_transcript"

    if media_type in ["image", "auto"] and HAS_PIL:
        try:
            ocr_text, meta_text = process_image(content)
            full_text = f"{ocr_text}\n\n[Metadata Signal]:\n{meta_text}".strip()
            if full_text:
                return full_text, "image_ocr"
        except Exception:
            pass

    return str(content), "user_input"


class MediaProcessor:
    def __init__(self) -> None:
        pass

    @classmethod
    def ingest(cls, content: Any, media_type: str = "auto") -> Tuple[str, str]:
        return ingest_media(content, media_type)

    @classmethod
    def process(cls, content: Any, media_type: str = "auto") -> Tuple[str, str]:
        return ingest_media(content, media_type)

    @classmethod
    def process_image(cls, image_input: Union[str, Path, bytes, Any], enhance_contrast: bool = True) -> Tuple[str, str]:
        return process_image(image_input, enhance_contrast)

    @classmethod
    def process_pdf(cls, pdf_input: Union[str, Path, bytes], min_text_len: int = 20) -> str:
        return process_pdf(pdf_input, min_text_len)

    @classmethod
    def process_audio(cls, audio_path: Union[str, Path], model_name: str = "base") -> str:
        return process_audio(audio_path, model_name)
