from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Tuple, Union

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
    from PIL import Image, ImageEnhance, ExifTags      # fixed: no "PIL" in this import
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pytesseract
    HAS_TESSERACT = True
    # Windows: uncomment if Tesseract isn't on PATH
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ImportError:
    HAS_TESSERACT = False

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
_AUDIO_EXT = (".mp3", ".wav", ".m4a")
_SKIP_INFO_KEYS = {"icc_profile", "exif", "dpi", "jfif", "jfif_version", "jfif_unit", "jfif_density"}


def _sniff(content: Any) -> str | None:
    """Work out what kind of media this is, including raw uploaded bytes."""
    if isinstance(content, (bytes, bytearray)):
        b = bytes(content)
        if b[:5] == b"%PDF-":
            return "pdf"
        if (b[:8] == b"\x89PNG\r\n\x1a\n" or b[:3] == b"\xff\xd8\xff"
                or b[:6] in (b"GIF87a", b"GIF89a") or (b[:4] == b"RIFF" and b[8:12] == b"WEBP")
                or b[:2] == b"BM"):
            return "image"
        return None
    if isinstance(content, (str, Path)):
        s = str(content).lower()
        if len(s) < 500 and "\n" not in s:                 # looks like a path, not pasted text
            if s.endswith(".pdf"):
                return "pdf"
            if s.endswith(_IMAGE_EXT):
                return "image"
            if s.endswith(_AUDIO_EXT):
                return "audio"
        return None
    if hasattr(content, "convert"):                        # PIL image object
        return "image"
    return None


def _to_text(value: Any) -> str:
    """Make EXIF/metadata values readable (they are often bytes, sometimes UTF-16)."""
    if isinstance(value, bytes):
        for enc in ("utf-16le", "utf-8"):
            try:
                out = value.decode(enc).replace("\x00", "").strip()
                if out and out.isprintable():
                    return out
            except UnicodeDecodeError:
                continue
        return value.decode("utf-8", errors="ignore").replace("\x00", "").strip()
    return str(value).strip()


def extract_image_metadata(img: Any) -> str:
    """Extract EXIF and embedded text fields (metadata smuggling)."""
    chunks = []
    try:
        exif = img.getexif()                                # public API (was _getexif)
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, str(tag_id))
            text = _to_text(value)
            if len(text) >= 4:
                chunks.append(f"EXIF {tag}: {text}")
        for tag_id, value in exif.get_ifd(0x8769).items():  # Exif sub-IFD holds UserComment
            tag = ExifTags.TAGS.get(tag_id, str(tag_id))
            text = _to_text(value)
            if len(text) >= 4:
                chunks.append(f"EXIF {tag}: {text}")
    except Exception:
        pass

    info = getattr(img, "info", None)
    if isinstance(info, dict):
        for k, v in info.items():
            if k in _SKIP_INFO_KEYS or not isinstance(v, (str, bytes)):
                continue                                    # skips ICC profile binary blobs
            text = _to_text(v)
            if len(text) >= 4:
                chunks.append(f"Metadata {k}: {text}")
    return "\n".join(chunks)


def preprocess_image_for_ocr(img: Any) -> Any:
    """Grayscale + contrast boost to expose faint / near-invisible text."""
    return ImageEnhance.Contrast(img.convert("L")).enhance(2.5)


def process_image(image_input: Union[str, Path, bytes, Any], enhance_contrast: bool = True) -> Tuple[str, str]:
    """Returns (ocr_text, metadata_text)."""
    if not HAS_PIL:
        raise ImportError("Pillow is required for image processing.")
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(bytes(image_input)))
    elif hasattr(image_input, "convert"):
        img = image_input
    else:
        raise ValueError("Unsupported image input type.")

    metadata_text = extract_image_metadata(img)
    proc = preprocess_image_for_ocr(img) if enhance_contrast else img

    ocr_text = ""
    if HAS_TESSERACT:
        try:
            ocr_text = pytesseract.image_to_string(proc)
        except Exception:
            ocr_text = ""
    return ocr_text.strip(), metadata_text.strip()


def process_pdf(pdf_input: Union[str, Path, bytes], min_text_len: int = 20) -> str:
    """Extract text per page; OCR pages that have no text layer."""
    if not HAS_FITZ:
        raise ImportError("PyMuPDF (pymupdf) is required for PDF processing.")
    if isinstance(pdf_input, (str, Path)):
        doc = fitz.open(pdf_input)
    elif isinstance(pdf_input, (bytes, bytearray)):
        doc = fitz.open(stream=bytes(pdf_input), filetype="pdf")
    else:
        raise ValueError("Unsupported PDF input type.")

    pages = []
    for n, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if len(text) < min_text_len and HAS_PIL:
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr, _ = process_image(img)
            combined = f"{text}\n{ocr}".strip()
            if combined:
                pages.append(f"--- Page {n} (Scanned OCR) ---\n{combined}")
        else:
            pages.append(f"--- Page {n} ---\n{text}")
    doc.close()
    return "\n\n".join(pages)


def process_audio(audio_path: Union[str, Path], model_name: str = "base") -> str:
    if not HAS_WHISPER:
        raise ImportError("openai-whisper is required for audio transcription.")
    model = whisper.load_model(model_name)
    return model.transcribe(str(audio_path)).get("text", "").strip()


def ingest_media(content: Any, media_type: str = "auto") -> Tuple[str, str]:
    """Normalise any input to (text, modality_label). Empty text is a valid result."""
    if media_type == "auto":
        media_type = _sniff(content) or "text"

    if media_type == "pdf":
        return process_pdf(content), "pdf_document"
    if media_type == "audio":
        return process_audio(content), "audio_transcript"
    if media_type == "image":
        ocr, meta = process_image(content)
        parts = [ocr] + ([f"[Image metadata]\n{meta}"] if meta else [])
        return "\n\n".join(p for p in parts if p).strip(), "image_ocr"
    return str(content), "user_input"


class MediaProcessor:
    """Class interface for preprocess.py and other shield modules."""

    @classmethod
    def ingest(cls, content: Any, media_type: str = "auto") -> Tuple[str, str]:
        return ingest_media(content, media_type)

    process = ingest

    @classmethod
    def process_image(cls, image_input, enhance_contrast: bool = True):
        return process_image(image_input, enhance_contrast)

    @classmethod
    def process_pdf(cls, pdf_input, min_text_len: int = 20):
        return process_pdf(pdf_input, min_text_len)

    @classmethod
    def process_audio(cls, audio_path, model_name: str = "base"):
        return process_audio(audio_path, model_name)