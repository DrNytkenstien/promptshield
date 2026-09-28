from __future__ import annotations

import io
import re
import shutil
from pathlib import Path
from typing import Tuple, Union, Optional, Dict, Any, List

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
    Image = None  # type: ignore
    ImageEnhance = None  # type: ignore
    ExifTags = None  # type: ignore
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

# Graceful import for OpenCV
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


# ---------------------------------------------------------------------------
# Text sanitization helpers
# ---------------------------------------------------------------------------

# Marker that the media layer prepends to any threat it detects (e.g. malicious QR code).
# Downstream stages (preprocess, shield, fuse) look for it via has_security_alert().
SECURITY_ALERT_MARKER = "[SECURITY ALERT"
SECURITY_ALERT_RE = re.compile(r"\[\s*SECURITY\s+ALERT\b", re.IGNORECASE)


def has_security_alert(text: Any) -> bool:
    """True if `text` contains a '[SECURITY ALERT ...' tag (also '[SECURITY ALERT - QR THREAT DETECTED]')."""
    return bool(text) and bool(SECURITY_ALERT_RE.search(str(text)))


# Control characters and non-printable code points (tab, LF and CR are kept).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Metadata keys that hold binary blobs. These are always skipped.
_BINARY_KEYS = {
    "icc_profile",
    "exif",
    "photoshop",
    "xmp",
    "iptc",
    "makernote",
    "usercomment",
    "printimatching",
    "componentsconfiguration",
    "filesource",
    "scenetype",
    "thumbnail",
    "interoperabilityindex",
}

# Cap on the length of any single metadata value.
_MAX_METADATA_VALUE_LEN = 500

_IMAGE_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".ico",
)


def _decode_printable(data: bytes) -> Optional[str]:
    """Decode bytes only if they form clean, printable UTF-8 text; otherwise None."""
    data = bytes(data).rstrip(b"\x00")
    if not data:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not text.strip():
        return None
    if _CONTROL_CHARS.search(text) or "\ufffd" in text:
        return None
    if not all(ch.isprintable() or ch in "\t\n\r" for ch in text):
        return None
    return text


def sanitize_text(text: Any) -> str:
    """Strip control characters / non-printable bytes so no binary junk leaks out.

    Raw bytes are only kept if they decode cleanly to printable UTF-8; otherwise
    they are dropped entirely (empty string is returned).
    """
    if text is None:
        return ""
    if isinstance(text, (bytes, bytearray)):
        decoded = _decode_printable(bytes(text))
        if decoded is None:
            return ""
        text = decoded
    elif not isinstance(text, str):
        text = str(text)
    text = _CONTROL_CHARS.sub("", text)
    text = text.replace("\ufffd", "")
    return text.strip()


def _is_binary_key(key: Any) -> bool:
    name = str(key).strip().lower()
    return name in _BINARY_KEYS or "xmp" in name or "icc" in name


def _coerce_value(value: Any) -> Optional[str]:
    """Convert a metadata value to clean text, or None if it should be dropped."""
    if isinstance(value, (bytes, bytearray)):
        text = _decode_printable(bytes(value))
    elif isinstance(value, str):
        text = sanitize_text(value)
    elif isinstance(value, (bool, int, float)):
        text = str(value)
    elif isinstance(value, (tuple, list)):
        parts = [_coerce_value(v) for v in value]
        # Any binary element poisons the whole value.
        text = None if any(p is None for p in parts) else ", ".join(parts)  # type: ignore[arg-type]
    elif isinstance(value, dict):
        text = None
    else:
        # e.g. IFDRational and other numeric-like PIL types
        text = sanitize_text(str(value))

    if not text:
        return None
    text = sanitize_text(text)
    if not text:
        return None
    return text[:_MAX_METADATA_VALUE_LEN]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def extract_image_metadata(img: Any) -> str:
    """Extract textual EXIF / image-info metadata, skipping all binary data."""
    metadata_chunks: List[str] = []

    # EXIF
    try:
        exif = img.getexif() if hasattr(img, "getexif") else None
        if not exif and hasattr(img, "_getexif"):
            exif = img._getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id) if ExifTags else tag_id
                if _is_binary_key(tag):
                    continue
                clean = _coerce_value(value)
                if clean:
                    metadata_chunks.append(f"EXIF {sanitize_text(tag)}: {clean}")
    except Exception:
        pass

    # Format-specific info (PNG text chunks, JPEG comments, etc.)
    try:
        info = getattr(img, "info", None)
        if isinstance(info, dict):
            for k, v in info.items():
                if _is_binary_key(k):
                    continue
                if not isinstance(v, (str, bytes, bytearray)):
                    continue
                clean = _coerce_value(v)
                if clean:
                    metadata_chunks.append(f"Metadata {sanitize_text(k)}: {clean}")
    except Exception:
        pass

    return "\n".join(metadata_chunks)


def preprocess_image_for_ocr(img: Any) -> Any:
    gray = img.convert("L")
    enhancer = ImageEnhance.Contrast(gray)
    return enhancer.enhance(2.5)


def extract_exif(img: Any) -> str:
    """Extract EXIF metadata using extract_image_metadata."""
    return sanitize_text(extract_image_metadata(img))


def extract_ocr(img: Any, enhance_contrast: bool = True) -> str:
    """Extract OCR text using pytesseract with fallback."""
    if not HAS_TESSERACT:
        return ""
    try:
        processed_img = preprocess_image_for_ocr(img) if enhance_contrast else img
        return sanitize_text(pytesseract.image_to_string(processed_img))
    except Exception:
        return ""


def _load_image(image_input: Any) -> Optional[Any]:
    """Turn bytes / path / PIL image into a loaded PIL image, or None on failure."""
    if not HAS_PIL:
        return None
    try:
        if isinstance(image_input, Image.Image):
            return image_input
        if isinstance(image_input, (bytes, bytearray)):
            img = Image.open(io.BytesIO(bytes(image_input)))
        elif isinstance(image_input, (str, Path)):
            img = Image.open(image_input)
        else:
            return None
        img.load()
        return img
    except Exception:
        return None


def process_image(image_bytes: Union[bytes, Any], enhance_contrast: bool = True) -> str:
    """Extract QR payloads, EXIF metadata and OCR text from an image.

    Accepts raw bytes, a file path, or a PIL Image. Returns clean, printable text
    only; undecodable input yields an empty string instead of raw binary.
    """
    image = _load_image(image_bytes)
    if image is None:
        return ""

    extracted_parts: List[str] = []

    # 1. Scan QR Codes for Malware & Injections
    for qr in extract_and_scan_qr_codes(image):
        payload = sanitize_text(qr["payload"])
        if payload:
            extracted_parts.append(f"[QR Content]: {payload}")
        if qr["is_malicious"]:
            flags_str = " | ".join(qr["threat_flags"])
            extracted_parts.append(f"{SECURITY_ALERT_MARKER} - QR THREAT DETECTED]: {flags_str}")

    # 2. Extract EXIF metadata
    exif_text = extract_exif(image)
    if exif_text:
        extracted_parts.append(f"[EXIF Metadata]: {exif_text}")

    # 3. Perform standard OCR text extraction
    ocr_text = extract_ocr(image, enhance_contrast)
    if ocr_text:
        extracted_parts.append(f"[OCR Text]: {ocr_text}")

    return sanitize_text("\n".join(extracted_parts))


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
        page_text = sanitize_text(page.get_text())

        if len(page_text) < min_text_len and HAS_PIL:
            try:
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                image_text = process_image(img)
                combined = f"{page_text}\n{image_text}".strip()
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
    return sanitize_text(result.get("text", ""))


def _looks_like_image_input(content: Any) -> bool:
    if isinstance(content, (bytes, bytearray)):
        return True
    if HAS_PIL and isinstance(content, Image.Image):
        return True
    if isinstance(content, (str, Path)):
        return str(content).lower().endswith(_IMAGE_EXTENSIONS)
    return False


def ingest_media(content: Any, media_type: str = "auto") -> Tuple[str, str]:
    if isinstance(content, (str, Path)) and str(content).lower().endswith(".pdf"):
        media_type = "pdf"

    if media_type == "pdf":
        return process_pdf(content), "pdf_document"

    if media_type == "audio" or (isinstance(content, (str, Path)) and str(content).lower().endswith((".mp3", ".wav", ".m4a"))):
        return process_audio(content), "audio_transcript"

    if media_type == "image" or (media_type == "auto" and _looks_like_image_input(content)):
        if _load_image(content) is not None:
            # Image decoded successfully: return whatever text it has, even if empty.
            # Never fall through to str(content), which would leak raw bytes.
            return process_image(content), "image_ocr"

    # Fallback: treat as plain user input, never emitting raw binary.
    return sanitize_text(content), "user_input"


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
        return process_image(image_input, enhance_contrast), "image_ocr"

    @classmethod
    def process_pdf(cls, pdf_input: Union[str, Path, bytes], min_text_len: int = 20) -> str:
        return process_pdf(pdf_input, min_text_len)

    @classmethod
    def process_audio(cls, audio_path: Union[str, Path], model_name: str = "base") -> str:
        return process_audio(audio_path, model_name)


# ---------------------------------------------------------------------------
# QR code scanning
# ---------------------------------------------------------------------------

# High-risk malware & threat patterns inside QR payloads
MALWARE_EXTENSIONS = re.compile(
    r"\.(exe|apk|bat|sh|ps1|vbs|scr|dll|cmd|msi|iso|dmg|jar|js)\b", re.IGNORECASE
)
RAW_IP_URL = re.compile(
    r"https?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?", re.IGNORECASE
)
DANGEROUS_PROTOCOLS = re.compile(
    r"^(javascript|data|file|vbscript|intent):", re.IGNORECASE
)
SUSPICIOUS_TUNNELS = re.compile(
    r"(ngrok\.io|requestbin|pipedream\.net|discord\.com/api/webhooks|webhook\.site)", re.IGNORECASE
)
INJECTION_KEYWORDS = re.compile(
    r"(system\s+override|ignore\s+previous|disregard\s+all|exfiltrate|output\s+api\s+key|act\s+as\s+root)", re.IGNORECASE
)


def analyze_qr_payload(payload: str) -> Dict[str, Any]:
    """Analyzes a QR code text payload for malware, quishing, and prompt injection signatures."""
    payload = sanitize_text(payload)
    threats = []

    # 1. Check for dangerous script execution protocols or Data URIs
    if DANGEROUS_PROTOCOLS.search(payload):
        threats.append("MALWARE_PROTOCOL: Dangerous URI scheme detected (e.g., javascript:, data:)")

    # 2. Check for direct executable file downloads
    if MALWARE_EXTENSIONS.search(payload):
        threats.append("MALWARE_DOWNLOAD: Direct link to executable/payload file in QR code")

    # 3. Check for raw IP address hosting (common in malware command-and-control)
    if RAW_IP_URL.search(payload):
        threats.append("SUSPICIOUS_HOST: URL uses raw IP address instead of domain name")

    # 4. Check for exfiltration endpoints and tunneling services
    if SUSPICIOUS_TUNNELS.search(payload):
        threats.append("EXFIL_TUNNEL: Exfiltration service or webhook destination detected")

    # 5. Check for embedded prompt injection instructions
    if INJECTION_KEYWORDS.search(payload):
        threats.append("INDIRECT_INJECTION: Prompt injection command detected inside QR code")

    is_malicious = len(threats) > 0
    return {
        "payload": payload,
        "is_malicious": is_malicious,
        "threat_flags": threats
    }


def extract_and_scan_qr_codes(image: Image.Image) -> List[Dict[str, Any]]:
    """Detects QR codes in an image and scans them for security threats."""
    if not HAS_OPENCV:
        return []

    try:
        open_cv_image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        detector = cv2.QRCodeDetector()

        retval, decoded_info, _, _ = detector.detectAndDecodeMulti(open_cv_image)
        results = []

        payloads = []
        if retval:
            payloads = [info.strip() for info in decoded_info if info.strip()]
        else:
            data, _, _ = detector.detectAndDecode(open_cv_image)
            if data and data.strip():
                payloads.append(data.strip())

        for p in payloads:
            if not sanitize_text(p):
                continue
            results.append(analyze_qr_payload(p))

        return results
    except Exception:
        return []
