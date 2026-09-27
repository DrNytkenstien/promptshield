"""Step 1 of the pipeline: clean up content and pull out anything hidden from humans.

Attackers hide instructions where a person will not see them but an AI will:
CSS-hidden text, HTML comments, zero-width characters, Unicode "tag" characters,
look-alike letters, base64 blobs. This module:

  1. separates VISIBLE text (what a human sees) from HIDDEN text (what only the AI sees),
  2. records every trick it found as a Finding (hiding text is itself a warning sign),
  3. splits everything into sentence-sized Chunks so each can be scored separately
     and the dashboard can highlight exactly which part is the attack.

Known limitation (say this to judges): we only understand INLINE styles. Text hidden by
CSS classes defined in <style> blocks or external stylesheets is treated as visible.
"""
from __future__ import annotations
from promptshield.media import MediaProcessor
from promptshield.media import MediaProcessor

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Comment

from .config import SETTINGS
from .types import Chunk, Finding

# ---------------------------------------------------------------- unicode tricks
ZERO_WIDTH = {"​", "‌", "‍", "⁠", "﻿", "᠎"}
BIDI_CONTROLS = {chr(c) for c in list(range(0x202A, 0x202F)) + list(range(0x2066, 0x206A))}
TAG_BLOCK = range(0xE0000, 0xE0080)     # "ASCII smuggling": invisible copies of ASCII letters

# Scripts that are often mixed into Latin words to fool keyword filters (look-alike letters)
_CONFUSABLE_SCRIPTS = ("CYRILLIC", "GREEK")

# ---------------------------------------------------------------- CSS hiding rules
_HIDDEN_STYLE_PATTERNS = [
    (re.compile(r"display\s*:\s*none", re.I), "display:none"),
    (re.compile(r"visibility\s*:\s*hidden", re.I), "visibility:hidden"),
    (re.compile(r"opacity\s*:\s*0(\.0+)?\s*(;|$)", re.I), "opacity:0"),
    (re.compile(r"font-size\s*:\s*[01](\.\d+)?\s*(px|pt|em|rem)?\s*(;|$)", re.I), "font-size:0/1"),
    (re.compile(r"(left|top|text-indent)\s*:\s*-\d{3,}", re.I), "moved off-screen"),
    (re.compile(r"(height|width)\s*:\s*0(px)?\s*;.*overflow\s*:\s*hidden", re.I), "zero-size box"),
]
# white or near-white text colour (we assume a white email background)
_WHITE_TEXT = re.compile(
    r"(?<![-\w])color\s*:\s*(#fff\b|#ffffff\b|white\b|rgb\(\s*25[0-5]\s*,\s*25[0-5]\s*,\s*25[0-5]\s*\))", re.I)

_BASE64 = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


@dataclass
class Preprocessed:
    visible_text: str
    hidden_texts: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)


# ================================================================= public entry point
media_processor = MediaProcessor()
def preprocess(content: str | bytes | Path, modality: str = "auto") -> Preprocessed:
    """Clean `content` (HTML, plain text, or media files) and split it into scoreable chunks."""
    findings: list[Finding] = []
    hidden: list[str] = []

    # 2. Add media extraction block here (before HTML parsing)
    if isinstance(content, (bytes, Path)) or modality != "auto":
        media_res = media_processor.process(content, modality=modality)
        content = media_res["text"]
        if media_res.get("metadata_text"):
            hidden.append(media_res["metadata_text"])
            findings.append(
                Finding(
                    kind="media_metadata",
                    detail=f"Extracted EXIF/metadata from {media_res['modality']}",
                    text=media_res["metadata_text"],
                )
            )

    if _looks_like_html(content):
        visible, html_hidden, html_findings = _split_html(content)
        hidden.extend(html_hidden)
        findings.extend(html_findings)
    else:
        visible = content

    # Unicode tricks are checked on everything (visible and hidden)
    visible, uni_hidden, uni_findings = _clean_unicode(visible)
    hidden.extend(uni_hidden)
    findings.extend(uni_findings)
    cleaned_hidden = []
    for h in hidden:
        h2, extra_hidden, extra_findings = _clean_unicode(h)
        cleaned_hidden.append(h2)
        cleaned_hidden.extend(extra_hidden)
        findings.extend(extra_findings)
    hidden = [h for h in cleaned_hidden if h.strip()]

    # Encoded payloads and exfiltration links
    b64_hidden, b64_findings = _decode_base64(visible + "\n" + "\n".join(hidden))
    hidden.extend(b64_hidden)
    findings.extend(b64_findings)
    findings.extend(_find_exfil_links(visible + "\n" + "\n".join(hidden)))

    chunks = _chunk(visible, hidden)
    return Preprocessed(visible_text=_squash_ws(visible), hidden_texts=hidden,
                        findings=findings, chunks=chunks)


# ================================================================= HTML
def _looks_like_html(text: str) -> bool:
    return bool(re.search(r"<\s*(html|body|div|p|span|table|br|a|img|font|!--)\b", text, re.I))


def _style_hidden_reason(style: str) -> str | None:
    for pattern, reason in _HIDDEN_STYLE_PATTERNS:
        if pattern.search(style):
            return reason
    if _WHITE_TEXT.search(style):
        return "white-on-white text"
    return None


def _element_hidden_reason(el) -> str | None:
    """Is this element (or any ancestor) hidden from a human reader?"""
    node = el
    while node is not None and getattr(node, "name", None):
        if node.name in {"script", "style", "head", "template", "noscript"}:
            return f"inside <{node.name}>"
        if node.has_attr("hidden"):
            return "hidden attribute"
        style = node.get("style", "")
        if style:
            reason = _style_hidden_reason(style)
            if reason:
                return reason
        node = node.parent
    return None


def _split_html(html: str) -> tuple[str, list[str], list[Finding]]:
    soup = BeautifulSoup(html, "html.parser")
    visible_parts: list[str] = []
    hidden_parts: list[str] = []
    findings: list[Finding] = []

    # 1) HTML comments
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        text = str(c).strip()
        if text:
            hidden_parts.append(text)
            findings.append(Finding(kind="html_comment", detail="Text inside an HTML comment", text=text))
        c.extract()

    # 2) Attribute text a human doesn't normally read (alt/title/aria-label/meta content)
    for el in soup.find_all(True):
        for attr in ("alt", "title", "aria-label"):
            val = el.get(attr)
            if isinstance(val, str) and len(val.split()) >= 4:
                hidden_parts.append(val)
                findings.append(Finding(kind="attribute_text", detail=f"Long text in {attr}= attribute", text=val))
        if el.name == "meta" and el.get("content") and len(el["content"].split()) >= 4:
            hidden_parts.append(el["content"])
            findings.append(Finding(kind="meta_text", detail="Text inside a <meta> tag", text=el["content"]))

    # 3) Text nodes: visible or hidden?
    for s in soup.find_all(string=True):
        text = str(s)
        if not text.strip():
            continue
        reason = _element_hidden_reason(s.parent)
        if reason:
            if reason in {"inside <style>", "inside <script>"}:
                continue            # code, not prose
            hidden_parts.append(text.strip())
            findings.append(Finding(kind="hidden_css", detail=f"Text hidden from humans ({reason})",
                                    text=text.strip()))
        else:
            visible_parts.append(text)

    return " ".join(visible_parts), hidden_parts, findings


# ================================================================= Unicode
def _clean_unicode(text: str) -> tuple[str, list[str], list[Finding]]:
    findings: list[Finding] = []
    hidden: list[str] = []

    # ASCII smuggling: decode Unicode tag characters back into the hidden message
    tag_chars = [ch for ch in text if ord(ch) in TAG_BLOCK]
    if tag_chars:
        smuggled = "".join(chr(ord(ch) - 0xE0000) for ch in tag_chars if 0x20 <= ord(ch) - 0xE0000 < 0x7F)
        if smuggled.strip():
            hidden.append(smuggled)
            findings.append(Finding(kind="unicode_tags", detail="Invisible Unicode tag characters (ASCII smuggling)",
                                    text=smuggled))
        text = "".join(ch for ch in text if ord(ch) not in TAG_BLOCK)

    zw = sum(1 for ch in text if ch in ZERO_WIDTH)
    if zw:
        findings.append(Finding(kind="zero_width", detail=f"{zw} zero-width characters (can split words to dodge filters)"))
        text = "".join(ch for ch in text if ch not in ZERO_WIDTH)

    bidi = sum(1 for ch in text if ch in BIDI_CONTROLS)
    if bidi:
        findings.append(Finding(kind="bidi_control", detail=f"{bidi} text-direction control characters"))
        text = "".join(ch for ch in text if ch not in BIDI_CONTROLS)

    mixed = _mixed_script_words(text)
    if mixed:
        findings.append(Finding(kind="homoglyph", detail="Words mixing Latin with look-alike letters: " + ", ".join(mixed[:5])))

    # NFKC turns full-width and other compatibility forms into plain characters
    text = unicodedata.normalize("NFKC", text)
    text = _fold_confusables(text)
    return text, hidden, findings


def _mixed_script_words(text: str) -> list[str]:
    out = []
    for word in re.findall(r"\w{3,}", text):
        scripts = set()
        for ch in word:
            if ch.isalpha():
                name = unicodedata.name(ch, "")
                if name.startswith("LATIN"):
                    scripts.add("LATIN")
                elif name.startswith(_CONFUSABLE_SCRIPTS):
                    scripts.add(name.split()[0])
        if "LATIN" in scripts and len(scripts) > 1:
            out.append(word)
    return out


# Small map of the most common Cyrillic/Greek look-alikes -> Latin, so rules still match
_CONFUSABLE_MAP = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
    "ο": "o", "α": "a", "ε": "e", "ι": "i", "ν": "v", "Ο": "O", "Α": "A", "Ε": "E", "Ι": "I",
})


def _fold_confusables(text: str) -> str:
    return text.translate(_CONFUSABLE_MAP)


# ================================================================= encoded payloads / exfil
def _decode_base64(text: str) -> tuple[list[str], list[Finding]]:
    hidden, findings = [], []
    for match in _BASE64.findall(text):
        try:
            decoded = base64.b64decode(match + "=" * (-len(match) % 4), validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        printable = sum(ch.isprintable() or ch.isspace() for ch in decoded) / max(len(decoded), 1)
        if printable > 0.95 and len(decoded.split()) >= 3:
            hidden.append(decoded)
            findings.append(Finding(kind="base64", detail="Base64-encoded text decoded", text=decoded))
    return hidden, findings


def _find_exfil_links(text: str) -> list[Finding]:
    findings = []
    for url in _MD_IMAGE.findall(text):
        if "?" in url or "{" in url:
            findings.append(Finding(kind="exfil_image",
                                    detail="Markdown image whose URL carries data (can leak info when rendered)",
                                    text=url))
    return findings


# ================================================================= chunking
def _squash_ws(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\s*\n\s*", "\n", text)).strip()


def _split_sentences(text: str) -> list[str]:
    text = _squash_ws(text)
    pieces, buf = [], ""
    for sent in _SENTENCE_SPLIT.split(text):
        sent = sent.strip()
        if not sent:
            continue
        if len(buf) + len(sent) + 1 <= SETTINGS.max_chunk_chars and len(buf) < 60:
            buf = (buf + " " + sent).strip()       # merge very short sentences
        else:
            if buf:
                pieces.append(buf)
            buf = sent
    if buf:
        pieces.append(buf)
    # hard cap: long sentences get cut so no text is ever ignored by a model's length limit
    out = []
    for p in pieces:
        while len(p) > SETTINGS.max_chunk_chars:
            cut = p.rfind(" ", 0, SETTINGS.max_chunk_chars)
            cut = cut if cut > 0 else SETTINGS.max_chunk_chars
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out


def _chunk(visible: str, hidden: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sent in _split_sentences(visible):
        chunks.append(Chunk(index=len(chunks), text=sent, hidden=False))
    for h in hidden:
        for sent in _split_sentences(h):
            chunks.append(Chunk(index=len(chunks), text=sent, hidden=True))
    return chunks
