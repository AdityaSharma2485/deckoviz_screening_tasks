import io
import os
import re
import time
from typing import List, Tuple

import pdfplumber
from gtts import gTTS


ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
OUTPUTS_DIR = os.path.join(ASSETS_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _read_pdf_text_lines(pdf_source) -> List[List[str]]:
    """Read PDF and return a list of pages, each a list of lines."""
    pages_lines: List[List[str]] = []
    try:
        with pdfplumber.open(pdf_source) as pdf:
            for page in pdf.pages:
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
                pages_lines.append(lines)
    except Exception:
        return []
    return pages_lines


def _detect_headers_footers(pages_lines: List[List[str]]) -> Tuple[set, set]:
    """Detect frequent header/footer lines across pages to remove them."""
    if not pages_lines:
        return set(), set()

    first_line_counts = {}
    last_line_counts = {}
    for lines in pages_lines:
        if not lines:
            continue
        first = lines[0].strip()
        last = lines[-1].strip()
        first_line_counts[first] = first_line_counts.get(first, 0) + 1
        last_line_counts[last] = last_line_counts.get(last, 0) + 1

    threshold = max(2, int(0.5 * len(pages_lines)))
    common_headers = {k for k, v in first_line_counts.items() if v >= threshold}
    common_footers = {k for k, v in last_line_counts.items() if v >= threshold}
    return common_headers, common_footers


def _clean_lines(pages_lines: List[List[str]]) -> List[str]:
    """Remove headers/footers, page numbers, and join hyphenated words."""
    headers, footers = _detect_headers_footers(pages_lines)
    cleaned: List[str] = []

    for lines in pages_lines:
        buf: List[str] = []
        for ln_idx, line in enumerate(lines):
            raw = line.strip()
            if not raw:
                continue
            if raw in headers and ln_idx == 0:
                continue
            if raw in footers and ln_idx == len(lines) - 1:
                continue
            if re.match(r"^(page\s*)?\d+\s*$", raw, flags=re.IGNORECASE):
                continue
            if re.match(r"^\d+\s*/\s*\d+$", raw):
                continue
            buf.append(raw)

        # Join hyphenations and rebuild paragraphs
        paragraph = []
        for i, ln in enumerate(buf):
            if not paragraph:
                paragraph.append(ln)
                continue
            prev = paragraph[-1]
            if prev.endswith("-"):
                paragraph[-1] = prev[:-1] + ln.lstrip()
            else:
                # If line looks like a continuation of sentence, add space
                if prev and not prev.endswith(('.', '!', '?', '"', "'")):
                    paragraph[-1] = prev + " " + ln.lstrip()
                else:
                    paragraph.append(ln)

        cleaned.extend(paragraph)

    # Normalize whitespace and remove lingering artifacts
    normalized: List[str] = []
    for p in cleaned:
        p = re.sub(r"\s+", " ", p)
        p = re.sub(r"\s+([,.;:!?])", r"\1", p)
        normalized.append(p.strip())

    return normalized


def _choose_tld(voice_choice: str) -> str:
    choice = (voice_choice or "").lower()
    if "british" in choice:
        return "co.uk"
    if "australian" in choice:
        return "com.au"
    if "indian" in choice:
        return "co.in"
    if "irish" in choice:
        return "ie"
    if "south africa" in choice or "south african" in choice:
        return "co.za"
    # default American
    return "com"


def convert_pdf_to_audio(pdf_file, voice_choice: str) -> str:
    """Convert a PDF to an MP3 audiobook file using gTTS.

    - Extracts text with pdfplumber and cleans it
    - Joins hyphenated words, removes headers/footers/page numbers
    - Maps voice choices to gTTS accents via TLD; gTTS does not support gendered voices
    - Saves file into assets/outputs and returns the path
    """
    # Support both path and streamlit UploadedFile-like objects
    pdf_source = pdf_file
    if hasattr(pdf_file, "read"):
        # Reset pointer to the beginning to ensure proper reading
        try:
            pdf_file.seek(0)
        except Exception:
            pass
        pdf_source = io.BytesIO(pdf_file.read())

    pages_lines = _read_pdf_text_lines(pdf_source)
    if not pages_lines:
        raise RuntimeError("Failed to read any text from PDF")

    paragraphs = _clean_lines(pages_lines)
    full_text = "\n\n".join(paragraphs)

    # gTTS conversion
    tld = _choose_tld(voice_choice)
    try:
        tts = gTTS(text=full_text[:4000], lang="en", tld=tld)  # limit length for demo stability
        ts = int(time.time())
        out_path = os.path.join(OUTPUTS_DIR, f"audiobook_{ts}.mp3")
        tts.save(out_path)
        return out_path
    except Exception as e:
        raise RuntimeError(f"gTTS synthesis failed: {e}")


