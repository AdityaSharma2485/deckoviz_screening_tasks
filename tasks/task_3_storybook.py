import json
import os
import time
from typing import List, Dict, Any

import requests
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None


ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
OUTPUTS_DIR = os.path.join(ASSETS_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _maybe_use_gemini_sections(text_input: str) -> List[str]:
    if genai is None:
        return []
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return []
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Split the following text into 5-7 coherent, page-sized sections. "
            "Return strictly JSON with schema {\"sections\": [\"...\"]}. No extra text.\n\n" + text_input
        )
        resp = model.generate_content(prompt)
        raw = resp.text if hasattr(resp, "text") else str(resp)
        # Remove fences
        if raw.strip().startswith("```") and raw.strip().endswith("```"):
            raw = raw.strip().split("\n", 1)[1].rsplit("\n", 1)[0]
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("sections"), list):
            sections = [s for s in data["sections"] if isinstance(s, str) and s.strip()]
            return sections[:7]
    except Exception:
        return []
    return []


def _fallback_sections(text_input: str) -> List[str]:
    words = text_input.split()
    if not words:
        return []
    target_sections = 5
    per = max(1, len(words) // target_sections)
    sections: List[str] = []
    for i in range(0, len(words), per):
        chunk = " ".join(words[i:i + per]).strip()
        if chunk:
            sections.append(chunk)
        if len(sections) >= 7:
            break
    return sections


def _maybe_use_gemini_prompt(section_text: str) -> str:
    if genai is None:
        return ""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Create a short, descriptive image prompt (max 8 words) for this story section. "
            "Return strictly JSON {\"prompt\": \"...\"}.\n\n" + section_text
        )
        resp = model.generate_content(prompt)
        raw = resp.text if hasattr(resp, "text") else str(resp)
        if raw.strip().startswith("```") and raw.strip().endswith("```"):
            raw = raw.strip().split("\n", 1)[1].rsplit("\n", 1)[0]
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("prompt"), str):
            return data["prompt"].strip()
    except Exception:
        return ""
    return ""


def _fallback_prompt(section_text: str) -> str:
    text = section_text.strip().split(".")[0][:40]
    if not text:
        text = "Story scene"
    return f"Illustration: {text}"


def _download_placeholder(text: str, path: str) -> None:
    from urllib.parse import quote_plus

    url = f"https://placehold.co/600x400/EEE/31343C?text={quote_plus(text)}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)


def _draw_wrapped_text(c: canvas.Canvas, text: str, x: float, y: float, max_width: float, leading: float) -> float:
    from reportlab.pdfbase.pdfmetrics import stringWidth

    lines: List[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        buf = []
        while words:
            buf.append(words.pop(0))
            w = stringWidth(" ".join(buf + ([words[0]] if words else [])), "Helvetica", 12)
            if w > max_width:
                lines.append(" ".join(buf))
                buf = []
        if buf:
            lines.append(" ".join(buf))
        lines.append("")  # paragraph break
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def create_storybook(text_input: str) -> str:
    """Create a storybook PDF alternating text and placeholder images.

    - Uses Gemini to chunk text and generate prompts if available; otherwise falls back
    - Downloads placeholder images from placehold.co with prompt text
    - Assembles a PDF alternating pages
    - Returns output PDF path
    """
    if not text_input or not text_input.strip():
        raise ValueError("Empty text input")

    sections = _maybe_use_gemini_sections(text_input)
    if not sections:
        sections = _fallback_sections(text_input)
    if not sections:
        raise ValueError("Unable to create sections from input text")

    prompts: List[str] = []
    for sec in sections:
        prompt = _maybe_use_gemini_prompt(sec)
        if not prompt:
            prompt = _fallback_prompt(sec)
        prompts.append(prompt)

    ts = int(time.time())
    pdf_path = os.path.join(OUTPUTS_DIR, f"storybook_{ts}.pdf")

    # Prepare images
    image_paths: List[str] = []
    for i, p in enumerate(prompts):
        img_path = os.path.join(OUTPUTS_DIR, f"placeholder_{ts}_{i+1}.png")
        try:
            _download_placeholder(p, img_path)
            image_paths.append(img_path)
        except Exception:
            # Skip image if download fails
            image_paths.append("")

    # Build PDF
    c = canvas.Canvas(pdf_path, pagesize=LETTER)
    width, height = LETTER
    margin = 0.8 * inch
    text_width = width - 2 * margin
    y_start = height - margin

    for idx, sec in enumerate(sections):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y_start, f"Page {idx + 1}: Story")
        c.setFont("Helvetica", 12)
        y = y_start - 0.3 * inch
        y = _draw_wrapped_text(c, sec, margin, y, text_width, leading=14)
        c.showPage()

        # Image page
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y_start, f"Page {idx + 1}: Illustration")
        if image_paths[idx]:
            try:
                iw = width - 2 * margin
                ih = iw * (400.0 / 600.0)
                c.drawImage(image_paths[idx], margin, y_start - ih - 0.2 * inch, width=iw, height=ih, preserveAspectRatio=True)
            except Exception:
                c.setFont("Helvetica", 12)
                c.drawString(margin, y_start - 0.4 * inch, "[Image failed to render]")
        else:
            c.setFont("Helvetica", 12)
            c.drawString(margin, y_start - 0.4 * inch, "[Image unavailable]")
        c.showPage()

    c.save()
    return pdf_path


