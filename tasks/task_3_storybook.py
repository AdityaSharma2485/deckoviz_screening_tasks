import json
import os
import time
from io import BytesIO
from base64 import b64decode
from typing import List, Dict, Any
import re

import streamlit as st
import requests
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None


ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
OUTPUTS_DIR = os.path.join(ASSETS_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Mapping of core ReportLab fonts to their bold counterparts
CORE_BOLD_MAP: Dict[str, str] = {
    "Helvetica": "Helvetica-Bold",
    "Times-Roman": "Times-Bold",
    "Courier": "Courier-Bold",
}


def _maybe_use_gemini_sections(text_input: str, target_sections: int = 6, per_section_words: int = 110) -> List[str]:
    if genai is None:
        return []
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return []
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        words = max(80, min(140, int(per_section_words)))
        prompt = (
            "You are an expert children's book editor. "
            f"Split the story into exactly {int(target_sections)} coherent scene sections (between 3 and 10 overall). "
            f"Each section should be a self-contained scene of around {words} words (±20), easy to illustrate, and avoid duplicating prior content. "
            "Prefer vivid, concrete actions and imagery. Avoid meta commentary.\n\n"
            "Return strictly JSON with schema {\"sections\": [\"...\"]} and nothing else.\n\n"
            + text_input
        )
        resp = model.generate_content(prompt)
        raw = resp.text if hasattr(resp, "text") else str(resp)
        # Remove fences
        if raw.strip().startswith("```") and raw.strip().endswith("```"):
            raw = raw.strip().split("\n", 1)[1].rsplit("\n", 1)[0]
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("sections"), list):
            sections = [s for s in data["sections"] if isinstance(s, str) and s.strip()]
            return sections[: max(1, int(target_sections))]
    except Exception:
        return []
    return []


def _fallback_sections(text_input: str, target_sections: int = 6, per_section_words: int = 110) -> List[str]:
    """Sentence-aware splitting into approximately equal sections.

    Keeps sentences intact and balances section sizes by word count.
    """
    text = re.sub(r"\s+", " ", text_input).strip()
    if not text:
        return []
    target = max(1, int(target_sections))
    desired = max(80, min(140, int(per_section_words)))
    # Split into sentences keeping punctuation (simple heuristic)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text]

    total_words = sum(len(s.split()) for s in sentences)
    # Aim for desired words per section but cap by evenly distributing the text
    target_words = max(1, min(desired, max(1, total_words // target + 10)))

    sections: List[str] = []
    current: List[str] = []
    current_words = 0
    for sent in sentences:
        sent_words = len(sent.split())
        if current and (current_words + sent_words) > target_words and len(sections) < target - 1:
            sections.append(" ".join(current).strip())
            current = []
            current_words = 0
        current.append(sent)
        current_words += sent_words

    if current:
        sections.append(" ".join(current).strip())

    # Ensure exactly target sections by merging or splitting as needed
    if len(sections) > target:
        # Merge adjacent sections until we reach target
        while len(sections) > target:
            merged: List[str] = []
            i = 0
            while i < len(sections):
                if i + 1 < len(sections) and len(merged) < target - (len(sections) - i - 1):
                    merged.append((sections[i] + " " + sections[i + 1]).strip())
                    i += 2
                else:
                    merged.append(sections[i])
                    i += 1
            sections = merged
    elif len(sections) < target:
        # If too few, re-balance by words with a simple chunking
        words = text.split()
        per = max(1, len(words) // target)
        sections = []
        for i in range(0, len(words), per):
            chunk = " ".join(words[i:i + per]).strip()
            if chunk:
                sections.append(chunk)
            if len(sections) >= target:
                break

    return sections[:target]


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
            "Create a concise but vivid illustration prompt (max 20 words) for this children's story scene. "
            "Describe character(s), setting, mood, composition, and lighting in a neutral style (no proper names). "
            "Return strictly JSON {\"prompt\": \"...\"} with no extra text.\n\n" + section_text
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


def generate_segmind_image(prompt: str) -> bytes:
    """Generate a storybook-style image from Segmind using a text prompt.

    Enhances the prompt for a consistent illustrated style and returns raw image bytes.
    """
    api_key = None
    try:
        api_key = st.secrets["SEGMIND_API_KEY"]
    except Exception:
        pass
    if not api_key:
        raise RuntimeError("Missing Segmind API key in Streamlit secrets: 'SEGMIND_API_KEY'")

    enhanced_prompt = (
        "storybook illustration, children's picture book, vibrant colors, soft shading, whimsical, "
        "clean outlines, friendly characters, depth of field, high detail, "
        + prompt
    )

    url = "https://api.segmind.com/v1/ssd-1b"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }
    # Keep payload conservative to reduce generation time and timeouts
    payload: Dict[str, Any] = {
        "prompt": enhanced_prompt,
        "width": 576,   # Slightly smaller for faster generation
        "height": 768,
        "steps": 20,
        "guidance_scale": 7.0,
        "num_images": 1,
        "negative_prompt": (
            "text, watermark, signature, logo, low quality, blurry, deformed, extra limbs, "
            "cropped, worst quality, lowres, jpeg artifacts, username, captions"
        ),
    }

    # Retry with exponential backoff on timeouts and transient server errors
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=(10, 180))
            # Retry on certain status codes
            if resp.status_code in {429, 500, 502, 503, 504}:
                last_error = RuntimeError(f"Segmind server responded {resp.status_code}")
                raise last_error
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "").lower()
            if content_type.startswith("image/"):
                return resp.content
            # Some APIs respond with JSON containing base64-encoded images
            try:
                data = resp.json()
                if isinstance(data, dict):
                    if isinstance(data.get("image"), str):
                        return b64decode(data["image"])  # type: ignore[arg-type]
                    if isinstance(data.get("output"), list) and data["output"]:
                        first = data["output"][0]
                        if isinstance(first, str):
                            if "," in first:
                                first = first.split(",", 1)[1]
                            return b64decode(first)
            except Exception:
                pass
            # If we got here, format was unexpected
            raise RuntimeError("Unexpected Segmind API response format; could not obtain image bytes")
        except requests.Timeout as e:
            last_error = e
        except requests.RequestException as e:
            last_error = e

        # Backoff before next attempt
        sleep_seconds = 2 * (attempt + 1)
        try:
            time.sleep(sleep_seconds)
        except Exception:
            pass

    if last_error:
        raise RuntimeError(f"Segmind generation failed after retries: {last_error}")
    raise RuntimeError("Segmind generation failed after retries")


def _draw_wrapped_text(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    max_width: float,
    leading: float,
    font_name: str,
    font_size: int,
) -> float:
    from reportlab.pdfbase.pdfmetrics import stringWidth

    lines: List[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        buf = []
        while words:
            buf.append(words.pop(0))
            w = stringWidth(" ".join(buf + ([words[0]] if words else [])), font_name, font_size)
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


def create_sections(text_input: str, desired_sections: int = 6, per_section_words: int = 110) -> List[str]:
    if not text_input or not text_input.strip():
        return []
    # Clamp desired sections to 3..10
    desired = max(3, min(10, int(desired_sections)))
    sections = _maybe_use_gemini_sections(text_input, target_sections=desired, per_section_words=per_section_words)
    if not sections:
        sections = _fallback_sections(text_input, target_sections=desired, per_section_words=per_section_words)
    # Normalize to exactly desired count if model returned a different number
    if len(sections) != desired:
        sections = _fallback_sections(" ".join(sections), target_sections=desired, per_section_words=per_section_words)
    return sections[:desired]


def create_prompts(sections: List[str]) -> List[str]:
    prompts: List[str] = []
    for sec in sections:
        prompt = _maybe_use_gemini_prompt(sec)
        if not prompt:
            prompt = _fallback_prompt(sec)
        prompts.append(prompt)
    return prompts


def create_storybook(
    sections: List[str],
    image_bytes_list: List[bytes],
    body_font: str = "Helvetica",
    body_font_size: int = 12,
    layout: str = "alternate",
) -> str:
    """Assemble a storybook PDF.

    layout: "alternate" for text page then image page, or "combined" for image left + text right on one page.

    Returns the path to the generated PDF.
    """
    if not sections:
        raise ValueError("No sections provided")
    if not image_bytes_list or len(image_bytes_list) != len(sections):
        raise ValueError("Images list must match sections length")

    ts = int(time.time())
    pdf_path = os.path.join(OUTPUTS_DIR, f"storybook_{ts}.pdf")

    c = canvas.Canvas(pdf_path, pagesize=LETTER)
    width, height = LETTER
    margin = 0.8 * inch
    text_width = width - 2 * margin
    y_start = height - margin

    heading_font = CORE_BOLD_MAP.get(body_font, "Helvetica-Bold")
    heading_font_size = max(body_font_size + 2, 12)
    leading = max(body_font_size + 2, int(body_font_size * 1.3))

    if layout == "combined":
        # One page per section with image left, text right
        for idx, sec in enumerate(sections):
            c.setFont(heading_font, heading_font_size)
            c.drawString(margin, y_start, f"Page {idx + 1}")

            # Image column
            image_col_x = margin
            image_col_w = (width - 2 * margin) * 0.48
            image_col_h = height - 2 * margin - 0.5 * inch
            try:
                img_reader = ImageReader(BytesIO(image_bytes_list[idx]))
                aspect = 800.0 / 600.0
                draw_w = image_col_w
                draw_h = draw_w * aspect
                if draw_h > image_col_h:
                    draw_h = image_col_h
                    draw_w = draw_h / aspect
                c.drawImage(
                    img_reader,
                    image_col_x,
                    y_start - draw_h - 0.2 * inch,
                    width=draw_w,
                    height=draw_h,
                    preserveAspectRatio=True,
                )
            except Exception:
                c.setFont(body_font, body_font_size)
                c.drawString(image_col_x, y_start - 0.4 * inch, "[Image failed to render]")

            # Text column
            text_col_x = margin + image_col_w + 0.4 * inch
            text_col_w = (width - 2 * margin) - image_col_w - 0.4 * inch
            c.setFont(body_font, body_font_size)
            text_start_y = y_start - 0.3 * inch
            _ = _draw_wrapped_text(
                c,
                sec,
                text_col_x,
                text_start_y,
                text_col_w,
                leading=float(leading),
                font_name=body_font,
                font_size=int(body_font_size),
            )
            c.showPage()
    else:
        # Alternate pages: text page then image page for each section
        for idx, sec in enumerate(sections):
            # Text page
            c.setFont(heading_font, heading_font_size)
            c.drawString(margin, y_start, f"Page {idx + 1}: Story")
            c.setFont(body_font, body_font_size)
            y = y_start - 0.3 * inch
            _ = _draw_wrapped_text(
                c,
                sec,
                margin,
                y,
                text_width,
                leading=float(leading),
                font_name=body_font,
                font_size=int(body_font_size),
            )
            c.showPage()

            # Image page
            c.setFont(heading_font, heading_font_size)
            c.drawString(margin, y_start, f"Page {idx + 1}: Illustration")
            try:
                img_reader = ImageReader(BytesIO(image_bytes_list[idx]))
                available_width = width - 2 * margin
                available_height = height - 2 * margin - 0.5 * inch
                aspect = 800.0 / 600.0
                draw_w = available_width
                draw_h = draw_w * aspect
                if draw_h > available_height:
                    draw_h = available_height
                    draw_w = draw_h / aspect
                c.drawImage(
                    img_reader,
                    margin,
                    y_start - draw_h - 0.2 * inch,
                    width=draw_w,
                    height=draw_h,
                    preserveAspectRatio=True,
                )
            except Exception:
                c.setFont(body_font, body_font_size)
                c.drawString(margin, y_start - 0.4 * inch, "[Image failed to render]")
            c.showPage()

    c.save()
    return pdf_path



