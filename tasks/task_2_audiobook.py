import io
import os
import re
import time
import asyncio
import threading
from typing import List, Tuple, Dict, Optional

import pdfplumber
import streamlit as st
import edge_tts  # Microsoft Edge TTS

# --- Constants ---
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
OUTPUTS_DIR = os.path.join(ASSETS_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

VOICE_MAP = {
    "American Male": "en-US-GuyNeural",
    "American Female": "en-US-JennyNeural",
    "British Male": "en-GB-RyanNeural",
    "British Female": "en-GB-SoniaNeural",
}

# =========================================================
# PDF TEXT EXTRACTION & CLEANING
# =========================================================
def _read_pdf_text_lines(pdf_source) -> List[List[str]]:
    pages_lines: List[List[str]] = []
    try:
        with pdfplumber.open(pdf_source) as pdf:
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
                pages_lines.append(lines)
    except Exception as e:
        st.error(f"Error reading PDF with pdfplumber: {e}")
        return []
    return pages_lines


def _detect_headers_footers(pages_lines: List[List[str]]) -> Tuple[set, set]:
    if not pages_lines or len(pages_lines) < 3:
        return set(), set()
    first_line_counts, last_line_counts = {}, {}
    for lines in pages_lines:
        if not lines:
            continue
        first, last = lines[0].strip(), lines[-1].strip()
        if len(first) < 70:
            first_line_counts[first] = first_line_counts.get(first, 0) + 1
        if len(last) < 70:
            last_line_counts[last] = last_line_counts.get(last, 0) + 1
    threshold = max(2, int(0.5 * len(pages_lines)))
    common_headers = {k for k, v in first_line_counts.items() if v >= threshold}
    common_footers = {k for k, v in last_line_counts.items() if v >= threshold}
    return common_headers, common_footers


def _clean_text(pages_lines: List[List[str]]) -> str:
    good_lines = []
    headers, footers = _detect_headers_footers(pages_lines)
    ignore_set = headers.union(footers)

    for page in pages_lines:
        for line in page:
            if line in ignore_set:
                continue
            line_lower = line.lower()
            if "header:" in line_lower or "footer:" in line_lower or "page" in line_lower:
                continue
            if "title:" in line_lower or "author:" in line_lower or "test script" in line_lower:
                continue
            if line.strip().isdigit():
                continue
            good_lines.append(line)

    full_text = " ".join(good_lines)
    full_text = re.sub(r"-\s+", "", full_text)
    full_text = re.sub(r"\s+", " ", full_text)
    return full_text.strip()


# =========================================================
# TTS SYNTHESIS HELPERS
# =========================================================
async def _synthesize_audio_edge(text: str, voice_label: str, output_path: str) -> None:
    """Single voice synthesis (async). voice_label is the human-facing label used in VOICE_MAP."""
    voice_name = VOICE_MAP.get(voice_label)
    if not voice_name:
        raise ValueError(f"Invalid voice label: {voice_label}")
    communicate = edge_tts.Communicate(text, voice_name)
    await communicate.save(output_path)


def _synthesize_with_retries(
    text: str,
    voice_label: str,
    output_path: str,
    attempts: int = 2,
) -> Tuple[bool, Optional[str]]:
    """Run synthesis with retry; returns (success, error_message)."""
    last_err: Optional[str] = None
    for attempt in range(1, attempts + 1):
        try:
            # Ensure old file removed to avoid stale success
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass
            asyncio.run(_synthesize_audio_edge(text, voice_label, output_path))
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True, None
            last_err = "File not created or empty after synthesis."
        except Exception as e:
            last_err = f"Attempt {attempt} failed: {e}"
            # Small backoff between attempts
            time.sleep(0.8 * attempt)
    return False, last_err


def _run_synthesis_thread(
    text: str,
    voice_label: str,
    output_path: str,
    result_dict: Dict[str, Dict[str, Optional[str]]],
    lock: threading.Lock,
    attempts: int = 2,
):
    """Thread target for concurrent synthesis."""
    success, err = _synthesize_with_retries(text, voice_label, output_path, attempts=attempts)
    with lock:
        result_dict[voice_label] = {
            "path": output_path if success else "",
            "error": None if success else err,
        }


# =========================================================
# PUBLIC API
# =========================================================
def preprocess_pdf(pdf_file) -> str:
    """Extract and clean text only. Accepts a Streamlit UploadedFile or raw bytes."""
    try:
        if hasattr(pdf_file, "read"):
            try:
                pdf_file.seek(0)
            except Exception:
                pass
            pdf_source = io.BytesIO(pdf_file.read())
        else:
            pdf_source = io.BytesIO(pdf_file)
        pages_lines = _read_pdf_text_lines(pdf_source)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {e}")

    if not pages_lines:
        raise RuntimeError("Failed to read any text from the uploaded PDF.")

    cleaned_text = _clean_text(pages_lines)
    if not cleaned_text:
        raise RuntimeError("Could not extract any meaningful text to synthesize.")
    return cleaned_text


def synthesize_all_voices(
    cleaned_text: str,
    parallel: bool = False,
    attempts: int = 2,
) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Synthesize all voices.

    Returns a dict:
    {
      "American Male": {"path": ".../audiobook_American_Male.mp3", "error": None},
      "British Female": {"path": "", "error": "Attempt 1 failed: ..."}
      ...
    }

    parallel=False (default) runs voices sequentially (more stable on some hosts).
    Set parallel=True to use threads.
    """
    results: Dict[str, Dict[str, Optional[str]]] = {}
    if not parallel:
        # Sequential (safer)
        for voice_label in VOICE_MAP.keys():
            safe_label = voice_label.replace(" ", "_")
            output_path = os.path.join(OUTPUTS_DIR, f"audiobook_{safe_label}.mp3")
            success, err = _synthesize_with_retries(cleaned_text, voice_label, output_path, attempts=attempts)
            results[voice_label] = {
                "path": output_path if success else "",
                "error": err if not success else None,
            }
        return results

    # Parallel version
    threads: List[threading.Thread] = []
    lock = threading.Lock()

    for voice_label in VOICE_MAP.keys():
        safe_label = voice_label.replace(" ", "_")
        output_path = os.path.join(OUTPUTS_DIR, f"audiobook_{safe_label}.mp3")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
        t = threading.Thread(
            target=_run_synthesis_thread,
            args=(cleaned_text, voice_label, output_path, results, lock, attempts),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    return results


def synthesize_single_voice(cleaned_text: str, voice_label: str, attempts: int = 2) -> Dict[str, Optional[str]]:
    """
    Regenerate a single voice (used to retry failed ones).
    Returns {"path": "...", "error": "..."}.
    """
    if voice_label not in VOICE_MAP:
        return {"path": "", "error": f"Unknown voice label '{voice_label}'"}
    safe_label = voice_label.replace(" ", "_")
    output_path = os.path.join(OUTPUTS_DIR, f"audiobook_{safe_label}.mp3")
    success, err = _synthesize_with_retries(cleaned_text, voice_label, output_path, attempts=attempts)
    return {
        "path": output_path if success else "",
        "error": err if not success else None,
    }


def preprocess_and_synthesize_all_voices(pdf_file) -> Tuple[str, Dict[str, Dict[str, Optional[str]]]]:
    """
    Convenience combined operation (sequential).
    Returns (cleaned_text, synthesis_results_dict).
    """
    cleaned_text = preprocess_pdf(pdf_file)
    results = synthesize_all_voices(cleaned_text, parallel=False)
    return cleaned_text, results