import io
import os
import re
import time
import asyncio
import threading
from typing import List, Tuple, Dict

import pdfplumber
import streamlit as st
import edge_tts  # NEW: Import the edge-tts library

# --- Constants ---
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
OUTPUTS_DIR = os.path.join(ASSETS_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# --- NEW: Voice Mapping for Microsoft Edge TTS ---
# These are the voice names for the high-quality Edge voices.
VOICE_MAP = {
    "American Male": "en-US-GuyNeural",
    "American Female": "en-US-JennyNeural",
    "British Male": "en-GB-RyanNeural",
    "British Female": "en-GB-SoniaNeural",
}

# --- PDF Text Extraction and Cleaning (Unchanged) ---

def _read_pdf_text_lines(pdf_source) -> List[List[str]]:
    """Read PDF and return a list of pages, each a list of lines."""
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
    """Detect frequent header/footer lines across pages to remove them."""
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
    """
    A more aggressive cleaning function to remove headers, footers, page numbers,
    and other metadata from the extracted text.
    """
    good_lines = []
    # Detect repeating headers/footers to build a set of lines to ignore
    headers, footers = _detect_headers_footers(pages_lines)
    ignore_set = headers.union(footers)

    for page in pages_lines:
        for line in page:
            # Rule 1: Skip if the line is in our ignore set
            if line in ignore_set:
                continue
            
            # Rule 2: Skip if the line looks like a header/footer/page marker
            # This catches things the automatic detection might miss.
            line_lower = line.lower()
            if "header:" in line_lower or "footer:" in line_lower or "page" in line_lower:
                continue

            # Rule 3: Skip if the line is likely part of the initial title block
            if "title:" in line_lower or "author:" in line_lower or "test script" in line_lower:
                continue

            # Rule 4: Skip if the line is just a number (likely a page number)
            if line.strip().isdigit():
                continue

            # If none of the above rules apply, it's probably good content
            good_lines.append(line)

    # Join the good lines into a single block of text
    full_text = " ".join(good_lines)

    # Perform final cleanup on the entire text block
    # Join words that were hyphenated across lines
    full_text = re.sub(r"-\s+", "", full_text)
    # Normalize all whitespace to single spaces
    full_text = re.sub(r"\s+", " ", full_text)

    return full_text.strip()


# --- REWRITTEN: Microsoft Edge TTS Synthesis ---

async def _synthesize_audio_edge(text: str, voice_choice: str, output_path: str) -> None:
    """Synthesize audio using the edge-tts library and save to a file."""
    voice_name = VOICE_MAP.get(voice_choice)
    if not voice_name:
        raise ValueError("Invalid voice choice selected.")
        
    communicate = edge_tts.Communicate(text, voice_name)
    await communicate.save(output_path)


def _run_synthesis(cleaned_text: str, voice_label: str, output_path: str, result_map: Dict[str, str], lock: threading.Lock) -> None:
    """Thread target to synthesize using edge-tts for a specific voice.

    Writes the output file and records the path into result_map under the voice label.
    """
    try:
        asyncio.run(_synthesize_audio_edge(cleaned_text, voice_label, output_path))
        with lock:
            result_map[voice_label] = output_path
    except Exception as synthesis_error:
        # Record failure with empty path so UI can handle gracefully
        with lock:
            result_map[voice_label] = ""


# --- Public API functions ---

def preprocess_pdf(pdf_file) -> str:
    """Extract and clean text only. Accepts an UploadedFile or raw bytes."""
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


def synthesize_all_voices(cleaned_text: str) -> Dict[str, str]:
    """Synthesize all voices concurrently for the given cleaned text."""
    threads: List[threading.Thread] = []
    result_paths: Dict[str, str] = {}
    lock = threading.Lock()

    for voice_label in VOICE_MAP.keys():
        safe_label = voice_label.replace(" ", "_")
        output_path = os.path.join(OUTPUTS_DIR, f"audiobook_{safe_label}.mp3")
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass

        thread = threading.Thread(
            target=_run_synthesis,
            args=(cleaned_text, voice_label, output_path, result_paths, lock),
            daemon=True,
        )
        threads.append(thread)
        thread.start()

    for t in threads:
        t.join()

    return result_paths

def preprocess_and_synthesize_all_voices(pdf_file) -> Tuple[str, Dict[str, str]]:
    """Extract and clean text, then synthesize all voices concurrently.

    Returns a tuple of (cleaned_text, voice_to_path_map).
    """
    # Step 1: Extraction & Cleaning
    try:
        # Ensure we have a bytes-like source for pdfplumber
        if hasattr(pdf_file, "read"):
            try:
                pdf_file.seek(0)
            except Exception:
                pass
            pdf_source = io.BytesIO(pdf_file.read())
        else:
            # Assume bytes
            pdf_source = io.BytesIO(pdf_file)
        pages_lines = _read_pdf_text_lines(pdf_source)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {e}")

    if not pages_lines:
        raise RuntimeError("Failed to read any text from the uploaded PDF.")

    cleaned_text = _clean_text(pages_lines)
    if not cleaned_text:
        raise RuntimeError("Could not extract any meaningful text to synthesize.")

    # Step 2: Multi-threaded synthesis
    threads: List[threading.Thread] = []
    result_paths: Dict[str, str] = {}
    lock = threading.Lock()

    timestamp = int(time.time())
    for voice_label in VOICE_MAP.keys():
        safe_label = voice_label.replace(" ", "_")
        output_path = os.path.join(OUTPUTS_DIR, f"audiobook_{safe_label}.mp3")
        # Remove any stale file from previous runs to avoid confusion
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass

        thread = threading.Thread(
            target=_run_synthesis,
            args=(cleaned_text, voice_label, output_path, result_paths, lock),
            daemon=True,
        )
        threads.append(thread)
        thread.start()

    for t in threads:
        t.join()

    return cleaned_text, result_paths
