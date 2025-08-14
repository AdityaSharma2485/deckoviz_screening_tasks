# Deckoviz Screening Test Submission

## Overview
This Streamlit app showcases three AI-powered tasks in a single interface:

- Natural Language Search: A mini-RAG system that embeds and searches mock personas with sentence-transformers and ChromaDB, then asks an LLM to produce structured, ranked matches.
- PDF to Audiobook: Extracts and cleans text from PDFs and converts it to MP3 using gTTS with accent options.
- Storybook Creator: Splits input text into sections, generates short prompts, fetches placeholder images, and assembles a PDF storybook with text and images.

## Tech Stack
- Streamlit for UI
- sentence-transformers (all-MiniLM-L6-v2) for embeddings
- ChromaDB for local vector search
- Google Gemini (google-generativeai) for LLM steps (optional with graceful fallbacks)
- pdfplumber for PDF text extraction
- gTTS for text-to-speech (accent via TLD; no gender differentiation)
- reportlab for PDF generation
- requests + placehold.co for placeholder images

## Setup and Usage
1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Set your Gemini API key to enable LLM features:
   - On Windows PowerShell:
     ```powershell
     $env:GOOGLE_API_KEY = "YOUR_KEY"
     ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Design Choices
- ChromaDB was chosen for simplicity and reliability in local development with persistent storage.
- The search task includes negative constraint filtering by scanning retrieved documents for forbidden keywords.
- Placeholder images from placehold.co were used for the storybook to avoid any paid API requirements while remaining fully functional.
- All API calls and file operations are wrapped in try/except with sensible fallbacks to keep the demo robust.

## Notes
- Mock personas are stored under `assets/personas`. Initialize the search engine once to index them.
- Generated artifacts (MP3s, PDFs, placeholders) are saved under `assets/outputs`.
- If the LLM is unavailable or the API key is not set, both the search and storybook tasks fall back to deterministic logic so the app remains usable.



