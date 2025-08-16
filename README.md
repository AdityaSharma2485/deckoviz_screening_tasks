# Deckoviz Screening Tasks

A Streamlit application implementing three AI-adjacent tasks with graceful fallbacks:

1. Natural Language Persona Search (lightweight local RAG)
2. PDF → Multi‑Voice Audiobook (Edge Neural TTS)
3. Storybook Creator (sectioning, prompt generation, placeholder imagery, per‑page narration + PDF export)
4. Persona Showcase (interactive browsing of curated dataset)

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Task Details](#task-details)
  - [1. Natural Language Persona Search](#1-natural-language-persona-search)
  - [2. PDF → Audiobook](#2-pdf--audiobook)
  - [3. Storybook Creator](#3-storybook-creator)
  - [4. Persona Showcase](#4-persona-showcase)
- [Development Narrative (System Enhancements)](#development-narrative-system-enhancements)
- [Query Syntax & Constraints](#query-syntax--constraints)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Persona Dataset](#persona-dataset)
- [Performance Notes](#performance-notes)
- [Limitations](#limitations)
- [Enhancement Ideas](#enhancement-ideas)

---

## Overview
This repository showcases pragmatic implementations of:
- Local semantic search over a set of richly structured personas (no remote vector service required).
- Robust PDF text extraction + cleaning -> multi-voice neural TTS generation for immediate playback.
- Story segmentation and illustrative prompt generation with optional Gemini augmentation; fallback heuristics keep the feature usable offline.
- Exportable, customizable storybook PDFs (font, size, layout) plus on-demand page narration.

Design goals:
- Deterministic, inspectable fallbacks when LLM access is missing.
- Minimal external dependencies beyond model downloads and optional Gemini.
- Clear separation between UI orchestration (Streamlit) and task logic (tasks/).

---

## Key Features
- Persistent ChromaDB vector store; one-time persona embedding using SentenceTransformer (all-MiniLM-L6-v2).
- Negative constraint parsing (e.g., “no smokers”, “don't want remote roles”).
- Four pre-generated premium Edge Neural voices for audiobooks:
  - American Male (en-US-GuyNeural)
  - American Female (en-US-JennyNeural)
  - British Male (en-GB-RyanNeural)
  - British Female (en-GB-SoniaNeural)
- Header/footer and boilerplate suppression in PDFs (frequency-based detection).
- Hybrid section splitting: Gemini JSON extraction if available; otherwise proportional word chunk fallback.
- Compact image prompt generation (≤8 words) with LLM/fallback symmetry.
- Interactive persona carousel using personas_index.csv metadata.
- Per-page TTS narration in Storybook mode (currently en-US-JennyNeural).

---

## Architecture
High-level flow:

```
Streamlit (app.py)
├── Task 1: task_1_search.py
│     ├─ Initialize persistent Chroma collection (assets/outputs/chroma)
│     ├─ Embed persona .txt files
│     ├─ Extract negative constraints
│     ├─ Similarity retrieval + (optional) Gemini reasoning
│
├── Task 2: task_2_audiobook.py
│     ├─ pdfplumber extraction
│     ├─ Header/footer detection
│     ├─ Text cleaning
│     ├─ Parallel (sequential in UI) Edge TTS synthesis for 4 voices
│
├── Task 3: task_3_storybook.py
│     ├─ Section creation (Gemini JSON or fallback)
│     ├─ Prompt generation (Gemini JSON or fallback)
│     ├─ Placeholder image generation (generate_segmind_image)
│     ├─ PDF assembly (reportlab)
│     └─ Per-page TTS (Edge)
│
└── Personas Toolkit: personas_tools.py
      ├─ Validation
      ├─ Tag & region inference
      ├─ Dataset README generation
```

Persistent / generated artifacts under assets/outputs.

---

## Task Details

### 1. Natural Language Persona Search
- Embedding model: all-MiniLM-L6-v2 (SentenceTransformer).
- Vector DB: Chroma (cosine).
- Indexing: First call creates collection; subsequent calls short-circuit (no change detection yet).
- Negative constraints: Regex patterns identify phrases following triggers (e.g., “no”, “don't want”, “avoid”).
- Result formatting: If Gemini available, can enrich/explain results; fallback returns raw similarity ordering.
- Improvement options: file hashing for re-index, attribute-aware pre-filtering (age/location/tags).

### 2. PDF → Audiobook
- Extraction: pdfplumber per page with low tolerance tweaks.
- Cleaning:
  - Frequency-based header/footer removal.
  - Drop lines containing “page”, “header:”, “footer:” heuristically.
  - (Additional cleanup steps in unseen lines may apply.)
- TTS: edge-tts library; synthesizes four voices on button click; stored paths in session state for instant playback/download.
- Enhancement targets: text chunking for very long documents, progress meter per voice, configurable voice set.

### 3. Storybook Creator
- Input: Raw text or uploaded PDF (reuse Task 2 extraction).
- Sectioning: Gemini JSON {sections:[...]} or word-count fallback (cap 7).
- Prompt generation: Gemini JSON {"prompt": "..."} or heuristic (_fallback_prompt).
- Image generation: generate_segmind_image(pr) (implementation not shown—document assumptions about determinism).
- Per-page narration: Edge TTS voice (currently fixed).
- Export: reportlab with selectable font, size, and layout (“alternate” = image/text on separate pages?; “combined” = same page).
- Potential improvements: semantic sentence boundary segmentation, caching prompts/images, multi-voice option.

### 4. Persona Showcase
- Uses personas_index.csv (name, age, location, profession, tags, file).
- Rotational navigation with previous/next buttons.
- Full persona file view in an expander (raw text).

---

## Development Narrative (System Enhancements)

1. Natural Language Search
This was the most complex task, and I went through a full professional development cycle to get it right.  

The Challenge: Initial testing of a standard RAG pipeline revealed a critical "recall" problem. The system failed on complex, multi-faceted queries (e.g., "musician in Europe with tech interests") because a single query vector became too generalized to find the best specialists.  

The Solution: To solve this, I re-architected the system to use Multi-Query Retrieval. The app now uses the Gemini LLM to first decompose the user's query into several simpler, focused sub-queries. It runs a vector search for each, aggregates the unique candidates, and then re-ranks the combined pool. This dramatically improved the search results, fixing the failures I had identified.  

Final Touches: The final prompt to the LLM was refined to include "action points" (conversation starters), making the results more practical for the user.  

2. PDF to Audiobook Converter  
This task evolved significantly to prioritize audio quality and user experience.  

TTS Engine - A Deliberate Pivot: I evaluated several TTS engines. gTTS was too robotic, and Google Cloud's API, while high-quality, introduced authentication and billing hurdles. I made the deliberate decision to pivot to the edge-tts library, which provides access to Microsoft's excellent neural voices for free, without API keys, while still offering the required voice diversity.  

Performance Optimization: To create a non-blocking UI, the app pre-generates all four voice options in the background using multi-threading on PDF upload. The results are cached, allowing the user to switch between voices and get instant playback.  

3. Storybook Creator  
This task demonstrates a multi-modal pipeline from text to a final, formatted document.  

Content Generation: I used the Gemini LLM for two key creative steps: first, to act as an editor and break the source text into logical "scenes," and second, to act as an art director and generate a descriptive image prompt for each scene.  

Image Generation: To ensure the project is fully runnable without paid API keys, I integrated the Segmind API, which offers a free tier for high-quality image generation.  

Fallback Logic: Both the text chunking and image prompting steps include deterministic, non-LLM fallbacks. If the Gemini API is unavailable, the system will still function by using word-count-based text splitting and simple keyword extraction for prompts.  

---

## Query Syntax & Constraints
Supported negative constraint patterns (case-insensitive examples):
- “no smokers”
- “don't want remote roles”
- “do not want freelancers”
- “avoid lawyers”
- “without relocation”

Internally, patterns capture the token(s) after the trigger words; terms are lowercased and excluded if they appear in persona text. Multi-word phrases may overmatch; avoid very broad exclusions like “no a”.

---

## Technology Stack
- Python 3.10+ (recommended)
- Streamlit (UI)
- SentenceTransformers (embeddings)
- ChromaDB (vector persistence)
- Google Generative AI (optional Gemini 1.5 Flash)
- pdfplumber (text extraction)
- edge-tts (Microsoft Edge Neural voices)
- reportlab (PDF generation)
- requests (image fetching)
- pandas (persona index browsing)

---

## Project Structure
```
.
├── app.py
├── tasks/
│   ├── task_1_search.py
│   ├── task_2_audiobook.py
│   ├── task_3_storybook.py
│   ├── personas_tools.py
│   └── __init__.py
├── assets/
│   ├── personas/
│   │   ├── README_PERSONAS.md
│   │   ├── personas_index.csv
│   │   └── *.txt  (50 persona bios)
│   └── outputs/
│       ├── chroma/        (vector store)
│       ├── audio/         (generated MP3s)
│       ├── storybooks/    (exported PDFs)
│       └── images/        (downloaded images if cached)
├── requirements.txt
└── README.md
```
---

## Persona Dataset
- 50 personas; uniform schema (Name, Age, Location, Profession, Backstory, Core Motivation, Fears & Insecurities, Hobbies & Passions, Media Diet, Communication Style, Quirk or Contradiction, Bio & Current Focus).
- personas_tools.py:
  - Validates field presence.
  - Infers regions via REGION_MAP and domain tags via regex KEYWORD_TAGS.
  - Produces personas_index.csv and supports README generation.
- README_PERSONAS.md summarizes coverage (regions/domains).

---

## Performance Notes
- All-MiniLM-L6-v2 model (~80MB) loads quickly on CPU; embedding 50 personas is near-instant (<1s typical).
- Chroma persistence avoids repeated embeddings.
- Edge TTS synthesis time scales with text length; generating four voices serially may be slow for large PDFs (consider concurrent tasks).
- Gemini calls (if used) add network latency; fallback avoids blocking.

---

## Limitations
- No automatic re-index if persona files change after first initialization.
- No chunking strategy for extremely large PDF audiobook texts (possible TTS failures).
- Single narration voice in Storybook (fixed).
- No analytics / instrumentation (logging limited to Streamlit UI messages).
- Absence of test suite / CI pipeline.

---

## Enhancement Ideas
- File content hashing for incremental persona re-index.
- Add similarity scores & applied negative filters in search output for transparency.
- Configurable voice set & parallel synthesis (asyncio gather).
- Token-based chunking for TTS with seamless MP3 concatenation.
- Image generation via local diffusion or stable API with caching layer.
- Add unit tests for regex negative constraint parsing, header/footer detection, fallback segmentation, prompt fallback.

---
