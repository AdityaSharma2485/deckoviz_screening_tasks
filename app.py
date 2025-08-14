import os
import json
import time
from typing import Any, Dict

import streamlit as st

from tasks.task_1_search import initialize_search_engine, perform_search
from tasks.task_2_audiobook import convert_pdf_to_audio
from tasks.task_3_storybook import create_storybook


ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
OUTPUTS_DIR = os.path.join(ASSETS_DIR, "outputs")
PERSONAS_DIR = os.path.join(ASSETS_DIR, "personas")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


st.set_page_config(page_title="Deckoviz Screening App", layout="centered")
st.title("Deckoviz Screening Test Submission")


with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Natural Language Search", "PDF to Audiobook", "Storybook Creator"],
        index=0,
    )
    st.markdown("---")
    st.caption("Ensure GOOGLE_API_KEY is set for LLM-powered features.")


if page == "Natural Language Search":
    st.subheader("Task 1: Natural Language Search (RAG)")
    if st.button("Initialize/Search Engine"):
        with st.spinner("Initializing ChromaDB and indexing personas..."):
            info = initialize_search_engine()
        st.success(info.get("message", "Initialized"))

    query = st.text_area("Describe who you're looking for", height=120, placeholder="Example: Outdoorsy non-smoker in NYC who loves tennis. Don't want smokers.")
    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a query")
        else:
            with st.spinner("Searching and reasoning with Gemini..."):
                results: Dict[str, Any] = perform_search(query)
            st.write("Results:")
            st.json(results)

elif page == "PDF to Audiobook":
    st.subheader("Task 2: PDF to Audiobook")
    pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    voice = st.radio("Voice/Accent", ["American (Default)", "British", "Australian", "Indian", "Irish", "South African"], horizontal=True)
    if st.button("Convert to MP3"):
        if not pdf:
            st.warning("Please upload a PDF file")
        else:
            with st.spinner("Extracting, cleaning, and synthesizing..."):
                try:
                    out_path = convert_pdf_to_audio(pdf, voice)
                    st.success("MP3 generated!")
                    audio_bytes = open(out_path, "rb").read()
                    st.audio(audio_bytes, format="audio/mp3")
                    st.download_button("Download MP3", data=audio_bytes, file_name=os.path.basename(out_path))
                except Exception as e:
                    st.error(f"Conversion failed: {e}")

elif page == "Storybook Creator":
    st.subheader("Task 3: Storybook Creator")
    text_input = st.text_area("Enter story text", height=200, placeholder="Once upon a time...")
    if st.button("Create Storybook PDF"):
        if not text_input.strip():
            st.warning("Please enter some story text")
        else:
            with st.spinner("Generating story sections, prompts, and assembling PDF..."):
                try:
                    pdf_path = create_storybook(text_input)
                    st.success("Storybook PDF created!")
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    st.download_button("Download Storybook PDF", data=pdf_bytes, file_name=os.path.basename(pdf_path), mime="application/pdf")
                except Exception as e:
                    st.error(f"Failed to create storybook: {e}")


