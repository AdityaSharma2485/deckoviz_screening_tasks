import os
import streamlit as st
import pandas as pd

from tasks.task_1_search import initialize_search_engine, perform_search
from tasks.task_2_audiobook import preprocess_pdf, synthesize_all_voices
from tasks.task_3_storybook import create_storybook


ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
OUTPUTS_DIR = os.path.join(ASSETS_DIR, "outputs")
PERSONAS_DIR = os.path.join(ASSETS_DIR, "personas")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


st.set_page_config(page_title="Deckoviz Screening App", layout="centered")
st.title("Deckoviz Screening Test Submission")


@st.cache_data
def load_persona_data():
    """Load personas_index.csv into a pandas DataFrame (cached)."""
    csv_path = os.path.join(PERSONAS_DIR, "personas_index.csv")
    df = pd.read_csv(csv_path)
    return df

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Natural Language Search", "PDF to Audiobook", "Storybook Creator", "Persona Showcase"],
        index=1,  # Start on the audiobook page for convenience
    )
    st.markdown("---")
    st.info("Ensure your Gemini API Key is set in `.streamlit/secrets.toml` for all AI-powered features.")


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
                results = perform_search(query)
            st.write("Results:")
            st.json(results)

elif page == "PDF to Audiobook":
    st.subheader("Task 2: PDF to Audiobook")
    st.markdown("This version pre-processes your PDF and pre-generates all four Edge TTS voice options for instant playback.")

    if "audiobook_text" not in st.session_state:
        st.session_state["audiobook_text"] = None
    if "audiobook_paths" not in st.session_state:
        st.session_state["audiobook_paths"] = None
    if "last_pdf_name" not in st.session_state:
        st.session_state["last_pdf_name"] = None

    pdf = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_upload")

    # If a new PDF is uploaded, clear previous session state and preprocess only
    if pdf is not None:
        current_name = getattr(pdf, "name", None)
        if current_name != st.session_state.get("last_pdf_name"):
            st.session_state["last_pdf_name"] = current_name
            st.session_state["audiobook_text"] = None
            st.session_state["audiobook_paths"] = None
            with st.spinner("Preprocessing PDF (extracting and cleaning text)..."):
                try:
                    cleaned_text = preprocess_pdf(pdf)
                    st.session_state["audiobook_text"] = cleaned_text
                    st.success("Text extracted!")
                except Exception as e:
                    st.error(f"Preprocessing failed: {e}")

    if st.session_state.get("audiobook_text"):
        with st.expander("View Extracted Text", expanded=False):
            st.write(st.session_state["audiobook_text"])

        # Generate all voices on-demand via button
        if st.session_state.get("audiobook_paths") is None:
            if st.button("Generate Voices (All at once)"):
                with st.spinner("Generating all voice options..."):
                    try:
                        st.session_state["audiobook_paths"] = synthesize_all_voices(st.session_state["audiobook_text"])
                        st.success("All voices generated!")
                    except Exception as e:
                        st.error(f"Voice generation failed: {e}")

        # If voices exist, let the user select and play/download instantly
        if st.session_state.get("audiobook_paths"):
            voice = st.radio(
                "Choose a Voice",
                ["American Male", "American Female", "British Male", "British Female"],
                horizontal=True,
            )
            chosen_path = st.session_state["audiobook_paths"].get(voice)
            if chosen_path and os.path.exists(chosen_path):
                audio_bytes = open(chosen_path, "rb").read()
                st.audio(audio_bytes, format="audio/mp3")
                st.download_button("Download MP3", data=audio_bytes, file_name=os.path.basename(chosen_path))
            else:
                st.info("Click 'Generate Voices' to produce audio files for playback.")

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

elif page == "Persona Showcase":
    st.subheader("Persona Showcase")
    try:
        df = load_persona_data()
    except Exception as e:
        st.error(f"Failed to load persona index: {e}")
        df = None

    if df is not None and not df.empty:
        total = len(df)
        if "persona_index" not in st.session_state:
            st.session_state.persona_index = 0

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Previous"):
                st.session_state.persona_index = (st.session_state.persona_index - 1) % total
        with col2:
            if st.button("Next"):
                st.session_state.persona_index = (st.session_state.persona_index + 1) % total

        idx = st.session_state.persona_index % total
        row = df.iloc[idx]

        name = row.get("name", "Unknown")
        age = row.get("age", "?")
        location = row.get("location", "Unknown")
        profession = row.get("profession", "")
        tags = row.get("tags", "")
        filename = row.get("file", None)

        st.subheader(str(name))

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Age", str(age))
        with m2:
            st.metric("Location", str(location))

        st.markdown(f"**Profession:** {profession}")
        st.markdown(f"**Tags:** {tags}")

        if filename:
            persona_path = os.path.join(PERSONAS_DIR, filename)
            with st.expander("View Full Persona Bio"):
                try:
                    with open(persona_path, "r", encoding="utf-8") as f:
                        bio = f.read()
                    st.text(bio)
                except Exception as e:
                    st.warning(f"Unable to load persona bio: {e}")
        else:
            st.info("No persona file specified for this entry.")
        st.caption(f"Viewing {idx + 1} of {total}")
    else:
        st.info("No personas available to display.")


