import os
import asyncio
import streamlit as st
import pandas as pd
import edge_tts

from tasks.task_1_search import initialize_search_engine, perform_search
from tasks.task_2_audiobook import preprocess_pdf, synthesize_all_voices
from tasks.task_3_storybook import (
    create_sections,
    create_prompts,
    generate_segmind_image,
    create_storybook,
)


ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
OUTPUTS_DIR = os.path.join(ASSETS_DIR, "outputs")
PERSONAS_DIR = os.path.join(ASSETS_DIR, "personas")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


st.set_page_config(page_title="Storybook Creator", layout="centered")


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

    # Session storage for interactive viewer
    if "storybook_sections" not in st.session_state:
        st.session_state["storybook_sections"] = []
    if "storybook_images" not in st.session_state:
        st.session_state["storybook_images"] = []  # List[bytes]
    if "storybook_page" not in st.session_state:
        st.session_state["storybook_page"] = 0
    if "storybook_audio" not in st.session_state:
        st.session_state["storybook_audio"] = {}  # Dict[int, bytes]

    # Input mode (Text or PDF)
    input_mode = st.radio("Choose input type", ["Text", "PDF"], horizontal=True)

    story_text = ""
    uploaded_pdf = None
    if input_mode == "Text":
        story_text = st.text_area("Enter story text", height=200, placeholder="Once upon a time...")
    else:
        uploaded_pdf = st.file_uploader("Upload a PDF story", type=["pdf"], key="storybook_pdf")
        if uploaded_pdf is not None and st.checkbox("Preview extracted text before generating", value=False):
            try:
                with st.spinner("Extracting and cleaning text from PDF..."):
                    preview_text = preprocess_pdf(uploaded_pdf)
                with st.expander("Extracted Text Preview", expanded=False):
                    st.write(preview_text)
            except Exception as e:
                st.warning(f"Preview extraction failed: {e}")

    # Controls line: sections and per-section word slider
    col_sections, col_words = st.columns([1, 1])
    with col_sections:
        num_sections = st.number_input(
            "Sections",
            min_value=3,
            max_value=10,
            value=6,
            step=1,
            help="How many story pages to generate (image + text)",
        )
    with col_words:
        per_section_words = st.slider("Words per section", min_value=80, max_value=140, value=110, step=5)

    # Full-width generate button below
    generate_clicked = st.button("Generate Storybook", use_container_width=True)

    if generate_clicked:
        # Resolve source text
        if input_mode == "PDF":
            if uploaded_pdf is None:
                st.warning("Please upload a PDF or switch to Text input.")
                st.stop()
            try:
                with st.spinner("Extracting and cleaning text from PDF..."):
                    story_text = preprocess_pdf(uploaded_pdf)
            except Exception as e:
                st.error(f"Failed to extract text from PDF: {e}")
                st.stop()
        if not story_text.strip():
            st.warning("Please enter some story text.")
        else:
            with st.spinner("Creating summarized sections and generating images..."):
                try:
                    sections = create_sections(story_text, desired_sections=num_sections, per_section_words=int(per_section_words))
                    if not sections:
                        st.error("Could not create sections from the provided text.")
                        st.stop()
                    prompts = create_prompts(sections)
                    images: list[bytes] = []
                    progress = st.progress(0)
                    for i, pr in enumerate(prompts):
                        img_bytes = generate_segmind_image(pr)
                        images.append(img_bytes)
                        progress.progress(int((i + 1) / max(1, len(prompts)) * 100))
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.stop()

                st.session_state["storybook_sections"] = sections
                st.session_state["storybook_images"] = images
                st.session_state["storybook_page"] = 0
                st.session_state["storybook_audio"] = {}
                st.success("Story generated! Use the viewer below.")

    # Viewer
    sections = st.session_state.get("storybook_sections", [])
    images = st.session_state.get("storybook_images", [])
    if sections and images and len(sections) == len(images):
        total_pages = len(sections)
        page = st.session_state.get("storybook_page", 0)
        page = max(0, min(page, total_pages - 1))
        st.session_state["storybook_page"] = page

        left, right = st.columns([1, 1])
        with left:
            try:
                st.image(images[page], use_container_width=True)
            except Exception:
                st.warning("Unable to display image for this page.")
        with right:
            st.markdown(f"### Page {page + 1}")
            # Render text with chosen font size
            fs_px = 16
            styled = f"<div style='font-size:{fs_px}px; line-height:1.5;'>{sections[page]}</div>"
            st.markdown(styled, unsafe_allow_html=True)

            # Per-page TTS
            narrate_key = f"narrate_{page}"
            if st.button("🔊 Narrate this page", key=narrate_key):
                async def _synthesize_page_audio(text: str) -> bytes:
                    tts = edge_tts.Communicate(text, "en-US-JennyNeural")
                    audio_data = bytearray()
                    async for chunk in tts.stream():
                        if chunk["type"] == "audio":
                            audio_data.extend(chunk["data"])
                    return bytes(audio_data)

                try:
                    audio_bytes = asyncio.run(_synthesize_page_audio(sections[page]))
                    st.session_state["storybook_audio"][page] = audio_bytes
                except Exception as e:
                    st.error(f"TTS failed: {e}")

            if page in st.session_state["storybook_audio"]:
                st.audio(st.session_state["storybook_audio"][page], format="audio/mp3")

        nav1, mid, nav2 = st.columns([1, 2, 1])
        with nav1:
            if st.button("⬅️ Previous", disabled=(total_pages <= 1)):
                st.session_state["storybook_page"] = (page - 1) % total_pages
        with nav2:
            if st.button("Next ➡️", disabled=(total_pages <= 1)):
                st.session_state["storybook_page"] = (page + 1) % total_pages

        # PDF export controls (font, size, layout)
        st.markdown("---")
        exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 1])
        with exp_col1:
            export_font = st.selectbox(
                "Export font",
                ["Helvetica", "Times-Roman", "Courier"],
                index=0,
            )
        with exp_col2:
            export_font_size = st.number_input("Export font size", min_value=10, max_value=18, value=12, step=1)
        with exp_col3:
            export_layout = st.radio("Layout", ["alternate", "combined"], index=0, horizontal=True)

        # Single-step export: clicking the download button triggers the file download
        try:
            pdf_path = create_storybook(
                sections=sections,
                image_bytes_list=images,
                body_font=export_font,
                body_font_size=int(export_font_size),
                layout=export_layout,
            )
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                "Export PDF",
                data=pdf_bytes,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Failed to generate PDF: {e}")

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


