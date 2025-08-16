import os
import asyncio
from typing import List
import streamlit as st
import pandas as pd
import edge_tts

# Ensure modern SQLite for chromadb
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")  # type: ignore
except Exception:
    pass

# Fixed path handling for Streamlit Cloud
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
OUTPUTS_DIR = os.path.join(ASSETS_DIR, "outputs")
PERSONAS_DIR = os.path.join(ASSETS_DIR, "personas")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

st.set_page_config(page_title="Storybook Creator", layout="centered")

@st.cache_data
def load_persona_data():
    csv_path = os.path.join(PERSONAS_DIR, "personas_index.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()  # Return empty DataFrame if file doesn't exist
    return pd.read_csv(csv_path)

# --------------------------------------------------
# Sidebar Navigation (Fixed)
# --------------------------------------------------
NAV_PAGES = ["Natural Language Search", "PDF to Audiobook", "Storybook Creator", "Persona Showcase"]

with st.sidebar:
    st.header("Navigation")
    # Fixed navigation - removed conflicting key
    selected_page = st.radio("Go to", NAV_PAGES, index=1)
    st.markdown("---")

# Use the selected page directly instead of session state for navigation
page = selected_page

# --------------------------------------------------
# Page: Natural Language Search
# --------------------------------------------------
if page == "Natural Language Search":
    from tasks.task_1_search import initialize_search_engine, perform_search
    st.subheader("Task 1: Natural Language Search (RAG)")
    st.caption("Multi-query retrieval + heuristic re-ranking.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Initialize / Load Collection"):
            with st.spinner("Initializing..."):
                info = initialize_search_engine()
            st.success(info.get("message", "Initialized"))
    with c2:
        if st.button("Force Reindex"):
            with st.spinner("Re-indexing..."):
                info = initialize_search_engine(force_reindex=True)
            st.success(info.get("message", "Re-index complete"))

    # Persist the query using a key so it doesn't clear on rerun
    query = st.text_area(
        "Describe who you're looking for",
        height=140,
        placeholder="Example: Outdoorsy non-smoker in NYC who loves tennis. Don't want smokers.",
        key="rag_query",
    )


    if st.button("Search", key="rag_search_button"):
        if not query.strip():
            st.warning("Please enter a query")
        else:
            with st.spinner("Searching..."):
                results = perform_search(query)
            st.json(results)

# --------------------------------------------------
# Page: PDF to Audiobook
# --------------------------------------------------
elif page == "PDF to Audiobook":
    try:
        from tasks.task_2_audiobook import (
            preprocess_pdf,
            synthesize_all_voices,
            synthesize_single_voice,
        )

        st.subheader("Task 2: PDF to Audiobook")
        st.caption("Upload a PDF, extract text, generate all voices, then choose & play.")

        # Initialize session state with proper keys
        session_keys = {
            "audiobook_text": None,
            "audiobook_results": {},
            "last_pdf_name": None
        }
        
        for key, default in session_keys.items():
            if key not in st.session_state:
                st.session_state[key] = default

        pdf = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_upload")

        if pdf is not None:
            current_name = getattr(pdf, "name", None)
            if current_name != st.session_state.last_pdf_name:
                st.session_state.last_pdf_name = current_name
                st.session_state.audiobook_text = None
                st.session_state.audiobook_results = {}
                with st.spinner("Extracting and cleaning text..."):
                    try:
                        cleaned_text = preprocess_pdf(pdf)
                        st.session_state.audiobook_text = cleaned_text
                        st.success("Text extracted.")
                    except Exception as e:
                        st.error(f"Preprocessing failed: {e}")

        if st.session_state.audiobook_text:
            with st.expander("View Extracted Text"):
                txt = st.session_state.audiobook_text
                st.write(txt[:10000] + ("..." if len(txt) > 10000 else ""))

            if not st.session_state.audiobook_results and st.button("Generate All Voices"):
                with st.spinner("Generating voices..."):
                    try:
                        st.session_state.audiobook_results = synthesize_all_voices(
                            st.session_state.audiobook_text,
                            parallel=False,
                            attempts=2,
                        )
                    except Exception as e:
                        st.error(f"Voice generation failed: {e}")

            results = st.session_state.audiobook_results
            if results:
                voice = st.radio(
                    "Choose a Voice",
                    list(results.keys()),
                    horizontal=True,
                    key="voice_selection",
                )

                info = results.get(voice, {})
                path = info.get("path") or ""
                err = info.get("error")
                if path and os.path.exists(path):
                    try:
                        with open(path, "rb") as f:
                            audio_bytes = f.read()
                        st.audio(audio_bytes, format="audio/mp3")
                        st.download_button(
                            f"Download {voice}",
                            data=audio_bytes,
                            file_name=os.path.basename(path),
                        )
                    except Exception as e:
                        st.error(f"Failed to load audio file: {e}")
                else:
                    st.error(f"{voice} not available.")
                    if err:
                        st.caption(f"Error: {err}")
                    if st.button(f"Retry {voice}", key=f"retry_{voice}"):
                        with st.spinner(f"Retrying {voice}..."):
                            try:
                                retry_info = synthesize_single_voice(st.session_state.audiobook_text, voice, attempts=2)
                                st.session_state.audiobook_results[voice] = retry_info
                                st.rerun()  # Fixed: replaced experimental_rerun
                            except Exception as e:
                                st.error(f"Retry failed: {e}")
    except ImportError as e:
        st.error(f"Task 2 module not available: {e}")

# --------------------------------------------------
# Page: Storybook Creator
# --------------------------------------------------
elif page == "Storybook Creator":
    try:
        from tasks.task_2_audiobook import preprocess_pdf
        from tasks.task_3_storybook import (
            create_sections,
            create_prompts,
            generate_segmind_image,
            create_storybook,
        )

        st.subheader("Task 3: Storybook Creator")
        
        # Initialize session state with better key management
        if "storybook_sections" not in st.session_state:
            st.session_state.storybook_sections = []
        if "storybook_images" not in st.session_state:
            st.session_state.storybook_images = []
        if "storybook_page" not in st.session_state:
            st.session_state.storybook_page = 0
        if "storybook_audio" not in st.session_state:
            st.session_state.storybook_audio = {}
        if "last_generation_params" not in st.session_state:
            st.session_state.last_generation_params = None

        input_mode = st.radio("Input Type", ["Text", "PDF"], horizontal=True, key="storybook_input_mode")
        
        if input_mode == "Text":
            story_text = st.text_area("Enter story text", height=180, placeholder="Once upon a time...", key="storybook_text")
        else:
            uploaded_pdf = st.file_uploader("Upload a PDF story", type=["pdf"], key="storybook_pdf")
            if uploaded_pdf is not None and st.checkbox("Preview extracted text before generating", value=False, key="preview_checkbox"):
                try:
                    with st.spinner("Extracting text from PDF..."):
                        preview_text = preprocess_pdf(uploaded_pdf)
                    with st.expander("Extracted Text Preview"):
                        st.write(preview_text)
                except Exception as e:
                    st.warning(f"Preview extraction failed: {e}")

        # Separate parameter controls from generation
        st.markdown("### Generation Parameters")
        col_sections, col_words = st.columns(2)
        with col_sections:
            num_sections = st.number_input("Sections", 3, 10, 6, step=1, key="num_sections")
        with col_words:
            per_section_words = st.slider("Words per section", 80, 140, 110, step=5, key="per_section_words")

        # Track current parameters
        current_params = {
            "sections": num_sections,
            "words": per_section_words,
            "input_mode": input_mode
        }

        # Check if parameters changed (but don't reset page yet)
        params_changed = st.session_state.last_generation_params != current_params

        # Only show generate button if content exists
        can_generate = False
        story_text_to_use = ""
        
        if input_mode == "PDF":
            if uploaded_pdf is not None:
                can_generate = True
        else:
            story_text_to_use = st.session_state.get("storybook_text", "")
            if story_text_to_use.strip():
                can_generate = True

        if can_generate:
            if st.button("Generate Storybook", use_container_width=True, key="generate_storybook_btn"):
                # Extract text if PDF mode
                if input_mode == "PDF":
                    try:
                        with st.spinner("Extracting text from PDF..."):
                            story_text_to_use = preprocess_pdf(uploaded_pdf)
                    except Exception as e:
                        st.error(f"Failed to extract text from PDF: {e}")
                        st.stop()

                if not story_text_to_use.strip():
                    st.warning("Please enter some story text.")
                else:
                    # Preserve current page if possible
                    current_page_before_generation = st.session_state.get("storybook_page", 0)
                    
                    with st.spinner("Creating sections & generating images..."):
                        try:
                            sections = create_sections(
                                story_text_to_use,
                                desired_sections=num_sections,
                                per_section_words=int(per_section_words),
                            )
                            if not sections:
                                st.error("Could not create sections from text.")
                                st.stop()
                            
                            prompts = create_prompts(sections)
                            images: List[bytes] = []
                            progress = st.progress(0)
                            for i, pr in enumerate(prompts):
                                try:
                                    img_bytes = generate_segmind_image(pr)
                                    images.append(img_bytes)
                                except Exception as e:
                                    st.warning(f"Failed to generate image {i+1}: {e}")
                                    images.append(b'')
                                progress.progress(int((i + 1) / len(prompts) * 100))
                        except Exception as e:
                            st.error(f"Generation failed: {e}")
                            st.stop()

                        # Update content
                        st.session_state.storybook_sections = sections
                        st.session_state.storybook_images = images
                        st.session_state.last_generation_params = current_params.copy()
                        
                        # Smart page preservation
                        new_total_pages = len(sections)
                        if current_page_before_generation < new_total_pages:
                            # Keep the same page if it's still valid
                            st.session_state.storybook_page = current_page_before_generation
                        else:
                            # Reset to last valid page
                            st.session_state.storybook_page = max(0, new_total_pages - 1)
                        
                        # Clear audio cache as content changed
                        st.session_state.storybook_audio = {}
                        st.success("Story generated! Scroll down to view.")
        else:
            st.info("Please provide story content (text or PDF) to generate storybook.")

        # Display existing storybook (separate from generation)
        sections = st.session_state.get("storybook_sections", [])
        images = st.session_state.get("storybook_images", [])
        
        if sections and images and len(sections) == len(images):
            st.markdown("---")
            st.markdown("### Generated Storybook")
            
            total_pages = len(sections)
            current_page = st.session_state.get("storybook_page", 0)
            
            # Ensure page is within bounds (safety check)
            current_page = max(0, min(current_page, total_pages - 1))
            st.session_state.storybook_page = current_page

            # Display current page
            left, right = st.columns([1, 1])
            with left:
                try:
                    if images[current_page]:
                        st.image(images[current_page], use_container_width=True)
                    else:
                        st.info("Image not available for this page.")
                except Exception as e:
                    st.warning(f"Unable to display image: {e}")
            
            with right:
                st.markdown(f"### Page {current_page + 1} of {total_pages}")
                
                # Show parameter change warning if applicable
                if params_changed and st.session_state.last_generation_params:
                    st.info("⚠️ Parameters have changed. Generate new storybook to apply changes.")
                
                styled_text = f"<div style='font-size:16px; line-height:1.5;'>{sections[current_page]}</div>"
                st.markdown(styled_text, unsafe_allow_html=True)

                # Audio narration
                narrate_key = f"narrate_{current_page}"
                if st.button("🔊 Narrate this page", key=narrate_key):
                    try:
                        def synthesize_audio(text: str) -> bytes:
                            async def _synthesize():
                                tts = edge_tts.Communicate(text, "en-US-JennyNeural")
                                audio_data = bytearray()
                                async for chunk in tts.stream():
                                    if chunk["type"] == "audio":
                                        audio_data.extend(chunk["data"])
                                return bytes(audio_data)
                            return asyncio.run(_synthesize())
                        
                        with st.spinner("Generating audio..."):
                            audio_bytes = synthesize_audio(sections[current_page])
                            st.session_state.storybook_audio[current_page] = audio_bytes
                    except Exception as e:
                        st.error(f"TTS failed: {e}")
                        
                if current_page in st.session_state.storybook_audio:
                    st.audio(st.session_state.storybook_audio[current_page], format="audio/mp3")

            # Navigation controls
            nav1, mid, nav2 = st.columns([1, 2, 1])
            with nav1:
                if st.button("⬅️ Previous", disabled=(current_page <= 0), key="prev_storybook"):
                    st.session_state.storybook_page = max(0, current_page - 1)
                    st.rerun()
            with mid:
                st.markdown(f"<div style='text-align:center; padding-top:8px;'>Page {current_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
            with nav2:
                if st.button("Next ➡️", disabled=(current_page >= total_pages - 1), key="next_storybook"):
                    st.session_state.storybook_page = min(total_pages - 1, current_page + 1)
                    st.rerun()

            # Export options
            st.markdown("---")
            st.markdown("### Export Options")
            exp_col1, exp_col2, exp_col3 = st.columns(3)
            with exp_col1:
                export_font = st.selectbox("Font", ["Helvetica", "Times-Roman", "Courier"], key="export_font")
            with exp_col2:
                export_font_size = st.number_input("Font Size", 10, 18, 12, step=1, key="export_font_size")
            with exp_col3:
                export_layout = st.radio("Layout", ["alternate", "combined"], horizontal=True, key="export_layout")

            if st.button("Generate PDF Export", key="generate_pdf_btn"):
                try:
                    with st.spinner("Creating PDF..."):
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
                            "📄 Download PDF",
                            data=pdf_bytes,
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf",
                            key="download_storybook_pdf",
                        )
                        st.success("PDF generated successfully!")
                except Exception as e:
                    st.error(f"Failed to generate PDF: {e}")
        
        elif sections or images:  # Partial content exists
            st.warning("Storybook generation incomplete. Please regenerate.")
    
    except ImportError as e:
        st.error(f"Task 3 module not available: {e}")
        
# --------------------------------------------------
# Page: Persona Showcase
# --------------------------------------------------
elif page == "Persona Showcase":
    st.subheader("Persona Showcase")
    try:
        df = load_persona_data()
        
        if df is not None and not df.empty:
            total = len(df)
            if "persona_index" not in st.session_state:
                st.session_state.persona_index = 0

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Previous", key="persona_prev"):
                    st.session_state.persona_index = (st.session_state.persona_index - 1) % total
            with col2:
                if st.button("Next", key="persona_next"):
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
            m1.metric("Age", str(age))
            m2.metric("Location", str(location))
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
    except Exception as e:
        st.error(f"Failed to load persona data: {e}")