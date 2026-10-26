"""
Transcription & Translation Page
Supports: Audio, Video, PDF, Images, Real-time Voice, Real-time Video
"""
import streamlit as st
import os
import tempfile
import time
from pathlib import Path
import pdfplumber
from PIL import Image
import speech_recognition as sr
import threading
import queue
import pytesseract
from utils.translator import IndianLanguageTranslator
from utils.whisper_model import WhisperTranscriber

# Initialize translator and whisper model
translator = IndianLanguageTranslator()
# Cache the whisper model to avoid reloading
@st.cache_resource
def get_whisper_model():
    return WhisperTranscriber(model_size='base')

whisper_model = get_whisper_model()

def render():
    st.markdown('<h1 class="main-header">Biomedical Translation Hub</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-modal translation: Text, Voice, Video, and Documents</p>', unsafe_allow_html=True)

    # Initialize session state for real-time features
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'recorded_text' not in st.session_state:
        st.session_state.recorded_text = ""

    # Tabs for different input modes
    tab1, tab2, tab3, tab4 = st.tabs(["📁 File Upload", "🎤 Live Voice", "📹 Live Video", "📝 Text Input"])

    with tab1:
        render_file_upload()

    with tab2:
        render_live_voice()

    with tab3:
        render_live_video()
        
    with tab4:
        render_text_input()

def render_file_upload():
    st.markdown("### Upload Medical Documents or Media")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'jpg', 'jpeg', 'png', 'wav', 'mp3', 'mp4', 'm4a'],
        help="Supports PDF, Images (JPG/PNG), Audio (WAV/MP3), Video (MP4)"
    )

    if uploaded_file:
        file_type = uploaded_file.type
        st.info(f"Processing {file_type} file: {uploaded_file.name}")
        
        extracted_text = ""
        
        # Process based on file type
        if "pdf" in file_type:
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text + "\n"
            st.success("✅ PDF Text Extracted")
            
        elif "image" in file_type:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Real OCR using pytesseract
            try:
                extracted_text = pytesseract.image_to_string(image)
                if not extracted_text.strip():
                    st.warning("No text detected in image.")
                else:
                    st.success("✅ OCR Text Extracted")
            except Exception as e:
                st.error(f"OCR Error: {e}")
                extracted_text = f"Error extracting text from image: {e}"
            
        elif "audio" in file_type or "video" in file_type or "mp4" in uploaded_file.name:
            # Save uploaded file to temp file for Whisper
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            if "video" in file_type:
                st.video(uploaded_file)
            else:
                st.audio(uploaded_file)
                
            with st.spinner("Transcribing with Whisper Model..."):
                try:
                    result = whisper_model.transcribe(tmp_path)
                    extracted_text = result['text']
                    
                    # Save to session state for evaluation
                    st.session_state.transcriptions.append({
                        'text': extracted_text,
                        'confidence': result.get('confidence', 0.85),
                        'audio_features': {'duration': result.get('segments', [{}])[-1].get('end', 0) if result.get('segments') else 0},
                        'timestamp': time.time(),
                        'type': 'audio/video'
                    })
                    
                    st.success("✅ Transcription Complete")
                except Exception as e:
                    st.error(f"Transcription Error: {e}")
                    extracted_text = f"Error transcribing media: {e}"
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

        if extracted_text:
            st.text_area("Extracted Content", extracted_text, height=150)
            render_translation_ui(extracted_text, key_suffix="file")

def render_live_voice():
    st.markdown("### Real-time Medical Voice Translation")
    
    audio_value = st.audio_input("Record voice note")

    if audio_value:
        st.audio(audio_value)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_value.getvalue())
            tmp_path = tmp_file.name
            
        with st.spinner("Transcribing voice..."):
            try:
                result = whisper_model.transcribe(tmp_path)
                text = result['text']
                
                # Save to session state for evaluation
                st.session_state.transcriptions.append({
                    'text': text,
                    'confidence': result.get('confidence', 0.85),
                    'audio_features': {'duration': result.get('segments', [{}])[-1].get('end', 0) if result.get('segments') else 0},
                    'timestamp': time.time(),
                    'type': 'voice'
                })
                
                st.success("✅ Transcription Complete")
                st.info(text)
                render_translation_ui(text, key_suffix="voice")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                 if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

def render_live_video():
    st.markdown("### Live AR Video Translation")
    st.info("Using Computer Vision to detect medical text in real-time video stream.")
    
    # Use Streamlit's camera input
    img_file_buffer = st.camera_input("Scan Medical Report or Label")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        st.success("Image Captured")
        
        with st.spinner("Extracting text..."):
            try:
                detected_text = pytesseract.image_to_string(image)
                if not detected_text.strip():
                     st.warning("No text detected. Please try again with better lighting/focus.")
                else:
                    st.text_area("Detected Text via OCR", detected_text)
                    render_translation_ui(detected_text, key_suffix="video")
            except Exception as e:
                st.error(f"OCR Error: {e}")

def render_text_input():
    st.markdown("### Direct Text Input")
    text = st.text_area("Enter medical text here...", height=150)
    if text:
        render_translation_ui(text, key_suffix="text")


def render_translation_ui(source_text, key_suffix):
    st.markdown("---")
    st.markdown("#### 🌐 Translation")

    languages = translator.get_supported_languages()

    col1, col2 = st.columns(2)

    with col1:
        target_lang = st.selectbox(
            "Target Language",
            options=list(languages.keys()),
            format_func=lambda x: languages[x],
            key=f"lang_{key_suffix}",
        )

        with col2:
            preserve_medical = st.checkbox(
                "Preserve Medical Terms (English)",
                value=True,
                key=f"pres_{key_suffix}",
            )

            if st.button("Translate", key=f"btn_{key_suffix}", type="primary"):
                if not isinstance(source_text, str) or not source_text.strip():
                    st.warning("Please enter some text to translate.")
                    return

                with st.spinner("Translating text..."):
                    try:
                        translation = translator.translate_text(
                            source_text,
                            target_lang,
                            preserve_medical_terms=preserve_medical,
                        )
                    except Exception as e:
                        st.error(f"Translation failed: {e}")
                        return

                    if not isinstance(translation, str) or not translation.strip():
                        st.error("Translation failed. Please try again later.")
                        return

                    if translation.startswith("[Translation failed:") or translation.startswith("[Error:"):
                        st.error(translation)
                        return

                    st.success("Translation generated successfully.")
                    st.markdown(f"**{languages[target_lang]}**")

                    st.text_area(
                        "Translated Text",
                        value=translation,
                        height=150,
                        key=f"result_{key_suffix}",
                    )

                    st.download_button(
                        "Download Translation",
                        data=translation.encode("utf-8"),
                        file_name=f"translation_{target_lang}.txt",
                        mime="text/plain; charset=utf-8",
                        key=f"download_{key_suffix}",
                    )
