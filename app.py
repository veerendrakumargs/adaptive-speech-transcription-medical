import streamlit as st
import os
from pathlib import Path
import pytesseract
import shutil

# Failsafe: explicitly point to tesseract if PATH fails
if shutil.which("tesseract") is None:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configure page
st.set_page_config(
    page_title="PolyglotTranscribe - Biomedical Speech-to-Text",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Root variables and base styling */
    :root {
        --primary: #0f72ff;
        --primary-dark: #0a56cc;
        --secondary: #7c3aed;
        --accent: #06b6d4;
        --success: #0ea54e;
        --warning: #ea580c;
        --error: #dc2626;
        --neutral-50: #f9fafb;
        --neutral-100: #f3f4f6;
        --neutral-200: #e5e7eb;
        --neutral-300: #d1d5db;
        --neutral-400: #9ca3af;
        --neutral-600: #4b5563;
        --neutral-700: #374151;
        --neutral-800: #1f2937;
        --neutral-900: #111827;
        --radius-sm: 6px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.1);
        --shadow-lg: 0 10px 15px rgba(0,0,0,0.1);
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Main content area with optimized spacing */
    .main .block-container {
        padding: 2.5rem 3rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Headers with gradient and professional styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(120deg, #0f72ff 0%, #0a56cc 50%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .sub-header {
        font-size: 1.1rem;
        font-weight: 500;
        color: #ffffff;
        margin-bottom: 2.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 2px solid #4b5563;
        line-height: 1.6;
    }
    
    /* Section headers with enhanced visibility */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #4b5563;
        transition: var(--transition);
    }
    
    /* Cards and containers with enhanced visual depth */
    .info-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 1.75rem;
        border-radius: var(--radius-md);
        border: 1px solid #4b5563;
        box-shadow: var(--shadow-sm);
        margin: 1rem 0;
        transition: var(--transition);
        color: #e5e7eb;
    }
    
    .info-card:hover {
        border-color: #7c3aed;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.15);
        transform: translateY(-2px);
    }
    
    .feature-card {
        background: linear-gradient(135deg, #0f72ff 0%, #7c3aed 100%);
        padding: 2rem;
        border-radius: var(--radius-md);
        color: white;
        margin: 1rem 0;
        box-shadow: 0 0 20px rgba(15, 114, 255, 0.2);
        transition: var(--transition);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 0 30px rgba(15, 114, 255, 0.3);
    }
    
    /* Buttons with smooth interactions */
    .stButton > button {
        border-radius: var(--radius-md);
        font-weight: 600;
        transition: var(--transition);
        font-size: 1rem;
        padding: 0.75rem 1.5rem;
        border: none;
        box-shadow: var(--shadow-sm);
        background: linear-gradient(135deg, #0f72ff 0%, #7c3aed 100%);
        color: white;
    }
    
    .stButton > button:hover {
        box-shadow: 0 0 20px rgba(15, 114, 255, 0.3);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Tabs with enhanced styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        border-bottom: 2px solid #4b5563;
        padding: 0 0 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        border-radius: var(--radius-md) var(--radius-md) 0 0;
        color: #9ca3af;
        transition: var(--transition);
        border: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(15, 114, 255, 0.1);
        color: #0f72ff;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(120deg, #0f72ff 0%, #7c3aed 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(15, 114, 255, 0.4);
    }
    
    /* Sidebar with modern gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f2937 0%, #111827 100%);
        border-right: 1px solid #4b5563;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        font-weight: 500;
        padding: 0.75rem 1rem;
        border-radius: var(--radius-md);
        transition: var(--transition);
        margin-bottom: 0.5rem;
        color: #e5e7eb;
    }
    
    [data-testid="stSidebar"] .stRadio > label:hover {
        background-color: rgba(15, 114, 255, 0.15);
        transform: translateX(2px);
        color: #ffffff;
    }
    
    /* File uploader with better visual hierarchy */
    [data-testid="stFileUploader"] {
        border: 2px dashed #0f72ff;
        border-radius: var(--radius-lg);
        padding: 2.5rem;
        background: linear-gradient(135deg, rgba(15, 114, 255, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%);
        transition: var(--transition);
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #7c3aed;
        background: linear-gradient(135deg, rgba(15, 114, 255, 0.15) 0%, rgba(124, 58, 237, 0.15) 100%);
    }
    
    /* Text input and select styling */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: var(--radius-md);
        border: 1.5px solid #4b5563;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: var(--transition);
        background: #1f2937;
        color: #e5e7eb;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #0f72ff;
        box-shadow: 0 0 0 3px rgba(15, 114, 255, 0.2);
    }
    
    /* Checkboxes and radio buttons */
    .stCheckbox > label {
        padding: 0.5rem 0;
        transition: var(--transition);
        color: #e5e7eb;
    }
    
    .stCheckbox > label:hover {
        color: #0f72ff;
    }
    
    /* Expander styling */
    .stExpander {
        border: 1px solid #4b5563;
        border-radius: var(--radius-md);
        transition: var(--transition);
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
    }
    
    .stExpander:hover {
        border-color: #0f72ff;
        box-shadow: 0 0 15px rgba(15, 114, 255, 0.1);
    }
    
    /* Messages and alerts */
    .stAlert {
        border-radius: var(--radius-md);
        border-left: 4px solid;
        padding: 1rem 1.5rem;
        color: #e5e7eb;
    }
    
    /* Markdown and text */
    .stMarkdown {
        margin-bottom: 1rem;
    }
    
    .stMarkdown h3 {
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .stMarkdown h4 {
        color: #e5e7eb;
        font-weight: 600;
    }
    
    .stMarkdown p {
        line-height: 1.6;
        color: #d1d5db;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        border-radius: var(--radius-sm);
        background: linear-gradient(90deg, #0f72ff 0%, #7c3aed 100%);
    }
    
    /* Table styling */
    .stDataframe {
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
        color: #e5e7eb;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1.5rem 1rem;
        }
        
        .main-header {
            font-size: 2rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'transcriptions' not in st.session_state:
    st.session_state.transcriptions = []
if 'current_transcription' not in st.session_state:
    st.session_state.current_transcription = None
if 'model_size' not in st.session_state:
    st.session_state.model_size = 'base'
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = 0
if 'training_active' not in st.session_state:
    st.session_state.training_active = False
if 'audio_settings' not in st.session_state:
    st.session_state.audio_settings = {
        'sample_rate': 16000,
        'noise_threshold': 0.02,
        'enable_noise_reduction': True,
        'trim_silence': True
    }

with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0 2rem 0;'>
            <h2 style='margin: 0; color: #0f72ff;'>🏥 PolyglotTranscribe</h2>
            <p style='margin: 0.5rem 0 0 0; color: #ffffff; font-size: 0.9rem;'>
                Medical Speech-to-Text
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation Menu
    st.markdown("**📍 Navigation**")
    page = st.radio(
        "Select Page",
        ["🏠 Home", "🎙️ Transcription", "📊 Evaluation", 
         "📈 Visualization", "🎯 Training", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Quick Stats Section
    st.markdown("**📊 Quick Stats**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Transcriptions", len(st.session_state.transcriptions), delta=None, delta_color="off")
    with col2:
        st.metric("Model", st.session_state.model_size.upper(), delta=None, delta_color="off")
    
    st.markdown("---")
    
    # Theme Settings
    st.markdown("**🎨 Appearance**")
    theme_toggle = st.toggle("Dark Mode", value=st.session_state.theme == 'dark')
    if theme_toggle:
        st.session_state.theme = 'dark'
    else:
        st.session_state.theme = 'light'
    
    st.markdown("---")
    
    # Technology Info
    with st.expander("ℹ️ Technology Stack"):
        st.markdown("""
        - OpenAI Whisper
        - GPT-5 Translation
        - Streamlit
        - PyTorch
        - Plotly Charts
        """, unsafe_allow_html=False)

try:
    if page == "🏠 Home":
        from pages import home
        home.render()
    elif page == "🎙️ Transcription":
        from pages import transcription
        transcription.render()
    elif page == "📊 Evaluation":
        from pages import evaluation
        evaluation.render()
    elif page == "📈 Visualization":
        from pages import visualization
        visualization.render()
    elif page == "🎯 Training":
        from pages import training
        training.render()
    elif page == "⚙️ Settings":
        from pages import settings
        settings.render()
except ImportError as e:
    st.error(f"Error loading page: {str(e)}")
    st.info("Please ensure all page files are properly configured.")
