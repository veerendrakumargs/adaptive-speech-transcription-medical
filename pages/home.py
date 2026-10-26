"""
Home page for PolyglotTranscribe
Clean, professional landing page with feature overview
"""
import streamlit as st

def render():
    """Render enhanced home page with improved layouts and spacing"""
    
    # Hero Header
    st.markdown('<h1 class="main-header">Welcome to PolyglotTranscribe</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enterprise-Grade Biomedical Speech-to-Text System for Medical Professionals</p>', unsafe_allow_html=True)
    
    # Quick Actions with improved spacing
    st.markdown("### Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🎙️ Start Transcription", use_container_width=True, type="primary"):
            st.success("Navigate to Transcription page using sidebar")
    
    with col2:
        if st.button("📊 View Metrics", use_container_width=True):
            st.success("Navigate to Evaluation page using sidebar")
    
    with col3:
        if st.button("📈 See Analytics", use_container_width=True):
            st.success("Navigate to Visualization page using sidebar")
    
    with col4:
        if st.button("⚙️ Settings", use_container_width=True):
            st.success("Navigate to Settings page using sidebar")

    st.markdown("---")
    
    # System Status with enhanced cards
    st.markdown("### System Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Transcriptions",
            value=len(st.session_state.transcriptions),
            delta="Ready to process" if len(st.session_state.transcriptions) == 0 else None
        )
    
    with col2:
        st.metric(
            label="Active Model",
            value=f"Whisper {st.session_state.model_size.capitalize()}",
            delta="Loaded"
        )
    
    with col3:
        import os
        api_status = "Connected" if os.getenv("OPENAI_API_KEY") else "Not configured"
        st.metric(
            label="API Status",
            value=api_status,
            delta="Translation ready" if os.getenv("OPENAI_API_KEY") else "Setup needed"
        )
    
    with col4:
        st.metric(
            label="Languages",
            value="12+",
            delta="Indian regional"
        )
    
    st.markdown("---")
    
    # Key Features Section with enhanced layout
    st.markdown("### Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎯 Multi-Modal Input
        - **Audio Files**: WAV, MP3, FLAC, M4A
        - **Video Files**: MP4, AVI, MOV (auto extraction)
        - **Real-time Recording**: Direct microphone input
        - **Batch Processing**: Handle multiple files
        
        #### 🤖 Advanced AI Models
        - **Whisper AI**: 5 model sizes (tiny to large)
        - **Word Timestamps**: Precise timing
        - **Confidence Scores**: Reliability metrics
        - **Auto Language Detection**: 90+ languages
        
        #### 🌍 Indian Language Support
        - **12+ Regional Languages**: Complete coverage
        - **Medical Term Preservation**: Keep technical terms
        - **Context-Aware Translation**: GPT-5 powered
        - **High Accuracy**: Optimized for medical content
        """)
    
    with col2:
        st.markdown("""
        #### 🏥 Medical NLP
        - **Biomedical Entity Recognition**: SciSpacy integration
        - **Automatic Extraction**: Diseases, medications, procedures
        - **Clinical Highlighting**: Color-coded medical terms
        - **Anatomical Terms**: Comprehensive coverage
        
        #### 📊 Comprehensive Metrics
        - **WER & CER**: Industry-standard error rates
        - **Accuracy Metrics**: Precision, recall, F1-score
        - **Visual Analytics**: Interactive charts
        - **Performance Tracking**: Historical data
        
        #### 🔊 Audio Processing
        - **Noise Reduction**: Spectral subtraction algorithm
        - **Silence Trimming**: Automatic cleanup
        - **Sample Rate Adjustment**: 8kHz to 48kHz
        - **Audio Normalization**: Consistent quality
        """)
    
    st.markdown("---")
    
    # How It Works with improved visual hierarchy
    st.markdown("### How It Works")
    
    step1, step2, step3, step4 = st.columns(4)
    
    with step1:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 1.5rem; background: linear-gradient(135deg, rgba(15, 114, 255, 0.15) 0%, rgba(124, 58, 237, 0.15) 100%); border-radius: 12px; border: 2px solid #0f72ff;'>
            <h3 style='color: #00d9ff; margin: 0; font-size: 2rem;'>1</h3>
            <h4 style='margin: 0.5rem 0; font-size: 1rem; color: #ffffff;'>Upload</h4>
            <p style='font-size: 0.9rem; margin: 0; color: #d1d5db;'>
                Upload audio, video or record live
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with step2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 1.5rem; background: linear-gradient(135deg, rgba(14, 165, 78, 0.15) 0%, rgba(34, 197, 94, 0.15) 100%); border-radius: 12px; border: 2px solid #0ea54e;'>
            <h3 style='color: #22c55e; margin: 0; font-size: 2rem;'>2</h3>
            <h4 style='margin: 0.5rem 0; font-size: 1rem; color: #ffffff;'>Process</h4>
            <p style='font-size: 0.9rem; margin: 0; color: #d1d5db;'>
                AI transcribes with medical NLP
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with step3:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 1.5rem; background: linear-gradient(135deg, rgba(124, 58, 237, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%); border-radius: 12px; border: 2px solid #7c3aed;'>
            <h3 style='color: #c084fc; margin: 0; font-size: 2rem;'>3</h3>
            <h4 style='margin: 0.5rem 0; font-size: 1rem; color: #ffffff;'>Translate</h4>
            <p style='font-size: 0.9rem; margin: 0; color: #d1d5db;'>
                Convert to Indian regional languages
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with step4:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 1.5rem; background: linear-gradient(135deg, rgba(234, 88, 12, 0.15) 0%, rgba(251, 146, 60, 0.15) 100%); border-radius: 12px; border: 2px solid #ea580c;'>
            <h3 style='color: #fb923c; margin: 0; font-size: 2rem;'>4</h3>
            <h4 style='margin: 0.5rem 0; font-size: 1rem; color: #ffffff;'>Export</h4>
            <p style='font-size: 0.9rem; margin: 0; color: #d1d5db;'>
                Download in TXT, JSON, PDF format
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Supported Languages with improved grid
    st.markdown("### Supported Indian Languages")
    
    lang_col1, lang_col2, lang_col3, lang_col4 = st.columns(4)
    
    with lang_col1:
        st.markdown("""
        **Dravidian Languages**
        - Tamil (தமிழ்)
        - Telugu (తెలుగు)
        - Kannada (ಕನ್ನಡ)
        - Malayalam (മലയാളം)
        """)
    
    with lang_col2:
        st.markdown("""
        **Indo-Aryan Languages**
        - Hindi (हिंदी)
        - Bengali (বাংলা)
        - Marathi (मराठी)
        - Punjabi (ਪੰਜਾਬੀ)
        """)
    
    with lang_col3:
        st.markdown("""
        **Eastern Languages**
        - Odia (ଓଡ଼ିଆ)
        - Assamese (অসমীয়া)
        - Gujarati (ગુજરાતી)
        """)
    
    with lang_col4:
        st.markdown("""
        **Additional**
        - Urdu (اردو)
        - English (English)
        - Plus many more
        """)
    
    st.markdown("---")
    
    # Research & Technology
    with st.expander("📚 Research Foundation & Technology"):
        st.markdown("""
        ### Research Foundation
        **Paper**: "End-to-End Speech Recognition for Medical Conversations"  
        **Authors**: Chung-Cheng Chiu et al. (2018)  
        **Focus**: High-accuracy medical conversation transcription with domain-specific optimization
        
        ### Technology Stack
        - **Speech Recognition**: OpenAI Whisper (PyTorch-based)
        - **Translation**: GPT-5 via OpenAI API
        - **Medical NLP**: SciSpacy with biomedical models
        - **Audio Processing**: Librosa, Soundfile
        - **Deep Learning**: PyTorch, Transformers
        - **Visualization**: Plotly, Matplotlib
        - **Framework**: Streamlit
        """)
    
    # Getting Started
    with st.expander("🎯 Getting Started Guide"):
        st.markdown("""
        ### Step-by-Step Tutorial
        
        1. **Configure API Key** (for translation)
           - Go to Settings → API Keys
           - Add your OpenAI API key
           - Test the connection
        
        2. **Upload Your First File**
           - Navigate to Transcription page
           - Upload an audio or video file
           - Select Whisper model size
        
        3. **Enable Medical Features**
           - Check "Extract Biomedical Entities"
           - Enable "Highlight Clinical Terms"
           - Adjust audio settings if needed
        
        4. **Process & Review**
           - Click "Start Transcription"
           - Wait for processing to complete
           - Review extracted medical terms
        
        5. **Translate (Optional)**
           - Select target Indian language
           - Choose medical term preservation
           - Export final results
        """)
    
    # Call to Action
    st.markdown("---")
    st.markdown("### Ready to Begin?")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Transcription Now", type="primary", use_container_width=True):
            st.success("Navigate to Transcription page using sidebar")
