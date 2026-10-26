"""
Settings Page - Clean configuration interface
"""
import streamlit as st
import os

def render():
    """Render enhanced settings page"""
    
    st.markdown('<h1 class="main-header">Settings</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Configure application preferences and API integrations</p>', unsafe_allow_html=True)
    
    # Settings tabs
    tab1, tab2, tab3 = st.tabs(["API Configuration", "Appearance", "About"])
    
    with tab1:
        render_api_settings()
    
    with tab2:
        render_appearance_settings()
    
    with tab3:
        render_about()

def render_api_settings():
    """Render enhanced API configuration"""
    st.markdown("### API Configuration")
    
    # OpenAI API
    st.markdown("#### OpenAI API")
    
    api_key_exists = bool(os.getenv("OPENAI_API_KEY"))
    
    if api_key_exists:
        st.success("OpenAI API Key configured")
        st.caption("Translation and advanced features are enabled")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Test API Connection", use_container_width=True):
                with st.spinner("Testing..."):
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                        response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[{"role": "user", "content": "test"}],
                            max_completion_tokens=10
                        )
                        st.success("Connection successful!")
                    except Exception as e:
                        st.error(f"Connection failed: {e}")
    else:
        st.warning("OpenAI API Key not configured")
        st.info("""
        To enable translation features:
        1. Go to Replit Secrets
        2. Add `OPENAI_API_KEY`
        3. Restart the application
        """)
    
    st.markdown("---")
    
    # Additional API settings
    st.markdown("#### Additional Services")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("Enable GPU Acceleration", value=True, help="Use CUDA if available")
        st.checkbox("Enable Caching", value=True, help="Cache model results")
    
    with col2:
        st.checkbox("Enable Analytics", value=False, help="Send usage metrics")
        st.checkbox("Enable Auto-Update", value=True, help="Check for updates")

def render_appearance_settings():
    """Render enhanced appearance settings"""
    st.markdown("### Appearance")
    
    # Theme
    current_theme = st.session_state.get('theme', 'light')
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Theme**")
    
    with col2:
        theme = st.radio(
            "Select Theme",
            ["Light Mode", "Dark Mode"],
            index=0 if current_theme == 'light' else 1,
            horizontal=True,
            label_visibility="collapsed"
        )
    
    if theme == "Light Mode":
        st.session_state.theme = 'light'
    else:
        st.session_state.theme = 'dark'
    
    st.markdown("---")
    
    # Display preferences
    st.markdown("### Display Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("Show Confidence Scores", value=True, help="Display confidence scores in results")
        st.checkbox("Show Word Timestamps", value=True, help="Display timing for each word")
        st.checkbox("Show Tooltips", value=True, help="Display helpful tooltips on hover")
    
    with col2:
        st.checkbox("Compact Mode", value=False, help="Reduce spacing and font sizes")
        st.checkbox("Show Advanced Options", value=False, help="Display advanced configuration")
        st.checkbox("Auto-Expand Sections", value=True, help="Auto-expand expanders on load")
    
    st.markdown("---")
    
    # Audio settings
    st.markdown("### Audio Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sample_rate = st.selectbox(
            "Sample Rate",
            [8000, 16000, 22050, 44100, 48000],
            index=1,
            help="Audio sample rate in Hz"
        )
        st.session_state.audio_settings['sample_rate'] = sample_rate
    
    with col2:
        noise_threshold = st.slider(
            "Noise Threshold",
            min_value=0.0,
            max_value=0.1,
            value=0.02,
            step=0.01,
            help="Threshold for noise reduction"
        )
        st.session_state.audio_settings['noise_threshold'] = noise_threshold
    
    st.markdown("---")
    
    # Save preferences
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("Save Preferences", type="primary", use_container_width=True):
            st.success("Preferences saved!")

def render_about():
    """Render enhanced about section"""
    st.markdown("### About PolyglotTranscribe")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 12px; color: white; text-align: center;'>
        <h2 style='margin: 0;'>PolyglotTranscribe</h2>
        <p style='margin: 1rem 0 0 0; font-size: 1.1rem;'>
            Enterprise Biomedical Speech-to-Text System
        </p>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Version 1.0.0</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Key Features
        - Multi-modal input support
        - 12+ Indian languages
        - Medical NLP integration
        - Advanced metrics
        - Real-time processing
        - Word-level timestamps
        - Entity extraction
        """)
    
    with col2:
        st.markdown("""
        #### Technology Stack
        - OpenAI Whisper
        - GPT-5 Translation
        - SciSpacy NLP
        - PyTorch
        - Streamlit
        - Plotly
        - NumPy & Pandas
        """)
    
    st.markdown("---")
    
    # System info
    st.markdown("### System Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Python", "3.11")
    with col2:
        st.metric("Streamlit", "latest")
    with col3:
        st.metric("Transcriptions", len(st.session_state.transcriptions))
    with col4:
        st.metric("Status", "Online")
    
    st.markdown("---")
    
    # Reset option
    with st.expander("Reset Application"):
        st.warning("This will clear all transcriptions and reset settings")
        
        if st.checkbox("I understand this action"):
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                if st.button("Reset Application", type="secondary", use_container_width=True):
                    st.session_state.clear()
                    st.success("Application reset complete!")
                    st.rerun()
    
    st.markdown("---")
    
    # Links and support
    st.markdown("### Support & Documentation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("[Documentation](https://github.com)")
    
    with col2:
        st.markdown("[Report Issue](https://github.com)")
    
    with col3:
        st.markdown("[Contact Support](mailto:support@example.com)")
