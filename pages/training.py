"""
Model Training Page - Clean interface for training configuration
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time

def render():
    """Render enhanced training page"""
    
    st.markdown('<h1 class="main-header">Model Training</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Configure and monitor model training for medical domain adaptation</p>', unsafe_allow_html=True)
    
    # Training tabs
    tab1, tab2, tab3 = st.tabs(["Configuration", "Training Progress", "Model Management"])
    
    with tab1:
        render_configuration()
    
    with tab2:
        render_training_progress()
    
    with tab3:
        render_model_management()

def render_configuration():
    """Render enhanced training configuration"""
    st.markdown("### Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Settings**")
        
        base_model = st.selectbox(
            "Base Model",
            ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium"],
            index=1,
            help="Choose base model size for transfer learning"
        )
        
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
            value=5e-5,
            format_func=lambda x: f"{x:.0e}",
            help="Higher LR = faster training but may diverge"
        )
        
        batch_size = st.selectbox(
            "Batch Size",
            [4, 8, 16, 32],
            index=1,
            help="Larger batches = faster but need more memory"
        )
        
        num_epochs = st.slider(
            "Number of Epochs",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="More epochs = better accuracy but longer training"
        )
    
    with col2:
        st.markdown("**Optimization**")
        
        optimizer = st.selectbox(
            "Optimizer",
            ["AdamW", "Adam", "SGD"],
            index=0,
            help="AdamW is recommended for modern deep learning"
        )
        
        scheduler = st.selectbox(
            "LR Scheduler",
            ["Linear", "Cosine", "Exponential"],
            index=1,
            help="Learning rate decay strategy"
        )
        
        col_opts1, col_opts2 = st.columns(2)
        
        with col_opts1:
            freeze_encoder = st.checkbox("Freeze Encoder Layers", value=True, help="Only train decoder")
        
        with col_opts2:
            mixed_precision = st.checkbox("Mixed Precision (FP16)", value=True, help="Faster and less memory")
    
    # Save configuration
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("Save Configuration", type="primary", use_container_width=True):
            st.session_state.training_config = {
                'base_model': base_model,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'optimizer': optimizer,
                'scheduler': scheduler
            }
            st.success("Configuration saved!")

def render_training_progress():
    """Render enhanced training progress monitor"""
    st.markdown("### Training Progress")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Start Training", type="primary", use_container_width=True):
            st.session_state.training_active = True
            st.session_state.training_progress = 0
            st.rerun()
    
    with col2:
        if st.button("Pause", use_container_width=True):
            st.session_state.training_active = False
            st.rerun()
    
    with col3:
        if st.button("Resume", use_container_width=True):
            st.session_state.training_active = True
            st.rerun()
    
    with col4:
        if st.button("Stop", use_container_width=True):
            st.session_state.training_active = False
            st.session_state.training_progress = 0
            st.rerun()
    
    st.markdown("---")
    
    # Progress display with improved visualization
    if st.session_state.get('training_progress', 0) > 0:
        progress = st.session_state.training_progress
        
        st.progress(progress / 100, text=f"Training Progress: {progress}%")
        
        # Metrics with improved layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Epoch", f"{(progress // 10) + 1}/10")
        with col2:
            st.metric("Training Loss", f"{2.0 * np.exp(-0.02 * progress):.4f}")
        with col3:
            st.metric("Val Loss", f"{2.0 * np.exp(-0.015 * progress):.4f}")
        with col4:
            remaining = (100 - progress) * 0.5
            st.metric("Est. Time Left", f"{remaining:.1f}m" if remaining > 0 else "Done")
        
        # Live loss chart
        steps = np.arange(0, progress, 2)
        train_losses = 2.0 * np.exp(-0.02 * steps)
        val_losses = 2.0 * np.exp(-0.015 * steps)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=steps,
            y=train_losses,
            mode='lines',
            name='Training Loss',
            line=dict(color='#3b82f6', width=2),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
        
        fig.add_trace(go.Scatter(
            x=steps,
            y=val_losses,
            mode='lines',
            name='Validation Loss',
            line=dict(color='#ef4444', width=2)
        ))
        
        fig.update_layout(
            title="Training Progress",
            xaxis_title="Step",
            yaxis_title="Loss",
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            xaxis_gridwidth=1,
            yaxis_gridwidth=1,
            xaxis_gridcolor='#e2e8f0',
            yaxis_gridcolor='#e2e8f0'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Training log
        with st.expander("Training Log"):
            st.write("""
            [INFO] Starting training...
            [INFO] Epoch 1/10 - Training Loss: 1.94, Val Loss: 2.01
            [INFO] Epoch 2/10 - Training Loss: 1.75, Val Loss: 1.82
            [INFO] Epoch 3/10 - Training Loss: 1.52, Val Loss: 1.61
            ...
            """)
        
        # Auto-increment
        if st.session_state.training_active and progress < 100:
            time.sleep(0.1)
            st.session_state.training_progress += 1
            st.rerun()
    else:
        st.info("Click 'Start Training' to begin model training")

def render_model_management():
    """Render enhanced model management interface"""
    st.markdown("### Saved Models")
    
    models_df = pd.DataFrame({
        'Model Name': ['medical-whisper-base-v1', 'medical-whisper-small-v2', 'medical-whisper-base-v3'],
        'Base': ['whisper-base', 'whisper-small', 'whisper-base'],
        'WER': ['15%', '12%', '13%'],
        'Date': ['2025-11-01', '2025-11-02', '2025-11-03'],
        'Status': ['Ready', 'Ready', 'Latest']
    })
    
    st.dataframe(models_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Model actions
    st.markdown("### Model Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Select Model**")
        selected_model = st.selectbox(
            "Choose Model",
            models_df['Model Name'].tolist(),
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("**Actions**")
        
        col_a1, col_a2, col_a3 = st.columns(3)
        
        with col_a1:
            if st.button("Load", use_container_width=True):
                st.success(f"Loaded {selected_model}")
        
        with col_a2:
            if st.button("Deploy", use_container_width=True):
                st.success(f"Deployed {selected_model}")
        
        with col_a3:
            if st.button("Delete", use_container_width=True):
                st.warning(f"Deleted {selected_model}")
    
    st.markdown("---")
    
    # Model comparison
    st.markdown("### Model Performance Comparison")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=models_df['Model Name'],
        y=models_df['WER'].str.rstrip('%').astype(float),
        marker_color=['#3b82f6', '#667eea', '#764ba2'],
        text=models_df['WER'],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Word Error Rate by Model",
        yaxis_title="WER (%)",
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_showgrid=False,
        yaxis_showgrid=True,
        yaxis_gridwidth=1,
        yaxis_gridcolor='#e2e8f0'
    )
    
    st.plotly_chart(fig, use_container_width=True)
