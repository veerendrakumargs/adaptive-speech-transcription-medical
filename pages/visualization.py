"""
Performance Visualization Page - Clean interactive charts
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def render():
    """Render enhanced visualization page with better charts"""
    
    st.markdown('<h1 class="main-header">Performance Visualization</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive charts and analytics for model performance</p>', unsafe_allow_html=True)
    
    # Chart selection with improved layout
    st.markdown("### Select Visualization")
    
    chart_type = st.radio(
        "Choose chart type",
        ["Training Progress", "Model Comparison", "Language Performance", "Statistical Analysis"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if chart_type == "Training Progress":
        render_training_charts()
    elif chart_type == "Model Comparison":
        render_model_comparison()
    elif chart_type == "Language Performance":
        render_language_analysis()
    else:
        render_statistical_analysis()

def render_training_charts():
    """Render enhanced training progress visualizations"""
    st.markdown("### Training Progress")
    
    # Generate sample training data
    epochs = np.arange(1, 51)
    train_loss = 2.5 * np.exp(-0.08 * epochs) + np.random.normal(0, 0.05, len(epochs))
    val_loss = 2.5 * np.exp(-0.07 * epochs) + np.random.normal(0, 0.08, len(epochs))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs, y=train_loss,
        mode='lines',
        name='Training Loss',
        line=dict(color='#3b82f6', width=3),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)',
        hovertemplate='<b>Epoch %{x}</b><br>Training Loss: %{y:.4f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs, y=val_loss,
        mode='lines',
        name='Validation Loss',
        line=dict(color='#ef4444', width=3),
        hovertemplate='<b>Epoch %{x}</b><br>Validation Loss: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Loss Curves Over Epochs",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=500,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        xaxis_gridwidth=1,
        yaxis_gridwidth=1,
        xaxis_gridcolor='#e2e8f0',
        yaxis_gridcolor='#e2e8f0'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Training Loss", f"{train_loss[-1]:.4f}")
    with col2:
        st.metric("Final Validation Loss", f"{val_loss[-1]:.4f}")
    with col3:
        gap = val_loss[-1] - train_loss[-1]
        st.metric("Train-Val Gap", f"{gap:.4f}", delta="Good" if gap < 0.2 else "Overfitting detected")

def render_model_comparison():
    """Render enhanced model comparison charts"""
    st.markdown("### Model Size Comparison")
    
    models_df = pd.DataFrame({
        'Model': ['Tiny', 'Base', 'Small', 'Medium', 'Large'],
        'WER': [0.28, 0.15, 0.12, 0.09, 0.07],
        'Speed (x realtime)': [32, 16, 8, 3, 1.2],
        'Size (MB)': [75, 145, 488, 1540, 2880]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_wer = px.bar(
            models_df,
            x='Model',
            y='WER',
            title="Word Error Rate by Model",
            color='WER',
            color_continuous_scale='RdYlGn_r',
            text='WER'
        )
        
        fig_wer.update_traces(
            texttemplate='%{text:.2%}',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>WER: %{y:.2%}<extra></extra>'
        )
        
        fig_wer.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_showgrid=False,
            yaxis_showgrid=True,
            yaxis_gridwidth=1,
            yaxis_gridcolor='#e2e8f0'
        )
        
        st.plotly_chart(fig_wer, use_container_width=True)
    
    with col2:
        fig_speed = px.bar(
            models_df,
            x='Model',
            y='Speed (x realtime)',
            title="Processing Speed by Model",
            color='Speed (x realtime)',
            color_continuous_scale='Viridis',
            text='Speed (x realtime)'
        )
        
        fig_speed.update_traces(
            texttemplate='%{text:.1f}x',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Speed: %{y:.1f}x realtime<extra></extra>'
        )
        
        fig_speed.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_showgrid=False,
            yaxis_showgrid=True,
            yaxis_gridwidth=1,
            yaxis_gridcolor='#e2e8f0'
        )
        
        st.plotly_chart(fig_speed, use_container_width=True)
    
    # Model specifications
    st.markdown("### Model Specifications")
    st.dataframe(models_df, use_container_width=True, hide_index=True)

def render_language_analysis():
    """Render enhanced language performance analysis"""
    st.markdown("### Language-Wise Performance")
    
    lang_data = pd.DataFrame({
        'Language': ['English', 'Hindi', 'Tamil', 'Telugu', 'Malayalam', 'Kannada'],
        'Accuracy': [0.88, 0.82, 0.78, 0.80, 0.76, 0.79],
        'Samples': [150, 95, 68, 72, 54, 61]
    })
    
    fig = px.bar(
        lang_data,
        x='Language',
        y='Accuracy',
        title="Accuracy by Language",
        color='Accuracy',
        color_continuous_scale='RdYlGn',
        text='Accuracy'
    )
    
    fig.update_traces(
        texttemplate='%{text:.1%}',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.1%}<br>Samples: %{customdata}<extra></extra>',
        customdata=lang_data['Samples']
    )
    
    fig.update_layout(
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_showgrid=False,
        yaxis_showgrid=True,
        yaxis_gridwidth=1,
        yaxis_gridcolor='#e2e8f0',
        coloraxis_colorbar=dict(title="Accuracy")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Language statistics table
    st.markdown("### Language Statistics")
    st.dataframe(lang_data, use_container_width=True, hide_index=True)
    
    # Best and worst performers
    col1, col2 = st.columns(2)
    
    with col1:
        best_idx = lang_data['Accuracy'].idxmax()
        st.metric("Best Performer", f"{lang_data.loc[best_idx, 'Language']}", 
                 f"{lang_data.loc[best_idx, 'Accuracy']:.1%}")
    
    with col2:
        worst_idx = lang_data['Accuracy'].idxmin()
        st.metric("Needs Improvement", f"{lang_data.loc[worst_idx, 'Language']}", 
                 f"{lang_data.loc[worst_idx, 'Accuracy']:.1%}")

def render_statistical_analysis():
    """Render enhanced statistical distributions"""
    st.markdown("### Statistical Analysis")
    
    # WER distribution
    wer_samples = np.random.beta(2, 10, 1000) * 0.5
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=wer_samples,
        nbinsx=40,
        marker_color='#3b82f6',
        name='WER Distribution',
        hovertemplate='<b>WER Range:</b> %{x:.2%}<br><b>Frequency:</b> %{y}<extra></extra>'
    ))
    
    # Add mean and median lines
    mean_wer = np.mean(wer_samples)
    median_wer = np.median(wer_samples)
    
    fig.add_vline(
        x=mean_wer,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_wer:.2%}",
        annotation_position="top right"
    )
    
    fig.add_vline(
        x=median_wer,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Median: {median_wer:.2%}",
        annotation_position="top left"
    )
    
    fig.update_layout(
        title="Word Error Rate Distribution",
        xaxis_title="WER",
        yaxis_title="Frequency",
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        xaxis_gridwidth=1,
        yaxis_gridwidth=1,
        xaxis_gridcolor='#e2e8f0',
        yaxis_gridcolor='#e2e8f0'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics with better layout
    st.markdown("### Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean WER", f"{np.mean(wer_samples):.2%}")
    with col2:
        st.metric("Median WER", f"{np.median(wer_samples):.2%}")
    with col3:
        st.metric("Std Dev", f"{np.std(wer_samples):.2%}")
    with col4:
        st.metric("Min WER", f"{np.min(wer_samples):.2%}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Max WER", f"{np.max(wer_samples):.2%}")
    with col2:
        q1 = np.percentile(wer_samples, 25)
        st.metric("Q1 (25%)", f"{q1:.2%}")
    with col3:
        q3 = np.percentile(wer_samples, 75)
        st.metric("Q3 (75%)", f"{q3:.2%}")
    with col4:
        iqr = q3 - q1
        st.metric("IQR", f"{iqr:.2%}")
