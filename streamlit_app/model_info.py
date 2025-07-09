"""
Model information and details module for Streamlit app
"""

import streamlit as st
import os
import json
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def load_experiment_metadata(model_path):
    """Load experiment metadata for a model"""
    model_name = os.path.basename(model_path).replace('.h5', '')
    
    # Try to find corresponding experiment directory
    parent_dir = Path(model_path).parent.parent
    experiments_dir = parent_dir / "experiments"
    
    if not experiments_dir.exists():
        return None
    
    # Look for experiment folder that matches the model name
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir() and model_name in exp_dir.name:
            metadata_file = exp_dir / "experiment_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    return json.load(f), exp_dir
    
    return None, None

def load_training_history(exp_dir):
    """Load training history if available"""
    if not exp_dir:
        return None
    
    history_file = exp_dir / "logs" / "training_history.json"
    if history_file.exists():
        with open(history_file, 'r') as f:
            return json.load(f)
    return None

def load_experiment_report(exp_dir):
    """Load experiment markdown report if available"""
    if not exp_dir:
        return None
    
    report_file = exp_dir / "README.md"
    if report_file.exists():
        with open(report_file, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def plot_training_history(history):
    """Create training history plots"""
    if not history:
        return None, None
    
    epochs = list(range(1, len(history['accuracy']) + 1))
    
    # Accuracy plot
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=epochs, y=history['accuracy'],
        mode='lines+markers', name='Training Accuracy',
        line=dict(color='#1f77b4', width=3)
    ))
    if 'val_accuracy' in history:
        fig_acc.add_trace(go.Scatter(
            x=epochs, y=history['val_accuracy'],
            mode='lines+markers', name='Validation Accuracy',
            line=dict(color='#ff7f0e', width=3)
        ))
    
    fig_acc.update_layout(
        title="Model Accuracy Over Time",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        height=400,
        hovermode='x unified'
    )
    
    # Loss plot
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=epochs, y=history['loss'],
        mode='lines+markers', name='Training Loss',
        line=dict(color='#d62728', width=3)
    ))
    if 'val_loss' in history:
        fig_loss.add_trace(go.Scatter(
            x=epochs, y=history['val_loss'],
            mode='lines+markers', name='Validation Loss',
            line=dict(color='#ff9896', width=3)
        ))
    
    fig_loss.update_layout(
        title="Model Loss Over Time",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=400,
        hovermode='x unified'
    )
    
    return fig_acc, fig_loss

def display_model_details(model_path):
    """Display comprehensive model details"""
    
    st.subheader("🔍 Model Details")
    
    # Basic model info
    model_name = os.path.basename(model_path)
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model File", model_name)
    with col2:
        st.metric("File Size", f"{model_size:.1f} MB")
    with col3:
        st.metric("Format", "Keras HDF5")
    
    # Load experiment metadata
    metadata, exp_dir = load_experiment_metadata(model_path)
    
    if metadata:
        st.subheader("📊 Experiment Information")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Accuracy", f"{metadata['final_accuracy']*100:.2f}%")
        with col2:
            st.metric("Best Val Accuracy", f"{metadata['best_val_accuracy']*100:.2f}%")
        with col3:
            st.metric("Total Epochs", metadata['total_epochs'])
        with col4:
            st.metric("Model Type", metadata['model_type'].upper())
        
        # Training parameters
        with st.expander("🛠️ Training Parameters", expanded=False):
            params_df = pd.DataFrame([
                {"Parameter": "Batch Size", "Value": metadata['args'].get('batch_size', 'N/A')},
                {"Parameter": "Learning Rate", "Value": metadata['args'].get('learning_rate', 'N/A')},
                {"Parameter": "Epochs", "Value": metadata['args'].get('epochs', 'N/A')},
                {"Parameter": "Model Type", "Value": metadata['model_type']},
                {"Parameter": "Timestamp", "Value": metadata['timestamp']},
            ])
            st.dataframe(params_df, use_container_width=True, hide_index=True)
        
        # Training history plots
        history = load_training_history(exp_dir)
        if history:
            st.subheader("📈 Training History")
            
            fig_acc, fig_loss = plot_training_history(history)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_acc, use_container_width=True)
            with col2:
                st.plotly_chart(fig_loss, use_container_width=True)
        
        # Experiment report
        report = load_experiment_report(exp_dir)
        if report:
            st.subheader("📋 Experiment Report")
            with st.expander("View Full Report", expanded=False):
                st.markdown(report)
    
    else:
        st.warning("⚠️ No experiment metadata found for this model. This model might have been trained outside the experiment tracking system.")

def display_model_comparison():
    """Display comparison of all available models"""
    
    st.subheader("🏆 Model Comparison")
    
    # Get all experiments
    parent_dir = Path(__file__).parent.parent
    experiments_dir = parent_dir / "experiments"
    
    if not experiments_dir.exists():
        st.error("No experiments directory found.")
        return
    
    experiments = []
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir():
            metadata_file = exp_dir / "experiment_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                experiments.append(metadata)
    
    if not experiments:
        st.error("No experiment data found.")
        return
    
    # Create comparison DataFrame
    comparison_data = []
    for exp in experiments:
        comparison_data.append({
            'Model': exp['model_type'].upper(),
            'Experiment ID': exp['experiment_id'],
            'Final Accuracy': f"{exp['final_accuracy']*100:.2f}%",
            'Val Accuracy': f"{exp['best_val_accuracy']*100:.2f}%",
            'Epochs': exp['total_epochs'],
            'Date': exp['timestamp'][:10]
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Final Accuracy', ascending=False)
    
    # Display table
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Performance visualization
    fig = px.bar(
        df, 
        x='Model', 
        y=[float(acc.rstrip('%')) for acc in df['Final Accuracy']],
        title="Model Performance Comparison",
        labels={'y': 'Accuracy (%)', 'x': 'Model Type'},
        color=[float(acc.rstrip('%')) for acc in df['Final Accuracy']],
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def display_architecture_info():
    """Display information about different model architectures"""
    
    st.subheader("🏗️ Model Architectures")
    
    architectures = {
        "ANN (Artificial Neural Network)": {
            "description": "Basic feedforward neural network with dense layers",
            "strengths": ["Simple architecture", "Fast training", "Good baseline"],
            "use_case": "Basic classification tasks, establishing baseline performance"
        },
        "CNN (Convolutional Neural Network)": {
            "description": "Uses 1D convolutions to detect patterns in MFCC features",
            "strengths": ["Pattern recognition", "Feature extraction", "Translation invariant"],
            "use_case": "Audio feature pattern detection, spectral analysis"
        },
        "Improved CNN": {
            "description": "Enhanced CNN with batch normalization and dropout",
            "strengths": ["Better generalization", "Reduced overfitting", "Stable training"],
            "use_case": "Improved audio classification with regularization"
        },
        "Residual CNN": {
            "description": "CNN with skip connections for deeper networks",
            "strengths": ["Deeper networks", "Gradient flow", "Higher accuracy"],
            "use_case": "Complex audio pattern recognition, best performance"
        },
        "LSTM (Long Short-Term Memory)": {
            "description": "Recurrent network that captures temporal dependencies",
            "strengths": ["Sequential patterns", "Memory", "Temporal modeling"],
            "use_case": "Time-series audio analysis, temporal feature learning"
        }
    }
    
    for arch_name, info in architectures.items():
        with st.expander(f"📐 {arch_name}", expanded=False):
            st.write(f"**Description:** {info['description']}")
            st.write("**Key Strengths:**")
            for strength in info['strengths']:
                st.write(f"• {strength}")
            st.write(f"**Best Use Case:** {info['use_case']}")
