"""
Music Genre Classification - Streamlit Web App
Upload an audio file and get instant genre predictions!
"""

import streamlit as st
import os
import sys
import tempfile
import json
import numpy as np
import pandas as pd
import librosa
import warnings
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Add parent directory to Python path to import project modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import custom modules
try:
    from model_info import display_model_details, display_model_comparison, display_architecture_info
    from app_utils import display_genre_info, create_feature_info_section, app_info_section, display_tips_and_tricks, load_experiment_summary
except ImportError as e:
    st.warning(f"Could not import custom modules: {e}. Some features may be limited.")

# Define genre labels (matching the actual model training order)
GENRE_MAPPING = {
    0: "disco",
    1: "metal", 
    2: "reggae",
    3: "blues",
    4: "rock",
    5: "classical",
    6: "jazz",
    7: "hiphop",
    8: "country",
    9: "pop"
}

GENRE_LABELS = [GENRE_MAPPING[i] for i in range(len(GENRE_MAPPING))]

# Audio processing parameters (matching config.py)
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
NUM_SEGMENTS = 10

def extract_features(file_path, num_segments=NUM_SEGMENTS):
    """Extract MFCC features from audio file using the same logic as predict.py"""
    try:
        # Load audio file
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Ensure audio is long enough
        min_length = SAMPLES_PER_TRACK
        if len(signal) < min_length:
            # Pad with zeros if too short
            signal = np.pad(signal, (0, min_length - len(signal)), mode='constant')
        elif len(signal) > min_length:
            # Truncate if too long
            signal = signal[:min_length]
        
        samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
        expected_vects_per_segment = int(samples_per_segment / HOP_LENGTH) + 1
        
        features = []
        
        # Extract features from each segment
        for i in range(num_segments):
            start_sample = samples_per_segment * i
            finish_sample = start_sample + samples_per_segment
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=signal[start_sample:finish_sample],
                sr=sr,
                n_fft=N_FFT,
                n_mfcc=N_MFCC,
                hop_length=HOP_LENGTH
            )
            
            mfcc = mfcc.T  # Transpose to get time x features
            
            # Pad or truncate to expected length
            if len(mfcc) > expected_vects_per_segment:
                mfcc = mfcc[:expected_vects_per_segment]
            elif len(mfcc) < expected_vects_per_segment:
                pad_length = expected_vects_per_segment - len(mfcc)
                mfcc = np.pad(mfcc, ((0, pad_length), (0, 0)), mode='constant')
            
            features.append(mfcc)
        
        return np.array(features)
        
    except Exception as e:
        raise Exception(f"Error extracting features: {str(e)}")

# Remove the problematic import
# try:
#     from src.data_preprocessing import extract_features
#     from src.config import GENRE_LABELS
# except ImportError as e:
#     st.error(f"❌ **Import Error**: Could not import project modules. {str(e)}")
#     st.error("Make sure you're running this app from the correct directory structure.")
#     st.stop()

# Page configuration
st.set_page_config(
    page_title="🎵 Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-score {
        font-size: 2rem;
        font-weight: bold;
        color: #ff6b6b;
    }
    .genre-label {
        font-size: 2.5rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .upload-section {
        border: 2px dashed #1f77b4;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MusicGenrePredictor:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.genre_mapping = GENRE_MAPPING
    
    def load_model(self, model_path):
        """Load a trained model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.model_name = os.path.basename(model_path)
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def predict_genre(self, audio_file):
        """Predict genre from audio file using the same logic as predict.py"""
        if self.model is None:
            raise ValueError("No model loaded")
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_path = tmp_file.name
            
            # Extract features (segments)
            features = extract_features(tmp_path)
            
            # Prepare input shape for model
            if len(self.model.input_shape) == 4:  # CNN model expects (batch, height, width, channels)
                features = features[..., np.newaxis]
            
            # Get predictions for all segments
            segment_predictions = []
            segment_probabilities = []
            
            for segment in features:
                # Add batch dimension
                segment_input = np.expand_dims(segment, axis=0)
                
                # Predict
                pred_proba = self.model.predict(segment_input, verbose=0)[0]
                pred_class = np.argmax(pred_proba)
                
                segment_predictions.append(pred_class)
                segment_probabilities.append(pred_proba)
            
            # Average predictions across segments
            avg_probabilities = np.mean(segment_probabilities, axis=0)
            final_prediction = np.argmax(avg_probabilities)
            confidence = np.max(avg_probabilities)
            
            # Get genre name using correct mapping
            predicted_genre = self.genre_mapping[final_prediction]
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            return {
                'genre': predicted_genre,
                'confidence': float(confidence),
                'probabilities': {
                    self.genre_mapping[i]: float(prob) 
                    for i, prob in enumerate(avg_probabilities)
                },
                'segment_predictions': [int(p) for p in segment_predictions],
                'segment_agreement': float(np.mean(np.array(segment_predictions) == final_prediction))
            }
            
        except Exception as e:
            # Clean up temporary file if it exists
            if 'tmp_path' in locals():
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            raise e

def get_available_models():
    """Get list of available trained models"""
    # Try multiple possible paths for the models directory
    possible_paths = [
        parent_dir / "models",  # ../models from streamlit_app
        Path(__file__).parent.parent / "models",  # Alternative parent reference
        Path("../models"),  # Relative path
        Path("models"),  # Current directory
    ]
    
    models_dir = None
    for path in possible_paths:
        if path.exists() and path.is_dir():
            models_dir = path
            break
    
    if not models_dir:
        st.error(f"Models directory not found. Searched in: {[str(p) for p in possible_paths]}")
        return []
    
    model_files = []
    for model_file in models_dir.glob("*.h5"):
        model_files.append(str(model_file))
    
    if not model_files:
        st.warning(f"No .h5 model files found in {models_dir}")
        st.info("Available files in models directory:")
        for file in models_dir.iterdir():
            st.write(f"- {file.name}")
    
    return sorted(model_files)

def plot_prediction_confidence(probabilities):
    """Create a confidence plot"""
    genres = list(probabilities.keys())
    confidences = list(probabilities.values())
    
    # Sort by confidence
    sorted_data = sorted(zip(genres, confidences), key=lambda x: x[1], reverse=True)
    genres, confidences = zip(*sorted_data)
    
    # Create bar chart
    fig = px.bar(
        x=confidences,
        y=genres,
        orientation='h',
        title="Genre Prediction Confidence",
        labels={'x': 'Confidence Score', 'y': 'Genre'},
        color=confidences,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def plot_audio_waveform(audio_file):
    """Plot audio waveform"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        # Load audio
        y, sr = librosa.load(tmp_path, duration=30)  # Load first 30 seconds
        
        # Create time axis
        time = np.linspace(0, len(y)/sr, len(y))
        
        # Create waveform plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time,
            y=y,
            mode='lines',
            name='Waveform',
            line=dict(color='#1f77b4', width=1)
        ))
        
        fig.update_layout(
            title="Audio Waveform",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            height=300,
            showlegend=False
        )
        
        # Clean up
        os.unlink(tmp_path)
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting waveform: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 Music Genre Classifier</h1>', unsafe_allow_html=True)
    st.markdown("**Upload an audio file and discover its genre using deep learning!**")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Prediction", 
        "🔍 Model Details", 
        "📊 Comparison", 
        "📚 Learn More", 
        "ℹ️ About"
    ])
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = MusicGenrePredictor()
    
    # Sidebar for model selection (shared across tabs)
    with st.sidebar:
        st.header("🤖 Model Configuration")
        
        # Debug information
        with st.expander("🔍 Debug Info", expanded=False):
            st.write(f"**Current file:** {__file__}")
            st.write(f"**Parent dir:** {parent_dir}")
            st.write(f"**Models path:** {parent_dir / 'models'}")
            st.write(f"**Models exists:** {(parent_dir / 'models').exists()}")
        
        # Get available models
        available_models = get_available_models()
        
        if not available_models:
            st.error("No trained models found in the models/ directory!")
            st.error("⚠️ Please train a model first using the main training scripts.")
            return
        
        # Model selection
        selected_model = st.selectbox(
            "Choose a trained model:",
            available_models,
            format_func=lambda x: os.path.basename(x)
        )
        
        # Load model button
        if st.button("🔄 Load Model"):
            with st.spinner("Loading model..."):
                if st.session_state.predictor.load_model(selected_model):
                    st.success(f"✅ Model loaded: {os.path.basename(selected_model)}")
                else:
                    st.error("❌ Failed to load model")
        
        # Model info
        if st.session_state.predictor.model is not None:
            st.info(f"**Current Model:** {st.session_state.predictor.model_name}")
            st.info(f"**Supported Genres:** {', '.join(GENRE_LABELS[:5])}...")
    
    # Tab 1: Main Prediction Interface
    with tab1:
        prediction_interface()
    
    # Tab 2: Model Details
    with tab2:
        if st.session_state.predictor.model is not None:
            model_path = None
            for model in available_models:
                if os.path.basename(model) == st.session_state.predictor.model_name:
                    model_path = model
                    break
            
            if model_path:
                try:
                    display_model_details(model_path)
                except:
                    st.error("Could not load model details. Please ensure model_info module is available.")
                    basic_model_info(model_path)
            else:
                st.warning("Model path not found for detailed analysis.")
        else:
            st.warning("⚠️ Please load a model first to see detailed information.")
    
    # Tab 3: Model Comparison
    with tab3:
        try:
            display_model_comparison()
        except:
            st.error("Could not load model comparison. Please ensure experiment data is available.")
            st.info("Train multiple models to see comparison data here.")
        
        # Architecture information
        try:
            display_architecture_info()
        except:
            basic_architecture_info()
    
    # Tab 4: Learn More
    with tab4:
        try:
            display_genre_info()
            create_feature_info_section()
        except:
            basic_genre_info()
            basic_feature_info()
    
    # Tab 5: About
    with tab5:
        try:
            app_info_section()
            display_tips_and_tricks()
        except:
            basic_about_info()
        
        # Show experiment summary if available
        try:
            summary = load_experiment_summary()
            if summary:
                st.subheader("📋 Experiment Summary Report")
                with st.expander("View Full Summary", expanded=False):
                    st.markdown(summary)
        except:
            pass

def prediction_interface():
    """Main prediction interface"""
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Audio File")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an audio file...",
            type=['mp3', 'wav', 'flac', 'm4a'],
            help="Supported formats: MP3, WAV, FLAC, M4A"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.success(f"✅ **File uploaded:** {uploaded_file.name}")
            st.info(f"**File size:** {uploaded_file.size / 1024:.1f} KB")
            
            # Audio player
            st.audio(uploaded_file, format='audio/wav')
            
            # Show waveform
            st.subheader("🌊 Audio Waveform")
            uploaded_file.seek(0)  # Reset file pointer
            waveform_fig = plot_audio_waveform(uploaded_file)
            if waveform_fig:
                st.plotly_chart(waveform_fig, use_container_width=True)
    
    with col2:
        st.header("🎯 Prediction Results")
        
        if uploaded_file is not None and st.session_state.predictor.model is not None:
            
            # Prediction button
            if st.button("🚀 Predict Genre", type="primary", use_container_width=True):
                
                with st.spinner("🎵 Analyzing audio features..."):
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        result = st.session_state.predictor.predict_genre(uploaded_file)
                        
                        # Display main prediction
                        st.markdown(f"""
                        <div class="prediction-box">
                            <div class="genre-label">{result['genre']}</div>
                            <div class="confidence-score">{result['confidence']:.1%} Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show segment information
                        st.info(f"**Segment Agreement:** {result['segment_agreement']:.1%} ({len([p for p in result['segment_predictions'] if p == np.argmax([result['probabilities'][GENRE_MAPPING[i]] for i in range(len(GENRE_MAPPING))])])}/{len(result['segment_predictions'])} segments agreed)")
                        
                        # Show detailed probabilities
                        st.subheader("📊 Detailed Confidence Scores")
                        confidence_fig = plot_prediction_confidence(result['probabilities'])
                        st.plotly_chart(confidence_fig, use_container_width=True)
                        
                        # Show top 3 predictions
                        st.subheader("🏆 Top 3 Predictions")
                        sorted_probs = sorted(result['probabilities'].items(), 
                                            key=lambda x: x[1], reverse=True)
                        
                        for i, (genre, prob) in enumerate(sorted_probs[:3]):
                            medal = ["🥇", "🥈", "🥉"][i]
                            st.write(f"{medal} **{genre}**: {prob:.1%}")
                        
                        # Save prediction history
                        if 'prediction_history' not in st.session_state:
                            st.session_state.prediction_history = []
                        
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'filename': uploaded_file.name,
                            'predicted_genre': result['genre'],
                            'confidence': result['confidence'],
                            'model': st.session_state.predictor.model_name
                        })
                        
                    except Exception as e:
                        st.error(f"❌ **Prediction failed:** {str(e)}")
        
        elif st.session_state.predictor.model is None:
            st.warning("⚠️ Please load a model first from the sidebar.")
        
        elif uploaded_file is None:
            st.info("📁 Please upload an audio file to get started.")
    
    # Prediction History
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        st.header("📝 Prediction History")
        
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df = history_df.sort_values('timestamp', ascending=False)
        
        # Display as table
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

def basic_model_info(model_path):
    """Basic model info fallback"""
    st.subheader("🔍 Basic Model Information")
    model_name = os.path.basename(model_path)
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model File", model_name)
    with col2:
        st.metric("File Size", f"{model_size:.1f} MB")
    with col3:
        st.metric("Format", "Keras HDF5")

def basic_architecture_info():
    """Basic architecture info fallback"""
    st.subheader("🏗️ Model Architectures")
    st.write("Information about different neural network architectures used in this project.")

def basic_genre_info():
    """Basic genre info fallback"""
    st.subheader("🎵 Music Genres")
    st.write("This system can classify music into 10 different genres:")
    for i, genre in enumerate(GENRE_LABELS, 1):
        st.write(f"{i}. **{genre.title()}**")

def basic_feature_info():
    """Basic feature info fallback"""
    st.subheader("🔊 Audio Features")
    st.write("The system uses MFCC (Mel-Frequency Cepstral Coefficients) features for classification.")

def basic_about_info():
    """Basic about info fallback"""
    st.subheader("ℹ️ About This Application")
    st.write("This is a music genre classification system built with deep learning.")
    st.write("Upload audio files to get instant genre predictions!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🎵 <strong>Music Genre Classification System</strong> 🎵<br>
        Built with Streamlit, TensorFlow, and Librosa<br>
        <em>Upload any audio file and discover its genre instantly!</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
