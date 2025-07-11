"""
Utilities and helper functions for the Streamlit app
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import json

def load_experiment_summary():
    """Load the experiment summary from results directory"""
    parent_dir = Path(__file__).parent.parent
    results_dir = parent_dir / "results"
    summary_file = results_dir / "experiment_summary.md"
    
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def get_genre_info():
    """Get information about music genres"""
    genre_info = {
        "blues": {
            "description": "A music genre characterized by specific chord progressions, call-and-response vocals, and blues scales.",
            "characteristics": ["12-bar blues progression", "Dominant 7th chords", "Blue notes", "Call and response"],
            "artists": ["B.B. King", "Muddy Waters", "Robert Johnson"]
        },
        "classical": {
            "description": "Art music produced in Western musical tradition, typically featuring orchestral instruments.",
            "characteristics": ["Complex harmonies", "Orchestral arrangements", "Formal structures", "Dynamic contrast"],
            "artists": ["Mozart", "Beethoven", "Bach"]
        },
        "country": {
            "description": "A genre rooted in American folk music, often featuring guitars, fiddles, and storytelling lyrics.",
            "characteristics": ["Acoustic guitars", "Storytelling lyrics", "Twangy vocals", "Simple chord progressions"],
            "artists": ["Hank Williams", "Johnny Cash", "Dolly Parton"]
        },
        "disco": {
            "description": "Dance music with a steady four-on-the-floor beat, prominent bass lines, and orchestral arrangements.",
            "characteristics": ["Four-on-the-floor beat", "Prominent bass", "Orchestral strings", "Danceable rhythm"],
            "artists": ["Bee Gees", "ABBA", "Chic"]
        },
        "hiphop": {
            "description": "A cultural movement including rap music, characterized by rhythmic speech over strong beats.",
            "characteristics": ["Rap vocals", "Strong beats", "Sampling", "Rhythmic speech"],
            "artists": ["Grandmaster Flash", "Run-DMC", "Public Enemy"]
        },
        "jazz": {
            "description": "A genre characterized by swing, blue notes, complex chords, and improvisation.",
            "characteristics": ["Improvisation", "Swing rhythm", "Complex chords", "Blue notes"],
            "artists": ["Miles Davis", "John Coltrane", "Duke Ellington"]
        },
        "metal": {
            "description": "A genre characterized by heavy guitar riffs, powerful vocals, and aggressive rhythms.",
            "characteristics": ["Heavy guitar riffs", "Powerful drums", "Aggressive vocals", "Fast tempo"],
            "artists": ["Black Sabbath", "Iron Maiden", "Metallica"]
        },
        "pop": {
            "description": "Popular music designed for mass appeal, often featuring catchy melodies and hooks.",
            "characteristics": ["Catchy melodies", "Verse-chorus structure", "Radio-friendly", "Mass appeal"],
            "artists": ["Michael Jackson", "Madonna", "The Beatles"]
        },
        "reggae": {
            "description": "A genre from Jamaica characterized by rhythmic patterns and socially conscious lyrics.",
            "characteristics": ["Off-beat rhythm", "Bass-heavy", "Rastafarian themes", "Relaxed tempo"],
            "artists": ["Bob Marley", "Jimmy Cliff", "Peter Tosh"]
        },
        "rock": {
            "description": "A genre centered around electric guitars, bass, drums, and strong rhythms.",
            "characteristics": ["Electric guitars", "Strong backbeat", "4/4 time", "Verse-chorus structure"],
            "artists": ["The Rolling Stones", "Led Zeppelin", "The Who"]
        }
    }
    return genre_info

def display_genre_info():
    """Display information about music genres"""
    st.subheader("🎵 Music Genre Information")
    
    genre_info = get_genre_info()
    
    # Genre selector
    selected_genre = st.selectbox(
        "Choose a genre to learn more:",
        list(genre_info.keys()),
        format_func=lambda x: x.title()
    )
    
    if selected_genre:
        info = genre_info[selected_genre]
        
        st.markdown(f"### {selected_genre.title()}")
        st.write(info["description"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Key Characteristics:**")
            for char in info["characteristics"]:
                st.write(f"• {char}")
        
        with col2:
            st.markdown("**Notable Artists:**")
            for artist in info["artists"]:
                st.write(f"• {artist}")

def create_feature_info_section():
    """Create a section explaining MFCC features"""
    st.subheader("🔊 Audio Feature Extraction")
    
    with st.expander("What are MFCC Features?", expanded=False):
        st.markdown("""
        **MFCC (Mel-Frequency Cepstral Coefficients)** are the key features used for music genre classification:
        
        **What they represent:**
        - Capture the shape of the spectral envelope
        - Mimic human auditory perception
        - Compress audio information into meaningful coefficients
        
        **Our Configuration:**
        - **13 MFCC coefficients** per time frame
        - **130 time steps** (covering 30 seconds of audio)
        - **Sample Rate:** 22,050 Hz
        - **Hop Length:** 512 samples (~23ms per frame)
        
        **Why MFCCs work for genre classification:**
        - Different genres have distinct spectral characteristics
        - MFCCs capture timbral information crucial for genre identification
        - Robust to noise and variations in recording quality
        """)
    
    with st.expander("Model Input Shape Explained", expanded=False):
        st.markdown("""
        **Input Tensor Shape: (1, 130, 13)**
        
        - **Batch Size (1):** Processing one audio file at a time
        - **Time Steps (130):** Sequence of 130 time frames
        - **Features (13):** 13 MFCC coefficients per time frame
        
        **This creates a 2D feature map** that neural networks can process to:
        - **CNNs:** Detect local patterns in the spectral features
        - **LSTMs:** Capture temporal dependencies across time
        - **Dense layers:** Learn genre-specific feature combinations
        """)

def app_info_section():
    """Display information about the application"""
    st.subheader("ℹ️ About This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Purpose:**
        This application demonstrates music genre classification using deep learning. Upload any audio file and get instant predictions!
        
        **🔬 Technology Stack:**
        - **Frontend:** Streamlit
        - **ML Framework:** TensorFlow/Keras
        - **Audio Processing:** Librosa
        - **Visualization:** Plotly
        - **Feature Extraction:** MFCC
        """)
    
    with col2:
        st.markdown("""
        **📊 Supported Models:**
        - Artificial Neural Networks (ANN)
        - Convolutional Neural Networks (CNN)
        - Improved CNN with regularization
        - Residual CNN with skip connections
        - Long Short-Term Memory (LSTM)
        
        **🎵 Supported Genres:**
        Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock
        """)
    
    st.markdown("""
    **🚀 How to Use:**
    1. **Select a Model:** Choose from your trained models in the sidebar
    2. **Load the Model:** Click the load button and wait for confirmation
    3. **Upload Audio:** Drag and drop or browse for your music file
    4. **Get Predictions:** Click predict and see the results instantly!
    """)

def display_tips_and_tricks():
    """Display usage tips and troubleshooting"""
    st.subheader("💡 Tips & Tricks")
    
    with st.expander("🎵 Best Practices for Audio Files", expanded=False):
        st.markdown("""
        **File Format Recommendations:**
        - **Best:** WAV files (uncompressed, highest quality)
        - **Good:** MP3 files with high bitrate (320 kbps)
        - **Acceptable:** FLAC, M4A files
        
        **Audio Quality Tips:**
        - Use high-quality recordings for better predictions
        - Avoid heavily compressed or low-bitrate files
        - Ensure audio is clear without heavy distortion
        - Mono or stereo both work fine
        
        **File Size Guidelines:**
        - Optimal: 3-10 MB per file
        - Maximum recommended: 50 MB
        - Longer files are automatically truncated to 30 seconds
        """)
    
    with st.expander("🤖 Model Selection Guide", expanded=False):
        st.markdown("""
        **For Best Accuracy:**
        - Use **Residual CNN** - typically highest performance
        - **Improved CNN** - good balance of speed and accuracy
        
        **For Faster Predictions:**
        - Use **ANN** - fastest inference time
        - **Basic CNN** - good speed/accuracy tradeoff
        
        **For Temporal Analysis:**
        - Use **LSTM** - captures time-based patterns
        - Good for music with strong temporal structure
        """)
    
    with st.expander("🔧 Troubleshooting", expanded=False):
        st.markdown("""
        **Common Issues:**
        
        **"No models found"**
        - Ensure you have trained models in the `models/` directory
        - Check the debug info in the sidebar for path details
        
        **"Prediction failed"**
        - Try a different audio file format (WAV recommended)
        - Ensure the audio file isn't corrupted
        - Check that the model loaded successfully
        
        **"Feature extraction error"**
        - Verify the audio file is valid and not empty
        - Try converting to WAV format first
        - Check file permissions
        """)
