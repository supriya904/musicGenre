# 🎵 Music Genre Classification - Streamlit Web App

A beautiful web interface for predicting music genres using your trained deep learning models!

## Features

### 🎯 **Core Functionality**
- **Audio File Upload**: Support for MP3, WAV, FLAC, M4A formats
- **Real-time Prediction**: Instant genre classification with confidence scores
- **Model Selection**: Choose from any trained model in your models/ directory
- **Interactive Visualizations**: Audio waveforms and confidence bar charts

### 📊 **Visual Features**
- **Audio Waveform Display**: See the shape of your music
- **Confidence Bar Chart**: Visual breakdown of all genre probabilities
- **Top 3 Predictions**: Medal system showing best predictions
- **Prediction History**: Track all your classifications

### 🎨 **User Experience**
- **Modern UI**: Beautiful gradient designs and intuitive layout
- **Responsive Design**: Works on desktop and mobile
- **Real-time Audio Player**: Listen to your uploaded files
- **Progress Indicators**: Visual feedback during processing

## Quick Start

### 1. Install Dependencies
```bash
# Install the updated requirements (from project root)
pip install -r requirements.txt
```

### 2. Ensure You Have Trained Models
Make sure you have at least one trained model in the `models/` directory:
```bash
# Train a model if you haven't already
python train.py --model residual_cnn --epochs 50
```

### 3. Launch the Streamlit App
```bash
# From the project root directory
cd streamlit_app
streamlit run app.py
```

### 4. Open in Browser
The app will automatically open at `http://localhost:8501`

## How to Use

### Step 1: Load a Model
1. In the sidebar, select a trained model from the dropdown
2. Click "🔄 Load Model" to load it
3. Wait for the success confirmation

### Step 2: Upload Audio
1. Click "Choose an audio file..." in the upload section
2. Select any music file (MP3, WAV, FLAC, M4A)
3. The file will be automatically loaded and you'll see:
   - File information
   - Audio player
   - Waveform visualization

### Step 3: Get Predictions
1. Click "🚀 Predict Genre" button
2. Wait for the analysis to complete
3. View your results:
   - **Main prediction** with confidence percentage
   - **Detailed confidence chart** for all genres
   - **Top 3 predictions** with medal rankings

### Step 4: Track History
- All predictions are automatically saved in the history table
- View past predictions with timestamps and model info
- Clear history when needed

## Supported Audio Formats

- **MP3**: Most common format
- **WAV**: Uncompressed audio
- **FLAC**: Lossless compression
- **M4A**: Apple's audio format

## Model Compatibility

The app works with any trained model from your project:
- ANN (Artificial Neural Network)
- CNN (Convolutional Neural Network) 
- Improved CNN
- Residual CNN
- LSTM (Long Short-Term Memory)

## Technical Details

### Audio Processing
- **Feature Extraction**: Uses the same MFCC features as training
- **Preprocessing**: Automatic normalization and formatting
- **Duration**: Processes full audio files (not limited to 30 seconds)

### Performance
- **Fast Inference**: Predictions typically complete in 1-3 seconds
- **Memory Efficient**: Processes files temporarily without storing permanently
- **Error Handling**: Graceful handling of unsupported files or corrupted audio

## Troubleshooting

### Common Issues

**"No trained models found"**
- Make sure you have `.h5` files in the `models/` directory
- Train at least one model using the main training scripts

**"Failed to load model"**
- Check that the model file isn't corrupted
- Ensure the model was saved properly during training

**"Audio processing error"**
- Verify the audio file isn't corrupted
- Try converting to a different format (MP3/WAV work best)
- Check file size (very large files may cause memory issues)

### Performance Tips
- **Use smaller audio files** (under 10MB) for faster processing
- **Close other applications** if experiencing memory issues
- **Use WAV format** for most reliable processing

## File Structure
```
streamlit_app/
├── app.py              # Main Streamlit application
└── README.md           # This documentation

Required in parent directory:
├── models/             # Your trained models (.h5 files)
├── src/               # Project source code
│   ├── config.py      # Configuration and genre labels
│   └── data_preprocessing.py  # Feature extraction
└── requirements.txt   # Updated with Streamlit dependencies
```

## Customization

### Adding New Genres
To support additional genres, update `GENRE_LABELS` in `src/config.py`

### Styling Changes
Modify the CSS in the `st.markdown()` sections of `app.py`

### Feature Modifications
Adjust the feature extraction in `src/data_preprocessing.py`

---

## 🎵 Enjoy Classifying Music! 🎵

Upload any song and watch the AI predict its genre in real-time!
