# Music Genre Classification

A deep learning project for classifying music genres using various neural network architectures including ANN, CNN, LSTM, and ensemble methods.

## Project Structure

```
musicGenre/
├── src/
│   ├── config.py              # Configuration parameters
│   ├── data_preprocessing.py  # Audio preprocessing utilities
│   ├── models.py              # Neural network models
│   └── evaluation.py          # Model evaluation utilities
├── data/                      # Data directory (created automatically)
├── models/                    # Saved models directory (created automatically)
├── results/                   # Results directory (created automatically)
├── train.py                   # Main training script
├── compare_models.py          # Model comparison script
├── predict.py                 # Inference script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Features

- **Multiple Model Architectures**: ANN, Regularized ANN, CNN, Residual CNN, LSTM
- **Advanced Preprocessing**: MFCC feature extraction with librosa
- **Comprehensive Evaluation**: Confusion matrices, classification reports, training visualizations
- **Model Comparison**: Side-by-side comparison of different architectures
- **Ensemble Methods**: Combine multiple models for improved accuracy
- **Inference Pipeline**: Easy prediction on new audio files
- **Modular Design**: Clean, maintainable code structure

## Installation

1. Clone or download the project
2. Create a virtual environment (recommended):
   ```bash
   python -m venv music_env
   music_env\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup

1. Download the GTZAN dataset or use your own music dataset
2. Update the `DATASET_PATH` in `src/config.py` to point to your dataset directory
3. The dataset should be organized as:
   ```
   genres_original/
   ├── blues/
   ├── classical/
   ├── country/
   ├── disco/
   ├── hiphop/
   ├── jazz/
   ├── metal/
   ├── pop/
   ├── reggae/
   └── rock/
   ```

## Usage

### 1. Data Preprocessing and Training

```bash
# Preprocess data and train a CNN model
python train.py --model_type improved_cnn --preprocess --epochs 50 --save_model

# Train other model types
python train.py --model_type regularized_ann --epochs 40
python train.py --model_type lstm --epochs 30
```

Available model types:
- `ann`: Basic Artificial Neural Network
- `regularized_ann`: ANN with dropout and L2 regularization
- `cnn`: Original CNN from notebook
- `improved_cnn`: Enhanced CNN with batch normalization
- `residual_cnn`: CNN with residual connections
- `lstm`: Long Short-Term Memory network

### 2. Model Comparison

```bash
# Compare all model architectures
python compare_models.py
```

This will:
- Train all model types
- Compare their performance
- Create ensemble predictions
- Analyze difficult samples

### 3. Making Predictions

```bash
# Predict genre for a single file
python predict.py --model models/improved_cnn_20231215_143022.h5 --audio path/to/song.wav

# Predict for multiple files in a directory
python predict.py --model models/improved_cnn_20231215_143022.h5 --audio path/to/music_folder/ --output predictions.json --probabilities

# Include normalization parameters for better accuracy
python predict.py --model models/improved_cnn_20231215_143022.h5 --audio song.wav --normalization results/improved_cnn_20231215_143022_normalization.json
```

## Model Performance

Based on the GTZAN dataset, typical performance metrics:

| Model | Accuracy | Notes |
|-------|----------|-------|
| Basic ANN | ~0.45 | Prone to overfitting |
| Regularized ANN | ~0.55 | Better generalization |
| Original CNN | ~0.65 | Good spatial feature extraction |
| Improved CNN | ~0.75 | Enhanced architecture |
| Residual CNN | ~0.77 | Best single model |
| LSTM | ~0.70 | Good for temporal patterns |
| Ensemble | ~0.80 | Combines multiple models |

## Configuration

Key parameters in `src/config.py`:

```python
# Audio processing
SAMPLE_RATE = 22050
DURATION = 30
N_MFCC = 13
NUM_SEGMENTS = 10

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50
DROPOUT_RATE = 0.3
```

## Advanced Features

### Custom Model Creation

Add new models in `src/models.py`:

```python
@staticmethod
def create_custom_model(input_shape, num_classes=10):
    model = Sequential([
        # Your custom architecture
    ])
    return model
```

### Custom Feature Extraction

Extend `AudioPreprocessor` in `src/data_preprocessing.py`:

```python
def extract_custom_features(self, signal, sr):
    # Extract additional features
    return features
```

### Ensemble Methods

Create custom ensemble strategies in `compare_models.py`:

```python
# Weighted ensemble
weighted_pred = w1*pred1 + w2*pred2 + w3*pred3
```

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Update dataset path in `config.py`
2. **Memory Error**: Reduce batch size or number of segments
3. **Low Accuracy**: Try data augmentation or different architectures
4. **Slow Training**: Use GPU acceleration with CUDA

### Performance Tips

- Use GPU for faster training: `pip install tensorflow-gpu`
- Increase `NUM_SEGMENTS` for more detailed analysis
- Try different audio features (chromagrams, spectrograms)
- Use data augmentation for better generalization

## Contributing

Feel free to contribute by:
- Adding new model architectures
- Implementing data augmentation techniques
- Improving preprocessing pipeline
- Adding support for new audio formats

## Genre Mapping

The project supports 10 music genres:
- 0: Disco
- 1: Metal
- 2: Reggae
- 3: Blues
- 4: Rock
- 5: Classical
- 6: Jazz
- 7: Hip-hop
- 8: Country
- 9: Pop

## Future Enhancements

- [ ] Support for more audio formats
- [ ] Real-time prediction
- [ ] Web interface
- [ ] Transfer learning with pre-trained audio models
- [ ] Multi-label classification
- [ ] Attention mechanisms
- [ ] Audio data augmentation

## License

This project is open source. Feel free to use and modify for your needs.
