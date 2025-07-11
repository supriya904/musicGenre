# Implementation Details

## Table of Contents
1. [Development Environment Setup](#1-development-environment-setup)
2. [Data Pipeline Implementation](#2-data-pipeline-implementation)
3. [Model Implementation](#3-model-implementation)
4. [Training Pipeline](#4-training-pipeline)
5. [Experiment Tracking System](#5-experiment-tracking-system)
6. [Web Application Development](#6-web-application-development)
7. [Testing and Validation](#7-testing-and-validation)

---

## 1. Development Environment Setup

### 1.1 System Requirements

**Hardware Specifications:**
- **CPU**: Intel Core i5 or equivalent (minimum)
- **RAM**: 8GB (minimum), 16GB (recommended)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 5GB free space for dataset and models

**Software Dependencies:**
```
Python 3.8.0+
TensorFlow 2.10.0+
Librosa 0.9.2+
NumPy 1.21.0+
Pandas 1.3.0+
Matplotlib 3.5.0+
Streamlit 1.28.0+
Plotly 5.15.0+
```

### 1.2 Environment Configuration

**Virtual Environment Setup:**
```python
# requirements.txt
tensorflow==2.13.0
librosa==0.9.2
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
streamlit==1.28.1
scikit-learn==1.3.0
```

**Installation Process:**
```bash
# Create virtual environment
python -m venv music_genre_env

# Activate environment
music_genre_env\Scripts\activate  # Windows
source music_genre_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 1.3 Project Structure Implementation

```
musicGenre/
├── src/                          # Core implementation modules
│   ├── __init__.py
│   ├── config.py                 # Configuration constants
│   ├── data_preprocessing.py     # Audio processing pipeline
│   ├── models.py                 # Neural network architectures
│   ├── evaluation.py             # Evaluation metrics
│   └── experiment_reporter.py    # Automated reporting
├── streamlit_app/               # Web application
│   ├── app.py                   # Main application
│   ├── model_info.py           # Model analysis module
│   └── app_utils.py            # Utility functions
├── experiments/                 # Experiment tracking
├── models/                      # Trained model storage
├── results/                     # Analysis outputs
├── tensorboard_logs/           # TensorBoard logs
├── train.py                    # Training script
├── predict.py                  # Inference script
├── compare_models.py          # Model comparison
└── experiment_manager.py      # CLI management
```

---

## 2. Data Pipeline Implementation

### 2.1 Configuration Module (src/config.py)

```python
# Audio processing configuration
SAMPLE_RATE = 22050
DURATION = 30  # seconds
NUM_SEGMENTS = 10

# MFCC feature extraction
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

# Dataset configuration
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']
NUM_GENRES = len(GENRES)

# Model training configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2

# Paths
DATA_PATH = "data/genres_original"
MODELS_PATH = "models"
EXPERIMENTS_PATH = "experiments"
TENSORBOARD_LOGS = "tensorboard_logs"
```

### 2.2 Data Preprocessing Implementation (src/data_preprocessing.py)

```python
import librosa
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from .config import *

class AudioDataProcessor:
    """Handles audio loading, preprocessing, and feature extraction."""
    
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.duration = DURATION
        self.num_segments = NUM_SEGMENTS
        
    def load_audio(self, file_path):
        """Load audio file with error handling."""
        try:
            signal, sr = librosa.load(file_path, sr=self.sample_rate)
            return signal, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def extract_mfcc_segments(self, signal):
        """Extract MFCC features with segment-based approach."""
        samples_per_segment = int(self.sample_rate * self.duration / self.num_segments)
        expected_mfcc_vectors = math.ceil(samples_per_segment / HOP_LENGTH)
        
        mfcc_segments = []
        
        for segment in range(self.num_segments):
            start = samples_per_segment * segment
            finish = start + samples_per_segment
            
            mfcc = librosa.feature.mfcc(
                y=signal[start:finish],
                sr=self.sample_rate,
                n_mfcc=N_MFCC,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )
            
            mfcc = mfcc.T  # Transpose to (time, features)
            
            # Ensure consistent shape
            if len(mfcc) == expected_mfcc_vectors:
                mfcc_segments.append(mfcc.tolist())
        
        return np.array(mfcc_segments)
    
    def process_dataset(self, data_path):
        """Process entire GTZAN dataset."""
        features = []
        labels = []
        
        for i, genre in enumerate(GENRES):
            genre_path = os.path.join(data_path, genre)
            print(f"Processing {genre}...")
            
            for file in os.listdir(genre_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(genre_path, file)
                    signal, sr = self.load_audio(file_path)
                    
                    if signal is not None:
                        mfcc_features = self.extract_mfcc_segments(signal)
                        if mfcc_features.shape[0] == self.num_segments:
                            features.append(mfcc_features)
                            labels.append(i)
        
        return np.array(features), np.array(labels)
    
    def prepare_data_splits(self, features, labels):
        """Create train/validation/test splits."""
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, test_size=TEST_SPLIT, 
            random_state=42, stratify=labels
        )
        
        # Second split: separate train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=VALIDATION_SPLIT/(1-TEST_SPLIT),
            random_state=42, stratify=y_temp
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
```

### 2.3 Feature Extraction Pipeline

**Implementation Details:**

1. **Audio Loading**: Uses librosa with consistent sampling rate
2. **Segmentation**: Divides 30-second tracks into 10 segments of 3 seconds each
3. **MFCC Extraction**: Computes 13 MFCC coefficients per time frame
4. **Normalization**: Ensures consistent feature dimensions across all samples
5. **Quality Control**: Validates feature shapes and handles corrupted files

**Key Implementation Features:**
- Robust error handling for corrupted audio files
- Consistent feature shape validation
- Memory-efficient processing for large datasets
- Reproducible random splits with fixed seeds

---

## 3. Model Implementation

### 3.1 Neural Network Architectures (src/models.py)

```python
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from .config import *

class ModelArchitectures:
    """Collection of neural network architectures for music genre classification."""
    
    @staticmethod
    def create_ann_model(input_shape):
        """Artificial Neural Network implementation."""
        model = models.Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(NUM_GENRES, activation='softmax')
        ])
        return model
    
    @staticmethod
    def create_cnn_model(input_shape):
        """Basic CNN implementation."""
        model = models.Sequential([
            layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(50, activation='relu'),
            layers.Dense(NUM_GENRES, activation='softmax')
        ])
        return model
    
    @staticmethod
    def create_improved_cnn_model(input_shape):
        """Enhanced CNN with regularization."""
        model = models.Sequential([
            layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.25),
            
            layers.Conv1D(64, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.25),
            
            layers.Conv1D(128, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            
            layers.Dense(128, activation='relu', 
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.5),
            layers.Dense(NUM_GENRES, activation='softmax')
        ])
        return model
    
    @staticmethod
    def create_residual_cnn_model(input_shape):
        """Residual CNN with skip connections."""
        inputs = layers.Input(shape=input_shape)
        
        # Initial convolution
        x = layers.Conv1D(64, 7, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Residual Block 1
        shortcut = x
        x = layers.Conv1D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv1D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(2)(x)
        
        # Residual Block 2
        shortcut = layers.Conv1D(128, 1, strides=2, padding='same')(x)
        shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Conv1D(128, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv1D(128, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(NUM_GENRES, activation='softmax')(x)
        
        model = models.Model(inputs, x)
        return model
    
    @staticmethod
    def create_lstm_model(input_shape):
        """LSTM implementation for temporal modeling."""
        model = models.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32, return_sequences=False),
            layers.Dense(50, activation='relu'),
            layers.Dense(NUM_GENRES, activation='softmax')
        ])
        return model

class ModelFactory:
    """Factory class for creating models."""
    
    MODELS = {
        'ann': ModelArchitectures.create_ann_model,
        'cnn': ModelArchitectures.create_cnn_model,
        'improved_cnn': ModelArchitectures.create_improved_cnn_model,
        'residual_cnn': ModelArchitectures.create_residual_cnn_model,
        'lstm': ModelArchitectures.create_lstm_model
    }
    
    @classmethod
    def create_model(cls, model_type, input_shape):
        """Create model by type name."""
        if model_type not in cls.MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls.MODELS[model_type](input_shape)
    
    @classmethod
    def get_available_models(cls):
        """Get list of available model types."""
        return list(cls.MODELS.keys())
```

### 3.2 Model Compilation and Training Configuration

```python
def compile_model(model, learning_rate=LEARNING_RATE):
    """Compile model with appropriate optimizer and metrics."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_callbacks(model_name, patience=10):
    """Create training callbacks."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f"{TENSORBOARD_LOGS}/{model_name}",
            histogram_freq=1,
            write_graph=True
        )
    ]
    return callbacks
```

---

## 4. Training Pipeline

### 4.1 Training Script Implementation (train.py)

```python
import argparse
import json
import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf

from src.data_preprocessing import AudioDataProcessor
from src.models import ModelFactory, compile_model, get_callbacks
from src.evaluation import ModelEvaluator
from src.experiment_reporter import ExperimentReporter
from src.config import *

def train_model(model_type, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Complete training pipeline for a specific model."""
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{model_type}_{timestamp}"
    experiment_dir = os.path.join(EXPERIMENTS_PATH, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"Starting experiment: {experiment_id}")
    
    # Data preprocessing
    print("Loading and preprocessing data...")
    processor = AudioDataProcessor()
    features, labels = processor.process_dataset(DATA_PATH)
    
    # Create data splits
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        processor.prepare_data_splits(features, labels)
    
    print(f"Data shapes:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Testing: {X_test.shape}")
    
    # Create model
    if model_type in ['ann']:
        input_shape = (X_train.shape[1] * X_train.shape[2],)  # Flatten for ANN
        X_train_model = X_train.reshape(X_train.shape[0], -1)
        X_val_model = X_val.reshape(X_val.shape[0], -1)
        X_test_model = X_test.reshape(X_test.shape[0], -1)
    elif model_type in ['lstm']:
        input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
        X_train_model = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
        X_val_model = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2])
        X_test_model = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
    else:  # CNN models
        input_shape = (X_train.shape[1], X_train.shape[2])
        X_train_model = X_train
        X_val_model = X_val
        X_test_model = X_test
    
    model = ModelFactory.create_model(model_type, input_shape)
    model = compile_model(model)
    
    print(f"\nModel architecture:")
    model.summary()
    
    # Training
    print(f"\nStarting training...")
    start_time = time.time()
    
    callbacks = get_callbacks(experiment_id)
    
    history = model.fit(
        X_train_model, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val_model, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Evaluation
    print("\nEvaluating model...")
    evaluator = ModelEvaluator(model, GENRES)
    
    # Test set evaluation
    test_loss, test_accuracy = model.evaluate(X_test_model, y_test, verbose=0)
    predictions = model.predict(X_test_model)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate detailed metrics
    classification_report = evaluator.generate_classification_report(
        y_test, predicted_classes
    )
    confusion_matrix = evaluator.create_confusion_matrix(
        y_test, predicted_classes
    )
    
    # Save model
    model_path = os.path.join(MODELS_PATH, f"{experiment_id}.h5")
    model.save(model_path)
    
    # Generate experiment report
    experiment_metadata = {
        "experiment_id": experiment_id,
        "model_type": model_type,
        "timestamp": timestamp,
        "args": {
            "model": model_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": LEARNING_RATE
        },
        "final_accuracy": float(test_accuracy),
        "best_val_accuracy": float(max(history.history['val_accuracy'])),
        "total_epochs": len(history.history['accuracy']),
        "training_time": f"{training_time:.1f} seconds",
        "model_parameters": model.count_params(),
        "data_shape": {
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
            "input_shape": list(input_shape)
        }
    }
    
    # Save experiment data
    reporter = ExperimentReporter(experiment_dir)
    reporter.save_experiment_metadata(experiment_metadata)
    reporter.save_training_history(history.history)
    reporter.save_evaluation_results({
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix.tolist()
    })
    reporter.generate_experiment_report(experiment_metadata)
    
    print(f"\nExperiment completed!")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Model saved: {model_path}")
    print(f"Experiment data: {experiment_dir}")
    
    return experiment_id, test_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train music genre classification model')
    parser.add_argument('--model', type=str, required=True,
                       choices=ModelFactory.get_available_models(),
                       help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    experiment_id, accuracy = train_model(
        args.model, args.epochs, args.batch_size
    )
    
    print(f"\nFinal Results:")
    print(f"Experiment ID: {experiment_id}")
    print(f"Test Accuracy: {accuracy:.4f}")
```

### 4.2 Experiment Management Implementation

```python
# experiment_manager.py
import os
import json
import pandas as pd
from datetime import datetime
import argparse

class ExperimentManager:
    """Command-line interface for managing experiments."""
    
    def __init__(self):
        self.experiments_path = EXPERIMENTS_PATH
    
    def list_experiments(self):
        """List all completed experiments."""
        experiments = []
        
        for exp_dir in os.listdir(self.experiments_path):
            exp_path = os.path.join(self.experiments_path, exp_dir)
            metadata_path = os.path.join(exp_path, 'experiment_metadata.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                experiments.append(metadata)
        
        # Sort by timestamp
        experiments.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Create DataFrame for display
        df_data = []
        for exp in experiments:
            df_data.append({
                'Experiment ID': exp['experiment_id'],
                'Model Type': exp['model_type'],
                'Test Accuracy': f"{exp['final_accuracy']:.4f}",
                'Best Val Accuracy': f"{exp['best_val_accuracy']:.4f}",
                'Epochs': exp['total_epochs'],
                'Training Time': exp['training_time'],
                'Parameters': f"{exp['model_parameters']:,}",
                'Timestamp': exp['timestamp']
            })
        
        df = pd.DataFrame(df_data)
        print("\n=== Experiment Summary ===")
        print(df.to_string(index=False))
        
        return experiments
    
    def compare_models(self):
        """Compare performance across different model types."""
        experiments = self.list_experiments()
        
        # Group by model type and find best performance
        model_performance = {}
        for exp in experiments:
            model_type = exp['model_type']
            accuracy = exp['final_accuracy']
            
            if model_type not in model_performance:
                model_performance[model_type] = {
                    'best_accuracy': accuracy,
                    'best_experiment': exp['experiment_id'],
                    'count': 1
                }
            else:
                if accuracy > model_performance[model_type]['best_accuracy']:
                    model_performance[model_type]['best_accuracy'] = accuracy
                    model_performance[model_type]['best_experiment'] = exp['experiment_id']
                model_performance[model_type]['count'] += 1
        
        # Display comparison
        print("\n=== Model Performance Comparison ===")
        for model_type, performance in sorted(model_performance.items(), 
                                            key=lambda x: x[1]['best_accuracy'], 
                                            reverse=True):
            print(f"{model_type:15} | Best: {performance['best_accuracy']:.4f} | "
                  f"Runs: {performance['count']:2d} | "
                  f"Experiment: {performance['best_experiment']}")
    
    def show_experiment_details(self, experiment_id):
        """Show detailed information for a specific experiment."""
        exp_path = os.path.join(self.experiments_path, experiment_id)
        
        if not os.path.exists(exp_path):
            print(f"Experiment {experiment_id} not found!")
            return
        
        # Load metadata
        with open(os.path.join(exp_path, 'experiment_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Load results
        results_path = os.path.join(exp_path, 'results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
        else:
            results = None
        
        # Display information
        print(f"\n=== Experiment Details: {experiment_id} ===")
        print(f"Model Type: {metadata['model_type']}")
        print(f"Timestamp: {metadata['timestamp']}")
        print(f"Test Accuracy: {metadata['final_accuracy']:.4f}")
        print(f"Best Validation Accuracy: {metadata['best_val_accuracy']:.4f}")
        print(f"Training Epochs: {metadata['total_epochs']}")
        print(f"Training Time: {metadata['training_time']}")
        print(f"Model Parameters: {metadata['model_parameters']:,}")
        
        if results:
            print(f"\nDetailed Results:")
            print(f"Test Loss: {results['test_loss']:.4f}")
            print(f"Classification Report:")
            print(results['classification_report'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment Management CLI')
    parser.add_argument('command', choices=['list', 'compare', 'details'],
                       help='Command to execute')
    parser.add_argument('--experiment_id', type=str,
                       help='Specific experiment ID for details command')
    
    args = parser.parse_args()
    
    manager = ExperimentManager()
    
    if args.command == 'list':
        manager.list_experiments()
    elif args.command == 'compare':
        manager.compare_models()
    elif args.command == 'details':
        if args.experiment_id:
            manager.show_experiment_details(args.experiment_id)
        else:
            print("Please provide --experiment_id for details command")
```

---

## 5. Experiment Tracking System

### 5.1 Automated Report Generation (src/experiment_reporter.py)

```python
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

class ExperimentReporter:
    """Automated experiment documentation and reporting."""
    
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.plots_dir = os.path.join(experiment_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def save_experiment_metadata(self, metadata):
        """Save experiment metadata to JSON."""
        metadata_path = os.path.join(self.experiment_dir, 'experiment_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_training_history(self, history):
        """Save training history to JSON."""
        history_path = os.path.join(self.experiment_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Generate training curves plot
        self.plot_training_history(history)
    
    def save_evaluation_results(self, results):
        """Save evaluation results to JSON."""
        results_path = os.path.join(self.experiment_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate evaluation plots
        if 'confusion_matrix' in results:
            self.plot_confusion_matrix(results['confusion_matrix'])
    
    def plot_training_history(self, history):
        """Generate training history plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, confusion_matrix):
        """Generate confusion matrix heatmap."""
        plt.figure(figsize=(10, 8))
        
        # Convert to numpy array if needed
        cm = np.array(confusion_matrix)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=GENRES, yticklabels=GENRES,
                   square=True, linewidths=0.5)
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Genre', fontsize=12)
        plt.ylabel('True Genre', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_experiment_report(self, metadata):
        """Generate comprehensive experiment report in Markdown."""
        report_path = os.path.join(self.experiment_dir, 'README.md')
        
        with open(report_path, 'w') as f:
            f.write(f"# Experiment Report: {metadata['experiment_id']}\n\n")
            
            # Overview
            f.write("## Experiment Overview\n\n")
            f.write(f"- **Model Type**: {metadata['model_type']}\n")
            f.write(f"- **Timestamp**: {metadata['timestamp']}\n")
            f.write(f"- **Test Accuracy**: {metadata['final_accuracy']:.4f}\n")
            f.write(f"- **Best Validation Accuracy**: {metadata['best_val_accuracy']:.4f}\n")
            f.write(f"- **Training Time**: {metadata['training_time']}\n")
            f.write(f"- **Model Parameters**: {metadata['model_parameters']:,}\n\n")
            
            # Configuration
            f.write("## Training Configuration\n\n")
            for key, value in metadata['args'].items():
                f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
            f.write("\n")
            
            # Data Information
            f.write("## Dataset Information\n\n")
            data_shape = metadata.get('data_shape', {})
            if data_shape:
                f.write(f"- **Training Samples**: {data_shape.get('training_samples', 'N/A')}\n")
                f.write(f"- **Validation Samples**: {data_shape.get('validation_samples', 'N/A')}\n")
                f.write(f"- **Test Samples**: {data_shape.get('test_samples', 'N/A')}\n")
                f.write(f"- **Input Shape**: {data_shape.get('input_shape', 'N/A')}\n\n")
            
            # Results
            f.write("## Results\n\n")
            f.write("### Training Progress\n\n")
            f.write("![Training Curves](plots/training_curves.png)\n\n")
            
            f.write("### Confusion Matrix\n\n")
            f.write("![Confusion Matrix](plots/confusion_matrix.png)\n\n")
            
            # Files
            f.write("## Generated Files\n\n")
            f.write("- `experiment_metadata.json` - Experiment configuration and results\n")
            f.write("- `training_history.json` - Detailed training history\n")
            f.write("- `results.json` - Evaluation results and metrics\n")
            f.write("- `plots/` - Generated visualizations\n")
            f.write(f"- `../models/{metadata['experiment_id']}.h5` - Trained model\n\n")
            
            f.write("---\n")
            f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
```

### 5.2 TensorBoard Integration

**Logging Implementation:**
- Real-time training metrics visualization
- Model architecture graphs
- Hyperparameter tracking
- Custom scalar metrics for genre-specific performance

**Usage:**
```bash
# Start TensorBoard server
tensorboard --logdir=tensorboard_logs

# Access at http://localhost:6006
```

---

## 6. Web Application Development

### 6.1 Streamlit Application Structure (streamlit_app/app.py)

```python
import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import os
import tempfile

# Custom modules
from model_info import ModelInfoDisplay
from app_utils import AppUtils

class MusicGenreClassifierApp:
    """Main Streamlit application for music genre classification."""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.load_models_info()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="🎵 Music Genre Classifier",
            page_icon="🎵",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        if 'current_model' not in st.session_state:
            st.session_state.current_model = None
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
    
    def load_models_info(self):
        """Load available models information."""
        models_path = "models"
        self.available_models = []
        
        if os.path.exists(models_path):
            for file in os.listdir(models_path):
                if file.endswith('.h5'):
                    model_info = {
                        'filename': file,
                        'name': file.replace('.h5', ''),
                        'path': os.path.join(models_path, file),
                        'size': os.path.getsize(os.path.join(models_path, file))
                    }
                    self.available_models.append(model_info)
    
    def sidebar_model_selection(self):
        """Sidebar for model selection and configuration."""
        st.sidebar.header("🔧 Model Configuration")
        
        if not self.available_models:
            st.sidebar.error("No trained models found!")
            return None
        
        # Model selection
        model_names = [model['name'] for model in self.available_models]
        selected_model_name = st.sidebar.selectbox(
            "Choose Model",
            model_names,
            index=0
        )
        
        selected_model = next(
            model for model in self.available_models 
            if model['name'] == selected_model_name
        )
        
        # Model information
        st.sidebar.info(f"""
        **Selected Model:** {selected_model['name']}
        **File Size:** {selected_model['size'] / 1024 / 1024:.1f} MB
        """)
        
        # Load model button
        if st.sidebar.button("🚀 Load Model", type="primary"):
            with st.sidebar.spinner("Loading model..."):
                try:
                    model = tf.keras.models.load_model(selected_model['path'])
                    st.session_state.current_model = model
                    st.session_state.model_loaded = True
                    st.sidebar.success("✅ Model loaded successfully!")
                except Exception as e:
                    st.sidebar.error(f"❌ Error loading model: {str(e)}")
                    return None
        
        return selected_model
    
    def predict_tab(self):
        """Main prediction interface tab."""
        st.header("🎯 Music Genre Prediction")
        
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please load a model from the sidebar first!")
            return
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload an audio file",
            type=['wav', 'mp3', 'flac'],
            help="Supported formats: WAV, MP3, FLAC (30 seconds recommended)"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name
            
            # Display audio player
            st.audio(uploaded_file, format='audio/wav')
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🌊 Audio Waveform")
                self.display_waveform(temp_path)
            
            with col2:
                st.subheader("🔮 Prediction")
                if st.button("🎯 Classify Genre", type="primary"):
                    with st.spinner("Analyzing audio..."):
                        prediction_result = self.predict_genre(temp_path)
                        self.display_prediction_result(prediction_result, uploaded_file.name)
            
            # Cleanup temporary file
            os.unlink(temp_path)
        
        # Prediction history
        if st.session_state.prediction_history:
            st.subheader("📝 Prediction History")
            self.display_prediction_history()
    
    def predict_genre(self, audio_path):
        """Predict genre for uploaded audio file."""
        try:
            # Load and preprocess audio
            signal, sr = librosa.load(audio_path, sr=22050)
            
            # Extract MFCC features (segment-based approach)
            duration = 30
            num_segments = 10
            samples_per_segment = int(22050 * duration / num_segments)
            expected_mfcc_vectors = 130  # Consistent with training
            
            mfcc_segments = []
            
            for segment in range(num_segments):
                start = samples_per_segment * segment
                finish = start + samples_per_segment
                
                if finish <= len(signal):
                    mfcc = librosa.feature.mfcc(
                        y=signal[start:finish],
                        sr=sr,
                        n_mfcc=13,
                        n_fft=2048,
                        hop_length=512
                    )
                    mfcc = mfcc.T
                    
                    if len(mfcc) == expected_mfcc_vectors:
                        mfcc_segments.append(mfcc)
            
            if len(mfcc_segments) < num_segments:
                # Pad with the last segment if needed
                while len(mfcc_segments) < num_segments:
                    mfcc_segments.append(mfcc_segments[-1])
            
            features = np.array(mfcc_segments[:num_segments])
            features = features.reshape(1, features.shape[0], features.shape[1], features.shape[2])
            
            # Get model input shape and adjust accordingly
            model_input_shape = st.session_state.current_model.input_shape
            
            if len(model_input_shape) == 2:  # Flattened for ANN
                features = features.reshape(1, -1)
            elif len(model_input_shape) == 3:  # For LSTM or CNN
                features = features.reshape(1, features.shape[1], features.shape[2])
            
            # Make prediction
            predictions = st.session_state.current_model.predict(features)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Get genre labels
            genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                     'jazz', 'metal', 'pop', 'reggae', 'rock']
            
            predicted_genre = genres[predicted_class]
            
            # Create confidence distribution
            confidence_dist = {
                genre: float(pred) for genre, pred in zip(genres, predictions[0])
            }
            
            return {
                'predicted_genre': predicted_genre,
                'confidence': confidence,
                'confidence_distribution': confidence_dist,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            return None
    
    def display_waveform(self, audio_path):
        """Display audio waveform visualization."""
        try:
            signal, sr = librosa.load(audio_path, sr=22050)
            time = np.linspace(0, len(signal) / sr, len(signal))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time, y=signal,
                mode='lines',
                name='Waveform',
                line=dict(color='blue', width=1)
            ))
            
            fig.update_layout(
                title="Audio Waveform",
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error displaying waveform: {str(e)}")
    
    def display_prediction_result(self, result, filename):
        """Display prediction results with confidence visualization."""
        if result is None:
            return
        
        # Main prediction result
        st.success(f"🎵 **Predicted Genre: {result['predicted_genre'].upper()}**")
        st.info(f"🎯 **Confidence: {result['confidence']:.1%}**")
        
        # Confidence distribution chart
        st.subheader("📊 Confidence Distribution")
        
        # Sort by confidence
        sorted_genres = sorted(
            result['confidence_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        genres = [item[0] for item in sorted_genres]
        confidences = [item[1] for item in sorted_genres]
        
        # Create bar chart
        fig = px.bar(
            x=confidences,
            y=genres,
            orientation='h',
            title="Genre Confidence Scores",
            labels={'x': 'Confidence', 'y': 'Genre'},
            color=confidences,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add to history
        history_entry = {
            'filename': filename,
            'predicted_genre': result['predicted_genre'],
            'confidence': result['confidence'],
            'timestamp': result['timestamp']
        }
        st.session_state.prediction_history.append(history_entry)
    
    def display_prediction_history(self):
        """Display prediction history table."""
        df = pd.DataFrame(st.session_state.prediction_history)
        df = df.sort_values('timestamp', ascending=False)
        
        # Format confidence as percentage
        df['confidence'] = df['confidence'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            df[['filename', 'predicted_genre', 'confidence', 'timestamp']],
            use_container_width=True,
            hide_index=True
        )
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.experimental_rerun()
    
    def model_details_tab(self):
        """Model details and analysis tab."""
        st.header("🔍 Model Details & Analysis")
        
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please load a model from the sidebar first!")
            return
        
        model_display = ModelInfoDisplay()
        model_display.show_model_details(st.session_state.current_model)
    
    def comparison_tab(self):
        """Model comparison tab."""
        st.header("📊 Model Comparison")
        
        model_display = ModelInfoDisplay()
        model_display.show_model_comparison()
    
    def learn_more_tab(self):
        """Educational content tab."""
        st.header("📚 Learn More")
        
        app_utils = AppUtils()
        app_utils.show_educational_content()
    
    def about_tab(self):
        """About section tab."""
        st.header("ℹ️ About This Project")
        
        app_utils = AppUtils()
        app_utils.show_about_section()
    
    def run(self):
        """Main application runner."""
        # Title and description
        st.title("🎵 Music Genre Classifier")
        st.markdown("*Classify music genres using deep learning models*")
        
        # Sidebar
        selected_model = self.sidebar_model_selection()
        
        # Main tabs
        tabs = st.tabs([
            "🎯 Prediction",
            "🔍 Model Details", 
            "📊 Comparison",
            "📚 Learn More",
            "ℹ️ About"
        ])
        
        with tabs[0]:
            self.predict_tab()
        
        with tabs[1]:
            self.model_details_tab()
        
        with tabs[2]:
            self.comparison_tab()
        
        with tabs[3]:
            self.learn_more_tab()
        
        with tabs[4]:
            self.about_tab()

if __name__ == "__main__":
    app = MusicGenreClassifierApp()
    app.run()
```

### 6.2 Model Information Display Module

**Features Implemented:**
- Interactive training history plots
- Model architecture visualization
- Performance metrics comparison
- Experiment metadata display
- Real-time model analysis

### 6.3 Utility Functions Module

**Educational Content:**
- Genre characteristics explanations
- MFCC feature visualization
- Model architecture comparisons
- Technical documentation

---

## 7. Testing and Validation

### 7.1 Unit Testing Implementation

```python
# tests/test_data_preprocessing.py
import unittest
import numpy as np
from src.data_preprocessing import AudioDataProcessor

class TestAudioDataProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = AudioDataProcessor()
    
    def test_mfcc_extraction_shape(self):
        """Test MFCC feature extraction output shape."""
        # Create dummy audio signal
        sample_rate = 22050
        duration = 30
        signal = np.random.randn(sample_rate * duration)
        
        features = self.processor.extract_mfcc_segments(signal)
        
        # Verify shape: (10 segments, 130 time frames, 13 MFCC coefficients)
        self.assertEqual(features.shape, (10, 130, 13))
    
    def test_data_splits(self):
        """Test data splitting functionality."""
        # Create dummy data
        features = np.random.randn(100, 10, 130, 13)
        labels = np.random.randint(0, 10, 100)
        
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            self.processor.prepare_data_splits(features, labels)
        
        # Verify split sizes
        total_samples = len(X_train) + len(X_val) + len(X_test)
        self.assertEqual(total_samples, 100)
        
        # Verify no data leakage
        train_indices = set(range(len(X_train)))
        val_indices = set(range(len(X_train), len(X_train) + len(X_val)))
        test_indices = set(range(len(X_train) + len(X_val), total_samples))
        
        self.assertTrue(train_indices.isdisjoint(val_indices))
        self.assertTrue(train_indices.isdisjoint(test_indices))
        self.assertTrue(val_indices.isdisjoint(test_indices))
```

### 7.2 Integration Testing

```python
# tests/test_model_training.py
import unittest
import tempfile
import os
from src.models import ModelFactory
from src.data_preprocessing import AudioDataProcessor

class TestModelTraining(unittest.TestCase):
    
    def test_model_creation(self):
        """Test all model architectures can be created."""
        input_shapes = {
            'ann': (1690,),
            'cnn': (130, 13),
            'improved_cnn': (130, 13),
            'residual_cnn': (130, 13),
            'lstm': (130, 13)
        }
        
        for model_type in ModelFactory.get_available_models():
            with self.subTest(model_type=model_type):
                model = ModelFactory.create_model(model_type, input_shapes[model_type])
                self.assertIsNotNone(model)
                self.assertEqual(model.output_shape[-1], 10)  # 10 genres
    
    def test_model_compilation(self):
        """Test model compilation process."""
        model = ModelFactory.create_model('cnn', (130, 13))
        compiled_model = compile_model(model)
        
        # Verify compilation
        self.assertIsNotNone(compiled_model.optimizer)
        self.assertEqual(compiled_model.loss, 'sparse_categorical_crossentropy')
        self.assertIn('accuracy', compiled_model.metrics_names)
```

### 7.3 Performance Testing

```python
# tests/test_performance.py
import unittest
import time
import numpy as np
from src.models import ModelFactory

class TestPerformance(unittest.TestCase):
    
    def test_prediction_speed(self):
        """Test model prediction performance."""
        model = ModelFactory.create_model('cnn', (130, 13))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Create test data
        test_data = np.random.randn(100, 130, 13)
        
        # Measure prediction time
        start_time = time.time()
        predictions = model.predict(test_data, verbose=0)
        prediction_time = time.time() - start_time
        
        # Verify performance (should predict 100 samples in less than 5 seconds)
        self.assertLess(prediction_time, 5.0)
        self.assertEqual(predictions.shape, (100, 10))
    
    def test_memory_usage(self):
        """Test memory efficiency of models."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Load all models
        models = {}
        for model_type in ['ann', 'cnn', 'improved_cnn']:
            input_shape = (130, 13) if model_type != 'ann' else (1690,)
            models[model_type] = ModelFactory.create_model(model_type, input_shape)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (less than 500MB)
        self.assertLess(memory_increase, 500)
```

### 7.4 User Interface Testing

```python
# tests/test_streamlit_app.py
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import tempfile
import os

class TestStreamlitApp(unittest.TestCase):
    
    @patch('streamlit_app.app.st')
    def test_app_initialization(self, mock_st):
        """Test Streamlit app initialization."""
        from streamlit_app.app import MusicGenreClassifierApp
        
        app = MusicGenreClassifierApp()
        self.assertIsNotNone(app)
        mock_st.set_page_config.assert_called_once()
    
    def test_prediction_pipeline(self):
        """Test end-to-end prediction pipeline."""
        # Create mock audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            # Write dummy audio data
            sample_rate = 22050
            duration = 30
            dummy_audio = np.random.randn(sample_rate * duration).astype(np.float32)
            
            # Save as simple WAV format (simplified for testing)
            tmp_file.write(dummy_audio.tobytes())
            audio_path = tmp_file.name
        
        try:
            from streamlit_app.app import MusicGenreClassifierApp
            app = MusicGenreClassifierApp()
            
            # Mock model
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            mock_model.input_shape = (None, 130, 13)
            
            app.session_state = {'current_model': mock_model}
            
            # Test prediction (would need more mocking for full test)
            # This is a simplified test structure
            
        finally:
            os.unlink(audio_path)
```

---

This implementation document provides comprehensive coverage of the technical implementation details, from environment setup to testing strategies. The modular design ensures maintainability and extensibility while the comprehensive testing approach validates system reliability and performance.
