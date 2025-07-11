# System Design and Architecture

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Data Pipeline Design](#2-data-pipeline-design)
3. [Neural Network Architectures](#3-neural-network-architectures)
4. [Experiment Management System](#4-experiment-management-system)
5. [User Interface Design](#5-user-interface-design)
6. [Implementation Architecture](#6-implementation-architecture)

---

## 1. System Overview

### 1.1 High-Level Architecture

The music genre classification system is designed as a modular, scalable framework that integrates multiple components for comprehensive model development, evaluation, and deployment.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Music Genre Classification System           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Data Pipeline │  │ Model Training  │  │ Web Interface   │  │
│  │   & Processing  │  │ & Evaluation    │  │ & Deployment    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│           │                     │                     │         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Feature         │  │ Experiment      │  │ Model           │  │
│  │ Extraction      │  │ Tracking        │  │ Serving         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Design Principles

**Modularity**: Each component is independently developed and can be tested in isolation
**Scalability**: System can handle increased data volume and model complexity
**Reproducibility**: All experiments are tracked and can be reproduced
**Usability**: User-friendly interfaces for both researchers and end-users
**Extensibility**: Easy to add new models, features, or evaluation metrics

### 1.3 System Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| Data Pipeline | Audio preprocessing and feature extraction | Librosa, NumPy |
| Model Training | Neural network implementation and training | TensorFlow/Keras |
| Experiment Tracking | Logging, monitoring, and comparison | TensorBoard, Custom |
| Web Interface | User interaction and real-time prediction | Streamlit |
| CLI Tools | Command-line experiment management | Python argparse |

---

## 2. Data Pipeline Design

### 2.1 Data Flow Architecture

```
Audio Files (WAV/MP3) 
         ↓
┌─────────────────────┐
│   Audio Loading     │ ← Librosa
│   - Format handling │
│   - Sampling rate   │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Preprocessing      │
│  - Duration norm.   │
│  - Segmentation     │
└─────────────────────┘
         ↓
┌─────────────────────┐
│ Feature Extraction  │
│  - MFCC calculation │
│  - Normalization    │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Data Formatting    │
│  - Shape alignment  │
│  - Batch preparation│
└─────────────────────┘
         ↓
    Model Training
```

### 2.2 Feature Extraction Pipeline

#### 2.2.1 Audio Preprocessing Specifications

```python
# Audio Processing Parameters
SAMPLE_RATE = 22050 Hz
DURATION = 30 seconds
SAMPLES_PER_TRACK = 661,500 samples
NUM_SEGMENTS = 10 segments per track
```

#### 2.2.2 MFCC Feature Configuration

```python
# MFCC Parameters
N_MFCC = 13 coefficients
N_FFT = 2048 samples
HOP_LENGTH = 512 samples
WINDOW_TYPE = Hann window
```

#### 2.2.3 Segment-Based Processing

```
Audio Track (30s)
│
├── Segment 1 (3s) → MFCC Features (130 x 13)
├── Segment 2 (3s) → MFCC Features (130 x 13)
├── Segment 3 (3s) → MFCC Features (130 x 13)
│   ...
└── Segment 10 (3s) → MFCC Features (130 x 13)
                           ↓
               Final Shape: (10, 130, 13)
```

### 2.3 Data Validation and Quality Control

**Input Validation:**
- Audio format compatibility check
- Duration and sampling rate verification
- File corruption detection

**Feature Quality Assurance:**
- NaN and infinity value detection
- Feature range validation
- Consistent shape verification

---

## 3. Neural Network Architectures

### 3.1 Architecture Design Philosophy

Each model architecture is designed to explore different aspects of audio pattern recognition:

- **ANN**: Baseline performance with simple dense connections
- **CNN**: Spatial pattern recognition in MFCC features
- **Improved CNN**: Enhanced generalization with regularization
- **Residual CNN**: Deep learning with gradient flow optimization
- **LSTM**: Temporal sequence modeling for music structure

### 3.2 Model Architecture Specifications

#### 3.2.1 Artificial Neural Network (ANN)

```
Input Layer (1690 features) - Flattened MFCC
         ↓
Dense Layer (512 neurons, ReLU)
         ↓
Dense Layer (256 neurons, ReLU)
         ↓
Dense Layer (128 neurons, ReLU)
         ↓
Output Layer (10 neurons, Softmax)
```

**Design Characteristics:**
- Simple feedforward architecture
- ReLU activation for hidden layers
- Dropout for basic regularization
- Total Parameters: ~1.2M

#### 3.2.2 Convolutional Neural Network (CNN)

```
Input Shape: (130, 13, 1)
         ↓
Conv1D(32 filters, kernel=3, ReLU)
         ↓
MaxPooling1D(pool_size=2)
         ↓
Conv1D(64 filters, kernel=3, ReLU)
         ↓
MaxPooling1D(pool_size=2)
         ↓
Conv1D(64 filters, kernel=3, ReLU)
         ↓
GlobalAveragePooling1D()
         ↓
Dense(50, ReLU)
         ↓
Dense(10, Softmax)
```

**Design Features:**
- 1D convolutions for temporal pattern detection
- Progressive filter increase (32→64→64)
- Global average pooling for dimensionality reduction
- Total Parameters: ~150K

#### 3.2.3 Improved CNN with Regularization

```
Input Shape: (130, 13, 1)
         ↓
Conv1D(32, kernel=3) + BatchNorm + ReLU
         ↓
MaxPooling1D(2) + Dropout(0.25)
         ↓
Conv1D(64, kernel=3) + BatchNorm + ReLU
         ↓
MaxPooling1D(2) + Dropout(0.25)
         ↓
Conv1D(128, kernel=3) + BatchNorm + ReLU
         ↓
GlobalAveragePooling1D()
         ↓
Dense(128, ReLU) + Dropout(0.5)
         ↓
Dense(10, Softmax)
```

**Enhanced Features:**
- Batch normalization for training stability
- Strategic dropout placement
- Increased model capacity (128 filters)
- L2 regularization on dense layers

#### 3.2.4 Residual CNN with Skip Connections

```
Input Shape: (130, 13, 1)
         ↓
Initial Conv1D(64, kernel=7)
         ↓
┌─ Residual Block 1 ─┐
│  Conv1D(64, k=3)   │
│  BatchNorm + ReLU  │ 
│  Conv1D(64, k=3)   │
│  BatchNorm         │
└─ Add + ReLU ───────┘
         ↓
┌─ Residual Block 2 ─┐
│  Conv1D(128, k=3)  │
│  BatchNorm + ReLU  │
│  Conv1D(128, k=3)  │
│  BatchNorm         │
└─ Add + ReLU ───────┘
         ↓
GlobalAveragePooling1D()
         ↓
Dense(10, Softmax)
```

**Residual Features:**
- Skip connections for gradient flow
- Deeper network without vanishing gradients
- Residual learning for feature refinement
- Efficient parameter usage

#### 3.2.5 Long Short-Term Memory (LSTM)

```
Input Shape: (130, 13)
         ↓
LSTM(64 units, return_sequences=True)
         ↓
Dropout(0.3)
         ↓
LSTM(64 units, return_sequences=True)
         ↓
Dropout(0.3)
         ↓
LSTM(32 units, return_sequences=False)
         ↓
Dense(50, ReLU)
         ↓
Dense(10, Softmax)
```

**Temporal Features:**
- Sequential processing of MFCC frames
- Memory cells for long-term dependencies
- Bidirectional option for complete context
- Gradient clipping for training stability

### 3.3 Model Comparison Matrix

| Architecture | Parameters | Strengths | Best For |
|--------------|------------|-----------|----------|
| ANN | 1.2M | Simple, fast training | Baseline comparison |
| CNN | 150K | Pattern recognition | Local feature detection |
| Improved CNN | 180K | Regularization | Balanced performance |
| Residual CNN | 200K | Deep learning | Complex pattern recognition |
| LSTM | 120K | Temporal modeling | Sequential dependencies |

---

## 4. Experiment Management System

### 4.1 Experiment Tracking Architecture

```
Experiment Execution
         ↓
┌─────────────────────┐
│  Metadata Capture   │
│  - Model config     │
│  - Hyperparameters  │
│  - Training args    │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Training Logging   │
│  - TensorBoard      │
│  - Custom metrics   │
│  - Progress tracking│
└─────────────────────┘
         ↓
┌─────────────────────┐
│ Results Storage     │
│  - Model weights    │
│  - Training history │
│  - Evaluation metrics│
└─────────────────────┘
         ↓
┌─────────────────────┐
│ Report Generation   │
│  - Markdown docs    │
│  - Visualizations   │
│  - Comparison tables│
└─────────────────────┘
```

### 4.2 Experiment Directory Structure

```
experiments/
├── {model_type}_{timestamp}/
│   ├── experiment_metadata.json
│   ├── README.md
│   ├── logs/
│   │   ├── training_history.json
│   │   ├── results.json
│   │   └── tensorboard/
│   ├── plots/
│   │   ├── training_curves.png
│   │   ├── confusion_matrix.png
│   │   └── classification_report.png
│   └── model/
│       └── {model_name}.h5
```

### 4.3 Experiment Metadata Schema

```json
{
  "experiment_id": "residual_cnn_20250709_194503",
  "model_type": "residual_cnn",
  "timestamp": "2025-07-09 19:45:03",
  "args": {
    "model": "residual_cnn",
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001
  },
  "final_accuracy": 0.8430,
  "best_val_accuracy": 0.8520,
  "total_epochs": 45,
  "training_time": "1250.5 seconds",
  "model_parameters": 200000
}
```

### 4.4 TensorBoard Integration

**Metrics Tracked:**
- Training and validation loss
- Training and validation accuracy
- Learning rate scheduling
- Model architecture visualization
- Hyperparameter comparison

**Custom Scalars:**
- Per-genre precision and recall
- Confusion matrix heatmaps
- Feature importance analysis

---

## 5. User Interface Design

### 5.1 Web Application Architecture

```
Streamlit Frontend
         ↓
┌─────────────────────┐
│   Model Manager     │
│   - Load models     │
│   - Model selection │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Audio Processor    │
│  - File upload      │
│  - Feature extract  │
└─────────────────────┘
         ↓
┌─────────────────────┐
│ Prediction Engine   │
│  - Model inference  │
│  - Result formatting│
└─────────────────────┘
         ↓
┌─────────────────────┐
│ Visualization       │
│  - Confidence plots │
│  - Audio waveforms  │
└─────────────────────┘
```

### 5.2 User Interface Components

#### 5.2.1 Multi-Tab Interface Design

```
┌─────────────────────────────────────────────────────────┐
│ 🎵 Music Genre Classifier                               │
├─────────────────────────────────────────────────────────┤
│ [🎯 Prediction] [🔍 Model Details] [📊 Comparison]      │
│ [📚 Learn More] [ℹ️ About]                              │
├─────────────────────────────────────────────────────────┤
│                    Tab Content                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 5.2.2 Prediction Interface Layout

```
┌─── Sidebar ───┐  ┌─── Main Content ──────────────────┐
│ Model Config  │  │ ┌─── Upload ───┐ ┌─── Results ───┐ │
│ - Model List  │  │ │ File Upload  │ │ Prediction    │ │
│ - Load Button │  │ │ Audio Player │ │ Confidence    │ │
│ - Model Info  │  │ │ Waveform     │ │ Top 3 Genres  │ │
│               │  │ └──────────────┘ └───────────────┘ │
│ Debug Info    │  │                                   │
│ - Paths       │  │ ┌─── History ────────────────────┐ │
│ - Status      │  │ │ Prediction History Table      │ │
└───────────────┘  │ └───────────────────────────────┘ │
                   └───────────────────────────────────┘
```

### 5.3 Visualization Components

#### 5.3.1 Confidence Score Display

```
┌─────────────────────────────────────┐
│           METAL                     │
│         84.3% Confidence            │
└─────────────────────────────────────┘

Genre Confidence Breakdown:
Metal     ████████████████████ 84.3%
Rock      ████████ 12.1%
Classical ██ 2.1%
Blues     █ 1.2%
...
```

#### 5.3.2 Audio Waveform Visualization

```
Amplitude
    ↑
0.5 ┤     ╭─╮    ╭─╮
    │    ╱   ╲  ╱   ╲
0.0 ┼────────────────────→ Time (seconds)
    │   ╱     ╲╱     ╲
-0.5┤  ╱               ╲
    0    5    10   15   20   25   30
```

### 5.4 Model Details Interface

```
┌─── Model Information ──────────────────────────────────┐
│ Model: residual_cnn_20250709_194503.h5                │
│ Size: 2.1 MB    Format: Keras HDF5    Accuracy: 84.30%│
├────────────────────────────────────────────────────────┤
│ ┌─ Training History ─┐  ┌─ Performance Metrics ──────┐ │
│ │ Accuracy/Loss Plot │  │ Precision: 0.847           │ │
│ │ Over Epochs        │  │ Recall: 0.843              │ │
│ └────────────────────┘  │ F1-Score: 0.844            │ │
│                         │ Segment Agreement: 89.2%   │ │
│ ┌─ Training Parameters ─└────────────────────────────┘ │
│ │ Batch Size: 32                                     │ │
│ │ Learning Rate: 0.001                               │ │
│ │ Epochs: 45/50                                      │ │
│ │ Training Time: 20 minutes                          │ │
│ └────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

---

## 6. Implementation Architecture

### 6.1 Project Structure

```
musicGenre/
├── src/                          # Core library
│   ├── config.py                 # Configuration constants
│   ├── data_preprocessing.py     # Audio processing
│   ├── models.py                 # Neural network architectures
│   ├── evaluation.py             # Metrics and evaluation
│   └── experiment_reporter.py    # Report generation
├── streamlit_app/               # Web interface
│   ├── app.py                   # Main Streamlit app
│   ├── model_info.py           # Model details module
│   └── app_utils.py            # Utility functions
├── experiments/                 # Experiment tracking
│   └── {experiment_folders}/
├── models/                      # Trained models
│   └── *.h5
├── results/                     # Analysis results
├── tensorboard_logs/           # TensorBoard logs
├── train.py                    # Training script
├── predict.py                  # Inference script
├── compare_models.py          # Model comparison
└── experiment_manager.py      # Experiment CLI
```

### 6.2 Technology Stack

#### 6.2.1 Core Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| Deep Learning | TensorFlow 2.x / Keras | Model implementation |
| Audio Processing | Librosa 0.9+ | Feature extraction |
| Data Science | NumPy, Pandas | Data manipulation |
| Visualization | Matplotlib, Plotly | Plotting and charts |
| Web Interface | Streamlit | User interface |
| Experiment Tracking | TensorBoard | Training monitoring |

#### 6.2.2 Development Tools

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Main programming language |
| Git | Version control |
| JSON | Configuration and metadata |
| Markdown | Documentation and reports |
| CLI | Command-line interfaces |

### 6.3 Deployment Architecture

```
Development Environment
         ↓
┌─────────────────────┐
│  Model Training     │
│  - Local GPU        │
│  - Experiment logs  │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Model Storage      │
│  - .h5 files        │
│  - Metadata JSON    │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Web Application    │
│  - Streamlit server │
│  - Model loading    │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  User Interface     │
│  - Browser access   │
│  - Real-time pred.  │
└─────────────────────┘
```

### 6.4 Performance Considerations

#### 6.4.1 Model Serving Optimization

- **Model Caching**: Loaded models cached in memory
- **Batch Processing**: Support for multiple file processing
- **Asynchronous Processing**: Non-blocking UI operations
- **Memory Management**: Efficient tensor operations

#### 6.4.2 Scalability Design

- **Modular Architecture**: Independent component scaling
- **Configuration Management**: Environment-specific settings
- **Resource Monitoring**: Memory and CPU usage tracking
- **Error Handling**: Graceful failure recovery

### 6.5 Security and Reliability

#### 6.5.1 Input Validation

- File format verification
- Size limit enforcement
- Audio content validation
- Path traversal prevention

#### 6.5.2 Error Handling

- Comprehensive exception catching
- User-friendly error messages
- Logging for debugging
- Graceful degradation

---

This design document provides a comprehensive overview of the system architecture, from data processing to user interface, ensuring a robust and scalable music genre classification system.
