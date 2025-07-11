# Music Genre Classification Using Deep Learning: A Comparative Study

## Abstract

Music genre classification is a fundamental task in music information retrieval that has significant applications in music recommendation systems, content organization, and audio analysis. This project presents a comprehensive comparative study of different deep learning architectures for automatic music genre classification using the GTZAN dataset. We implemented and evaluated five distinct neural network models: Artificial Neural Network (ANN), Convolutional Neural Network (CNN), Improved CNN with regularization, Residual CNN with skip connections, and Long Short-Term Memory (LSTM) networks.

Our approach utilizes Mel-Frequency Cepstral Coefficients (MFCC) as the primary audio features, extracting 13 MFCC coefficients across 130 time frames from 30-second audio segments. The study demonstrates that enhanced neural network architectures significantly outperform basic models, with our Residual CNN achieving the highest accuracy of 84.30%. We developed a complete machine learning pipeline including data preprocessing, multiple model architectures, comprehensive evaluation metrics, experiment tracking, and a user-friendly web interface for real-time genre prediction.

**Keywords:** Music Genre Classification, Deep Learning, MFCC Features, CNN, LSTM, Audio Processing

---

## 1. Introduction

### 1.1 Background and Motivation

Music genre classification is one of the most important tasks in Music Information Retrieval (MIR) systems. With the exponential growth of digital music content and streaming platforms, the need for automatic music organization and recommendation systems has become crucial. Traditional manual classification is time-consuming and subjective, making automated approaches essential for modern music applications.

The ability to automatically classify music genres has numerous practical applications:
- **Music Streaming Services**: Automatic playlist generation and music recommendation
- **Music Libraries**: Efficient organization and cataloging of large music collections
- **Music Production**: Assisting producers in understanding and creating genre-specific content
- **Music Research**: Analyzing trends and patterns in music evolution

### 1.2 Problem Statement

Manual music genre classification faces several challenges:
- **Scalability**: Human annotation cannot keep pace with the volume of new music releases
- **Subjectivity**: Genre boundaries are often fuzzy, leading to inconsistent classifications
- **Cost**: Manual classification requires significant human resources
- **Consistency**: Different annotators may classify the same music piece differently

These challenges motivate the development of automated systems that can consistently and efficiently classify music genres while maintaining high accuracy.

### 1.3 Research Objectives

The primary objectives of this research are:

1. **Comparative Analysis**: Systematically compare different deep learning architectures for music genre classification
2. **Feature Engineering**: Investigate the effectiveness of MFCC features for genre discrimination
3. **Model Enhancement**: Develop and evaluate improved neural network architectures with advanced techniques
4. **Performance Evaluation**: Establish comprehensive metrics for model comparison and validation
5. **Practical Implementation**: Create a complete system with user interface for real-world application

### 1.4 Scope and Limitations

**Scope:**
- Classification of 10 music genres from the GTZAN dataset
- Implementation of 5 different neural network architectures
- Comprehensive evaluation using multiple performance metrics
- Development of web-based interface for practical usage

**Limitations:**
- Limited to 10 predefined genres from GTZAN dataset
- Audio segments restricted to 30-second duration
- Single feature type (MFCC) for audio representation
- Balanced dataset assumption (100 samples per genre)

---

## 2. Methodology

### 2.1 Overview

Our methodology follows a systematic approach to compare different neural network architectures for music genre classification. We establish a baseline using simple traditional approaches and progressively introduce enhanced architectures to demonstrate improvement in classification performance.

### 2.2 Dataset and Audio Processing

#### 2.2.1 GTZAN Dataset
We utilize the GTZAN dataset, which consists of:
- **10 Music Genres**: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock
- **1000 Audio Files**: 100 samples per genre
- **Duration**: 30 seconds per audio file
- **Format**: WAV files at 22,050 Hz sampling rate

#### 2.2.2 Feature Extraction
We employ Mel-Frequency Cepstral Coefficients (MFCC) as our primary audio features:
- **13 MFCC coefficients** per time frame
- **130 time frames** per audio segment (covering 30 seconds)
- **Segment-based approach**: Each audio file is divided into 10 segments for robust feature representation
- **Final feature shape**: (10 segments, 130 time frames, 13 MFCC coefficients)

### 2.3 Baseline Methodology (Existing Approach)

#### 2.3.1 Simple Artificial Neural Network (ANN)
Our baseline model represents the traditional approach to music genre classification:

**Architecture:**
- **Input Layer**: Flattened MFCC features
- **Hidden Layers**: 2-3 fully connected dense layers
- **Activation**: ReLU activation functions
- **Output Layer**: 10 neurons with softmax activation for genre classification

**Characteristics:**
- Simple feedforward architecture
- Basic dense layer connections
- Minimal regularization
- Standard optimization techniques

This baseline establishes the performance benchmark that our enhanced methodologies aim to improve upon.

### 2.4 Enhanced Methodology (Our Approach)

We propose and implement four enhanced neural network architectures that progressively build upon the baseline model:

#### 2.4.1 Convolutional Neural Network (CNN)

**Innovation:** Instead of treating audio features as flat vectors, we leverage the spatial relationships in MFCC features using convolutional operations.

**Architecture Enhancements:**
- **1D Convolutional Layers**: Extract local patterns from MFCC time-series data
- **Pooling Layers**: Reduce dimensionality while preserving important features
- **Feature Maps**: Multiple filters to capture different audio patterns
- **Hierarchical Learning**: Progressive feature abstraction through multiple conv layers

**Key Improvements over Baseline:**
- Preserves temporal structure of audio features
- Automatic feature learning instead of manual feature engineering
- Better handling of local patterns in audio signals

#### 2.4.2 Improved CNN with Regularization

**Innovation:** Address overfitting issues and improve generalization through advanced regularization techniques.

**Architecture Enhancements:**
- **Batch Normalization**: Stabilizes training and accelerates convergence
- **Dropout Layers**: Prevents overfitting by randomly deactivating neurons
- **L2 Regularization**: Weight decay to reduce model complexity
- **Optimized Architecture**: Carefully tuned layer dimensions and parameters

**Key Improvements over Basic CNN:**
- Better generalization to unseen data
- Reduced overfitting
- More stable training process
- Improved validation performance

#### 2.4.3 Residual CNN with Skip Connections

**Innovation:** Implement residual learning to enable deeper networks and better gradient flow.

**Architecture Enhancements:**
- **Skip Connections**: Direct paths for gradient flow through the network
- **Residual Blocks**: Learn residual mappings instead of direct mappings
- **Deeper Architecture**: More layers without vanishing gradient problems
- **Identity Mapping**: Preserves information flow through the network

**Key Improvements over Improved CNN:**
- Enables training of deeper networks
- Solves vanishing gradient problem
- Better feature representation through residual learning
- Highest classification accuracy achieved

#### 2.4.4 Long Short-Term Memory (LSTM)

**Innovation:** Capture long-term temporal dependencies in music audio that other architectures may miss.

**Architecture Enhancements:**
- **Memory Cells**: Maintain information over long sequences
- **Gating Mechanisms**: Control information flow (forget, input, output gates)
- **Bidirectional Processing**: Analyze audio sequences in both directions
- **Temporal Modeling**: Explicitly model time-dependent patterns in music

**Key Improvements over CNN Approaches:**
- Better modeling of temporal dependencies
- Capture long-term patterns in music structure
- Explicit sequence modeling capabilities
- Complementary approach to spatial pattern detection

### 2.5 Model Training and Evaluation Strategy

#### 2.5.1 Training Configuration
- **Train/Validation/Test Split**: 60%/20%/20%
- **Batch Size**: 32 samples
- **Learning Rate**: Adaptive learning rate with decay
- **Epochs**: Maximum 50 with early stopping
- **Optimization**: Adam optimizer with momentum

#### 2.5.2 Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision, Recall, F1-Score**: Per-genre performance analysis
- **Confusion Matrix**: Detailed error analysis
- **Training History**: Loss and accuracy progression

#### 2.5.3 Experimental Design
Each model is trained and evaluated using:
- **Consistent Data Splits**: Same train/validation/test sets for fair comparison
- **Multiple Runs**: Statistical significance testing
- **Hyperparameter Optimization**: Grid search for optimal parameters
- **Cross-Validation**: Robust performance estimation

### 2.6 Implementation Framework

**Technologies Used:**
- **Deep Learning**: TensorFlow/Keras for model implementation
- **Audio Processing**: Librosa for feature extraction
- **Data Analysis**: NumPy, Pandas for data manipulation
- **Visualization**: Matplotlib, Plotly for result visualization
- **Web Interface**: Streamlit for user interaction
- **Experiment Tracking**: Custom logging and TensorBoard integration

This methodology provides a systematic framework for comparing different neural network architectures while progressively demonstrating the benefits of enhanced deep learning techniques for music genre classification.

---

*This document represents Part 1 of the comprehensive project report. The methodology section establishes our approach to systematically improve upon baseline techniques through enhanced neural network architectures.*
