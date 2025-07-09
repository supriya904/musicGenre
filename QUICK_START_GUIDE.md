# Music Genre Classification - Quick Start Guide

Welcome! This guide will walk you through using the music genre classification system step by step.

## 🚀 Prerequisites

1. **Python 3.8+** installed on your system
2. **Git** (optional, for cloning)
3. **10GB+ free disk space** for dataset and models

## 📋 Step 1: Environment Setup

### Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv music_env

# Activate environment
# On Windows:
music_env\Scripts\activate
# On macOS/Linux:
source music_env/bin/activate
```

### Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

## 📁 Step 2: Dataset Verification

Check if you have the dataset in the correct location:
```bash
# Your data folder should contain:
# data/
# ├── genres_original/     (audio files)
# ├── features_30_sec.csv  (pre-extracted features)
# └── features_3_sec.csv   (pre-extracted features)
```

If you don't have the dataset, download the GTZAN dataset and place it in the `data/` folder.

## 🎯 Step 3: First Training Run

### Option A: Quick Start (Using Pre-extracted Features)
```bash
# Train your first model - this is fast!
python train.py --model_type improved_cnn --epochs 20 --save_model
```

### Option B: Full Pipeline (Process raw audio)
```bash
# Process audio files and train (takes longer but more comprehensive)
python train.py --model_type improved_cnn --preprocess --epochs 50 --save_model
```

**Expected Output:**
```
Starting training run: improved_cnn_20250708_143022
Model type: improved_cnn
Epochs: 50
Batch size: 32
Experiment directory: experiments/improved_cnn_20250708_143022
--------------------------------------------------
[Training progress will show here]
--------------------------------------------------
🎉 Training completed successfully!
📊 Final test accuracy: 0.7661
📁 Experiment folder: experiments/improved_cnn_20250708_143022
📋 Detailed report: experiments/improved_cnn_20250708_143022/README.md
📈 TensorBoard logs: experiments/improved_cnn_20250708_143022/logs/
```

## 📊 Step 4: View Your Results

### View Experiment Report
```bash
# Open the markdown report in any text editor or markdown viewer
# Location: experiments/improved_cnn_20250708_143022/README.md
```

### Launch TensorBoard (Interactive Visualization)
```bash
# Start TensorBoard server
tensorboard --logdir=experiments/improved_cnn_20250708_143022/logs/

# Open your browser and go to: http://localhost:6006
```

## 🔬 Step 5: Experiment with Different Models

Try different architectures to see which performs best:

### Basic Neural Network
```bash
python train.py --model_type ann --epochs 30 --save_model
```

### Regularized Neural Network (Prevents Overfitting)
```bash
python train.py --model_type regularized_ann --epochs 40 --save_model
```

### Original CNN Architecture
```bash
python train.py --model_type cnn --epochs 30 --save_model
```

### Residual CNN (Advanced Architecture)
```bash
python train.py --model_type residual_cnn --epochs 50 --save_model
```

### LSTM (For Sequential Patterns)
```bash
python train.py --model_type lstm --epochs 40 --save_model
```

## 📈 Step 6: Compare Your Models

### List All Experiments
```bash
python experiment_manager.py --list
```

### Compare Performance
```bash
python experiment_manager.py --compare
```

### Generate Summary Report
```bash
python experiment_manager.py --summary
```

### Get Specific Experiment Details
```bash
python experiment_manager.py --details improved_cnn_20250708_143022
```

## 🎵 Step 7: Make Predictions on New Audio

### Single File Prediction
```bash
python predict.py --model models/improved_cnn_20250708_143022.h5 --audio path/to/your/song.wav
```

### Batch Prediction with Probabilities
```bash
python predict.py --model models/improved_cnn_20250708_143022.h5 --audio music_folder/ --output predictions.json --probabilities
```

## 🏆 Step 8: Advanced Usage

### Compare All Models at Once
```bash
python compare_models.py
```
This will:
- Train all model types
- Compare their performance
- Create ensemble predictions
- Generate comprehensive analysis

### Custom Training Parameters
```bash
# Custom epochs and batch size
python train.py --model_type improved_cnn --epochs 100 --batch_size 64 --save_model

# Train without saving model (for quick testing)
python train.py --model_type lstm --epochs 10
```

## 📚 Understanding Your Results

### What Files Are Created

After training, you'll find:

```
experiments/improved_cnn_20250708_143022/
├── README.md                    # 📋 Detailed experiment report
├── experiment_metadata.json     # 🔧 Configuration and summary
├── plots/
│   ├── training_history.png     # 📈 Training curves
│   ├── confusion_matrix.png     # 🎯 Classification results
│   └── model_architecture.png   # 🏗️ Model structure
└── logs/
    ├── training_history.json    # 📊 Raw training data
    ├── results.json             # 🎯 Performance metrics
    ├── normalization.json       # ⚙️ Data preprocessing params
    └── [tensorboard files]      # 📈 Interactive visualizations

models/
└── improved_cnn_20250708_143022.h5  # 💾 Trained model
```

### Performance Interpretation

- **Accuracy > 70%**: Good performance for music genre classification
- **Accuracy > 75%**: Very good performance
- **Accuracy > 80%**: Excellent performance

### Genre-Specific Performance
- **Classical & Metal**: Usually easiest to classify
- **Rock vs Country**: Often confused with each other
- **Jazz**: Can be challenging due to variety

## 🔧 Troubleshooting

### Common Issues

**Error: Dataset path not found**
```bash
# Check your data folder structure
ls data/
# Should show: genres_original/ features_30_sec.csv features_3_sec.csv
```

**Error: Memory issues**
```bash
# Reduce batch size
python train.py --model_type improved_cnn --batch_size 16 --epochs 30
```

**Error: TensorBoard not opening**
```bash
# Make sure you're using the correct path
tensorboard --logdir=experiments/[your_experiment_name]/logs/
```

**Low accuracy results**
```bash
# Try different models or more epochs
python train.py --model_type residual_cnn --epochs 100 --save_model
```

## 🎯 Recommended Workflow for New Users

1. **Start Simple**: Train improved_cnn for 20 epochs
2. **View Results**: Check the markdown report and TensorBoard
3. **Try Different Models**: Experiment with lstm, residual_cnn
4. **Compare**: Use experiment_manager.py to compare results
5. **Best Model**: Train your best-performing architecture for more epochs
6. **Predict**: Use your best model to classify new songs

## 📞 Next Steps

Once comfortable with the basics:
- Experiment with different hyperparameters
- Try ensemble methods (`compare_models.py`)
- Analyze difficult samples in the reports
- Consider data augmentation techniques
- Explore the TensorBoard visualizations in detail

---

## 🏁 Quick Commands Summary

```bash
# Essential workflow
pip install -r requirements.txt
python train.py --model_type improved_cnn --epochs 50 --save_model
python experiment_manager.py --compare
tensorboard --logdir=experiments/[experiment_name]/logs/
python predict.py --model models/[model_name].h5 --audio [audio_file]
```

Happy experimenting! 🎵🤖
