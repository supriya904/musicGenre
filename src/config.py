# Configuration file for music genre classification project

import os

# Audio processing parameters
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# MFCC parameters
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
NUM_SEGMENTS = 10

# Data paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_PATH = os.path.join(DATA_DIR, "genres_original")
JSON_PATH = os.path.join(DATA_DIR, "processed_data.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Experiment tracking directories
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
TENSORBOARD_DIR = os.path.join(PROJECT_ROOT, "tensorboard_logs")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

# Genre mapping
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

# Model parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50
TEST_SIZE = 0.25
VALIDATION_SIZE = 0.2

# Training parameters
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
DROPOUT_RATE = 0.3
L2_REG = 0.001
