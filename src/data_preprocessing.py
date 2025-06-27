"""
Data preprocessing utilities for music genre classification
"""

import os
import json
import math
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import *

class AudioPreprocessor:
    """Class for preprocessing audio data and extracting features"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, 
                 n_fft=N_FFT, hop_length=HOP_LENGTH):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_mfcc_features(self, dataset_path, json_path, num_segments=NUM_SEGMENTS):
        """
        Extract MFCC features from audio files and save to JSON
        
        Args:
            dataset_path (str): Path to the dataset directory
            json_path (str): Path to save the processed data
            num_segments (int): Number of segments to divide each track into
        """
        # Data storage dictionary
        data = {
            "mapping": [],
            "mfcc": [],
            "labels": [],
            "features_info": {
                "n_mfcc": self.n_mfcc,
                "sample_rate": self.sample_rate,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "num_segments": num_segments
            }
        }
        
        samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
        expected_vects_per_segment = math.ceil(samples_per_segment / self.hop_length)
        
        print(f"Processing {num_segments} segments per track")
        print(f"Expected MFCC vectors per segment: {expected_vects_per_segment}")
        
        # Loop through all genres
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
            # Skip root directory
            if dirpath == dataset_path:
                continue
                
            # Extract genre name
            genre_name = os.path.basename(dirpath)
            data["mapping"].append(genre_name)
            print(f"\nProcessing genre: {genre_name}")
            
            # Process each audio file
            for file_idx, filename in enumerate(filenames):
                if not filename.endswith('.wav'):
                    continue
                    
                # Skip problematic files
                if filename == "jazz.00054.wav":
                    print(f"Skipping {filename} (file size issue)")
                    continue
                
                try:
                    file_path = os.path.join(dirpath, filename)
                    signal, sr = librosa.load(file_path, sr=self.sample_rate)
                    
                    # Extract features from each segment
                    for segment_idx in range(num_segments):
                        start_sample = samples_per_segment * segment_idx
                        finish_sample = start_sample + samples_per_segment
                        
                        # Extract MFCC features
                        mfcc = librosa.feature.mfcc(
                            y=signal[start_sample:finish_sample],
                            sr=sr,
                            n_fft=self.n_fft,
                            n_mfcc=self.n_mfcc,
                            hop_length=self.hop_length
                        )
                        
                        mfcc = mfcc.T  # Transpose to get time x features
                        
                        # Store MFCC if it has expected length
                        if len(mfcc) == expected_vects_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i - 1)  # i-1 because we skip root
                            
                    if (file_idx + 1) % 10 == 0:
                        print(f"Processed {file_idx + 1} files in {genre_name}")
                        
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue
        
        # Save processed data
        print(f"\nSaving processed data to {json_path}")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Total samples: {len(data['mfcc'])}")
        print(f"Genres: {data['mapping']}")
        
        return data
    
    def extract_additional_features(self, signal, sr):
        """
        Extract additional audio features beyond MFCC
        
        Args:
            signal: Audio signal
            sr: Sample rate
            
        Returns:
            dict: Dictionary of extracted features
        """
        features = {}
        
        # Spectral features
        features['spectral_centroid'] = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(signal)[0]
        
        # Chroma features
        features['chroma'] = librosa.feature.chroma_stft(y=signal, sr=sr)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=signal, sr=sr)
        features['tempo'] = tempo
        
        return features

class DataLoader:
    """Class for loading and preparing data for training"""
    
    @staticmethod
    def load_processed_data(json_path):
        """
        Load processed data from JSON file
        
        Args:
            json_path (str): Path to the JSON file
            
        Returns:
            tuple: (inputs, targets, mapping)
        """
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            inputs = np.array(data["mfcc"])
            targets = np.array(data["labels"])
            mapping = data["mapping"]
            
            print(f"Loaded data shape: {inputs.shape}")
            print(f"Labels shape: {targets.shape}")
            print(f"Number of classes: {len(mapping)}")
            
            return inputs, targets, mapping
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {json_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    @staticmethod
    def prepare_data_for_training(inputs, targets, test_size=TEST_SIZE, 
                                validation_size=VALIDATION_SIZE, random_state=42):
        """
        Split data into train, validation, and test sets
        
        Args:
            inputs: Input features
            targets: Target labels
            test_size: Proportion of test data
            validation_size: Proportion of validation data from training data
            random_state: Random seed
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            inputs, targets, test_size=test_size, 
            random_state=random_state, stratify=targets
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size, 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @staticmethod
    def prepare_for_cnn(X_train, X_val, X_test):
        """
        Prepare data for CNN by adding channel dimension
        
        Args:
            X_train, X_val, X_test: Input arrays
            
        Returns:
            tuple: Reshaped arrays with added channel dimension
        """
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis] 
        X_test = X_test[..., np.newaxis]
        
        return X_train, X_val, X_test

def normalize_data(X_train, X_val, X_test):
    """
    Normalize data using training set statistics
    
    Args:
        X_train, X_val, X_test: Input arrays
        
    Returns:
        tuple: Normalized arrays and normalization parameters
    """
    # Calculate statistics from training data only
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Normalize all sets using training statistics
    X_train_norm = (X_train - mean) / (std + 1e-8)
    X_val_norm = (X_val - mean) / (std + 1e-8)
    X_test_norm = (X_test - mean) / (std + 1e-8)
    
    return X_train_norm, X_val_norm, X_test_norm, {"mean": mean, "std": std}
