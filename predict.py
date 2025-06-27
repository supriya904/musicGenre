"""
Inference script for making predictions on new audio files
"""

import os
import sys
import argparse
import librosa
import numpy as np
import tensorflow as tf
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import *
from src.data_preprocessing import AudioPreprocessor

class MusicGenrePredictor:
    """Class for making predictions on new audio files"""
    
    def __init__(self, model_path, normalization_path=None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
            normalization_path: Path to normalization parameters (optional)
        """
        self.model = tf.keras.models.load_model(model_path)
        self.preprocessor = AudioPreprocessor()
        self.genre_mapping = GENRE_MAPPING
        
        # Load normalization parameters if provided
        self.norm_params = None
        if normalization_path and os.path.exists(normalization_path):
            with open(normalization_path, 'r') as f:
                norm_data = json.load(f)
                self.norm_params = {
                    'mean': np.array(norm_data['mean']),
                    'std': np.array(norm_data['std'])
                }
                print("Loaded normalization parameters")
        
        print(f"Model loaded from {model_path}")
        print(f"Input shape: {self.model.input_shape}")
    
    def preprocess_audio_file(self, audio_path, num_segments=NUM_SEGMENTS):
        """
        Preprocess a single audio file for prediction
        
        Args:
            audio_path: Path to audio file
            num_segments: Number of segments to extract
            
        Returns:
            Preprocessed features
        """
        try:
            # Load audio file
            signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            
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
            raise Exception(f"Error preprocessing audio file {audio_path}: {str(e)}")
    
    def normalize_features(self, features):
        """Apply normalization to features"""
        if self.norm_params is not None:
            features = (features - self.norm_params['mean']) / (self.norm_params['std'] + 1e-8)
        return features
    
    def predict_genre(self, audio_path, return_probabilities=False):
        """
        Predict genre for a single audio file
        
        Args:
            audio_path: Path to audio file
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Prediction results
        """
        # Preprocess audio
        features = self.preprocess_audio_file(audio_path)
        
        # Normalize features
        features = self.normalize_features(features)
        
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
        
        # Get genre name
        predicted_genre = self.genre_mapping[final_prediction]
        
        result = {
            'predicted_class': int(final_prediction),
            'predicted_genre': predicted_genre,
            'confidence': float(confidence),
            'segment_predictions': [int(p) for p in segment_predictions],
            'segment_agreement': float(np.mean(np.array(segment_predictions) == final_prediction))
        }
        
        if return_probabilities:
            result['probabilities'] = avg_probabilities.tolist()
            result['genre_probabilities'] = {
                self.genre_mapping[i]: float(prob) 
                for i, prob in enumerate(avg_probabilities)
            }
        
        return result
    
    def predict_batch(self, audio_paths, return_probabilities=False):
        """
        Predict genres for multiple audio files
        
        Args:
            audio_paths: List of audio file paths
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, audio_path in enumerate(audio_paths):
            print(f"Processing {i+1}/{len(audio_paths)}: {os.path.basename(audio_path)}")
            
            try:
                result = self.predict_genre(audio_path, return_probabilities)
                result['file_path'] = audio_path
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                results.append({
                    'file_path': audio_path,
                    'error': str(e)
                })
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Predict music genre for audio files')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--audio', type=str, required=True,
                       help='Path to audio file or directory containing audio files')
    parser.add_argument('--normalization', type=str,
                       help='Path to normalization parameters file')
    parser.add_argument('--output', type=str,
                       help='Path to save prediction results (JSON format)')
    parser.add_argument('--probabilities', action='store_true',
                       help='Include class probabilities in output')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found!")
        return
    
    # Initialize predictor
    try:
        predictor = MusicGenrePredictor(args.model, args.normalization)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Get audio files to process
    audio_files = []
    if os.path.isfile(args.audio):
        # Single file
        audio_files = [args.audio]
    elif os.path.isdir(args.audio):
        # Directory of files
        supported_formats = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        for file in os.listdir(args.audio):
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                audio_files.append(os.path.join(args.audio, file))
        
        if not audio_files:
            print(f"No supported audio files found in {args.audio}")
            return
    else:
        print(f"Error: {args.audio} is not a valid file or directory!")
        return
    
    print(f"Found {len(audio_files)} audio file(s) to process")
    
    # Make predictions
    if len(audio_files) == 1:
        print(f"\nPredicting genre for: {os.path.basename(audio_files[0])}")
        result = predictor.predict_genre(audio_files[0], args.probabilities)
        
        print(f"\nPrediction Results:")
        print(f"Predicted Genre: {result['predicted_genre']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Segment Agreement: {result['segment_agreement']:.3f}")
        
        if args.probabilities:
            print(f"\nGenre Probabilities:")
            for genre, prob in sorted(result['genre_probabilities'].items(), 
                                    key=lambda x: x[1], reverse=True):
                print(f"  {genre}: {prob:.3f}")
        
        results = [result]
    else:
        print(f"\nProcessing {len(audio_files)} files...")
        results = predictor.predict_batch(audio_files, args.probabilities)
        
        # Print summary
        successful_predictions = [r for r in results if 'error' not in r]
        print(f"\nSuccessfully processed: {len(successful_predictions)}/{len(audio_files)} files")
        
        if successful_predictions:
            # Show genre distribution
            genre_counts = {}
            for result in successful_predictions:
                genre = result['predicted_genre']
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            print(f"\nPredicted Genre Distribution:")
            for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {genre}: {count} files")
    
    # Save results if output path specified
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")

if __name__ == "__main__":
    main()
