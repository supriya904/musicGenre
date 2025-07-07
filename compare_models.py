"""
Compare different models and create ensemble predictions
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import *
from src.data_preprocessing import DataLoader, normalize_data
from src.models import MusicGenreModels, ModelTrainer
from src.evaluation import ModelEvaluator, PredictionAnalyzer

class ModelComparator:
    """Class for comparing multiple models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.data_loader = DataLoader()
        self.evaluator = ModelEvaluator()
        
    def load_data(self):
        """Load and prepare data for all models"""
        print("Loading and preparing data...")
        
        # Load processed data
        inputs, targets, mapping = self.data_loader.load_processed_data(JSON_PATH)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.prepare_data_for_training(
            inputs, targets
        )
        
        # Normalize data
        X_train_norm, X_val_norm, X_test_norm, norm_params = normalize_data(X_train, X_val, X_test)
        
        self.data = {
            'X_train': X_train_norm,
            'X_val': X_val_norm, 
            'X_test': X_test_norm,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'mapping': mapping,
            'norm_params': norm_params
        }
        
        print(f"Data loaded: {X_train_norm.shape[0]} training samples")
        
    def train_all_models(self, epochs=30):
        """Train all model types"""
        if not hasattr(self, 'data'):
            self.load_data()
            
        model_configs = [
            ('ANN', 'ann'),
            ('Regularized ANN', 'regularized_ann'),
            ('CNN', 'improved_cnn'),
            ('Residual CNN', 'residual_cnn'),
            ('LSTM', 'lstm')
        ]
        
        for name, model_type in model_configs:
            print(f"\nTraining {name}...")
            
            # All models now use the same data format
            X_train = self.data['X_train']
            X_val = self.data['X_val']
            X_test = self.data['X_test']
            input_shape = X_train.shape[1:]
            
            # Create model
            if model_type == 'ann':
                model = MusicGenreModels.create_ann_model(input_shape)
            elif model_type == 'regularized_ann':
                model = MusicGenreModels.create_regularized_ann_model(input_shape)
            elif model_type == 'improved_cnn':
                model = MusicGenreModels.create_improved_cnn_model(input_shape)
            elif model_type == 'residual_cnn':
                model = MusicGenreModels.create_residual_cnn_model(input_shape)
            elif model_type == 'lstm':
                model = MusicGenreModels.create_lstm_model(input_shape)
            
            # Train model
            trainer = ModelTrainer(model)
            trainer.compile_model()
            
            history = trainer.train(
                X_train, self.data['y_train'],
                X_val, self.data['y_val'],
                epochs=epochs,
                batch_size=32
            )
            
            # Evaluate model
            results = self.evaluator.evaluate_model(model, X_test, self.data['y_test'], verbose=False)
            
            # Store results
            self.models[name] = {
                'model': model,
                'history': history,
                'test_data': (X_test, self.data['y_test'])
            }
            self.results[name] = results
            
            print(f"{name} - Test Accuracy: {results['accuracy']:.4f}")
    
    def compare_models(self):
        """Compare all trained models"""
        if not self.results:
            print("No models trained yet. Please run train_all_models() first.")
            return
            
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        
        # Display comparison
        self.evaluator.compare_models(self.results)
        
        # Find best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        print(f"\nBest performing model: {best_model}")
        print(f"Best accuracy: {self.results[best_model]['accuracy']:.4f}")
        
        return best_model
    
    def create_ensemble(self):
        """Create ensemble predictions from multiple models"""
        if not self.models:
            print("No models available for ensemble. Please train models first.")
            return
        
        print("\nCreating ensemble predictions...")
        
        # Get predictions from all models
        ensemble_predictions = []
        ensemble_probabilities = []
        
        for name, model_info in self.models.items():
            model = model_info['model']
            X_test, y_test = model_info['test_data']
            
            # Get predictions
            pred_proba = model.predict(X_test, verbose=0)
            ensemble_probabilities.append(pred_proba)
            
            print(f"Added {name} to ensemble")
        
        # Average probabilities
        avg_probabilities = np.mean(ensemble_probabilities, axis=0)
        ensemble_pred = np.argmax(avg_probabilities, axis=1)
        
        # Evaluate ensemble
        from sklearn.metrics import accuracy_score, classification_report
        
        ensemble_accuracy = accuracy_score(self.data['y_test'], ensemble_pred)
        
        print(f"\nEnsemble Results:")
        print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
        print("\nEnsemble Classification Report:")
        print(classification_report(self.data['y_test'], ensemble_pred, 
                                  target_names=self.data['mapping']))
        
        # Compare with individual models
        print(f"\nAccuracy Comparison:")
        print("-" * 40)
        for name in self.results.keys():
            individual_acc = self.results[name]['accuracy']
            improvement = ensemble_accuracy - individual_acc
            print(f"{name:<20}: {individual_acc:.4f} ({improvement:+.4f})")
        print(f"{'Ensemble':<20}: {ensemble_accuracy:.4f}")
        
        return ensemble_pred, avg_probabilities
    
    def analyze_difficult_samples(self):
        """Analyze samples that are difficult to classify"""
        if not self.models:
            print("No models available. Please train models first.")
            return
        
        print("\nAnalyzing difficult samples...")
        
        # Find samples that multiple models get wrong
        all_predictions = {}
        
        for name, model_info in self.models.items():
            model = model_info['model']
            X_test, y_test = model_info['test_data']
            
            pred_proba = model.predict(X_test, verbose=0)
            pred = np.argmax(pred_proba, axis=1)
            
            all_predictions[name] = pred
        
        # Find samples where most models are wrong
        y_test = self.data['y_test']
        difficult_samples = []
        
        for i in range(len(y_test)):
            correct_predictions = 0
            for name in all_predictions:
                if all_predictions[name][i] == y_test[i]:
                    correct_predictions += 1
            
            # If less than half of models got it right, consider it difficult
            if correct_predictions < len(all_predictions) / 2:
                difficult_samples.append(i)
        
        print(f"Found {len(difficult_samples)} difficult samples ({len(difficult_samples)/len(y_test)*100:.1f}%)")
        
        # Analyze genre distribution of difficult samples
        difficult_genres = [y_test[i] for i in difficult_samples]
        genre_counts = {}
        for genre in difficult_genres:
            genre_name = self.data['mapping'][genre]
            genre_counts[genre_name] = genre_counts.get(genre_name, 0) + 1
        
        print("\nDifficult samples by genre:")
        for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {genre}: {count} samples")
        
        return difficult_samples

def main():
    """Main function to run model comparison"""
    print("Music Genre Classification - Model Comparison")
    print("=" * 50)
    
    # Check if processed data exists
    if not os.path.exists(JSON_PATH):
        print(f"Error: Processed data file {JSON_PATH} not found!")
        print("Please run train.py with --preprocess flag first")
        return
    
    # Create comparator
    comparator = ModelComparator()
    
    # Load data
    comparator.load_data()
    
    # Train all models
    print("\nTraining all models...")
    comparator.train_all_models(epochs=25)  # Reduced epochs for comparison
    
    # Compare models
    best_model = comparator.compare_models()
    
    # Create ensemble
    ensemble_pred, ensemble_proba = comparator.create_ensemble()
    
    # Analyze difficult samples
    difficult_samples = comparator.analyze_difficult_samples()
    
    print(f"\nModel comparison completed!")
    print(f"Best individual model: {best_model}")
    print(f"Consider using ensemble for improved performance")

if __name__ == "__main__":
    main()
