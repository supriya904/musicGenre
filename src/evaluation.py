"""
Utilities for training evaluation, visualization, and model comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from src.config import GENRE_MAPPING

class ModelEvaluator:
    """Class for evaluating and comparing models"""
    
    def __init__(self, class_names=None):
        if class_names is None:
            self.class_names = list(GENRE_MAPPING.values())
        else:
            self.class_names = class_names
    
    def evaluate_model(self, model, X_test, y_test, verbose=True):
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            X_test, y_test: Test data
            verbose: Whether to print detailed results
            
        Returns:
            dict: Evaluation metrics
        """
        # Get predictions
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        if verbose:
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print("\nDetailed Classification Report:")
            print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, history, title="Training History"):
        """
        Plot training and validation metrics
        
        Args:
            history: Training history from model.fit()
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss', marker='o')
        ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_confidence(self, y_true, y_pred_proba, num_samples=20):
        """
        Plot prediction confidence for sample predictions
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            num_samples: Number of samples to display
        """
        indices = np.random.choice(len(y_true), num_samples, replace=False)
        
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            if i >= 20:
                break
                
            true_label = self.class_names[y_true[idx]]
            pred_probs = y_pred_proba[idx]
            pred_label = self.class_names[np.argmax(pred_probs)]
            confidence = np.max(pred_probs)
            
            # Create bar plot of probabilities
            axes[i].bar(range(len(self.class_names)), pred_probs)
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})')
            axes[i].set_xticks(range(len(self.class_names)))
            axes[i].set_xticklabels(self.class_names, rotation=45, ha='right')
            axes[i].set_ylabel('Probability')
        
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, models_results):
        """
        Compare multiple models
        
        Args:
            models_results: Dict with model names as keys and evaluation results as values
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(models_results.keys())
        
        # Create comparison dataframe
        comparison_data = {}
        for metric in metrics:
            comparison_data[metric] = [models_results[model][metric] for model in model_names]
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            axes[i].bar(model_names, comparison_data[metric])
            axes[i].set_title(f'{metric.capitalize()} Comparison')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(comparison_data[metric]):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Print comparison table
        print("\nModel Comparison Summary:")
        print("-" * 60)
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 60)
        for model in model_names:
            results = models_results[model]
            print(f"{model:<20} {results['accuracy']:<10.3f} {results['precision']:<10.3f} "
                  f"{results['recall']:<10.3f} {results['f1_score']:<10.3f}")

class PredictionAnalyzer:
    """Class for analyzing individual predictions"""
    
    def __init__(self, class_names=None):
        if class_names is None:
            self.class_names = list(GENRE_MAPPING.values())
        else:
            self.class_names = class_names
    
    def predict_single_sample(self, model, X_sample, y_true=None):
        """
        Predict and analyze a single sample
        
        Args:
            model: Trained model
            X_sample: Single input sample
            y_true: True label (optional)
            
        Returns:
            dict: Prediction results
        """
        # Ensure correct shape for prediction
        if len(X_sample.shape) == len(model.input_shape) - 1:
            X_sample = np.expand_dims(X_sample, axis=0)
        
        # Get prediction
        pred_proba = model.predict(X_sample, verbose=0)
        pred_class = np.argmax(pred_proba, axis=1)[0]
        confidence = np.max(pred_proba)
        
        result = {
            'predicted_class': pred_class,
            'predicted_genre': self.class_names[pred_class],
            'confidence': confidence,
            'probabilities': pred_proba[0]
        }
        
        if y_true is not None:
            result['true_class'] = y_true
            result['true_genre'] = self.class_names[y_true]
            result['correct'] = pred_class == y_true
        
        return result
    
    def analyze_misclassifications(self, model, X_test, y_test, num_examples=10):
        """
        Analyze misclassified examples
        
        Args:
            model: Trained model
            X_test, y_test: Test data
            num_examples: Number of misclassified examples to analyze
        """
        # Get predictions
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Find misclassified examples
        misclassified = np.where(y_pred != y_test)[0]
        
        if len(misclassified) == 0:
            print("No misclassifications found!")
            return
        
        print(f"Found {len(misclassified)} misclassified examples")
        print(f"Analyzing first {min(num_examples, len(misclassified))} examples:\n")
        
        for i, idx in enumerate(misclassified[:num_examples]):
            true_genre = self.class_names[y_test[idx]]
            pred_genre = self.class_names[y_pred[idx]]
            confidence = np.max(y_pred_proba[idx])
            
            print(f"Example {i+1}:")
            print(f"  True genre: {true_genre}")
            print(f"  Predicted genre: {pred_genre}")
            print(f"  Confidence: {confidence:.3f}")
            
            # Show top 3 predictions
            top_3_idx = np.argsort(y_pred_proba[idx])[-3:][::-1]
            print("  Top 3 predictions:")
            for j, class_idx in enumerate(top_3_idx):
                print(f"    {j+1}. {self.class_names[class_idx]}: {y_pred_proba[idx][class_idx]:.3f}")
            print()

def save_evaluation_results(results, filepath):
    """
    Save evaluation results to file
    
    Args:
        results: Evaluation results dictionary
        filepath: Path to save results
    """
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {filepath}")

def load_evaluation_results(filepath):
    """
    Load evaluation results from file
    
    Args:
        filepath: Path to load results from
        
    Returns:
        dict: Loaded results
    """
    import json
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    # Convert relevant lists back to numpy arrays
    if 'predictions' in results:
        results['predictions'] = np.array(results['predictions'])
    if 'probabilities' in results:
        results['probabilities'] = np.array(results['probabilities'])
    
    return results
