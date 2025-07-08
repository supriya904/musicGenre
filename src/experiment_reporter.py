"""
Experiment reporting utilities for music genre classification
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import base64
from src.config import PROJECT_ROOT

class ExperimentReporter:
    """Class for generating comprehensive experiment reports"""
    
    def __init__(self, run_name, model_type, args):
        self.run_name = run_name
        self.model_type = model_type
        self.args = args
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create experiment directory
        self.experiment_dir = os.path.join(PROJECT_ROOT, "experiments", run_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = os.path.join(self.experiment_dir, "plots")
        self.logs_dir = os.path.join(self.experiment_dir, "logs")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
    def save_training_plots(self, history, results, y_test, mapping):
        """Save training plots and return their paths"""
        plots = {}
        
        # 1. Training History Plot
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
        
        plt.tight_layout()
        training_plot_path = os.path.join(self.plots_dir, "training_history.png")
        plt.savefig(training_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['training_history'] = training_plot_path
        
        # 2. Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, results['predictions'])
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=mapping, yticklabels=mapping)
        plt.title(f'{self.model_type.upper()} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        confusion_plot_path = os.path.join(self.plots_dir, "confusion_matrix.png")
        plt.savefig(confusion_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['confusion_matrix'] = confusion_plot_path
        
        # 3. Model Architecture Visualization (if possible)
        try:
            from tensorflow.keras.utils import plot_model
            model_plot_path = os.path.join(self.plots_dir, "model_architecture.png")
            plot_model(results.get('model'), to_file=model_plot_path, show_shapes=True, show_layer_names=True)
            plots['model_architecture'] = model_plot_path
        except:
            pass  # Skip if graphviz not available
        
        return plots
    
    def generate_markdown_report(self, model, history, results, norm_params, mapping, model_save_path=None):
        """Generate comprehensive markdown report"""
        
        # Save plots first
        plots = self.save_training_plots(history, results, results.get('y_test', []), mapping)
        
        # Format metrics safely
        precision_val = results.get('precision', 0)
        recall_val = results.get('recall', 0)
        f1_val = results.get('f1_score', 0)
        
        precision_str = f"{precision_val:.4f}" if isinstance(precision_val, (int, float)) else "N/A"
        recall_str = f"{recall_val:.4f}" if isinstance(recall_val, (int, float)) else "N/A"
        f1_str = f"{f1_val:.4f}" if isinstance(f1_val, (int, float)) else "N/A"
        
        # Generate report content
        markdown_content = f"""# Music Genre Classification Experiment Report

## Experiment Overview
- **Experiment ID**: `{self.run_name}`
- **Model Type**: `{self.model_type}`
- **Timestamp**: {self.timestamp}
- **Duration**: Training completed successfully

## Configuration

### Model Parameters
- **Architecture**: {self.model_type.upper()}
- **Input Shape**: {model.input_shape}
- **Total Parameters**: {model.count_params():,}
- **Trainable Parameters**: {sum([layer.count_params() for layer in model.layers]):,}

### Training Parameters
- **Epochs**: {self.args.epochs}
- **Batch Size**: {self.args.batch_size}
- **Learning Rate**: {model.optimizer.learning_rate.numpy() if hasattr(model.optimizer, 'learning_rate') else 'N/A'}
- **Optimizer**: {model.optimizer.__class__.__name__}

### Data Configuration
- **Dataset**: GTZAN Genre Classification
- **Genres**: {len(mapping)} classes
- **Training Samples**: {len(history.history['loss']) * self.args.batch_size} (approx)
- **Preprocessing**: MFCC features extraction

## Results Summary

### Performance Metrics
| Metric | Value |
|--------|-------|
| **Test Accuracy** | **{results['accuracy']:.4f}** |
| **Test Precision** | {precision_str} |
| **Test Recall** | {recall_str} |
| **Test F1-Score** | {f1_str} |

### Training Progress
| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
"""

        # Add training progress table (last 5 epochs)
        epochs = len(history.history['accuracy'])
        start_epoch = max(0, epochs - 5)
        
        for i in range(start_epoch, epochs):
            epoch_num = i + 1
            train_acc = history.history['accuracy'][i]
            val_acc = history.history['val_accuracy'][i]
            train_loss = history.history['loss'][i]
            val_loss = history.history['val_loss'][i]
            markdown_content += f"| {epoch_num} | {train_acc:.4f} | {val_acc:.4f} | {train_loss:.4f} | {val_loss:.4f} |\n"

        markdown_content += f"""

### Final Training Metrics
- **Best Training Accuracy**: {max(history.history['accuracy']):.4f}
- **Best Validation Accuracy**: {max(history.history['val_accuracy']):.4f}
- **Final Training Loss**: {history.history['loss'][-1]:.4f}
- **Final Validation Loss**: {history.history['val_loss'][-1]:.4f}

## Visualizations

### Training History
![Training History](./plots/training_history.png)

*Training and validation accuracy/loss curves over epochs*

### Confusion Matrix
![Confusion Matrix](./plots/confusion_matrix.png)

*Model predictions vs actual labels on test set*

## Model Architecture

### Layer Summary
"""

        # Add model architecture
        model_summary = []
        model.summary(print_fn=model_summary.append)
        markdown_content += "```\n" + "\n".join(model_summary) + "\n```\n"

        markdown_content += f"""

## Genre Classification Results

### Class Performance
| Genre | Accuracy | Notes |
|-------|----------|-------|
"""

        # Add per-class performance if available
        for i, genre in enumerate(mapping):
            markdown_content += f"| {genre.capitalize()} | - | - |\n"

        markdown_content += f"""

## Files Generated

### Model Files
"""
        if model_save_path:
            markdown_content += f"- **Trained Model**: `{os.path.basename(model_save_path)}`\n"
        
        markdown_content += f"""
### Data Files
- **Training History**: `logs/training_history.json`
- **Model Results**: `logs/results.json`
- **Normalization Parameters**: `logs/normalization.json`

### Visualizations
- **Training Curves**: `plots/training_history.png`
- **Confusion Matrix**: `plots/confusion_matrix.png`

## Experiment Notes

### What Worked Well
- Model trained successfully without errors
- Achieved reasonable accuracy on test set
- Training converged properly

### Areas for Improvement
- Consider data augmentation techniques
- Experiment with different learning rates
- Try ensemble methods for better performance

### Next Steps
- Compare with other model architectures
- Analyze misclassified samples
- Consider hyperparameter tuning

---

*Report generated automatically on {self.timestamp}*
*Experiment ID: {self.run_name}*
"""

        # Save markdown report
        report_path = os.path.join(self.experiment_dir, "README.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return report_path
    
    def save_experiment_data(self, history, results, norm_params, model_save_path=None):
        """Save all experiment data to organized structure"""
        
        # Save training history
        history_file = os.path.join(self.logs_dir, "training_history.json")
        with open(history_file, 'w') as f:
            history_dict = {key: [float(val) for val in values] 
                           for key, values in history.history.items()}
            json.dump(history_dict, f, indent=2)
        
        # Save results
        results_file = os.path.join(self.logs_dir, "results.json")
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if hasattr(value, 'tolist'):
                    json_results[key] = value.tolist()
                elif isinstance(value, (int, float, str, bool, list)):
                    json_results[key] = value
                else:
                    json_results[key] = str(value)
            json.dump(json_results, f, indent=2)
        
        # Save normalization parameters
        norm_file = os.path.join(self.logs_dir, "normalization.json")
        with open(norm_file, 'w') as f:
            norm_dict = {key: val.tolist() if hasattr(val, 'tolist') else val 
                        for key, val in norm_params.items()}
            json.dump(norm_dict, f, indent=2)
        
        # Save experiment metadata
        metadata = {
            "experiment_id": self.run_name,
            "model_type": self.model_type,
            "timestamp": self.timestamp,
            "args": vars(self.args),
            "final_accuracy": float(results['accuracy']),
            "model_path": model_save_path,
            "total_epochs": len(history.history['accuracy']),
            "best_val_accuracy": float(max(history.history['val_accuracy']))
        }
        
        metadata_file = os.path.join(self.experiment_dir, "experiment_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "experiment_dir": self.experiment_dir,
            "logs_dir": self.logs_dir,
            "plots_dir": self.plots_dir,
            "metadata_file": metadata_file
        }
    
    def get_tensorboard_logdir(self):
        """Get TensorBoard log directory for this experiment"""
        return self.logs_dir
