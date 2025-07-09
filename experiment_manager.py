"""
Experiment management utilities for music genre classification
"""

import os
import json
import pandas as pd
from datetime import datetime
from src.config import PROJECT_ROOT, EXPERIMENTS_DIR

class ExperimentManager:
    """Class for managing and comparing experiments"""
    
    def __init__(self):
        self.experiments_dir = EXPERIMENTS_DIR
    
    def list_experiments(self):
        """List all available experiments"""
        if not os.path.exists(self.experiments_dir):
            print("No experiments directory found.")
            return []
        
        experiments = []
        for exp_dir in os.listdir(self.experiments_dir):
            exp_path = os.path.join(self.experiments_dir, exp_dir)
            if os.path.isdir(exp_path):
                metadata_file = os.path.join(exp_path, "experiment_metadata.json")
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    experiments.append(metadata)
        
        return sorted(experiments, key=lambda x: x['timestamp'], reverse=True)
    
    def compare_experiments(self, output_file=None):
        """Compare all experiments and create a summary"""
        experiments = self.list_experiments()
        
        if not experiments:
            print("No experiments found to compare.")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for exp in experiments:
            comparison_data.append({
                'Experiment ID': exp['experiment_id'],
                'Model Type': exp['model_type'],
                'Timestamp': exp['timestamp'],
                'Final Accuracy': f"{exp['final_accuracy']*100:.2f}%",
                'Best Val Accuracy': f"{exp['best_val_accuracy']*100:.2f}%",
                'Total Epochs': exp['total_epochs'],
                'Batch Size': exp['args'].get('batch_size', 'N/A'),
                'Learning Rate': exp['args'].get('learning_rate', 'N/A')
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Print to console
        print("\n📊 EXPERIMENT COMPARISON SUMMARY")
        print("=" * 80)
        print(df.to_string(index=False))
        
        # Save to file if requested
        if output_file:
            output_path = os.path.join(PROJECT_ROOT, output_file)
            df.to_csv(output_path, index=False)
            print(f"\n💾 Comparison saved to: {output_path}")
        
        # Find best performing experiment
        best_exp = max(experiments, key=lambda x: x['final_accuracy'])
        print(f"\n🏆 BEST PERFORMING EXPERIMENT:")
        print(f"   ID: {best_exp['experiment_id']}")
        print(f"   Model: {best_exp['model_type']}")
        print(f"   Accuracy: {best_exp['final_accuracy']*100:.2f}%")
        print(f"   Val Accuracy: {best_exp['best_val_accuracy']*100:.2f}%")
        
        return df
    
    def get_experiment_details(self, experiment_id):
        """Get detailed information about a specific experiment"""
        exp_path = os.path.join(self.experiments_dir, experiment_id)
        
        if not os.path.exists(exp_path):
            print(f"Experiment {experiment_id} not found.")
            return None
        
        # Load metadata
        metadata_file = os.path.join(exp_path, "experiment_metadata.json")
        if not os.path.exists(metadata_file):
            print(f"Metadata file not found for experiment {experiment_id}")
            return None
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load training history
        history_file = os.path.join(exp_path, "logs", "training_history.json")
        history = None
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        
        # Load results
        results_file = os.path.join(exp_path, "logs", "results.json")
        results = None
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
        
        return {
            'metadata': metadata,
            'history': history,
            'results': results,
            'experiment_path': exp_path
        }
    
    def create_summary_report(self):
        """Create a comprehensive summary report of all experiments"""
        experiments = self.list_experiments()
        
        if not experiments:
            print("No experiments found.")
            return
        
        report_content = f"""# Music Genre Classification - Experiment Summary

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview
- Total Experiments: {len(experiments)}
- Best Accuracy: {max(exp['final_accuracy'] for exp in experiments)*100:.2f}%
- Date Range: {min(exp['timestamp'] for exp in experiments)} to {max(exp['timestamp'] for exp in experiments)}

## Experiment Results Summary

| Experiment ID | Model Type | Accuracy | Val Accuracy | Epochs | Date |
|---------------|------------|----------|--------------|--------|------|
"""
        
        for exp in experiments:
            report_content += f"| {exp['experiment_id']} | {exp['model_type']} | {exp['final_accuracy']*100:.2f}% | {exp['best_val_accuracy']*100:.2f}% | {exp['total_epochs']} | {exp['timestamp'][:10]} |\n"
        
        # Model type analysis
        model_types = {}
        for exp in experiments:
            model_type = exp['model_type']
            if model_type not in model_types:
                model_types[model_type] = []
            model_types[model_type].append(exp['final_accuracy'])
        
        report_content += f"""

## Model Type Performance

| Model Type | Experiments | Best Accuracy | Avg Accuracy |
|------------|-------------|---------------|--------------|
"""
        
        for model_type, accuracies in model_types.items():
            best_acc = max(accuracies)
            avg_acc = sum(accuracies) / len(accuracies)
            count = len(accuracies)
            report_content += f"| {model_type} | {count} | {best_acc*100:.2f}% | {avg_acc*100:.2f}% |\n"
        
        # Best experiment details
        best_exp = max(experiments, key=lambda x: x['final_accuracy'])
        report_content += f"""

## Best Performing Experiment

**Experiment ID:** {best_exp['experiment_id']}
- **Model Type:** {best_exp['model_type']}
- **Final Accuracy:** {best_exp['final_accuracy']*100:.2f}%
- **Best Validation Accuracy:** {best_exp['best_val_accuracy']*100:.2f}%
- **Total Epochs:** {best_exp['total_epochs']}
- **Timestamp:** {best_exp['timestamp']}

## Next Steps
1. Analyze the best performing model architecture
2. Consider hyperparameter tuning based on successful experiments
3. Experiment with ensemble methods using top performers
4. Investigate data augmentation techniques

---
*This report was generated automatically by the Experiment Manager*
"""
        
        # Save report
        results_dir = os.path.join(PROJECT_ROOT, "results")
        os.makedirs(results_dir, exist_ok=True)
        report_path = os.path.join(results_dir, "experiment_summary.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📋 Summary report saved to: {report_path}")
        return report_path

def main():
    """CLI for experiment management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage music genre classification experiments')
    parser.add_argument('--list', action='store_true', help='List all experiments')
    parser.add_argument('--compare', action='store_true', help='Compare all experiments')
    parser.add_argument('--summary', action='store_true', help='Create summary report')
    parser.add_argument('--details', type=str, help='Get details for specific experiment ID')
    parser.add_argument('--output', type=str, help='Output file for comparison (CSV format)')
    
    args = parser.parse_args()
    
    manager = ExperimentManager()
    
    if args.list:
        experiments = manager.list_experiments()
        print(f"\n📁 Found {len(experiments)} experiments:")
        for exp in experiments:
            print(f"   {exp['experiment_id']} ({exp['model_type']}) - Accuracy: {exp['final_accuracy']*100:.2f}%")
    
    elif args.compare:
        manager.compare_experiments(args.output)
    
    elif args.summary:
        manager.create_summary_report()
    
    elif args.details:
        details = manager.get_experiment_details(args.details)
        if details:
            print(f"\n📊 Experiment Details: {args.details}")
            print(f"   Model Type: {details['metadata']['model_type']}")
            print(f"   Final Accuracy: {details['metadata']['final_accuracy']*100:.2f}%")
            print(f"   Best Val Accuracy: {details['metadata']['best_val_accuracy']*100:.2f}%")
            print(f"   Timestamp: {details['metadata']['timestamp']}")
            print(f"   Path: {details['experiment_path']}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
