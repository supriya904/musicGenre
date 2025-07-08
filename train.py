"""
Main training script for music genre classification
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import *
from src.data_preprocessing import AudioPreprocessor, DataLoader, normalize_data
from src.models import MusicGenreModels, ModelTrainer
from src.evaluation import ModelEvaluator, save_evaluation_results
from src.experiment_reporter import ExperimentReporter

def main():
    parser = argparse.ArgumentParser(description='Train music genre classification model')
    parser.add_argument('--model_type', type=str, default='improved_cnn',
                       choices=['ann', 'regularized_ann', 'cnn', 'improved_cnn', 'residual_cnn', 'lstm'],
                       help='Type of model to train')
    parser.add_argument('--preprocess', action='store_true',
                       help='Run data preprocessing')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--save_model', action='store_true',
                       help='Save the trained model')
    
    args = parser.parse_args()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model_type}_{timestamp}"
    
    # Initialize experiment reporter
    reporter = ExperimentReporter(run_name, args.model_type, args)
    
    print(f"Starting training run: {run_name}")
    print(f"Model type: {args.model_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Experiment directory: {reporter.experiment_dir}")
    print("-" * 50)
    
    # Step 1: Data preprocessing (if requested)
    if args.preprocess:
        print("Starting data preprocessing...")
        preprocessor = AudioPreprocessor()
        
        if not os.path.exists(DATASET_PATH):
            print(f"Error: Dataset path {DATASET_PATH} not found!")
            print("Please update the DATASET_PATH in config.py")
            return
        
        preprocessor.extract_mfcc_features(DATASET_PATH, JSON_PATH)
        print("Data preprocessing completed!")
    
    # Step 2: Load and prepare data
    print("Loading processed data...")
    
    if not os.path.exists(JSON_PATH):
        print(f"Error: Processed data file {JSON_PATH} not found!")
        print("Please run with --preprocess flag first")
        return
    
    data_loader = DataLoader()
    inputs, targets, mapping = data_loader.load_processed_data(JSON_PATH)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.prepare_data_for_training(
        inputs, targets
    )
    
    # Normalize data
    X_train, X_val, X_test, norm_params = normalize_data(X_train, X_val, X_test)
    print("Data normalization completed!")
    
    # Step 3: Create model
    print(f"Creating {args.model_type} model...")
    
    # All models now use the same input shape (time_steps, features)
    input_shape = X_train.shape[1:]
    
    # Create model based on type
    if args.model_type == 'ann':
        model = MusicGenreModels.create_ann_model(input_shape)
    elif args.model_type == 'regularized_ann':
        model = MusicGenreModels.create_regularized_ann_model(input_shape)
    elif args.model_type == 'cnn':
        # Use original CNN architecture adapted for 1D
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten, Dense, Dropout
        
        model = Sequential([
            Conv1D(64, 3, activation="relu", input_shape=input_shape),
            MaxPooling1D(3, strides=2, padding="same"),
            BatchNormalization(),
            
            Conv1D(32, 3, activation="relu"),
            MaxPooling1D(3, strides=2, padding="same"),
            BatchNormalization(),
            
            Conv1D(32, 2, activation="relu"),
            MaxPooling1D(2, strides=2, padding="same"),
            BatchNormalization(),
            
            Conv1D(16, 1, activation="relu"),
            MaxPooling1D(1, strides=2, padding="same"),
            BatchNormalization(),
            
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(10, activation="softmax")
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    elif args.model_type == 'improved_cnn':
        model = MusicGenreModels.create_improved_cnn_model(input_shape)
    elif args.model_type == 'residual_cnn':
        model = MusicGenreModels.create_residual_cnn_model(input_shape)
    elif args.model_type == 'lstm':
        model = MusicGenreModels.create_lstm_model(input_shape)
    
    print("Model created successfully!")
    model.summary()
    
    # Step 4: Train model
    print("Starting model training...")
    trainer = ModelTrainer(model)
    trainer.compile_model()
    
    # Set up model save path if requested
    model_save_path = None
    if args.save_model:
        model_save_path = os.path.join(MODELS_DIR, f"{run_name}.h5")
        callbacks = trainer.get_callbacks(model_save_path, reporter.get_tensorboard_logdir())
    else:
        callbacks = trainer.get_callbacks(tensorboard_log_dir=reporter.get_tensorboard_logdir())
    
    # Train the model
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )
    
    print("Training completed!")
    
    # Step 5: Evaluate model
    print("Evaluating model...")
    evaluator = ModelEvaluator(mapping)
    
    # Evaluate on test set
    results = evaluator.evaluate_model(model, X_test, y_test)
    results['y_test'] = y_test  # Add for confusion matrix generation
    
    # Generate comprehensive experiment report
    print("Generating experiment report...")
    report_path = reporter.generate_markdown_report(
        model, history, results, norm_params, mapping, model_save_path
    )
    
    # Save all experiment data in organized structure
    experiment_files = reporter.save_experiment_data(
        history, results, norm_params, model_save_path
    )
    
    # Legacy: Also save to old results directory for backward compatibility
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f"{run_name}_results.json")
    save_evaluation_results(results, results_file)
    
    # Save training history (legacy)
    history_file = os.path.join(results_dir, f"{run_name}_history.json")
    with open(history_file, 'w') as f:
        history_dict = {key: [float(val) for val in values] 
                       for key, values in history.history.items()}
        json.dump(history_dict, f, indent=2)
    
    # Save normalization parameters (legacy)
    norm_file = os.path.join(results_dir, f"{run_name}_normalization.json")
    with open(norm_file, 'w') as f:
        norm_dict = {key: val.tolist() if hasattr(val, 'tolist') else val 
                    for key, val in norm_params.items()}
        json.dump(norm_dict, f, indent=2)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📊 Final test accuracy: {results['accuracy']:.4f}")
    print(f"📁 Experiment folder: {reporter.experiment_dir}")
    print(f"📋 Detailed report: {report_path}")
    print(f"📈 TensorBoard logs: {reporter.get_tensorboard_logdir()}")
    print(f"📊 Legacy results: {results_dir}")
    
    if args.save_model:
        print(f"💾 Model saved to: {model_save_path}")
    
    print(f"\n🚀 To view TensorBoard:")
    print(f"   tensorboard --logdir='{reporter.get_tensorboard_logdir()}'")
    print(f"   Then open: http://localhost:6006")

if __name__ == "__main__":
    main()
