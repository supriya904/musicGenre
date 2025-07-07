"""
Neural network models for music genre classification
"""

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, 
    Flatten, Dropout, LSTM, BatchNormalization, 
    GlobalAveragePooling1D, GlobalAveragePooling2D, Input, Add,
    TimeDistributed, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from src.config import *

class MusicGenreModels:
    """Collection of neural network models for music genre classification"""
    
    @staticmethod
    def create_ann_model(input_shape, num_classes=10):
        """
        Create a basic Artificial Neural Network
        
        Args:
            input_shape: Shape of input data (excluding batch dimension)
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Flatten(input_shape=input_shape),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @staticmethod
    def create_regularized_ann_model(input_shape, num_classes=10, 
                                   dropout_rate=DROPOUT_RATE, l2_reg=L2_REG):
        """
        Create an ANN with regularization to prevent overfitting
        
        Args:
            input_shape: Shape of input data
            num_classes: Number of output classes
            dropout_rate: Dropout probability
            l2_reg: L2 regularization factor
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Flatten(input_shape=input_shape),
            
            Dense(512, activation='relu', kernel_regularizer=l2(l2_reg)),
            Dropout(dropout_rate),
            
            Dense(256, activation='relu', kernel_regularizer=l2(l2_reg * 2)),
            Dropout(dropout_rate),
            
            Dense(64, activation='relu', kernel_regularizer=l2(l2_reg * 5)),
            Dropout(dropout_rate),
            
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @staticmethod
    def create_improved_cnn_model(input_shape, num_classes=10, dropout_rate=DROPOUT_RATE):
        """
        Create an improved CNN model for 1D audio data (MFCC features)
        Input shape should be (time_steps, features) = (130, 13)
        
        Args:
            input_shape: Shape of input data (time_steps, features)
            num_classes: Number of output classes
            dropout_rate: Dropout probability
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # First convolutional block
            Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),
            
            # Second convolutional block
            Conv1D(128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            Conv1D(128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),
            
            # Third convolutional block
            Conv1D(256, kernel_size=3, activation='relu'),
            BatchNormalization(),
            Dropout(0.25),
            
            # Global average pooling instead of Flatten
            GlobalAveragePooling1D(),
            
            # Dense layers
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @staticmethod
    def create_residual_cnn_model(input_shape, num_classes=10):
        """
        Create a CNN with residual connections for 1D audio data
        
        Args:
            input_shape: Shape of input data (time_steps, features)
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=input_shape)
        
        # Initial conv
        x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        
        # Residual block 1
        shortcut = x
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = MaxPooling1D(2)(x)
        x = Dropout(0.25)(x)
        
        # Residual block 2
        shortcut = Conv1D(128, 1, padding='same')(x)  # Match dimensions
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = MaxPooling1D(2)(x)
        x = Dropout(0.25)(x)
        
        # Final layers
        x = GlobalAveragePooling1D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @staticmethod
    def create_lstm_model(input_shape, num_classes=10):
        """
        Create an LSTM model for sequential audio data
        
        Args:
            input_shape: Shape of input data (time_steps, features)
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @staticmethod
    def create_cnn_lstm_hybrid(input_shape, num_classes=10):
        """
        Create a hybrid CNN-LSTM model for 1D audio data
        
        Args:
            input_shape: Shape of input data (time_steps, features)
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # CNN layers for local feature extraction
            Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),
            
            Conv1D(128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            Dropout(0.25),
            
            # LSTM layers for sequential modeling
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            
            # Dense layers
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

class ModelTrainer:
    """Class for training neural network models"""
    
    def __init__(self, model, learning_rate=LEARNING_RATE):
        self.model = model
        self.learning_rate = learning_rate
        self.history = None
        
    def compile_model(self, optimizer=None, loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy']):
        """
        Compile the model with specified parameters
        
        Args:
            optimizer: Keras optimizer (default: Adam)
            loss: Loss function
            metrics: List of metrics to track
        """
        if optimizer is None:
            optimizer = Adam(learning_rate=self.learning_rate)
            
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
    def get_callbacks(self, model_save_path=None):
        """
        Get training callbacks
        
        Args:
            model_save_path: Path to save best model
            
        Returns:
            List of callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=REDUCE_LR_PATIENCE,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        if model_save_path:
            callbacks.append(
                ModelCheckpoint(
                    model_save_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1
                )
            )
            
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, epochs=EPOCHS, 
              batch_size=BATCH_SIZE, callbacks=None):
        """
        Train the model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: List of callbacks
            
        Returns:
            Training history
        """
        if callbacks is None:
            callbacks = self.get_callbacks()
            
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            Test metrics
        """
        return self.model.evaluate(X_test, y_test, verbose=1)
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        return self.model.predict(X)
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
