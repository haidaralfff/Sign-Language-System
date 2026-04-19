"""
Training pipeline for SIBI gesture recognition model.

Implements:
- Two-phase training (head-only then fine-tuning)
- ModelCheckpoint and TensorBoard callbacks
- Class weight balancing using sklearn
- Learning rate scheduling
- Comprehensive logging and checkpointing
"""

import logging
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration container for training parameters."""
    
    def __init__(
        self,
        phase1_epochs: int = 50,
        phase2_epochs: int = 30,
        batch_size: int = 32,
        phase1_lr: float = 0.001,
        phase2_lr: float = 1e-5,
        early_stopping_patience: int = 15,
        reduce_lr_patience: int = 5,
        reduce_lr_factor: float = 0.5,
        validation_split: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initialize training configuration.
        
        Args:
            phase1_epochs (int): Training epochs for head-only phase
            phase2_epochs (int): Training epochs for fine-tuning phase
            batch_size (int): Batch size for training
            phase1_lr (float): Learning rate for phase 1
            phase2_lr (float): Learning rate for phase 2 fine-tuning
            early_stopping_patience (int): EarlyStopping patience
            reduce_lr_patience (int): ReduceLROnPlateau patience
            reduce_lr_factor (float): LR reduction factor
            validation_split (float): Validation set ratio
            random_state (int): Random seed for reproducibility
        """
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs
        self.batch_size = batch_size
        self.phase1_lr = phase1_lr
        self.phase2_lr = phase2_lr
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.validation_split = validation_split
        self.random_state = random_state
    
    def __repr__(self) -> str:
        return (
            f"TrainingConfig(\n"
            f"  phase1_epochs={self.phase1_epochs},\n"
            f"  phase2_epochs={self.phase2_epochs},\n"
            f"  batch_size={self.batch_size},\n"
            f"  phase1_lr={self.phase1_lr},\n"
            f"  phase2_lr={self.phase2_lr},\n"
            f"  validation_split={self.validation_split}\n"
            f")"
        )


def compute_class_weights(
    y_train: np.ndarray,
    num_classes: int
) -> Dict[int, float]:
    """
    Compute balanced class weights to handle imbalanced datasets.
    
    Args:
        y_train (np.ndarray): One-hot encoded training labels (N, num_classes)
        num_classes (int): Number of classes
        
    Returns:
        Dict[int, float]: Class weights {class_index: weight}
        
    Example:
        >>> class_weights = compute_class_weights(y_train, num_classes=26)
        >>> # Use in model.fit(..., class_weight=class_weights)
    """
    # Convert one-hot to class indices
    y_train_classes = np.argmax(y_train, axis=1)
    
    # Compute class weights
    classes = np.unique(y_train_classes)
    weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=y_train_classes
    )
    
    class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}
    
    logger.info("Class weights (for imbalanced dataset handling):")
    for cls in sorted(class_weight_dict.keys()):
        logger.info(f"  Class {cls}: {class_weight_dict[cls]:.4f}")
    
    return class_weight_dict


def create_callbacks(
    model_dir: str = 'model',
    log_dir: str = 'logs',
    early_stopping_patience: int = 15,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    phase: str = 'phase1'
) -> list:
    """
    Create training callbacks for monitoring and optimization.
    
    Callbacks include:
    - ModelCheckpoint: Save best model based on val_accuracy
    - TensorBoard: Real-time training visualization
    - EarlyStopping: Stop training when validation metric plateaus
    - ReduceLROnPlateau: Reduce learning rate when metric plateaus
    
    Args:
        model_dir (str): Directory to save model checkpoints
        log_dir (str): Directory for TensorBoard logs
        early_stopping_patience (int): Patience for early stopping
        reduce_lr_patience (int): Patience for LR reduction
        reduce_lr_factor (float): Factor to multiply LR by
        phase (str): 'phase1' or 'phase2' for naming
        
    Returns:
        list: List of Keras callbacks
    """
    # Create directories
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # ModelCheckpoint - saves best model based on validation accuracy
    checkpoint_path = os.path.join(
        model_dir,
        f'best_model_{phase}.h5'
    )
    model_checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )
    logger.info(f"ModelCheckpoint will save to: {checkpoint_path}")
    
    # TensorBoard - visualization in tensorboard
    tensorboard_log_dir = os.path.join(
        log_dir,
        f'{phase}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    tensorboard = TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    logger.info(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")
    
    # EarlyStopping - stop if validation accuracy doesn't improve
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    )
    logger.info(f"EarlyStopping patience set to: {early_stopping_patience} epochs")
    
    # ReduceLROnPlateau - reduce learning rate if metric plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=1e-7,
        verbose=1,
        mode='max'
    )
    logger.info(f"ReduceLROnPlateau factor: {reduce_lr_factor}, patience: {reduce_lr_patience}")
    
    return [model_checkpoint, tensorboard, early_stopping, reduce_lr]


def train_model_phase1(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainingConfig,
    class_weights: Optional[Dict[int, float]] = None,
) -> Tuple[tf.keras.Model, object]:
    """
    Phase 1: Train only the head layers (base model frozen for transfer learning).
    
    Args:
        model (tf.keras.Model): Compiled model
        X_train (np.ndarray): Training images
        y_train (np.ndarray): Training labels (one-hot)
        X_val (np.ndarray): Validation images
        y_val (np.ndarray): Validation labels (one-hot)
        config (TrainingConfig): Training configuration
        class_weights (Dict, optional): Class weights for imbalanced data
        
    Returns:
        Tuple[tf.keras.Model, object]: (trained_model, training_history)
    """
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: HEAD-ONLY TRAINING")
    logger.info("="*60)
    logger.info(f"Training for {config.phase1_epochs} epochs")
    logger.info(f"Learning rate: {config.phase1_lr}")
    logger.info(f"Batch size: {config.batch_size}")
    
    # Create callbacks
    callbacks = create_callbacks(
        early_stopping_patience=config.early_stopping_patience,
        reduce_lr_patience=config.reduce_lr_patience,
        reduce_lr_factor=config.reduce_lr_factor,
        phase='phase1'
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.phase1_epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    logger.info(f"Phase 1 training completed!")
    logger.info(f"Final train accuracy: {history.history['accuracy'][-1]:.4f}")
    logger.info(f"Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return model, history


def train_model_phase2(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainingConfig,
    num_layers_unfreeze: int = 30,
    class_weights: Optional[Dict[int, float]] = None,
) -> Tuple[tf.keras.Model, object]:
    """
    Phase 2: Fine-tune the full model with unfrozen base layers.
    
    Args:
        model (tf.keras.Model): Model from phase 1
        X_train (np.ndarray): Training images
        y_train (np.ndarray): Training labels (one-hot)
        X_val (np.ndarray): Validation images
        y_val (np.ndarray): Validation labels (one-hot)
        config (TrainingConfig): Training configuration
        num_layers_unfreeze (int): Number of layers from end to unfreeze
        class_weights (Dict, optional): Class weights for imbalanced data
        
    Returns:
        Tuple[tf.keras.Model, object]: (fine_tuned_model, training_history)
    """
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: FINE-TUNING")
    logger.info("="*60)
    
    # Unfreeze layers
    for layer in model.layers[-num_layers_unfreeze:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.phase2_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Unfroze last {num_layers_unfreeze} layers for fine-tuning")
    logger.info(f"Training for {config.phase2_epochs} epochs")
    logger.info(f"Learning rate: {config.phase2_lr}")
    logger.info(f"Batch size: {config.batch_size}")
    
    # Create callbacks
    callbacks = create_callbacks(
        early_stopping_patience=config.early_stopping_patience,
        reduce_lr_patience=config.reduce_lr_patience,
        reduce_lr_factor=config.reduce_lr_factor,
        phase='phase2'
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.phase2_epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    logger.info(f"Phase 2 fine-tuning completed!")
    logger.info(f"Final train accuracy: {history.history['accuracy'][-1]:.4f}")
    logger.info(f"Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return model, history


def train_custom_cnn(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainingConfig,
    class_weights: Optional[Dict[int, float]] = None,
) -> Tuple[tf.keras.Model, object]:
    """
    Train custom CNN model (single phase).
    
    Args:
        model (tf.keras.Model): Compiled model
        X_train (np.ndarray): Training images
        y_train (np.ndarray): Training labels (one-hot)
        X_val (np.ndarray): Validation images
        y_val (np.ndarray): Validation labels (one-hot)
        config (TrainingConfig): Training configuration
        class_weights (Dict, optional): Class weights for imbalanced data
        
    Returns:
        Tuple[tf.keras.Model, object]: (trained_model, training_history)
    """
    logger.info("\n" + "="*60)
    logger.info("TRAINING CUSTOM CNN")
    logger.info("="*60)
    logger.info(f"Training for {config.phase1_epochs} epochs")
    logger.info(f"Learning rate: {config.phase1_lr}")
    logger.info(f"Batch size: {config.batch_size}")
    
    # Create callbacks
    callbacks = create_callbacks(
        early_stopping_patience=config.early_stopping_patience,
        reduce_lr_patience=config.reduce_lr_patience,
        reduce_lr_factor=config.reduce_lr_factor,
        phase='custom_cnn'
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.phase1_epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    logger.info(f"Training completed!")
    logger.info(f"Final train accuracy: {history.history['accuracy'][-1]:.4f}")
    logger.info(f"Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return model, history


def save_model(
    model: tf.keras.Model,
    save_path: str = 'model/cnn_model.h5'
) -> None:
    """
    Save trained model to disk.
    
    Args:
        model (tf.keras.Model): Model to save
        save_path (str): Path to save H5 model file
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    logger.info(f"Model saved to: {save_path}")


def prepare_training_data(
    X: np.ndarray,
    y: np.ndarray,
    validation_split: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and validation sets.
    
    Args:
        X (np.ndarray): Features
        y (np.ndarray): Labels (one-hot or class indices)
        validation_split (float): Validation set ratio
        random_state (int): Random seed
        
    Returns:
        Tuple: (X_train, X_val, y_train, y_val)
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=validation_split,
        random_state=random_state,
        stratify=np.argmax(y, axis=1) if len(y.shape) > 1 and y.shape[1] > 1 else y
    )
    
    logger.info(f"Data split:")
    logger.info(f"  - Training: {X_train.shape[0]} samples")
    logger.info(f"  - Validation: {X_val.shape[0]} samples")
    
    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    # Example usage
    print("Training module loaded successfully!")
    print("\nExample usage:")
    print("""
    from train import TrainingConfig, compute_class_weights, train_custom_cnn
    from utils.preprocessing import load_sibi_dataset
    from utils.model_builder import build_model, compile_model
    
    # Load data
    X, y, label_dict = load_sibi_dataset('dataset/SIBI/')
    
    # Build model
    model, _ = build_model('custom', num_classes=26)
    model = compile_model(model)
    
    # Compute class weights
    class_weights = compute_class_weights(y, num_classes=26)
    
    # Configure training
    config = TrainingConfig(
        phase1_epochs=100,
        batch_size=32,
        phase1_lr=0.001,
    )
    
    # Train
    model, history = train_custom_cnn(
        model, X_train, y_train, X_val, y_val,
        config, class_weights=class_weights
    )
    """)
