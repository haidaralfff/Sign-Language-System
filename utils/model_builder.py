"""
Model architecture definitions for SIBI gesture recognition.

Provides:
- Custom CNN architecture (enhanced with GlobalAveragePooling2D and SpatialDropout2D)
- MobileNetV2 transfer learning architecture with fine-tuning support
- Model compilation and configuration utilities
"""

import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, Sequential
from tensorflow.keras.applications import MobileNetV2
from typing import Tuple, Optional
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_custom_cnn(
    num_classes: int = 26,
    img_size: int = 64,
    input_channels: int = 3,
    l2_reg: float = 0.001,
    dropout_rate: float = 0.3,
    spatial_dropout_rate: float = 0.25
) -> keras.Model:
    """
    Build enhanced custom CNN model for gesture recognition.
    
    Architecture:
    - Conv Block 1: Conv2D(32) → BatchNorm → MaxPool → SpatialDropout2D
    - Conv Block 2: Conv2D(64) → BatchNorm → MaxPool → SpatialDropout2D
    - Conv Block 3: Conv2D(128) → BatchNorm → MaxPool → SpatialDropout2D
    - Conv Block 4: Conv2D(256) → BatchNorm → MaxPool → SpatialDropout2D
    - GlobalAveragePooling2D (reduces spatial dims, prevents overfitting)
    - Dense: 256 (ReLU, BatchNorm, Dropout)
    - Dense: 128 (ReLU, BatchNorm, Dropout)
    - Output: Dense(num_classes, softmax)
    
    Args:
        num_classes (int): Number of output classes (default 26 for A-Z)
        img_size (int): Input image size (square, default 64)
        input_channels (int): Number of input channels (1 for grayscale, 3 for RGB, default 3)
        l2_reg (float): L2 regularization factor (default 0.001)
        dropout_rate (float): Dropout rate for dense layers (default 0.3)
        spatial_dropout_rate (float): SpatialDropout2D rate (default 0.25)
        
    Returns:
        keras.Model: Compiled Keras Sequential model
        
    Example:
        >>> model = build_custom_cnn(num_classes=26, img_size=64)
        >>> model.summary()
        >>> # Total params: ~1.7M
    """
    model = Sequential([
        # Input layer
        layers.Input(shape=(img_size, img_size, input_channels)),
        
        # Conv Block 1: 32 filters
        layers.Conv2D(
            32, 
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name='conv2d_1'
        ),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.MaxPooling2D(pool_size=(2, 2), name='max_pool_1'),
        layers.SpatialDropout2D(spatial_dropout_rate, name='spatial_dropout_1'),
        
        # Conv Block 2: 64 filters
        layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name='conv2d_2'
        ),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='max_pool_2'),
        layers.SpatialDropout2D(spatial_dropout_rate, name='spatial_dropout_2'),
        
        # Conv Block 3: 128 filters
        layers.Conv2D(
            128,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name='conv2d_3'
        ),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.MaxPooling2D(pool_size=(2, 2), name='max_pool_3'),
        layers.SpatialDropout2D(spatial_dropout_rate, name='spatial_dropout_3'),
        
        # Conv Block 4: 256 filters (NEW)
        layers.Conv2D(
            256,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name='conv2d_4'
        ),
        layers.BatchNormalization(name='batch_norm_4'),
        layers.MaxPooling2D(pool_size=(2, 2), name='max_pool_4'),
        layers.SpatialDropout2D(spatial_dropout_rate, name='spatial_dropout_4'),
        
        # Global Average Pooling (better than Flatten for preventing overfitting)
        layers.GlobalAveragePooling2D(name='global_avg_pool'),
        
        # Dense head
        layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name='dense_1'
        ),
        layers.BatchNormalization(name='batch_norm_5'),
        layers.Dropout(dropout_rate, name='dropout_1'),
        
        layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name='dense_2'
        ),
        layers.BatchNormalization(name='batch_norm_6'),
        layers.Dropout(dropout_rate, name='dropout_2'),
        
        # Output layer
        layers.Dense(
            num_classes,
            activation='softmax',
            name='output'
        ),
    ])
    
    logger.info("Custom CNN model built:")
    logger.info(f"  - Input shape: ({img_size}, {img_size}, {input_channels})")
    logger.info(f"  - Output classes: {num_classes}")
    logger.info(f"  - L2 Regularization: {l2_reg}")
    logger.info(f"  - Spatial Dropout: {spatial_dropout_rate}")
    logger.info(f"  - Dense Dropout: {dropout_rate}")
    
    return model


def build_mobilenetv2_transfer(
    num_classes: int = 26,
    img_size: int = 128,
    input_channels: int = 3,
    freeze_base: bool = True,
    base_dropout: float = 0.5,
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Build MobileNetV2 transfer learning model for gesture recognition.
    
    Architecture:
    - MobileNetV2 base (ImageNet pre-trained, frozen or trainable)
    - GlobalAveragePooling2D
    - Dense(256, relu) → BatchNorm → Dropout(0.5)
    - Dense(num_classes, softmax)
    
    Supports two-phase training:
    1. Phase 1: Train head only (freeze_base=True, ~50 epochs)
    2. Phase 2: Fine-tune full model (freeze_base=False, ~30 epochs, lr/10)
    
    Args:
        num_classes (int): Number of output classes (default 26)
        img_size (int): Input image size (square, default 128 for MobileNetV2)
        input_channels (int): Number of input channels (1 or 3, default 3)
        freeze_base (bool): Freeze base model during initial training (default True)
        base_dropout (float): Dropout rate in dense layers (default 0.5)
        learning_rate (float): Initial learning rate (default 0.001)
        
    Returns:
        keras.Model: Compiled Keras Model with transfer learning
        
    Example:
        >>> model = build_mobilenetv2_transfer(num_classes=26, img_size=128)
        >>> model.summary()
        >>> # Phase 1: Train 50 epochs with frozen base
        >>> # Phase 2: Unfreeze, set lr to 1e-5, train 30 more epochs
    """
    # Validate input channels
    if input_channels not in [1, 3]:
        logger.warning(f"MobileNetV2 expects 3 channels. Got {input_channels}. "
                      "Converting grayscale to RGB by duplicating channels.")
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    if freeze_base:
        base_model.trainable = False
        logger.info("MobileNetV2 base model layers frozen for phase 1 training")
    else:
        logger.info("MobileNetV2 base model layers trainable for fine-tuning")
    
    # Build model
    model = Sequential([
        layers.Input(shape=(img_size, img_size, input_channels)),
        
        # If grayscale, convert to RGB by repeating channels
        layers.Lambda(
            lambda x: tf.repeat(x, 3, axis=-1) if input_channels == 1 else x,
            name='channel_expand'
        ) if input_channels == 1 else layers.Lambda(lambda x: x),
        
        # MobileNetV2 base
        base_model,
        
        # Global average pooling
        layers.GlobalAveragePooling2D(name='global_avg_pool'),
        
        # Dense head
        layers.Dense(
            256,
            activation='relu',
            name='dense_1'
        ),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(base_dropout, name='dropout_1'),
        
        # Output layer
        layers.Dense(
            num_classes,
            activation='softmax',
            name='output'
        ),
    ])
    
    logger.info("MobileNetV2 transfer learning model built:")
    logger.info(f"  - Input shape: ({img_size}, {img_size}, {input_channels})")
    logger.info(f"  - Output classes: {num_classes}")
    logger.info(f"  - Base model frozen: {freeze_base}")
    logger.info(f"  - Dense dropout: {base_dropout}")
    
    return model, base_model


def build_model(
    model_type: str = 'custom',
    num_classes: int = 26,
    img_size: int = 64,
    input_channels: int = 3,
    **kwargs
) -> Tuple[keras.Model, Optional[keras.Model]]:
    """
    Factory function to build model based on type.
    
    Args:
        model_type (str): 'custom' or 'mobilenetv2' (default 'custom')
        num_classes (int): Number of classes (default 26)
        img_size (int): Input image size (default 64 for custom, 128 for MobileNetV2)
        input_channels (int): Number of input channels (default 3)
        **kwargs: Additional arguments passed to specific builder
        
    Returns:
        Tuple[keras.Model, Optional[keras.Model]]:
            - (model, None) for custom CNN
            - (model, base_model) for MobileNetV2
            
    Example:
        >>> # Custom CNN
        >>> model, _ = build_model('custom', num_classes=26)
        >>> 
        >>> # MobileNetV2
        >>> model, base_model = build_model('mobilenetv2', num_classes=26)
    """
    if model_type.lower() == 'custom':
        model = build_custom_cnn(
            num_classes=num_classes,
            img_size=img_size,
            input_channels=input_channels,
            **kwargs
        )
        return model, None
        
    elif model_type.lower() == 'mobilenetv2':
        # Default img_size for MobileNetV2 is 128
        if img_size == 64:
            logger.info("Adjusting img_size to 128 for MobileNetV2 (optimal input size)")
            img_size = 128
        
        model, base_model = build_mobilenetv2_transfer(
            num_classes=num_classes,
            img_size=img_size,
            input_channels=input_channels,
            **kwargs
        )
        return model, base_model
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                        f"Choose 'custom' or 'mobilenetv2'")


def compile_model(
    model: keras.Model,
    optimizer_type: str = 'adam',
    learning_rate: float = 0.001,
    loss: str = 'categorical_crossentropy',
    metrics: list = None
) -> keras.Model:
    """
    Compile model with optimizer, loss, and metrics.
    
    Args:
        model (keras.Model): Model to compile
        optimizer_type (str): 'adam', 'sgd', 'rmsprop' (default 'adam')
        learning_rate (float): Learning rate (default 0.001)
        loss (str): Loss function (default 'categorical_crossentropy')
        metrics (list): Evaluation metrics (default ['accuracy'])
        
    Returns:
        keras.Model: Compiled model
    """
    if metrics is None:
        metrics = ['accuracy']
    
    # Create optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_type.lower() == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    logger.info(f"Model compiled:")
    logger.info(f"  - Optimizer: {optimizer_type} (lr={learning_rate})")
    logger.info(f"  - Loss: {loss}")
    logger.info(f"  - Metrics: {metrics}")
    
    return model


def get_model_summary(model: keras.Model) -> str:
    """
    Get detailed model summary as string.
    
    Args:
        model (keras.Model): Model to summarize
        
    Returns:
        str: Model summary text
    """
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    return '\n'.join(summary_lines)


def count_trainable_parameters(model: keras.Model) -> Tuple[int, int]:
    """
    Count trainable and total parameters in model.
    
    Args:
        model (keras.Model): Model to analyze
        
    Returns:
        Tuple[int, int]: (trainable_params, total_params)
    """
    trainable_count = sum([tf.keras.backend.count_params(w) 
                          for w in model.trainable_weights])
    total_count = sum([tf.keras.backend.count_params(w) 
                      for w in model.weights])
    
    return trainable_count, total_count


def freeze_base_model(model: keras.Model, num_layers_to_freeze: int = None) -> keras.Model:
    """
    Freeze specific layers in model for transfer learning.
    
    Args:
        model (keras.Model): Model to modify
        num_layers_to_freeze (int): Number of layers to freeze from start
                                   If None, freeze all except last layer
                                   
    Returns:
        keras.Model: Model with frozen layers
    """
    if num_layers_to_freeze is None:
        # Freeze all but last layer
        num_layers_to_freeze = len(model.layers) - 1
    
    for layer in model.layers[:num_layers_to_freeze]:
        layer.trainable = False
    
    trainable, total = count_trainable_parameters(model)
    logger.info(f"Froze {num_layers_to_freeze} layers. "
               f"Trainable params: {trainable:,} / {total:,}")
    
    return model


def unfreeze_model(model: keras.Model) -> keras.Model:
    """
    Unfreeze all layers in model for fine-tuning.
    
    Args:
        model (keras.Model): Model to modify
        
    Returns:
        keras.Model: Model with all layers trainable
    """
    for layer in model.layers:
        layer.trainable = True
    
    trainable, total = count_trainable_parameters(model)
    logger.info(f"Unfroze all layers. Trainable params: {trainable:,} / {total:,}")
    
    return model


if __name__ == "__main__":
    import sys
    
    # Set seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("\n" + "="*60)
    print("CUSTOM CNN MODEL")
    print("="*60)
    
    # Build and summarize custom CNN
    model_cnn, _ = build_model('custom', num_classes=26, img_size=64, input_channels=3)
    compile_model(model_cnn)
    trainable, total = count_trainable_parameters(model_cnn)
    print(f"\nTotal params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    
    print("\n" + "="*60)
    print("MOBILENETV2 TRANSFER LEARNING MODEL")
    print("="*60)
    
    # Build and summarize MobileNetV2
    model_mobile, base_model = build_model(
        'mobilenetv2',
        num_classes=26,
        img_size=128,
        input_channels=3,
        freeze_base=True
    )
    compile_model(model_mobile, learning_rate=0.001)
    trainable, total = count_trainable_parameters(model_mobile)
    print(f"\nTotal params: {total:,}")
    print(f"Trainable params: {trainable:,} (base frozen)")
    
    print("\n" + "="*60)
