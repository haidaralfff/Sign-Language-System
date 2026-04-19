"""
Utilities package for SIBI gesture recognition system.

Modules:
- preprocessing: Dataset loading, validation, and preprocessing
- model_builder: CNN and transfer learning model architectures
- visualization: Plotting, evaluation visualization, and Grad-CAM
"""

from .preprocessing import validate_dataset, load_sibi_dataset, apply_data_augmentation
from .model_builder import (
    build_custom_cnn,
    build_mobilenetv2_transfer,
    build_model,
    compile_model,
    count_trainable_parameters,
    freeze_base_model,
    unfreeze_model,
)
from .visualization import GradCAM, generate_gradcam_batch, generate_gradcam_comparison

__all__ = [
    'validate_dataset',
    'load_sibi_dataset',
    'apply_data_augmentation',
    'build_custom_cnn',
    'build_mobilenetv2_transfer',
    'build_model',
    'compile_model',
    'count_trainable_parameters',
    'freeze_base_model',
    'unfreeze_model',
    'GradCAM',
    'generate_gradcam_batch',
    'generate_gradcam_comparison',
]
