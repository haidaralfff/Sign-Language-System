"""
Dataset preprocessing, loading, and validation utilities for SIBI gesture recognition.

This module provides functions for:
- Dataset validation and quality checks
- Image loading and preprocessing with CLAHE and Gaussian blur
- Stratified sampling to balance the dataset
- Grayscale and RGB mode support
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_dataset(base_path: str) -> Dict:
    """
    Validate dataset quality and integrity.
    
    Checks for:
    - Count of images per class
    - Corrupted or unreadable image files
    - Class imbalance (warns if max/min ratio > 3x)
    - Missing classes
    
    Args:
        base_path (str): Path to dataset root directory (e.g., 'dataset/SIBI/')
        
    Returns:
        Dict: Summary dictionary containing:
            - 'class_counts': {class_name: image_count}
            - 'corrupted_files': List of corrupted file paths
            - 'is_imbalanced': Boolean indicating class imbalance
            - 'imbalance_ratio': max_count / min_count
            - 'total_images': Total valid images
            - 'total_classes': Number of classes found
            - 'summary': Human-readable summary string
            
    Example:
        >>> validation = validate_dataset('dataset/SIBI/')
        >>> print(validation['summary'])
        >>> if validation['is_imbalanced']:
        ...     print(f"Dataset imbalanced! Ratio: {validation['imbalance_ratio']:.2f}x")
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        logger.error(f"Dataset path does not exist: {base_path}")
        raise FileNotFoundError(f"Dataset path not found: {base_path}")
    
    class_counts = {}
    corrupted_files = []
    total_images = 0
    
    logger.info(f"Starting dataset validation at: {base_path}")
    
    # Iterate through all class folders (A-Z)
    for class_folder in sorted(base_path.iterdir()):
        if not class_folder.is_dir():
            continue
            
        class_name = class_folder.name
        image_count = 0
        
        # Check all image files in class folder
        for img_file in class_folder.glob('*'):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                continue
            
            # Try to read image to detect corruption
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    corrupted_files.append(str(img_file))
                    logger.warning(f"Corrupted image: {img_file}")
                else:
                    image_count += 1
                    total_images += 1
            except Exception as e:
                corrupted_files.append(str(img_file))
                logger.warning(f"Error reading {img_file}: {str(e)}")
        
        if image_count > 0:
            class_counts[class_name] = image_count
            logger.info(f"Class {class_name}: {image_count} valid images")
        else:
            logger.warning(f"Class {class_name}: No valid images found!")
    
    # Check for class imbalance
    if class_counts:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        is_imbalanced = imbalance_ratio > 3.0
    else:
        is_imbalanced = False
        imbalance_ratio = 0.0
    
    # Identify low-count classes
    low_count_classes = [
        (cls, count) for cls, count in class_counts.items() 
        if count < 100
    ]
    
    # Generate summary
    summary_lines = [
        f"\n{'='*60}",
        f"DATASET VALIDATION SUMMARY",
        f"{'='*60}",
        f"Total Classes Found: {len(class_counts)}",
        f"Total Valid Images: {total_images}",
        f"Corrupted/Unreadable Files: {len(corrupted_files)}",
        f"Dataset Imbalanced: {'YES' if is_imbalanced else 'NO'}",
        f"Imbalance Ratio (max/min): {imbalance_ratio:.2f}x",
    ]
    
    if low_count_classes:
        summary_lines.append(f"\n⚠️  Classes with < 100 images:")
        for cls, count in sorted(low_count_classes, key=lambda x: x[1]):
            summary_lines.append(f"   - {cls}: {count} images")
    
    if corrupted_files:
        summary_lines.append(f"\n⚠️  Corrupted Files ({len(corrupted_files)}):")
        for file in corrupted_files[:5]:  # Show first 5
            summary_lines.append(f"   - {file}")
        if len(corrupted_files) > 5:
            summary_lines.append(f"   ... and {len(corrupted_files) - 5} more")
    
    if is_imbalanced:
        summary_lines.append(f"\n💡 Recommendation: Use class_weight='balanced' during training")
    
    summary_lines.append(f"{'='*60}\n")
    summary = '\n'.join(summary_lines)
    
    logger.info(summary)
    
    return {
        'class_counts': class_counts,
        'corrupted_files': corrupted_files,
        'is_imbalanced': is_imbalanced,
        'imbalance_ratio': imbalance_ratio,
        'total_images': total_images,
        'total_classes': len(class_counts),
        'low_count_classes': low_count_classes,
        'summary': summary
    }


def load_sibi_dataset(
    base_path: str,
    img_size: int = 64,
    grayscale: bool = False,
    max_per_class: int = 500,
    apply_clahe: bool = True,
    apply_blur: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Load and preprocess SIBI dataset with stratified sampling.
    
    Implements:
    - Grayscale and RGB mode support
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Gaussian blur for noise reduction
    - Stratified sampling (max_per_class) to balance dataset
    - Image normalization to [0, 1] range
    
    Args:
        base_path (str): Path to dataset root (e.g., 'dataset/SIBI/')
        img_size (int): Target image size (square, default 64)
        grayscale (bool): Convert to grayscale if True, else RGB (default False)
        max_per_class (int): Maximum images per class for stratified sampling (default 500)
        apply_clahe (bool): Apply CLAHE for contrast enhancement (default True)
        apply_blur (bool): Apply Gaussian blur for noise reduction (default True)
        
    Returns:
        Tuple[np.ndarray, np.ndarray, Dict]: 
            - X: Image array, shape (N, img_size, img_size, channels)
              channels=1 if grayscale, 3 if RGB
            - y: One-hot encoded labels, shape (N, num_classes)
            - label_dict: Dict mapping class names to indices
            
    Example:
        >>> X, y, label_dict = load_sibi_dataset(
        ...     'dataset/SIBI/',
        ...     img_size=64,
        ...     grayscale=False,
        ...     max_per_class=500
        ... )
        >>> print(X.shape, y.shape)
        >>> print(f"Classes: {len(label_dict)}")
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {base_path}")
    
    images = []
    labels = []
    label_dict = {}
    class_idx = 0
    samples_per_class = defaultdict(int)
    
    logger.info(f"Loading dataset from: {base_path}")
    logger.info(f"Config - Size: {img_size}x{img_size}, Grayscale: {grayscale}, "
                f"Max per class: {max_per_class}, CLAHE: {apply_clahe}, Blur: {apply_blur}")
    
    # Create CLAHE object for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if apply_clahe else None
    
    # Iterate through class folders
    for class_folder in sorted(base_path.iterdir()):
        if not class_folder.is_dir():
            continue
        
        class_name = class_folder.name
        label_dict[class_name] = class_idx
        
        logger.info(f"Loading class '{class_name}' (index: {class_idx})")
        
        # Load images for this class
        for img_file in sorted(class_folder.glob('*')):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                continue
            
            # Check stratified sampling limit
            if samples_per_class[class_name] >= max_per_class:
                logger.debug(f"Reached max_per_class limit ({max_per_class}) for {class_name}")
                continue
            
            try:
                # Read image
                img = cv2.imread(str(img_file))
                if img is None:
                    logger.warning(f"Failed to read: {img_file}")
                    continue
                
                # Convert to grayscale if requested
                if grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize image
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                
                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                if apply_clahe:
                    if grayscale:
                        img = clahe.apply(img)
                    else:
                        # Apply to each channel separately
                        for i in range(3):
                            img[:, :, i] = clahe.apply(img[:, :, i])
                
                # Apply Gaussian blur for noise reduction
                if apply_blur:
                    img = cv2.GaussianBlur(img, (3, 3), 0)
                
                # Normalize to [0, 1]
                img = img.astype(np.float32) / 255.0
                
                # Reshape for grayscale
                if grayscale:
                    img = np.expand_dims(img, axis=-1)
                
                images.append(img)
                labels.append(class_idx)
                samples_per_class[class_name] += 1
                
            except Exception as e:
                logger.warning(f"Error processing {img_file}: {str(e)}")
                continue
        
        logger.info(f"  → Loaded {samples_per_class[class_name]} images for class '{class_name}'")
        class_idx += 1
    
    if not images:
        raise ValueError("No images loaded from dataset. Check dataset path and image files.")
    
    # Convert to numpy arrays
    X = np.array(images, dtype=np.float32)
    
    # One-hot encode labels
    from tensorflow.keras.utils import to_categorical
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    y = to_categorical(labels, num_classes=num_classes)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset Loading Complete!")
    logger.info(f"  - Total Images: {len(images)}")
    logger.info(f"  - Total Classes: {num_classes}")
    logger.info(f"  - Image Shape: {X.shape}")
    logger.info(f"  - Label Shape: {y.shape}")
    logger.info(f"{'='*60}\n")
    
    return X, y, label_dict


def apply_data_augmentation(
    X_train: np.ndarray,
    rotation_range: float = 15,
    shift_range: float = 0.1,
    shear_range: float = 0.2,
    zoom_range: float = 0.2,
    horizontal_flip: bool = True,
    fill_mode: str = 'nearest'
) -> object:
    """
    Create an ImageDataGenerator for data augmentation.
    
    Args:
        X_train (np.ndarray): Training images
        rotation_range (float): Rotation range in degrees (default 15)
        shift_range (float): Shift range as fraction of width/height (default 0.1)
        shear_range (float): Shear range (default 0.2)
        zoom_range (float): Zoom range (default 0.2)
        horizontal_flip (bool): Enable horizontal flip (default True)
        fill_mode (str): Filling mode for new pixels (default 'nearest')
        
    Returns:
        ImageDataGenerator: Configured augmentation generator
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    augment_generator = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=shift_range,
        height_shift_range=shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode
    )
    
    logger.info("Data augmentation generator created with:")
    logger.info(f"  - Rotation: ±{rotation_range}°")
    logger.info(f"  - Shift: ±{shift_range*100:.0f}%")
    logger.info(f"  - Shear: {shear_range}")
    logger.info(f"  - Zoom: {zoom_range}")
    logger.info(f"  - Horizontal flip: {horizontal_flip}")
    
    return augment_generator


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Set seeds for reproducibility
    np.random.seed(42)
    
    # Validate dataset
    dataset_path = "dataset/SIBI/"
    
    try:
        validation_result = validate_dataset(dataset_path)
        print(validation_result['summary'])
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Load dataset (if validation passed)
    try:
        X, y, label_dict = load_sibi_dataset(
            dataset_path,
            img_size=64,
            grayscale=False,
            max_per_class=500,
            apply_clahe=True,
            apply_blur=True
        )
        print(f"Dataset loaded successfully!")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"Classes: {list(label_dict.keys())}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
