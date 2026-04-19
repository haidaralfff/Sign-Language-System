"""
Grad-CAM (Gradient-weighted Class Activation Mapping) visualization.

Implements:
- Gradient-weighted class activation mapping
- Heatmap overlay on original images
- Per-class Grad-CAM visualization
- Batch processing with automatic layer detection
"""

import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GradCAM:
    """Grad-CAM implementation for CNN models."""
    
    def __init__(
        self,
        model: tf.keras.Model,
        layer_name: Optional[str] = None,
        label_dict: Optional[Dict] = None
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model (tf.keras.Model): Trained model
            layer_name (str, optional): Name of convolutional layer for Grad-CAM
                                       If None, automatically selects last Conv2D layer
            label_dict (Dict, optional): Class name to index mapping
        """
        self.model = model
        self.label_dict = label_dict or {i: f"Class_{i}" for i in range(26)}
        self.idx_to_class = {v: k for k, v in self.label_dict.items()}
        
        # Auto-detect layer if not specified
        if layer_name is None:
            layer_name = self._find_last_conv_layer()
        
        self.layer_name = layer_name
        self.conv_layer = self.model.get_layer(layer_name)
        
        logger.info(f"Grad-CAM initialized:")
        logger.info(f"  - Target layer: {layer_name}")
        logger.info(f"  - Layer output shape: {self.conv_layer.output.shape}")
    
    def _find_last_conv_layer(self) -> str:
        """Find last Conv2D layer in model."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                logger.info(f"Auto-detected Conv2D layer: {layer.name}")
                return layer.name
        
        raise ValueError("No Conv2D layer found in model")
    
    def generate(
        self,
        img_array: np.ndarray,
        class_idx: int,
        eps: float = 1e-8
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for an image and class.
        
        Args:
            img_array (np.ndarray): Input image, shape (H, W, C) or (1, H, W, C)
            class_idx (int): Target class index
            eps (float): Small value to avoid division by zero
            
        Returns:
            np.ndarray: Grad-CAM heatmap, shape (H, W)
        """
        # Ensure 4D input
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        # Build gradient model
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.conv_layer.output, self.model.output]
        )
        
        # Compute gradients with respect to input
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]
        
        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Pool gradients across spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Compute Grad-CAM
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(
            tf.multiply(pooled_grads, conv_outputs),
            axis=-1
        ).numpy()
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + eps)
        
        return heatmap
    
    def overlay_heatmap(
        self,
        img_array: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image.
        
        Args:
            img_array (np.ndarray): Original image (0-1 float or 0-255 uint8), shape (H, W, C)
            heatmap (np.ndarray): Grad-CAM heatmap, shape (H, W)
            alpha (float): Blending alpha (0-1)
            colormap (int): OpenCV colormap ID
            
        Returns:
            np.ndarray: Overlaid image, shape (H, W, 3), uint8
        """
        # Convert image to uint8 if needed
        if img_array.dtype == np.float32 or img_array.dtype == np.float64:
            img_uint8 = (img_array * 255).astype(np.uint8)
        else:
            img_uint8 = img_array.astype(np.uint8)
        
        # Handle grayscale
        if len(img_uint8.shape) == 2:
            img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        elif img_uint8.shape[2] == 1:
            img_uint8 = cv2.cvtColor(img_uint8[:, :, 0], cv2.COLOR_GRAY2BGR)
        
        # Resize heatmap to match image
        h, w = img_uint8.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            colormap
        )
        
        # Blend
        overlaid = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlaid
    
    def visualize_class(
        self,
        img_array: np.ndarray,
        class_idx: int,
        alpha: float = 0.5,
        figsize: Tuple[int, int] = (15, 5)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Visualize Grad-CAM for a specific class.
        
        Args:
            img_array (np.ndarray): Input image
            class_idx (int): Target class index
            alpha (float): Blending alpha
            figsize (Tuple): Figure size for matplotlib
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - original_img: Original image
                - heatmap: Grad-CAM heatmap
                - overlaid: Overlaid image
        """
        # Generate heatmap
        heatmap = self.generate(img_array, class_idx)
        
        # Ensure image is 3D
        if len(img_array.shape) == 4:
            img_display = img_array[0]
        else:
            img_display = img_array
        
        # Overlay
        overlaid = self.overlay_heatmap(img_display, heatmap, alpha=alpha)
        
        return img_display, heatmap, overlaid


def generate_gradcam_batch(
    model: tf.keras.Model,
    images: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    label_dict: Optional[Dict] = None,
    layer_name: Optional[str] = None,
    results_dir: str = 'results/gradcam',
    num_samples: int = 3,
    save_plots: bool = True
) -> None:
    """
    Generate Grad-CAM visualizations for batch of images.
    
    Args:
        model (tf.keras.Model): Trained model
        images (np.ndarray): Input images, shape (N, H, W, C)
        true_labels (np.ndarray, optional): True class indices
        label_dict (Dict, optional): Class name to index mapping
        layer_name (str, optional): Target conv layer name
        results_dir (str): Directory to save visualizations
        num_samples (int): Number of samples per class to visualize
        save_plots (bool): Whether to save plots to disk
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    if label_dict is None:
        label_dict = {i: f"Class_{i}" for i in range(26)}
    
    idx_to_class = {v: k for k, v in label_dict.items()}
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, layer_name=layer_name, label_dict=label_dict)
    
    logger.info(f"Generating Grad-CAM for {len(images)} images...")
    
    if true_labels is not None:
        # Group images by class
        for class_idx in range(len(label_dict)):
            class_name = idx_to_class[class_idx]
            class_mask = np.argmax(true_labels, axis=1) == class_idx if len(true_labels.shape) > 1 else true_labels == class_idx
            class_images = images[class_mask]
            
            if len(class_images) == 0:
                logger.warning(f"No images found for class {class_name}")
                continue
            
            logger.info(f"Processing class: {class_name} ({len(class_images)} images)")
            
            # Process up to num_samples images per class
            for sample_idx in range(min(num_samples, len(class_images))):
                img = class_images[sample_idx]
                
                # Generate Grad-CAM
                img_display, heatmap, overlaid = gradcam.visualize_class(img, class_idx)
                
                # Plot
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original
                if len(img_display.shape) == 3 and img_display.shape[2] == 3:
                    axes[0].imshow(img_display.astype(np.uint8))
                else:
                    axes[0].imshow(img_display.astype(np.uint8), cmap='gray')
                axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
                axes[0].axis('off')
                
                # Heatmap
                axes[1].imshow(heatmap, cmap='jet')
                axes[1].set_title(f'Grad-CAM Heatmap\n({class_name})', fontsize=12, fontweight='bold')
                axes[1].axis('off')
                
                # Overlaid
                axes[2].imshow(overlaid)
                axes[2].set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')
                axes[2].axis('off')
                
                plt.suptitle(f'Class: {class_name}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                if save_plots:
                    save_path = Path(results_dir) / f'{class_name}_sample_{sample_idx}.png'
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    logger.info(f"  Saved: {save_path}")
                
                plt.close()
    else:
        # Process all images
        for idx in range(min(10, len(images))):
            img = images[idx]
            
            # Predict
            pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
            pred_class_idx = np.argmax(pred)
            
            # Generate Grad-CAM
            img_display, heatmap, overlaid = gradcam.visualize_class(img, pred_class_idx)
            
            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            if len(img_display.shape) == 3 and img_display.shape[2] == 3:
                axes[0].imshow(img_display.astype(np.uint8))
            else:
                axes[0].imshow(img_display.astype(np.uint8), cmap='gray')
            axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title(f'Grad-CAM Heatmap', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
            axes[2].imshow(overlaid)
            axes[2].set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')
            axes[2].axis('off')
            
            pred_class = idx_to_class.get(pred_class_idx, f"Class_{pred_class_idx}")
            plt.suptitle(f'Image {idx} - Predicted: {pred_class} ({pred[pred_class_idx]*100:.1f}%)',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_plots:
                save_path = Path(results_dir) / f'image_{idx}_gradcam.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved: {save_path}")
            
            plt.close()
    
    logger.info(f"Grad-CAM visualization complete! Results saved to: {results_dir}")


def generate_gradcam_comparison(
    model: tf.keras.Model,
    img_array: np.ndarray,
    label_dict: Dict[str, int],
    layer_name: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 12),
    alpha: float = 0.5
) -> None:
    """
    Generate Grad-CAM visualizations for all classes (comparison).
    
    Args:
        model (tf.keras.Model): Trained model
        img_array (np.ndarray): Input image (H, W, C)
        label_dict (Dict): Class name to index mapping
        layer_name (str, optional): Target conv layer
        figsize (Tuple): Figure size
        alpha (float): Blending alpha
    """
    idx_to_class = {v: k for k, v in label_dict.items()}
    gradcam = GradCAM(model, layer_name=layer_name, label_dict=label_dict)
    
    # Get prediction
    pred_probs = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
    pred_class = np.argmax(pred_probs)
    
    # Create grid for all classes
    num_classes = len(label_dict)
    num_cols = 4
    num_rows = (num_classes + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    
    for class_idx in range(num_classes):
        ax = axes[class_idx]
        
        # Generate Grad-CAM
        _, heatmap, overlaid = gradcam.visualize_class(img_array, class_idx, alpha=alpha)
        
        # Display
        ax.imshow(overlaid)
        
        # Title with confidence
        class_name = idx_to_class[class_idx]
        confidence = pred_probs[class_idx] * 100
        
        if class_idx == pred_class:
            ax.set_title(f'{class_name}\n({confidence:.1f}%)\n✓ PREDICTED',
                        fontsize=10, fontweight='bold', color='green')
        else:
            ax.set_title(f'{class_name}\n({confidence:.1f}%)',
                        fontsize=10, fontweight='bold')
        
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_classes, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Grad-CAM module loaded successfully!")
    print("\nExample usage:")
    print("""
    from utils.visualization import generate_gradcam_batch, GradCAM
    
    # Generate Grad-CAM for batch
    generate_gradcam_batch(
        model, X_val,
        true_labels=y_val,
        label_dict=label_dict,
        results_dir='results/gradcam',
        num_samples=2
    )
    
    # Or use GradCAM class directly
    gradcam = GradCAM(model, label_dict=label_dict)
    img, heatmap, overlaid = gradcam.visualize_class(test_image, class_idx=0)
    """)
