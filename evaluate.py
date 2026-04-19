"""
Comprehensive evaluation module for SIBI gesture recognition models.

Provides:
- Classification report (precision, recall, F1 per class)
- Confusion matrix visualization (normalized heatmap)
- Top-3 accuracy calculation
- Model evaluation and report generation
- Results saved to evaluation_report.txt
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from tensorflow.keras.metrics import TopKCategoricalAccuracy
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_model(
    model: tf.keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    label_dict: Dict[str, int],
    results_dir: str = 'results',
    confidence_threshold: float = 0.5
) -> Dict:
    """
    Comprehensive model evaluation with multiple metrics.
    
    Generates:
    - Classification report (sklearn format)
    - Confusion matrix
    - Top-3 accuracy
    - Per-class metrics
    - Saves results to results/evaluation_report.txt
    
    Args:
        model (tf.keras.Model): Trained model
        X_val (np.ndarray): Validation images
        y_val (np.ndarray): Validation labels (one-hot encoded)
        label_dict (Dict): Class name to index mapping
        results_dir (str): Directory to save results
        confidence_threshold (float): Minimum confidence for prediction
        
    Returns:
        Dict: Evaluation metrics dictionary containing:
            - 'accuracy': Overall accuracy
            - 'top3_accuracy': Top-3 accuracy
            - 'classification_report': sklearn classification report dict
            - 'confusion_matrix': Normalized confusion matrix
            - 'per_class_metrics': Per-class precision/recall/f1
            - 'summary': Human-readable summary string
            
    Example:
        >>> results = evaluate_model(model, X_val, y_val, label_dict)
        >>> print(results['summary'])
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*60)
    logger.info("MODEL EVALUATION")
    logger.info("="*60)
    
    # Get predictions
    logger.info("Generating predictions...")
    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_val, axis=1)
    
    # Create reverse label dict (index to class name)
    idx_to_class = {v: k for k, v in label_dict.items()}
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    logger.info(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Top-3 accuracy
    top3_metric = TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
    top3_metric.update_state(y_val, y_pred_probs)
    top3_accuracy = top3_metric.result().numpy()
    logger.info(f"Top-3 Accuracy: {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)")
    
    # Classification report
    logger.info("Generating classification report...")
    class_report = classification_report(
        y_true, y_pred,
        target_names=[idx_to_class[i] for i in range(len(label_dict))],
        output_dict=True,
        zero_division=0
    )
    
    # Print classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        y_true, y_pred,
        target_names=[idx_to_class[i] for i in range(len(label_dict))],
        zero_division=0
    ))
    
    # Confusion matrix
    logger.info("Computing confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Per-class metrics
    per_class_metrics = {}
    for class_idx in range(len(label_dict)):
        class_name = idx_to_class[class_idx]
        if class_idx in class_report:
            metrics = class_report[class_idx]
            per_class_metrics[class_name] = {
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
                'f1-score': metrics.get('f1-score', 0.0),
                'support': int(metrics.get('support', 0)),
            }
    
    # Low-performing classes (F1 < 0.7)
    low_f1_classes = [
        (name, metrics['f1-score'])
        for name, metrics in per_class_metrics.items()
        if metrics['f1-score'] < 0.7
    ]
    
    # Generate summary
    summary_lines = [
        "\n" + "="*60,
        "EVALUATION SUMMARY",
        "="*60,
        f"Validation Samples: {len(X_val)}",
        f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)",
        f"Top-3 Accuracy: {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)",
        f"Total Classes: {len(label_dict)}",
    ]
    
    if low_f1_classes:
        summary_lines.append(f"\n⚠️  Classes with F1 < 0.70:")
        for class_name, f1_score in sorted(low_f1_classes, key=lambda x: x[1]):
            summary_lines.append(f"  - {class_name}: {f1_score:.4f}")
    
    summary_lines.append("\n" + "="*60)
    summary = '\n'.join(summary_lines)
    logger.info(summary)
    
    # Save results to file
    report_path = Path(results_dir) / 'evaluation_report.txt'
    _save_evaluation_report(
        report_path,
        accuracy,
        top3_accuracy,
        class_report,
        cm_normalized,
        per_class_metrics,
        X_val.shape,
        label_dict
    )
    
    return {
        'accuracy': float(accuracy),
        'top3_accuracy': float(top3_accuracy),
        'classification_report': class_report,
        'confusion_matrix': cm_normalized,
        'per_class_metrics': per_class_metrics,
        'y_pred': y_pred,
        'y_true': y_true,
        'y_pred_probs': y_pred_probs,
        'summary': summary
    }


def plot_confusion_matrix(
    cm_normalized: np.ndarray,
    label_dict: Dict[str, int],
    results_dir: str = 'results',
    figsize: Tuple[int, int] = (16, 14),
    save: bool = True
) -> None:
    """
    Plot normalized confusion matrix as heatmap.
    
    Args:
        cm_normalized (np.ndarray): Normalized confusion matrix
        label_dict (Dict): Class name to index mapping
        results_dir (str): Directory to save plot
        figsize (Tuple): Figure size (width, height)
        save (bool): Whether to save the plot
        
    Example:
        >>> plot_confusion_matrix(cm, label_dict)
        >>> # Saves to: results/confusion_matrix.png
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Create reverse label dict
    idx_to_class = {v: k for k, v in label_dict.items()}
    class_names = [idx_to_class[i] for i in range(len(label_dict))]
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'},
        square=True,
        vmin=0,
        vmax=1
    )
    
    plt.title('Normalized Confusion Matrix - SIBI Gesture Recognition', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save:
        save_path = Path(results_dir) / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to: {save_path}")
    
    plt.show()


def plot_training_history(
    history_dict: dict,
    results_dir: str = 'results',
    save: bool = True
) -> None:
    """
    Plot training history (accuracy and loss).
    
    Args:
        history_dict (dict): Training history from model.fit()
        results_dir (str): Directory to save plot
        save (bool): Whether to save the plot
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history_dict.get('accuracy', []), label='Train', linewidth=2)
    axes[0].plot(history_dict.get('val_accuracy', []), label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history_dict.get('loss', []), label='Train', linewidth=2)
    axes[1].plot(history_dict.get('val_loss', []), label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        save_path = Path(results_dir) / 'training_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Training history plot saved to: {save_path}")
    
    plt.show()


def plot_per_class_metrics(
    per_class_metrics: Dict,
    metric: str = 'f1-score',
    results_dir: str = 'results',
    save: bool = True
) -> None:
    """
    Plot per-class metrics as bar chart.
    
    Args:
        per_class_metrics (Dict): Per-class metrics from evaluate_model
        metric (str): Metric to plot ('precision', 'recall', 'f1-score')
        results_dir (str): Directory to save plot
        save (bool): Whether to save the plot
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    classes = list(per_class_metrics.keys())
    values = [per_class_metrics[cls][metric] for cls in classes]
    
    # Sort by value
    sorted_pairs = sorted(zip(classes, values), key=lambda x: x[1])
    classes_sorted, values_sorted = zip(*sorted_pairs)
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(classes_sorted, values_sorted, color='steelblue', alpha=0.8)
    
    # Color bars based on threshold
    for bar, val in zip(bars, values_sorted):
        if val < 0.7:
            bar.set_color('coral')
        elif val < 0.85:
            bar.set_color('gold')
        else:
            bar.set_color('lightgreen')
    
    plt.xlabel('Class', fontsize=11)
    plt.ylabel(metric.capitalize(), fontsize=11)
    plt.title(f'Per-Class {metric.capitalize()}', fontsize=12, fontweight='bold')
    plt.xticks(rotation=0)
    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Threshold (0.70)')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save:
        save_path = Path(results_dir) / f'per_class_{metric}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Per-class metrics plot saved to: {save_path}")
    
    plt.show()


def _save_evaluation_report(
    report_path: Path,
    accuracy: float,
    top3_accuracy: float,
    class_report: dict,
    cm_normalized: np.ndarray,
    per_class_metrics: dict,
    data_shape: tuple,
    label_dict: dict
) -> None:
    """
    Save detailed evaluation report to text file.
    
    Args:
        report_path (Path): Path to save report
        accuracy (float): Overall accuracy
        top3_accuracy (float): Top-3 accuracy
        class_report (dict): Classification report dict
        cm_normalized (np.ndarray): Normalized confusion matrix
        per_class_metrics (dict): Per-class metrics
        data_shape (tuple): Shape of validation data
        label_dict (dict): Class mapping
    """
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("SIBI GESTURE RECOGNITION - EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Validation Samples: {data_shape[0]}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Top-3 Accuracy: {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)\n")
        f.write(f"Total Classes: {len(label_dict)}\n\n")
        
        f.write("PER-CLASS METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-"*70 + "\n")
        
        for class_name in sorted(per_class_metrics.keys()):
            metrics = per_class_metrics[class_name]
            f.write(
                f"{class_name:<10} "
                f"{metrics['precision']:<12.4f} "
                f"{metrics['recall']:<12.4f} "
                f"{metrics['f1-score']:<12.4f} "
                f"{metrics['support']:<10}\n"
            )
        
        f.write("\n" + "="*70 + "\n")
    
    logger.info(f"Evaluation report saved to: {report_path}")


if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
    print("\nExample usage:")
    print("""
    from evaluate import evaluate_model, plot_confusion_matrix
    
    # Evaluate model
    results = evaluate_model(model, X_val, y_val, label_dict)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        results['confusion_matrix'],
        label_dict,
        results_dir='results'
    )
    
    # Plot per-class F1 scores
    plot_per_class_metrics(
        results['per_class_metrics'],
        metric='f1-score',
        results_dir='results'
    )
    """)
