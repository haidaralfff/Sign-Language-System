"""
Complete end-to-end example for SIBI Gesture Recognition System.

This script demonstrates the full pipeline from data loading to evaluation and inference.
Run this script to see all components in action.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Import utilities
from utils.preprocessing import validate_dataset, load_sibi_dataset
from utils.model_builder import build_model, compile_model, count_trainable_parameters
from train import (
    TrainingConfig,
    compute_class_weights,
    prepare_training_data,
    train_custom_cnn,
    save_model
)
from evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_training_history,
    plot_per_class_metrics
)
from utils.visualization import generate_gradcam_batch


def main():
    """Run complete pipeline."""
    
    print("\n" + "="*70)
    print("SIBI GESTURE RECOGNITION - COMPLETE PIPELINE")
    print("="*70 + "\n")
    
    # ========== STEP 1: DATASET VALIDATION ==========
    print("\n[STEP 1] Validating Dataset...")
    print("-" * 70)
    
    dataset_path = 'dataset/SIBI/'
    try:
        validation = validate_dataset(dataset_path)
        print(validation['summary'])
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure your dataset is organized in: dataset/SIBI/")
        print("With subdirectories: A/, B/, C/, ... Z/")
        return
    
    # ========== STEP 2: DATA LOADING & PREPROCESSING ==========
    print("\n[STEP 2] Loading and Preprocessing Dataset...")
    print("-" * 70)
    
    try:
        X, y, label_dict = load_sibi_dataset(
            dataset_path,
            img_size=64,
            grayscale=False,           # RGB mode
            max_per_class=500,         # Stratified sampling
            apply_clahe=True,          # Contrast enhancement
            apply_blur=True            # Noise reduction
        )
        
        # Prepare training/validation split
        X_train, X_val, y_train, y_val = prepare_training_data(
            X, y,
            validation_split=0.2,
            random_state=42
        )
        
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Training: {X_train.shape}")
        print(f"  - Validation: {X_val.shape}")
        print(f"  - Classes: {len(label_dict)}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # ========== STEP 3: COMPUTE CLASS WEIGHTS ==========
    print("\n[STEP 3] Computing Class Weights...")
    print("-" * 70)
    
    class_weights = compute_class_weights(y_train, num_classes=len(label_dict))
    print("✓ Class weights computed for imbalanced dataset handling")
    
    # ========== STEP 4: BUILD MODEL ==========
    print("\n[STEP 4] Building Model Architecture...")
    print("-" * 70)
    
    print("\nAvailable options:")
    print("  1) Custom CNN (Enhanced) - Lightweight, fast, ~1.7M params")
    print("  2) MobileNetV2 Transfer Learning - High accuracy, ~3.7M params")
    print("\nSelect model type (1 or 2): ", end="")
    
    try:
        choice = input().strip()
    except:
        choice = "1"
    
    if choice == "2":
        model_type = 'mobilenetv2'
        img_size = 128
        print("\nBuilding MobileNetV2 Transfer Learning Model...")
        model, base_model = build_model(
            model_type='mobilenetv2',
            num_classes=len(label_dict),
            img_size=128,
            freeze_base=True
        )
    else:
        model_type = 'custom'
        img_size = 64
        print("\nBuilding Custom CNN Model...")
        model, _ = build_model(
            model_type='custom',
            num_classes=len(label_dict),
            img_size=64
        )
    
    # Compile model
    model = compile_model(
        model,
        optimizer_type='adam',
        learning_rate=0.001,
        metrics=['accuracy']
    )
    
    trainable, total = count_trainable_parameters(model)
    print(f"\n✓ Model compiled successfully!")
    print(f"  - Total parameters: {total:,}")
    print(f"  - Trainable parameters: {trainable:,}")
    
    # ========== STEP 5: TRAINING ==========
    print("\n[STEP 5] Training Model...")
    print("-" * 70)
    
    config = TrainingConfig(
        phase1_epochs=50,
        phase2_epochs=0 if model_type == 'custom' else 30,
        batch_size=32,
        phase1_lr=0.001,
        phase2_lr=1e-5
    )
    
    print(f"\nTraining Configuration:")
    print(f"  - Epochs: {config.phase1_epochs}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Learning rate: {config.phase1_lr}")
    
    # Train model
    model, history = train_custom_cnn(
        model,
        X_train, y_train,
        X_val, y_val,
        config,
        class_weights=class_weights
    )
    
    # Save model
    save_model(model, 'model/cnn_model.h5')
    
    # ========== STEP 6: EVALUATION ==========
    print("\n[STEP 6] Evaluating Model...")
    print("-" * 70)
    
    results = evaluate_model(
        model,
        X_val, y_val,
        label_dict,
        results_dir='results'
    )
    
    print(results['summary'])
    
    # ========== STEP 7: VISUALIZATIONS ==========
    print("\n[STEP 7] Generating Visualizations...")
    print("-" * 70)
    
    print("\nGenerating...")
    print("  - Confusion Matrix Heatmap...")
    plot_confusion_matrix(results['confusion_matrix'], label_dict, save=True)
    
    print("  - Training History Plot...")
    plot_training_history(history.history, save=True)
    
    print("  - Per-Class Metrics (F1-Score)...")
    plot_per_class_metrics(results['per_class_metrics'], metric='f1-score', save=True)
    
    # ========== STEP 8: GRAD-CAM ==========
    print("\n[STEP 8] Generating Grad-CAM Visualizations...")
    print("-" * 70)
    
    try:
        print("Generating Grad-CAM heatmaps (this may take a few minutes)...")
        generate_gradcam_batch(
            model,
            X_val,
            true_labels=y_val,
            label_dict=label_dict,
            results_dir='results/gradcam',
            num_samples=2,
            save_plots=True
        )
        print("✓ Grad-CAM visualizations generated!")
    except Exception as e:
        print(f"⚠️  Grad-CAM generation skipped: {e}")
    
    # ========== NEXT STEPS ==========
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    
    print("\n📊 Results Summary:")
    print(f"  - Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  - Top-3 Accuracy: {results['top3_accuracy']*100:.2f}%")
    print(f"  - Model saved: model/cnn_model.h5")
    print(f"  - Evaluation report: results/evaluation_report.txt")
    print(f"  - Visualizations: results/")
    
    print("\n🚀 Next Steps:")
    print("\n1. Real-Time Inference (Webcam):")
    print("   python realtime_inference.py --model model/cnn_model.h5")
    
    print("\n2. View TensorBoard Logs:")
    print("   tensorboard --logdir logs/")
    
    print("\n3. Analyze Individual Predictions:")
    print("   from utils.visualization import GradCAM")
    print("   gradcam = GradCAM(model, label_dict=label_dict)")
    print("   img, heatmap, overlaid = gradcam.visualize_class(test_image, class_idx=0)")
    
    print("\n4. Fine-tune Model (Transfer Learning only):")
    print("   # See train.py for Phase 2 fine-tuning implementation")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

