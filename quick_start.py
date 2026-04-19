"""
Quick Start Demo - SIBI Gesture Recognition
Menjalankan pipeline lengkap dalam satu script
"""

import numpy as np
import tensorflow as tf
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("\n" + "="*70)
print("SIBI GESTURE RECOGNITION - QUICK START DEMO")
print("="*70 + "\n")

# ========== STEP 1: IMPORT MODULES ==========
print("[STEP 1] Importing modules...")
try:
    from utils.preprocessing import load_sibi_dataset, apply_data_augmentation
    from utils.model_builder import build_model, compile_model, count_trainable_parameters
    from train import (
        TrainingConfig,
        compute_class_weights,
        prepare_training_data,
        train_custom_cnn,
        save_model
    )
    from evaluate import evaluate_model, plot_confusion_matrix
    print("✓ Semua modul berhasil dimuat!\n")
except ImportError as e:
    print(f"✗ Error: {e}")
    exit(1)

# ========== STEP 2: LOAD DATASET ==========
print("[STEP 2] Loading dataset...")
try:
    X, y, label_dict = load_sibi_dataset(
        'dataset/SIBI/',
        img_size=64,
        grayscale=False,
        max_per_class=100,  # Reduced for quick demo
        apply_clahe=True,
        apply_blur=True
    )
    print(f"✓ Dataset loaded: {X.shape}\n")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# ========== STEP 3: PREPARE DATA ==========
print("[STEP 3] Preparing training/validation split...")
X_train, X_val, y_train, y_val = prepare_training_data(X, y, validation_split=0.2)
print(f"✓ Training: {X_train.shape}, Validation: {X_val.shape}\n")

# ========== STEP 4: COMPUTE CLASS WEIGHTS ==========
print("[STEP 4] Computing class weights...")
class_weights = compute_class_weights(y_train, num_classes=len(label_dict))
print(f"✓ Class weights computed\n")

# ========== STEP 5: BUILD MODEL ==========
print("[STEP 5] Building Custom CNN model...")
model, _ = build_model(
    model_type='custom',
    num_classes=len(label_dict),
    img_size=64
)
model = compile_model(model, learning_rate=0.001)

trainable, total = count_trainable_parameters(model)
print(f"✓ Model built! Total params: {total:,}\n")

# ========== STEP 6: TRAIN MODEL ==========
print("[STEP 6] Training model (50 epochs)...")
config = TrainingConfig(
    phase1_epochs=50,
    batch_size=32,
    phase1_lr=0.001,
    early_stopping_patience=10,
    reduce_lr_patience=5
)

model, history = train_custom_cnn(
    model,
    X_train, y_train,
    X_val, y_val,
    config,
    class_weights=class_weights
)

save_model(model, 'model/cnn_model.h5')
print("✓ Model trained and saved!\n")

# ========== STEP 7: EVALUATE ==========
print("[STEP 7] Evaluating model...")
results = evaluate_model(model, X_val, y_val, label_dict)
print(results['summary'])

# ========== COMPLETE ==========
print("\n" + "="*70)
print("✓ QUICK START DEMO SELESAI!")
print("="*70)
print("\n📊 Hasil Training:")
print(f"  - Accuracy: {results['accuracy']*100:.2f}%")
print(f"  - Top-3 Accuracy: {results['top3_accuracy']*100:.2f}%")
print(f"  - Model disimpan: model/cnn_model.h5")

print("\n🚀 Langkah Berikutnya:")
print("\n1. Real-Time Inference:")
print("   python realtime_inference.py --model model/cnn_model.h5")
print("\n2. View TensorBoard:")
print("   tensorboard --logdir logs/")
print("\n3. Train dengan MobileNetV2:")
print("   # Lihat main.py untuk contoh lengkap")
print("\n" + "="*70 + "\n")
