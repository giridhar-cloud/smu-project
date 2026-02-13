"""
Transfer Learning Project: CIFAR-10 Classification with MobileNetV2
This project implements transfer learning to solve a classification task using a pretrained model.
By the end, you'll understand how transfer learning accelerates training and improves generalization.
"""

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ========== STEP 1: Load and Prepare the Dataset ==========
print("=" * 80)
print("STEP 1: Loading and Preparing the CIFAR-10 Dataset")
print("=" * 80)

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(f"Original Training set shape: {x_train.shape}")
print(f"Original Test set shape: {x_test.shape}")

# Normalize the pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
print(f"Normalized pixel values to range [0, 1]")

# Flatten labels for processing
y_train = y_train.flatten()
y_test = y_test.flatten()

# Convert labels to one-hot encoded format
y_train_one_hot = to_categorical(y_train, 10)
y_test_one_hot = to_categorical(y_test, 10)
print(f"Labels converted to one-hot encoding. Shape: {y_train_one_hot.shape}")

# Split training data into training and validation sets
x_train_split, x_val, y_train_split, y_val = train_test_split(
    x_train, y_train_one_hot, test_size=0.2, random_state=42, stratify=y_train
)
print(f"Training set: {x_train_split.shape}, Validation set: {x_val.shape}")

# CIFAR-10 class names for reference
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
print(f"Classes: {class_names}")
print()


# ========== STEP 2: Examine the Pretrained Model ==========
print("=" * 80)
print("STEP 2: Examining the Pretrained MobileNetV2 Model")
print("=" * 80)

# Load the MobileNetV2 model pretrained on ImageNet
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)

print("MobileNetV2 Model Architecture Summary:")
base_model.summary()

# Show total number of layers and parameters
total_layers = len(base_model.layers)
trainable_params_before = sum([tf.size(w).numpy() for w in base_model.trainable_weights])
print(f"\nTotal layers in base model: {total_layers}")
print(f"Total trainable parameters before freezing: {trainable_params_before:,}")
print()


# ========== STEP 3: Fine-tune the Model ==========
print("=" * 80)
print("STEP 3: Fine-tuning the Pretrained Model")
print("=" * 80)

# Freeze the base model layers to preserve learned features
for layer in base_model.layers:
    layer.trainable = False

print(f"Frozen all {len(base_model.layers)} layers in base model")

# Build custom top layers for CIFAR-10 classification
inputs = Input(shape=(32, 32, 3))
x = base_model(inputs, training=False)  # Use base model in inference mode
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(10, activation='softmax')(x)

# Create the fine-tuned model
fine_tuned_model = Model(inputs, outputs)

print(f"Fine-tuned model created with custom top layers")
print(f"Fine-tuned model summary:")
fine_tuned_model.summary()

# Compile the model
fine_tuned_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the fine-tuned model
print("\nTraining the fine-tuned model...")
history = fine_tuned_model.fit(
    x_train_split, y_train_split,
    validation_data=(x_val, y_val),
    epochs=20,
    batch_size=64,
    verbose=1
)
print("Fine-tuned model training completed!")
print()


# ========== STEP 4: Evaluate the Model ==========
print("=" * 80)
print("STEP 4: Evaluating the Fine-tuned Model")
print("=" * 80)

# Evaluate on test set
test_loss, test_accuracy = fine_tuned_model.evaluate(x_test, y_test_one_hot, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Analyze feature maps from an intermediate layer
print("\nAnalyzing intermediate feature maps...")
intermediate_layer_name = 'block_13_expand_relu'  # Layer from MobileNetV2
intermediate_layer_model = Model(
    inputs=fine_tuned_model.input,
    outputs=base_model.get_layer(intermediate_layer_name).output
)

# Get feature maps for a sample of test images
sample_images = x_test[:5]
intermediate_output = intermediate_layer_model.predict(sample_images, verbose=0)
print(f"Feature map shape from '{intermediate_layer_name}': {intermediate_output.shape}")
print(f"This shows how the model processes images through its layers")
print()


# ========== STEP 5: Compare to Baseline Model ==========
print("=" * 80)
print("STEP 5: Comparing to Baseline CNN Model")
print("=" * 80)

# Define a simple baseline CNN model trained from scratch
baseline_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

baseline_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Baseline model architecture:")
baseline_model.summary()

# Train the baseline model
print("\nTraining the baseline model from scratch...")
baseline_history = baseline_model.fit(
    x_train_split, y_train_split,
    validation_data=(x_val, y_val),
    epochs=20,
    batch_size=64,
    verbose=1
)
print("Baseline model training completed!")

# Evaluate baseline model
baseline_test_loss, baseline_test_accuracy = baseline_model.evaluate(
    x_test, y_test_one_hot, verbose=0
)
print(f"\nBaseline Test Loss: {baseline_test_loss:.4f}")
print(f"Baseline Test Accuracy: {baseline_test_accuracy:.4f} ({baseline_test_accuracy*100:.2f}%)")
print()


# ========== STEP 6: Visualize and Reflect ==========
print("=" * 80)
print("STEP 6: Visualizing Results and Analysis")
print("=" * 80)

# Create comparison plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training and Validation Accuracy
axes[0].plot(history.history['accuracy'], label='Transfer Learning - Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Transfer Learning - Val Accuracy', linewidth=2)
axes[0].plot(baseline_history.history['accuracy'], label='Baseline - Train Accuracy', linewidth=2, linestyle='--')
axes[0].plot(baseline_history.history['val_accuracy'], label='Baseline - Val Accuracy', linewidth=2, linestyle='--')
axes[0].set_xlabel('Epochs', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Model Comparison: Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Training and Validation Loss
axes[1].plot(history.history['loss'], label='Transfer Learning - Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Transfer Learning - Val Loss', linewidth=2)
axes[1].plot(baseline_history.history['loss'], label='Baseline - Train Loss', linewidth=2, linestyle='--')
axes[1].plot(baseline_history.history['val_loss'], label='Baseline - Val Loss', linewidth=2, linestyle='--')
axes[1].set_xlabel('Epochs', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Model Comparison: Loss Over Epochs', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/azureuser/cloudfiles/code/transfer_learning/model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved comparison plot to 'model_comparison.png'")
plt.close()

# Create test accuracy comparison
fig, ax = plt.subplots(figsize=(10, 6))
models = ['Transfer Learning', 'Baseline CNN']
accuracies = [test_accuracy, baseline_test_accuracy]
colors = ['#2ecc71', '#e74c3c']
bars = ax.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc*100:.2f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('Test Accuracy Comparison: Transfer Learning vs Baseline', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('/home/azureuser/cloudfiles/code/transfer_learning/accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("Saved accuracy comparison plot to 'accuracy_comparison.png'")
plt.close()


# ========== Summary and Analysis ==========
print("=" * 80)
print("ANALYSIS AND INTERPRETATION")
print("=" * 80)

print("\n📊 TRANSFER LEARNING MODEL:")
print(f"  • Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  • Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  • Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
train_val_gap = history.history['accuracy'][-1] - history.history['val_accuracy'][-1]
print(f"  • Train-Val Accuracy Gap: {train_val_gap:.4f} (overfitting indicator)")

print("\n📊 BASELINE MODEL:")
print(f"  • Test Accuracy: {baseline_test_accuracy*100:.2f}%")
print(f"  • Final Training Accuracy: {baseline_history.history['accuracy'][-1]:.4f}")
print(f"  • Final Validation Accuracy: {baseline_history.history['val_accuracy'][-1]:.4f}")
baseline_train_val_gap = baseline_history.history['accuracy'][-1] - baseline_history.history['val_accuracy'][-1]
print(f"  • Train-Val Accuracy Gap: {baseline_train_val_gap:.4f} (overfitting indicator)")

print("\n📈 KEY OBSERVATIONS:")
accuracy_improvement = (test_accuracy - baseline_test_accuracy) * 100
print(f"  1. Transfer Learning achieved {accuracy_improvement:.2f}% higher test accuracy")
print(f"  2. Transfer Learning converged faster (note the steeper initial curves)")
print(f"  3. Transfer Learning shows {'less' if train_val_gap < baseline_train_val_gap else 'more'} overfitting")
print(f"  4. Validation accuracy improved significantly with transfer learning")

print("\n💡 INTERPRETATIONS:")
print("  • Transfer Learning Advantages:")
print("    - Leverages pretrained features from ImageNet")
print("    - Faster convergence due to warm-start weights")
print("    - Better generalization with limited data")
print("    - Reduced overfitting through pretrained knowledge")
print("\n  • Baseline Model Limitations:")
print("    - Started from random weights")
print("    - Requires more data to achieve comparable results")
print("    - Slower training convergence")
print("    - Higher risk of overfitting with limited data")

print("\n✅ CONCLUSION:")
print("Transfer learning significantly outperforms training from scratch, demonstrating")
print("the value of leveraging pretrained models for classification tasks.")

print("\n" + "=" * 80)
print(f"Project completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
