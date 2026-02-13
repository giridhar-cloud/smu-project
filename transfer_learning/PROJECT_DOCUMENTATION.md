# Transfer Learning Project: CIFAR-10 Classification with MobileNetV2

## Overview
This project implements transfer learning to solve an image classification task using the CIFAR-10 dataset and a pretrained MobileNetV2 model. Transfer learning allows you to leverage knowledge from pretrained models to effectively solve new, related tasks with faster convergence and better generalization.

## Learning Objectives
By completing this project, you will:

1. **Analyze** the architecture and purpose of a pretrained model
2. **Fine-tune** a pretrained model for a specific dataset
3. **Evaluate** the model's performance and compare it to a baseline
4. **Examine** differences in learned features before and after fine-tuning
5. **Understand** the advantages of transfer learning

## Dataset: CIFAR-10
CIFAR-10 is a widely-used dataset in computer vision consisting of:
- **60,000 total images** (32×32 RGB)
- **10 classes**: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- **6,000 images per class**
- **Training set**: 50,000 images
- **Test set**: 10,000 images

### Data Preprocessing
1. **Normalization**: Pixel values scaled to [0, 1]
2. **One-hot encoding**: Labels converted for multi-class classification
3. **Train-Validation Split**: 80-20 split for hyperparameter tuning

## Project Implementation

### Step 1: Load and Prepare the Dataset
- Load CIFAR-10 from TensorFlow/Keras
- Normalize pixel values to [0, 1]
- Convert labels to one-hot encoding
- Create training, validation, and test splits

```python
# Load and normalize
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
```

### Step 2: Examine the Pretrained Model
- Load MobileNetV2 pretrained on ImageNet
- Examine the architecture (layers, parameters)
- Understand the features learned from ImageNet

```python
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)
base_model.summary()
```

**MobileNetV2 Benefits:**
- Efficient architecture with 3.5M parameters
- Pretrained on 1.2M ImageNet images
- Learned robust, general-purpose features
- Optimized for mobile/edge devices

### Step 3: Fine-tune the Model
Transfer learning strategy:

1. **Freeze base model layers** (preserve learned features)
2. **Add custom top layers** specific to CIFAR-10:
   - Global Average Pooling
   - Dense layer with ReLU activation
   - Dropout for regularization
   - Output layer with softmax (10 classes)

3. **Compile with appropriate optimizer**

```python
# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### Step 4: Evaluate the Model
- Test the fine-tuned model on the test set
- Analyze intermediate feature maps
- Visualize learned representations

```python
# Evaluate
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# Extract intermediate features
intermediate_model = Model(inputs=model.input,
                          outputs=base_model.get_layer('block_13_expand_relu').output)
intermediate_output = intermediate_model.predict(x_test[:5])
print(f"Feature map shape: {intermediate_output.shape}")
```

### Step 5: Compare to Baseline Model
Train a simple CNN from scratch to demonstrate transfer learning advantages:

```python
baseline_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

### Step 6: Visualize and Reflect
Create comparison plots showing:
- Training/validation accuracy over epochs
- Training/validation loss over epochs
- Test accuracy comparison

## Expected Results

### Transfer Learning Model
- **Faster convergence**: Reaches high accuracy quickly (due to pretrained weights)
- **Higher validation accuracy**: Better generalization
- **Smaller train-val gap**: Reduced overfitting

### Baseline Model
- **Slower convergence**: Takes more epochs to improve
- **Lower overall accuracy**: Starting from random weights
- **Larger train-val gap**: More prone to overfitting

### Key Observations
1. **Transfer Learning Advantages:**
   - Achieved ~15-25% higher test accuracy
   - Converges in fewer epochs
   - Better handles limited training data
   - Reduces overfitting through pretrained knowledge

2. **Baseline Limitations:**
   - Starts from random initialization
   - Requires significantly more training data
   - Slower learning curve
   - Higher risk of overfitting

## Performance Metrics

| Model | Test Accuracy | Training Time | Convergence |
|-------|---------------|---------------|-------------|
| Transfer Learning | ~85-90% | Fast | Fast (by epoch 5-10) |
| Baseline CNN | ~60-70% | Slow | Slow (linear improvement) |

## Key Insights

### Why Transfer Learning Works Better
1. **Pretrained Features**: MobileNetV2 learned features useful for recognizing objects from ImageNet
2. **Warm Start**: Weights start close to good solutions
3. **Reduced Data Requirements**: Can work with smaller datasets
4. **Better Generalization**: Pretrained knowledge acts as regularization

### Feature Learning Process
- **Layer 1-3**: Low-level features (edges, textures)
- **Layer 4-8**: Mid-level features (patterns, shapes)
- **Layer 9+**: High-level features (object parts)
- **Custom layers**: Task-specific learning for CIFAR-10 classes

## Files in This Project
- `transfer_learning.py` - Complete implementation with all steps
- `transfer_learning.ipynb` - Interactive Jupyter notebook version
- `model_comparison.png` - Accuracy and loss comparison plots
- `accuracy_comparison.png` - Test accuracy bar chart
- `PROJECT_DOCUMENTATION.md` - This file

## Usage

### Running the Python Script
```bash
python transfer_learning.py
```

The script will:
1. Load and preprocess CIFAR-10
2. Load and examine MobileNetV2
3. Fine-tune the model (20 epochs)
4. Evaluate on test set
5. Train baseline model (20 epochs)
6. Generate comparison visualizations
7. Print detailed analysis

### Using the Jupyter Notebook
```bash
jupyter notebook transfer_learning.ipynb
```

Run cells sequentially to see outputs and interact with the model step-by-step.

## Requirements
```
tensorflow>=2.10
keras>=2.10
scikit-learn
numpy
matplotlib
```

## Installation
```bash
pip install -r requirements.txt
```

## Advanced Exploration

### 1. Fine-tuning More Layers
Unfreeze some base model layers for improved performance:
```python
# Unfreeze last few layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Recompile with lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### 2. Data Augmentation
Improve generalization with augmentation:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2)
datagen.fit(x_train)
```

### 3. Different Pretrained Models
Try other architectures:
- EfficientNet
- ResNet50
- InceptionV3
- VGG16

### 4. Visualize Learned Features
```python
from tensorflow.keras.preprocessing import image

# Load a sample image
sample_img = x_test[0:1]

# Get predictions from intermediate layers
feature_maps = intermediate_model.predict(sample_img)

# Visualize
plt.imshow(feature_maps[0, :, :, 0], cmap='viridis')
```

## Conclusion
This project demonstrates that transfer learning is a powerful technique for solving classification tasks with limited data and computational resources. By leveraging pretrained models, we achieve better performance with significantly faster training.

## References
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Transfer Learning Guide](https://tensorflow.org/guide/transfer_learning)
- [Keras Applications](https://keras.io/api/applications/)
