"""
Edge AI Prototype: Recyclable Item Classification Model Training

This script trains a lightweight CNN model for classifying recyclable items.
The model is designed to be optimized for edge devices like Raspberry Pi.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration
IMG_SIZE = 128  # Reduced size for edge devices
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 4  # Paper, Plastic, Glass, Metal
LEARNING_RATE = 0.001

# Class names for recyclable items
CLASS_NAMES = ['paper', 'plastic', 'glass', 'metal']


def create_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """
    Create a lightweight CNN model optimized for edge devices.
    
    Architecture:
    - Uses depthwise separable convolutions for efficiency
    - Batch normalization for stability
    - Global average pooling to reduce parameters
    - Dropout for regularization
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First block - depthwise separable convolution
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third block
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth block - smaller filters
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Global average pooling instead of flatten to reduce parameters
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ])
    
    return model


def load_synthetic_data():
    """
    Generate synthetic data for demonstration.
    In a real scenario, you would load actual images from a dataset.
    """
    print("Generating synthetic training data...")
    print("Note: In production, replace this with actual image dataset loading")
    
    # Generate synthetic images (simulating real dataset)
    num_samples = 800  # Total samples
    X = np.random.rand(num_samples, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    
    # Create class labels
    y = np.random.randint(0, NUM_CLASSES, num_samples)
    y = keras.utils.to_categorical(y, NUM_CLASSES)
    
    # Add some structure to make it more realistic
    for i in range(num_samples):
        class_idx = np.argmax(y[i])
        # Add class-specific patterns
        if class_idx == 0:  # Paper - lighter colors
            X[i] = X[i] * 0.7 + 0.3
        elif class_idx == 1:  # Plastic - medium colors
            X[i] = X[i] * 0.8 + 0.2
        elif class_idx == 2:  # Glass - darker with highlights
            X[i] = X[i] * 0.6 + 0.4
        else:  # Metal - metallic colors
            X[i] = X[i] * 0.5 + 0.5
    
    return X, y


def load_real_data(data_dir='data'):
    """
    Load real image data from directory structure.
    Expected structure:
    data/
        paper/
        plastic/
        glass/
        metal/
    """
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found. Using synthetic data.")
        return None, None
    
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_file)
                try:
                    img = keras.preprocessing.image.load_img(
                        img_path, target_size=(IMG_SIZE, IMG_SIZE)
                    )
                    img_array = keras.preprocessing.image.img_to_array(img)
                    img_array = img_array / 255.0  # Normalize
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    if len(images) == 0:
        return None, None
    
    X = np.array(images)
    y = keras.utils.to_categorical(labels, NUM_CLASSES)
    
    return X, y


def train_model():
    """Main training function"""
    print("=" * 60)
    print("Edge AI Model Training - Recyclable Item Classification")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path('models')
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    X, y = load_real_data()
    if X is None:
        print("\nUsing synthetic data for demonstration...")
        X, y = load_synthetic_data()
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Class names: {CLASS_NAMES}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, 
        stratify=np.argmax(y_train, axis=1)
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Data augmentation for better generalization
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomFlip("horizontal"),
    ])
    
    # Create model
    print("\nCreating model architecture...")
    model = create_model()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    # Print model summary
    print("\nModel Architecture:")
    print("-" * 60)
    model.summary()
    
    # Calculate model size
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Prepare training data with augmentation
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Train model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    
    test_loss, test_accuracy, test_top_k = model.evaluate(
        X_test, y_test, verbose=1
    )
    
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Top-K Accuracy: {test_top_k:.4f}")
    
    # Save final model
    model_path = output_dir / 'recyclable_classifier.h5'
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'test_accuracy': float(test_accuracy),
        'test_top_k_accuracy': float(test_top_k),
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs': len(history.history['loss']),
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"Training history saved to: {history_path}")
    
    # Plot training curves
    plot_training_curves(history, output_dir)
    
    return model, history, test_accuracy


def plot_training_curves(history, output_dir):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {output_dir / 'training_curves.png'}")
    plt.close()


if __name__ == '__main__':
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    
    # Train the model
    model, history, test_accuracy = train_model()
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
    print("\nNext steps:")
    print("1. Run convert_to_tflite.py to convert the model to TensorFlow Lite")
    print("2. Run test_tflite.py to test the converted model")

