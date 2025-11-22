"""
TensorFlow Lite Conversion Script

This script converts the trained Keras model to TensorFlow Lite format
for deployment on edge devices like Raspberry Pi.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import json

# Configuration
MODEL_PATH = 'models/recyclable_classifier.h5'
TFLITE_MODEL_PATH = 'models/recyclable_classifier.tflite'
TFLITE_QUANTIZED_PATH = 'models/recyclable_classifier_quantized.tflite'
REPRESENTATIVE_DATASET_SIZE = 100


def load_model(model_path):
    """Load the trained Keras model"""
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    return model


def representative_dataset_generator():
    """
    Generate representative dataset for quantization.
    In production, use actual validation/test images.
    """
    print("Generating representative dataset for quantization...")
    for _ in range(REPRESENTATIVE_DATASET_SIZE):
        # Generate random sample (in production, use real images)
        sample = np.random.rand(1, 128, 128, 3).astype(np.float32)
        yield [sample]


def convert_to_tflite(model, output_path, quantize=False):
    """
    Convert Keras model to TensorFlow Lite format.
    
    Args:
        model: Keras model to convert
        output_path: Path to save the TFLite model
        quantize: Whether to apply quantization (reduces model size)
    """
    print(f"\nConverting model to TensorFlow Lite...")
    print(f"Quantization: {'Enabled' if quantize else 'Disabled'}")
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Apply dynamic range quantization (8-bit weights, float activations)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # For full integer quantization (optional, more aggressive)
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.uint8
        # converter.inference_output_type = tf.uint8
        # converter.representative_dataset = representative_dataset_generator
    
    # Convert model
    tflite_model = converter.convert()
    
    # Save model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get model size
    model_size = len(tflite_model) / (1024 * 1024)  # Size in MB
    print(f"Model saved to: {output_path}")
    print(f"Model size: {model_size:.2f} MB")
    
    return tflite_model, model_size


def get_model_info(model_path):
    """Get information about the TFLite model"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    info = {
        'input_shape': input_details[0]['shape'].tolist(),
        'input_dtype': str(input_details[0]['dtype']),
        'output_shape': output_details[0]['shape'].tolist(),
        'output_dtype': str(output_details[0]['dtype']),
    }
    
    return info


def compare_models():
    """Compare original Keras model with TFLite versions"""
    print("=" * 60)
    print("Model Conversion: Keras to TensorFlow Lite")
    print("=" * 60)
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run train_model.py first to train the model.")
        return
    
    # Load original model
    model = load_model(MODEL_PATH)
    
    # Get original model size
    original_size = Path(MODEL_PATH).stat().st_size / (1024 * 1024)
    print(f"\nOriginal Keras model size: {original_size:.2f} MB")
    
    # Convert to TFLite (float32)
    print("\n" + "-" * 60)
    tflite_model, tflite_size = convert_to_tflite(
        model, TFLITE_MODEL_PATH, quantize=False
    )
    
    # Convert to TFLite (quantized)
    print("\n" + "-" * 60)
    tflite_quantized, quantized_size = convert_to_tflite(
        model, TFLITE_QUANTIZED_PATH, quantize=True
    )
    
    # Get model information
    print("\n" + "=" * 60)
    print("Model Information:")
    print("=" * 60)
    
    print("\nFloat32 TFLite Model:")
    float_info = get_model_info(TFLITE_MODEL_PATH)
    for key, value in float_info.items():
        print(f"  {key}: {value}")
    
    print("\nQuantized TFLite Model:")
    quantized_info = get_model_info(TFLITE_QUANTIZED_PATH)
    for key, value in quantized_info.items():
        print(f"  {key}: {value}")
    
    # Size comparison
    print("\n" + "=" * 60)
    print("Size Comparison:")
    print("=" * 60)
    print(f"Original Keras model:     {original_size:.2f} MB")
    print(f"TFLite (float32):         {tflite_size:.2f} MB")
    print(f"TFLite (quantized):       {quantized_size:.2f} MB")
    print(f"\nSize reduction (float32):  {(1 - tflite_size/original_size)*100:.1f}%")
    print(f"Size reduction (quantized): {(1 - quantized_size/original_size)*100:.1f}%")
    
    # Save conversion report
    report = {
        'original_model_size_mb': round(original_size, 2),
        'tflite_float32_size_mb': round(tflite_size, 2),
        'tflite_quantized_size_mb': round(quantized_size, 2),
        'size_reduction_float32_percent': round((1 - tflite_size/original_size)*100, 1),
        'size_reduction_quantized_percent': round((1 - quantized_size/original_size)*100, 1),
        'float32_model_info': float_info,
        'quantized_model_info': quantized_info,
        'models': {
            'keras': MODEL_PATH,
            'tflite_float32': TFLITE_MODEL_PATH,
            'tflite_quantized': TFLITE_QUANTIZED_PATH
        }
    }
    
    report_path = Path('models/conversion_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nConversion report saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("Conversion completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run test_tflite.py to test the TFLite models")
    print("2. Deploy the quantized model to your edge device (Raspberry Pi)")


if __name__ == '__main__':
    print(f"TensorFlow version: {tf.__version__}")
    compare_models()

