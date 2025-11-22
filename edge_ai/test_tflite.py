"""
TensorFlow Lite Model Testing Script

This script tests the converted TFLite models on sample data
and compares performance with the original Keras model.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from pathlib import Path
import json
from datetime import datetime

# Configuration
KERAS_MODEL_PATH = 'models/recyclable_classifier.h5'
TFLITE_MODEL_PATH = 'models/recyclable_classifier.tflite'
TFLITE_QUANTIZED_PATH = 'models/recyclable_classifier_quantized.tflite'
NUM_TEST_SAMPLES = 100
NUM_INFERENCE_RUNS = 10  # For timing measurements

CLASS_NAMES = ['paper', 'plastic', 'glass', 'metal']


def load_test_data(num_samples=NUM_TEST_SAMPLES):
    """Generate test data"""
    print(f"Generating {num_samples} test samples...")
    # In production, load actual test images
    test_data = np.random.rand(num_samples, 128, 128, 3).astype(np.float32)
    return test_data


def test_keras_model(model_path, test_data):
    """Test the original Keras model"""
    print("\n" + "=" * 60)
    print("Testing Keras Model")
    print("=" * 60)
    
    model = keras.models.load_model(model_path)
    
    # Warm-up run
    _ = model.predict(test_data[:1], verbose=0)
    
    # Time inference
    start_time = time.time()
    predictions = model.predict(test_data, verbose=0)
    inference_time = time.time() - start_time
    
    avg_time_per_sample = inference_time / len(test_data)
    
    print(f"Total inference time: {inference_time:.4f} seconds")
    print(f"Average time per sample: {avg_time_per_sample*1000:.2f} ms")
    print(f"Throughput: {len(test_data)/inference_time:.2f} samples/second")
    
    return predictions, avg_time_per_sample


def test_tflite_model(model_path, test_data, model_name="TFLite"):
    """Test a TFLite model"""
    print("\n" + "=" * 60)
    print(f"Testing {model_name} Model")
    print("=" * 60)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output dtype: {output_details[0]['dtype']}")
    
    # Prepare input
    input_data = test_data.astype(input_details[0]['dtype'])
    
    # Warm-up run
    interpreter.set_tensor(input_details[0]['index'], input_data[:1])
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]['index'])
    
    # Time inference
    predictions = []
    start_time = time.time()
    
    for i in range(len(test_data)):
        interpreter.set_tensor(input_details[0]['index'], input_data[i:i+1])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output[0])
    
    inference_time = time.time() - start_time
    avg_time_per_sample = inference_time / len(test_data)
    
    predictions = np.array(predictions)
    
    print(f"Total inference time: {inference_time:.4f} seconds")
    print(f"Average time per sample: {avg_time_per_sample*1000:.2f} ms")
    print(f"Throughput: {len(test_data)/inference_time:.2f} samples/second")
    
    return predictions, avg_time_per_sample


def compare_predictions(keras_preds, tflite_preds, quantized_preds):
    """Compare predictions between models"""
    print("\n" + "=" * 60)
    print("Prediction Comparison")
    print("=" * 60)
    
    # Get predicted classes
    keras_classes = np.argmax(keras_preds, axis=1)
    tflite_classes = np.argmax(tflite_preds, axis=1)
    quantized_classes = np.argmax(quantized_preds, axis=1)
    
    # Calculate agreement
    keras_tflite_agreement = np.mean(keras_classes == tflite_classes)
    keras_quantized_agreement = np.mean(keras_classes == quantized_classes)
    
    print(f"Keras vs TFLite (float32) agreement: {keras_tflite_agreement*100:.2f}%")
    print(f"Keras vs TFLite (quantized) agreement: {keras_quantized_agreement*100:.2f}%")
    
    # Calculate prediction differences
    keras_tflite_diff = np.mean(np.abs(keras_preds - tflite_preds))
    keras_quantized_diff = np.mean(np.abs(keras_preds - quantized_preds))
    
    print(f"\nMean absolute difference (Keras vs TFLite float32): {keras_tflite_diff:.6f}")
    print(f"Mean absolute difference (Keras vs TFLite quantized): {keras_quantized_diff:.6f}")
    
    # Show some example predictions
    print("\nExample Predictions (first 5 samples):")
    print("-" * 60)
    for i in range(min(5, len(keras_preds))):
        keras_class = CLASS_NAMES[keras_classes[i]]
        keras_conf = keras_preds[i][keras_classes[i]]
        tflite_class = CLASS_NAMES[tflite_classes[i]]
        tflite_conf = tflite_preds[i][tflite_classes[i]]
        quantized_class = CLASS_NAMES[quantized_classes[i]]
        quantized_conf = quantized_preds[i][quantized_classes[i]]
        
        print(f"\nSample {i+1}:")
        print(f"  Keras:      {keras_class} ({keras_conf:.4f})")
        print(f"  TFLite:     {tflite_class} ({tflite_conf:.4f})")
        print(f"  Quantized:  {quantized_class} ({quantized_conf:.4f})")
    
    return {
        'keras_tflite_agreement': float(keras_tflite_agreement),
        'keras_quantized_agreement': float(keras_quantized_agreement),
        'mean_abs_diff_float32': float(keras_tflite_diff),
        'mean_abs_diff_quantized': float(keras_quantized_diff)
    }


def benchmark_inference(model_path, test_data, model_type="keras"):
    """Benchmark inference speed with multiple runs"""
    print(f"\nBenchmarking {model_type} model ({NUM_INFERENCE_RUNS} runs)...")
    
    times = []
    
    if model_type == "keras":
        model = keras.models.load_model(model_path)
        for _ in range(NUM_INFERENCE_RUNS):
            start = time.time()
            _ = model.predict(test_data, verbose=0)
            times.append(time.time() - start)
    else:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_data = test_data.astype(input_details[0]['dtype'])
        
        for _ in range(NUM_INFERENCE_RUNS):
            start = time.time()
            for i in range(len(test_data)):
                interpreter.set_tensor(input_details[0]['index'], input_data[i:i+1])
                interpreter.invoke()
                _ = interpreter.get_tensor(output_details[0]['index'])
            times.append(time.time() - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'avg_time_per_sample': avg_time / len(test_data),
        'throughput': len(test_data) / avg_time
    }


def run_tests():
    """Run all tests"""
    print("=" * 60)
    print("TensorFlow Lite Model Testing")
    print("=" * 60)
    
    # Check if models exist
    if not Path(KERAS_MODEL_PATH).exists():
        print(f"Error: Keras model not found at {KERAS_MODEL_PATH}")
        print("Please run train_model.py first.")
        return
    
    if not Path(TFLITE_MODEL_PATH).exists():
        print(f"Error: TFLite model not found at {TFLITE_MODEL_PATH}")
        print("Please run convert_to_tflite.py first.")
        return
    
    # Load test data
    test_data = load_test_data()
    
    # Test Keras model
    keras_preds, keras_time = test_keras_model(KERAS_MODEL_PATH, test_data)
    
    # Test TFLite float32 model
    tflite_preds, tflite_time = test_tflite_model(
        TFLITE_MODEL_PATH, test_data, "TFLite (Float32)"
    )
    
    # Test TFLite quantized model
    if Path(TFLITE_QUANTIZED_PATH).exists():
        quantized_preds, quantized_time = test_tflite_model(
            TFLITE_QUANTIZED_PATH, test_data, "TFLite (Quantized)"
        )
    else:
        print(f"\nWarning: Quantized model not found at {TFLITE_QUANTIZED_PATH}")
        quantized_preds = None
        quantized_time = None
    
    # Compare predictions
    if quantized_preds is not None:
        comparison = compare_predictions(keras_preds, tflite_preds, quantized_preds)
    else:
        comparison = None
    
    # Benchmark inference
    print("\n" + "=" * 60)
    print("Performance Benchmarking")
    print("=" * 60)
    
    keras_benchmark = benchmark_inference(KERAS_MODEL_PATH, test_data, "keras")
    tflite_benchmark = benchmark_inference(TFLITE_MODEL_PATH, test_data, "tflite")
    
    if Path(TFLITE_QUANTIZED_PATH).exists():
        quantized_benchmark = benchmark_inference(
            TFLITE_QUANTIZED_PATH, test_data, "tflite_quantized"
        )
    else:
        quantized_benchmark = None
    
    # Print benchmark results
    print("\nKeras Model:")
    print(f"  Average time: {keras_benchmark['avg_time']:.4f}s ± {keras_benchmark['std_time']:.4f}s")
    print(f"  Time per sample: {keras_benchmark['avg_time_per_sample']*1000:.2f} ms")
    print(f"  Throughput: {keras_benchmark['throughput']:.2f} samples/s")
    
    print("\nTFLite (Float32) Model:")
    print(f"  Average time: {tflite_benchmark['avg_time']:.4f}s ± {tflite_benchmark['std_time']:.4f}s")
    print(f"  Time per sample: {tflite_benchmark['avg_time_per_sample']*1000:.2f} ms")
    print(f"  Throughput: {tflite_benchmark['throughput']:.2f} samples/s")
    print(f"  Speedup vs Keras: {keras_benchmark['avg_time']/tflite_benchmark['avg_time']:.2f}x")
    
    if quantized_benchmark:
        print("\nTFLite (Quantized) Model:")
        print(f"  Average time: {quantized_benchmark['avg_time']:.4f}s ± {quantized_benchmark['std_time']:.4f}s")
        print(f"  Time per sample: {quantized_benchmark['avg_time_per_sample']*1000:.2f} ms")
        print(f"  Throughput: {quantized_benchmark['throughput']:.2f} samples/s")
        print(f"  Speedup vs Keras: {keras_benchmark['avg_time']/quantized_benchmark['avg_time']:.2f}x")
        print(f"  Speedup vs TFLite float32: {tflite_benchmark['avg_time']/quantized_benchmark['avg_time']:.2f}x")
    
    # Save test report
    report = {
        'test_timestamp': datetime.now().isoformat(),
        'num_test_samples': NUM_TEST_SAMPLES,
        'num_inference_runs': NUM_INFERENCE_RUNS,
        'keras_benchmark': keras_benchmark,
        'tflite_benchmark': tflite_benchmark,
        'quantized_benchmark': quantized_benchmark,
        'prediction_comparison': comparison,
        'speedup_tflite_vs_keras': float(keras_benchmark['avg_time'] / tflite_benchmark['avg_time']),
        'speedup_quantized_vs_keras': float(keras_benchmark['avg_time'] / quantized_benchmark['avg_time']) if quantized_benchmark else None,
    }
    
    report_path = Path('models/test_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nTest report saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("Testing completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    print(f"TensorFlow version: {tf.__version__}")
    run_tests()

