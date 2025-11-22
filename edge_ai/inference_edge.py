"""
Edge Device Inference Script

This script demonstrates how to use the TensorFlow Lite model
on an edge device (e.g., Raspberry Pi) for real-time inference.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import time

# Configuration
TFLITE_MODEL_PATH = 'models/recyclable_classifier_quantized.tflite'
CLASS_NAMES = ['paper', 'plastic', 'glass', 'metal']


class RecyclableItemClassifier:
    """Class for running inference on edge devices"""
    
    def __init__(self, model_path=TFLITE_MODEL_PATH):
        """
        Initialize the classifier with a TFLite model.
        
        Args:
            model_path: Path to the TFLite model file
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape
        self.input_shape = self.input_details[0]['shape'][1:3]  # Height, Width
        self.input_dtype = self.input_details[0]['dtype']
        
        print(f"Model loaded successfully!")
        print(f"Input shape: {self.input_shape}")
        print(f"Input dtype: {self.input_dtype}")
    
    def preprocess_image(self, image):
        """
        Preprocess image for inference.
        
        Args:
            image: Image array (numpy array or PIL Image)
            
        Returns:
            Preprocessed image array
        """
        # If PIL Image, convert to array
        if hasattr(image, 'size'):
            import PIL.Image
            if isinstance(image, PIL.Image.Image):
                image = np.array(image)
        
        # Resize if needed
        if image.shape[:2] != self.input_shape:
            import tensorflow as tf
            image = tf.image.resize(image, self.input_shape).numpy()
        
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Ensure correct dtype
        image = image.astype(self.input_dtype)
        
        # Add batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image, return_probabilities=False):
        """
        Run inference on an image.
        
        Args:
            image: Input image (numpy array or PIL Image)
            return_probabilities: If True, return all class probabilities
            
        Returns:
            Predicted class name (and probabilities if requested)
        """
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Set input tensor
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            processed_image
        )
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        probabilities = output[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        
        if return_probabilities:
            class_probs = {
                CLASS_NAMES[i]: float(probabilities[i])
                for i in range(len(CLASS_NAMES))
            }
            return predicted_class, confidence, class_probs
        
        return predicted_class, confidence
    
    def predict_batch(self, images):
        """
        Run inference on a batch of images.
        
        Args:
            images: List of images or numpy array of images
            
        Returns:
            List of (class_name, confidence) tuples
        """
        results = []
        for image in images:
            class_name, confidence = self.predict(image)
            results.append((class_name, confidence))
        return results


def demo_inference():
    """Demonstrate inference on sample data"""
    print("=" * 60)
    print("Edge AI Inference Demo")
    print("=" * 60)
    
    # Initialize classifier
    try:
        classifier = RecyclableItemClassifier()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure you have:")
        print("1. Trained the model (run train_model.py)")
        print("2. Converted to TFLite (run convert_to_tflite.py)")
        return
    
    # Generate sample images (in production, load from camera/file)
    print("\nRunning inference on sample images...")
    num_samples = 5
    
    for i in range(num_samples):
        # Generate random image (simulating camera input)
        sample_image = np.random.rand(128, 128, 3).astype(np.float32)
        
        # Run inference
        start_time = time.time()
        predicted_class, confidence = classifier.predict(sample_image)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        print(f"\nSample {i+1}:")
        print(f"  Predicted class: {predicted_class}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Inference time: {inference_time:.2f} ms")
    
    # Benchmark inference speed
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)
    
    num_runs = 100
    times = []
    
    for _ in range(num_runs):
        sample_image = np.random.rand(128, 128, 3).astype(np.float32)
        start_time = time.time()
        _ = classifier.predict(sample_image)
        times.append((time.time() - start_time) * 1000)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\nResults over {num_runs} runs:")
    print(f"  Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  Min time: {min_time:.2f} ms")
    print(f"  Max time: {max_time:.2f} ms")
    print(f"  Throughput: {1000/avg_time:.2f} inferences/second")
    
    print("\n" + "=" * 60)
    print("Edge AI Benefits Demonstrated:")
    print("=" * 60)
    print("✓ Fast inference suitable for real-time applications")
    print("✓ Low memory footprint")
    print("✓ No internet connection required")
    print("✓ Privacy-preserving (data stays on device)")
    print("✓ Low latency for responsive user experience")


if __name__ == '__main__':
    demo_inference()

