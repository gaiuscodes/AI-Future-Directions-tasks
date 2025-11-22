"""
Visualization Script

This script creates visualizations from the training and test results.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_training_history():
    """Plot training history from JSON file"""
    history_path = Path('models/training_history.json')
    
    if not history_path.exists():
        print(f"Training history not found at {history_path}")
        print("Please run train_model.py first.")
        return
    
    history = load_json(history_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Accuracy plot
    axes[0].plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0].axhline(y=history['test_accuracy'], color='g', linestyle='--', 
                    label=f"Test Accuracy: {history['test_accuracy']:.4f}", linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[1].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path('models/training_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training analysis saved to: {output_path}")
    plt.close()

def plot_performance_comparison():
    """Plot performance comparison between models"""
    test_path = Path('models/test_report.json')
    
    if not test_path.exists():
        print(f"Test report not found at {test_path}")
        print("Please run test_tflite.py first.")
        return
    
    report = load_json(test_path)
    
    # Extract data
    keras_time = report['keras_benchmark']['avg_time_per_sample'] * 1000  # ms
    tflite_time = report['tflite_benchmark']['avg_time_per_sample'] * 1000  # ms
    
    models = ['Keras\n(Float32)', 'TFLite\n(Float32)']
    times = [keras_time, tflite_time]
    colors = ['#FF6B6B', '#4ECDC4']
    
    if report.get('quantized_benchmark'):
        quantized_time = report['quantized_benchmark']['avg_time_per_sample'] * 1000
        models.append('TFLite\n(Quantized)')
        times.append(quantized_time)
        colors.append('#95E1D3')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Inference time comparison
    bars = axes[0].bar(models, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Inference Time (ms)', fontsize=12)
    axes[0].set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f} ms',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Throughput comparison
    keras_throughput = report['keras_benchmark']['throughput']
    tflite_throughput = report['tflite_benchmark']['throughput']
    throughputs = [keras_throughput, tflite_throughput]
    
    if report.get('quantized_benchmark'):
        quantized_throughput = report['quantized_benchmark']['throughput']
        throughputs.append(quantized_throughput)
    
    bars = axes[1].bar(models, throughputs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Throughput (samples/second)', fontsize=12)
    axes[1].set_title('Throughput Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, throughput in zip(bars, throughputs):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{throughput:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path('models/performance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Performance comparison saved to: {output_path}")
    plt.close()

def plot_model_size_comparison():
    """Plot model size comparison"""
    conversion_path = Path('models/conversion_report.json')
    
    if not conversion_path.exists():
        print(f"Conversion report not found at {conversion_path}")
        print("Please run convert_to_tflite.py first.")
        return
    
    report = load_json(conversion_path)
    
    models = ['Keras\n(.h5)', 'TFLite\n(Float32)', 'TFLite\n(Quantized)']
    sizes = [
        report['original_model_size_mb'],
        report['tflite_float32_size_mb'],
        report['tflite_quantized_size_mb']
    ]
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(models, sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Model Size (MB)', fontsize=12)
    ax.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels and reduction percentages
    for i, (bar, size) in enumerate(zip(bars, sizes)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{size:.2f} MB',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        if i > 0:
            reduction = (1 - size / sizes[0]) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height / 2,
                   f'↓{reduction:.1f}%',
                   ha='center', va='center', fontsize=9, 
                   color='white', fontweight='bold')
    
    plt.tight_layout()
    output_path = Path('models/size_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Size comparison saved to: {output_path}")
    plt.close()

def create_summary_dashboard():
    """Create a comprehensive summary dashboard"""
    print("\n" + "=" * 60)
    print("Creating Visualization Dashboard")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    Path('models').mkdir(exist_ok=True)
    
    # Generate all visualizations
    plot_training_history()
    plot_performance_comparison()
    plot_model_size_comparison()
    
    print("\n" + "=" * 60)
    print("All visualizations created successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - models/training_analysis.png")
    print("  - models/performance_comparison.png")
    print("  - models/size_comparison.png")

if __name__ == '__main__':
    create_summary_dashboard()

