"""
Complete Pipeline Script

This script runs the complete Edge AI pipeline:
1. Train the model
2. Convert to TensorFlow Lite
3. Test the models
4. Run edge inference demo

Run this script to execute the entire workflow.
"""

import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print("\n" + "=" * 60)
    print(description)
    print("=" * 60)
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=False
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError: {description} failed!")
        print(f"Exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False


def main():
    """Run the complete pipeline"""
    print("=" * 60)
    print("Edge AI Prototype - Complete Pipeline")
    print("=" * 60)
    print("\nThis script will:")
    print("1. Train the model")
    print("2. Convert to TensorFlow Lite")
    print("3. Test the models")
    print("4. Run edge inference demo")
    print("\nPress Ctrl+C to cancel at any time.\n")
    
    # Check if we're in the right directory
    if not Path('train_model.py').exists():
        print("Error: Please run this script from the edge_ai directory")
        sys.exit(1)
    
    steps = [
        ("python train_model.py", "Step 1: Training the model"),
        ("python convert_to_tflite.py", "Step 2: Converting to TensorFlow Lite"),
        ("python test_tflite.py", "Step 3: Testing the models"),
        ("python inference_edge.py", "Step 4: Edge inference demo"),
    ]
    
    for command, description in steps:
        success = run_command(command, description)
        if not success:
            print(f"\nPipeline stopped at: {description}")
            print("Please fix the error and try again.")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print("\nAll models and reports are available in the 'models/' directory.")
    print("See EDGE_AI_REPORT.md for detailed results and analysis.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)

