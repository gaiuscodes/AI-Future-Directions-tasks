"""
Main Entry Point for Smart Agriculture Simulation System

This script provides a command-line interface to run the complete system.
"""

import argparse
import sys
from pathlib import Path
from simulation_system import SmartAgricultureSimulator
from yield_prediction_model import train_yield_model


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Smart Agriculture Simulation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models for all crops
  python main.py --train

  # Run simulation with default settings
  python main.py --simulate

  # Run simulation for specific crop
  python main.py --simulate --crop corn --duration 30

  # Train and simulate
  python main.py --train --simulate --crop wheat
        """
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train AI models for yield prediction'
    )
    
    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Run smart agriculture simulation'
    )
    
    parser.add_argument(
        '--crop',
        type=str,
        default='wheat',
        choices=['wheat', 'corn', 'rice', 'soybean'],
        help='Crop type (default: wheat)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='ensemble',
        choices=['ensemble', 'random_forest', 'neural_network'],
        help='AI model type (default: ensemble)'
    )
    
    parser.add_argument(
        '--devices',
        type=int,
        default=3,
        help='Number of IoT devices (default: 3)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=5,
        help='Simulation duration in minutes (default: 5)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Data collection interval in seconds (default: 30)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=2000,
        help='Number of training samples (default: 2000)'
    )
    
    args = parser.parse_args()
    
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    # Train models if requested
    if args.train:
        print("=" * 60)
        print("Training AI Models")
        print("=" * 60)
        
        crops = ['wheat', 'corn', 'rice', 'soybean'] if args.crop == 'all' else [args.crop]
        
        for crop in crops:
            print(f"\nTraining {crop} model...")
            try:
                train_yield_model(
                    crop_type=crop,
                    model_type=args.model,
                    n_samples=args.samples
                )
            except Exception as e:
                print(f"Error training {crop} model: {e}")
                sys.exit(1)
        
        print("\n✓ Model training completed!")
    
    # Run simulation if requested
    if args.simulate:
        print("\n" + "=" * 60)
        print("Starting Simulation")
        print("=" * 60)
        
        try:
            simulator = SmartAgricultureSimulator(
                num_devices=args.devices,
                crop_type=args.crop,
                model_type=args.model
            )
            
            results = simulator.run_simulation(
                duration_minutes=args.duration,
                interval_seconds=args.interval
            )
            
            # Generate visualizations
            print("\nGenerating visualizations...")
            simulator.visualize_results()
            
            print("\n✓ Simulation completed successfully!")
            
        except Exception as e:
            print(f"\nError during simulation: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # If neither train nor simulate, show help
    if not args.train and not args.simulate:
        parser.print_help()
        print("\n" + "=" * 60)
        print("Quick Start:")
        print("=" * 60)
        print("1. Train models:    python main.py --train")
        print("2. Run simulation:   python main.py --simulate")
        print("3. Both:            python main.py --train --simulate")
        print("=" * 60)


if __name__ == '__main__':
    main()

