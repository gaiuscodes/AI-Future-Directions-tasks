"""
Smart Agriculture Simulation System

This module simulates a complete smart agriculture system with IoT sensors
and AI-powered yield prediction.
"""

import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from sensors import SensorNetwork, IoTDevice
from yield_prediction_model import CropYieldPredictor
import matplotlib.pyplot as plt
from collections import deque


class SmartAgricultureSimulator:
    """
    Complete simulation system for smart agriculture
    """
    
    def __init__(self, num_devices=3, crop_type='wheat', model_type='ensemble'):
        """
        Initialize simulator
        
        Args:
            num_devices: Number of IoT devices to simulate
            crop_type: Type of crop being monitored
            model_type: AI model type for predictions
        """
        self.num_devices = num_devices
        self.crop_type = crop_type
        self.model_type = model_type
        
        # Initialize sensor network
        self.sensor_network = SensorNetwork()
        self._setup_devices()
        
        # Initialize AI model
        self.yield_predictor = None
        self._load_or_train_model()
        
        # Data storage
        self.historical_data = deque(maxlen=1000)  # Store last 1000 readings
        self.predictions_history = []
        
        # Statistics
        self.stats = {
            'total_readings': 0,
            'total_predictions': 0,
            'avg_yield_prediction': 0,
            'alerts_triggered': 0
        }
    
    def _setup_devices(self):
        """Setup IoT devices in the network"""
        # All sensors available
        all_sensors = [
            'soil_moisture', 'temperature', 'humidity', 'light_intensity',
            'ph_level', 'npk', 'rainfall', 'wind_speed'
        ]
        
        # Create devices at different locations
        base_lat = 40.7128
        base_lon = -74.0060
        
        for i in range(self.num_devices):
            device_id = f"AGRI_{i+1:03d}"
            # Spread devices across a field
            location = (
                base_lat + (i * 0.001),
                base_lon + (i * 0.001)
            )
            
            # Each device has a subset of sensors
            if i == 0:
                sensors = all_sensors  # Full sensor suite
            elif i == 1:
                sensors = ['soil_moisture', 'temperature', 'humidity', 'npk']
            else:
                sensors = ['soil_moisture', 'temperature', 'light_intensity', 'ph_level']
            
            device = IoTDevice(device_id, location, sensors)
            self.sensor_network.add_device(device)
        
        print(f"✓ Initialized {self.num_devices} IoT devices")
    
    def _load_or_train_model(self):
        """Load existing model or train a new one"""
        model_path = f"models/yield_predictor_{self.crop_type}_{self.model_type}.pkl"
        if self.model_type == 'neural_network':
            model_path = model_path.replace('.pkl', '.h5')
        
        if Path(model_path).exists():
            print(f"✓ Loading existing model: {model_path}")
            self.yield_predictor = CropYieldPredictor(model_type=self.model_type)
            self.yield_predictor.load_model(model_path)
        else:
            print(f"⚠ Model not found. Training new model...")
            from yield_prediction_model import train_yield_model
            self.yield_predictor, _ = train_yield_model(
                crop_type=self.crop_type,
                model_type=self.model_type,
                n_samples=2000
            )
    
    def collect_sensor_data(self):
        """Collect data from all sensors"""
        readings = self.sensor_network.collect_all_readings()
        ml_data = self.sensor_network.get_readings_dataframe_format()
        
        # Store historical data
        self.historical_data.append({
            'timestamp': datetime.now(),
            'data': ml_data
        })
        
        self.stats['total_readings'] += 1
        
        return ml_data
    
    def predict_yield(self, sensor_data):
        """
        Predict crop yield based on sensor data
        
        Args:
            sensor_data: Dictionary with sensor readings
            
        Returns:
            Predicted yield and confidence
        """
        # Prepare feature vector
        features = np.array([[
            sensor_data.get('soil_moisture', 50),
            sensor_data.get('temperature', 25),
            sensor_data.get('humidity', 60),
            sensor_data.get('light_intensity', 500),
            sensor_data.get('ph_level', 6.5),
            sensor_data.get('nitrogen', 200),
            sensor_data.get('phosphorus', 100),
            sensor_data.get('potassium', 225),
            sensor_data.get('rainfall', 5),
            sensor_data.get('wind_speed', 5)
        ]])
        
        # Make prediction
        prediction = self.yield_predictor.predict(features)[0]
        
        # Store prediction
        self.predictions_history.append({
            'timestamp': datetime.now(),
            'prediction': prediction,
            'sensor_data': sensor_data
        })
        
        self.stats['total_predictions'] += 1
        self.stats['avg_yield_prediction'] = np.mean([
            p['prediction'] for p in self.predictions_history[-100:]
        ])
        
        return prediction
    
    def check_alerts(self, sensor_data, yield_prediction):
        """
        Check for conditions that require alerts
        
        Args:
            sensor_data: Current sensor readings
            yield_prediction: Predicted yield
            
        Returns:
            List of alerts
        """
        alerts = []
        
        # Check soil moisture
        moisture = sensor_data.get('soil_moisture', 50)
        if moisture < 30:
            alerts.append({
                'type': 'warning',
                'message': f'Low soil moisture: {moisture:.1f}% - Irrigation needed',
                'severity': 'high'
            })
        elif moisture > 80:
            alerts.append({
                'type': 'warning',
                'message': f'High soil moisture: {moisture:.1f}% - Risk of waterlogging',
                'severity': 'medium'
            })
        
        # Check temperature
        temp = sensor_data.get('temperature', 25)
        if temp < 5:
            alerts.append({
                'type': 'warning',
                'message': f'Low temperature: {temp:.1f}°C - Risk of frost damage',
                'severity': 'high'
            })
        elif temp > 35:
            alerts.append({
                'type': 'warning',
                'message': f'High temperature: {temp:.1f}°C - Heat stress risk',
                'severity': 'medium'
            })
        
        # Check pH
        ph = sensor_data.get('ph_level', 6.5)
        if ph < 5.5 or ph > 8.0:
            alerts.append({
                'type': 'warning',
                'message': f'Suboptimal pH: {ph:.1f} - Soil amendment needed',
                'severity': 'medium'
            })
        
        # Check nutrients
        nitrogen = sensor_data.get('nitrogen', 200)
        if nitrogen < 100:
            alerts.append({
                'type': 'warning',
                'message': f'Low nitrogen: {nitrogen:.1f} mg/kg - Fertilization needed',
                'severity': 'high'
            })
        
        # Check yield prediction
        if yield_prediction < 2000:  # Threshold depends on crop
            alerts.append({
                'type': 'info',
                'message': f'Low yield prediction: {yield_prediction:.0f} kg/hectare',
                'severity': 'low'
            })
        
        if alerts:
            self.stats['alerts_triggered'] += len(alerts)
        
        return alerts
    
    def generate_recommendations(self, sensor_data, yield_prediction):
        """
        Generate actionable recommendations
        
        Args:
            sensor_data: Current sensor readings
            yield_prediction: Predicted yield
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        moisture = sensor_data.get('soil_moisture', 50)
        if moisture < 40:
            recommendations.append({
                'action': 'irrigation',
                'priority': 'high',
                'message': 'Start irrigation system to maintain optimal soil moisture (40-60%)',
                'duration': '30-60 minutes'
            })
        
        nitrogen = sensor_data.get('nitrogen', 200)
        if nitrogen < 150:
            recommendations.append({
                'action': 'fertilization',
                'priority': 'medium',
                'message': 'Apply nitrogen fertilizer to improve nutrient levels',
                'amount': '50-100 kg/hectare'
            })
        
        ph = sensor_data.get('ph_level', 6.5)
        if ph < 6.0:
            recommendations.append({
                'action': 'soil_amendment',
                'priority': 'low',
                'message': 'Consider adding lime to raise soil pH',
                'amount': '2-5 tons/hectare'
            })
        
        return recommendations
    
    def run_simulation_step(self):
        """Run one simulation step"""
        # Collect sensor data
        sensor_data = self.collect_sensor_data()
        
        # Predict yield
        yield_prediction = self.predict_yield(sensor_data)
        
        # Check alerts
        alerts = self.check_alerts(sensor_data, yield_prediction)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(sensor_data, yield_prediction)
        
        return {
            'timestamp': datetime.now(),
            'sensor_data': sensor_data,
            'yield_prediction': yield_prediction,
            'alerts': alerts,
            'recommendations': recommendations
        }
    
    def run_simulation(self, duration_minutes=60, interval_seconds=60):
        """
        Run continuous simulation
        
        Args:
            duration_minutes: How long to run simulation
            interval_seconds: Time between readings
        """
        print("\n" + "=" * 60)
        print("Starting Smart Agriculture Simulation")
        print("=" * 60)
        print(f"Crop Type: {self.crop_type}")
        print(f"Number of Devices: {self.num_devices}")
        print(f"AI Model: {self.model_type}")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Interval: {interval_seconds} seconds")
        print("=" * 60 + "\n")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        step = 0
        
        results = []
        
        try:
            while datetime.now() < end_time:
                step += 1
                result = self.run_simulation_step()
                results.append(result)
                
                # Print status
                print(f"\n[Step {step}] {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Yield Prediction: {result['yield_prediction']:.0f} kg/hectare")
                
                if result['alerts']:
                    print(f"  ⚠ Alerts: {len(result['alerts'])}")
                    for alert in result['alerts'][:2]:  # Show first 2
                        print(f"    - {alert['message']}")
                
                if result['recommendations']:
                    print(f"  💡 Recommendations: {len(result['recommendations'])}")
                    for rec in result['recommendations'][:2]:  # Show first 2
                        print(f"    - {rec['message']}")
                
                # Wait for next interval
                if datetime.now() < end_time:
                    time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            print("\n\nSimulation interrupted by user")
        
        # Save results
        self._save_simulation_results(results)
        
        # Print summary
        self._print_summary()
        
        return results
    
    def _save_simulation_results(self, results):
        """Save simulation results to file"""
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        # Save as JSON
        json_path = output_dir / f'simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, default=str, indent=2)
        
        # Save as CSV
        csv_data = []
        for result in results:
            row = {
                'timestamp': result['timestamp'],
                'yield_prediction': result['yield_prediction'],
                'soil_moisture': result['sensor_data'].get('soil_moisture'),
                'temperature': result['sensor_data'].get('temperature'),
                'humidity': result['sensor_data'].get('humidity'),
                'light_intensity': result['sensor_data'].get('light_intensity'),
                'ph_level': result['sensor_data'].get('ph_level'),
                'nitrogen': result['sensor_data'].get('nitrogen'),
                'num_alerts': len(result['alerts']),
                'num_recommendations': len(result['recommendations'])
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_path = output_dir / f'simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"\n✓ Results saved to:")
        print(f"  - {json_path}")
        print(f"  - {csv_path}")
    
    def _print_summary(self):
        """Print simulation summary"""
        print("\n" + "=" * 60)
        print("Simulation Summary")
        print("=" * 60)
        print(f"Total Readings: {self.stats['total_readings']}")
        print(f"Total Predictions: {self.stats['total_predictions']}")
        print(f"Average Yield Prediction: {self.stats['avg_yield_prediction']:.0f} kg/hectare")
        print(f"Alerts Triggered: {self.stats['alerts_triggered']}")
        
        if self.predictions_history:
            predictions = [p['prediction'] for p in self.predictions_history]
            print(f"\nYield Prediction Statistics:")
            print(f"  Min: {min(predictions):.0f} kg/hectare")
            print(f"  Max: {max(predictions):.0f} kg/hectare")
            print(f"  Mean: {np.mean(predictions):.0f} kg/hectare")
            print(f"  Std: {np.std(predictions):.0f} kg/hectare")
    
    def visualize_results(self):
        """Create visualizations of simulation results"""
        if not self.predictions_history:
            print("No data to visualize. Run simulation first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Yield predictions over time
        timestamps = [p['timestamp'] for p in self.predictions_history]
        yields = [p['prediction'] for p in self.predictions_history]
        
        axes[0, 0].plot(timestamps, yields, 'b-', linewidth=2)
        axes[0, 0].set_title('Yield Predictions Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Predicted Yield (kg/hectare)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Sensor data trends
        if self.historical_data:
            recent_data = list(self.historical_data)[-50:]  # Last 50 readings
            timestamps_sensor = [d['timestamp'] for d in recent_data]
            moisture = [d['data'].get('soil_moisture', 0) for d in recent_data]
            temperature = [d['data'].get('temperature', 0) for d in recent_data]
            
            ax2 = axes[0, 1]
            ax2_twin = ax2.twinx()
            
            line1 = ax2.plot(timestamps_sensor, moisture, 'g-', label='Soil Moisture', linewidth=2)
            line2 = ax2_twin.plot(timestamps_sensor, temperature, 'r-', label='Temperature', linewidth=2)
            
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Soil Moisture (%)', color='g')
            ax2_twin.set_ylabel('Temperature (°C)', color='r')
            ax2.set_title('Sensor Data Trends', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='upper left')
        
        # Yield distribution
        axes[1, 0].hist(yields, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Yield Prediction Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Predicted Yield (kg/hectare)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Feature importance (if available)
        axes[1, 1].text(0.5, 0.5, 'Feature Importance\n(Requires model analysis)', 
                       ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Feature Importance', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        output_path = Path('results') / f'simulation_visualization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")
        plt.close()


if __name__ == '__main__':
    # Create and run simulation
    simulator = SmartAgricultureSimulator(
        num_devices=3,
        crop_type='wheat',
        model_type='ensemble'
    )
    
    # Run simulation for 5 minutes with 30-second intervals
    results = simulator.run_simulation(duration_minutes=5, interval_seconds=30)
    
    # Create visualizations
    simulator.visualize_results()

