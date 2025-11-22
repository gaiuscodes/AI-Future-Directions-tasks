"""
Crop Yield Prediction AI Model

This module implements a machine learning model to predict crop yields
based on IoT sensor data and environmental factors.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from datetime import datetime
from pathlib import Path
import joblib


class CropYieldPredictor:
    """
    AI Model for predicting crop yields based on sensor data
    """
    
    def __init__(self, model_type='ensemble'):
        """
        Initialize yield predictor
        
        Args:
            model_type: 'ensemble', 'neural_network', or 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'soil_moisture', 'temperature', 'humidity', 'light_intensity',
            'ph_level', 'nitrogen', 'phosphorus', 'potassium',
            'rainfall', 'wind_speed'
        ]
        self.training_history = None
    
    def create_neural_network(self, input_dim):
        """
        Create a neural network model for yield prediction
        
        Architecture:
        - Input layer: 10 features (sensor readings)
        - Hidden layers: 3 dense layers with dropout
        - Output layer: Single value (yield in kg/hectare)
        """
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='linear')  # Linear for regression
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def generate_synthetic_data(self, n_samples=1000, crop_type='wheat'):
        """
        Generate synthetic training data based on realistic relationships
        
        Args:
            n_samples: Number of samples to generate
            crop_type: Type of crop (affects yield ranges)
            
        Returns:
            DataFrame with features and target (yield)
        """
        np.random.seed(42)
        
        # Crop-specific optimal ranges and yield factors
        crop_configs = {
            'wheat': {
                'optimal_temp': 20,
                'optimal_moisture': 50,
                'base_yield': 3000,  # kg/hectare
                'yield_range': (2000, 5000)
            },
            'corn': {
                'optimal_temp': 25,
                'optimal_moisture': 60,
                'base_yield': 8000,
                'yield_range': (5000, 12000)
            },
            'rice': {
                'optimal_temp': 28,
                'optimal_moisture': 70,
                'base_yield': 4000,
                'yield_range': (2500, 6000)
            },
            'soybean': {
                'optimal_temp': 22,
                'optimal_moisture': 55,
                'base_yield': 2500,
                'yield_range': (1500, 4000)
            }
        }
        
        config = crop_configs.get(crop_type, crop_configs['wheat'])
        
        data = []
        
        for _ in range(n_samples):
            # Generate sensor readings with some correlation
            temp = np.random.normal(config['optimal_temp'], 5)
            moisture = np.random.normal(config['optimal_moisture'], 10)
            humidity = np.random.normal(60, 15)
            light = np.random.uniform(400, 800)
            ph = np.random.normal(6.5, 0.5)
            nitrogen = np.random.uniform(150, 250)
            phosphorus = np.random.uniform(50, 150)
            potassium = np.random.uniform(150, 300)
            rainfall = np.random.exponential(5)
            wind_speed = np.random.exponential(5)
            
            # Calculate yield based on environmental factors
            # This simulates real relationships
            yield_factor = 1.0
            
            # Temperature effect (optimal range)
            temp_effect = 1.0 - abs(temp - config['optimal_temp']) / 20
            temp_effect = max(0.5, min(1.2, temp_effect))
            
            # Moisture effect
            moisture_effect = 1.0 - abs(moisture - config['optimal_moisture']) / 30
            moisture_effect = max(0.6, min(1.1, moisture_effect))
            
            # pH effect (optimal around 6.5-7)
            ph_effect = 1.0 - abs(ph - 6.75) / 2
            ph_effect = max(0.7, min(1.1, ph_effect))
            
            # Nutrient effect (NPK)
            npk_effect = (nitrogen/200 + phosphorus/100 + potassium/225) / 3
            npk_effect = max(0.8, min(1.2, npk_effect))
            
            # Light effect
            light_effect = min(1.0, light / 600)
            
            # Rainfall effect (moderate is good, too much is bad)
            if rainfall < 2:
                rain_effect = 0.9  # Too dry
            elif rainfall > 20:
                rain_effect = 0.85  # Too wet
            else:
                rain_effect = 1.0  # Optimal
            
            # Combine effects
            yield_factor = (temp_effect * 0.25 + 
                          moisture_effect * 0.25 + 
                          ph_effect * 0.15 + 
                          npk_effect * 0.20 + 
                          light_effect * 0.10 + 
                          rain_effect * 0.05)
            
            # Add some randomness
            yield_factor *= np.random.uniform(0.9, 1.1)
            
            # Calculate final yield
            yield_value = config['base_yield'] * yield_factor
            yield_value = np.clip(yield_value, config['yield_range'][0], 
                                config['yield_range'][1])
            
            data.append({
                'soil_moisture': round(moisture, 2),
                'temperature': round(temp, 2),
                'humidity': round(humidity, 2),
                'light_intensity': round(light, 2),
                'ph_level': round(ph, 2),
                'nitrogen': round(nitrogen, 2),
                'phosphorus': round(phosphorus, 2),
                'potassium': round(potassium, 2),
                'rainfall': round(rainfall, 2),
                'wind_speed': round(wind_speed, 2),
                'yield': round(yield_value, 2),
                'crop_type': crop_type
            })
        
        return pd.DataFrame(data)
    
    def prepare_data(self, df):
        """
        Prepare data for training
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            X (features), y (target)
        """
        X = df[self.feature_names].values
        y = df['yield'].values
        
        return X, y
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of epochs (for neural network)
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if self.model_type == 'neural_network':
            # Neural network
            input_dim = X_train_scaled.shape[1]
            self.model = self.create_neural_network(input_dim)
            
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                validation_data = (X_val_scaled, y_val)
            else:
                validation_data = None
            
            history = self.model.fit(
                X_train_scaled, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=32,
                verbose=1
            )
            self.training_history = history.history
        
        elif self.model_type == 'random_forest':
            # Random Forest
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
        
        elif self.model_type == 'ensemble':
            # Ensemble: Random Forest + Gradient Boosting
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            gb = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            
            rf.fit(X_train_scaled, y_train)
            gb.fit(X_train_scaled, y_train)
            
            self.model = {'random_forest': rf, 'gradient_boosting': gb}
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Feature array
            
        Returns:
            Predicted yields
        """
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'neural_network':
            predictions = self.model.predict(X_scaled, verbose=0).flatten()
        elif self.model_type == 'random_forest':
            predictions = self.model.predict(X_scaled)
        elif self.model_type == 'ensemble':
            rf_pred = self.model['random_forest'].predict(X_scaled)
            gb_pred = self.model['gradient_boosting'].predict(X_scaled)
            predictions = (rf_pred + gb_pred) / 2
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with metrics
        """
        predictions = self.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Calculate percentage error
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'mape': float(mape)
        }
        
        return metrics, predictions
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_dir = Path(filepath).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if self.model_type == 'neural_network':
            self.model.save(filepath)
        elif self.model_type == 'random_forest':
            joblib.dump(self.model, filepath)
        elif self.model_type == 'ensemble':
            joblib.dump(self.model, filepath)
        
        # Save scaler
        scaler_path = filepath.replace('.h5', '_scaler.pkl').replace('.pkl', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to: {filepath}")
        print(f"Scaler saved to: {scaler_path}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        if self.model_type == 'neural_network':
            self.model = keras.models.load_model(filepath)
        else:
            self.model = joblib.load(filepath)
        
        # Load scaler
        scaler_path = filepath.replace('.h5', '_scaler.pkl').replace('.pkl', '_scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        
        print(f"Model loaded from: {filepath}")


def train_yield_model(crop_type='wheat', model_type='ensemble', n_samples=2000):
    """
    Train a crop yield prediction model
    
    Args:
        crop_type: Type of crop
        model_type: Type of model to train
        n_samples: Number of training samples
    """
    print("=" * 60)
    print(f"Crop Yield Prediction Model Training - {crop_type.upper()}")
    print("=" * 60)
    
    # Initialize predictor
    predictor = CropYieldPredictor(model_type=model_type)
    
    # Generate training data
    print(f"\nGenerating {n_samples} training samples...")
    df = predictor.generate_synthetic_data(n_samples=n_samples, crop_type=crop_type)
    
    print(f"\nDataset statistics:")
    print(df.describe())
    
    # Prepare data
    X, y = predictor.prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    print(f"\nTraining {model_type} model...")
    predictor.train(X_train, y_train, X_val, y_val, epochs=50)
    
    # Evaluate
    print("\nEvaluating on test set...")
    metrics, predictions = predictor.evaluate(X_test, y_test)
    
    print("\n" + "=" * 60)
    print("Model Performance Metrics")
    print("=" * 60)
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"RMSE: {metrics['rmse']:.2f} kg/hectare")
    print(f"MAE: {metrics['mae']:.2f} kg/hectare")
    print(f"MAPE: {metrics['mape']:.2f}%")
    
    # Save model
    model_path = f"models/yield_predictor_{crop_type}_{model_type}.pkl"
    if model_type == 'neural_network':
        model_path = model_path.replace('.pkl', '.h5')
    
    predictor.save_model(model_path)
    
    # Save metrics
    metrics_path = f"models/yield_metrics_{crop_type}_{model_type}.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'crop_type': crop_type,
            'model_type': model_type,
            'n_samples': n_samples,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nMetrics saved to: {metrics_path}")
    
    return predictor, metrics


if __name__ == '__main__':
    # Train models for different crops
    crops = ['wheat', 'corn', 'rice', 'soybean']
    model_types = ['ensemble', 'random_forest']
    
    for crop in crops:
        for model_type in model_types:
            print("\n" + "=" * 60)
            train_yield_model(crop_type=crop, model_type=model_type, n_samples=2000)

