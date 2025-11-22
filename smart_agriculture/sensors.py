"""
Smart Agriculture IoT Sensors Module

This module defines all sensors needed for a smart agriculture system
and simulates sensor data collection.
"""

from dataclasses import dataclass
from typing import Dict, List
import random
import numpy as np
from datetime import datetime, timedelta
import json


@dataclass
class SensorReading:
    """Represents a single sensor reading"""
    timestamp: datetime
    sensor_id: str
    sensor_type: str
    value: float
    unit: str
    location: tuple  # (latitude, longitude) or (x, y) coordinates


class SoilMoistureSensor:
    """
    Soil Moisture Sensor
    Measures volumetric water content in soil (0-100%)
    """
    SENSOR_TYPE = "soil_moisture"
    UNIT = "percentage"
    MIN_VALUE = 0.0
    MAX_VALUE = 100.0
    
    @staticmethod
    def read(optimal_range=(40, 60), noise_level=2.0):
        """
        Simulate soil moisture reading
        
        Args:
            optimal_range: Optimal moisture range (min, max)
            noise_level: Random noise level
            
        Returns:
            Sensor reading value
        """
        base_value = np.random.uniform(optimal_range[0], optimal_range[1])
        noise = np.random.normal(0, noise_level)
        value = np.clip(base_value + noise, 0, 100)
        return round(value, 2)


class TemperatureSensor:
    """
    Temperature Sensor
    Measures air and soil temperature in Celsius
    """
    SENSOR_TYPE = "temperature"
    UNIT = "celsius"
    MIN_VALUE = -20.0
    MAX_VALUE = 50.0
    
    @staticmethod
    def read(optimal_temp=25.0, daily_variation=5.0, noise_level=0.5):
        """
        Simulate temperature reading with daily variation
        
        Args:
            optimal_temp: Optimal temperature
            daily_variation: Daily temperature variation
            noise_level: Random noise level
            
        Returns:
            Temperature reading in Celsius
        """
        # Simulate daily temperature cycle
        hour = datetime.now().hour
        daily_factor = np.sin((hour - 6) * np.pi / 12)  # Peak at 2 PM
        base_value = optimal_temp + daily_variation * daily_factor
        noise = np.random.normal(0, noise_level)
        value = np.clip(base_value + noise, -20, 50)
        return round(value, 2)


class HumiditySensor:
    """
    Air Humidity Sensor
    Measures relative humidity (0-100%)
    """
    SENSOR_TYPE = "humidity"
    UNIT = "percentage"
    MIN_VALUE = 0.0
    MAX_VALUE = 100.0
    
    @staticmethod
    def read(optimal_range=(40, 70), noise_level=3.0):
        """
        Simulate humidity reading
        
        Args:
            optimal_range: Optimal humidity range
            noise_level: Random noise level
            
        Returns:
            Humidity reading
        """
        base_value = np.random.uniform(optimal_range[0], optimal_range[1])
        noise = np.random.normal(0, noise_level)
        value = np.clip(base_value + noise, 0, 100)
        return round(value, 2)


class LightIntensitySensor:
    """
    Light Intensity Sensor (PAR - Photosynthetically Active Radiation)
    Measures light intensity in micromoles per square meter per second
    """
    SENSOR_TYPE = "light_intensity"
    UNIT = "μmol/m²/s"
    MIN_VALUE = 0.0
    MAX_VALUE = 2000.0
    
    @staticmethod
    def read(optimal_range=(400, 800), noise_level=20.0):
        """
        Simulate light intensity reading with daily cycle
        
        Args:
            optimal_range: Optimal light intensity range
            noise_level: Random noise level
            
        Returns:
            Light intensity reading
        """
        hour = datetime.now().hour
        # Simulate sunlight cycle (0 at night, peak at noon)
        if 6 <= hour <= 18:
            sunlight_factor = np.sin((hour - 6) * np.pi / 12)
        else:
            sunlight_factor = 0
        
        base_value = optimal_range[0] + (optimal_range[1] - optimal_range[0]) * sunlight_factor
        noise = np.random.normal(0, noise_level)
        value = np.clip(base_value + noise, 0, 2000)
        return round(value, 2)


class PHLevelSensor:
    """
    Soil pH Sensor
    Measures soil acidity/alkalinity (0-14 scale)
    """
    SENSOR_TYPE = "ph_level"
    UNIT = "pH"
    MIN_VALUE = 0.0
    MAX_VALUE = 14.0
    
    @staticmethod
    def read(optimal_range=(6.0, 7.5), noise_level=0.1):
        """
        Simulate pH reading
        
        Args:
            optimal_range: Optimal pH range
            noise_level: Random noise level
            
        Returns:
            pH reading
        """
        base_value = np.random.uniform(optimal_range[0], optimal_range[1])
        noise = np.random.normal(0, noise_level)
        value = np.clip(base_value + noise, 0, 14)
        return round(value, 2)


class NPKSensor:
    """
    NPK Sensor (Nitrogen, Phosphorus, Potassium)
    Measures nutrient levels in soil
    """
    SENSOR_TYPE = "npk"
    UNIT = "mg/kg"
    MIN_VALUE = 0.0
    MAX_VALUE = 500.0
    
    @staticmethod
    def read(optimal_n=(150, 250), optimal_p=(50, 150), optimal_k=(150, 300), noise_level=5.0):
        """
        Simulate NPK readings
        
        Args:
            optimal_n: Optimal nitrogen range
            optimal_p: Optimal phosphorus range
            optimal_k: Optimal potassium range
            noise_level: Random noise level
            
        Returns:
            Dictionary with N, P, K values
        """
        n = np.clip(np.random.uniform(optimal_n[0], optimal_n[1]) + 
                   np.random.normal(0, noise_level), 0, 500)
        p = np.clip(np.random.uniform(optimal_p[0], optimal_p[1]) + 
                   np.random.normal(0, noise_level), 0, 500)
        k = np.clip(np.random.uniform(optimal_k[0], optimal_k[1]) + 
                   np.random.normal(0, noise_level), 0, 500)
        
        return {
            'nitrogen': round(n, 2),
            'phosphorus': round(p, 2),
            'potassium': round(k, 2)
        }


class RainfallSensor:
    """
    Rainfall Sensor
    Measures precipitation in millimeters
    """
    SENSOR_TYPE = "rainfall"
    UNIT = "mm"
    MIN_VALUE = 0.0
    MAX_VALUE = 100.0
    
    @staticmethod
    def read(daily_rainfall=0.0, noise_level=0.5):
        """
        Simulate rainfall reading
        
        Args:
            daily_rainfall: Daily rainfall amount
            noise_level: Random noise level
            
        Returns:
            Rainfall reading in mm
        """
        value = max(0, daily_rainfall + np.random.normal(0, noise_level))
        return round(value, 2)


class WindSpeedSensor:
    """
    Wind Speed Sensor
    Measures wind speed in km/h
    """
    SENSOR_TYPE = "wind_speed"
    UNIT = "km/h"
    MIN_VALUE = 0.0
    MAX_VALUE = 100.0
    
    @staticmethod
    def read(avg_wind_speed=5.0, noise_level=2.0):
        """
        Simulate wind speed reading
        
        Args:
            avg_wind_speed: Average wind speed
            noise_level: Random noise level
            
        Returns:
            Wind speed reading
        """
        value = max(0, avg_wind_speed + np.random.normal(0, noise_level))
        return round(value, 2)


class IoTDevice:
    """
    Represents an IoT device with multiple sensors
    """
    
    def __init__(self, device_id: str, location: tuple, sensors: List[str]):
        """
        Initialize IoT device
        
        Args:
            device_id: Unique device identifier
            location: (latitude, longitude) coordinates
            sensors: List of sensor types to include
        """
        self.device_id = device_id
        self.location = location
        self.sensors = sensors
        self.sensor_instances = self._initialize_sensors()
    
    def _initialize_sensors(self):
        """Initialize sensor instances"""
        sensor_map = {
            'soil_moisture': SoilMoistureSensor,
            'temperature': TemperatureSensor,
            'humidity': HumiditySensor,
            'light_intensity': LightIntensitySensor,
            'ph_level': PHLevelSensor,
            'npk': NPKSensor,
            'rainfall': RainfallSensor,
            'wind_speed': WindSpeedSensor,
        }
        
        return {sensor: sensor_map[sensor]() for sensor in self.sensors 
                if sensor in sensor_map}
    
    def collect_readings(self) -> List[SensorReading]:
        """
        Collect readings from all sensors
        
        Returns:
            List of sensor readings
        """
        readings = []
        timestamp = datetime.now()
        
        for sensor_type, sensor_class in self.sensor_instances.items():
            if sensor_type == 'npk':
                # NPK sensor returns a dictionary
                npk_values = sensor_class.read()
                for nutrient, value in npk_values.items():
                    reading = SensorReading(
                        timestamp=timestamp,
                        sensor_id=f"{self.device_id}_{sensor_type}_{nutrient}",
                        sensor_type=f"{sensor_type}_{nutrient}",
                        value=value,
                        unit=sensor_class.UNIT,
                        location=self.location
                    )
                    readings.append(reading)
            else:
                value = sensor_class.read()
                reading = SensorReading(
                    timestamp=timestamp,
                    sensor_id=f"{self.device_id}_{sensor_type}",
                    sensor_type=sensor_type,
                    value=value,
                    unit=sensor_class.UNIT,
                    location=self.location
                )
                readings.append(reading)
        
        return readings
    
    def to_dict(self) -> Dict:
        """Convert device to dictionary"""
        return {
            'device_id': self.device_id,
            'location': self.location,
            'sensors': self.sensors
        }


class SensorNetwork:
    """
    Manages a network of IoT devices
    """
    
    def __init__(self):
        self.devices: Dict[str, IoTDevice] = {}
    
    def add_device(self, device: IoTDevice):
        """Add a device to the network"""
        self.devices[device.device_id] = device
    
    def collect_all_readings(self) -> List[SensorReading]:
        """Collect readings from all devices"""
        all_readings = []
        for device in self.devices.values():
            readings = device.collect_readings()
            all_readings.extend(readings)
        return all_readings
    
    def get_readings_dataframe_format(self) -> Dict:
        """
        Get readings in a format suitable for ML model
        
        Returns:
            Dictionary with sensor values organized by type
        """
        readings = self.collect_all_readings()
        
        data = {
            'timestamp': [],
            'soil_moisture': [],
            'temperature': [],
            'humidity': [],
            'light_intensity': [],
            'ph_level': [],
            'nitrogen': [],
            'phosphorus': [],
            'potassium': [],
            'rainfall': [],
            'wind_speed': []
        }
        
        for reading in readings:
            if reading.sensor_type.startswith('soil_moisture'):
                data['soil_moisture'].append(reading.value)
            elif reading.sensor_type.startswith('temperature'):
                data['temperature'].append(reading.value)
            elif reading.sensor_type.startswith('humidity'):
                data['humidity'].append(reading.value)
            elif reading.sensor_type.startswith('light_intensity'):
                data['light_intensity'].append(reading.value)
            elif reading.sensor_type.startswith('ph_level'):
                data['ph_level'].append(reading.value)
            elif reading.sensor_type.startswith('npk_nitrogen'):
                data['nitrogen'].append(reading.value)
            elif reading.sensor_type.startswith('npk_phosphorus'):
                data['phosphorus'].append(reading.value)
            elif reading.sensor_type.startswith('npk_potassium'):
                data['potassium'].append(reading.value)
            elif reading.sensor_type.startswith('rainfall'):
                data['rainfall'].append(reading.value)
            elif reading.sensor_type.startswith('wind_speed'):
                data['wind_speed'].append(reading.value)
        
        # Use average if multiple readings of same type
        for key in data:
            if key != 'timestamp' and data[key]:
                data[key] = np.mean(data[key])
            elif key == 'timestamp':
                data[key] = readings[0].timestamp if readings else None
        
        return data


# Sensor Specifications Summary
SENSOR_SPECIFICATIONS = {
    "soil_moisture": {
        "name": "Soil Moisture Sensor",
        "type": "Capacitive/Resistive",
        "range": "0-100%",
        "accuracy": "±2%",
        "power": "3.3V, 5mA",
        "communication": "I2C/SPI/Analog",
        "cost": "$10-50",
        "use_case": "Monitor soil water content for irrigation"
    },
    "temperature": {
        "name": "Temperature Sensor",
        "type": "DS18B20/DHT22",
        "range": "-20°C to 50°C",
        "accuracy": "±0.5°C",
        "power": "3.3V, 1-5mA",
        "communication": "OneWire/I2C",
        "cost": "$2-10",
        "use_case": "Monitor air and soil temperature"
    },
    "humidity": {
        "name": "Humidity Sensor",
        "type": "DHT22/SHT31",
        "range": "0-100% RH",
        "accuracy": "±2% RH",
        "power": "3.3V, 1-5mA",
        "communication": "I2C/SPI",
        "cost": "$5-15",
        "use_case": "Monitor air humidity levels"
    },
    "light_intensity": {
        "name": "Light Intensity Sensor (PAR)",
        "type": "BH1750/TSL2561",
        "range": "0-2000 μmol/m²/s",
        "accuracy": "±5%",
        "power": "3.3V, 0.1-1mA",
        "communication": "I2C",
        "cost": "$5-20",
        "use_case": "Measure photosynthetically active radiation"
    },
    "ph_level": {
        "name": "Soil pH Sensor",
        "type": "Electrochemical",
        "range": "0-14 pH",
        "accuracy": "±0.1 pH",
        "power": "5V, 5-10mA",
        "communication": "Analog/I2C",
        "cost": "$20-100",
        "use_case": "Monitor soil acidity/alkalinity"
    },
    "npk": {
        "name": "NPK Sensor",
        "type": "Electrochemical",
        "range": "0-500 mg/kg each",
        "accuracy": "±5%",
        "power": "5V, 20-50mA",
        "communication": "RS485/Modbus",
        "cost": "$50-200",
        "use_case": "Measure nitrogen, phosphorus, potassium levels"
    },
    "rainfall": {
        "name": "Rainfall Sensor",
        "type": "Tipping Bucket/Optical",
        "range": "0-100 mm/h",
        "accuracy": "±0.2 mm",
        "power": "5V, 10-20mA",
        "communication": "Digital/Analog",
        "cost": "$30-150",
        "use_case": "Measure precipitation"
    },
    "wind_speed": {
        "name": "Wind Speed Sensor",
        "type": "Anemometer",
        "range": "0-100 km/h",
        "accuracy": "±2%",
        "power": "12V, 50-100mA",
        "communication": "Analog/Pulse",
        "cost": "$50-200",
        "use_case": "Monitor wind conditions"
    }
}


if __name__ == '__main__':
    # Demo: Create sensor network
    print("=" * 60)
    print("Smart Agriculture IoT Sensor Network Demo")
    print("=" * 60)
    
    # Create network
    network = SensorNetwork()
    
    # Add devices with different sensor configurations
    device1 = IoTDevice(
        device_id="AGRI_001",
        location=(40.7128, -74.0060),  # Example coordinates
        sensors=['soil_moisture', 'temperature', 'humidity', 'light_intensity', 'ph_level']
    )
    
    device2 = IoTDevice(
        device_id="AGRI_002",
        location=(40.7130, -74.0062),
        sensors=['soil_moisture', 'temperature', 'npk', 'rainfall']
    )
    
    network.add_device(device1)
    network.add_device(device2)
    
    # Collect readings
    print("\nCollecting sensor readings...")
    readings = network.collect_all_readings()
    
    print(f"\nTotal readings collected: {len(readings)}")
    print("\nSample readings:")
    for reading in readings[:5]:
        print(f"  {reading.sensor_id}: {reading.value} {reading.unit}")
    
    # Get data for ML model
    ml_data = network.get_readings_dataframe_format()
    print("\n\nData formatted for ML model:")
    for key, value in ml_data.items():
        if value:
            print(f"  {key}: {value}")

