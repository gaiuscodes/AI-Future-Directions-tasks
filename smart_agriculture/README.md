# Smart Agriculture Simulation System

An AI-powered IoT-based smart agriculture system for monitoring crop conditions and predicting yields using machine learning.

## 🌾 Overview

This system simulates a complete smart agriculture solution that:
- Monitors crop conditions using multiple IoT sensors
- Processes sensor data through an AI model
- Predicts crop yields based on environmental factors
- Provides alerts and recommendations for optimal farming

## 📋 Features

- **8 Types of IoT Sensors**: Soil moisture, temperature, humidity, light intensity, pH, NPK, rainfall, wind speed
- **AI Yield Prediction**: Multiple ML models (Random Forest, Gradient Boosting, Neural Network, Ensemble)
- **Real-time Monitoring**: Continuous sensor data collection and processing
- **Alert System**: Automatic detection of suboptimal conditions
- **Recommendations**: Actionable insights for irrigation, fertilization, and soil management
- **Data Visualization**: Charts and graphs for analysis

## 🔧 System Components

### 1. IoT Sensors (`sensors.py`)

**Sensor Specifications:**

| Sensor | Range | Accuracy | Use Case |
|--------|-------|----------|----------|
| Soil Moisture | 0-100% | ±2% | Monitor irrigation needs |
| Temperature | -20°C to 50°C | ±0.5°C | Track air/soil temperature |
| Humidity | 0-100% RH | ±2% RH | Monitor air humidity |
| Light Intensity | 0-2000 μmol/m²/s | ±5% | Measure PAR for photosynthesis |
| pH Level | 0-14 pH | ±0.1 pH | Monitor soil acidity |
| NPK | 0-500 mg/kg | ±5% | Measure nutrients (N, P, K) |
| Rainfall | 0-100 mm/h | ±0.2 mm | Track precipitation |
| Wind Speed | 0-100 km/h | ±2% | Monitor wind conditions |

### 2. AI Model (`yield_prediction_model.py`)

**Model Types:**
- **Random Forest**: Fast, interpretable, good for tabular data
- **Gradient Boosting**: High accuracy, handles non-linear relationships
- **Neural Network**: Deep learning approach, captures complex patterns
- **Ensemble**: Combines multiple models for best performance

**Features Used:**
- Soil moisture, temperature, humidity
- Light intensity, pH level
- Nitrogen, Phosphorus, Potassium (NPK)
- Rainfall, wind speed

**Output:** Crop yield prediction in kg/hectare

### 3. Simulation System (`simulation_system.py`)

Complete simulation that:
- Manages multiple IoT devices
- Collects sensor data continuously
- Runs AI predictions in real-time
- Generates alerts and recommendations
- Stores historical data
- Creates visualizations

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Train the AI Model

```bash
python yield_prediction_model.py
```

This will train models for different crop types (wheat, corn, rice, soybean).

#### 2. Run Simulation

```bash
python simulation_system.py
```

Or use the simulator programmatically:

```python
from simulation_system import SmartAgricultureSimulator

# Create simulator
simulator = SmartAgricultureSimulator(
    num_devices=3,
    crop_type='wheat',
    model_type='ensemble'
)

# Run simulation
results = simulator.run_simulation(
    duration_minutes=60,
    interval_seconds=60
)

# Visualize results
simulator.visualize_results()
```

#### 3. Test Sensors

```bash
python sensors.py
```

## 📊 Data Flow

See `data_flow_diagram.md` for detailed architecture and data flow diagrams.

**Simplified Flow:**
```
Sensors → Gateway → Cloud → AI Processing → Predictions → Dashboard/Alerts
```

## 📁 Project Structure

```
smart_agriculture/
├── sensors.py                    # IoT sensor definitions and simulation
├── yield_prediction_model.py     # AI model for yield prediction
├── simulation_system.py          # Complete system simulation
├── data_flow_diagram.md          # Architecture and data flow diagrams
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── SMART_AGRICULTURE_REPORT.md   # Comprehensive report
├── models/                       # Trained AI models (generated)
└── results/                      # Simulation results (generated)
```

## 🎯 Use Cases

1. **Precision Agriculture**: Optimize resource usage (water, fertilizer)
2. **Yield Forecasting**: Predict harvest yields for planning
3. **Disease Prevention**: Early detection of suboptimal conditions
4. **Resource Management**: Automated irrigation and fertilization
5. **Data-Driven Decisions**: Make informed farming decisions

## 📈 Model Performance

Expected metrics for yield prediction:
- **R² Score**: 0.85-0.95
- **RMSE**: 200-500 kg/hectare
- **MAE**: 150-400 kg/hectare
- **MAPE**: 5-15%

*Note: Performance depends on data quality and crop type*

## 🔍 Example Output

```
[Step 1] 2024-01-15 10:30:00
  Yield Prediction: 3420 kg/hectare
  ⚠ Alerts: 1
    - Low soil moisture: 28.5% - Irrigation needed
  💡 Recommendations: 1
    - Start irrigation system to maintain optimal soil moisture (40-60%)
```

## 🛠️ Customization

### Add New Sensors

Edit `sensors.py` and add a new sensor class following the existing pattern.

### Modify AI Model

Edit `yield_prediction_model.py` to:
- Change model architecture
- Add new features
- Adjust hyperparameters

### Configure Simulation

Modify `simulation_system.py` to:
- Change number of devices
- Adjust sensor configurations
- Modify alert thresholds
- Customize recommendations

## 📚 Documentation

- **Data Flow Diagram**: See `data_flow_diagram.md`
- **Comprehensive Report**: See `SMART_AGRICULTURE_REPORT.md`
- **Sensor Specifications**: See `sensors.py` (SENSOR_SPECIFICATIONS)

## 🔬 Technical Details

### Sensor Communication
- **Protocols**: I2C, SPI, OneWire, RS485, Analog
- **Power**: 3.3V-12V, 1-100mA depending on sensor
- **Update Rate**: 1-5 minutes (configurable)

### AI Model Architecture
- **Input**: 10 features (sensor readings)
- **Hidden Layers**: 2-3 dense layers (for neural network)
- **Output**: Single value (yield in kg/hectare)
- **Training**: Supervised learning with historical data

### Data Processing
- **Preprocessing**: Normalization, feature engineering
- **Storage**: Time-series database for sensor data
- **Real-time**: Sub-second inference on edge devices

## 🌍 Real-World Deployment

### Hardware Requirements
- Raspberry Pi or similar edge device
- Sensor modules (see specifications)
- Power supply and connectivity
- Optional: Gateway for cloud communication

### Software Requirements
- Python 3.8+
- TensorFlow Lite (for edge deployment)
- MQTT broker (for IoT communication)
- Database (PostgreSQL/InfluxDB)

### Cloud Infrastructure
- Data storage (time-series database)
- ML model serving (TensorFlow Serving)
- API server (REST/GraphQL)
- Dashboard (Web application)

## 📝 License

This project is provided for educational and demonstration purposes.

## 🙏 Acknowledgments

- IoT sensor manufacturers and communities
- Machine learning frameworks (TensorFlow, scikit-learn)
- Open-source agriculture projects

---

**Version**: 1.0  
**Last Updated**: 2024

For detailed technical documentation, see `SMART_AGRICULTURE_REPORT.md`.

