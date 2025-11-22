# Smart Agriculture System - Data Flow Diagram

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SMART AGRICULTURE SYSTEM                          │
│                      AI-Powered IoT Monitoring                            │
└─────────────────────────────────────────────────────────────────────────┘

```

## Data Flow Diagram

```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    FIELD SENSORS (IoT Devices)                   │
    │                                                                   │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
    │  │   Device 1   │  │   Device 2   │  │   Device N   │         │
    │  │  (Field A)   │  │  (Field B)   │  │  (Field N)   │         │
    │  └──────────────┘  └──────────────┘  └──────────────┘         │
    │         │                  │                  │                  │
    │    ┌────┴────┐       ┌────┴────┐       ┌────┴────┐            │
    │    │ Sensors │       │ Sensors │       │ Sensors │            │
    │    └─────────┘       └─────────┘       └─────────┘            │
    │         │                  │                  │                  │
    └─────────┼──────────────────┼──────────────────┼──────────────────┘
              │                  │                  │
              │  Sensor Data     │                  │
              │  (JSON/Protocol) │                  │
              ▼                  ▼                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              GATEWAY / EDGE COMPUTING LAYER                 │
    │                                                               │
    │  • Data Aggregation                                          │
    │  • Protocol Conversion (MQTT/HTTP/CoAP)                      │
    │  • Local Processing & Filtering                              │
    │  • Data Validation                                            │
    │  • Time-series Buffer                                        │
    └─────────────────────────────────────────────────────────────┘
              │
              │  Processed Data Stream
              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              CLOUD / CENTRAL PROCESSING LAYER                │
    │                                                               │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │           DATA STORAGE & MANAGEMENT                  │   │
    │  │  • Time-series Database (InfluxDB/TimescaleDB)      │   │
    │  │  • Relational Database (PostgreSQL)                 │   │
    │  │  • Data Warehouse (Historical Data)                  │   │
    │  └─────────────────────────────────────────────────────┘   │
    │                                                               │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │              AI/ML PROCESSING ENGINE                 │   │
    │  │                                                       │   │
    │  │  ┌───────────────────────────────────────────────┐   │   │
    │  │  │   Feature Engineering & Preprocessing         │   │   │
    │  │  │   • Normalization                             │   │   │
    │  │  │   • Feature Selection                         │   │   │
    │  │  │   • Time-series Aggregation                    │   │   │
    │  │  └───────────────────────────────────────────────┘   │   │
    │  │                    │                                 │   │
    │  │                    ▼                                 │   │
    │  │  ┌───────────────────────────────────────────────┐   │   │
    │  │  │      AI MODEL: Yield Prediction               │   │   │
    │  │  │  • Random Forest / Gradient Boosting          │   │   │
    │  │  │  • Neural Network                             │   │   │
    │  │  │  • Ensemble Methods                           │   │   │
    │  │  └───────────────────────────────────────────────┘   │   │
    │  │                    │                                 │   │
    │  │                    ▼                                 │   │
    │  │  ┌───────────────────────────────────────────────┐   │   │
    │  │  │      Predictions & Insights                  │   │   │
    │  │  │  • Yield Forecast                             │   │   │
    │  │  │  • Risk Assessment                            │   │   │
    │  │  │  • Recommendations                            │   │   │
    │  │  └───────────────────────────────────────────────┘   │   │
    │  └─────────────────────────────────────────────────────┘   │
    │                                                               │
    └─────────────────────────────────────────────────────────────┘
              │
              │  Predictions & Alerts
              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    APPLICATION LAYER                          │
    │                                                               │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
    │  │   Dashboard  │  │  Mobile App │  │  API Server │      │
    │  │  (Web UI)    │  │  (Farmers)  │  │  (3rd Party) │      │
    │  └──────────────┘  └──────────────┘  └──────────────┘      │
    │                                                               │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │              ACTUATOR CONTROL                        │   │
    │  │  • Irrigation Systems                               │   │
    │  │  • Fertilizer Dispensers                            │   │
    │  │  • Climate Control                                  │   │
    │  └─────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘
```

## Detailed Data Flow Process

### 1. Data Collection Phase

```
Sensors → Raw Data → Local Processing → Transmission
   │          │            │                 │
   │          │            │                 │
   ▼          ▼            ▼                 ▼
[Read]   [Format]    [Validate]        [Send to Gateway]
```

**Sensor Data Format:**
```json
{
  "device_id": "AGRI_001",
  "timestamp": "2024-01-15T10:30:00Z",
  "location": {"lat": 40.7128, "lon": -74.0060},
  "sensors": {
    "soil_moisture": 45.2,
    "temperature": 22.5,
    "humidity": 65.0,
    "light_intensity": 650.0,
    "ph_level": 6.8,
    "nitrogen": 180.5,
    "phosphorus": 95.2,
    "potassium": 220.0,
    "rainfall": 2.5,
    "wind_speed": 8.3
  }
}
```

### 2. Data Processing Pipeline

```
Raw Sensor Data
      │
      ▼
┌─────────────────┐
│ Data Validation │  → Check ranges, detect outliers
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Data Cleaning   │  → Remove noise, handle missing values
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Feature Eng.    │  → Create derived features, aggregations
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Normalization   │  → Scale features for ML model
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Model Input     │  → Ready for AI prediction
└─────────────────┘
```

### 3. AI Model Processing Flow

```
Preprocessed Sensor Data (10 features)
              │
              ▼
    ┌─────────────────────┐
    │  Feature Vector      │
    │  [moisture, temp,    │
    │   humidity, light,   │
    │   pH, N, P, K,       │
    │   rain, wind]        │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  AI Model           │
    │  (Ensemble/Neural)  │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Yield Prediction   │
    │  (kg/hectare)       │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Confidence Score   │
    │  Risk Assessment    │
    └─────────────────────┘
```

### 4. Feedback Loop

```
Predictions
      │
      ▼
┌─────────────────┐
│ Recommendations │  → Irrigation, fertilization, etc.
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Actuator Control│  → Automated actions
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ New Sensor Data │  → Continuous monitoring
└─────────────────┘
      │
      ▼
    (Loop)
```

## Data Flow Components

### Input Layer (Sensors)
- **Soil Moisture**: Continuous monitoring
- **Temperature**: Air and soil temperature
- **Humidity**: Relative humidity levels
- **Light Intensity**: PAR measurements
- **pH Level**: Soil acidity/alkalinity
- **NPK**: Nutrient levels (N, P, K)
- **Rainfall**: Precipitation monitoring
- **Wind Speed**: Wind conditions

### Processing Layer
1. **Data Aggregation**: Combine readings from multiple sensors
2. **Time-series Processing**: Handle temporal data
3. **Feature Engineering**: Create derived features
4. **Model Inference**: Run AI predictions
5. **Post-processing**: Format results

### Output Layer
1. **Predictions**: Yield forecasts
2. **Alerts**: Anomaly detection
3. **Recommendations**: Actionable insights
4. **Visualizations**: Charts and dashboards
5. **Control Signals**: Actuator commands

## Communication Protocols

```
Sensors → Gateway:    LoRaWAN / Zigbee / WiFi
Gateway → Cloud:      MQTT / HTTP / CoAP
Cloud → Applications: REST API / WebSocket
Cloud → Actuators:    MQTT / HTTP
```

## Data Storage Architecture

```
┌─────────────────────────────────────────┐
│         TIME-SERIES DATABASE            │
│  (InfluxDB / TimescaleDB)               │
│  • Raw sensor readings                  │
│  • High-frequency data (1 min intervals)│
│  • Retention: 1 year                     │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│      RELATIONAL DATABASE                 │
│  (PostgreSQL)                            │
│  • Processed features                   │
│  • Model predictions                     │
│  • User data & configurations           │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│      DATA WAREHOUSE                      │
│  (BigQuery / Redshift)                   │
│  • Historical data (years)               │
│  • Aggregated statistics                 │
│  • Analytics & reporting                 │
└─────────────────────────────────────────┘
```

## Real-time Processing Flow

```
Sensor Reading (every 1-5 minutes)
         │
         ▼
    Buffer (5-10 readings)
         │
         ▼
    Aggregate Features
         │
         ▼
    Run AI Model
         │
         ▼
    Generate Prediction
         │
         ▼
    Update Dashboard
         │
         ▼
    Trigger Alerts (if needed)
         │
         ▼
    Send Control Signals (if automated)
```

## Edge Computing Option

For low-latency requirements, AI model can run on edge:

```
Sensors → Edge Gateway → Local AI Model → Immediate Actions
                │
                └──→ Sync to Cloud (async)
```

This enables:
- **Sub-second response times**
- **Offline operation**
- **Reduced bandwidth**
- **Privacy (local processing)**

