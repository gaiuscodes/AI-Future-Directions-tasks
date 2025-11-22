# Quick Start Guide - Smart Agriculture System

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train AI Models

```bash
# Train models for all crops
python main.py --train

# Or train for specific crop
python main.py --train --crop wheat
```

### Step 3: Run Simulation

```bash
# Run with default settings (5 minutes, wheat, 3 devices)
python main.py --simulate

# Custom simulation
python main.py --simulate --crop corn --duration 10 --devices 5
```

## 📊 What You'll See

The simulation will:
1. ✅ Collect sensor data from IoT devices
2. ✅ Run AI predictions for crop yield
3. ✅ Generate alerts for suboptimal conditions
4. ✅ Provide actionable recommendations
5. ✅ Save results and create visualizations

## 📁 Output Files

After running, check:
- `models/` - Trained AI models
- `results/` - Simulation data and visualizations

## 🎯 Example Commands

```bash
# Complete workflow
python main.py --train --simulate --crop wheat

# Quick test (short simulation)
python main.py --simulate --duration 1 --interval 10

# Train multiple crops
python main.py --train --crop wheat
python main.py --train --crop corn
python main.py --train --crop rice
```

## 📖 Full Documentation

- **README.md** - Complete user guide
- **SMART_AGRICULTURE_REPORT.md** - Technical report
- **data_flow_diagram.md** - Architecture diagrams

