#  EV Battery Intelligence System

### Anomaly Detection • SoC • SoH • RUL • EV Range Prediction using Hybrid AI Models

This project implements a **complete battery intelligence pipeline** for Electric Vehicles including:

* Battery Anomaly Detection
* State of Charge (SoC) Estimation
* State of Health (SoH) Prediction
* Remaining Useful Life (RUL) Prediction
* EV Driving Range Prediction
* Hybrid Deep Learning + ML Ensemble Architecture

The system combines **Deep Neural Networks, LSTM, Residual Networks, and Gradient Boosting** to provide accurate battery analytics.

---

#  Project Explanation Video

Watch the complete explanation of our model and implementation:

🔗 https://www.youtube.com/playlist?list=PLMowFToGv2T4U9zJG3LClz9bpXaAKRlWC
This video explains:

* Full pipeline architecture
* Anomaly detection model
* SoC, SoH, RUL models
* EV range prediction ensemble
* Training and results visualization

---

#  Team Members

* P.Nagashiva (Team Lead) – 524162
* S.Sridhar – 524179
* T.Rishik – 524181
* S.Tharun – 524177

---

#  Project Objective

The goal of this project is to build an **end-to-end EV battery prediction pipeline** that:

* Detects abnormal battery behaviour
* Predicts battery charge level (SoC)
* Estimates battery degradation (SoH)
* Predicts Remaining Useful Life (RUL)
* Predicts EV driving range
* Uses ensemble learning for high accuracy
* Provides visualization for battery degradation trends

---

#  Model Architecture Overview

The pipeline consists of **four major AI modules**

##  Battery Anomaly Detection Model

* Isolation Forest based anomaly detection
* Detects abnormal battery behaviour
* Identifies sudden voltage/current spikes
* Detects abnormal degradation patterns
* Flags unsafe battery operating conditions

---

##  SoC Estimation Model

* Deep Neural Network (DNN)
* Learns battery voltage, current, temperature patterns
* Predicts real-time battery charge level
* Used as input for SoH model

---

##  SoH Prediction Model

* Residual Deep Neural Network
* Uses skip connections
* Predicts battery degradation over cycles
* Learns nonlinear aging behaviour

---

##  RUL Prediction Model

* LSTM based time-series model
* Uses cycle history sequences
* Predicts remaining battery life
* Captured temporal degradation trend

---

##  EV Range Prediction Model

Hybrid Ensemble Model:

* Deep Neural Network (Residual DNN)
* Gradient Boosting Regressor (GBM)
* Blended prediction (DNN + GBM)

This module predicts:

* Remaining EV driving range (km)
* Based on SoC, SoH, RUL and environment features

---

#  Full Pipeline Flow

Raw Battery Data
↓
Anomaly Detection Model
↓
SoC Model
↓
SoH Model
↓
RUL Model
↓
EV Range Model (DNN + GBM Ensemble)
↓
Final Battery Intelligence Output

---

#  Final Pipeline Performance

| Module              | MAE    | RMSE   | R²      |
| ------------------- | ------ | ------ | ------- |
| SoC Estimation      | 0.0028 | 0.0035 | -0.0528 |
| SoH Prediction      | 0.0058 | 0.0074 | 0.9775  |
| RUL Prediction      | 2.2908 | 2.9606 | 0.9879  |
| EV Range Prediction | 0.6026 | 0.7860 | 0.9979  |

---

#  Setup Instructions

## 1. Clone Project

```bash
git clone <repo_link>
cd EV_Battery_Project
```

---

## 2. Install Requirements

```bash
pip install -r requirements.txt
```

Required libraries:

* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn

---

## 3. Run Complete Pipeline

```bash
python main.py
```

The script will:

1. Load battery dataset
2. Run anomaly detection
3. Train SoC model
4. Train SoH model
5. Train RUL model
6. Train EV Range ensemble model
7. Generate predictions
8. Plot degradation curves
9. Show performance metrics

---

#  Output Visualizations

The pipeline generates:

* EV Range Prediction Plot
* SoH Degradation Curve
* RUL Prediction Curve
* Anomaly Detection Plot
* Actual vs Predicted Graph
* Ensemble Model Comparison

All outputs saved in:

```
assets/
 ├── EV_Range_Prediction.png
 ├── degradation_overview.png
 ├── rul_prediction.png
 └── model_performance.png
```

---

#  Project Structure

```
EV_Battery_Project
│
├── Code/
│   ├── main.py
│   ├── EV_project_ml.ipynb
│   ├── requirements.txt
│   ├── README.md
│
├── data/
│   ├── battery_dataset.csv
│   └── cleaned_battery_data.pkl
│
├── assets/
│   ├── plots
│   └── results
```

---

#  Features Used

The models use:

* Voltage
* Current
* Temperature
* Cycle count
* Capacity
* Internal resistance
* Charge time
* Discharge time

---

#  Example Output

```
Anomaly Detected : False
Predicted SoC : 0.84
Predicted SoH : 0.91
Predicted RUL : 52 cycles
Predicted EV Range : 253.5 km
```

---

#  Key Highlights

✔ End-to-end EV battery pipeline
✔ Anomaly detection included
✔ Hybrid deep learning + ML ensemble
✔ Residual neural network architecture
✔ LSTM time-series modeling
✔ High accuracy (R² = 0.997 EV range)
✔ Real-world battery degradation modeling
✔ Clean modular architecture

---

#  System Requirements

Python 3.8+
8GB RAM minimum
GPU optional (CUDA supported)
2GB disk space

---

#  Future Improvements

* Real-time EV dashboard
* IoT battery monitoring
* Cloud deployment
* Mobile app integration
* Transformer-based RUL model

---

#  Project Demonstration

The system predicts:

* Battery degradation
* Remaining battery life
* EV driving range
* Charging efficiency
* Battery anomalies

This project can be used for:

* Electric vehicles
* Battery management systems
* Smart charging stations
* EV fleet analytics

---

#  License

This project is for academic and research purposes only.
