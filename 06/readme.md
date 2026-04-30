# Microgrid Energy Forecasting and Optimization Project

This project implements a hybrid deep learning and reinforcement learning framework for microgrid energy management. It combines a CNN-LSTM model for forecasting renewable generation and load demand, along with a PPO-based reinforcement learning agent for optimized energy distribution.

---

## 📌 Project Explanation  
Watch our project explanation:  
https://youtu.be/fjA-_VL7jrc  

---

## 👥 Team Members  
- T. Teena Anjusha – 524182  
- N. Sai Ambika Akshaya – 524154  
- P. Manasa Varshini – 524160  

---

## ⚙️ Project Overview  

This project consists of two major components:

### 1. Forecasting Module (CNN + LSTM)  
Predicts:
- PV (Solar) Production  
- Wind Production  
- Electric Demand  

Uses time-series data with feature engineering:
- Lag features  
- Rolling mean  
- Temporal features  

---

### 2. Optimization Module (Reinforcement Learning - PPO)  
Learns optimal energy distribution between:
- Solar  
- Wind  
- Grid  

Goal:
- Maximize renewable usage  
- Ensure demand satisfaction  

---

## 📂 Dataset Description  

This project uses a **single dataset file**:

- **Database.csv** → Used for both:
  - Forecasting model training (CNN + LSTM)
  - Reinforcement Learning environment (PPO)

### Dataset Contains:
- Solar Irradiance (DHI, DNI, GHI)  
- Weather data (Temperature, Humidity, Wind Speed)  
- Load demand values  
- Time-related features  

📥 **Dataset Download Link:**  
https://data.mendeley.com/datasets/fdfftr3tc2/1  

*(This dataset is hosted on Mendeley Data, a research data repository that allows datasets to be shared, cited, and reused in scientific projects.)* :contentReference[oaicite:0]{index=0}  

---

## 🛠️ Setup Instructions  

### 1. Dataset Preparation  
Ensure the following file is present in the root directory:

- `Database.csv`

---

### 2. Install Requirements  

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow gym stable-baselines3
```

(Optional for GPU support):
```bash
pip install tensorflow-gpu
```

---

### 3. Run the Project  

```bash
python main.py
```

The script will:
- Load and preprocess data from `Database.csv`
- Train CNN-LSTM forecasting model  
- Evaluate predictions using MAE, RMSE, R²  
- Train PPO reinforcement learning agent  
- Optimize energy distribution  
- Display graphs and performance metrics  

---

## 📊 Results and Outputs  

The project generates:
- Training vs Validation Loss Curve  
- Load Forecasting Graph  
- Scatter Plot (Actual vs Predicted Load)  
- Energy Distribution Graph (Solar, Wind, Grid)  
- Reward Curve (RL Performance)  

Saved model:
- `final_microgrid_model.h5`

---

## 📁 Project Structure  

```
├── main.py                     # Full implementation (Forecasting + RL)
├── Database.csv               # Dataset for both forecasting and RL
├── final_microgrid_model.h5   # Saved trained model
├── assets/                    # Output graphs (optional)
```

---

## 🤖 Model Details  

### Forecasting Model (CNN + LSTM)  

**Input Features:**
- Solar Irradiance (DHI, DNI, GHI)  
- Weather (Temperature, Humidity, Wind Speed)  
- Time Features (Hour, Day, Month)  
- Encoded categorical variables  
- Lag features & rolling mean  

**Architecture:**
- Conv1D (feature extraction)  
- Batch Normalization  
- LSTM (temporal learning)  
- Dense layers  
- Dropout (regularization)  

**Loss Function:**
- Mean Squared Error (MSE)  

---

### Reinforcement Learning Model (PPO)  

**State Space:**
- Solar production  
- Wind production  
- Load demand  

**Action Space:**
- Allocation weights for:
  - Solar  
  - Wind  
  - Grid  

**Reward Function Includes:**
- Demand-supply balance  
- Renewable energy usage  
- Grid usage penalty  
- Incentives for efficient renewable utilization  

---

## 🖥️ Sample Output (Console)  

```
Step 1
Demand: 0.532
Solar Used: 0.210
Wind Used : 0.180
Grid Used : 0.142
Renewable %: 73.30%
Reward: 14.23
----------------------------------------
```

---

## 📈 Evaluation Metrics  

- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)  
- R² Score (Model Accuracy)  

---

## 💻 System Requirements  

- Python 3.8+  
- 8GB RAM minimum  
- GPU (optional but recommended)  
- 2GB Disk Space  

---

## 🚀 Key Highlights  

- Hybrid AI approach (Deep Learning + Reinforcement Learning)  
- Single dataset pipeline for simplicity  
- Multi-output forecasting  
- Reward-engineered energy optimization  
- Scalable for smart grid applications  

---

## 🔮 Future Improvements  

- Add battery storage optimization  
- Include real-time deployment  
- Integrate pricing and cost optimization  
- Use advanced RL models (SAC, TD3)  

---

## ✅ Conclusion  

This project demonstrates how AI can be effectively used to forecast energy demand and optimize renewable energy utilization in microgrids, contributing to smarter and more sustainable power systems using a **unified dataset approach**.
