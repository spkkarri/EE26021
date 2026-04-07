# ⚡ Advanced Smart Grid Energy Optimization using Reinforcement Learning

This project presents an **AI-powered Smart Grid Energy Management System** using **Reinforcement Learning (RL)** to optimize energy distribution, improve efficiency, and ensure grid stability.

The system compares:

* **Dueling Double DQN + Prioritized Experience Replay (PER)**
* **PPO (Proximal Policy Optimization)**
* **Rule-Based Baseline**

---

## 👥 Team Members

* **Ved Prakash Mishra (Team Lead)** – 524187
* **Nitish Kumar** – 524157
* **Shivesh Kumar Jha** – 524175
* **Vipul Kumar** – 524116

---

## 🎥 Project Demo

👉 https://youtu.be/syUTNftDMbQ?si=WCCgt6VHo51anUHa

---

## 📌 Project Overview

Smart grids require intelligent systems to manage:

* Renewable energy (solar, wind)
* Battery storage
* Dynamic demand
* Cost optimization

Traditional methods fail in real-time adaptability.
This project uses **Reinforcement Learning agents** to learn optimal decisions dynamically.

---

## 🧠 Models Used

### 🔹 1. Dueling Double DQN + PER

* Dueling architecture (Value + Advantage)
* Double DQN (reduces overestimation)
* Prioritized Experience Replay
* High performance in discrete action spaces 

### 🔹 2. PPO Actor-Critic

* Stable policy optimization
* Actor-Critic architecture
* Better exploration-exploitation balance 

### 🔹 3. Rule-Based System

* Baseline comparison using heuristics

---

## ⚙️ Environment Details

The system uses a custom smart grid environment:

* **State (9 features):**

  * Solar, Wind, Grid availability
  * Battery State of Charge
  * Demand
  * Time (sin/cos encoding)
  * Temperature, Sunlight

* **Actions (5):**

  * Charge battery
  * Discharge battery
  * Buy from grid
  * Use renewables
  * Idle

* **Reward Function:**

  * Maximize renewable usage
  * Minimize grid cost (Time-of-Use pricing)
  * Penalize unmet demand
  * Penalize battery misuse 

---

## 📊 Results & Performance

### 🔹 Energy Efficiency

* DQN: **42.9%**
* PPO: **41.4%**
* Rule-Based: **42.0%**

### 🔹 Grid Stability Index

* DQN: **0.750 (Best)**
* PPO: **0.719**
* Rule-Based: **0.735**

### 🔹 Cost Reduction

* ~69% for all models

### 🔹 Average Reward

* DQN: **-10.09 (Best)**
* PPO: **-12.79**
* Rule-Based: **-11.43** 

👉 **Conclusion:**
DQN with PER performs best overall.

---

## 📈 Training Output

The system generates:

* Reward curves
* Cumulative reward comparison
* Efficiency and stability graphs

File:

```
training.png
```

---

## 🖥️ Dashboard

A real-time web dashboard is implemented using Flask:

### Features:

* Live grid simulation
* Agent switching (DQN / PPO)
* Energy visualization
* Performance metrics

Run:

```bash
python app.py
```

Then open:

```
smart_grid_rl.html
```

API example:

```
GET /step?agent=dqn
```

---

## ⚙️ Setup Instructions

### 1️⃣ Download Dataset

```bash
cd data
chmod +x download_dataset.sh
./download_dataset.sh
```

---

### 2️⃣ Install Dependencies

```bash
cd Code
pip install -r requirements.txt
```

Dependencies include:

* numpy, pandas, matplotlib
* flask (for dashboard) 

---

### 3️⃣ Train Models

```bash
python train.py
```

This will generate:

* `dqn_weights.npz`
* `ppo_weights.npz`
* `metrics.txt`
* `training.png` 

---

### 4️⃣ Run Dashboard

```bash
python app.py
```

---

## 📁 Project Structure

```
EE26021/26/Team_LastDigits/

├── Code/
│   ├── train.py
│   ├── app.py
│   ├── dqn_agent.py
│   ├── ppo_agent.py
│   ├── smart_grid_env.py
│   ├── requirements.txt
│
├── data/
│   ├── download_dataset.sh
│   └── smart_grid_rl_dataset.xlsx
│
├── assets/
│   ├── training.png
│   ├── Final_Output.jpg
│   ├── Comparison_Output.jpg
│   └── report.pdf
```

---

## 🔬 Research Reference

This project is based on:

📄 *Energy Optimization for Smart Grids Using Reinforcement Learning*
(IEEE INCET 2025) 

---

## 🚀 Key Contributions

✔ Real-world dataset (8760 hours)
✔ Advanced RL algorithms (DQN + PPO)
✔ Time-of-Use pricing integration
✔ Dashboard visualization
✔ Comparison with traditional methods

---

## ⚠️ Notes

* Run training before dashboard
* Ensure dataset is downloaded
* GPU optional but recommended

---

## 📌 Conclusion

This project demonstrates how **Reinforcement Learning can significantly improve smart grid energy management**, making systems more efficient, stable, and cost-effective.

---

