# ⚡ Smart Grid Energy Optimization using Reinforcement Learning (DQN & PPO)

This project implements an **Intelligent Energy Management System** using **Reinforcement Learning (RL)** to optimize energy distribution in smart grids.

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

The project uses **Reinforcement Learning agents** to:

* Optimize energy usage
* Reduce grid dependency
* Improve stability
* Maximize renewable energy usage

Models used:

* ✅ Dueling Double DQN
* ✅ PPO Actor-Critic
* ✅ Rule-Based baseline

---

## ⚙️ Setup Instructions

### 1️⃣ Download Dataset

```bash
cd data
chmod +x download_dataset.sh
./download_dataset.sh
```

---

### 2️⃣ Install Requirements

```bash
cd Code
pip install -r requirements.txt
```

---

### 3️⃣ Run Training

```bash
cd Code
python train.py
```

---

### 4️⃣ Run Dashboard

```bash
cd Code
python app.py
```

Then open:

```
smart_grid_rl.html
```

---

## 📊 Results

| Method     | Efficiency | Stability | Cost |
| ---------- | ---------- | --------- | ---- |
| Rule-Based | 42.0%      | 0.735     | ~69% |
| DQN        | **42.9%**  | **0.750** | ~69% |
| PPO        | 41.4%      | 0.719     | ~69% |

👉 **DQN gives best performance**

---

## 📁 Project Structure (SUBMISSION FORMAT)

```
EE26021/26/Team_LastDigits/

├── Code/
│   ├── app.py
│   ├── train.py
│   ├── dqn_agent.py
│   ├── ppo_agent.py
│   ├── smart_grid_env.py
│   ├── requirements.txt
│   ├── smart_grid_rl.html
│
├── assets/
│   ├── training.png
│   ├── dqn_weights.npz
│   ├── ppo_weights.npz
│   └── metrics.txt
│
├── data/
│   ├── download_dataset.sh
│   └── smart_grid_rl_dataset.xlsx (optional or via script)
```

---

## 📊 Outputs

* 📈 Training Graph → `training.png`
* 📄 Metrics → `metrics.txt`
* 🌐 Dashboard → `smart_grid_rl.html`

---

## 🧠 Key Features

✔ Reinforcement Learning-based optimization
✔ Real-world dataset
✔ Battery + renewable integration
✔ Time-of-Use pricing
✔ Interactive dashboard
✔ Model comparison

---

## 🖥️ Requirements

* Python 3.8+
* 8GB RAM
* 2GB storage
* (Optional) GPU

---

## 📌 Conclusion

This project demonstrates that **Reinforcement Learning significantly improves smart grid performance**, making systems more efficient, stable, and cost-effective.

---
