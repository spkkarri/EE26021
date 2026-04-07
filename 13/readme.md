# Smart Grid Energy Forecasting — Liander 2024

Multi-model ensemble pipeline for 24-hour ahead energy forecasting on the Liander 2024 smart meter dataset. Combines LightGBM, N-HiTS, and TFT base models in a meta-ensemble stacker, with a decision engine for real-time grid management.

---

## Team Members

- Dasari Surya Venkata Arjun (Team Lead) - 524123
- Harsha - 524140
- K Mayur Swamy - 524135

## Project Explanation

Watch our project explanation: [Demo Link](https://drive.google.com/file/d/1P-YsZVuxTlRR5XoNNayg4JDwd0U4UU52/view?usp=drive_link)

---

## Results (Holdout: Nov–Dec 2024)

| Track | Model | R² | MAE |
|---|---|---|---|
| Load | Meta Ensemble | **0.879** | 0.062 |
| Solar | Meta Ensemble | **0.650** | 0.020 |
| Wind | Meta Ensemble | **0.740** | 0.105 |

Meta model beats all base models on every track. Load H+1 R² = 0.915.

---

## Project Structure

```
smart_grid_management/
├── src/
│   ├── preprocessing/
│   │   ├── load_assets.py       ← load raw parquet files
│   │   ├── pre_process.py       ← clean, impute, normalize
│   │   ├── feature_eng.py       ← lag features, cyclic encoding, rolling stats
│   │   └── SPLITS.py            ← train/holdout split (Jan–Oct / Nov–Dec)
│   └── models/
│       ├── lgbm_model.py        ← LightGBM OOF training (5-fold walk-forward)
│       ├── nhits_model.py       ← N-HiTS OOF training (GPU)
│       ├── tft_model.py         ← TFT OOF training (GPU)
│       ├── meta_model.py        ← LightGBM stacking ensemble
│       ├── validate_holdout.py  ← holdout evaluation + plots
│       └── decision_engine.py   ← 10-hour grid forecast + actions
├── scripts/
│   ├── download_dataset.py      ← download Liander 2024 from Hugging Face
│   └── collect_model_files.sh   ← zip trained models for sharing
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/arjun1623-code/smart_grid_management
cd smart_grid_management
pip install -r requirements.txt
```

---

## Option A — Use Pre-trained Models (Recommended, ~2 min)

Download the pre-trained model files and calibration data:

**Google Drive:** https://drive.google.com/drive/folders/1mA6rjgsGlKxxbyrqmbH3LvMCR6v4oyEt?usp=sharing
```bash
# Unzip into the correct location
unzip smart_grid_models.zip -d data/processed/

# Run decision engine immediately
python src/models/decision_engine.py
```

That's it. The decision engine loads all models, calibrates confidence from holdout data, and prints a 10-hour forecast with grid management recommendations.

---

## Option B — Retrain from Scratch (~4–5 hours, GPU recommended)

### Step 1: Download dataset

```bash
python scripts/download_dataset.py
```

This downloads the Liander 2024 smart meter dataset from Hugging Face to `data/raw/`.

If the automatic download fails, manually download from:
 https://huggingface.co/datasets/OpenSTEF/liander2024-energy-forecasting-benchmark
and place the parquet files in `data/raw/`.

### Step 2: Preprocessing

```bash
python src/preprocessing/load_assets.py
python src/preprocessing/pre_process.py
python src/preprocessing/feature_eng.py
python src/preprocessing/SPLITS.py
```

Expected output:
```
load:  (1164480, 48) rows
solar: (145560, 23)  rows
wind:  (146280, 24)  rows
Train: Jan 2024 → Oct 2024
Val:   Nov 2024 → Dec 2024
```

### Step 3: Train base models

```bash
# LightGBM — runs on CPU, ~2 hours
python src/models/lgbm_model.py

# N-HiTS — GPU recommended, ~30 min
python src/models/nhits_model.py

# TFT — GPU recommended, ~45 min
python src/models/tft_model.py
```

### Step 4: Train meta-model and validate

```bash
python src/models/meta_model.py
python src/models/validate_holdout.py
```

### Step 5: Run decision engine

```bash
python src/models/decision_engine.py
```

---

## Decision Engine Output

When you run `decision_engine.py`, you see:

```
======================================================================
  SMART GRID DECISION ENGINE
  Reference Time : 2024-12-31 11:45 UTC
  Forecast Window: Next 10 hours (H+1 to H+10)
======================================================================

📊 LOAD TRACK — Next 10 Hours
  H+n  Forecast  Stress  Confidence  Alert
  H+1    0.712   0.712   MED  (54%)  ✅ NORMAL
  H+2    0.748   0.748   MED  (56%)  🟡 WATCH
  ...

⚡ NET GRID SUMMARY
  H+n  Net Load  Renew%  Cost Signal  Congestion
  H+1     0.302   18.3%       NORMAL  NORMAL
  ...

🎯 RECOMMENDED ACTIONS
  → ✅ All systems nominal — no action required
```

**Congestion thresholds:**
- 🔴 CRITICAL — Stress Index > 0.90 → immediate demand curtailment
- 🟠 WARNING  — Stress Index > 0.85 → activate demand response
- 🟡 WATCH    — Stress Index > 0.70 → monitor closely
- ✅ NORMAL   — Stress Index < 0.70 → no action

**Confidence levels** (calibrated from Nov–Dec 2024 holdout):
- HIGH — residual std ≤ 0.05
- MED  — residual std ≤ 0.10
- LOW  — residual std ≤ 0.20

**Grid metrics computed:**
- Net Load = Load − Solar − Wind
- Stress Index = Load / upper_limit
- Renewable % = (Solar + Wind) / Load
- Cost Signal: OVERSUPPLY / NORMAL / HIGH COST

Output also saved to: `data/processed/decision_engine/decision_output.csv`

---

## Architecture

```
Raw Data (Liander 2024)
        ↓
  Preprocessing
  (clean, normalize, feature engineering)
        ↓
  ┌─────────────────────────────┐
  │  Base Models (OOF Training) │
  │  ├── LightGBM  R²=0.767    │
  │  ├── N-HiTS    R²=0.697    │
  │  └── TFT       R²=0.684    │
  └─────────────┬───────────────┘
                ↓
  Meta-Model Stacker (LightGBM)
  (OOF predictions + context features)
                ↓
  Holdout Validation (Nov-Dec 2024)
  Load R²=0.879 | Solar R²=0.650 | Wind R²=0.740
                ↓
  Decision Engine
  (10-hour forecast + grid actions)
```

---

## Requirements

```
python >= 3.10
torch >= 2.0
lightgbm
pandas
numpy
scikit-learn
matplotlib
huggingface_hub
```

Install all:
```bash
pip install -r requirements.txt
```

---

## Notes

- GCN was architecturally planned as a 4th base model for spatial asset dependencies. Not implemented due to absence of geographic coordinates in the Liander 2024 metadata.
- Training was done on RTX 5070 Laptop (8.5GB VRAM). CPU training works but is slower (~2x for neural models).