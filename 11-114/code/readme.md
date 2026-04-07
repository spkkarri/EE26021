# **Hybrid LSTM-Transformer for Battery State of Health (SOH) Prediction**

This project implements a deep learning architecture that combines **Long Short-Term Memory (LSTM)** networks and **Transformers** to predict the State of Health (SOH) of lithium-ion batteries using the **NASA Battery Dataset**.

---

## **Project Overview**
The goal of this model is to accurately assess battery degradation by processing time-series data including voltage, current, and temperature. By utilizing a hybrid approach, the model captures short-term temporal dependencies through LSTMs and long-range contextual relationships through Transformer self-attention mechanisms.

---

## **Data Pipeline**
*   **Dataset:** NASA SOH Data consisting of multiple battery discharge cycles.
*   **Features:** The model uses four key features: `Voltage_measured`, `Current_measured`, `Temperature_measured`, and `cycle_id`.
*   **Preprocessing:** 
    *   **Scaling:** Features are normalized using `MinMaxScaler` fitted strictly on training data to prevent data leakage.
    *   **Sliding Windows:** Data is structured into **30-cycle sliding windows** (`SEQUENCE_LENGTH = 30`) to provide temporal context for each prediction.
*   **Data Split:** An **80/20 split** is performed by battery file (not just random cycles) to ensure the model generalizes to entirely new batteries.

---

## **Model Architecture**
The `HybridLSTMTransformer` is composed of the following stages:

1.  **LSTM Feature Extractor:** Two stacked LSTM layers with **64 hidden units** process the input sequence to extract local temporal patterns.
2.  **Transformer Blocks:** Two Transformer blocks follow the LSTM to refine features.
    *   **Multi-Head Self-Attention:** Uses **4 attention heads** to attend to different representation subspaces simultaneously.
    *   **Feed-Forward Network (FFN):** A two-layer network that expands the data from **64 to 128 dimensions** (using **ReLU** activation) to learn complex non-linear relationships before compressing it back to 64.
3.  **Output Layer:** A final fully connected layer projects the 64-dimensional refined features into a **single scalar value** representing the predicted SOH.

---

## **Training & Evaluation**
*   **Loss Function:** Mean Squared Error (**MSE**).
*   **Optimizer:** **AdamW** (Learning Rate: 1e-3, Weight Decay: 1e-2), which is optimized for Transformer-based architectures.
*   **Learning Rate Scheduler:** `StepLR` reduces the learning rate by half every 20 epochs.
*   **Regularization:** **Dropout of 0.3** is applied within Transformer blocks, and **Early Stopping** (patience of 20) is used to prevent overfitting.
*   **Metrics:** Performance is evaluated using Mean Absolute Error (**MAE**) and Root Mean Squared Error (**RMSE**).

---
## 📁 Dataset

We use the publicly available `.mat` datasets from NASA’s Prognostics Data Repository:

- `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0008.mat`
## Data Drive Link:
Dataset = https://drive.google.com/drive/folders/1MbgZrASiTTbc7bM3Yi368yPJ6obH7a6o?usp=sharing
---
## Demo Video Link:
LInk = 
## **How to Use**
1.  **Environment Setup:** Ensure `torch`, `pandas`, `numpy`, and `sklearn` are installed.
2.  **Data Path:** Update the `CSV_FOLDER` variable to point to your NASA battery CSV files.
3.  **Training:** Run the training loop to generate the `best_hybrid_soh_model.pth` file.
4.  **Evaluation:** The script will automatically load the best model and visualize predictions against actual SOH values.
---
## Team information
-Team number = 11
-Team lead roll number(last three digits) = 114
-Course: EE2621 – Machine Learning for Engineers
### Team members:
-524107 - charan teja
-524113 - jagadeesh chandra prasad
-524114 - Omkar
-524121 - sanyasi naidu


