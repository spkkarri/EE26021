# UAV Based Insulator Classification

**Submitted By:** 524130, 524132, 524126

---
## Project Explanation
Watch our project explanation:
[![Link]](https://youtu.be/0OfKZTr8JqA?si=rbcc74WTH42dybyu)
[![Link]](https://docs.google.com/presentation/d/1_yVkvcKhMTV5uDg88F0pgqw12CJ9FBCWpO-XHBE5SDg/edit?usp=sharing)
#  Project Overview
To accurately detect and classify three primary types of electrical insulators in near-real-time using the **YOLOv8n** (You Only Look Once v8) architecture.
* **Target Classes:** Porcelain, Glass, and Composite insulators.

#  Key Features
- **Model:** Utilizes YOLOv8n (Nano), the state-of-the-art in real-time object detection.
- **Multi-Class Classification:** Specifically trained to distinguish between three distinct industrial materials: Porcelain, Glass, and Composite.
- **Precision:** Achieved a Precision of 0.943 and a mAP50 of 0.937, ensuring reliable detection even in complex outdoor environments.
- **Streamlit UI:** A clean, web-based interface for ground station operators to visualize data.
- **Dynamic Thresholding:** A live "Confidence Slider" that allows users to adjust detection sensitivity for "Composite" insulators that are often missed by fixed-threshold models.
- **Automated Reporting:** Instantly generates a material count (e.g., "3 Glass, 1 Porcelain found").

#  Technical Architecture
The project utilizes the **YOLOv8n (Nano)** model, which is the fastest and most efficient version of the YOLOv8 family.
- **Backbone:** Uses **C2f modules** for efficient feature extraction.
- **Neck:** Implements **PANet** (Path Aggregation Network) to better capture features at different scales.
- **Head:** A **Decoupled Head** that separates the classification and localization tasks for higher accuracy.

#  Technology Stack
| Component | Tools Used |
| :--- | :--- |
| **Language** | Python 3.13+ |
| **Deep Learning** | Ultralytics YOLOv8 |
| **Web Framework** | Streamlit |
| **Image Processing** | OpenCV, Pillow |
| **Data Handling** | Pandas, NumPy |
| **Environment** | Python Virtual Environment (venv) |

---

##  Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/neeharikajasti/EE26021.git
   ```
2. **Navigate to Folder:**
   ```bash
   cd UAV_Based_Insulator_Classification
   ```
3. **Create Virtual Environment:**
   ```bash
     python -m venv venv
   ```
4. **Activate Environment:**


   Windows: venv\Scripts\activate


   Mac/Linux: source venv/bin/activate

6. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
7. **Run the Dashboard:**
```bash
streamlit run dashboard.py
```
# Deployment Strategy
​

**Environment:** 

Local Python (venv) running on a ground station laptop.
​

**Interface:**

A custom Streamlit Web Dashboard that allows for:
​

**Manual photo uploads:**

Processing high-resolution UAV stills.
​

**Real-time summary tables:**

Instant breakdown of detected insulator materials.

# Project Structure
```bash
UAV_Based_Insulator_Classification/
├── venv/                 # Virtual Environment (local libraries)
├── models/               # Contains trained weights (best.pt)
├── samples/              # Sample UAV images for testing
├── dashboard.py          # Main Streamlit UI Script
├── requirements.txt      # Dependency List for reproducibility
├── data.yaml             # Class definitions (Porcelain, Glass, etc.)
└── README.md             # Project Documentation
```

# Results
<img width="1000" height="1000" alt="confusion_matrix (2)" src="https://github.com/user-attachments/assets/df06e386-af57-46fe-a2b8-5a5be3fdf7be" />
<img width="2400" height="1200" alt="results (3)" src="https://github.com/user-attachments/assets/b50a7633-e3df-47b1-9fa2-24e36887a54c" />
