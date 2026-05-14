# ♟️ Human-Centric Chess AI (ChessGPT)

> NIT Andhra Pradesh | EE2621 Introduction to Machine Learning
# Rajiv Rajpoot (524166)
# Anand Pal(524109)
# Raj Pandey(524165)

# Project Demo Video link
https://drive.google.com/drive/folders/18yfmvwLHsPjRX8T78bxDBuYRpmSyixzI?usp=sharing

A full-stack AI system that predicts human-like chess moves, evaluates gameplay using Centipawn Loss (CPL), and estimates player ELO.

---

## 🚀 Features

- Transformer-based Chess Model (ChessGPT)
- Human-like move prediction
- Centipawn Loss (CPL) evaluation
- Move + Score dual prediction
- ELO estimation
- Streamlit UI
- Stockfish integration

---

## 📂 Project Structure

```
.
├── data/
├── checkpoints/
├── model.py
├── config.py
├── train.py
├── parse_pgn.py
├── analysis.py
├── chess_app.py
└── README.md
```

---

## ⚙️ Installation

```bash
git clone <your-repo-link>
cd chess-ai
pip install -r requirements.txt
```

---

## ▶️ Usage

### Data Preparation
```bash
python parse_pgn.py
```

### Train Model
```bash
python train.py
```

### Run Analysis
```bash
python analysis.py
```

### Run UI
```bash
streamlit run chess_app.py
```

---

## 📊 Metrics

- Accuracy (Top-1, Top-5)
- ACPL (Average CPL)
- Move Agreement
- CPL Distribution

---

## 🧠 Tech Stack

- Python
- PyTorch
- NumPy
- Streamlit
- Stockfish

---

## 📌 Notes

- Uses Lichess PGN dataset
- Filters applied: ELO, moves, time control
- Designed for human-like play prediction



