# femto-GPT 51.1M
### Self-Attention · Decoder-Only Transformer Architecture · Language Modeling

[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)]()
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-green)]()
[![Hugging Face](https://img.shields.io/badge/deployment-HuggingFace-yellow)]()
[![Vercel](https://img.shields.io/badge/frontend-Vercel-black)]()
[![LoRA](https://img.shields.io/badge/fine--tuning-LoRA-purple)]()
[![DDP](https://img.shields.io/badge/training-DDP%202%20GPUs-orange)]()

---

> Lightweight GPT-style language model built from scratch, trained on multiple datasets, and deployed as a full end-to-end system.

### **Live Demo (Web Page):** https://femto-gpt.vercel.app/
Backend: FastAPI (Dockerized on Hugging Face Spaces)

---

## Team Members
  - Renganath Chokkalingam (Team Lead) - 524170
  - Adwaith Rajani Krishnan - 524103
  - Aswan Sanjeev - 524111


---

## 🏗️ Architecture

| Component        | Value   |
|------------------|--------|
| Token Embedding Dimensions  | 512   |
| Total Parameters  | 51.1 Million   |
| Transformer Block Layers          | 8      |
| Attention Heads | 8      |
| Context Length  | 256    |
| Vocabulary Size | 50,257 |

---

## Project Highlights

### ⚡ **DDP training across 2 GPUs**  
  Introduced Distributed Data Parallel to overcome CUDA out-of-memory constraints and scale training by splitting workloads across two GPUs.

### 🧠 **Decoder-only Transformer built from scratch**  
  Implemented the GPT architecture end-to-end to deeply understand attention mechanisms, token interactions, and training dynamics.

### 🔁 **LoRA fine-tuning for domain adaptation**  
  Instead of retraining the entire model, LoRA was used due to the limited size of a custom synthetic U.S. affairs dataset (generated using Claude Sonnet 4.6) — leveraging the base model’s learned grammar and structure while adapting it to domain-specific knowledge.

### 🎯 **Custom batching strategy (sentence-aligned sampling)**  
  Designed batching based on the requirement that inputs resemble properly starting sentences — e.g., “The president addressed the nation…” instead of mid-sequence fragments like “…he wanted to go and tr…” — improving grammatical structure and overall coherence in a small model.

### 🌐 **Full pipeline: Training → Containerization → API → Client UI**  
  Built and deployed a complete system — covering training, Dockerized containerization, API inference, and frontend delivery.

---

## 🧠 Model Architecture

```
Input Text
   │
   ▼
Tokenization (BPE)
   │
   ▼
Token IDs ────────────────► Token Embedding (50257 × 512)
                                   │
                                   ▼
                    + Positional Embedding (256 × 512)
                                   │
                                   ▼
                                Dropout
                                   │
                                   ▼
               ┌───────────────────────────────────────────────────────┐
               │                 Transformer Block × 8                 │
               │                                                       │
               │   ┌───────────────────────────────────────────────┐   │
               │   │               LayerNorm (ln1)                 │   │
               │   └──────────────┬────────────────────────────────┘   │
               │                  ▼                                    │
               │        c_attn (Linear → 3 × n_embd)                   │
               │                  │                                    │
               │                  ▼                                    │
               │           Split → Q, K, V                             │
               │                  │                                    │
               │                  ▼                                    │
               │      Multi-Head Reshape + Transpose                   │
               │                  │                                    │
               │                  ▼                                    │
               │        Scaled Dot-Product Attention                   │
               │      softmax(QKᵀ / √d) with causal mask               │
               │                  │                                    │
               │                  ▼                                    │
               │         Attention Output (att @ V)                    │
               │                  │                                    │
               │                  ▼                                    │
               │       Merge Heads → c_proj (Linear)                   │
               │                  │                                    │
               │                  ▼                                    │
               │           Residual Connection                         │
               │                                                       │
               │   ┌───────────────────────────────────────────────┐   │
               │   │               LayerNorm (ln2)                 │   │
               │   └──────────────┬────────────────────────────────┘   │
               │                  ▼                                    │
               │         MLP: c_fc → GELU → c_proj                     │
               │                  │                                    │
               │                  ▼                                    │
               │            Residual Connection                        │
               │                                                       │
               └───────────────────────────────────────────────────────┘
                                   │
                                   ▼
                           Final LayerNorm (ln_f)
                                   │
                                   ▼
                  Linear Head (tied with embedding weights)
                                   │
                                   ▼
                            Logits (50,257)
                                   │
                                   ▼
                                Softmax
                                   │
                                   ▼
                         Next Token Prediction
```
---

## 🧠 Models

### 🌐 WEBedu Model (FineWeb-Edu)

Optimized for **educational and structured web text** (science, history, technical explanations).

**Dataset**  
FineWeb-Edu — curated educational websites and technical guides  

**Training Tokens**  
400M  

**Metrics**  
- Train Loss: 3.6942  
- Validation Loss: 3.7563  
- Perplexity: e³·⁷⁵  
- Shortlisted Tokens: 42 / 50,257

**Training Curve**  
<p align="center">
  <img src="loss_curves/fine_web_loss_curve.png" width="60%">
</p>

---

### 🏛️ U.S. Affairs Model (LoRA Fine-tuned)

Specialized in **U.S. political, civic, and public affairs language understanding**.

**Base Model**  
FineWeb-Edu  

**Dataset**  
Custom synthetic dataset made with Claude Sonnet 4.6 (U.S. affairs: presidential, wars, civic topics)  

**Fine-tuned**  
Yes (LoRA)  

**Training Tokens**  
39,490  

**Metrics**  
- Train Loss: 3.0140  
- Validation Loss: 3.8601
- Original Dataset Val Loss: 4.0663
- Perplexity: e³·⁸⁶ (Shortlisted Tokens: 47 / 50,257)

**Training Curve**  
<p align="center">
  <img src="loss_curves/lora_us_affairs_loss_curve.png" width="60%">
</p>

---

### 📖 Narrative Model (TinyStories)

Built for **coherent storytelling, scene continuity, and expressive long-form generation**.

**Dataset**  
TinyStories — fiction passages and screenplay-style samples  

**Training Tokens**  
473.8M  

**Metrics**  
- Train Loss: 1.4308  
- Validation Loss: 1.4368  
- Perplexity: e¹·⁴³  
- Shortlisted Tokens: 4 / 50,257

**Training Curve**  
<p align="center">
  <img src="loss_curves/tiny_stories_loss_curve.png" width="60%">
</p>


---

## ⚙️ Training Pipeline

```
[Raw Data]
     ↓
[Memmap Binary Format]
     ↓
[Custom get_batch]
     ↓
[Training Loop]
     ↓
[DDP (2 GPUs)]
     ↓
[Model Checkpoints]
     ↓
[LoRA Fine-tuning]
```

---

## 🌐 Deployment

```
[User]
   ↓
[Client UI (Vercel hosted)]
   ↓
[FastAPI Backend (Dockerized on Hugging Face Spaces)]
   ↓
[Femto-GPT Models]
```

---

## 🔌 API

```
POST /fineweb-edu/generate
POST /tiny-stories/generate
POST /lora-us-affairs/generate
```
**API:** https://renganathc-femto-gpt-extension.hf.space/  

<p align="center">
  <img src="loss_curves/postman.png" width="85%">
</p>

---

## 📓 Training Notebooks and Logs

- FineWeb-Edu: https://www.kaggle.com/code/renganathc/femto-gpt-fine-web-edu  
- Tiny Stories: https://www.kaggle.com/code/renganathc/femto-gpt-tiny-stories  
- LoRA Fine-tuned US Affairs: https://www.kaggle.com/code/renganathc/femto-lora-us-pres-affairs  

---

## 💡 Why Femto?

Best name I could come up with while creating the repo xD
