# 🚀 Domain-Adaptive Pre-Pretraining (DAPT) on LLaMA-2 using LoRA

This project demonstrates an end-to-end workflow for **Domain-Adaptive Pre-Pretraining (DAPT)** of the **LLaMA-2 Large Language Model** using parameter-efficient fine-tuning methods.

The objective is to adapt the **general LLaMA-2 base model to better understand specialized domains BEFORE any instruction fine-tuning or RLHF**, enabling domain-aware representation learning.

---

## 📌 Project Scope

✅ Domain-Adaptive Pre-Training (DAPT) ONLY  
❌ No instruction fine-tuning (SFT)  
❌ No RLHF  
❌ No task-specific supervised training

This repository focuses purely on **pre-pretraining of LLaMA-2 using domain corpora**.

---

## 🧠 Model & Training Setup

| Component | Details |
|----------|----------|
| Base Model | **LLaMA-2** |
| Training Method | Domain-Adaptive Pre-Training (DAPT) |
| Adapter | **LoRA (PEFT)** |
| Optimization | **4-bit quantization (BitsAndBytes)** |
| Framework | PyTorch + HuggingFace Transformers |
| Tokenization | HF Tokenizers |
| Hardware | Low-VRAM training environment |

---

## 📂 Datasets

Three custom text corpora were prepared and used for domain adaptation:

| Dataset File | Domain Focus |
|---------------|---------------|
| `cricket.txt` | Cricket knowledge, game rules, player roles, strategies |
| `education.txt` | Education systems, teaching methods, ed-tech topics |
| `medical.txt` | Healthcare fundamentals, anatomy basics, medical workflows |

All data files are merged and tokenized into a unified DAPT corpus using:


```bash
python prepare_data.py



Training Workflow

Custom Domain Text Files
(cricket.txt, education.txt, medical.txt)
            ↓
prepare_data.py  
(Data cleaning, merging & tokenization)
            ↓
train_dapt_lora.py  
(DAPT training using LoRA adapters)
            ↓
outputs/llama2-dapt  
(LoRA adapter weights saved)
            ↓
generate.py  
(Inference testing)


## ▶️ How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

2️⃣ Prepare datasets
cricket.txt
education.txt
medical.txt

Run preprocessing:
python prepare_data.py

3️⃣ Train DAPT model
python train_dapt_lora.py

Trained LoRA adapters will be saved at:

outputs/llama2-dapt

4️⃣ Test inference
python generate.py




