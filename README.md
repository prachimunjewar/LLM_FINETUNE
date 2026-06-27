# 🧠 MedLLM — Domain-Adaptive LLM Fine-Tuning Pipeline

> Fine-tuning Mistral-7B on Medical QA using QLoRA (4-bit), PEFT, TRL SFTTrainer,
> evaluated with ROUGE + BERTScore + RAGAS, tracked via W&B, deployed with FastAPI + Streamlit.

---
## Live Demo : https://llmfinetune-wruquzawfnttka2ntwtd7e.streamlit.app/ 
## 🎯 Business Problem
General-purpose LLMs hallucinate on medical questions and lack domain precision.
This project answers:
- **Can we fine-tune a 7B model on consumer GPU with QLoRA?**
- **How much does fine-tuning improve domain accuracy?**
- **How do we evaluate and deploy it in production?**

---

## 🔑 Key Results
- Fine-tuned Mistral-7B using QLoRA — **99.3% fewer trainable parameters** vs full fine-tuning
- **34% improvement** in ROUGE-L over base model on medical QA
- BERTScore F1: 0.91 (fine-tuned) vs 0.74 (base)
- Deployed FastAPI inference endpoint + Streamlit comparison dashboard

---

## 🏗️ Architecture
```
MedQA Dataset (HuggingFace)
        ↓
01: Data Preprocessing & Prompt Formatting
        ↓
02: QLoRA Fine-Tuning (4-bit, PEFT + TRL SFTTrainer)
        ↓
03: Evaluation (ROUGE, BERTScore, Perplexity)
        ↓
04: W&B Experiment Tracking & Analysis
        ↓
05: FastAPI Inference Endpoint
        ↓
06: Streamlit Comparison Dashboard
```

---

## 🛠️ Tech Stack
| Category | Tools |
|---|---|
| Base Model | Mistral-7B-Instruct-v0.3 |
| Fine-Tuning | LoRA, QLoRA, PEFT, TRL SFTTrainer |
| Quantization | bitsandbytes (4-bit NF4) |
| Evaluation | ROUGE, BERTScore, RAGAS, Perplexity |
| Tracking | Weights & Biases (W&B) |
| Deployment | FastAPI + Streamlit |
| Infrastructure | Google Colab / Kaggle (free T4 GPU) |

---

## 📁 Project Structure
```
llm_finetune/
├── notebooks/
│   ├── 01_data_preprocessing.py
│   ├── 02_qlora_finetuning.py        ← Run on Colab/Kaggle GPU
│   ├── 03_evaluation.py
│   ├── 04_wandb_analysis.py
│   └── 05_merge_and_export.py
├── api/
│   └── main.py                        ← FastAPI inference server
├── dashboard/
│   └── app.py                         ← Streamlit comparison UI
├── requirements.txt
├── requirements_training.txt           ← GPU-only deps
└── README.md
```

---

## ⚙️ How to Run

### Local (CPU — preprocessing, eval, dashboard)
```bash
pip install -r requirements.txt
python notebooks/01_data_preprocessing.py
python notebooks/03_evaluation.py      # after training
streamlit run dashboard/app.py
```

### GPU Training (Google Colab / Kaggle)
```
1. Upload notebooks/02_qlora_finetuning.py to Colab
2. Runtime → Change runtime → T4 GPU
3. pip install -r requirements_training.txt
4. Run notebook
5. Download adapter/ folder to your project
```

### API
```bash
uvicorn api.main:app --reload
# Visit http://localhost:8000/docs
```

---

## 📝 Dataset
👉 https://huggingface.co/datasets/medalpaca/medical_meadow_medqa

---

## 📄 License
MIT — free for portfolio use.
