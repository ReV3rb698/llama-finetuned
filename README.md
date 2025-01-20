---
base_model: meta-llama/Meta-Llama-3-8B
library_name: peft
---

# 🦙 Llama 3 Fine-Tuned Model

## 📌 Overview
This repository contains a fine-tuned **Llama 3 (Meta-Llama-3-8B)** model, trained using **LoRA (Low-Rank Adaptation)**. The model specializes in **job description processing**, extracting **key job responsibilities and required qualifications**.

## 📜 Model Details

- **Base Model:** `meta-llama/Meta-Llama-3-8B`
- **Fine-tuning Library:** PEFT (Parameter-Efficient Fine-Tuning)
- **Language(s):** English
- **License:** Apache 2.0
- **Hardware Used for Training:** NVIDIA A100 (Google Colab Pro)
- **Fine-tuned For:** Job description processing, skills extraction, and structured job posting generation.

## 🔍 **How to Use the Model**
To load and generate text with the fine-tuned model:

```python
import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

# Load the fine-tuned tokenizer and model
tokenizer = PreTrainedTokenizerFast.from_pretrained("path-to-your-model")
model = LlamaForCausalLM.from_pretrained("path-to-your-model")

# Example job description input
input_text = """Job Title: Cloud Engineer
Company: Google
Location: Singapore

Responsibilities:
- Manage cloud infrastructure
- Automate deployments
- Ensure security compliance

Describe the key skills required for this job."""

# Tokenize and generate
inputs = tokenizer(input_text, return_tensors="pt")
with torch.no_grad():
    output = model.generate(inputs['input_ids'], max_length=200)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## 📌 What Was Done

This project fine-tunes the **Meta-Llama-3-8B** model using **LoRA (Low-Rank Adaptation)** on a dataset of job descriptions.

### 🔹 **Steps Taken:**
1. **Data Preparation:**
   - A dataset of job postings was collected and structured into `job_skills.jsonl`.
   - The dataset was formatted into JSONL to make it suitable for LLM fine-tuning.

2. **Fine-Tuning in Google Colab:**
   - `Llama_Colab_Training.ipynb` was used to fine-tune the Llama 3 model.
   - Applied **LoRA** to optimize training efficiency and reduce VRAM usage.
   - Used **PEFT (Parameter-Efficient Fine-Tuning)** for faster adaptation.

3. **Model Training Details:**
   - Training was done on **Google Colab A100 GPUs**.
   - **Hyperparameters:** batch size 2, learning rate 5e-5, 3 epochs.
   - The fine-tuned model was saved in **safetensors format**.

4. **Model Usage:**
   - `main.py` demonstrates how to load the fine-tuned model.
   - The model takes job descriptions and generates **key job skills**.

5. **Repository Updates:**
   - Improved `.gitignore` to exclude large files & model checkpoints.
   - Structured `README.md` to include training details and dataset usage.

---

# Source of the dataset:
[Google Job Skills Dataset](https://www.kaggle.com/datasets/niyamatalmass/google-job-skills?resource=download)

