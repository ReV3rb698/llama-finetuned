import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM, LlamaConfig
from safetensors import safe_open
import os
import csv
from tqdm import tqdm

# Load tokenizer and model
tokenizer = PreTrainedTokenizerFast.from_pretrained(os.getcwd())
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cpu")  # Use CPU
model_path = "adapter_model.safetensors"

with safe_open(model_path, framework="pt") as f:
    lora_state_dict = {key.replace("base_model.model.", ""): f.get_tensor(key) for key in f.keys()}

config = LlamaConfig.from_pretrained(os.getcwd())
model = LlamaForCausalLM(config).to(device)

filtered_state_dict = {key: value for key, value in lora_state_dict.items() if "lora_" not in key}
model.load_state_dict(filtered_state_dict, strict=False)
model.resize_token_embeddings(len(tokenizer))
model.eval()

# Path to input and output files
input_csv = "job_skills.csv"
output_txt = "llm_results.txt"

# Process CSV with batch generation
batch_size = 5
batch_inputs = []

with open(input_csv, newline='', encoding='utf-8') as csvfile, open(output_txt, 'w', encoding='utf-8') as outfile:
    reader = csv.DictReader(csvfile)
    total_rows = sum(1 for _ in open(input_csv, "r", encoding='utf-8')) - 1
    csvfile.seek(0)

    for idx, row in enumerate(tqdm(reader, total=total_rows, desc="Processing rows", unit="row", dynamic_ncols=True)):
        input_text = (
            f"Job Title: {row['Title']}\n"
            f"Company: {row['Company']}\n"
            f"Responsibilities: {row['Responsibilities'][:150]}\n\n"
            "Describe key skills for this job."
        )
        batch_inputs.append(input_text)

        if len(batch_inputs) == batch_size or idx == total_rows - 1:
            inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    max_new_tokens=20,
                    do_sample=True,
                    top_k=10,
                    temperature=1.0,
                )
            decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            for inp, out in zip(batch_inputs, decoded_outputs):
                outfile.write(f"Input:\n{inp}\n\nOutput:\n{out}\n\n{'-' * 80}\n")
            batch_inputs = []
