import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM, LlamaConfig
from safetensors import safe_open
import os
import csv

# Use PreTrainedTokenizerFast instead of LlamaTokenizer to load the pre-trained tokenizer
# The tokenizer is used to convert raw text into a format that can be processed by the model
# We're using the current working directory (os.getcwd()) as the path to the pre-trained model

# Use PreTrainedTokenizerFast instead of LlamaTokenizer
#tokenizer = PreTrainedTokenizerFast.from_pretrained("C:/Users/xetro/Downloads/llama-finetuned")
tokenizer = PreTrainedTokenizerFast.from_pretrained(os.getcwd())
# Set the pad token to the eos token
tokenizer.pad_token = tokenizer.eos_token

print("done tokenizer")

# Load the model using safetensors
#model_path = "C:/Users/xetro/Downloads/llama-finetuned/adapter_model.safetensors"
# Safetensors is a library that allows us to load and manipulate large models efficiently
# We're loading the model from a file called "adapter_model.safetensors" in the current working directory
model_path = "adapter_model.safetensors"

print("done model path")

# Using safetensors to load the model
with safe_open(model_path, framework="pt") as f:
    # We're loading the LoRA (Low-Rank Adaptation) weights from the model file
    # LoRA is a technique for fine-tuning large language models
    # We're removing the "base_model.model." prefix from the weight names
    lora_state_dict = {key.replace("base_model.model.", ""): f.get_tensor(key) for key in f.keys()}

print("done lora state dict")

# Initialize the model configuration
#config = LlamaConfig.from_pretrained("C:/Users/xetro/Downloads/llama-finetuned")
# We're loading the pre-trained model configuration from the current working directory
config = LlamaConfig.from_pretrained(os.getcwd())

print("done config")

# Initialize the model with the configuration
# We're creating a new instance of the LlamaForCausalLM model with the loaded configuration
model = LlamaForCausalLM(config)

print("done model")

'''

# Update the base model weights with the LoRA weights
model_state_dict = model.state_dict()
model_state_dict.update(lora_state_dict)

# Load the updated state dictionary into the model
model.load_state_dict(model_state_dict)
'''

# Filter out LoRA-specific keys from the LoRA state dictionary
filtered_state_dict = {
    key: value for key, value in lora_state_dict.items() if "lora_" not in key
}

print("done filtered state dict")

# Load the filtered weights into the base model
model.load_state_dict(filtered_state_dict, strict=False)
model.resize_token_embeddings(len(tokenizer))


print("done load state dict")


# Ensure the model is in evaluation mode
model.eval()

print("done eval")

# Path to input CSV and output TXT
input_csv = "job_skills.csv"
output_txt = "llm_results.txt"

# Open CSV and process each row
with open(input_csv, newline='', encoding='utf-8') as csvfile, open(output_txt, 'w', encoding='utf-8') as outfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        # Generate input text for the model
        input_text = (
            f"Job Title: {row['Title']}\n"
            f"Company: {row['Company']}\n"
            f"Category: {row['Category']}\n"
            f"Location: {row['Location']}\n\n"
            f"Responsibilities: {row['Responsibilities']}\n\n"
            f"Minimum Qualifications: {row['Minimum Qualifications']}\n\n"
            f"Preferred Qualifications: {row['Preferred Qualifications']}\n\n"
            "Describe the key skills required for this job."
        )

        # Tokenize input text
        inputs = tokenizer(input_text, return_tensors="pt")

        # Generate model output
        with torch.no_grad():
            #output = model.generate(inputs['input_ids'], max_length=300)
            output = model.generate(inputs['input_ids'], max_new_tokens=100)

        # Decode output and write to file
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        outfile.write(f"Input:\n{input_text}\n\nOutput:\n{decoded_output}\n\n{'-' * 80}\n")