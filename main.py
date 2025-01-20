import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM, LlamaConfig
from safetensors import safe_open
import os

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

# Example job data from the CSV (or JSONL)
job_data = {
    "title": "Google Cloud Program Manager",
    "company": "Google",
    "location": "Singapore",
    "responsibilities": "Shape, shepherd, ship, and show technical programs designed to support the work of cloud customer engineers...",
    "min_qualifications": "BA/BS degree or equivalent practical experience. 3 years of experience in program and/or project management in cloud computing...",
    "pref_qualifications": "Experience in the business technology market as a program manager in SaaS, cloud computing..."
}

# Formulate the input text for the model
input_text = f"Job Title: {job_data['title']}\nCompany: {job_data['company']}\nLocation: {job_data['location']}\n\n" \
             f"Responsibilities: {job_data['responsibilities']}\n\n" \
             f"Minimum Qualifications: {job_data['min_qualifications']}\n\n" \
             f"Preferred Qualifications: {job_data['pref_qualifications']}\n\n" \
             "Describe the key skills required for this job."

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt")

print("debugging inputs")
print("Input IDs:", inputs["input_ids"])
print("Vocabulary size:", tokenizer.vocab_size)

# Verify all token IDs are within range
max_id = max(inputs["input_ids"][0])
print(f"Max token ID: {max_id}")
if max_id >= model.config.vocab_size:
    print("Error: Token ID exceeds vocabulary size!")


print("done inputs")

# Generate output from the model
with torch.no_grad():  # Disable gradient calculation for inference
    output = model.generate(inputs['input_ids'], max_length=100)

# Decode the output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print("done decoded output")

# Print the generated text
print("Generated Text:", decoded_output)