import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM, LlamaConfig
from safetensors import safe_open

# Use PreTrainedTokenizerFast instead of LlamaTokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("C:/Users/xetro/Downloads/llama-finetuned")

# Load the model using safetensors
model_path = "C:/Users/xetro/Downloads/llama-finetuned/adapter_model.safetensors"

# Using safetensors to load the model
with safe_open(model_path, framework="pt") as f:
    lora_state_dict = {key.replace("base_model.model.", ""): f.get_tensor(key) for key in f.keys()}

# Initialize the model configuration
config = LlamaConfig.from_pretrained("C:/Users/xetro/Downloads/llama-finetuned")

# Initialize the model with the configuration
model = LlamaForCausalLM(config)



# Update the base model weights with the LoRA weights
model_state_dict = model.state_dict()
model_state_dict.update(lora_state_dict)

# Load the updated state dictionary into the model
model.load_state_dict(model_state_dict)

# Ensure the model is in evaluation mode
model.eval()

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

# Generate output from the model
with torch.no_grad():  # Disable gradient calculation for inference
    output = model.generate(inputs['input_ids'], max_length=100)

# Decode the output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated text
print("Generated Text:", decoded_output)