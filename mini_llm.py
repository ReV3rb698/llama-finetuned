from itertools import chain
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import csv
from tqdm import tqdm

# Template for input to the model
template = """
Job Title: {title}
Company: {company}
Responsibilities: {responsibilities}

Question: Describe the key skills required for this job.
"""

# Initialize the model
model = OllamaLLM(model="smollm:135m")
prompt = ChatPromptTemplate.from_template(template)

# Input and output file paths
input_csv = "job_skills.csv"
output_txt = "llm_results.txt"

# Wipe the output file at the start
with open(output_txt, "w", encoding="utf-8") as outfile:
    outfile.write("")  # Empty the file

# Read the CSV and process each row
with open(input_csv, newline="", encoding="utf-8") as csvfile, open(output_txt, "a", encoding="utf-8") as outfile:
    reader = csv.DictReader(csvfile)
    total_rows = sum(1 for _ in open(input_csv, "r", encoding="utf-8")) - 1  # Get total rows (excluding header)
    csvfile.seek(0)  # Reset the CSV file pointer after counting rows

    for row in tqdm(reader, total=total_rows, desc="Processing rows", unit="row", dynamic_ncols=True):
        # Prepare the input for the model
        input_text = prompt.format(
            title=row["Title"],
            company=row["Company"],
            responsibilities=row["Responsibilities"][:150],  # Truncate for brevity
        )

        # Generate output from the model
        result = model.invoke(input=input_text)

        # Write the input and output to the output file
        outfile.write(f"Input:\n{input_text}\n\nOutput:\n{result}\n\n{'-' * 80}\n")
