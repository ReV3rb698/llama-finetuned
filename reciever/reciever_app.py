from connexion import NoContent
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import re
import json
import logging.config
import yaml
import torch
import connexion

# Template for input to the model
template = """
Job Title: {title}
Company: {company}
Responsibilities: {responsibilities}
My Skills and Experience: {skills}
Question: Describe skills required for this job and compare with the skills I have. Output the result in the following JSON format:
{{
    "matches": [
        {{
            "skill": "<skill>",
            "your_experience": "<your_experience>"
        }}
    ],
    "non_matches": [
        {{
            "reason": "<reason>",
            "skill": "<skill>"
        }}
    ],
    "skills_required": {{
        "communication_and_collaboration": [
            "<skill>"
        ],
        "integration_expertise": [
            "<skill>"
        ],
        "programming_languages": [
            "<skill>"
        ]
    }},
    "suggested_skills": [
        {{
            "description": "<description>",
            "skill": "<skill>"
        }}
    ]
}}
"""
ollama_model = "llama3.2"
# Initialize the model
model = OllamaLLM(model=ollama_model if torch.cuda.is_available() else ollama_model, device="cuda" if torch.cuda.is_available() else "cpu")
prompt = ChatPromptTemplate.from_template(template)

# Initialize Logger
with open('log_conf.yml', 'r') as f:
    log_config = yaml.safe_load(f)

logging.config.dictConfig(log_config)
logger = logging.getLogger('basicLogger')

def request_skills(body):
    """
    Processes the input data and returns the skills comparison result.

    Args:
        body (dict): A dictionary containing 'title', 'company', 'responsibilities', and 'skills'.

    Returns:
        tuple: The result from the model invocation and the status code.
    """
    input_text = prompt.format(
        title=body['title'],
        company=body['company'],
        responsibilities=body['responsibilities'][:150] if len(body['responsibilities']) > 150 else body['responsibilities'],
        skills=", ".join(body['skills'])
    )
    try:
        result = model.invoke(input=input_text)
        logger.info("Skills request processed")
        logger.debug(f"Model output: {result}")

        # Extract JSON from the result using regular expression
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            json_result = json.loads(json_match.group())
            return json_result, 200
        else:
            logger.error("No JSON found in the model output")
            return {"error": "No JSON found in the model output"}, 500
    except Exception as e:
        logger.error(f"Error invoking model: {e}")
        return {"error": "Failed to process skills request"}, 500

app = connexion.FlaskApp(__name__, specification_dir='')
app.add_api("llm_reciever_api.yml", strict_validation=True, validate_responses=True)

if __name__ == "__main__":
    app.run(port=5000)