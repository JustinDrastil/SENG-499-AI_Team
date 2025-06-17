import os
import google.generativeai as genai


# Chosen Generative AI Model
DEFAULT_MODEL = "gemini-2.0-flash"

# API Key Configuration with error handling
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("Missing Google_API_KEY environment variable - see README for instructions")
genai.configure(api_key=GEMINI_API_KEY)

# Models array populated by initialize()
models = {}

# Loads prompts from "../config/"
def load_prompt(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "..", "config", f"{filename}.txt")
    with open(config_path, "r") as f:
        return f.read().strip()

# Creates global models dictionary (Function is called in "../app.py")
def initialize():
    global models
    models = {
        "api": genai.GenerativeModel(
            model_name=DEFAULT_MODEL,
            system_instruction=load_prompt("system_prompt")
        ),
        "answer": genai.GenerativeModel(
            model_name=DEFAULT_MODEL,
            system_instruction=load_prompt("answer")
        )
    }

def generate_response(context_text, query, model_key, token=None):
    if model_key == "api" and token:
        # Load and inject the system prompt with the token
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "..", "config", f"system_prompt.txt")
            with open(config_path, "r") as f:
                system_prompt = f.read().replace("{token}", token)
        except FileNotFoundError:
            return "Error: system_prompt.txt not found"

        # Override the context_text with the system prompt and query
        prompt = (
            f"{system_prompt}\n\n"
            f"Example Input:\n{query.strip()}"
        )
    else:
        # Default formatting
        prompt = (
            f"QUESTION:\n{query.strip()}\n\n"
            f"CONTEXT:\n{context_text}\n\n"
        )

    try:
        model = models[model_key]
        response = model.generate_content(prompt)
        return response.text
    except KeyError:
        return f"Error: model '{model_key}' not found"

def build_second_llm_prompt(user_query: str, json_data: str) -> str:
    return f"""You are a helpful assistant that translates structured API data into clear, conversational answers for end users.

The user asked the following question:
"{user_query}"

Below is the JSON response from the Ocean Networks Canada API:
{json_data}

Using only the information in the JSON, write a clear and informative natural language answer for the user. 

Include relevant details such as:
- Location name and description
- Latitude and longitude
- Number of deployments
- Whether device data is available
- A link to the dataSearchURL (if provided)

Do not make up any information not found in the JSON. Be concise, factual, and user-friendly."""
