import os
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime

# Adds .env file entries (GOOGLE_API_KEY) to the environment
load_dotenv()

# Chosen Generative AI Model
DEFAULT_MODEL = "gemini-2.0-flash"
# Token limit for Gemini Pro (adjust as needed)
TOKEN_LIMIT = 1048575

# API Key Configuration with error handling
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("Missing GOOGLE_API_KEY environment variable - see README for instructions")
genai.configure(api_key=GEMINI_API_KEY)

# Models array populated by initialize()
models = {}

# Loads prompts from "../config/"
def load_prompt(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "..", "config", f"{filename}.txt")
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    current_date = datetime.now().strftime("%Y-%m-%d")
    return content.replace("{current_date}", current_date)

# Creates global models dictionary (Function is called in "../app.py")
def initialize():
    global models
    models = {
        "api": genai.GenerativeModel(
            model_name=DEFAULT_MODEL,
            system_instruction=load_prompt("api")
        ),
        "answer": genai.GenerativeModel(
            model_name=DEFAULT_MODEL,
            system_instruction=load_prompt("answer")
        )
    }

def generate_response(context_text, query, model_key, token=None, message_history=None):

    history_text = message_history if message_history else ""

    if model_key == "api" and token:
        # Load and inject the system prompt with the token
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "..", "config", f"api.txt")
            with open(config_path, "r") as f:
                system_prompt = f.read()
        except FileNotFoundError:
            return "Error: system_prompt.txt not found"

        # Override the context_text with the system prompt and query
        prompt = (
            f"{system_prompt}\n\n"
            f"{history_text}\n\n"
            f"Example Input:\n{query.strip()}"
        )
    else:
        # Default formatting
        prompt = (
            f"QUESTION:\n{query.strip()}\n\n"
            f"HISTORY:\n{history_text}\n\n"
            f"CONTEXT:\n{context_text}\n\n"
        )

    try:
        model = models[model_key]
        response = model.generate_content(prompt)
        return response.text
    except KeyError:
        return f"Error: model '{model_key}' not found"

def check_prompt_length(prompt):
    model = models.get("answer")  # or "answer" depending on your case
    if model is None:
        raise ValueError("Model 'answer' not initialized. Did you forget to call initialize()?")

    token_count = model.count_tokens(prompt).total_tokens
    return token_count > TOKEN_LIMIT


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
