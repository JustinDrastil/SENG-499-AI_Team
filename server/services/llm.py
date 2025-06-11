import os
import google.generativeai as genai


GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

models = {}

def load_prompt(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "..", "config", f"{filename}.txt")
    with open(config_path, "r") as f:
        return f.read().strip()
    
def initialize():
    global models
    models = {
        "api": genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=load_prompt("api")
        ),
        "answer": genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=load_prompt("answer")
        )
    }

def generate_response(context_text, query, model_key):
    

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