import subprocess

def run_llama3(query, context):
    prompt = f"""Context:
{context}

Question:
{query}

Answer:"""

    result = subprocess.run(
        ["ollama", "run", "llama3", prompt],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return "[Error running LLaMA 3 with Ollama]"

    return result.stdout.strip()
