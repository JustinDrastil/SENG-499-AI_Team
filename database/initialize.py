import requests
import os

API_URL = "http://localhost:5001/add"
INPUT_DIR = r"./documents"
COLLECTION_NAME = "new_collection"

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
total = len(files)

for idx, filename in enumerate(files, 1):
    file_path = os.path.join(INPUT_DIR, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        text_content = f.read()
    payload = {
        "collection_name": COLLECTION_NAME,
        "text": text_content,
        "doc_tag": filename
    }
    try:
        response = requests.post(API_URL, json=payload)
        status = response.status_code
        if status != 200:
            print(f"{idx}/{total} files added: {filename} | Status: {status}")
            print("Error response:", response.text)
        else:
            print(f"{idx}/{total} files added: {filename} | Status: {status}")
    except Exception as e:
        print(f"{idx}/{total} files added: {filename} | Exception: {e}")
