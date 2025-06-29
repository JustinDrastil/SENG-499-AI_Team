import requests
import json

# Constants that don't change
URL = "http://localhost:5001/search"
HEADERS = {"Content-Type": "application/json"}
COLLECTION_NAME = "new_collection"
TOKEN = "5e3aec6d-8ed0-49bc-9e96-7980704c17ef"
MESSAGE_HISTORY = [
    {"actor": "user", "message": "this is a message"},
    {"actor": "system", "message": "this is an answer"}
]

def main():
    print("Welcome to the ONC assistant query interface.")
    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break

        payload = {
            "query": query,
            "collection_name": COLLECTION_NAME,
            "token": TOKEN,
            "message_history": json.dumps(MESSAGE_HISTORY)
        }

        try:
            response = requests.post(URL, headers=HEADERS, data=json.dumps(payload))
            response.raise_for_status()
            print("\nResponse:")
            print(response.json())  # Or use response.text if it's not JSON
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
