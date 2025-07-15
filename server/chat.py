import subprocess
import json
import random


SEARCH_ENDPOINT = "http://localhost:5001/search"
COLLECTION_NAME = "new_collection"
ONC_API_TOKEN = "" # <-- Put your onc token here 
message_history = []

def make_curl_request(query, message_history):
    payload = {
        "query": query,
        "collection_name": COLLECTION_NAME,
        "token": ONC_API_TOKEN,
        "message_history": message_history
    }

    json_data = json.dumps(payload)

    curl_command = [
        "curl",
        "-s",
        "-X", "POST",
        SEARCH_ENDPOINT,
        "-H", "Content-Type: application/json",
        "-d", json_data
    ]

    # For debugging purposes, shows curl request with current message_history
    #print("\n[DEBUG] curl command:", " ".join(f"'{arg}'" if ' ' in arg else arg for arg in curl_command), "\n")

    try:
        result = subprocess.run(curl_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error during curl execution:")
        print(e.stderr)
        return None

def main():
    print("\nWelcome to the ONC Query Assistant!\nPlease input your query when prompted by the USER_MESSAGE input prompt.\nType 'exit' to quit.\n\n")

    while True:
        user_message = input("USER_MESSAGE:\n ").strip()
        if user_message.lower() in ["exit", "quit", "goodbye"]:
            print("\nSYSTEM_MESSAGE:\n", random.choice(["Goodbye!", "See you later!", "Take care!", "OK bud!", "Bye for now!", "Have a great day!"]), "\n")
            break

        response_text = make_curl_request(user_message, message_history)
        if not response_text:
            continue

        try:
            response_json = json.loads(response_text)
            system_message = response_json.get("answer", "[No answer returned]")
            print("\nSYSTEM_MESSAGE:\n", system_message, "\n")

            message_history.append({"actor": "user", "message": user_message})
            message_history.append({"actor": "system", "message": system_message})

        except json.JSONDecodeError:
            print("Error: Invalid JSON response from server.")
            print(response_text)

if __name__ == "__main__":
    main()
