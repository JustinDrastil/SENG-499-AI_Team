# SENG499 AI Subteam Repository for Retrieval Augmented Generation (RAG)

## Setting up your local machine
1. Clone the Github Repository to a folder on your local machine
2. In your terminal, use cd to change your working directory to the new repo (SENG-499-AI_Team)
3. Create a Python Virtual Environment (venv):
	- Command: python -m venv venv
4. Activate the new venv:
	- Linux/macOS: source venv/bin/activate
	- Windows (Command Prompt): venv\Scripts\activate.bat
	- Windows (PowerShell): venv\Scripts\Activate.ps1
5. You are now working within an isolated Python environment.
6. Install the dependencies required for the project's python scripts (time <5 minutes):
	- Command: pip install -r requirements.txt
7. Generate an API Key:
   - Visit [Google AI Studio](https://aistudio.google.com/app/apikey) and create a new API key.

8. Configure Environment Variable:
   - **Windows (PowerShell)**:
     ```powershell
     $env:GOOGLE_API_KEY = "your_api_key_here"
     ```
   - **Linux/macOS (Terminal)**:
     ```bash
     export GOOGLE_API_KEY="your_api_key_here"
     ```

   Replace `your_api_key_here` with the API key you generated from Google AI Studio.
9. Once complete, you should be able to run all non-legacy scripts in the repository.
10. To deactivate the venv:
	- Command: deactivate

## server folder
- app.py: Run this to start the flask server which can be used to perform CRUD operations on the vector database (ChromaDB)

## database folder
- documents folder: Contains text documents that are embedded into ChromaDB on initialization
- chroma_store: Placeholder database for testing purposes, run initialize.py to replace this with an up to date version if needed (time: 15-40+ minutes)
- initialize.py: Once the server is started, run this to create the database and add all documents from the repo's document folder

## legacy folder
- contains legacy code from previous sprint cycles that is not currently being used

## requirements.txt
- contains a list of all python packages that are required by different scripts in the repository
- add new modules to this list when they become required by a script
