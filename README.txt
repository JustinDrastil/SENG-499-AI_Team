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
7. Once complete, you should be able to run all non-legacy scripts in the repository.
8. To deactivate the venv:
	- Command: deactivate

## server folder
- app.py: Run this to start the flask server which can be used to perform CRUD operations on the vector database (ChromaDB)

## database folder
- documents folder: Contains text documents that are embedded into ChromaDB on initialization
- initialize.py: Once the server is started, run this to create the database and add all documents from the repo's document folder

## legacy folder
- contains legacy code that is no longer required

## requirements.txt
- contains a list of all python packages that are required by different scripts in the repository
- add new modules to this list when they become required by a script
