from flask import request, jsonify
from services.chromadb_ops import add_documents, delete_documents, search_documents

def register_routes(app):
    @app.route("/")
    def home():
        return "Flask server is running! Use /add, /delete, or /search endpoints with a 'collection_name' parameter."

    @app.route("/add", methods=["POST"])
    def add():
        data = request.json
        result = add_documents(data)
        return jsonify(result)

    @app.route("/delete", methods=["POST"])
    def delete():
        data = request.json
        result = delete_documents(data)
        return jsonify(result)

    @app.route("/search", methods=["POST"])
    def search():
        data = request.json
        result = search_documents(data)
        return jsonify(result)
