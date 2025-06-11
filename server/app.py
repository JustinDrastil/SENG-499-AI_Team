from flask import Flask
from routes.endpoints import register_routes
from services.llm import initialize

app = Flask(__name__)
initialize()
register_routes(app)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
