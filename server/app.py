from flask import Flask
from routes.endpoints import register_routes

app = Flask(__name__)
register_routes(app)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
