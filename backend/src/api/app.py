from flask import Flask
from flask_cors import CORS
from src.api.routes.prediction_routes import prediction_bp
import os

app = Flask(__name__)
CORS(app)

app.register_blueprint(prediction_bp)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
