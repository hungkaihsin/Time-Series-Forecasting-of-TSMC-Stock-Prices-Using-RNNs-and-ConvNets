from flask import Flask, request, jsonify
from flask_cors import CORS
from src.api.routes.prediction_routes import prediction_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(prediction_bp)



if __name__ == '__main__':
    app.run(debug=True)