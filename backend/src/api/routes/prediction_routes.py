from flask import Blueprint, request, jsonify
from src.api.models.prediction import (
    lstm_prediction,
    gru_prediction,
    conv1d_prediction,
    ffn_prediction,
)

prediction_bp = Blueprint("prediction", __name__, url_prefix="/api")


@prediction_bp.post("/predict")
def predict():
    model = request.get_json().get("model")
    if model == "lstm":
        result = lstm_prediction()
    elif model == "gru":
        result = gru_prediction()
    elif model == "conv1d":
        result = conv1d_prediction()
    elif model == "ffn":
        result = ffn_prediction()
    else:
        return jsonify({"error": "Invalid model name"}), 400

    return jsonify(result)
        
