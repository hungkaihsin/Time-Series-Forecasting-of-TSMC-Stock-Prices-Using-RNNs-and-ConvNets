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
    if model not in ["lstm", "gru", "conv1d", "ffn"]:
        return jsonify({"error": "Invalid model name"}), 400
    
    # Map model names to their corresponding prediction functions
    prediction_functions = {
        "lstm": lstm_prediction,
        "gru": gru_prediction,
        "conv1d": conv1d_prediction,
        "ffn": ffn_prediction,
    }

    # Get the prediction function from the dictionary
    prediction_function = prediction_functions.get(model)

    # Call the selected function
    result = prediction_function()

    return jsonify(result)