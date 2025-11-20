from flask import Blueprint, request, jsonify
from src.api.models.prediction import run_prediction

prediction_bp = Blueprint("prediction", __name__, url_prefix="/api")


@prediction_bp.post("/predict")
def predict():
    model = request.get_json().get("model")
    if model not in ["lstm", "gru", "conv1d", "ffn"]:
        return jsonify({"error": "Invalid model name"}), 400

    # Call the run_prediction function with the selected model
    result = run_prediction(model)

    return jsonify(result)