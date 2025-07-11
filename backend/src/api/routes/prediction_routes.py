from flask import Blueprint, request, jsonify
from src.api.models.prediction import get_prediction

prediction_bp = Blueprint("prediction", __name__, url_prefix="/api")


@prediction_bp.post("/predict")
def predict():
    model = request.get_json().get("model")
    if model not in ["lstm", "gru", "conv1d", "ffn"]:
        return jsonify({"error": "Invalid model name"}), 400
    
    # The model name for ffn is ffn_model in the tuner project
    if model == "ffn":
        model = "ffn_model"

    result = get_prediction(model)

    return jsonify(result)