"""Attention endpoints shared by both web modes."""
from flask import Blueprint, current_app, jsonify, request
from mrmodn_backend.services.attention import visualize

attention_bp = Blueprint("attention", __name__, url_prefix="/mrmodn/api/v1")


@attention_bp.post("/attention-visualization")
def attention_visualization():
    sequence = (request.get_json(silent=True) or {}).get("rnaSequence", "").strip()
    if not sequence:
        return jsonify(error="No sequence provided"), 400
    return jsonify(visualize(current_app.config["MODEL"], current_app.config["MODEL_DEVICE"], current_app.config["MODEL_CFG"], sequence))


@attention_bp.get("/attention-comparison")
def attention_comparison():
    return jsonify(samples=[], class_names=[], model_names=[])

