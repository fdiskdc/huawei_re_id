"""NextGen sample and embedding datasets."""
import random
from flask import Blueprint, jsonify, request
from mrmodn_backend.services.dataset_loader import load_json

datasets_bp = Blueprint("datasets", __name__, url_prefix="/mrmodn/api/v1")


@datasets_bp.get("/sample-sequence")
def sample_sequence():
    sequences = load_json("sample_sequences.json").get("sequences", [])
    return jsonify(sequence=random.choice(sequences) if sequences else "")


def _umap(name: str):
    data = load_json(name)
    limit = request.args.get("limit", type=int)
    if limit and limit > 0 and len(data.get("points", [])) > limit:
        data = dict(data)
        data["points"] = random.sample(data["points"], limit)
        data["metadata"] = dict(data.get("metadata", {}), subsampled=True)
    response = jsonify(data)
    response.headers["Cache-Control"] = "public, max-age=3600"
    return response


@datasets_bp.get("/umap-data")
def umap_human():
    return _umap("umap_human_data.json")


@datasets_bp.get("/umap-cora-data")
def umap_cora():
    return _umap("umap_cora_data.json")

