"""REST API for deterministic four-sample ReID body heatmaps."""

from __future__ import annotations

from flask import Blueprint, jsonify, send_file

from config import config, get_logger
from mrmodn_backend.models.reid_runtime import get_reid_model, get_reid_runtime
from mrmodn_backend.services.reid_dataset import dataset_service
from mrmodn_backend.services.reid_heatmap import STAGES, heatmap_service


reid_bp = Blueprint("reid", __name__, url_prefix="/api/v1/reid")
logger = get_logger("api.reid")


def _service_unavailable(message: str = "ReID visualization service is unavailable"):
    return jsonify({"error": message, "status": "unavailable"}), 503


@reid_bp.get("/meta")
def get_reid_meta():
    try:
        dataset_service.validate()
        model, checkpoint = get_reid_model()
        splits = checkpoint["data_splits"]
        return jsonify(
            {
                "status": "ready",
                "device": "cpu",
                "torchThreads": config.REID_TORCH_NUM_THREADS,
                "batchSize": config.REID_BATCH_SIZE,
                "gridShape": [model.grid_height, model.grid_width],
                "displaySize": [config.REID_DISPLAY_SIZE, config.REID_DISPLAY_SIZE],
                "stages": list(STAGES),
                "defaultSplit": "test",
                "splits": list(splits),
                "totalBatches": {
                    name: len({int(pid) for pid in identities}) // 2
                    for name, identities in splits.items()
                },
                "modelConfig": checkpoint["model_config"],
                "checkpoint": "outputs/best.pt",
                "loaded": get_reid_runtime().is_loaded,
            }
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.exception("ReID metadata initialization failed")
        return _service_unavailable(str(exc))


@reid_bp.get("/batches/<int:batch_index>")
def get_reid_batch(batch_index: int):
    from flask import request

    split = request.args.get("split", "test").strip().lower()
    if split not in {"train", "test", "val"}:
        return jsonify({"error": "split must be train, test, or val"}), 400
    try:
        result = heatmap_service.generate_batch(split, batch_index)
        for sample in result["samples"]:
            sample["imageUrl"] = (
                f"/mrmodn/api/v1/reid/samples/{sample['sampleId']}/image"
            )
        return jsonify(result)
    except (KeyError, IndexError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        logger.exception("ReID data or checkpoint is unavailable")
        return _service_unavailable(str(exc))
    except RuntimeError:
        logger.exception("ReID batch computation failed")
        return jsonify({"error": "ReID batch computation failed"}), 500


@reid_bp.get("/samples/<sample_id>/image")
def get_reid_sample_image(sample_id: str):
    if len(sample_id) != 24 or any(character not in "0123456789abcdef" for character in sample_id):
        return jsonify({"error": "sample not found"}), 404
    path = dataset_service.resolve_sample_image(sample_id)
    if path is None:
        return jsonify({"error": "sample not found"}), 404
    return send_file(path, mimetype="image/jpeg", conditional=True, etag=True, max_age=3600)
