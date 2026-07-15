"""NextGen comparison API backed by bundled workbook/CSV data."""
import csv
from flask import Blueprint, jsonify
from mrmodn_backend.services.dataset_loader import DATA_DIR

comparison_bp = Blueprint("comparison", __name__, url_prefix="/mrmodn/api/v1")


def _csv(name):
    with (DATA_DIR / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@comparison_bp.get("/model-comparison")
def model_comparison():
    models = []
    for name, filename in (("mRModN", "res.csv"), ("MultiRM", "multirm_res.csv"), ("ModX", "modx_res.csv")):
        rows = _csv(filename)
        metrics = {k: float(v) for k, v in (rows[0] if rows else {}).items() if v not in (None, "") and k.lower() not in {"model", "class"}}
        models.append({"name": name.lower(), "display_name": name, "metrics": metrics})
    metric_names = sorted({key for model in models for key in model["metrics"]})
    return jsonify(models=models, metric_names=metric_names)


@comparison_bp.get("/mrmodn-classification-heatmap")
def classification_heatmap():
    rows = _csv("res.csv")
    return jsonify(model_name="mRModN", classes=[r.get("class", str(i)) for i, r in enumerate(rows)],
                   metric_names=list(rows[0].keys()) if rows else [], data=rows)


@comparison_bp.get("/dataset-comparison-heatmap")
def dataset_comparison():
    rows = _csv("res.csv")
    return jsonify(dataset_names=["Human"], model_names=["mRModN"],
                   metric_names=list(rows[0].keys()) if rows else [], row_labels=[r.get("class", str(i)) for i, r in enumerate(rows)], data=rows)


@comparison_bp.get("/mrmodn-localization")
def localization():
    rows = _csv("mrmodn_loc.csv")
    return jsonify(model_name="mRModN", classes=[r.get("class", str(i)) for i, r in enumerate(rows)],
                   class_names=[r.get("class", str(i)) for i, r in enumerate(rows)], k_labels=list(rows[0].keys())[1:] if rows else [],
                   k_values=list(range(max(len(rows[0]) - 1, 0))) if rows else [],
                   heatmap=[[float(v) for v in list(r.values())[1:] if v] for r in rows], statistics=_csv("statistic_loc.csv"))


@comparison_bp.get("/mrmodn-loc-comparison")
def localization_comparison():
    files = (("mRModN", "mrmodn_loc.csv"), ("MultiRM", "multirm_loc.csv"), ("ModX", "modx_loc.csv"))
    rows = [(name, _csv(filename)) for name, filename in files]
    labels = list(rows[0][1][0].keys())[1:] if rows and rows[0][1] else []
    heatmap = [[float(v) for row in values for v in list(row.values())[1:] if v] for _, values in rows]
    return jsonify(model_names=[name for name, _ in rows], k_labels=labels, k_values=list(range(len(labels))), heatmap=heatmap)

