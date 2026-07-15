"""Generate four aligned 9x5 body heatmaps for a four-sample ReID batch."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import torch

from config import config
from mrmodn_backend.models.reid_runtime import get_reid_model, get_reid_runtime
from mrmodn_backend.services.reid_dataset import ReIDSample, dataset_service


STAGES = ("position_embedding", "transformer_0", "transformer_1", "classifier")


def _normalize_batch(values: torch.Tensor, robust: bool) -> tuple[torch.Tensor, float, float]:
    detached = values.detach().float().cpu()
    flat = detached.flatten()
    if robust and flat.numel() > 2:
        low = torch.quantile(flat, 0.01)
        high = torch.quantile(flat, 0.99)
    else:
        low = flat.min()
        high = flat.max()
    denominator = high - low
    if not torch.isfinite(denominator) or denominator.item() <= 1e-12:
        normalized = torch.zeros_like(detached)
    else:
        normalized = ((detached - low) / denominator).clamp(0.0, 1.0)
    return normalized, float(flat.min().item()), float(flat.max().item())


class ReIDHeatmapService:
    def __init__(self) -> None:
        self._cache: OrderedDict[tuple[int, str, int], dict[str, Any]] = OrderedDict()
        self._cache_lock = threading.Lock()

    def _cache_get(self, key: tuple[int, str, int]) -> dict[str, Any] | None:
        with self._cache_lock:
            value = self._cache.get(key)
            if value is not None:
                self._cache.move_to_end(key)
            return value

    def _cache_put(self, key: tuple[int, str, int], value: dict[str, Any]) -> None:
        with self._cache_lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > max(config.REID_CACHE_SIZE, 1):
                self._cache.popitem(last=False)

    def generate_batch(self, split: str, batch_index: int) -> dict[str, Any]:
        runtime = get_reid_runtime()
        cache_key = (runtime.checkpoint_mtime_ns, split, batch_index)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        model, checkpoint = get_reid_model()
        data_splits = checkpoint["data_splits"]
        if split not in data_splits:
            raise KeyError(f"unsupported split: {split}")
        samples, total_batches = dataset_service.get_batch(data_splits[split], batch_index)
        images = dataset_service.preprocess(samples, model.image_height, model.image_width)

        with runtime.inference_lock, torch.enable_grad():
            outputs = model.forward_visualization(images)
            logits = outputs["logits"]
            predicted = logits.argmax(dim=1)
            selected = logits.gather(1, predicted[:, None]).sum()
            gradients = torch.autograd.grad(selected, outputs["feature_map"], retain_graph=False)[0]

            position = torch.linalg.vector_norm(outputs["position_tokens"], ord=2, dim=-1)
            position = position.reshape(-1, model.grid_height, model.grid_width)
            if len(outputs["attentions"]) < 2:
                raise RuntimeError("ReID checkpoint must contain two Transformer blocks")
            transformer_0 = outputs["attentions"][0][:, :, 0, 1:].mean(dim=1)
            transformer_1 = outputs["attentions"][1][:, :, 0, 1:].mean(dim=1)
            transformer_0 = transformer_0.reshape(-1, model.grid_height, model.grid_width)
            transformer_1 = transformer_1.reshape(-1, model.grid_height, model.grid_width)
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            classifier = torch.relu((weights * outputs["feature_map"]).sum(dim=1))
            probabilities = torch.softmax(logits.detach(), dim=1)
            scores = probabilities.gather(1, predicted[:, None]).squeeze(1)

        stage_tensors = {
            "position_embedding": position,
            "transformer_0": transformer_0,
            "transformer_1": transformer_1,
            "classifier": classifier,
        }
        normalized: dict[str, dict[str, Any]] = {}
        for stage, values in stage_tensors.items():
            maps, raw_min, raw_max = _normalize_batch(
                values, robust=stage in {"position_embedding", "classifier"}
            )
            normalized[stage] = {
                "values": maps.tolist(),
                "rawMin": raw_min,
                "rawMax": raw_max,
            }

        label_to_pid = checkpoint["label_to_pid"]
        response_samples = []
        for index, sample in enumerate(samples):
            class_index = int(predicted[index].item())
            response_samples.append(
                self._serialize_sample(
                    sample,
                    class_index,
                    int(label_to_pid.get(class_index, label_to_pid.get(str(class_index), -1))),
                    float(scores[index].item()),
                    normalized,
                    index,
                )
            )

        result = {
            "batchId": f"{split}-{batch_index:06d}",
            "batchIndex": batch_index,
            "batchSize": len(response_samples),
            "totalBatches": total_batches,
            "gridShape": [model.grid_height, model.grid_width],
            "displaySize": [config.REID_DISPLAY_SIZE, config.REID_DISPLAY_SIZE],
            "stages": list(STAGES),
            "samples": response_samples,
        }
        self._cache_put(cache_key, result)
        return result

    @staticmethod
    def _serialize_sample(
        sample: ReIDSample,
        class_index: int,
        predicted_pid: int,
        score: float,
        normalized: dict[str, dict[str, Any]],
        index: int,
    ) -> dict[str, Any]:
        return {
            "sampleId": sample.sample_id,
            "pid": sample.pid,
            "camera": sample.camera,
            "modality": sample.modality,
            "relativePath": sample.relative_path,
            "prediction": {
                "classIndex": class_index,
                "pid": predicted_pid,
                "score": score,
            },
            "heatmaps": {
                stage: {
                    "values": data["values"][index],
                    "rawMin": data["rawMin"],
                    "rawMax": data["rawMax"],
                }
                for stage, data in normalized.items()
            },
        }


heatmap_service = ReIDHeatmapService()
