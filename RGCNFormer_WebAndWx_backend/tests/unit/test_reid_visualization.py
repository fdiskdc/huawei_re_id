"""Focused tests for the CPU ReID visualization path."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import config
from mrmodn_backend.models.reid import CNNTransformerReID, load_pretrained_reid
from mrmodn_backend.services.reid_dataset import ReIDDatasetService
from mrmodn_backend.services.reid_heatmap import _normalize_batch


def _small_model() -> CNNTransformerReID:
    model = CNNTransformerReID(
        num_classes=5,
        image_height=64,
        image_width=32,
        transformer_dim=32,
        transformer_depth=2,
        num_heads=4,
        embedding_dim=16,
        dropout=0.0,
    )
    return model.eval()


def test_visualization_forward_matches_normal_forward():
    torch.manual_seed(1)
    model = _small_model()
    images = torch.randn(2, 3, 64, 32)
    with torch.no_grad():
        normal_embedding, normal_logits = model(images)
        visualization = model.forward_visualization(images)
    torch.testing.assert_close(visualization["embedding"], normal_embedding)
    torch.testing.assert_close(visualization["logits"], normal_logits)
    assert visualization["position_tokens"].shape == (2, 2, 32)
    assert len(visualization["attentions"]) == 2
    assert visualization["attentions"][0].shape == (2, 4, 3, 3)


def test_checkpoint_loader_uses_model_config_and_strict_state(tmp_path: Path):
    model = _small_model()
    checkpoint = {
        "model_config": model.get_config(),
        "model": model.state_dict(),
        "label_to_pid": {index: index + 10 for index in range(5)},
        "data_splits": {"train": [1, 2], "test": [3, 4], "val": [5, 6]},
    }
    path = tmp_path / "best.pt"
    torch.save(checkpoint, path)
    loaded, metadata = load_pretrained_reid(path, "cpu")
    assert loaded.get_config() == model.get_config()
    assert metadata["data_splits"]["test"] == [3, 4]
    assert next(loaded.parameters()).device.type == "cpu"


def test_dataset_returns_two_rgb_ir_pairs(tmp_path: Path, monkeypatch):
    for camera in range(1, 7):
        (tmp_path / f"cam{camera}").mkdir()
    for pid in (1, 2):
        for camera in (1, 3):
            directory = tmp_path / f"cam{camera}" / f"{pid:04d}"
            directory.mkdir()
            Image.new("RGB", (16, 32), color=(pid * 20, camera * 20, 50)).save(
                directory / "sample.jpg"
            )
    monkeypatch.setattr(config, "REID_DATA_ROOT", str(tmp_path))
    service = ReIDDatasetService()
    samples, total = service.get_batch([1, 2], 0)
    assert total == 1
    assert len(samples) == 4
    assert [(sample.pid, sample.modality) for sample in samples] == [
        (1, "visible"),
        (1, "infrared"),
        (2, "visible"),
        (2, "infrared"),
    ]
    tensor = service.preprocess(samples, 64, 32)
    assert tensor.shape == (4, 3, 64, 32)
    assert service.resolve_sample_image(samples[0].sample_id) == samples[0].path
    assert service.resolve_sample_image("../outside") is None


def test_batch_normalization_is_finite_and_shared():
    values = torch.tensor([[[0.0, 1.0]], [[2.0, 3.0]]])
    normalized, raw_min, raw_max = _normalize_batch(values, robust=False)
    assert np.isfinite(normalized.numpy()).all()
    assert normalized.min().item() == 0.0
    assert normalized.max().item() == 1.0
    assert (raw_min, raw_max) == (0.0, 3.0)
