"""Thread-safe, lazy CPU runtime for the ReID visualization model."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch

from config import config
from mrmodn_backend.models.reid import CNNTransformerReID, load_pretrained_reid


class ReIDRuntime:
    def __init__(self) -> None:
        self._model: CNNTransformerReID | None = None
        self._checkpoint: dict[str, Any] | None = None
        self._load_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        if config.REID_TORCH_NUM_THREADS < 1:
            raise ValueError("REID_TORCH_NUM_THREADS must be positive")
        torch.set_num_threads(config.REID_TORCH_NUM_THREADS)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def checkpoint_mtime_ns(self) -> int:
        path = Path(config.REID_CHECKPOINT_PATH)
        return path.stat().st_mtime_ns if path.is_file() else 0

    def load_model(self) -> tuple[CNNTransformerReID, dict[str, Any]]:
        if self._model is not None and self._checkpoint is not None:
            return self._model, self._checkpoint
        with self._load_lock:
            if self._model is None or self._checkpoint is None:
                self._model, self._checkpoint = load_pretrained_reid(
                    config.REID_CHECKPOINT_PATH, config.REID_DEVICE
                )
        return self._model, self._checkpoint


_runtime = ReIDRuntime()


def get_reid_runtime() -> ReIDRuntime:
    return _runtime


def get_reid_model() -> tuple[CNNTransformerReID, dict[str, Any]]:
    return _runtime.load_model()
