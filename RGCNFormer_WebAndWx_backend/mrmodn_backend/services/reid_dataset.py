"""Read-only, deterministic SYSU-MM01 batches for ReID visualization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from config import config


VISIBLE_CAMERAS = (1, 2, 4, 5)
THERMAL_CAMERAS = (3, 6)
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass(frozen=True)
class ReIDSample:
    sample_id: str
    pid: int
    camera: int
    modality: str
    path: Path
    relative_path: str


class ReIDDatasetService:
    def __init__(self) -> None:
        self.data_root = Path(config.REID_DATA_ROOT).expanduser().resolve()
        self._sample_paths: dict[str, Path] = {}

    def validate(self) -> None:
        if not self.data_root.is_dir():
            raise FileNotFoundError("Configured SYSU-MM01 directory is unavailable")
        missing = [f"cam{camera}" for camera in range(1, 7) if not (self.data_root / f"cam{camera}").is_dir()]
        if missing:
            raise FileNotFoundError("SYSU-MM01 camera directories are incomplete")

    def _records(self, pid: int, cameras: Iterable[int], modality: str) -> list[ReIDSample]:
        records: list[ReIDSample] = []
        for camera in cameras:
            directory = self.data_root / f"cam{camera}" / f"{pid:04d}"
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.jpg")):
                relative = path.relative_to(self.data_root).as_posix()
                sample_id = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:24]
                resolved = path.resolve()
                self._sample_paths[sample_id] = resolved
                records.append(
                    ReIDSample(sample_id, pid, camera, modality, resolved, relative)
                )
        return records

    def _valid_identities(self, identities: Iterable[int]) -> list[int]:
        valid: list[int] = []
        for pid in sorted({int(value) for value in identities}):
            visible = self._records(pid, VISIBLE_CAMERAS, "visible")
            thermal = self._records(pid, THERMAL_CAMERAS, "infrared")
            if visible and thermal:
                valid.append(pid)
        return valid

    def batch_count(self, identities: Iterable[int]) -> int:
        self.validate()
        return len(self._valid_identities(identities)) // 2

    def get_batch(self, identities: Iterable[int], batch_index: int) -> tuple[list[ReIDSample], int]:
        self.validate()
        valid = self._valid_identities(identities)
        total_batches = len(valid) // 2
        if total_batches < 1:
            raise ValueError("The selected split has fewer than two paired identities")
        if batch_index < 0 or batch_index >= total_batches:
            raise IndexError(f"batch index must be between 0 and {total_batches - 1}")

        selected = valid[batch_index * 2 : batch_index * 2 + 2]
        samples: list[ReIDSample] = []
        for pid in selected:
            visible = self._records(pid, VISIBLE_CAMERAS, "visible")
            thermal = self._records(pid, THERMAL_CAMERAS, "infrared")
            samples.append(visible[batch_index % len(visible)])
            samples.append(thermal[batch_index % len(thermal)])
        if len(samples) != config.REID_BATCH_SIZE:
            raise RuntimeError("ReID visualization batch must contain exactly four samples")
        return samples, total_batches

    def preprocess(self, samples: list[ReIDSample], height: int, width: int) -> torch.Tensor:
        tensors: list[torch.Tensor] = []
        for sample in samples:
            with Image.open(sample.path) as source:
                image = source.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
                array = np.asarray(image, dtype=np.float32) / 255.0
            normalized = (array - IMAGENET_MEAN) / IMAGENET_STD
            tensors.append(torch.from_numpy(normalized).permute(2, 0, 1))
        return torch.stack(tensors, dim=0).contiguous()

    def resolve_sample_image(self, sample_id: str) -> Path | None:
        path = self._sample_paths.get(sample_id)
        if path is None or not path.is_file():
            return None
        try:
            path.relative_to(self.data_root)
        except ValueError:
            return None
        return path


dataset_service = ReIDDatasetService()
