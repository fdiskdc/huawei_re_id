"""SYSU-MM01 PyTorch data pipeline / SYSU-MM01 的 PyTorch 数据流水线。"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, Sampler


VISIBLE_CAMERAS = (1, 2, 4, 5)
THERMAL_CAMERAS = (3, 6)
INDOOR_VISIBLE_CAMERAS = (1, 2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ImageRecord:
    """一张图像及其协议信息 / One image and its protocol metadata."""

    path: Path
    pid: int
    camid: int
    label: int = -1


def _read_identity_file(path: Path) -> list[int]:
    """读取逗号分隔的身份文件 / Read a comma-separated identity file."""

    if not path.is_file():
        raise FileNotFoundError(f"Identity split file not found / 身份划分文件不存在: {path}")
    values = path.read_text(encoding="utf-8").replace("\n", ",").split(",")
    return sorted({int(value.strip()) for value in values if value.strip()})


def _list_images(directory: Path) -> list[Path]:
    """稳定地列出图像文件 / List image files in deterministic order."""

    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _scan_records(
    data_root: Path,
    identities: Sequence[int],
    cameras: Sequence[int],
    label_map: dict[int, int] | None = None,
) -> list[ImageRecord]:
    """按身份和相机扫描记录 / Scan records by identity and camera."""

    records: list[ImageRecord] = []
    for pid in sorted(identities):
        for camid in cameras:
            directory = data_root / f"cam{camid}" / f"{pid:04d}"
            label = -1 if label_map is None else label_map[pid]
            records.extend(ImageRecord(path, pid, camid, label) for path in _list_images(directory))
    return records


def build_identity_splits(
    data_root: str | Path,
    ratio: Sequence[int] = (7, 2, 1),
    seed: int = 0,
) -> dict[str, list[int]]:
    """按身份构建 train:test:val 划分 / Build identity-level train:test:val splits.

    Identity-level splitting prevents images of the same person from leaking across
    subsets. Counts use largest-remainder allocation so every available identity is used.
    身份级划分避免同一人的图片跨集合泄漏；最大余数法保证所有可用身份都被使用。
    """

    if len(ratio) != 3 or any(value <= 0 for value in ratio):
        raise ValueError("ratio must contain three positive values for train:test:val")
    root = Path(data_root).expanduser().resolve()
    available_ids = _read_identity_file(root / "exp" / "available_id.txt")
    paired_ids = [
        pid
        for pid in available_ids
        if any(_list_images(root / f"cam{camid}" / f"{pid:04d}") for camid in VISIBLE_CAMERAS)
        and any(_list_images(root / f"cam{camid}" / f"{pid:04d}") for camid in THERMAL_CAMERAS)
    ]
    if len(paired_ids) < 3:
        raise RuntimeError("At least three paired identities are required / 至少需要三个双模态身份")

    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(np.asarray(paired_ids, dtype=np.int64)).tolist()
    weights = np.asarray(ratio, dtype=np.float64)
    exact_counts = weights / weights.sum() * len(shuffled)
    counts = np.floor(exact_counts).astype(np.int64)
    remaining = len(shuffled) - int(counts.sum())
    remainder_order = np.argsort(-(exact_counts - counts), kind="stable")
    for index in remainder_order[:remaining]:
        counts[index] += 1

    train_count, test_count, val_count = (int(value) for value in counts)
    train_end = train_count
    test_end = train_end + test_count
    splits = {
        "train": sorted(int(pid) for pid in shuffled[:train_end]),
        "test": sorted(int(pid) for pid in shuffled[train_end:test_end]),
        "val": sorted(int(pid) for pid in shuffled[test_end : test_end + val_count]),
    }
    if any(not identities for identities in splits.values()):
        raise RuntimeError("Every split must contain at least one identity")
    return splits


def _to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    """将 PIL 图像转为标准化 CHW tensor / Convert a PIL image to a normalized CHW tensor."""

    # copy=True prevents a non-writable NumPy view from being shared with PyTorch.
    # copy=True 防止 PyTorch 共享只读的 NumPy 视图。
    array = np.array(image, dtype=np.float32, copy=True) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


class TrainTransform:
    """旧版增强的轻量复现 / Lightweight reproduction of the legacy augmentation."""

    def __init__(self, height: int = 288, width: int = 144, padding: int = 10) -> None:
        self.height = height
        self.width = width
        self.padding = padding

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        image = ImageOps.expand(image, border=self.padding, fill=0)

        max_x = image.width - self.width
        max_y = image.height - self.height
        left = random.randint(0, max_x)
        top = random.randint(0, max_y)
        image = image.crop((left, top, left + self.width, top + self.height))
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return _to_normalized_tensor(image)


class TestTransform:
    """确定性的测试预处理 / Deterministic test preprocessing."""

    def __init__(self, height: int = 288, width: int = 144) -> None:
        self.height = height
        self.width = width

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        return _to_normalized_tensor(image)


class PairedSYSUDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """返回同身份 RGB/IR 图像对 / Return a same-identity RGB/IR image pair.

    The dataset index is a ``(visible_index, thermal_index)`` tuple generated by
    :class:`CrossModalIdentityBatchSampler`.
    Dataset 索引由采样器生成，是 ``(visible_index, thermal_index)`` 二元组。
    """

    def __init__(
        self,
        visible_records: Sequence[ImageRecord],
        thermal_records: Sequence[ImageRecord],
        transform: TrainTransform,
        label_to_pid: dict[int, int],
    ) -> None:
        self.visible_records = list(visible_records)
        self.thermal_records = list(thermal_records)
        self.transform = transform
        self.label_to_pid = dict(label_to_pid)
        self.num_classes = len(self.label_to_pid)

        self.visible_by_label = self._group_by_label(self.visible_records)
        self.thermal_by_label = self._group_by_label(self.thermal_records)
        if set(self.visible_by_label) != set(self.thermal_by_label):
            raise ValueError("Visible/thermal identities are not aligned / RGB 与 IR 身份未对齐")

    @staticmethod
    def _group_by_label(records: Sequence[ImageRecord]) -> dict[int, list[int]]:
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            grouped[record.label].append(index)
        return dict(grouped)

    def __getitem__(self, index_pair: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, int]:
        visible_index, thermal_index = index_pair
        visible_record = self.visible_records[visible_index]
        thermal_record = self.thermal_records[thermal_index]
        if visible_record.label != thermal_record.label:
            raise RuntimeError("Sampler produced mismatched identities / 采样器生成了不匹配的身份")

        with Image.open(visible_record.path) as image:
            visible = self.transform(image.convert("RGB"))
        with Image.open(thermal_record.path) as image:
            # Infrared images are converted to three channels for the shared CNN.
            # 红外图转成三通道，以便与可见光共享 CNN。
            thermal = self.transform(image.convert("RGB"))
        return visible, thermal, visible_record.label

    def __len__(self) -> int:
        return max(len(self.visible_records), len(self.thermal_records))


class CrossModalIdentityBatchSampler(Sampler[list[tuple[int, int]]]):
    """P×K 跨模态身份批采样器 / P-by-K cross-modal identity batch sampler.

    Each batch contains ``identities_per_batch * instances_per_identity`` pairs;
    training concatenates both modalities, so the network batch is twice this size.
    每批含 P×K 个图像对；训练时拼接两种模态，因此网络实际 batch 为 2×P×K。
    """

    def __init__(
        self,
        dataset: PairedSYSUDataset,
        identities_per_batch: int,
        instances_per_identity: int,
        seed: int = 0,
    ) -> None:
        if identities_per_batch < 2:
            raise ValueError("identities_per_batch must be >= 2 for triplet loss")
        if instances_per_identity < 1:
            raise ValueError("instances_per_identity must be >= 1")
        if identities_per_batch > dataset.num_classes:
            raise ValueError("identities_per_batch exceeds the number of training identities")

        self.dataset = dataset
        self.identities_per_batch = identities_per_batch
        self.instances_per_identity = instances_per_identity
        self.seed = seed
        self.epoch = 0
        samples_per_modality = max(len(dataset.visible_records), len(dataset.thermal_records))
        self.num_batches = math.ceil(
            samples_per_modality / (identities_per_batch * instances_per_identity)
        )

    def set_epoch(self, epoch: int) -> None:
        """设置 epoch 以获得可复现的新采样 / Set epoch for reproducible fresh sampling."""

        self.epoch = epoch

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        generator = np.random.default_rng(self.seed + self.epoch)
        labels = np.array(sorted(self.dataset.visible_by_label), dtype=np.int64)

        for _ in range(self.num_batches):
            selected = generator.choice(labels, self.identities_per_batch, replace=False)
            batch: list[tuple[int, int]] = []
            for label_value in selected.tolist():
                label = int(label_value)
                visible_pool = self.dataset.visible_by_label[label]
                thermal_pool = self.dataset.thermal_by_label[label]
                visible = generator.choice(
                    visible_pool,
                    self.instances_per_identity,
                    replace=len(visible_pool) < self.instances_per_identity,
                )
                thermal = generator.choice(
                    thermal_pool,
                    self.instances_per_identity,
                    replace=len(thermal_pool) < self.instances_per_identity,
                )
                batch.extend(zip(visible.tolist(), thermal.tolist()))
            yield batch

    def __len__(self) -> int:
        return self.num_batches


class SYSUTestDataset(Dataset[tuple[torch.Tensor, int, int]]):
    """SYSU query/gallery 数据集 / SYSU query/gallery dataset."""

    def __init__(self, records: Sequence[ImageRecord], transform: TestTransform) -> None:
        self.records = list(records)
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, record.pid, record.camid

    def __len__(self) -> int:
        return len(self.records)


def build_train_dataset(
    data_root: str | Path,
    height: int = 288,
    width: int = 144,
    identities: Sequence[int] | None = None,
) -> PairedSYSUDataset:
    """从显式身份或官方 train+val 构建训练集 / Build from explicit IDs or official train+val."""

    root = Path(data_root).expanduser().resolve()
    if identities is None:
        train_ids = _read_identity_file(root / "exp" / "train_id.txt")
        val_ids = _read_identity_file(root / "exp" / "val_id.txt")
        candidate_ids = sorted(set(train_ids + val_ids))
    else:
        candidate_ids = sorted({int(pid) for pid in identities})
        if not candidate_ids:
            raise ValueError("Training identities must not be empty")

    # Keep only identities available in both modalities, matching paired training.
    # 仅保留两种模态都存在的身份，以满足配对训练要求。
    visible_raw = _scan_records(root, candidate_ids, VISIBLE_CAMERAS)
    thermal_raw = _scan_records(root, candidate_ids, THERMAL_CAMERAS)
    visible_pids = {record.pid for record in visible_raw}
    thermal_pids = {record.pid for record in thermal_raw}
    common_pids = sorted(visible_pids & thermal_pids)
    if not common_pids:
        raise RuntimeError(f"No paired training identities found / 未找到成对训练身份: {root}")

    pid_to_label = {pid: label for label, pid in enumerate(common_pids)}
    label_to_pid = {label: pid for pid, label in pid_to_label.items()}
    visible = _scan_records(root, common_pids, VISIBLE_CAMERAS, pid_to_label)
    thermal = _scan_records(root, common_pids, THERMAL_CAMERAS, pid_to_label)
    return PairedSYSUDataset(visible, thermal, TrainTransform(height, width), label_to_pid)


def build_query_records(
    data_root: str | Path,
    identities: Sequence[int] | None = None,
) -> list[ImageRecord]:
    """构建 IR query（cam3/cam6）/ Build the IR query set (cam3/cam6)."""

    root = Path(data_root).expanduser().resolve()
    test_ids = (
        _read_identity_file(root / "exp" / "test_id.txt")
        if identities is None
        else sorted({int(pid) for pid in identities})
    )
    return _scan_records(root, test_ids, THERMAL_CAMERAS)


def build_gallery_records(
    data_root: str | Path,
    mode: str = "all",
    trial: int = 0,
    identities: Sequence[int] | None = None,
) -> list[ImageRecord]:
    """每个身份/相机随机取一张 RGB gallery / Sample one RGB gallery image per identity/camera."""

    root = Path(data_root).expanduser().resolve()
    test_ids = (
        _read_identity_file(root / "exp" / "test_id.txt")
        if identities is None
        else sorted({int(pid) for pid in identities})
    )
    if mode == "all":
        cameras = VISIBLE_CAMERAS
    elif mode == "indoor":
        cameras = INDOOR_VISIBLE_CAMERAS
    else:
        raise ValueError("mode must be 'all' or 'indoor' / mode 必须为 all 或 indoor")

    generator = random.Random(trial)
    records: list[ImageRecord] = []
    for pid in test_ids:
        for camid in cameras:
            images = _list_images(root / f"cam{camid}" / f"{pid:04d}")
            if images:
                records.append(ImageRecord(generator.choice(images), pid, camid))
    return records


def seed_worker(worker_id: int) -> None:
    """同步 NumPy/Python worker 随机种子 / Synchronize NumPy/Python worker seeds."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_train_loader(
    dataset: PairedSYSUDataset,
    identities_per_batch: int,
    instances_per_identity: int,
    workers: int,
    seed: int,
) -> tuple[DataLoader, CrossModalIdentityBatchSampler]:
    """创建 CUDA 友好的训练 loader / Create a CUDA-friendly training loader."""

    sampler = CrossModalIdentityBatchSampler(
        dataset,
        identities_per_batch=identities_per_batch,
        instances_per_identity=instances_per_identity,
        seed=seed,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return loader, sampler


def create_test_loader(
    records: Sequence[ImageRecord],
    height: int,
    width: int,
    batch_size: int,
    workers: int,
) -> DataLoader:
    """创建确定性 query/gallery loader / Create a deterministic query/gallery loader."""

    dataset = SYSUTestDataset(records, TestTransform(height, width))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
