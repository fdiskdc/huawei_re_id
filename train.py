"""CUDA PyTorch training for SYSU-MM01 / SYSU-MM01 的 CUDA PyTorch 训练脚本。"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import (
    ImageRecord,
    build_gallery_records,
    build_identity_splits,
    build_query_records,
    build_train_dataset,
    create_test_loader,
    create_train_loader,
)
from model import CNNTransformerReID


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="PyTorch CNN-Transformer cross-modal ReID / 跨模态行人重识别"
    )
    parser.add_argument("--data-root", type=Path, default=project_root / "SYSU-MM01")
    parser.add_argument("--output-dir", type=Path, default=project_root / "outputs")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--batch-size",
        "--identities-per-batch",
        dest="identities_per_batch",
        type=int,
        default=8,
        help="P: identities per batch / 每批身份数",
    )
    parser.add_argument(
        "--num-pos",
        "--instances-per-identity",
        dest="instances_per_identity",
        type=int,
        default=4,
        help="K: images per identity in each modality / 每身份每模态图像数",
    )
    parser.add_argument("--test-batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--img-height", type=int, default=288)
    parser.add_argument("--img-width", type=int, default=144)

    parser.add_argument("--transformer-dim", type=int, default=256)
    parser.add_argument("--transformer-depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--milestones", type=str, default="20,50")
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument("--triplet-weight", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CUDA automatic mixed precision / CUDA 自动混合精度",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-ratio",
        type=str,
        default="7:2:1",
        help="Identity ratio in train:test:val order / 身份级 train:test:val 比例",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--test-only", action="store_true")

    parser.add_argument("--mode", choices=("all", "indoor"), default="all")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--gallery-trials", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--early-stopping-metric",
        choices=("mAP", "rank1", "mINP"),
        default="mAP",
    )
    parser.add_argument("--disable-early-stopping", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """设置可复现随机种子 / Set reproducible random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device_name: str, gpu: int) -> torch.device:
    """默认强制 CUDA；CPU 仅用于显式调试 / Require CUDA by default; CPU is explicit debug only."""

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but PyTorch cannot access a GPU. "
                "请安装 CUDA 版 PyTorch 并检查 WSL2 GPU 映射；仅调试可传 --device cpu。"
            )
        if gpu < 0 or gpu >= torch.cuda.device_count():
            raise ValueError(f"Invalid GPU index / 无效 GPU 编号: {gpu}")
        torch.cuda.set_device(gpu)
        device = torch.device("cuda", gpu)
        print(f"CUDA device / CUDA 设备: {torch.cuda.get_device_name(gpu)}")
        return device
    print("WARNING: CPU debug mode; training is designed for CUDA / CPU 仅供调试")
    return torch.device("cpu")


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-hard triplet：最远正样本与最近负样本 / Furthest positive and closest negative."""

    distances = torch.cdist(embeddings, embeddings, p=2.0)
    same_identity = labels[:, None].eq(labels[None, :])
    diagonal = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    positive_mask = same_identity & ~diagonal
    negative_mask = ~same_identity

    if not torch.all(positive_mask.any(dim=1)) or not torch.all(negative_mask.any(dim=1)):
        raise RuntimeError(
            "Each anchor needs a positive and a negative; use P>=2 and K>=1 / "
            "每个锚点都需要正负样本，请使用 P>=2、K>=1"
        )

    hardest_positive = distances.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
    hardest_negative = distances.masked_fill(~negative_mask, float("inf")).min(dim=1).values
    loss = F.relu(hardest_positive - hardest_negative + margin).mean()
    accuracy = (hardest_negative > hardest_positive).float().mean()
    return loss, accuracy


class AverageMeter:
    """加权均值统计 / Weighted running average."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)

    def update(self, value: float, count: int) -> None:
        self.total += value * count
        self.count += count


class EarlyStopping:
    """跟踪验证集提升并决定是否停止 / Track validation improvement and decide when to stop."""

    def __init__(self, patience: int, min_delta: float, metric_name: str) -> None:
        if patience < 1:
            raise ValueError("early_stopping_patience must be >= 1")
        if min_delta < 0:
            raise ValueError("early_stopping_min_delta must be >= 0")
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.best_score = float("-inf")
        self.bad_validations = 0

    def update(self, score: float) -> tuple[bool, bool]:
        """返回 (是否提升, 是否应停止) / Return (improved, should_stop)."""

        improved = score > self.best_score + self.min_delta
        if improved:
            self.best_score = score
            self.bad_validations = 0
        else:
            self.bad_validations += 1
        return improved, self.bad_validations >= self.patience

    def state_dict(self) -> dict[str, float | int | str]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "metric_name": self.metric_name,
            "best_score": self.best_score,
            "bad_validations": self.bad_validations,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.patience = int(state.get("patience", self.patience))
        self.min_delta = float(state.get("min_delta", self.min_delta))
        self.metric_name = str(state.get("metric_name", self.metric_name))
        self.best_score = float(state.get("best_score", float("-inf")))
        self.bad_validations = int(state.get("bad_validations", 0))


def build_optimizer(model: CNNTransformerReID, args: argparse.Namespace) -> Optimizer:
    """为 CNN 和 Transformer/head 设置分组学习率 / Use grouped CNN and Transformer/head LRs."""

    cnn_parameters = list(model.cnn.parameters())
    cnn_ids = {id(parameter) for parameter in cnn_parameters}
    other_parameters = [parameter for parameter in model.parameters() if id(parameter) not in cnn_ids]
    groups = [
        {"params": cnn_parameters, "lr": args.lr * args.backbone_lr_mult},
        {"params": other_parameters, "lr": args.lr},
    ]
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            groups,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def parse_milestones(value: str) -> list[int]:
    milestones = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if any(item < 0 for item in milestones):
        raise ValueError("Milestones must be non-negative")
    return milestones


def parse_split_ratio(value: str) -> tuple[int, int, int]:
    """解析 train:test:val 比例 / Parse a train:test:val ratio."""

    normalized = value.replace(",", ":")
    parts = [item.strip() for item in normalized.split(":") if item.strip()]
    if len(parts) != 3:
        raise ValueError("--split-ratio must contain train:test:val, for example 7:2:1")
    ratio = tuple(int(item) for item in parts)
    if any(item <= 0 for item in ratio):
        raise ValueError("Every split ratio value must be positive")
    return ratio


def build_scheduler(optimizer: Optimizer, args: argparse.Namespace) -> LambdaLR:
    """复现旧训练的 warm-up + 阶段衰减 / Reproduce legacy warm-up and staged decay."""

    milestones = parse_milestones(args.milestones)

    def multiplier(epoch: int) -> float:
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            return float(epoch + 1) / float(args.warmup_epochs)
        decay_count = sum(epoch >= milestone for milestone in milestones)
        return args.lr_gamma**decay_count

    return LambdaLR(optimizer, lr_lambda=multiplier)


def save_checkpoint(
    path: Path,
    model: CNNTransformerReID,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    early_stopping: EarlyStopping,
    label_to_pid: dict[int, int],
    data_splits: dict[str, list[int]],
    args: argparse.Namespace,
) -> None:
    """原子保存训练状态 / Atomically save complete training state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    state = {
        "epoch": epoch,
        "early_stopping": early_stopping.state_dict(),
        "model": model.state_dict(),
        "model_config": model.get_config(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "label_to_pid": label_to_pid,
        "data_splits": data_splits,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found / checkpoint 不存在: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def train_one_epoch(
    model: CNNTransformerReID,
    loader: DataLoader,
    optimizer: Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    classification_loss: nn.Module,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    """执行一个 CUDA/AMP 训练 epoch / Run one CUDA/AMP training epoch."""

    model.train()
    loss_meter = AverageMeter()
    id_meter = AverageMeter()
    triplet_meter = AverageMeter()
    id_accuracy_meter = AverageMeter()
    triplet_accuracy_meter = AverageMeter()
    start = time.perf_counter()
    amp_enabled = args.amp and device.type == "cuda"

    progress = tqdm(loader, desc=f"Train {epoch + 1}/{args.epochs}", leave=False)
    for step, (visible, thermal, labels) in enumerate(progress):
        visible = visible.to(device, non_blocking=True)
        thermal = thermal.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images = torch.cat((visible, thermal), dim=0)
        targets = torch.cat((labels, labels), dim=0)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            embeddings, logits = model(images)
            identity_loss = classification_loss(logits, targets)
        # Distance calculation stays FP32 for numerical stability.
        # 距离计算保持 FP32，以提高数值稳定性。
        with torch.cuda.amp.autocast(enabled=False):
            triplet_loss, triplet_accuracy = batch_hard_triplet_loss(
                embeddings.float(), targets, args.triplet_margin
            )
            total_loss = identity_loss.float() + args.triplet_weight * triplet_loss

        scaler.scale(total_loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        sample_count = targets.shape[0]
        identity_accuracy = logits.argmax(dim=1).eq(targets).float().mean()
        loss_meter.update(total_loss.item(), sample_count)
        id_meter.update(identity_loss.item(), sample_count)
        triplet_meter.update(triplet_loss.item(), sample_count)
        id_accuracy_meter.update(identity_accuracy.item(), sample_count)
        triplet_accuracy_meter.update(triplet_accuracy.item(), sample_count)

        if (step + 1) % args.log_interval == 0:
            progress.set_postfix(
                loss=f"{loss_meter.average:.4f}",
                id_acc=f"{100.0 * id_accuracy_meter.average:.1f}%",
                tri_acc=f"{100.0 * triplet_accuracy_meter.average:.1f}%",
            )

    return {
        "loss": loss_meter.average,
        "identity_loss": id_meter.average,
        "triplet_loss": triplet_meter.average,
        "identity_accuracy": id_accuracy_meter.average,
        "triplet_accuracy": triplet_accuracy_meter.average,
        "seconds": time.perf_counter() - start,
    }


@torch.inference_mode()
def extract_features(
    model: CNNTransformerReID,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在 GPU 批量提取归一化特征 / Extract normalized features on the GPU in batches."""

    features: list[np.ndarray] = []
    pids: list[np.ndarray] = []
    camids: list[np.ndarray] = []
    for images, batch_pids, batch_camids in tqdm(loader, desc="Extract", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled and device.type == "cuda"):
            embeddings = model.forward_embedding(images)
        features.append(embeddings.float().cpu().numpy())
        pids.append(batch_pids.numpy())
        camids.append(batch_camids.numpy())
    return np.concatenate(features), np.concatenate(pids), np.concatenate(camids)


def _pad_cmc(cmc: np.ndarray, max_rank: int) -> np.ndarray:
    if cmc.size >= max_rank:
        return cmc[:max_rank]
    fill = float(cmc[-1]) if cmc.size else 0.0
    return np.pad(cmc, (0, max_rank - cmc.size), constant_values=fill)


def compute_sysu_metrics(
    distance_matrix: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_camids: np.ndarray,
    gallery_camids: np.ndarray,
    max_rank: int = 20,
) -> tuple[np.ndarray, float, float]:
    """按官方 SYSU 协议计算 CMC/mAP/mINP / Compute CMC/mAP/mINP with the SYSU protocol."""

    num_query, num_gallery = distance_matrix.shape
    if num_query == 0 or num_gallery == 0:
        raise ValueError("Query and gallery must be non-empty")
    max_rank = min(max_rank, num_gallery)
    indices = np.argsort(distance_matrix, axis=1)
    cmc_values: list[np.ndarray] = []
    average_precisions: list[float] = []
    inps: list[float] = []

    for query_index in range(num_query):
        query_pid = query_pids[query_index]
        query_camid = query_camids[query_index]
        order = indices[query_index]
        # This camera exclusion exactly follows the legacy SYSU evaluator.
        # 该相机排除规则与旧版 SYSU evaluator 保持一致。
        keep = ~((query_camid == 3) & (gallery_camids[order] == 2))
        ranked_pids = gallery_pids[order][keep]
        matches = ranked_pids == query_pid
        if not np.any(matches):
            continue

        # CMC de-duplicates gallery identities as required by the SYSU protocol.
        # SYSU 的 CMC 协议要求 gallery 身份去重。
        first_positions = np.unique(ranked_pids, return_index=True)[1]
        unique_ranked_pids = ranked_pids[np.sort(first_positions)]
        identity_cmc = (unique_ranked_pids == query_pid).astype(np.float32).cumsum()
        identity_cmc[identity_cmc > 1] = 1
        cmc_values.append(_pad_cmc(identity_cmc, max_rank))

        binary_matches = matches.astype(np.float32)
        cumulative = binary_matches.cumsum()
        positive_positions = np.flatnonzero(binary_matches)
        last_positive = int(positive_positions[-1])
        inps.append(float(cumulative[last_positive] / (last_positive + 1.0)))
        precisions = cumulative / (np.arange(binary_matches.size) + 1.0)
        average_precisions.append(float((precisions * binary_matches).sum() / binary_matches.sum()))

    if not cmc_values:
        raise RuntimeError("No valid SYSU queries / 没有可计算的 SYSU query")
    return (
        np.mean(np.stack(cmc_values), axis=0),
        float(np.mean(average_precisions)),
        float(np.mean(inps)),
    )


@torch.inference_mode()
def evaluate_sysu(
    model: CNNTransformerReID,
    data_root: Path,
    query_records: Sequence[ImageRecord],
    gallery_identities: Sequence[int],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """执行 IR→RGB 多 trial 评估 / Run multi-trial IR-to-RGB evaluation."""

    model.eval()
    amp_enabled = args.amp and device.type == "cuda"
    query_loader = create_test_loader(
        query_records,
        args.img_height,
        args.img_width,
        args.test_batch,
        args.workers,
    )
    query_features, query_pids, query_camids = extract_features(
        model, query_loader, device, amp_enabled
    )

    trial_metrics: list[tuple[np.ndarray, float, float]] = []
    for trial in range(args.gallery_trials):
        gallery_records = build_gallery_records(
            data_root,
            mode=args.mode,
            trial=trial,
            identities=gallery_identities,
        )
        gallery_loader = create_test_loader(
            gallery_records,
            args.img_height,
            args.img_width,
            args.test_batch,
            args.workers,
        )
        gallery_features, gallery_pids, gallery_camids = extract_features(
            model, gallery_loader, device, amp_enabled
        )
        # Embeddings are L2 normalized, so negative dot product is a distance.
        # embedding 已 L2 归一化，因此负点积可作为距离。
        distances = -(query_features @ gallery_features.T)
        trial_metrics.append(
            compute_sysu_metrics(
                distances,
                query_pids,
                gallery_pids,
                query_camids,
                gallery_camids,
            )
        )

    cmc = np.mean(np.stack([metric[0] for metric in trial_metrics]), axis=0)
    mean_ap = float(np.mean([metric[1] for metric in trial_metrics]))
    mean_inp = float(np.mean([metric[2] for metric in trial_metrics]))
    return {"cmc": cmc, "mAP": mean_ap, "mINP": mean_inp}


def print_metrics(metrics: dict[str, Any], prefix: str = "Eval") -> None:
    cmc = metrics["cmc"]

    def rank(index: int) -> float:
        return float(cmc[min(index, len(cmc) - 1)])

    print(
        f"{prefix}: Rank-1 {rank(0):.2%} | Rank-5 {rank(4):.2%} | "
        f"Rank-10 {rank(9):.2%} | Rank-20 {rank(19):.2%} | "
        f"mAP {metrics['mAP']:.2%} | mINP {metrics['mINP']:.2%}"
    )


def get_validation_score(metrics: dict[str, Any], metric_name: str) -> float:
    """读取早停指标 / Read the selected early-stopping metric."""

    if metric_name == "rank1":
        return float(metrics["cmc"][0])
    return float(metrics[metric_name])


def make_model_config(args: argparse.Namespace, num_classes: int) -> dict[str, Any]:
    return {
        "num_classes": num_classes,
        "image_height": args.img_height,
        "image_width": args.img_width,
        "transformer_dim": args.transformer_dim,
        "transformer_depth": args.transformer_depth,
        "num_heads": args.num_heads,
        "mlp_ratio": args.mlp_ratio,
        "embedding_dim": args.embedding_dim,
        "dropout": args.dropout,
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if args.eval_every < 1 or args.gallery_trials < 1:
        raise ValueError("eval_every and gallery_trials must be >= 1")

    set_seed(args.seed)
    device = select_device(args.device, args.gpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(args.resume.expanduser().resolve()) if args.resume else None
    if checkpoint is not None and "model_config" in checkpoint:
        # A resumed checkpoint owns its spatial shape because positional tokens are fixed.
        # 断点模型拥有其空间尺寸，因为位置 token 数是固定的。
        args.img_height = int(checkpoint["model_config"]["image_height"])
        args.img_width = int(checkpoint["model_config"]["image_width"])

    if checkpoint is not None and "data_splits" in checkpoint:
        # Reuse exact identities when resuming; never silently reshuffle data.
        # 恢复训练时复用原身份，绝不静默重新划分。
        data_splits = {
            name: [int(pid) for pid in checkpoint["data_splits"][name]]
            for name in ("train", "test", "val")
        }
    else:
        data_splits = build_identity_splits(
            data_root,
            ratio=parse_split_ratio(args.split_ratio),
            seed=args.split_seed,
        )

    (output_dir / "train_args.json").write_text(
        json.dumps(
            {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Loading SYSU-MM01 / 正在加载 SYSU-MM01 ...")
    train_dataset = build_train_dataset(
        data_root,
        args.img_height,
        args.img_width,
        identities=data_splits["train"],
    )
    val_query_records = build_query_records(data_root, identities=data_splits["val"])
    test_query_records = build_query_records(data_root, identities=data_splits["test"])
    print(
        f"Train IDs {train_dataset.num_classes}, RGB {len(train_dataset.visible_records)}, "
        f"IR {len(train_dataset.thermal_records)}"
    )
    print(
        f"Identity split train:test:val = {len(data_splits['train'])}:"
        f"{len(data_splits['test'])}:{len(data_splits['val'])}; "
        f"val queries {len(val_query_records)}, test queries {len(test_query_records)}"
    )

    if checkpoint is not None and "model_config" in checkpoint:
        model_config = dict(checkpoint["model_config"])
        if model_config["num_classes"] != train_dataset.num_classes:
            raise ValueError("Checkpoint class count does not match this dataset")
    else:
        model_config = make_model_config(args, train_dataset.num_classes)

    model = CNNTransformerReID(**model_config).to(device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    classification_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        metric_name=args.early_stopping_metric,
    )
    start_epoch = 0

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        early_state = checkpoint.get("early_stopping")
        if early_state:
            early_stopping.load_state_dict(early_state)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler_state = checkpoint.get("scaler")
            if scaler_state:
                scaler.load_state_dict(scaler_state)
        print(f"Resumed from epoch {start_epoch} / 从 epoch {start_epoch} 恢复")

    if args.test_only:
        if checkpoint is None:
            raise ValueError("--test-only requires --resume / 仅测试模式需要 --resume")
        metrics = evaluate_sysu(
            model,
            data_root,
            test_query_records,
            data_splits["test"],
            device,
            args,
        )
        print_metrics(metrics, prefix="Test")
        return

    train_loader, identity_sampler = create_train_loader(
        train_dataset,
        args.identities_per_batch,
        args.instances_per_identity,
        args.workers,
        args.seed,
    )
    print(
        f"Batches/epoch {len(train_loader)}, network batch "
        f"{2 * args.identities_per_batch * args.instances_per_identity}"
    )

    for epoch in range(start_epoch, args.epochs):
        identity_sampler.set_epoch(epoch)
        statistics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            classification_loss,
            device,
            epoch,
            args,
        )
        scheduler.step()
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        print(
            f"Epoch {epoch + 1:03d}: loss {statistics['loss']:.4f}, "
            f"ID acc {statistics['identity_accuracy']:.2%}, "
            f"Triplet acc {statistics['triplet_accuracy']:.2%}, "
            f"lr {learning_rates}, {statistics['seconds']:.1f}s"
        )

        should_stop = False
        should_evaluate = (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs
        if should_evaluate:
            metrics = evaluate_sysu(
                model,
                data_root,
                val_query_records,
                data_splits["val"],
                device,
                args,
            )
            print_metrics(metrics, prefix=f"Val epoch {epoch + 1}")
            validation_score = get_validation_score(metrics, early_stopping.metric_name)
            improved, should_stop = early_stopping.update(validation_score)
            print(
                f"Early stopping [{early_stopping.metric_name}]: current "
                f"{validation_score:.6f}, best {early_stopping.best_score:.6f}, "
                f"no-improve {early_stopping.bad_validations}/"
                f"{early_stopping.patience}"
            )
            if improved:
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    early_stopping,
                    train_dataset.label_to_pid,
                    data_splits,
                    args,
                )
            if args.disable_early_stopping:
                should_stop = False

        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            early_stopping,
            train_dataset.label_to_pid,
            data_splits,
            args,
        )
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch + 1:03d}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                early_stopping,
                train_dataset.label_to_pid,
                data_splits,
                args,
            )

        if should_stop:
            print(
                f"Early stopping at epoch {epoch + 1}: validation "
                f"{early_stopping.metric_name} did not improve for "
                f"{early_stopping.patience} checks / 触发早停"
            )
            break

    # Restore the validation-selected model before touching the held-out test split.
    # 在访问独立 test 集之前恢复由 val 选出的最佳模型。
    best_checkpoint_path = output_dir / "best.pt"
    if best_checkpoint_path.is_file():
        best_checkpoint = load_checkpoint(best_checkpoint_path)
        model.load_state_dict(best_checkpoint["model"], strict=True)
        print(f"Loaded best validation checkpoint / 已加载最佳验证模型: {best_checkpoint_path}")
    test_metrics = evaluate_sysu(
        model,
        data_root,
        test_query_records,
        data_splits["test"],
        device,
        args,
    )
    print_metrics(test_metrics, prefix="Final test")


if __name__ == "__main__":
    main()
