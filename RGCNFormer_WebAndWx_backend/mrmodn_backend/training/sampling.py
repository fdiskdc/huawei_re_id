"""
中文：多标签平衡采样器。提供 MultilabelBalancedBatchSampler 和 DynamicBalancedBatchSampler，确保每个训练批次中各类别样本均衡。
English: Multi-label balanced batch samplers. Provides MultilabelBalancedBatchSampler and DynamicBalancedBatchSampler to ensure balanced class representation per training batch.
"""
import numpy as np
from typing import List
from torch.utils.data import Sampler


class MultilabelBalancedBatchSampler(Sampler):
    """
    中文：多标签平衡批次采样器。确保每个批次包含所有 12 类的均衡样本表示，通过过采样解决极端类别不平衡问题。
    English: Multi-label balanced batch sampler. Ensures each batch contains balanced representation of all 12 classes, addressing extreme class imbalance via oversampling.
    """

    def __init__(self, dataset, train_indices: List[int], batch_size: int,
                 num_classes: int = 12, random_seed: int = 42):
        if batch_size % num_classes != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by "
                f"num_classes ({num_classes})"
            )

        self.dataset = dataset
        self.train_indices = train_indices
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.n_samples_per_class = batch_size // num_classes
        self.random_seed = random_seed

        np.random.seed(random_seed)

        self.class_buckets = self._build_class_buckets()

        min_bucket_size = min(len(bucket) for bucket in self.class_buckets)
        self.num_batches = min_bucket_size // self.n_samples_per_class

    def _build_class_buckets(self) -> List[np.ndarray]:
        """
        Organize train_indices into class-specific buckets.

        Returns:
            List of num_classes arrays with RELATIVE indices (positions within train_indices)
        """
        train_labels = self.dataset.y_12class[self.train_indices]

        class_buckets = []
        for class_idx in range(self.num_classes):
            mask = train_labels[:, class_idx] == 1
            class_indices = np.where(mask)[0]
            class_buckets.append(class_indices)

        for bucket in class_buckets:
            np.random.shuffle(bucket)

        return class_buckets

    def _get_samples_from_bucket(self, bucket_idx: int, n: int) -> np.ndarray:
        """
        Get n samples from a class bucket with cycling/oversampling.

        Args:
            bucket_idx: Index of the class bucket
            n: Number of samples to fetch

        Returns:
            Array of n sample indices
        """
        bucket = self.class_buckets[bucket_idx]
        bucket_size = len(bucket)

        if bucket_size >= n:
            samples = bucket[:n]
            self.class_buckets[bucket_idx] = np.concatenate([bucket[n:], bucket[:n]])
        else:
            np.random.shuffle(bucket)
            times = n // bucket_size + 1
            samples = np.tile(bucket, times)[:n]

        return samples

    def __iter__(self):
        """
        Generate batches with balanced class representation.
        """
        for batch_idx in range(self.num_batches):
            batch_indices = []

            for class_idx in range(self.num_classes):
                samples = self._get_samples_from_bucket(class_idx, self.n_samples_per_class)
                batch_indices.extend(samples.tolist())

            batch_indices = np.array(batch_indices)
            np.random.shuffle(batch_indices)

            yield batch_indices.tolist()

    def __len__(self) -> int:
        return self.num_batches


class DynamicBalancedBatchSampler(MultilabelBalancedBatchSampler):
    """
    中文：动态平衡批次采样器。训练前期使用均衡策略，后期切换到完整覆盖策略。
    English: Dynamic balanced batch sampler. Uses balanced strategy in early training, then switches to full coverage strategy.
    """

    def __init__(
        self,
        dataset,
        train_indices: List[int],
        batch_size: int,
        num_classes: int = 12,
        balance_ratio: float = 0.3,
        total_epochs: int = 100,
        random_seed: int = 42
    ):
        super().__init__(dataset, train_indices, batch_size, num_classes, random_seed)

        self.balance_ratio = balance_ratio
        self.total_epochs = total_epochs
        self.current_epoch = 0

        self.min_bucket_size = min(len(bucket) for bucket in self.class_buckets)
        self.max_bucket_size = max(len(bucket) for bucket in self.class_buckets)

        self.num_batches_balanced = self.min_bucket_size // self.n_samples_per_class
        self.num_batches_full = self.max_bucket_size // self.n_samples_per_class

        self.use_balanced_mode = True

    def set_epoch(self, epoch: int):
        """
        Set the current epoch to determine sampling strategy.

        Args:
            epoch: Current epoch number (1-indexed)
        """
        self.current_epoch = epoch
        transition_epoch = int(self.total_epochs * self.balance_ratio)
        self.use_balanced_mode = (epoch <= transition_epoch)

        if self.use_balanced_mode:
            self.num_batches = self.num_batches_balanced
        else:
            self.num_batches = self.num_batches_full

    def get_mode_info(self) -> str:
        mode = "BALANCED" if self.use_balanced_mode else "FULL_COVERAGE"
        if self.use_balanced_mode:
            return f"{mode} (rare class: {self.num_batches} batches)"
        else:
            coverage_pct = (self.num_batches_balanced / self.num_batches_full) * 100
            return f"{mode} (all classes: {self.num_batches} batches, {coverage_pct:.1f}% of balanced mode batches)"

    def __iter__(self):
        for batch_idx in range(self.num_batches):
            batch_indices = []

            for class_idx in range(self.num_classes):
                samples = self._get_samples_from_bucket(class_idx, self.n_samples_per_class)
                batch_indices.extend(samples.tolist())

            batch_indices = np.array(batch_indices)
            np.random.shuffle(batch_indices)

            yield batch_indices.tolist()
