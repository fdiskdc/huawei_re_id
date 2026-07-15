"""
Backward-compatible shim for common.

All code has been moved to:
- mrmodn_backend.core.constants   (MOD_NAMES, GROUP_TO_CLASS_INDICES, etc.)
- mrmodn_backend.core.config      (Config, get_logger)
- mrmodn_backend.training.sampling (MultilabelBalancedBatchSampler, DynamicBalancedBatchSampler)
- mrmodn_backend.training.engine   (train_epoch, test_epoch, compute_attention_supervision_loss)
- mrmodn_backend.training.metrics  (calculate_topk_recall, print_topk_table, etc.)

This file re-exports the public API for backward compatibility.
"""
from mrmodn_backend.core.constants import (
    MOD_NAMES,
    NUCLEOTIDE_GROUP_NAMES,
    INDEX_TO_NUCLEOTIDE,
    NUCLEOTIDE_GROUPS,
    GROUP_TO_INDEX,
    INDEX_TO_GROUP,
    GROUP_SIZES,
    GROUP_TO_CLASS_INDICES,
    PLANT_VALID_CLASS_INDICES,
    LABEL_MAPPING,
)
from mrmodn_backend.training.sampling import (
    MultilabelBalancedBatchSampler,
    DynamicBalancedBatchSampler,
)
from mrmodn_backend.training.engine import (
    train_epoch,
    test_epoch,
    compute_attention_supervision_loss,
)
from mrmodn_backend.training.metrics import (
    calculate_topk_recall,
    print_topk_table,
    calculate_comprehensive_localization_metrics,
    print_comprehensive_table,
)

import os
import json
import hashlib
import pickle
import logging
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader


def get_center_nucleotide(sequence: str) -> str:
    if len(sequence) >= 501:
        return sequence[500].upper()
    return 'A'


def load_config(config_path: str = 'model.json') -> Tuple:
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = config_dict.get('experiment_name', 'rna_classification')
    exp_folder = f"{exp_name}_{timestamp}"

    base_log_dir = config_dict.get('paths', {}).get('log_dir', './logs')
    log_dir = os.path.join(base_log_dir, exp_folder)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    config_save_path = os.path.join(log_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    class Config:
        pass

    data_cfg = config_dict.get('data', {})

    if data_cfg:
        Config.data_dir = data_cfg.get('human_data_dir', './human3')
        Config.human_data_dir = data_cfg.get('human_data_dir', './human3')
        Config.plant_data_dir = data_cfg.get('plant_data_dir', './plant')
        Config.cache_dir = data_cfg.get('cache_dir', './cache')
    else:
        Config.data_dir = config_dict.get('data_dir', './human3')
        Config.human_data_dir = Config.data_dir
        Config.plant_data_dir = './plant'
        Config.cache_dir = './cache'

    Config.checkpoint_dir = checkpoint_dir
    Config.log_dir = log_dir
    Config.experiment_name = exp_name
    Config.timestamp = timestamp

    class DataConfig:
        pass
    DataConfig.human_data_dir = Config.human_data_dir
    DataConfig.plant_data_dir = Config.plant_data_dir
    DataConfig.cache_dir = Config.cache_dir
    Config.data = DataConfig

    model_cfg = config_dict.get('model', {})
    Config.cnn_hidden_dim = model_cfg.get('cnn_hidden_dim', 64)
    Config.cnn_kernel_sizes = tuple(model_cfg.get('cnn_kernel_sizes', [1, 3, 5, 7]))
    Config.cnn_dropout = model_cfg.get('cnn_dropout', 0.1)
    Config.gcn_hidden_dim = model_cfg.get('gcn_hidden_dim', 128)
    Config.gcn_out_channels = model_cfg.get('gcn_out_channels', 128)
    Config.gcn_num_layers = model_cfg.get('gcn_num_layers', 3)
    Config.gcn_dropout = model_cfg.get('gcn_dropout', 0.3)
    Config.num_classes = model_cfg.get('num_classes', 12)
    Config.num_attn_heads = model_cfg.get('num_attn_heads', 4)
    Config.attn_dropout = model_cfg.get('attn_dropout', 0.1)
    Config.use_simple_pooling = model_cfg.get('use_simple_pooling', False)
    Config.use_hierarchical = model_cfg.get('use_hierarchical', False)
    Config.use_layer_norm = model_cfg.get('use_layer_norm', True)

    train_cfg = config_dict.get('training', {})
    Config.batch_size = train_cfg.get('batch_size', 32)
    Config.num_epochs = train_cfg.get('num_epochs', 100)
    Config.learning_rate = train_cfg.get('learning_rate', 1e-3)
    Config.weight_decay = train_cfg.get('weight_decay', 1e-4)
    Config.warmup_epochs = train_cfg.get('warmup_epochs', 5)
    Config.train_ratio = train_cfg.get('train_ratio', 0.7)
    Config.random_seed = train_cfg.get('random_seed', 42)
    Config.eval_threshold = train_cfg.get('eval_threshold', 0.5)
    Config.test_interval = train_cfg.get('test_interval', 1)
    Config.save_every_epoch = train_cfg.get('save_every_epoch', True)
    Config.save_best_only = train_cfg.get('save_best_only', False)

    Config.use_dynamic_sampler = train_cfg.get('use_dynamic_sampler', True)
    Config.balance_ratio = train_cfg.get('balance_ratio', 0.3)

    Config.use_amp = train_cfg.get('use_amp', True)
    Config.use_attention_supervision = train_cfg.get('use_attention_supervision', False)
    Config.attention_lambda = train_cfg.get('attention_lambda', 1.0)

    few_shot_cfg = config_dict.get('few_shot', {})
    Config.few_shot = few_shot_cfg

    Config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return Config, config_dict


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, metrics: Dict, filepath: str,
                   logger: Optional[logging.Logger] = None,
                   config_dict: Optional[Dict] = None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config_dict
    }
    torch.save(checkpoint, filepath)

    msg = f"Checkpoint saved to {filepath}"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def load_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
                   filepath: str, device: torch.device) -> Dict:
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint['metrics']
    }


def multi_label_disjoint_split(dataset, train_ratio: float = 0.7,
                               random_seed: int = 42, logger: Optional[logging.Logger] = None,
                               use_cache: bool = True, cache_dir: str = './cache') -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(random_seed)
    num_samples = len(dataset)
    num_classes = 12

    os.makedirs(cache_dir, exist_ok=True)

    cache_key = hashlib.md5(f"{dataset.data_dir}_{num_samples}_{train_ratio}_{random_seed}".encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f'labels_cache_{cache_key}.pkl')

    all_labels = None
    if use_cache:
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    if cache_data.get('num_samples') == num_samples:
                        all_labels = cache_data['labels']
                        if logger:
                            logger.info(f"Loaded cached labels from {cache_file}")
                        else:
                            print(f"Loaded cached labels from {cache_file}")
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to load cache: {e}")
                else:
                    print(f"Failed to load cache: {e}")

    if all_labels is None:
        if logger:
            logger.info(f"Collecting labels from {num_samples} samples...")
        else:
            print(f"Collecting labels from {num_samples} samples...")

        all_labels = dataset.y_12class[:num_samples].copy()

        if logger:
            logger.info(f"Loaded {len(all_labels)} labels directly from dataset array")
        else:
            print(f"Loaded {len(all_labels)} labels directly from dataset array")

        if use_cache:
            try:
                cache_data = {
                    'labels': all_labels,
                    'num_samples': num_samples,
                    'train_ratio': train_ratio,
                    'random_seed': random_seed
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                if logger:
                    logger.info(f"Saved cached labels to {cache_file}")
                else:
                    print(f"Saved cached labels to {cache_file}")
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to save cache: {e}")
                else:
                    print(f"Failed to save cache: {e}")

    log_msg = f"\n{'='*60}\nMulti-label Disjoint Split\n{'='*60}\n"
    log_msg += f"Total samples: {num_samples}\n"
    log_msg += f"Number of classes: {num_classes}\n"
    log_msg += f"Train ratio: {train_ratio}\n"

    class_sample_indices = {}
    for class_idx in range(num_classes):
        indices = np.where(all_labels[:, class_idx] == 1)[0]
        class_sample_indices[class_idx] = indices.tolist()
        mod_name = MOD_NAMES.get(class_idx, f'Class{class_idx}')
        log_msg += f"Class {class_idx} ({mod_name}): {len(indices)} positive samples\n"

    pre_train_indices_per_class = []
    pre_test_indices_per_class = []

    for class_idx, indices in class_sample_indices.items():
        if len(indices) == 0:
            pre_train_indices_per_class.append(set())
            pre_test_indices_per_class.append(set())
            continue

        shuffled = rng.permutation(indices).tolist()
        split_point = int(len(shuffled) * train_ratio)

        pre_train = set(shuffled[:split_point])
        pre_test = set(shuffled[split_point:])

        pre_train_indices_per_class.append(pre_train)
        pre_test_indices_per_class.append(pre_test)

        mod_name = MOD_NAMES.get(class_idx, f'Class{class_idx}')
        log_msg += f"Class {class_idx} ({mod_name}) split: {len(pre_train)} train, {len(pre_test)} pre-test\n"

    train_indices_union = set()
    test_indices_union = set()

    for pre_train in pre_train_indices_per_class:
        train_indices_union.update(pre_train)

    for pre_test in pre_test_indices_per_class:
        test_indices_union.update(pre_test)

    log_msg += f"\nUnion sizes:\n"
    log_msg += f"  Train union: {len(train_indices_union)}\n"
    log_msg += f"  Test union: {len(test_indices_union)}\n"

    conflicts = train_indices_union & test_indices_union
    if conflicts:
        log_msg += f"  Conflicts (samples in both unions): {len(conflicts)}\n"
        test_indices_union -= conflicts
        train_indices_union.update(conflicts)

    assert len(train_indices_union & test_indices_union) == 0, "Train and test sets are not disjoint!"

    train_indices = sorted(list(train_indices_union))
    test_indices = sorted(list(test_indices_union))

    log_msg += f"\nFinal split sizes:\n"
    log_msg += f"  Train: {len(train_indices)} samples\n"
    log_msg += f"  Test: {len(test_indices)} samples\n"
    log_msg += f"  Total: {len(train_indices) + len(test_indices)} samples\n"
    log_msg += f"{'='*60}\n"

    if logger:
        logger.info(log_msg)
    else:
        print(log_msg)

    return train_indices, test_indices


def get_smoothed_pos_weights(dataset, train_indices: List[int], num_classes: int = 12,
                            epsilon: float = 1e-6, logger: Optional[logging.Logger] = None,
                            use_balanced_weights: bool = False) -> torch.Tensor:
    if use_balanced_weights:
        pos_weights = np.ones(num_classes, dtype=np.float32)

        log_msg = f"\n{'='*60}\nClass Weights: BALANCED (equal weights for all classes)\n{'='*60}\n"
        log_msg += f"All classes: weight = 1.0\n"
        log_msg += f"{'='*60}\n"

        if logger:
            logger.info(log_msg)
        else:
            print(log_msg)

        return torch.FloatTensor(pos_weights)

    if logger:
        logger.info(f"Counting positive samples from {len(train_indices)} training samples...")
    else:
        print(f"Counting positive samples from {len(train_indices)} training samples...")

    train_labels = dataset.y_12class[train_indices]
    pos_counts = train_labels.sum(axis=0).astype(np.float32)

    raw_weights = 1.0 / np.sqrt(pos_counts + epsilon)

    min_weight = np.min(raw_weights)
    pos_weights = raw_weights / min_weight

    log_msg = f"\n{'='*60}\nSmoothed Class Weights (Square Root Reciprocal)\n{'='*60}\n"
    log_msg += f"{'Class':<12} {'Pos Count':<12} {'Weight':<10}\n"
    log_msg += f"{'-'*40}\n"
    for i in range(num_classes):
        mod_name = MOD_NAMES.get(i, f'Class{i}')
        log_msg += f"{i:<3} ({mod_name:<6}) {int(pos_counts[i]):<12} {pos_weights[i]:<10.4f}\n"
    log_msg += f"{'='*60}\n"

    if logger:
        logger.info(log_msg)
    else:
        print(log_msg)

    return torch.FloatTensor(pos_weights)


__all__ = [
    'MOD_NAMES',
    'NUCLEOTIDE_GROUP_NAMES',
    'INDEX_TO_NUCLEOTIDE',
    'NUCLEOTIDE_GROUPS',
    'GROUP_TO_INDEX',
    'INDEX_TO_GROUP',
    'GROUP_SIZES',
    'GROUP_TO_CLASS_INDICES',
    'PLANT_VALID_CLASS_INDICES',
    'LABEL_MAPPING',
    'get_center_nucleotide',
    'load_config',
    'save_checkpoint',
    'load_checkpoint',
    'multi_label_disjoint_split',
    'MultilabelBalancedBatchSampler',
    'DynamicBalancedBatchSampler',
    'get_smoothed_pos_weights',
    'train_epoch',
    'test_epoch',
    'compute_attention_supervision_loss',
    'calculate_topk_recall',
    'print_topk_table',
    'calculate_comprehensive_localization_metrics',
    'print_comprehensive_table',
]
