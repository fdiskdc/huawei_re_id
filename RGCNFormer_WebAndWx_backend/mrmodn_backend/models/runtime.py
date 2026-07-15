"""
中文：模型加载工具模块。提供从检查点加载 RNA_ClassQuery_Model 的统一入口。
English: Model loading utilities. Provides a single entry-point to load the RNA_ClassQuery_Model from checkpoint.
"""
import torch
from typing import Tuple, Any

from mrmodn_backend.core.config import config
from mrmodn_backend.core.paths import MODEL_CHECKPOINT_PATH
from mrmodn_backend.models.mrmodn import RNA_ClassQuery_Model


def load_model(device: torch.device = None) -> Tuple[RNA_ClassQuery_Model, torch.device, dict]:
    """
    Load the RNA_ClassQuery_Model from checkpoint.

    Args:
        device: Target device. If None, uses config.MODEL_DEVICE.

    Returns:
        (model, device, model_cfg) where model_cfg is the checkpoint's
        'config' dict (may be None for older checkpoints).
    """
    if device is None:
        device_str = config.MODEL_DEVICE
        if device_str == 'cuda' and not torch.cuda.is_available():
            device_str = 'cpu'
        device = torch.device(device_str)

    checkpoint_path = MODEL_CHECKPOINT_PATH
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_cfg = checkpoint.get('config', {})

    model_params = {}
    if model_cfg and 'model' in model_cfg:
        m = model_cfg['model']
        model_params = {
            'cnn_hidden_dim': m.get('cnn_hidden_dim', 64),
            'cnn_kernel_sizes': tuple(m.get('cnn_kernel_sizes', [1, 3, 5, 7])),
            'cnn_dropout': m.get('cnn_dropout', 0.1),
            'gcn_hidden_dim': m.get('gcn_hidden_dim', 128),
            'gcn_out_channels': m.get('gcn_out_channels', 128),
            'gcn_num_layers': m.get('gcn_num_layers', 3),
            'gcn_dropout': m.get('gcn_dropout', 0.3),
            'num_classes': m.get('num_classes', 12),
            'num_attn_heads': m.get('num_attn_heads', 4),
            'attn_dropout': m.get('attn_dropout', 0.1),
            'use_simple_pooling': m.get('use_simple_pooling', False),
            'use_hierarchical': m.get('use_hierarchical', False),
            'use_layer_norm': m.get('use_layer_norm', True),
        }
    else:
        model_params = {
            'cnn_hidden_dim': 64,
            'cnn_kernel_sizes': (1, 3, 5, 7),
            'cnn_dropout': 0.1,
            'gcn_hidden_dim': 128,
            'gcn_out_channels': 128,
            'gcn_num_layers': 3,
            'gcn_dropout': 0.3,
            'num_classes': 12,
            'num_attn_heads': 4,
            'attn_dropout': 0.1,
            'use_simple_pooling': False,
            'use_hierarchical': False,
            'use_layer_norm': True,
        }

    model = RNA_ClassQuery_Model(**model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, device, model_cfg
