"""
中文：RNA_ClassQuery_Model 的 ONNX 导出友好版本。移除了 .item() 调用，简化了数据依赖控制流，假设一致的输入张量形状。
English: ONNX export-friendly version of RNA_ClassQuery_Model. Removes .item() calls, simplifies data-dependent control flow, and assumes consistent input tensor shapes.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from typing import Optional, Tuple

from mrmodn_backend.core.constants import GROUP_TO_CLASS_INDICES


class ParallelCNNBlock(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_dim: int = 64, kernel_sizes: Tuple[int, ...] = (1, 3, 5, 7), use_layer_norm: bool = True, dropout: float = 0.1):
        super().__init__()
        self.out_channels = len(kernel_sizes) * hidden_dim
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=k, padding='same', bias=True) for k in kernel_sizes
        ])
        self.norm = nn.LayerNorm(normalized_shape=(self.out_channels, 1001)) if use_layer_norm else nn.BatchNorm1d(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        branch_outputs = [conv(x) for conv in self.conv_branches]
        concatenated = torch.cat(branch_outputs, dim=1)
        normalized = self.norm(concatenated)
        features = self.dropout(self.activation(normalized))
        features = features.transpose(1, 2)
        features = features.reshape(-1, self.out_channels)
        return features


class GCNBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 128, out_channels: int = 128, num_layers: int = 3, dropout: float = 0.3, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.input_proj = nn.Linear(in_channels, hidden_dim) if in_channels != hidden_dim else None
        self.gcn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            self.gcn_layers.append(GCNConv(in_dim, out_channels if i == num_layers - 1 else hidden_dim))
            self.norms.append(nn.LayerNorm(out_channels if i == num_layers - 1 else hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.input_proj is not None:
            x = self.input_proj(x)

        residual = x
        for idx, (gcn_layer, norm) in enumerate(zip(self.gcn_layers, self.norms)):
            x = gcn_layer(x, edge_index)
            x = norm(x)
            if idx < self.num_layers - 1:
                x = self.activation(x)
                x = self.dropout(x)
                if self.use_residual:
                    x = x + residual
                    residual = x
        return x


class HierarchicalClassQueryHeadPooling(nn.Module):
    def __init__(self, hidden_dim, num_classes, group_to_class_indices, dropout=0.1, use_layer_norm=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.group_to_class_indices = group_to_class_indices
        self.num_groups = 4
        self.group_names = ['A', 'C', 'G', 'U']
        self.group_queries = nn.Parameter(torch.randn(self.num_groups, hidden_dim))
        self.group_projectors = nn.ModuleList()
        for g_idx in range(self.num_groups):
            group_name = self.group_names[g_idx]
            num_subclasses = len(self.group_to_class_indices.get(group_name, []))
            projector = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, num_subclasses * hidden_dim)
            )
            self.group_projectors.append(projector)

        self.output_proj_12 = nn.Sequential(
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.output_proj_4 = nn.Sequential(
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.attention_scale = hidden_dim ** 0.5

    def _derive_class_queries(self):
        all_sub_queries = []
        for g_idx in range(self.num_groups):
             group_name = self.group_names[g_idx]
             if group_name not in self.group_to_class_indices: continue
             g_query = self.group_queries[g_idx].unsqueeze(0)
             sub_flat = self.group_projectors[g_idx](g_query)
             num_subs = len(self.group_to_class_indices[group_name])
             sub_queries = sub_flat.view(num_subs, self.hidden_dim)
             all_sub_queries.append(sub_queries)

        concatenated_queries = torch.cat(all_sub_queries, dim=0)

        final_queries = torch.zeros(self.num_classes, self.hidden_dim, device=concatenated_queries.device, dtype=concatenated_queries.dtype)

        current_pos = 0
        all_indices = []
        for group in ['A', 'C', 'G', 'U']:
            indices = self.group_to_class_indices.get(group, [])
            for i, class_idx in enumerate(indices):
                final_queries[class_idx] = concatenated_queries[current_pos + i]
            current_pos += len(indices)

        return final_queries

    def forward(self, node_features: torch.Tensor, batch: torch.Tensor):
        class_queries = self._derive_class_queries()
        group_queries = self.group_queries

        scores_12 = torch.matmul(class_queries, node_features.t()) / self.attention_scale
        attn_12 = torch.softmax(scores_12, dim=1)
        weighted_nodes_12 = torch.matmul(attn_12, node_features)
        logits_12 = self.output_proj_12(weighted_nodes_12).squeeze(-1)

        scores_4 = torch.matmul(group_queries, node_features.t()) / self.attention_scale
        attn_4 = torch.softmax(scores_4, dim=1)
        weighted_nodes_4 = torch.matmul(attn_4, node_features)
        logits_4 = self.output_proj_4(weighted_nodes_4).squeeze(-1)

        return logits_12.unsqueeze(0), logits_4.unsqueeze(0), attn_12.unsqueeze(0)


class RNA_ClassQuery_Model(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cnn_block = ParallelCNNBlock(
            in_channels=4, hidden_dim=kwargs['cnn_hidden_dim'], kernel_sizes=tuple(kwargs['cnn_kernel_sizes']),
            use_layer_norm=kwargs['use_layer_norm'], dropout=kwargs['cnn_dropout']
        )
        self.gcn_block = GCNBlock(
            in_channels=len(kwargs['cnn_kernel_sizes']) * kwargs['cnn_hidden_dim'],
            hidden_dim=kwargs['gcn_hidden_dim'], out_channels=kwargs['gcn_out_channels'],
            num_layers=kwargs['gcn_num_layers'], dropout=kwargs['gcn_dropout'], use_residual=True
        )
        self.class_query_head = HierarchicalClassQueryHeadPooling(
            hidden_dim=kwargs['gcn_out_channels'], num_classes=kwargs['num_classes'],
            group_to_class_indices=GROUP_TO_CLASS_INDICES, dropout=kwargs['attn_dropout'],
            use_layer_norm=kwargs.get('use_layer_norm', True)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None):
        if batch is None:
            batch_size = x.size(0)
            seq_len = x.size(2)
            batch = torch.arange(batch_size, device=x.device).repeat_interleave(seq_len)

        node_features = self.cnn_block(x, batch)
        node_features = self.gcn_block(node_features, edge_index)

        node_features_flat = node_features.view(-1, self.gcn_block.out_channels)
        batch_flat = torch.zeros(node_features_flat.size(0), dtype=torch.long, device=x.device)

        return self.class_query_head(node_features_flat, batch_flat)
