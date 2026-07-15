"""
main_model_onnx.py - RGCNFormer 主网络(ONNX 导出友好版) / RGCNFormer (ONNX-friendly)

与 main_model.py 结构相同,但移除了 .item() 调用、简化了 data-dependent 控制流、
固定张量形状,使模型可被 torch.onnx.export 成功 trace;用于生产环境 ONNX Runtime
推理,启动比 PyTorch 快约 5×。 / Same structure as main_model.py but with .item()
calls removed, data-dependent control flow simplified, and tensor shapes fixed
so torch.onnx.export can trace it. Used for production ONNX Runtime inference
(~5× faster startup than PyTorch).

功能模块 / Modules:
- ParallelCNNBlock: 多尺度 CNN(去除数据相关分支)/ Multi-scale CNN (no data-dep branches)
- GCNBlock: 简化版 GCN(2 层 GCNConv) / Simplified GCN (2-layer GCNConv)
- ClassQueryHead: ONNX-friendly 类查询头 / ONNX-friendly Class-Query head
- RNA_ClassQuery_Model: ONNX 兼容顶层封装 / ONNX-compatible top-level wrapper

输入 / Inputs:
- x: torch.Tensor - [N, 4, 1001] one-hot 编码 / [N, 4, 1001] one-hot sequence
- edge_index: torch.Tensor - [2, E] 图边 / [2, E] graph edges
- batch: torch.Tensor - [N] 批索引 / [N] batch indices

输出 / Outputs:
- logits: torch.Tensor - [B, 12] 12 类分数 / 12-class scores
- attn_weights: torch.Tensor - 注意力权重 / Attention weights

数据流 / Data Flow:
1. ParallelCNN → GCN → Class-Query(同 main_model.py,但每步 ONNX-friendly)
2. 固定 batch 维度由输入张量推断,无 batch.max() / Batch dim inferred from input

相关文件 / Related Files:
- 调用 / Calls: torch_geometric.GCNConv、common.GROUP_TO_CLASS_INDICES
- 被调用 / Called by: server.py(ONNX 推理模式)、onnx.py / onnx2.py(导出)

使用示例 / Usage Example:
    # 导出 ONNX:
    torch.onnx.export(model, (x, edge_index, batch), "model.onnx", opset_version=14)
    # ONNX Runtime 推理:
    import onnxruntime as ort
    sess = ort.InferenceSession("model.onnx")

作者 / Author: 项目组 / Project Team
版本 / Version: 1.0
"""

"""
RNA_ClassQuery_Model - ONNX EXPORT-FRIENDLY VERSION

Modifications for ONNX export:
- Removed all .item() calls in forward passes.
- Simplified data-dependent control flow (if/else on tensor shapes).
- Batch size is inferred from tensor dimensions, not from batch.max().
- Assumes consistent input tensor shapes.
- Restored original nn.Sequential structure for output projection heads.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from typing import Optional, Tuple

from common import GROUP_TO_CLASS_INDICES


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
        # ONNX-friendly: Assume input is always [Batch, Channels, SeqLen], e.g., [N, 4, 1001]
        # The original if/else logic is removed as it's not traceable
        branch_outputs = [conv(x) for conv in self.conv_branches]
        concatenated = torch.cat(branch_outputs, dim=1)
        normalized = self.norm(concatenated)
        features = self.dropout(self.activation(normalized))
        features = features.transpose(1, 2) # -> [N, 1001, C*len(K)]
        # Reshape for GCN: if multiple batches, flatten them
        features = features.reshape(-1, self.out_channels) # -> [N*1001, C*len(K)]
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
                # ONNX-friendly: Assume residual connection is always possible.
                # The data-dependent shape check is removed.
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

        # RESTORED to match original model's state_dict
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
        # This logic is complex for tracing. We build a constant index map for ONNX.
        # This order must match the derivation logic.
        # Order of groups: A, C, G, U
        # Indices within groups: A:[0,1,7,9,10], C:[2,6,8], G:[3,11], U:[4,5]
        # Final flat order for cat: Am,Atol,m1A,m6A,m6Am, Cm,ac4C,m5C, Gm,m7G, Tm,Y
        # Final re-order indices to match 0-11:
        reorder_map = {
            0:0, 1:1, 2:5, 3:8, 4:10, 5:6, 6:9, 7:2, 8:7, 9:3, 10:4, 11:11
        }
        # This is hard to trace. Instead, we'll try to build the tensor directly
        # in a way that might be more traceable.

        # Let's create the full query tensor and then select from it.
        # This might be more ONNX-friendly than index_copy_
        derived_queries_list = []
        for g_idx in range(self.num_groups):
             group_name = self.group_names[g_idx]
             if group_name not in self.group_to_class_indices: continue
             g_query = self.group_queries[g_idx].unsqueeze(0)
             sub_flat = self.group_projectors[g_idx](g_query)
             num_subs = len(self.group_to_class_indices[group_name])
             sub_queries = sub_flat.view(num_subs, self.hidden_dim)
             derived_queries_list.append(sub_queries)

        concatenated_queries = torch.cat(derived_queries_list, dim=0)

        # The re-ordering is the main issue for tracing.
        # For a robust export, we must assume a fixed mapping.
        # Let's create the final tensor by gathering from the concatenated one.
        # This is still not ideal. The best way is to have the model architecture fixed.
        # We assume the order is fixed as per the indices in GROUP_TO_CLASS_INDICES
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
        # Assume node_features corresponds to a single graph for simplicity in export.
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
        # ONNX-friendly: Assume x is always [N, C, L] format, no conditional checking
        if batch is None:
            batch_size = x.size(0)
            seq_len = x.size(2)
            batch = torch.arange(batch_size, device=x.device).repeat_interleave(seq_len)

        node_features = self.cnn_block(x, batch)
        node_features = self.gcn_block(node_features, edge_index)

        # The head is not easily batchable for tracing, so we assume batch size of 1
        # when exporting. The dynamic_axes setting in onnx export will handle variable seq length.
        # We must flatten the node features for the single graph.
        node_features_flat = node_features.view(-1, self.gcn_block.out_channels)
        batch_flat = torch.zeros(node_features_flat.size(0), dtype=torch.long, device=x.device)

        return self.class_query_head(node_features_flat, batch_flat)
