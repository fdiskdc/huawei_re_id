"""
中文：预测预处理工具。提供 one-hot 编码、序列填充/截断和 GCN 图数据构建等辅助函数。
English: Prediction preprocessing utilities. Provides helpers for one-hot encoding, sequence padding/truncation, and GCN graph data construction.
"""
import numpy as np


def one_hot_encode_sequence(sequence: str) -> np.ndarray:
    """
    Convert RNA sequence string to one-hot encoding.

    Args:
        sequence: RNA sequence string (A, C, G, U)

    Returns:
        One-hot encoded array of shape (len(sequence), 4)
    """
    one_hot_mapping = {
        'A': [1., 0., 0., 0.],
        'C': [0., 1., 0., 0.],
        'G': [0., 0., 1., 0.],
        'U': [0., 0., 0., 1.],
        'T': [0., 0., 0., 1.],  # Treat T as U
        'N': [0., 0., 0., 0.]
    }

    one_hot = np.zeros((len(sequence), 4), dtype=np.float32)
    for i, nucleotide in enumerate(sequence.upper()):
        if nucleotide in one_hot_mapping:
            one_hot[i] = one_hot_mapping[nucleotide]
        else:
            one_hot[i] = [0., 0., 0., 0.]  # Unknown nucleotide

    return one_hot


def pad_or_truncate_sequence(sequence: str, target_length: int = 1001):
    """
    Pad a short sequence with N's or truncate a long sequence to target_length.
    Padding/truncation is centered (equal left and right when possible).

    Args:
        sequence: Original RNA sequence string
        target_length: Desired sequence length (default 1001)

    Returns:
        Tuple of (processed_sequence, left_padding, left_trimming) where:
        - processed_sequence: the padded/truncated sequence
        - left_padding: number of N chars added on the left (0 if truncated)
        - left_trimming: number of chars removed from the left (0 if padded)
    """
    seq_len = len(sequence)
    left_padding = 0
    left_trimming = 0

    if seq_len == target_length:
        return sequence, left_padding, left_trimming

    if seq_len < target_length:
        padding_needed = target_length - seq_len
        left_pad = padding_needed // 2
        right_pad = padding_needed - left_pad
        left_padding = left_pad
        sequence = 'N' * left_pad + sequence + 'N' * right_pad
    else:
        excess = seq_len - target_length
        left_trim = excess // 2
        right_trim = excess - left_trim
        left_trimming = left_trim
        sequence = sequence[left_trim:seq_len - right_trim]

    return sequence, left_padding, left_trimming


def build_gcn_graph_data(edge_index, original_sequence, left_padding, left_trimming):
    """
    Build GCN graph data (nodes and edges) for visualization from an edge_index tensor.

    Args:
        edge_index: torch.Tensor of shape [2, E] with graph edges
        original_sequence: The original (un-padded) RNA sequence string
        left_padding: Number of N chars added on the left
        left_trimming: Number of chars removed from the left

    Returns:
        Tuple of (nodes, valid_edges) lists for JSON serialization
    """
    edge_index_np = edge_index.cpu().numpy()
    edges = []

    for i in range(int(edge_index_np.shape[1])):
        source = int(edge_index_np[0, i])
        target = int(edge_index_np[1, i])

        if not (0 <= source < len(original_sequence) + left_padding and 0 <= target < len(original_sequence) + left_padding):
            continue

        orig_source = source - left_padding + left_trimming
        orig_target = target - left_padding + left_trimming

        if not (0 <= orig_source < len(original_sequence) and 0 <= orig_target < len(original_sequence)):
            continue

        if source >= target:
            continue

        nuc_source = original_sequence[orig_source]
        nuc_target = original_sequence[orig_target]
        edges.append({
            "source": f"{nuc_source}{orig_source}",
            "target": f"{nuc_target}{orig_target}"
        })

    nodes = []
    for i in range(len(original_sequence)):
        nuc = original_sequence[i]
        nodes.append({
            "id": f"{nuc}{i}",
            "label": f"位置{i}: {nuc}",
            "data": {
                "index": i,
                "type": nuc,
                "name": f"{'腺嘌呤' if nuc == 'A' else '胞嘧啶' if nuc == 'C' else '鸟嘌呤' if nuc == 'G' else '尿嘧啶'}"
            }
        })

    valid_node_ids = {node["id"] for node in nodes}
    valid_edges = [
        edge for edge in edges
        if edge["source"] in valid_node_ids and edge["target"] in valid_node_ids
    ]

    return nodes, valid_edges
