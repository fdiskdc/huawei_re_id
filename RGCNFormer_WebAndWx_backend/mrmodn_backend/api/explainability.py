"""
中文：模型可解释性 API 端点。提供 Integrated Gradients 归因分析、GCN 聚合可视化、模型架构和计算图查询。
English: Model explainability API endpoints. Provides Integrated Gradients attribution, GCN aggregation visualization, model architecture, and computation graph queries.
"""
import json
import numpy as np
import torch
from torch_geometric.data import Batch
from captum.attr import IntegratedGradients
from flask import Blueprint, jsonify, request, current_app

from mrmodn_backend.core.config import get_logger
from mrmodn_backend.core.paths import get_model_graph_json_path
from mrmodn_backend.services.prediction import one_hot_encode_sequence, pad_or_truncate_sequence
from mrmodn_backend.services.model_inspection import extract_module_info
from mrmodn_backend.services.rna_structure import run_linearfold, build_edge_index_from_structure

explainability_bp = Blueprint('explainability', __name__)
logger = get_logger('api.explainability')


@explainability_bp.route('/mrmodn/api/v1/model-architecture', methods=['GET'])
def get_model_architecture():
    """
    中文：获取模型层级架构。返回 RNA_ClassQuery_Model 的 JSON 结构表示。
    English: Get model hierarchical architecture. Returns a JSON representation of the RNA_ClassQuery_Model structure.
    """
    try:
        logger.info("Extracting model architecture...")

        model = current_app.config['MODEL']

        # Extract model structure
        model_info = extract_module_info(model, "RNA_ClassQuery_Model")

        logger.info("Model architecture extracted successfully")

        return jsonify(model_info), 200

    except Exception as e:
        import traceback
        error_msg = f"Failed to extract model architecture: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500


@explainability_bp.route('/mrmodn/api/v1/model-graph', methods=['GET'])
def get_model_graph():
    """
    中文：获取 ONNX 模型计算图。从 JSON 文件加载模型图的节点和边数据。
    English: Get the ONNX model computation graph. Loads model graph nodes and edges from the JSON file.
    """
    try:
        logger.info("Loading model graph data...")

        # Read model graph from JSON file
        model_graph_path = get_model_graph_json_path()

        with open(model_graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        logger.info(f"Model graph loaded successfully: {len(graph_data.get('nodes', []))} nodes, {len(graph_data.get('edges', []))} edges")

        return jsonify(graph_data), 200

    except FileNotFoundError:
        logger.error(f"Model graph file not found: {model_graph_path}")
        return jsonify({
            "error": "Model graph file not found",
            "detail": f"The file {model_graph_path} does not exist. Please generate the model graph first.",
            "type": "FileNotFoundError"
        }), 404
    except Exception as e:
        import traceback
        error_msg = f"Failed to load model graph: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500


@explainability_bp.route('/mrmodn/api/v1/integrated-gradients', methods=['POST'])
def integrated_gradients():
    """
    中文：计算 RNA 序列预测的 Integrated Gradients 归因分数。返回带归因分数的节点和边。
    English: Compute Integrated Gradients attributions for RNA sequence prediction. Returns nodes and edges with attribution scores.
    """
    # Get data from request
    data = request.get_json()
    original_sequence = data.get('rnaSequence', '')
    target_class_id = data.get('targetClassId')

    # Validate inputs
    if not original_sequence:
        return jsonify({"error": "No sequence provided"}), 400

    if target_class_id is None or not (0 <= target_class_id < 12):
        return jsonify({"error": "Invalid targetClassId. Must be between 0 and 11"}), 400

    logger.info(f"Integrated Gradients: sequence length={len(original_sequence)}, target_class_id={target_class_id}")

    # Store original sequence for response
    sequence = original_sequence

    # For shorter sequences, pad to 1001; for longer sequences, truncate
    TARGET_LENGTH = 1001
    seq_len = len(sequence)

    # Track padding/trimming for index remapping
    left_padding = 0
    left_trimming = 0

    if seq_len != TARGET_LENGTH:
        if seq_len < TARGET_LENGTH:
            padding_needed = TARGET_LENGTH - seq_len
            left_pad = padding_needed // 2
            right_pad = padding_needed - left_pad
            left_padding = left_pad
            sequence = 'N' * left_pad + sequence + 'N' * right_pad
        else:
            excess = seq_len - TARGET_LENGTH
            left_trim = excess // 2
            right_trim = excess - left_trim
            left_trimming = left_trim
            sequence = sequence[left_trim:seq_len - right_trim]

    try:
        model = current_app.config['MODEL']
        device = current_app.config['MODEL_DEVICE']
        model_cfg = current_app.config['MODEL_CFG']

        # Step 1: Call LinearFold to get secondary structure
        structures = run_linearfold([sequence])
        structure = structures[0]

        # Step 2: Build edge index from structure
        edge_index = build_edge_index_from_structure(sequence, structure)

        # Step 3: Prepare model input
        x = one_hot_encode_sequence(sequence)
        x = torch.FloatTensor(x)  # Shape: [1001, 4]

        # Create batch tensor (single sample)
        batch = torch.zeros(len(sequence), dtype=torch.long)

        # Step 4: Create Batch object
        data_batch = Batch(x=x, edge_index=edge_index, batch=batch)
        data_batch = data_batch.to(device)

        # Step 5: Compute Integrated Gradients
        # Check if model is hierarchical
        if model_cfg.get('model', {}).get('use_hierarchical', False):
            # For hierarchical model, wrap the model to extract 12-class logits
            def forward_func(x, edge_index, batch):
                output = model(x, edge_index, batch)
                # output is a tuple: (logits_12class, logits_4class, attn_weights)
                logits_12class = output[0]  # Extract 12-class logits
                return logits_12class
        else:
            # For non-hierarchical model, use model directly
            def forward_func(x, edge_index, batch):
                output = model(x, edge_index, batch)
                return output

        # Instantiate IntegratedGradients
        ig = IntegratedGradients(forward_func)

        # Compute attributions
        with torch.enable_grad():
            # Baseline: zero tensor of same shape
            baseline = torch.zeros_like(x)

            # Compute attributions
            attributions = ig.attribute(
                x.unsqueeze(0),  # Add batch dimension
                baselines=baseline.unsqueeze(0),
                target=target_class_id,
                additional_forward_args=(edge_index, batch),
                internal_batch_size=1
            )

        # Step 6: Process attributions
        attributions = attributions.squeeze(0)  # Remove batch dimension [1001, 4]

        # Sum attributions across the one-hot encoding dimension to get per-nucleotide score
        node_attributions = attributions.sum(dim=1).cpu().numpy()  # [1001]

        # Step 7: Build GCN graph data with attribution scores
        edge_index_np = edge_index.cpu().numpy()
        edges = []

        # Calculate the valid range in model coordinates
        valid_start = left_padding
        valid_end = left_padding + len(original_sequence)

        # Process all edges
        for i in range(int(edge_index_np.shape[1])):
            source = int(edge_index_np[0, i])
            target = int(edge_index_np[1, i])

            # Only process edges within valid range
            if not (0 <= source < len(sequence) and 0 <= target < len(sequence)):
                continue

            # Map model indices to original indices
            orig_source = source - left_padding + left_trimming
            orig_target = target - left_padding + left_trimming

            # Only include edges within original sequence bounds
            if not (0 <= orig_source < len(original_sequence) and 0 <= orig_target < len(original_sequence)):
                continue

            # Only keep one direction (source < target) to avoid duplicates
            if source >= target:
                continue

            nuc_source = original_sequence[orig_source]
            nuc_target = original_sequence[orig_target]
            edges.append({
                "source": f"{nuc_source}{orig_source}",
                "target": f"{nuc_target}{orig_target}"
            })

        # Create nodes with attribution scores
        nodes = []
        for i in range(len(original_sequence)):
            nuc = original_sequence[i]
            # Map original index to model index
            model_index = i + left_padding - left_trimming
            attribution_score = float(node_attributions[model_index]) if 0 <= model_index < len(node_attributions) else 0.0

            nodes.append({
                "id": f"{nuc}{i}",
                "label": f"位置{i}: {nuc}",
                "data": {
                    "index": i,
                    "type": nuc,
                    "name": f"{'腺嘌呤' if nuc == 'A' else '胞嘧啶' if nuc == 'C' else '鸟嘌呤' if nuc == 'G' else '尿嘧啶'}",
                    "attributionScore": attribution_score
                }
            })

        # Create a set of valid node IDs for filtering edges
        valid_node_ids = {node["id"] for node in nodes}

        # Filter edges to only include those that reference valid nodes
        valid_edges = [
            edge for edge in edges
            if edge["source"] in valid_node_ids and edge["target"] in valid_node_ids
        ]

        logger.info(f"Integrated Gradients: 节点数={len(nodes)}, 有效边数={len(valid_edges)}")

        # Return response
        return jsonify({
            "nodes": nodes,
            "edges": valid_edges,
            "targetClassId": target_class_id
        }), 200

    except Exception as e:
        import traceback
        error_msg = f"Integrated Gradients error: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500


@explainability_bp.route('/mrmodn/api/v1/visualize-gcn-aggregation', methods=['POST'])
def visualize_gcn_aggregation():
    """
    中文：可视化 GCN 消息传递过程。返回指定目标节点在各 GCN 层的聚合详情和邻居消息强度。
    English: Visualize GCN message passing. Returns aggregation details and neighbor message strengths for a specific target node across GCN layers.
    """
    # Get data from request
    data = request.get_json()
    original_sequence = data.get('rnaSequence', '')
    target_node_idx = data.get('targetNodeIdx')

    # Validate inputs
    if not original_sequence:
        return jsonify({"error": "No sequence provided"}), 400

    if target_node_idx is None or not isinstance(target_node_idx, int):
        return jsonify({"error": "Invalid targetNodeIdx. Must be an integer"}), 400

    logger.info(f"GCN Aggregation Viz: sequence length={len(original_sequence)}, target_node_idx={target_node_idx}")

    # Store original sequence for response
    sequence = original_sequence

    # For shorter sequences, pad to 1001; for longer sequences, truncate
    TARGET_LENGTH = 1001
    seq_len = len(sequence)

    # Track padding/trimming for index remapping
    left_padding = 0
    left_trimming = 0

    if seq_len != TARGET_LENGTH:
        if seq_len < TARGET_LENGTH:
            padding_needed = TARGET_LENGTH - seq_len
            left_pad = padding_needed // 2
            right_pad = padding_needed - left_pad
            left_padding = left_pad
            sequence = 'N' * left_pad + sequence + 'N' * right_pad
        else:
            excess = seq_len - TARGET_LENGTH
            left_trim = excess // 2
            right_trim = excess - left_trim
            left_trimming = left_trim
            sequence = sequence[left_trim:seq_len - right_trim]

    # Map original target node index to model coordinates
    model_target_idx = target_node_idx + left_padding - left_trimming

    # Validate model target index
    if not (0 <= model_target_idx < len(sequence)):
        return jsonify({"error": f"Target node index out of bounds after padding/trimming"}), 400

    try:
        model = current_app.config['MODEL']
        device = current_app.config['MODEL_DEVICE']

        # Step 1: Call LinearFold to get secondary structure
        structures = run_linearfold([sequence])
        structure = structures[0]

        # Step 2: Build edge index from structure
        edge_index = build_edge_index_from_structure(sequence, structure)

        # Step 3: Prepare model input
        x = one_hot_encode_sequence(sequence)
        x = torch.FloatTensor(x)  # Shape: [1001, 4]

        # Create batch tensor (single sample)
        batch = torch.zeros(len(sequence), dtype=torch.long)

        # Step 4: Create Batch object
        data_batch = Batch(x=x, edge_index=edge_index, batch=batch)
        data_batch = data_batch.to(device)

        # Step 5: Run model with aggregation details
        with torch.no_grad():
            output, aggregation_details = model(
                data_batch.x,
                data_batch.edge_index,
                data_batch.batch,
                return_aggregation_details=True,
                target_node_idx=model_target_idx
            )

        # Step 6: Process aggregation details
        # Map model indices back to original indices
        processed_aggregation = []
        for layer_data in aggregation_details:
            processed_layer = {
                "layer": layer_data["layer"],
                "messages": []
            }

            for msg in layer_data["messages"]:
                # Map model index to original index
                model_from_idx = msg["from"]
                orig_from_idx = model_from_idx - left_padding + left_trimming

                # Only include messages within original sequence bounds
                if 0 <= orig_from_idx < len(original_sequence):
                    processed_layer["messages"].append({
                        "from": orig_from_idx,
                        "strength": msg["strength"]
                    })

            processed_aggregation.append(processed_layer)

        # Step 7: Build graph structure for visualization
        edge_index_np = edge_index.cpu().numpy()
        edges = []

        # Calculate the valid range in model coordinates
        valid_start = left_padding
        valid_end = left_padding + len(original_sequence)

        # Process all edges
        for i in range(int(edge_index_np.shape[1])):
            source = int(edge_index_np[0, i])
            target = int(edge_index_np[1, i])

            # Only process edges within valid range
            if not (valid_start <= source < valid_end and valid_start <= target < valid_end):
                continue

            # Map model indices to original indices
            orig_source = source - left_padding + left_trimming
            orig_target = target - left_padding + left_trimming

            # Only include edges within original sequence bounds
            if not (0 <= orig_source < len(original_sequence) and 0 <= orig_target < len(original_sequence)):
                continue

            # Only keep one direction (source < target) to avoid duplicates
            if orig_source >= orig_target:
                continue

            nuc_source = original_sequence[orig_source]
            nuc_target = original_sequence[orig_target]
            edges.append({
                "source": f"{nuc_source}{orig_source}",
                "target": f"{nuc_target}{orig_target}"
            })

        # Create nodes
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

        logger.info(f"GCN Aggregation: 节点数={len(nodes)}, 边数={len(edges)}, 层数={len(processed_aggregation)}")

        # Return response
        return jsonify({
            "targetNode": target_node_idx,
            "nodes": nodes,
            "edges": edges,
            "aggregationData": processed_aggregation
        }), 200

    except Exception as e:
        import traceback
        error_msg = f"GCN Aggregation error: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500
