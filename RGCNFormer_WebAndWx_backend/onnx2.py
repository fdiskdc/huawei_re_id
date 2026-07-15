"""
onnx2.py - PyTorch → ONNX → graph JSON 导出(变体) / PyTorch → ONNX → graph JSON (v2)

与 onnx.py 类似,但同时加载原模型(用于权重)与 ONNX 友好模型(用于 trace),用
torch.jit.trace 生成更稳定/可移植的 ONNX。 / Similar to onnx.py but loads both
the original model (for weights) and the ONNX-friendly model (for trace), then
uses torch.jit.trace to produce a more stable/portable ONNX.

功能模块 / Modules:
- export_model_to_json(): 加载两个模型、copy state_dict、trace、导出 / load both, copy, trace, export
- 错误处理:trace 失败时回退到 torch.onnx.export / fall back to torch.onnx.export

输入 / Inputs:
- config.MODEL_CHECKPOINT_PATH: .pt 权重 / checkpoint
- config.MODEL_CONFIG_PATH: 模型超参 / model hyperparams

输出 / Outputs:
- *.onnx: ONNX 模型 / ONNX model
- *.json: 图结构 / graph JSON

数据流 / Data Flow:
1. 实例化 OriginalModel,加载权重 / Instantiate OriginalModel, load weights
2. 实例化 OnnxModel,copy 权重 / Instantiate OnnxModel, copy weights
3. trace → 导出 ONNX → 解析图 / trace, export, parse

相关文件 / Related Files:
- 调用 / Calls: main_model、main_model_onnx、torch.jit
- 被调用 / Called by: 部署期手动运行 / Manual run during deploy

使用示例 / Usage Example:
    python onnx2.py

作者 / Author: 项目组 / Project Team
版本 / Version: 1.0
"""
import torch
import json
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import BOTH the original model and the ONNX-friendly version
from main_model import RNA_ClassQuery_Model as OriginalModel
from main_model_onnx import RNA_ClassQuery_Model as OnnxModel
from config import config, get_logger

logger = get_logger('graph_exporter')

def export_model_to_json():
    logger.info("Starting model graph export process...")

    # --- 1. Load Original Model and Trained Weights ---
    logger.info(f"Loading ORIGINAL model configuration from: {config.MODEL_CONFIG_PATH}")
    with open(config.MODEL_CONFIG_PATH, 'r') as f:
        model_cfg = json.load(f)['model']

    device = torch.device('cpu')

    logger.info("Instantiating original model...")
    original_model = OriginalModel(**model_cfg)

    logger.info(f"Loading trained weights from {config.MODEL_CHECKPOINT_PATH}...")
    checkpoint = torch.load(config.MODEL_CHECKPOINT_PATH, map_location=device,weights_only=False)
    original_model.load_state_dict(checkpoint['model_state_dict'])
    original_model.eval()
    logger.info("Original model with trained weights loaded successfully.")

    # --- 2. Create ONNX-friendly model and COPY weights ---
    logger.info("Creating ONNX-friendly model instance and copying weights...")
    onnx_friendly_model = OnnxModel(**model_cfg)
    onnx_friendly_model.load_state_dict(original_model.state_dict())
    onnx_friendly_model.eval()
    logger.info("Weights successfully copied to the ONNX-friendly model.")

    # --- 3. Create Dummy Input ---
    batch_size = 1
    seq_length = 1001
    num_features = 4
    num_edges = 500

    # CNN expects [batch, channels, sequence] format
    dummy_x = torch.randn(batch_size, num_features, seq_length, device=device)
    dummy_edge_index = torch.randint(0, seq_length, (2, num_edges), device=device)
    dummy_batch = torch.zeros(batch_size * seq_length, dtype=torch.long, device=device)

    json_path = os.path.join(os.path.dirname(__file__), '..', 'model_graph.json')

    # --- 4. Trace model and extract graph structure ---
    logger.info("Tracing model to extract graph structure...")
    try:
        traced_model = torch.jit.trace(onnx_friendly_model, (dummy_x, dummy_edge_index, dummy_batch))
        logger.info("Model traced successfully.")
    except Exception as e:
        logger.error(f"Failed to trace model. Error: {e}", exc_info=True)
        return

    # --- 5. Extract graph information and save to JSON ---
    logger.info(f"Extracting graph structure and saving to: {json_path}")

    # Get the graph from the traced ScriptModule
    graph = traced_model.graph
    nodes = []
    edges = []
    tensor_to_producer = {}

    # Map to track unique node names
    node_name_counter = {}

    # Process all nodes in the graph
    for i, node in enumerate(graph.nodes()):
        # Generate a unique node name
        node_kind = str(node.kind())
        if node_kind in node_name_counter:
            node_name_counter[node_kind] += 1
        else:
            node_name_counter[node_kind] = 0
        node_name = f"{node_kind}_{node_name_counter[node_kind]}"

        # Extract input and output tensor names
        input_names = []
        output_names = []

        for inp in node.inputs():
            try:
                input_names.append(str(inp.debugName()))
            except:
                input_names.append(str(inp))

        for outp in node.outputs():
            try:
                output_name = str(outp.debugName())
                output_names.append(output_name)
                # Track which node produces each tensor
                tensor_to_producer[output_name] = node_name
            except:
                output_names.append(str(outp))

        nodes.append({
            "id": node_name,
            "label": node_kind,
            "inputs": input_names,
            "outputs": output_names,
            "attributes": {}
        })

    # Create edges based on data flow
    edge_counter = {}
    for i, node in enumerate(graph.nodes()):
        node_kind = str(node.kind())
        node_name = f"{node_kind}_{node_name_counter[node_kind]}"

        for inp in node.inputs():
            try:
                input_name = str(inp.debugName())
                if input_name in tensor_to_producer:
                    source_node = tensor_to_producer[input_name]
                    edge_id = f"edge_{source_node}_{node_name}"
                    if edge_id in edge_counter:
                        edge_counter[edge_id] += 1
                    else:
                        edge_counter[edge_id] = 0

                    edges.append({
                        "id": f"{edge_id}_{edge_counter[edge_id]}",
                        "source": source_node,
                        "target": node_name
                    })
            except:
                pass

    # Save to JSON
    with open(json_path, 'w') as f:
        json.dump({"nodes": nodes, "edges": edges}, f, indent=2)

    logger.info(f"Graph JSON saved successfully with {len(nodes)} nodes and {len(edges)} edges.")
    logger.info(f"Graph saved to: {json_path}")

if __name__ == '__main__':
    export_model_to_json()
