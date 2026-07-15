import torch
import json
import os
import onnx as onnx

from mrmodn_backend.models.mrmodn import RNA_ClassQuery_Model
from mrmodn_backend.core.config import config, get_logger

logger = get_logger('graph_exporter')

def export_model_to_json():
    """
    Loads the model, exports it to ONNX, parses the ONNX graph,
    and saves the graph structure as a JSON file.
    """
    logger.info("Starting model graph export process...")

    # --- 1. Load Model ---
    logger.info(f"Loading model configuration from: {config.MODEL_CONFIG_PATH}")
    with open(config.MODEL_CONFIG_PATH, 'r') as f:
        model_cfg = json.load(f)['model']

    device = torch.device('cpu')
    model = RNA_ClassQuery_Model(**model_cfg)
    checkpoint = torch.load(config.MODEL_CHECKPOINT_PATH, map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Model loaded successfully on CPU.")

    # --- 2. Create Dummy Input ---
    # Create a dummy input tuple that matches the model's forward pass signature
    batch_size = 1
    seq_length = 1001
    num_features = 4
    num_edges = 500  # An arbitrary number of edges for the dummy graph

    dummy_x = torch.randn(batch_size, seq_length, num_features, device=device)
    dummy_edge_index = torch.randint(0, seq_length, (2, num_edges), device=device)
    dummy_batch = torch.zeros(batch_size * seq_length, dtype=torch.long, device=device)

    # The model's forward pass expects (x, edge_index, batch)
    # For ONNX export, we need to handle the case where x is already batched
    # So the dummy input will just be the x tensor, and others are passed as args
    dummy_input_for_export = dummy_x

    onnx_path = os.path.join(os.path.dirname(__file__), '..', '..', 'model.onnx')
    json_path = os.path.join(os.path.dirname(__file__), '..', '..', 'model_graph.json')

    # --- 3. Export to ONNX ---
    logger.info(f"Exporting model to ONNX at: {onnx_path}")
    try:
        # We need to wrap the model to match the expected inputs for tracing
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x, edge_index, batch):
                # The model returns a tuple, which is fine for ONNX
                return self.model(x, edge_index, batch)

        # The actual forward pass uses tensors, not PyG Batch objects
        # Redefine dummy inputs as tensors for the wrapper
        dummy_x_tensor = torch.randn(seq_length, num_features, device=device)

        # For scripting we might need to be more explicit
        # Let's try to trace with example inputs first
        torch.onnx.export(
            model,
            (dummy_x_tensor, dummy_edge_index, torch.zeros(seq_length, dtype=torch.long, device=device)),
            onnx_path,
            input_names=['x', 'edge_index', 'batch'],
            output_names=['output_12_class', 'output_4_class', 'attention_weights'],
            opset_version=14, # A reasonably modern opset
            dynamic_axes={
                'x': {0: 'sequence_length'},
                'edge_index': {1: 'num_edges'},
                'batch': {0: 'sequence_length'},
            },
            verbose=False
        )
        logger.info("Model exported to ONNX successfully.")
    except Exception as e:
        logger.error(f"Failed to export model to ONNX. This is often due to custom ops in torch_geometric.")
        logger.error(f"Error: {e}")
        return

    # --- 4. Parse ONNX to JSON ---
    logger.info(f"Parsing ONNX graph and saving to: {json_path}")
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph

    nodes = []
    edges = []

    tensor_to_producer = {}
    for node in graph.node:
        for output_name in node.output:
            tensor_to_producer[output_name] = node.name

    for node in graph.node:
        nodes.append({
            "id": node.name,
            "label": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        })

        for input_name in node.input:
            if input_name in tensor_to_producer:
                source_node_name = tensor_to_producer[input_name]
                edges.append({
                    "id": f"edge_{source_node_name}_{node.name}_{input_name}",
                    "source": source_node_name,
                    "target": node.name
                })

    graph_json = {"nodes": nodes, "edges": edges}

    with open(json_path, 'w') as f:
        json.dump(graph_json, f, indent=2)

    logger.info("Graph JSON saved successfully.")
    os.remove(onnx_path) # Clean up the large onnx file
    logger.info(f"Cleaned up temporary file: {onnx_path}")


if __name__ == '__main__':
    export_model_to_json()
