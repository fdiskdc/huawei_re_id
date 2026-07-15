"""Shared Classic/NextGen attention inference service."""
import numpy as np
import torch
from torch_geometric.data import Batch
from mrmodn_backend.core.constants import MOD_NAMES
from mrmodn_backend.services.prediction import one_hot_encode_sequence
from mrmodn_backend.services.rna_structure import run_linearfold, build_edge_index_from_structure


def visualize(model, device, model_cfg, original_sequence: str):
    target = 1001
    sequence = original_sequence.upper()
    left_padding = 0
    if len(sequence) < target:
        left_padding = (target - len(sequence)) // 2
        sequence = "N" * left_padding + sequence + "N" * (target - len(sequence) - left_padding)
    elif len(sequence) > target:
        trim = len(sequence) - target
        sequence = sequence[trim // 2: len(sequence) - (trim - trim // 2)]
    structure = run_linearfold([sequence])[0]
    edge_index = build_edge_index_from_structure(sequence, structure)
    encoded = torch.as_tensor(one_hot_encode_sequence(sequence), dtype=torch.float32)
    batch = Batch(x=encoded, edge_index=edge_index, batch=torch.zeros(target, dtype=torch.long)).to(device)
    with torch.no_grad():
        output = model(batch.x, batch.edge_index, batch.batch, return_attention=True)
    logits, attention = output[0], output[2] if model_cfg.get("use_hierarchical") else output[1]
    probabilities = torch.sigmoid(logits).cpu().numpy()[0]
    weights = attention.cpu().numpy()[0]
    classes = []
    for index, row in enumerate(weights):
        normalized = row / np.sum(row) if np.sum(row) else row
        classes.append({"index": index, "name": MOD_NAMES[index], "probability": float(probabilities[index]), "attention": normalized.tolist()})
    return {"sequence_length": len(original_sequence), "left_padding": left_padding, "classes": classes, "class_names": list(MOD_NAMES)}

