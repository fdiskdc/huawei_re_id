"""
Build and address cached per-class attention distributions.

The model always operates on 1001 positions. This module maps those model
coordinates back to the visible range of the original sequence and keeps the
format shared by local tasks, Docker tasks, and HTTP endpoints.
"""
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


ATTENTION_DISTRIBUTION_KEY_PREFIX = "attention_distribution:"
NORMALIZATION = "visible_modeled_range_sum_1"


def attention_distribution_cache_key(job_id: str) -> str:
    return f"{ATTENTION_DISTRIBUTION_KEY_PREFIX}{job_id}"


def build_attention_distribution(
    original_sequence: str,
    attn_weights: np.ndarray,
    probs_12class: Sequence[float],
    predictions_12class: Mapping[int, bool],
    thresholds_12class: Mapping[int, float],
    class_names: Iterable[str],
    left_padding: int = 0,
    left_trimming: int = 0,
    precision: int = 8,
) -> Dict:
    """Create the API/cache representation for a model attention tensor."""
    attention = np.asarray(attn_weights, dtype=np.float64)
    if attention.ndim != 2:
        raise ValueError(f"Expected a 2D attention tensor, got shape {attention.shape}")

    names = list(class_names)
    class_count, model_length = attention.shape
    if len(names) != class_count:
        raise ValueError(f"Expected {class_count} class names, got {len(names)}")
    if len(probs_12class) != class_count:
        raise ValueError(f"Expected {class_count} probabilities, got {len(probs_12class)}")

    original_length = len(original_sequence)
    if original_length <= model_length:
        model_start = left_padding
        model_end = min(model_start + original_length, model_length)
        modeled_start = 0
    else:
        model_start = 0
        model_end = model_length
        modeled_start = left_trimming

    visible_length = model_end - model_start
    modeled_end = modeled_start + visible_length

    classes = []
    for index, name in enumerate(names):
        visible_attention = attention[index, model_start:model_end].copy()
        attention_sum = float(visible_attention.sum())
        if attention_sum > 0:
            visible_attention /= attention_sum

        classes.append({
            "index": index,
            "name": name,
            "probability": round(float(probs_12class[index]), precision),
            "threshold": round(float(thresholds_12class[index]), precision),
            "is_predicted": bool(predictions_12class.get(index, False)),
            "attention": np.round(visible_attention, precision).tolist(),
        })

    return {
        "sequence_length": original_length,
        "modeled_range": {
            "start": modeled_start,
            "end": modeled_end,
        },
        "normalization": NORMALIZATION,
        "classes": classes,
    }
