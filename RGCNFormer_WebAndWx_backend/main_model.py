"""
Backward-compatible shim for main_model.

All code has been moved to mrmodn_backend.models.mrmodn.
This file re-exports the public API for backward compatibility.
"""
from mrmodn_backend.models.mrmodn import (
    ParallelCNNBlock,
    GCNBlock,
    ClassQueryHead,
    ClassQueryHeadPooling,
    HierarchicalClassQueryHeadPooling,
    RNA_ClassQuery_Model,
)

__all__ = [
    'ParallelCNNBlock',
    'GCNBlock',
    'ClassQueryHead',
    'ClassQueryHeadPooling',
    'HierarchicalClassQueryHeadPooling',
    'RNA_ClassQuery_Model',
]
