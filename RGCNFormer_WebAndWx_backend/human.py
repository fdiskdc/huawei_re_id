"""
Backward-compatible shim for human.

All code has been moved to:
- mrmodn_backend.services.rna_structure  (run_linearfold, build_edge_index_from_structure, build_sequential_edge_index)
- mrmodn_backend.data.human              (Mer100Dataset, helpers, constants)
- mrmodn_backend.core.constants          (LABEL_MAPPING, INDEX_TO_NUCLEOTIDE, etc.)

This file re-exports the public API for backward compatibility.
"""
from mrmodn_backend.core.constants import LABEL_MAPPING, INDEX_TO_NUCLEOTIDE, NUCLEOTIDE_GROUPS, MOD_NAMES
from mrmodn_backend.data.human import (
    ONE_HOT_MAPPING,
    DEFAULT_ONE_HOT,
    TARGET_LENGTH,
    BATCH_CACHE_FILE,
    _create_byte_to_onehot_mapping,
    _BYTE_TO_ONEHOT_MAPPING,
    one_hot_to_sequence,
    Mer100Dataset,
)
from mrmodn_backend.services.rna_structure import (
    run_linearfold,
    build_edge_index_from_structure,
    build_sequential_edge_index,
)

LINEARFOLD_PATH = 'LinearFold/linearfold'

__all__ = [
    'LABEL_MAPPING',
    'INDEX_TO_NUCLEOTIDE',
    'NUCLEOTIDE_GROUPS',
    'MOD_NAMES',
    'ONE_HOT_MAPPING',
    'DEFAULT_ONE_HOT',
    'TARGET_LENGTH',
    'BATCH_CACHE_FILE',
    'LINEARFOLD_PATH',
    '_create_byte_to_onehot_mapping',
    '_BYTE_TO_ONEHOT_MAPPING',
    'one_hot_to_sequence',
    'run_linearfold',
    'build_edge_index_from_structure',
    'build_sequential_edge_index',
    'Mer100Dataset',
]
