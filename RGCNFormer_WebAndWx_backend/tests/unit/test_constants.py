"""
Tests for common.py - Constants and mappings.

Verifies that all constant definitions are correct and consistent.
"""
import sys
import os
import importlib
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Mock torch and torch_geometric if not installed (constants don't need them)
for mod_name in ['torch', 'torch.nn', 'torch.utils', 'torch.utils.data',
                 'torch_geometric', 'torch_geometric.data', 'torch_geometric.loader']:
    try:
        importlib.import_module(mod_name)
    except ImportError:
        sys.modules[mod_name] = MagicMock()

from common import (
    MOD_NAMES,
    INDEX_TO_NUCLEOTIDE,
    NUCLEOTIDE_GROUP_NAMES,
    GROUP_TO_CLASS_INDICES,
    NUCLEOTIDE_GROUPS,
    GROUP_TO_INDEX,
    INDEX_TO_GROUP,
    GROUP_SIZES,
)


class TestMOD_NAMES:
    """Test MOD_NAMES constant."""

    def test_mod_names_has_12_entries(self):
        assert len(MOD_NAMES) == 12

    def test_mod_names_keys_are_0_to_11(self):
        assert set(MOD_NAMES.keys()) == set(range(12))

    def test_mod_names_values_are_strings(self):
        for key, value in MOD_NAMES.items():
            assert isinstance(value, str), f"Key {key} value is not str: {type(value)}"

    def test_mod_names_expected_values(self):
        expected = {
            0: 'Am', 1: 'Atol', 2: 'Cm',
            3: 'Gm', 4: 'Tm', 5: 'Y',
            6: 'ac4C', 7: 'm1A', 8: 'm5C',
            9: 'm6A', 10: 'm6Am', 11: 'm7G'
        }
        assert MOD_NAMES == expected


class TestINDEX_TO_NUCLEOTIDE:
    """Test INDEX_TO_NUCLEOTIDE mapping."""

    def test_index_to_nucleotide_has_12_entries(self):
        assert len(INDEX_TO_NUCLEOTIDE) == 12

    def test_index_to_nucleotide_keys_are_0_to_11(self):
        assert set(INDEX_TO_NUCLEOTIDE.keys()) == set(range(12))

    def test_index_to_nucleotide_values_are_valid(self):
        valid_nucleotides = {'A', 'C', 'G', 'U'}
        for key, value in INDEX_TO_NUCLEOTIDE.items():
            assert value in valid_nucleotides, f"Key {key} has invalid nucleotide: {value}"

    def test_index_to_nucleotide_mapping(self):
        expected = {
            0: 'A', 1: 'A',
            2: 'C',
            3: 'G',
            4: 'U', 5: 'U',
            6: 'C',
            7: 'A',
            8: 'C',
            9: 'A', 10: 'A',
            11: 'G'
        }
        assert INDEX_TO_NUCLEOTIDE == expected


class TestNUCLEOTIDE_GROUP_NAMES:
    """Test NUCLEOTIDE_GROUP_NAMES constant."""

    def test_group_names_has_4_entries(self):
        assert len(NUCLEOTIDE_GROUP_NAMES) == 4

    def test_group_names_are_correct(self):
        assert NUCLEOTIDE_GROUP_NAMES == ['A', 'C', 'G', 'U']


class TestGROUP_TO_CLASS_INDICES:
    """Test GROUP_TO_CLASS_INDICES mapping."""

    def test_group_to_class_has_4_keys(self):
        assert len(GROUP_TO_CLASS_INDICES) == 4

    def test_group_to_class_keys(self):
        assert set(GROUP_TO_CLASS_INDICES.keys()) == {'A', 'C', 'G', 'U'}

    def test_group_a_has_5_classes(self):
        assert len(GROUP_TO_CLASS_INDICES['A']) == 5

    def test_group_c_has_3_classes(self):
        assert len(GROUP_TO_CLASS_INDICES['C']) == 3

    def test_group_g_has_2_classes(self):
        assert len(GROUP_TO_CLASS_INDICES['G']) == 2

    def test_group_u_has_2_classes(self):
        assert len(GROUP_TO_CLASS_INDICES['U']) == 2

    def test_all_class_indices_covered(self):
        all_indices = []
        for indices in GROUP_TO_CLASS_INDICES.values():
            all_indices.extend(indices)
        assert sorted(all_indices) == list(range(12))

    def test_group_a_class_indices(self):
        assert GROUP_TO_CLASS_INDICES['A'] == [0, 1, 7, 9, 10]

    def test_group_c_class_indices(self):
        assert GROUP_TO_CLASS_INDICES['C'] == [2, 6, 8]

    def test_group_g_class_indices(self):
        assert GROUP_TO_CLASS_INDICES['G'] == [3, 11]

    def test_group_u_class_indices(self):
        assert GROUP_TO_CLASS_INDICES['U'] == [4, 5]


class TestNUCLEOTIDE_GROUPS:
    """Test NUCLEOTIDE_GROUPS constant."""

    def test_nucleotide_groups_has_4_keys(self):
        assert len(NUCLEOTIDE_GROUPS) == 4

    def test_nucleotide_groups_keys(self):
        assert set(NUCLEOTIDE_GROUPS.keys()) == {'A', 'C', 'G', 'U'}


class TestGROUP_TO_INDEX:
    """Test GROUP_TO_INDEX mapping."""

    def test_group_to_index_mapping(self):
        assert GROUP_TO_INDEX == {'A': 0, 'C': 1, 'G': 2, 'U': 3}


class TestINDEX_TO_GROUP:
    """Test INDEX_TO_GROUP mapping."""

    def test_index_to_group_mapping(self):
        assert INDEX_TO_GROUP == {0: 'A', 1: 'C', 2: 'G', 3: 'U'}


class TestGROUP_SIZES:
    """Test GROUP_SIZES constant."""

    def test_group_sizes(self):
        assert GROUP_SIZES == {'A': 5, 'C': 3, 'G': 2, 'U': 2}

    def test_group_sizes_sum_to_12(self):
        assert sum(GROUP_SIZES.values()) == 12
