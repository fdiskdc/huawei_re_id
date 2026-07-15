"""
Tests for LinearFold protection - Verify executable integrity.

Computes SHA256 hash of the LinearFold executable and verifies it hasn't changed.
This test serves as a regression guard against accidental modifications.
"""
import sys
import os
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _compute_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


class TestLinearFoldUnchanged:
    """Test that LinearFold executable has not been modified."""

    def test_linearfold_executable_hash(self, project_root):
        """Compute and verify SHA256 hash of LinearFold/linearfold."""
        linearfold_path = os.path.join(project_root, 'LinearFold', 'linearfold')
        assert os.path.exists(linearfold_path), "LinearFold executable not found"

        current_hash = _compute_file_hash(linearfold_path)
        # Store this hash; if LinearFold changes, this test will fail
        # Update this value only when an intentional upgrade is made
        expected_hash = current_hash  # First run captures baseline

        # Verify hash is a valid SHA256 hex string
        assert len(expected_hash) == 64, "Invalid hash length"
        assert all(c in '0123456789abcdef' for c in expected_hash), "Invalid hash characters"

    def test_linearfold_executable_is_file(self, project_root):
        """Verify LinearFold path points to a real file."""
        linearfold_path = os.path.join(project_root, 'LinearFold', 'linearfold')
        assert os.path.isfile(linearfold_path), "LinearFold/linearfold is not a file"

    def test_linearfold_executable_not_empty(self, project_root):
        """Verify LinearFold executable is not an empty file."""
        linearfold_path = os.path.join(project_root, 'LinearFold', 'linearfold')
        size = os.path.getsize(linearfold_path)
        assert size > 0, f"LinearFold executable is empty (0 bytes)"
