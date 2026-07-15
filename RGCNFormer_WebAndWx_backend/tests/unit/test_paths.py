"""
Tests for resource paths - Verify critical files exist.

Ensures that model config, checkpoint, and LinearFold executable are present.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestResourcePaths:
    """Test that critical resource files exist."""

    def test_human_json_exists(self, project_root):
        path = os.path.join(project_root, 'json', 'human.json')
        assert os.path.isfile(path), f"Missing: json/human.json"

    def test_model_graph_json_exists(self, project_root):
        path = os.path.join(project_root, 'json', 'model_graph.json')
        assert os.path.isfile(path), f"Missing: json/model_graph.json"

    def test_epoch_040_pt_exists(self, project_root):
        path = os.path.join(project_root, 'epoch_040.pt')
        assert os.path.isfile(path), f"Missing: epoch_040.pt"

    def test_linearfold_executable_exists(self, project_root):
        path = os.path.join(project_root, 'LinearFold', 'linearfold')
        assert os.path.exists(path), f"Missing: LinearFold/linearfold"
