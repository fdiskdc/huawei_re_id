"""
Tests for model loading - Verify model config and checkpoint integrity.

Tests that the model configuration JSON can be loaded, the checkpoint can be
loaded, and the model has the expected architecture components.
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestModelConfigLoading:
    """Test that model config JSON can be loaded correctly."""

    def test_human_json_is_valid_json(self, project_root):
        path = os.path.join(project_root, 'json', 'human.json')
        with open(path, 'r') as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_human_json_has_model_section(self, project_root):
        path = os.path.join(project_root, 'json', 'human.json')
        with open(path, 'r') as f:
            data = json.load(f)
        assert 'model' in data

    def test_model_config_has_cnn_params(self, project_root):
        path = os.path.join(project_root, 'json', 'human.json')
        with open(path, 'r') as f:
            data = json.load(f)
        model = data['model']
        assert 'cnn_hidden_dim' in model
        assert 'cnn_kernel_sizes' in model
        assert 'cnn_dropout' in model

    def test_model_config_has_gcn_params(self, project_root):
        path = os.path.join(project_root, 'json', 'human.json')
        with open(path, 'r') as f:
            data = json.load(f)
        model = data['model']
        assert 'gcn_hidden_dim' in model
        assert 'gcn_out_channels' in model
        assert 'gcn_num_layers' in model
        assert 'gcn_dropout' in model

    def test_model_config_has_num_classes(self, project_root):
        path = os.path.join(project_root, 'json', 'human.json')
        with open(path, 'r') as f:
            data = json.load(f)
        model = data['model']
        assert model['num_classes'] == 12

    def test_model_config_has_attention_params(self, project_root):
        path = os.path.join(project_root, 'json', 'human.json')
        with open(path, 'r') as f:
            data = json.load(f)
        model = data['model']
        assert 'num_attn_heads' in model
        assert 'attn_dropout' in model

    def test_model_config_has_hierarchical_flag(self, project_root):
        path = os.path.join(project_root, 'json', 'human.json')
        with open(path, 'r') as f:
            data = json.load(f)
        model = data['model']
        assert 'use_hierarchical' in model
        assert isinstance(model['use_hierarchical'], bool)


class TestModelCheckpointLoading:
    """Test that model checkpoint can be loaded."""

    def test_checkpoint_file_is_loadable(self, project_root):
        import torch
        path = os.path.join(project_root, 'epoch_040.pt')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert isinstance(checkpoint, dict)

    def test_checkpoint_has_model_state_dict(self, project_root):
        import torch
        path = os.path.join(project_root, 'epoch_040.pt')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert 'model_state_dict' in checkpoint

    def test_checkpoint_model_state_dict_is_dict(self, project_root):
        import torch
        path = os.path.join(project_root, 'epoch_040.pt')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert isinstance(checkpoint['model_state_dict'], dict)

    def test_checkpoint_model_state_dict_not_empty(self, project_root):
        import torch
        path = os.path.join(project_root, 'epoch_040.pt')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert len(checkpoint['model_state_dict']) > 0


class TestModelArchitecture:
    """Test model has expected architecture components."""

    def test_model_has_cnn_block(self, project_root):
        import torch
        from main_model import RNA_ClassQuery_Model

        path = os.path.join(project_root, 'json', 'human.json')
        with open(path, 'r') as f:
            model_config = json.load(f)
        cfg = model_config['model']

        model = RNA_ClassQuery_Model(
            cnn_hidden_dim=cfg['cnn_hidden_dim'],
            cnn_kernel_sizes=tuple(cfg['cnn_kernel_sizes']),
            cnn_dropout=cfg['cnn_dropout'],
            gcn_hidden_dim=cfg['gcn_hidden_dim'],
            gcn_out_channels=cfg['gcn_out_channels'],
            gcn_num_layers=cfg['gcn_num_layers'],
            gcn_dropout=cfg['gcn_dropout'],
            num_classes=cfg['num_classes'],
            num_attn_heads=cfg['num_attn_heads'],
            attn_dropout=cfg['attn_dropout'],
            use_simple_pooling=cfg['use_simple_pooling'],
            use_hierarchical=cfg['use_hierarchical'],
            use_layer_norm=cfg['use_layer_norm']
        )
        assert hasattr(model, 'cnn_block')

    def test_model_has_gcn_block(self, project_root):
        import torch
        from main_model import RNA_ClassQuery_Model

        path = os.path.join(project_root, 'json', 'human.json')
        with open(path, 'r') as f:
            model_config = json.load(f)
        cfg = model_config['model']

        model = RNA_ClassQuery_Model(
            cnn_hidden_dim=cfg['cnn_hidden_dim'],
            cnn_kernel_sizes=tuple(cfg['cnn_kernel_sizes']),
            cnn_dropout=cfg['cnn_dropout'],
            gcn_hidden_dim=cfg['gcn_hidden_dim'],
            gcn_out_channels=cfg['gcn_out_channels'],
            gcn_num_layers=cfg['gcn_num_layers'],
            gcn_dropout=cfg['gcn_dropout'],
            num_classes=cfg['num_classes'],
            num_attn_heads=cfg['num_attn_heads'],
            attn_dropout=cfg['attn_dropout'],
            use_simple_pooling=cfg['use_simple_pooling'],
            use_hierarchical=cfg['use_hierarchical'],
            use_layer_norm=cfg['use_layer_norm']
        )
        assert hasattr(model, 'gcn_block')

    def test_model_has_class_query_head(self, project_root):
        import torch
        from main_model import RNA_ClassQuery_Model

        path = os.path.join(project_root, 'json', 'human.json')
        with open(path, 'r') as f:
            model_config = json.load(f)
        cfg = model_config['model']

        model = RNA_ClassQuery_Model(
            cnn_hidden_dim=cfg['cnn_hidden_dim'],
            cnn_kernel_sizes=tuple(cfg['cnn_kernel_sizes']),
            cnn_dropout=cfg['cnn_dropout'],
            gcn_hidden_dim=cfg['gcn_hidden_dim'],
            gcn_out_channels=cfg['gcn_out_channels'],
            gcn_num_layers=cfg['gcn_num_layers'],
            gcn_dropout=cfg['gcn_dropout'],
            num_classes=cfg['num_classes'],
            num_attn_heads=cfg['num_attn_heads'],
            attn_dropout=cfg['attn_dropout'],
            use_simple_pooling=cfg['use_simple_pooling'],
            use_hierarchical=cfg['use_hierarchical'],
            use_layer_norm=cfg['use_layer_norm']
        )
        assert hasattr(model, 'class_query_head')

    def test_model_checkpoint_loads_into_model(self, project_root):
        import torch
        from main_model import RNA_ClassQuery_Model

        path = os.path.join(project_root, 'json', 'human.json')
        with open(path, 'r') as f:
            model_config = json.load(f)
        cfg = model_config['model']

        model = RNA_ClassQuery_Model(
            cnn_hidden_dim=cfg['cnn_hidden_dim'],
            cnn_kernel_sizes=tuple(cfg['cnn_kernel_sizes']),
            cnn_dropout=cfg['cnn_dropout'],
            gcn_hidden_dim=cfg['gcn_hidden_dim'],
            gcn_out_channels=cfg['gcn_out_channels'],
            gcn_num_layers=cfg['gcn_num_layers'],
            gcn_dropout=cfg['gcn_dropout'],
            num_classes=cfg['num_classes'],
            num_attn_heads=cfg['num_attn_heads'],
            attn_dropout=cfg['attn_dropout'],
            use_simple_pooling=cfg['use_simple_pooling'],
            use_hierarchical=cfg['use_hierarchical'],
            use_layer_norm=cfg['use_layer_norm']
        )

        ckpt_path = os.path.join(project_root, 'epoch_040.pt')
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
