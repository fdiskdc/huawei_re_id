"""
Tests for API routes - Verify Flask app routes are registered.

Tests that all expected API endpoints exist and the health endpoint responds.
The application factory is exercised with Redis and model loading mocked.
"""
import sys
import os
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _get_app():
    """Import and return the Flask app with mocked dependencies."""
    # Mock heavy dependencies before importing server
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    mock_redis.get.return_value = None
    mock_redis.setex.return_value = True

    from mrmodn_backend import app as app_module
    with patch('redis.Redis', return_value=mock_redis), \
         patch.object(app_module, 'load_model') as load_model:
        mock_model_instance = MagicMock()
        load_model.return_value = (mock_model_instance, 'cpu', {'use_hierarchical': True})
        return app_module.create_app()


class TestAPIRoutes:
    """Test that all API routes are registered."""

    def test_health_route_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/mrmodn/api/health' in rules

    def test_wx_login_route_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/mrmodn/api/v1/wx/login' in rules

    def test_wx_submit_task_route_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/mrmodn/api/v1/wx-submit-task' in rules

    def test_wx_task_progress_route_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/mrmodn/api/v1/wx-task-progress/<job_id>' in rules

    def test_submit_task_route_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/mrmodn/api/v1/submit-task' in rules

    def test_results_route_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/mrmodn/api/v1/results/<job_id>' in rules

    def test_model_architecture_route_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/mrmodn/api/v1/model-architecture' in rules

    def test_model_graph_route_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/mrmodn/api/v1/model-graph' in rules

    def test_integrated_gradients_route_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/mrmodn/api/v1/integrated-gradients' in rules

    def test_visualize_gcn_aggregation_route_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/mrmodn/api/v1/visualize-gcn-aggregation' in rules

    def test_reid_routes_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/mrmodn/api/v1/reid/meta' in rules
        assert '/mrmodn/api/v1/reid/batches/<int:batch_index>' in rules
        assert '/mrmodn/api/v1/reid/samples/<sample_id>/image' in rules

    def test_total_api_routes_count(self):
        """Verify we have at least 10 API routes (excluding static and HEAD/OPTIONS)."""
        app = _get_app()
        api_rules = {
            rule.rule for rule in app.url_map.iter_rules()
            if rule.rule.startswith('/mrmodn/api/')
        }
        assert len(api_rules) >= 10, f"Expected >= 10 API routes, got {len(api_rules)}: {api_rules}"


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_returns_200(self):
        app = _get_app()
        with app.test_client() as client:
            response = client.get('/mrmodn/api/health')
            assert response.status_code == 200

    def test_health_returns_json(self):
        app = _get_app()
        with app.test_client() as client:
            response = client.get('/mrmodn/api/health')
            data = json.loads(response.data)
            assert 'status' in data
            assert data['status'] == 'ok'
