"""Route-registration tests for the ReID blueprint on the Cluster backend."""

from flask import Flask

from mrmodn_backend.api.reid import reid_bp


def _app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(reid_bp)
    return app


def test_reid_routes_registered():
    rules = {rule.rule for rule in _app().url_map.iter_rules()}
    assert '/api/v1/reid/meta' in rules
    assert '/api/v1/reid/batches/<int:batch_index>' in rules
    assert '/api/v1/reid/samples/<sample_id>/image' in rules
