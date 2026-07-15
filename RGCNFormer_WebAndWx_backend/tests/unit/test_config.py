"""
Tests for config.py - Configuration loading and validation.

Verifies that the Config class loads correct default values and has all required attributes.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mrmodn_backend.core.config import config as _config_instance

# Get the Config class for attribute testing
Config = type(_config_instance)


class TestConfigDefaults:
    """Test default configuration values."""

    def test_redis_host_default(self, config):
        assert config.REDIS_HOST == 'localhost'

    def test_redis_port_default(self, config):
        assert config.REDIS_PORT == 6379

    def test_redis_db_default(self, config):
        assert config.REDIS_DB == 0

    def test_model_checkpoint_path_default(self, config):
        assert config.MODEL_CHECKPOINT_PATH.endswith('epoch_040.pt')

    def test_model_config_path_default(self, config):
        assert config.MODEL_CONFIG_PATH.endswith(os.path.join('json', 'human.json'))

    def test_model_device_default(self, config):
        assert config.MODEL_DEVICE == 'cpu'

    def test_model_target_length_default(self, config):
        assert config.MODEL_TARGET_LENGTH == 1001

    def test_flask_host_default(self, config):
        assert config.FLASK_HOST == '0.0.0.0'

    def test_flask_port_default(self, config):
        assert config.FLASK_PORT == 8000

    def test_flask_debug_default(self, config):
        assert config.FLASK_DEBUG is False

    def test_log_level_default(self, config):
        assert config.LOG_LEVEL == 'INFO'

    def test_redis_cache_ttl_default(self, config):
        assert config.REDIS_CACHE_TTL == 3600

    def test_celery_task_time_limit_default(self, config):
        assert config.CELERY_TASK_TIME_LIMIT == 3600

    def test_celery_task_soft_time_limit_default(self, config):
        assert config.CELERY_TASK_SOFT_TIME_LIMIT == 3000

    def test_default_top_k(self, config):
        assert config.DEFAULT_TOP_K == 3


class TestConfigAttributes:
    """Test that Config class has all required attributes."""

    def test_has_redis_attributes(self, config):
        assert hasattr(config, 'REDIS_HOST')
        assert hasattr(config, 'REDIS_PORT')
        assert hasattr(config, 'REDIS_DB')

    def test_has_celery_attributes(self, config):
        assert hasattr(config, 'CELERY_BROKER_URL')
        assert hasattr(config, 'CELERY_RESULT_BACKEND')
        assert hasattr(config, 'CELERY_TASK_TIME_LIMIT')
        assert hasattr(config, 'CELERY_TASK_SOFT_TIME_LIMIT')

    def test_has_model_attributes(self, config):
        assert hasattr(config, 'MODEL_CHECKPOINT_PATH')
        assert hasattr(config, 'MODEL_CONFIG_PATH')
        assert hasattr(config, 'MODEL_DEVICE')
        assert hasattr(config, 'MODEL_TARGET_LENGTH')

    def test_has_server_attributes(self, config):
        assert hasattr(config, 'FLASK_HOST')
        assert hasattr(config, 'FLASK_PORT')
        assert hasattr(config, 'FLASK_DEBUG')

    def test_has_logging_attributes(self, config):
        assert hasattr(config, 'LOG_LEVEL')
        assert hasattr(config, 'LOG_FORMAT')
        assert hasattr(config, 'LOG_FILE_MAX_BYTES')
        assert hasattr(config, 'LOG_FILE_BACKUP_COUNT')

    def test_has_cache_attributes(self, config):
        assert hasattr(config, 'REDIS_CACHE_TTL')

    def test_has_wechat_attributes(self, config):
        assert hasattr(config, 'WX_APPID')
        assert hasattr(config, 'WX_SECRET')
        assert hasattr(config, 'WX_LOGIN_URL')

    def test_has_threshold_attributes(self, config):
        assert hasattr(config, 'THRESHOLDS_12_CLASS')
        assert hasattr(config, 'THRESHOLDS_4_CLASS')


class TestThresholdDictionaries:
    """Test threshold dictionaries have correct keys."""

    def test_thresholds_12_class_has_12_keys(self, config):
        assert len(config.THRESHOLDS_12_CLASS) == 12

    def test_thresholds_12_class_keys_range(self, config):
        assert set(config.THRESHOLDS_12_CLASS.keys()) == set(range(12))

    def test_thresholds_12_class_values_are_floats(self, config):
        for key, value in config.THRESHOLDS_12_CLASS.items():
            assert isinstance(value, float), f"Key {key} value is not float: {type(value)}"

    def test_thresholds_12_class_values_in_range(self, config):
        for key, value in config.THRESHOLDS_12_CLASS.items():
            assert 0.0 <= value <= 1.0, f"Key {key} value out of range: {value}"

    def test_thresholds_4_class_has_4_keys(self, config):
        assert len(config.THRESHOLDS_4_CLASS) == 4

    def test_thresholds_4_class_keys_range(self, config):
        assert set(config.THRESHOLDS_4_CLASS.keys()) == set(range(4))

    def test_thresholds_4_class_values_are_floats(self, config):
        for key, value in config.THRESHOLDS_4_CLASS.items():
            assert isinstance(value, float), f"Key {key} value is not float: {type(value)}"

    def test_thresholds_4_class_values_in_range(self, config):
        for key, value in config.THRESHOLDS_4_CLASS.items():
            assert 0.0 <= value <= 1.0, f"Key {key} value out of range: {value}"
