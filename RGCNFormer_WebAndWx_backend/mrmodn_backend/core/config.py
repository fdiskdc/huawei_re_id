"""
中文：统一配置管理模块。从环境变量加载配置并提供合理默认值。
English: Unified configuration management with environment-variable overrides.
"""
import os
import logging
from typing import Dict

from mrmodn_backend.core.paths import MODEL_CHECKPOINT_PATH, HUMAN_JSON_PATH, REID_CHECKPOINT_PATH


class Config:
    """
    中文：应用配置类，从环境变量读取 Redis、Celery、模型、服务器、日志及微信等配置。
    English: Application configuration class. Reads Redis, Celery, model, server, logging, and WeChat settings from environment variables.
    """

    def __init__(self):
        """Initialize configuration from environment variables."""
        # Redis Configuration
        self.REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
        self.REDIS_DB = int(os.getenv('REDIS_DB', 0))

        # Celery Configuration
        self.CELERY_BROKER_URL = os.getenv(
            'CELERY_BROKER_URL',
            f'redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}'
        )
        self.CELERY_RESULT_BACKEND = os.getenv(
            'CELERY_RESULT_BACKEND',
            f'redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}'
        )
        self.CELERY_TASK_TIME_LIMIT = int(os.getenv('CELERY_TASK_TIME_LIMIT', 3600))
        self.CELERY_TASK_SOFT_TIME_LIMIT = int(os.getenv('CELERY_TASK_SOFT_TIME_LIMIT', 3000))

        # Model Configuration (paths resolved via core.paths)
        self.MODEL_CHECKPOINT_PATH = os.getenv('MODEL_CHECKPOINT_PATH', MODEL_CHECKPOINT_PATH)
        self.MODEL_CONFIG_PATH = os.getenv('MODEL_CONFIG_PATH', HUMAN_JSON_PATH)
        self.MODEL_DEVICE = os.getenv('MODEL_DEVICE', 'cpu')
        self.MODEL_TARGET_LENGTH = int(os.getenv('MODEL_TARGET_LENGTH', 1001))

        # ReID Model Configuration
        self.REID_CHECKPOINT_PATH = os.getenv('REID_CHECKPOINT_PATH', REID_CHECKPOINT_PATH)
        self.REID_DATA_ROOT = os.getenv('REID_DATA_ROOT', '/home/dc/vscode/re_id/SYSU-MM01')
        self.REID_DEVICE = os.getenv('REID_DEVICE', 'cpu')
        self.REID_BATCH_SIZE = 4  # Fixed batch size for visualization
        self.REID_HEATMAP_SIZE = (9, 5)  # Original grid shape [height, width]
        self.REID_DISPLAY_SIZE = 256  # Canvas display size in pixels
        self.REID_TORCH_NUM_THREADS = int(os.getenv('REID_TORCH_NUM_THREADS', '4'))
        self.REID_CACHE_SIZE = int(os.getenv('REID_CACHE_SIZE', '8'))

        if self.REID_DEVICE.lower() != 'cpu':
            raise ValueError("ReID model only supports CPU device. Set REID_DEVICE=cpu")

        # Server Configuration
        self.FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
        self.FLASK_PORT = int(os.getenv('FLASK_PORT', 8000))
        self.FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

        # Logging Configuration
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FORMAT = os.getenv(
            'LOG_FORMAT',
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        self.LOG_FILE_MAX_BYTES = int(os.getenv('LOG_FILE_MAX_BYTES', 10485760))  # 10MB
        self.LOG_FILE_BACKUP_COUNT = int(os.getenv('LOG_FILE_BACKUP_COUNT', 5))

        # Cache Configuration
        self.REDIS_CACHE_TTL = int(os.getenv('REDIS_CACHE_TTL', 3600))

        # WeChat Mini Program Configuration
        self.WX_APPID = os.getenv('WX_APPID', '')
        self.WX_SECRET = os.getenv('WX_SECRET', '')
        self.WX_LOGIN_URL = 'https://api.weixin.qq.com/sns/jscode2session'

        # Classification Thresholds (12-class)
        self.THRESHOLDS_12_CLASS: Dict[int, float] = {
            0: float(os.getenv('THRESHOLD_12_AM', 0.510)),
            1: float(os.getenv('THRESHOLD_12_ATOL', 0.400)),
            2: float(os.getenv('THRESHOLD_12_CM', 0.690)),
            3: float(os.getenv('THRESHOLD_12_GM', 0.710)),
            4: float(os.getenv('THRESHOLD_12_TM', 0.350)),
            5: float(os.getenv('THRESHOLD_12_Y', 0.150)),
            6: float(os.getenv('THRESHOLD_12_AC4C', 0.120)),
            7: float(os.getenv('THRESHOLD_12_M1A', 0.380)),
            8: float(os.getenv('THRESHOLD_12_M5C', 0.350)),
            9: float(os.getenv('THRESHOLD_12_M6A', 0.260)),
            10: float(os.getenv('THRESHOLD_12_M6AM', 0.570)),
            11: float(os.getenv('THRESHOLD_12_M7G', 0.130)),
        }

        # Classification Thresholds (4-class)
        self.THRESHOLDS_4_CLASS: Dict[int, float] = {
            0: float(os.getenv('THRESHOLD_4_A', 0.980)),
            1: float(os.getenv('THRESHOLD_4_C', 0.270)),
            2: float(os.getenv('THRESHOLD_4_G', 0.050)),
            3: float(os.getenv('THRESHOLD_4_U', 0.050)),
        }

        # Default Top-K
        self.DEFAULT_TOP_K = int(os.getenv('DEFAULT_TOP_K', 3))

    def setup_logging(self, name: str = None, log_file: str = None) -> logging.Logger:
        """
        Set up logging configuration.

        Args:
            name: Logger name (defaults to the calling module's name)
            log_file: Optional log file path for file handler

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)

        # Set log level
        numeric_level = getattr(logging, self.LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(numeric_level)

        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()

        # Create formatter
        log_format = self.LOG_FORMAT
        if log_format.lower() == "json":
            log_format = '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
        formatter = logging.Formatter(log_format)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self.LOG_FILE_MAX_BYTES,
                backupCount=self.LOG_FILE_BACKUP_COUNT
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger


# Global configuration instance
config = Config()


def get_logger(name: str = None, log_file: str = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (defaults to the calling module's name)
        log_file: Optional log file path for file handler

    Returns:
        Configured logger instance
    """
    return config.setup_logging(name, log_file)
