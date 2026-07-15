# DEPRECATED: This file is no longer used.
# Configuration is now handled by mrmodn_backend/core/config.py
# which reads REDIS_HOST from environment variables.
"""
Configuration management for mRModN backend.

Loads settings from environment variables with sensible defaults.
"""
import os
import logging
from typing import Dict


class Config:
    """Application configuration class."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        # 核心修改：优先读取环境变量 REDIS_HOST，默认指向 docker 服务名 'redis'
        self.REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
        self.REDIS_DB = int(os.getenv('REDIS_DB', 0))

        # 动态拼接 URL
        redis_url = f'redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}'
        self.CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', redis_url)
        self.CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', redis_url)
        
        # 微信配置（对应你之前的需求）
        self.WX_APPID = os.getenv('WX_APPID')
        self.WX_SECRET = os.getenv('WX_SECRET')
        
        if not self.WX_APPID or not self.WX_SECRET:
            print("❌ 警告: WX_APPID 或 WX_SECRET 未配置！")

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

        # Model Configuration
        self.MODEL_CHECKPOINT_PATH = os.getenv('MODEL_CHECKPOINT_PATH', 'epoch_040.pt')
        self.MODEL_CONFIG_PATH = os.getenv('MODEL_CONFIG_PATH', 'json/human.json')
        self.MODEL_DEVICE = os.getenv('MODEL_DEVICE', 'cpu')
        self.MODEL_TARGET_LENGTH = int(os.getenv('MODEL_TARGET_LENGTH', 1001))

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
            0: float(os.getenv('THRESHOLD_12_AM', 0.510)),     # Am
            1: float(os.getenv('THRESHOLD_12_ATOL', 0.400)),   # Atol
            2: float(os.getenv('THRESHOLD_12_CM', 0.690)),     # Cm
            3: float(os.getenv('THRESHOLD_12_GM', 0.710)),     # Gm
            4: float(os.getenv('THRESHOLD_12_TM', 0.350)),     # Tm
            5: float(os.getenv('THRESHOLD_12_Y', 0.150)),      # Y
            6: float(os.getenv('THRESHOLD_12_AC4C', 0.120)),   # ac4C
            7: float(os.getenv('THRESHOLD_12_M1A', 0.380)),   # m1A
            8: float(os.getenv('THRESHOLD_12_M5C', 0.350)),   # m5C
            9: float(os.getenv('THRESHOLD_12_M6A', 0.260)),   # m6A
            10: float(os.getenv('THRESHOLD_12_M6AM', 0.570)),  # m6Am
            11: float(os.getenv('THRESHOLD_12_M7G', 0.130)),   # m7G
        }

        # Classification Thresholds (4-class)
        self.THRESHOLDS_4_CLASS: Dict[int, float] = {
            0: float(os.getenv('THRESHOLD_4_A', 0.980)),  # A
            1: float(os.getenv('THRESHOLD_4_C', 0.270)),  # C
            2: float(os.getenv('THRESHOLD_4_G', 0.050)),  # G
            3: float(os.getenv('THRESHOLD_4_U', 0.050)),  # U
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
        formatter = logging.Formatter(self.LOG_FORMAT)

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
