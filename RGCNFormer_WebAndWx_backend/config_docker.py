"""
config_docker.py - 后端配置管理(Docker 环境) / Backend config (Docker)

与 config.py 结构相同,但默认 REDIS_HOST='redis'(docker-compose 中的服务名)而非
'localhost',并增加了 WX_APPID / WX_SECRET 用于微信登录。在 Docker 容器中跑
时使用。 / Same as config.py but defaults REDIS_HOST to 'redis' (the docker-compose
service name) instead of 'localhost', and adds WX_APPID / WX_SECRET for WeChat
login. Use when running inside Docker.

功能模块 / Modules:
- Config 类:同 config.py,默认 docker 网络下的 redis 服务名 / Config class
- get_logger(name): 统一 logger / unified logger
- WX_APPID / WX_SECRET: 微信小程序登录凭证 / WeChat mini-program login credentials

输入 / Inputs:
- 环境变量:同 config.py + WX_APPID + WX_SECRET / env vars

输出 / Outputs:
- config: Config 单例 / Config singleton
- logger: 日志 / logger

数据流 / Data Flow:
1. 实例化时读环境变量 / Read env on init
2. 校验 WX_APPID / WX_SECRET 是否配置 / Validate WeChat credentials
3. 被 Docker 容器中各模块使用 / Used by modules in Docker

相关文件 / Related Files:
- 调用 / Calls: os.getenv、logging
- 被调用 / Called by: server.py(Docker 模式)、tasks_docker.py、wx-login 路由

使用示例 / Usage Example:
    # 在 docker-compose 中设置环境变量:
    environment:
      - REDIS_HOST=redis
      - WX_APPID=wx...
      - WX_SECRET=...
    from config_docker import config

作者 / Author: 项目组 / Project Team
版本 / Version: 1.0
"""

"""
Configuration management for RGCNFormer backend.

Loads settings from environment variables with sensible defaults.
"""
import os
import logging
from typing import Dict

from config_file import load_backend_config


BACKEND_CONFIG_FILE = load_backend_config()


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

        # ReID body-heatmap visualization (CPU only)
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.REID_CHECKPOINT_PATH = os.getenv(
            'REID_CHECKPOINT_PATH',
            os.path.abspath(os.path.join(project_root, '..', 'outputs', 'best.pt')),
        )
        self.REID_DATA_ROOT = os.getenv(
            'REID_DATA_ROOT',
            os.path.abspath(os.path.join(project_root, '..', 'SYSU-MM01')),
        )
        self.REID_DEVICE = os.getenv('REID_DEVICE', 'cpu')
        self.REID_BATCH_SIZE = 4
        self.REID_HEATMAP_SIZE = (9, 5)
        self.REID_DISPLAY_SIZE = 256
        self.REID_TORCH_NUM_THREADS = int(os.getenv('REID_TORCH_NUM_THREADS', '4'))
        self.REID_CACHE_SIZE = int(os.getenv('REID_CACHE_SIZE', '8'))
        if self.REID_DEVICE.lower() != 'cpu':
            raise ValueError('ReID visualization only supports REID_DEVICE=cpu')

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

        # Model Comparison CSV Configuration
        self.MODEL_COMPARISON_CSV_DIR = os.getenv(
            'MODEL_COMPARISON_CSV_DIR',
            os.path.join(os.path.dirname(__file__), 'data')
        )
        self.MODEL_COMPARISON_FILES = {
            'DCPRES': 'DCPRES_cls.csv',
            'SCDGC': 'SCDGC_cls.csv',
            'DSCPS': 'DSCPS_cls.csv',
        }

        # Dataset Comparison Excel File
        self.DATASET_COMPARISON_XLSX = os.getenv(
            'DATASET_COMPARISON_XLSX',
            os.path.join(os.path.dirname(__file__), 'data', 'CORA-CITE-AMAP-BAT-EAT.xlsx')
        )

        # UMAP Visualization Data Paths
        self.UMAP_DATA_PATH = os.getenv(
            'UMAP_DATA_PATH',
            os.path.join(os.path.dirname(__file__), 'data', 'umap_human_data.json')
        )
        self.UMAP_CORA_DATA_PATH = os.getenv(
            'UMAP_CORA_DATA_PATH',
            os.path.join(os.path.dirname(__file__), 'data', 'umap_cora_data.json')
        )

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
