"""
中文：Flask 应用工厂模块。负责创建应用实例、初始化 Redis/模型连接并注册路由蓝图。
English: Flask application factory. Creates the app instance, initializes Redis/model connections, and registers route blueprints.
"""
import json
import redis
import torch
from flask import Flask
from flask_cors import CORS

from mrmodn_backend.core.config import config, get_logger
from mrmodn_backend.models.runtime import load_model
from mrmodn_backend.api.health import health_bp
from mrmodn_backend.api.wechat import wechat_bp
from mrmodn_backend.api.prediction import prediction_bp
from mrmodn_backend.api.explainability import explainability_bp
from mrmodn_backend.api.datasets import datasets_bp
from mrmodn_backend.api.comparison import comparison_bp
from mrmodn_backend.api.attention import attention_bp
from mrmodn_backend.api.reid import reid_bp


def create_app():
    """
    中文：创建并配置 Flask 应用实例，包括 Redis、模型加载和蓝图注册。
    English: Create and configure the Flask application, including Redis, model loading, and blueprint registration.
    """
    app = Flask(__name__)
    CORS(app)

    logger = get_logger('app')

    # Store config in app
    app.config['MODEL_CHECKPOINT_PATH'] = config.MODEL_CHECKPOINT_PATH

    # Setup Redis
    try:
        redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True
        )
        redis_client.ping()
        logger.info(f"Connected to Redis at {config.REDIS_HOST}:{config.REDIS_PORT}")
    except Exception as e:
        logger.warning(f"Redis not available: {e}")
        redis_client = None

    app.config['REDIS_CLIENT'] = redis_client

    # Load model
    model, device, model_cfg = load_model()
    app.config['MODEL'] = model
    app.config['MODEL_DEVICE'] = device
    app.config['MODEL_CFG'] = model_cfg

    # Load model config JSON
    with open(config.MODEL_CONFIG_PATH, 'r') as f:
        model_config = json.load(f)
    app.config['MODEL_CONFIG_JSON'] = model_config

    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(wechat_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(explainability_bp)
    app.register_blueprint(datasets_bp)
    app.register_blueprint(comparison_bp)
    app.register_blueprint(attention_bp)
    app.register_blueprint(reid_bp)

    return app
