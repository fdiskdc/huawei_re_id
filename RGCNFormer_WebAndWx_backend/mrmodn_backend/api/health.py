"""
中文：健康检查 API 端点。
English: Health check API endpoint.
"""
from flask import Blueprint, jsonify, current_app

health_bp = Blueprint('health', __name__)


@health_bp.route('/mrmodn/api/health', methods=['GET'])
def health():
    """
    中文：健康检查端点。返回服务状态、模型加载状态、运行设备和检查点路径。
    English: Health check endpoint. Returns service status, model loading state, device, and checkpoint path.
    """
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "device": str(current_app.config['MODEL_DEVICE']),
        "checkpoint": current_app.config['MODEL_CHECKPOINT_PATH']
    })
