"""
server.py - Flask HTTP API 入口 / Flask HTTP API entry

RGCNFormer 可视化系统后端的 HTTP 入口。提供 RNA 序列提交、单条/批量推理、异步
任务查询、Integrated Gradients 归因、UMAP 嵌入、注意力可视化等接口;通过 Celery
与 Redis 协调多 worker 长任务。 / Flask entry for RGCNFormer backend. Provides RNA
sequence submission, single/batch inference, async task polling, IG attribution,
UMAP embedding, and attention viz endpoints; coordinates long-running tasks across
workers via Celery + Redis.

功能模块 / Modules:
- Flask app 初始化(日志、CORS、Redis、模型)/ Flask app init (logging, CORS, Redis, model)
- 同步推理接口(/api/v1/submit-task, /api/v1/wx-submit-task) / Sync inference
- 异步任务轮询(/api/v1/get-result, /api/v1/wx-get-result) / Async result polling
- 可视化接口(/api/v1/ig, /api/v1/umap, attention) / Viz endpoints
- 微信登录(/api/v1/wx-login) / WeChat login

输入 / Inputs:
- HTTP 请求:JSON body 含 sequence(s)、taskType 等 / HTTP requests with JSON body
- 环境变量:.env 提供 REDIS_HOST / MODEL_PATH / LINEARFOLD_PATH / env vars from .env

输出 / Outputs:
- JSON 响应:jobId、分类概率、注意力权重、IG 归因、UMAP 坐标 / JSON responses

数据流 / Data Flow:
1. 启动时连接 Redis,加载 RNA_ClassQuery_Model 与 LinearFold 路径 / Connect Redis, load model
2. 接收请求 → 校验序列 → 走同步推理或投递 Celery 任务 / Validate → sync or async
3. 同步路径: 实时返回结果;异步路径: 返回 jobId,前端轮询 / Sync returns result; async returns jobId
4. 异步任务完成时把结果写入 Redis,轮询接口读取 / Async writes to Redis, polling reads

相关文件 / Related Files:
- 调用 / Calls: main_model.RNA_ClassQuery_Model、human.run_linearfold、common、tasks、config
- 被调用 / Called by: Gunicorn(wsgi.py)、Cluster_WebAndWx_WxFrontend、RGCNFormer_WebAndWx_WebFrontend

使用示例 / Usage Example:
    gunicorn -c gunicorn.conf.py wsgi:app
    curl -X POST http://localhost:8000/api/v1/submit-task -H 'Content-Type: application/json' -d '{"sequence":"ACGU..."}'

作者 / Author: 项目组 / Project Team
版本 / Version: 1.0
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import csv
import os
import redis
import hashlib
import uuid
import time
import numpy as np
import torch
import requests
from torch_geometric.data import Batch
from captum.attr import IntegratedGradients
from celery.result import AsyncResult

# Import local modules
from main_model import RNA_ClassQuery_Model
from human import run_linearfold, build_edge_index_from_structure
from common import INDEX_TO_NUCLEOTIDE
from attention_distribution import attention_distribution_cache_key
from tasks import celery_app, run_prediction_task
from config import config, get_logger
from mrmodn_backend.api.reid import reid_bp

# Initialize logger
logger = get_logger('server')

# ============================================================================
# Redis Connection (Shared Storage for Multi-Worker Setup)
# ============================================================================
try:
    redis_client = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        db=config.REDIS_DB,
        decode_responses=True
    )
    # Test connection
    redis_client.ping()
    logger.info(f"Connected to Redis successfully at {config.REDIS_HOST}:{config.REDIS_PORT}")
except Exception as e:
    logger.warning(f"Could not connect to Redis: {e}")
    logger.warning("Falling back to in-memory storage (not suitable for multi-worker gunicorn)")
    redis_client = None


def one_hot_encode_sequence(sequence: str) -> np.ndarray:
    """
    Convert RNA sequence string to one-hot encoding.

    Args:
        sequence: RNA sequence string (A, C, G, U)

    Returns:
        One-hot encoded array of shape (len(sequence), 4)
    """
    one_hot_mapping = {
        'A': [1., 0., 0., 0.],
        'C': [0., 1., 0., 0.],
        'G': [0., 0., 1., 0.],
        'U': [0., 0., 0., 1.],
        'T': [0., 0., 0., 1.],  # Treat T as U
        'N': [0., 0., 0., 0.]
    }

    one_hot = np.zeros((len(sequence), 4), dtype=np.float32)
    for i, nucleotide in enumerate(sequence.upper()):
        if nucleotide in one_hot_mapping:
            one_hot[i] = one_hot_mapping[nucleotide]
        else:
            one_hot[i] = [0., 0., 0., 0.]  # Unknown nucleotide

    return one_hot

# 初始化 Flask app
app = Flask(__name__)
# 设置 CORS，允许来自前端（例如 http://localhost:5173）的跨域请求
CORS(app)
app.register_blueprint(reid_bp)

# ============================================================================
# Global Model Loading (Load once at startup)
# ============================================================================

logger.info("Loading model and configuration...")

# Load configuration
with open(config.MODEL_CONFIG_PATH, 'r') as f:
    model_config = json.load(f)

model_cfg = model_config['model']

# Initialize model
device = config.MODEL_DEVICE
logger.info(f"Using device: {device}")

model = RNA_ClassQuery_Model(
    cnn_hidden_dim=model_cfg['cnn_hidden_dim'],
    cnn_kernel_sizes=tuple(model_cfg['cnn_kernel_sizes']),
    cnn_dropout=model_cfg['cnn_dropout'],
    gcn_hidden_dim=model_cfg['gcn_hidden_dim'],
    gcn_out_channels=model_cfg['gcn_out_channels'],
    gcn_num_layers=model_cfg['gcn_num_layers'],
    gcn_dropout=model_cfg['gcn_dropout'],
    num_classes=model_cfg['num_classes'],
    num_attn_heads=model_cfg['num_attn_heads'],
    attn_dropout=model_cfg['attn_dropout'],
    use_simple_pooling=model_cfg['use_simple_pooling'],
    use_hierarchical=model_cfg['use_hierarchical'],
    use_layer_norm=model_cfg['use_layer_norm']
)

# Load model weights
checkpoint_path = config.MODEL_CHECKPOINT_PATH
checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

logger.info(f"Model loaded successfully from {checkpoint_path}")
logger.info("Model is ready for predictions!")

# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "device": str(device),
        "checkpoint": checkpoint_path
    })

# ============================================================================
# Sample Sequence Endpoint (for Workspace Input Block)
# ============================================================================

_sample_sequences = None

def _load_sample_sequences():
    global _sample_sequences
    if _sample_sequences is None:
        try:
            data_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_sequences.json')
            with open(data_path, 'r') as f:
                _sample_sequences = json.load(f)['sequences']
        except Exception as e:
            logger.warning(f"Failed to load sample sequences: {e}")
            _sample_sequences = []
    return _sample_sequences

@app.route('/api/v1/sample-sequence', methods=['GET'])
def get_sample_sequence():
    """Return a random sample sequence for workspace input block."""
    sequences = _load_sample_sequences()
    if not sequences:
        return jsonify({'sequence': ''})
    import random
    seq = random.choice(sequences)
    return jsonify({'sequence': seq})

# ============================================================================
# WeChat Mini Program Login Endpoint
# ============================================================================

@app.route('/api/v1/wx/login', methods=['POST'])
def wx_login():
    """
    WeChat Mini Program login endpoint.
    Expects JSON body with 'loginCode' key.
    Returns openid after exchanging code with WeChat API.
    """
    # Get data from request
    data = request.get_json()
    
    # Validate input
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    login_code = data.get('loginCode')
    nickname = data.get('nickname')
    avatar_url = data.get('avatarUrl')
    
    if not login_code:
        return jsonify({"error": "loginCode is required"}), 400
    
    # Check if WeChat app credentials are configured
    if not config.WX_APPID or not config.WX_SECRET:
        logger.error("WeChat app credentials not configured")
        return jsonify({
            "error": "WeChat app credentials not configured",
            "detail": "Please set WX_APPID and WX_SECRET environment variables"
        }), 500
    
    logger.info(f"Processing WeChat login request for code: {login_code[:10]}...")
    
    try:
        # Call WeChat API to exchange code for openid and session_key
        params = {
            'appid': config.WX_APPID,
            'secret': config.WX_SECRET,
            'js_code': login_code,
            'grant_type': 'authorization_code'
        }
        
        response = requests.get(config.WX_LOGIN_URL, params=params, timeout=10)
        response_data = response.json()
        
        # Check if WeChat API returned an error
        if 'errcode' in response_data:
            error_msg = response_data.get('errmsg', 'Unknown WeChat API error')
            logger.error(f"WeChat API error: {response_data.get('errcode')} - {error_msg}")
            return jsonify({
                "error": f"WeChat API error: {response_data.get('errcode')}",
                "detail": error_msg
            }), 400
        
        # Extract openid and session_key
        openid = response_data.get('openid')
        session_key = response_data.get('session_key')
        
        if not openid:
            logger.error("WeChat API did not return openid")
            return jsonify({
                "error": "Failed to get openid from WeChat API",
                "detail": response_data
            }), 500
        
        # Check if user already exists in Redis
        # Use nickname and avatarUrl from frontend if provided
        user = {
            'openid': openid,
            'session_key': session_key,
            'nickname': nickname,
            'avatarUrl': avatar_url
        }
        
        if redis_client:
            try:
                user_key = f"wx_user:{openid}"
                user_data = redis_client.get(user_key)
                
                if user_data:
                    existing_user = json.loads(user_data)
                    logger.info(f"Existing user found: {openid}")
                    # Update session_key for security
                    # Use provided nickname/avatar from frontend, or keep existing if not provided
                    user['session_key'] = session_key
                    if nickname is not None:
                        user['nickname'] = nickname
                    else:
                        user['nickname'] = existing_user.get('nickname')
                    if avatar_url is not None:
                        user['avatarUrl'] = avatar_url
                    else:
                        user['avatarUrl'] = existing_user.get('avatarUrl')
                    redis_client.setex(
                        user_key,
                        30 * 24 * 3600,  # 30 days in seconds
                        json.dumps(user, ensure_ascii=False)
                    )
                else:
                    # Store new user data in Redis with 30 days TTL
                    redis_client.setex(
                        user_key,
                        30 * 24 * 3600,  # 30 days in seconds
                        json.dumps(user, ensure_ascii=False)
                    )
                    logger.info(f"New user created: {openid}")
            except Exception as e:
                logger.error(f"Redis error during user storage: {e}")
                # Continue with user object even if Redis fails
        
        # Prepare user info for response (exclude session_key for security)
        user_info = {
            'openid': user['openid'],
            'nickname': user.get('nickname'),
            'avatarUrl': user.get('avatarUrl')
        }
        
        # Return success response with user info
        return jsonify({
            "code": 0,
            "openid": openid,
            "data": user_info,
            "message": "Login successful"
        }), 200
        
    except requests.exceptions.Timeout:
        logger.error("WeChat API request timeout")
        return jsonify({
            "error": "WeChat API request timeout",
            "detail": "Failed to connect to WeChat server"
        }), 504
    except requests.exceptions.RequestException as e:
        logger.error(f"WeChat API request error: {e}")
        return jsonify({
            "error": "Failed to call WeChat API",
            "detail": str(e)
        }), 500
    except Exception as e:
        import traceback
        error_msg = f"WeChat login error: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500

# ============================================================================
# WeChat Mini Program Batch Task Submission Endpoint (Refactored)
# ============================================================================

@app.route('/api/v1/wx-submit-task', methods=['POST'])
def wx_submit_task():
    """
    Submit up to 5 prediction tasks for asynchronous processing with progress tracking.
    Designed for WeChat mini-program to handle multiple RNA sequences.
    Returns a batch job_id for tracking progress.
    Empty sequences are skipped (not processed).
    
    Request body format:
    {
        "rnaSequence1": "sequence1",
        "rnaSequence2": "sequence2",
        "rnaSequence3": "sequence3",
        "rnaSequence4": "sequence4",
        "rnaSequence5": "sequence5",
        "targetClassId": 0,  // Optional
        "topK": 10           // Optional
    }
    
    Response format:
    {
        "code": 200,
        "message": "任务已提交",
        "data": {
            "job_id": "your-generated-uuid"
        }
    }
    """
    # Import process_sequence_in_batch task
    from tasks import process_sequence_in_batch
    
    # Get data from request JSON body
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Extract sequences
    sequences = []
    for i in range(1, 6):
        seq_key = f'rnaSequence{i}'
        sequence = data.get(seq_key, '')
        sequences.append(sequence)
    
    # Extract optional parameters
    target_class_id = data.get('targetClassId')
    top_k = data.get('topK')
    
    # Validate: at least one non-empty sequence
    non_empty_sequences = [(i, seq) for i, seq in enumerate(sequences) if seq and seq.strip()]
    if not non_empty_sequences:
        return jsonify({"error": "At least one non-empty RNA sequence is required"}), 400
    
    logger.info(f"Received WeChat mini-program task submission with {len(non_empty_sequences)} valid sequences")
    
    # Generate unique batch job_id
    batch_job_id = str(uuid.uuid4())
    logger.info(f"Generated batch job_id: {batch_job_id}")
    
    # Initialize Redis state for this batch job
    if redis_client:
        try:
            redis_key = f'batch_job:{batch_job_id}'
            redis_client.hset(redis_key, 'status', 'PENDING')
            redis_client.hset(redis_key, 'total_sequences', str(len(non_empty_sequences)))
            redis_client.hset(redis_key, 'completed_sequences', '0')
            redis_client.hset(redis_key, 'results', json.dumps([]))
            redis_client.hset(redis_key, 'creation_time', str(int(time.time())))
            # Set TTL for batch job (24 hours)
            redis_client.expire(redis_key, 86400)
            logger.info(f"Initialized Redis state for batch {batch_job_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Redis state for batch {batch_job_id}: {e}")
            return jsonify({"error": "Failed to initialize batch job"}), 500
    
    # Submit Celery tasks for each non-empty sequence
    for index, sequence in non_empty_sequences:
        try:
            process_sequence_in_batch.apply_async(
                args=[batch_job_id, sequence, index, target_class_id, top_k]
            )
            logger.info(f"Submitted sequence {index} for batch {batch_job_id}")
        except Exception as e:
            import traceback
            logger.error(f"Failed to submit sequence {index} for batch {batch_job_id}: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            # Continue with other sequences even if one fails
    
    # Return batch job_id to client
    response = {
        "code": 200,
        "message": "任务已提交",
        "data": {
            "job_id": batch_job_id
        }
    }
    
    logger.info(f"WeChat mini-program batch task submitted successfully. Batch ID: {batch_job_id}")
    
    return jsonify(response), 202


# ============================================================================
# WeChat Mini Program Task Progress Polling Endpoint
# ============================================================================

@app.route('/api/v1/wx-task-progress/<job_id>', methods=['GET'])
def wx_task_progress(job_id):
    """
    Retrieve the progress of a batch task by job_id.
    Returns status, progress, and completed results.
    
    Response format:
    {
        "code": 200,
        "message": "成功",
        "data": {
            "job_id": "your-uuid",
            "status": "PROCESSING",
            "total_sequences": 5,
            "completed_sequences": 2,
            "results": [
                { "index": 0, "jobId": "...", "data": "..." },
                { "index": 1, "jobId": "...", "data": "..." }
            ]
        }
    }
    """
    if not redis_client:
        return jsonify({"error": "Redis not available"}), 500
    
    try:
        redis_key = f'batch_job:{job_id}'
        
        # Check if job_id exists in Redis
        if not redis_client.exists(redis_key):
            return jsonify({
                "code": 404,
                "message": "任务不存在",
                "error": "Job not found"
            }), 404
        
        # Get all fields from Redis hash
        job_data = redis_client.hgetall(redis_key)
        
        # Parse data
        status = job_data.get('status', 'UNKNOWN')
        total_sequences = int(job_data.get('total_sequences', 0))
        completed_sequences = int(job_data.get('completed_sequences', 0))
        results_json = job_data.get('results', '[]')
        results = json.loads(results_json) if results_json else []
        
        # Prepare response data
        data = {
            "job_id": job_id,
            "status": status,
            "total_sequences": total_sequences,
            "completed_sequences": completed_sequences,
            "results": results
        }
        
        logger.info(f"Retrieved progress for batch {job_id}: {completed_sequences}/{total_sequences} completed")
        
        response = {
            "code": 200,
            "message": "成功",
            "data": data
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        import traceback
        logger.error(f"Error retrieving task progress: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "code": 500,
            "message": "获取进度失败",
            "error": str(e)
        }), 500
@app.route('/api/v1/submit-task', methods=['POST'])
def submit_task():
    """
    Submit a prediction task for asynchronous processing.
    Returns immediately with a job_id and 'pending' status.
    The actual prediction runs in the background via Celery.
    """
    # Get data from request JSON body
    data = request.get_json()
    user_id = ""
    original_sequence = ""
    target_class_id = None
    top_k = None
    if data:
        user_id = data.get('userId', '')
        original_sequence = data.get('rnaSequence', '')
        target_class_id = data.get('targetClassId')  # Optional: specific class to visualize
        top_k = data.get('topK')  # Optional: number of top sites to display
        # Print logs for debugging
        logger.info(f"Received user_id: {user_id}, target_class_id: {target_class_id}, top_k: {top_k}")
        logger.info(f"Received sequence from frontend: {original_sequence[:50]}... (length: {len(original_sequence)})")
        if target_class_id is not None:
            logger.info(f"Target class ID: {target_class_id}")
        if top_k is not None:
            logger.info(f"Top-K value: {top_k}")

    # Validate sequence
    if not original_sequence:
        return jsonify({"error": "No sequence provided"}), 400

    # Step 1: Generate SHA256 hash of the RNA sequence as job_id
    job_id = hashlib.sha256(original_sequence.encode('utf-8')).hexdigest()
    logger.info(f"Generated job_id (SHA256 hash): {job_id}")

    # Step 2: Check Redis cache for existing result
    if redis_client:
        try:
            cached_result = redis_client.get(f"task:{job_id}")
            if cached_result:
                logger.info(f"Cache HIT for job_id: {job_id}")
                # Deserialize and return cached result immediately
                result = json.loads(cached_result)
                # Update jobId to match the hash
                result["jobId"] = job_id
                result["status"] = "completed"
                return jsonify(result), 200
            else:
                logger.info(f"Cache MISS for job_id: {job_id}")
        except Exception as e:
            logger.error(f"Redis cache error: {e}")
            # Continue with task submission if Redis fails
    else:
        logger.warning("Redis not available, skipping cache check")

    # Step 3: Submit Celery task for background processing
    try:
        # Submit the prediction task to Celery
        # Use the sequence hash as the task_id for consistency
        run_prediction_task.apply_async(
            args=[original_sequence, target_class_id, top_k],
            task_id=job_id
        )

        logger.info(f"Task {job_id} submitted to Celery for background processing")

        # Return immediately with 202 Accepted
        response = {
            "jobId": job_id,
            "status": "pending"
        }
        return jsonify(response), 202

    except Exception as e:
        import traceback
        error_msg = f"Failed to submit task: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500

# ============================================================================
# Results Retrieval Endpoint
# ============================================================================

@app.route('/api/v1/results/<job_id>', methods=['GET'])
def get_result(job_id):
    """
    Retrieve the result of a previously submitted task by job_id.
    First checks Redis cache for completed results.
    If not found, queries Celery result backend for task status.
    Returns:
    - 200 with full result if task completed and cached
    - 200 with {"status": "processing"} if task is still running
    - 200 with {"status": "failed", "error": "..."} if task failed
    - 404 if task not found
    """
    try:
        # Step 1: Check Redis cache first (fastest path)
        if redis_client:
            result_json = redis_client.get(f"task:{job_id}")
            if result_json:
                result = json.loads(result_json)
                return jsonify(result), 200
            logger.info(f"Cache MISS for job_id: {job_id} in results endpoint")

        # Step 2: Check Celery task status
        task = AsyncResult(job_id, app=celery_app)

        if task.state == 'PENDING':
            # Task is waiting to be processed or currently processing
            logger.info(f"Task {job_id} status: PENDING (processing)")
            return jsonify({
                "jobId": job_id,
                "status": "processing"
            }), 200
        elif task.state == 'STARTED':
            # Task is currently being processed (if task_track_started=True)
            logger.info(f"Task {job_id} status: STARTED (processing)")
            return jsonify({
                "jobId": job_id,
                "status": "processing"
            }), 200
        elif task.state == 'SUCCESS':
            # Task completed successfully - result should be in Redis by now
            # If we reach here, it means the result wasn't in Redis but task is done
            # This can happen if Redis caching failed in the task
            logger.info(f"Task {job_id} status: SUCCESS but not in cache")
            result = task.result
            if isinstance(result, dict):
                # Try to cache it now for future requests
                if redis_client:
                    try:
                        result_json = json.dumps(result, ensure_ascii=False)
                        redis_client.setex(f"task:{job_id}", config.REDIS_CACHE_TTL, result_json)
                    except Exception as e:
                        logger.error(f"Failed to cache result in Redis: {e}")
                return jsonify(result), 200
            else:
                return jsonify({"error": "Invalid result format"}), 500
        elif task.state == 'FAILURE':
            # Task failed with an exception
            logger.info(f"Task {job_id} status: FAILURE")
            error_info = task.info

            # Extract structured error information
            error_response = {
                "jobId": job_id,
                "status": "failed"
            }

            if isinstance(error_info, dict):
                # Check for structured error info from TaskError
                if 'error_info' in error_info:
                    structured_error = error_info['error_info']
                    error_response["error"] = structured_error.get('error_message', 'Unknown error')
                    error_response["errorType"] = structured_error.get('error_type', 'Unknown')
                    error_response["step"] = structured_error.get('step', 'unknown')
                else:
                    # Fallback for standard Celery error format
                    error_response["error"] = error_info.get('message', str(error_info))
                    error_response["errorType"] = error_info.get('error_type', type(error_info).__name__)
            else:
                # String error format
                error_response["error"] = str(error_info)
                error_response["errorType"] = "Unknown"

            return jsonify(error_response), 200
        elif task.state == 'RETRY':
            # Task is being retried
            logger.info(f"Task {job_id} status: RETRY")
            return jsonify({
                "jobId": job_id,
                "status": "processing"
            }), 200
        else:
            # Unknown state
            logger.info(f"Task {job_id} status: {task.state}")
            return jsonify({
                "jobId": job_id,
                "status": "unknown",
                "state": task.state
            }), 200

    except Exception as e:
        logger.error(f"Error retrieving job result: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({"error": "Failed to retrieve result"}), 500

# ============================================================================
# Integrated Gradients Endpoint
# ============================================================================

def extract_module_info(module, module_name=""):
    """
    Recursively extract information from a PyTorch nn.Module hierarchy.
    
    Args:
        module: PyTorch nn.Module object
        module_name: Name of the module (for display)
    
    Returns:
        Dictionary containing module structure information
    """
    # Get class name
    class_name = module.__class__.__name__
    
    # Build result dictionary
    result = {
        "name": module_name,
        "class": class_name,
        "details": "",
        "children": []
    }
    
    # Extract relevant details based on module type
    if class_name == "ParallelCNNBlock":
        details = f"Kernels: {module.kernel_sizes}, Hidden: {module.hidden_dim}"
        result["details"] = details
        
        # Add convolution branches as children
        for i, conv in enumerate(module.conv_branches):
            conv_info = {
                "name": f"conv_{i}",
                "class": "Conv1d",
                "details": f"kernel_size={module.kernel_sizes[i]}, out_channels={module.hidden_dim}",
                "children": []
            }
            result["children"].append(conv_info)
            
    elif class_name == "GCNBlock":
        details = f"Layers: {module.num_layers}, Hidden: {module.hidden_dim}, Out: {module.out_channels}"
        result["details"] = details
        
        # Add GCN layers as children
        for i, gcn_layer in enumerate(module.gcn_layers):
            gcn_info = {
                "name": f"gcn_layer_{i}",
                "class": "GCNConv",
                "details": f"layer_{i}",
                "children": []
            }
            result["children"].append(gcn_info)
            
    elif class_name == "ClassQueryHead":
        details = f"Classes: {module.num_classes}, Heads: {module.num_heads if hasattr(module, 'num_heads') else 'N/A'}"
        result["details"] = details
        
        # Add class queries as children
        for i in range(module.num_classes):
            query_info = {
                "name": f"class_query_{i}",
                "class": "LearnableQuery",
                "details": f"class_{i}",
                "children": []
            }
            result["children"].append(query_info)
            
    elif class_name == "ClassQueryHeadPooling":
        details = f"Classes: {module.num_classes}"
        result["details"] = details
        
        # Add class queries as children
        for i in range(module.num_classes):
            query_info = {
                "name": f"class_query_{i}",
                "class": "LearnableQuery",
                "details": f"class_{i}",
                "children": []
            }
            result["children"].append(query_info)
            
    elif class_name == "HierarchicalClassQueryHeadPooling":
        details = f"Classes: {module.num_classes}, Groups: {module.num_groups}"
        result["details"] = details
        
        # Add group queries as children
        for i, group_name in enumerate(module.group_names):
            group_info = {
                "name": f"group_query_{i}",
                "class": "GroupQuery",
                "details": f"{group_name} group",
                "children": []
            }
            
            # Add derived class queries
            if group_name in module.group_to_class_indices:
                class_indices = module.group_to_class_indices[group_name]
                for j, class_idx in enumerate(class_indices):
                    class_info = {
                        "name": f"class_query_{class_idx}",
                        "class": "DerivedClassQuery",
                        "details": f"class_{class_idx}",
                        "children": []
                    }
                    group_info["children"].append(class_info)
            
            result["children"].append(group_info)
    
    # Recursively process named children
    for name, child in module.named_children():
        if not child.__class__.__name__ in ["Sequential", "ModuleList", "ModuleDict"]:
            child_info = extract_module_info(child, name)
            result["children"].append(child_info)
    
    return result


@app.route('/api/v1/model-architecture', methods=['GET'])
def get_model_architecture():
    """
    Get the hierarchical structure of the RNA_ClassQuery_Model.
    Returns a JSON representation of the model's architecture.
    """
    try:
        logger.info("Extracting model architecture...")
        
        # Extract model structure
        model_info = extract_module_info(model, "RNA_ClassQuery_Model")
        
        logger.info("Model architecture extracted successfully")
        
        return jsonify(model_info), 200
        
    except Exception as e:
        import traceback
        error_msg = f"Failed to extract model architecture: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500


@app.route('/api/v1/results/<job_id>/attention-distribution', methods=['GET'])
def get_cached_attention_distribution(job_id):
    """Return the complete per-class attention distribution cached by a task."""
    if not redis_client:
        return jsonify({"error": "Redis not available"}), 503

    try:
        result_json = redis_client.get(attention_distribution_cache_key(job_id))
        if not result_json:
            return jsonify({
                "error": "Attention distribution not found",
                "message": "This task does not contain a cached attention distribution.",
            }), 404

        result = json.loads(result_json)
        predicted_only = request.args.get("predictedOnly", "true").lower() != "false"
        if predicted_only:
            result["classes"] = [
                class_data
                for class_data in result.get("classes", [])
                if class_data.get("is_predicted")
            ]

        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Failed to retrieve attention distribution for {job_id}: {e}")
        return jsonify({
            "error": "Failed to retrieve attention distribution",
            "detail": str(e),
        }), 500


@app.route('/api/v1/model-graph', methods=['GET'])
def get_model_graph():
    """
    Get the ONNX model computation graph (nodes and edges).
    Returns the model graph data from the JSON file.
    """
    try:
        logger.info("Loading model graph data...")
        
        # Read model graph from JSON file
        model_graph_path = 'json/model_graph.json'
        
        with open(model_graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        logger.info(f"Model graph loaded successfully: {len(graph_data.get('nodes', []))} nodes, {len(graph_data.get('edges', []))} edges")
        
        return jsonify(graph_data), 200
        
    except FileNotFoundError:
        logger.error(f"Model graph file not found: {model_graph_path}")
        return jsonify({
            "error": "Model graph file not found",
            "detail": f"The file {model_graph_path} does not exist. Please generate the model graph first.",
            "type": "FileNotFoundError"
        }), 404
    except Exception as e:
        import traceback
        error_msg = f"Failed to load model graph: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500


@app.route('/api/v1/integrated-gradients', methods=['POST'])
def integrated_gradients():
    """
    Compute Integrated Gradients attributions for RNA sequence prediction.
    """
    # Get data from request
    data = request.get_json()
    original_sequence = data.get('rnaSequence', '')
    target_class_id = data.get('targetClassId')
    
    # Validate inputs
    if not original_sequence:
        return jsonify({"error": "No sequence provided"}), 400
    
    if target_class_id is None or not (0 <= target_class_id < 12):
        return jsonify({"error": "Invalid targetClassId. Must be between 0 and 11"}), 400
    
    logger.info(f"Integrated Gradients: sequence length={len(original_sequence)}, target_class_id={target_class_id}")
    
    # Store original sequence for response
    sequence = original_sequence
    
    # For shorter sequences, pad to 1001; for longer sequences, truncate
    TARGET_LENGTH = 1001
    seq_len = len(sequence)
    
    # Track padding/trimming for index remapping
    left_padding = 0
    left_trimming = 0
    
    if seq_len != TARGET_LENGTH:
        if seq_len < TARGET_LENGTH:
            padding_needed = TARGET_LENGTH - seq_len
            left_pad = padding_needed // 2
            right_pad = padding_needed - left_pad
            left_padding = left_pad
            sequence = 'N' * left_pad + sequence + 'N' * right_pad
        else:
            excess = seq_len - TARGET_LENGTH
            left_trim = excess // 2
            right_trim = excess - left_trim
            left_trimming = left_trim
            sequence = sequence[left_trim:seq_len - right_trim]
    
    try:
        # Step 1: Call LinearFold to get secondary structure
        structures = run_linearfold([sequence])
        structure = structures[0]
        
        # Step 2: Build edge index from structure
        edge_index = build_edge_index_from_structure(sequence, structure)
        
        # Step 3: Prepare model input
        x = one_hot_encode_sequence(sequence)
        x = torch.FloatTensor(x)  # Shape: [1001, 4]
        
        # Create batch tensor (single sample)
        batch = torch.zeros(len(sequence), dtype=torch.long)
        
        # Step 4: Create Batch object
        data_batch = Batch(x=x, edge_index=edge_index, batch=batch)
        data_batch = data_batch.to(device)
        
        # Step 5: Compute Integrated Gradients
        # Check if model is hierarchical
        if model_cfg['use_hierarchical']:
            # For hierarchical model, wrap the model to extract 12-class logits
            def forward_func(x, edge_index, batch):
                output = model(x, edge_index, batch)
                # output is a tuple: (logits_12class, logits_4class, attn_weights)
                logits_12class = output[0]  # Extract 12-class logits
                return logits_12class
        else:
            # For non-hierarchical model, use model directly
            def forward_func(x, edge_index, batch):
                output = model(x, edge_index, batch)
                return output
        
        # Instantiate IntegratedGradients
        ig = IntegratedGradients(forward_func)
        
        # Compute attributions
        with torch.enable_grad():
            # Baseline: zero tensor of same shape
            baseline = torch.zeros_like(x)
            
            # Compute attributions
            attributions = ig.attribute(
                x.unsqueeze(0),  # Add batch dimension
                baselines=baseline.unsqueeze(0),
                target=target_class_id,
                additional_forward_args=(edge_index, batch),
                internal_batch_size=1
            )
        
        # Step 6: Process attributions
        attributions = attributions.squeeze(0)  # Remove batch dimension [1001, 4]
        
        # Sum attributions across the one-hot encoding dimension to get per-nucleotide score
        node_attributions = attributions.sum(dim=1).cpu().numpy()  # [1001]
        
        # Step 7: Build GCN graph data with attribution scores
        edge_index_np = edge_index.cpu().numpy()
        edges = []
        
        # Calculate the valid range in model coordinates
        valid_start = left_padding
        valid_end = left_padding + len(original_sequence)
        
        # Process all edges
        for i in range(int(edge_index_np.shape[1])):
            source = int(edge_index_np[0, i])
            target = int(edge_index_np[1, i])
            
            # Only process edges within valid range
            if not (0 <= source < len(sequence) and 0 <= target < len(sequence)):
                continue
            
            # Map model indices to original indices
            orig_source = source - left_padding + left_trimming
            orig_target = target - left_padding + left_trimming
            
            # Only include edges within original sequence bounds
            if not (0 <= orig_source < len(original_sequence) and 0 <= orig_target < len(original_sequence)):
                continue
            
            # Only keep one direction (source < target) to avoid duplicates
            if source >= target:
                continue
            
            nuc_source = original_sequence[orig_source]
            nuc_target = original_sequence[orig_target]
            edges.append({
                "source": f"{nuc_source}{orig_source}",
                "target": f"{nuc_target}{orig_target}"
            })
        
        # Create nodes with attribution scores
        nodes = []
        for i in range(len(original_sequence)):
            nuc = original_sequence[i]
            # Map original index to model index
            model_index = i + left_padding - left_trimming
            attribution_score = float(node_attributions[model_index]) if 0 <= model_index < len(node_attributions) else 0.0
            
            nodes.append({
                "id": f"{nuc}{i}",
                "label": f"位置{i}: {nuc}",
                "data": {
                    "index": i,
                    "type": nuc,
                    "name": f"{'腺嘌呤' if nuc == 'A' else '胞嘧啶' if nuc == 'C' else '鸟嘌呤' if nuc == 'G' else '尿嘧啶'}",
                    "attributionScore": attribution_score
                }
            })
        
        # Create a set of valid node IDs for filtering edges
        valid_node_ids = {node["id"] for node in nodes}
        
        # Filter edges to only include those that reference valid nodes
        valid_edges = [
            edge for edge in edges
            if edge["source"] in valid_node_ids and edge["target"] in valid_node_ids
        ]
        
        logger.info(f"Integrated Gradients: 节点数={len(nodes)}, 有效边数={len(valid_edges)}")
        
        # Return response
        return jsonify({
            "nodes": nodes,
            "edges": valid_edges,
            "targetClassId": target_class_id
        }), 200
        
    except Exception as e:
        import traceback
        error_msg = f"Integrated Gradients error: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500


@app.route('/api/v1/visualize-gcn-aggregation', methods=['POST'])
def visualize_gcn_aggregation():
    """
    Visualize GCN message passing for a specific target node.
    Returns aggregation details showing message strengths from neighbors at each GCN layer.
    """
    # Get data from request
    data = request.get_json()
    original_sequence = data.get('rnaSequence', '')
    target_node_idx = data.get('targetNodeIdx')
    
    # Validate inputs
    if not original_sequence:
        return jsonify({"error": "No sequence provided"}), 400
    
    if target_node_idx is None or not isinstance(target_node_idx, int):
        return jsonify({"error": "Invalid targetNodeIdx. Must be an integer"}), 400
    
    logger.info(f"GCN Aggregation Viz: sequence length={len(original_sequence)}, target_node_idx={target_node_idx}")
    
    # Store original sequence for response
    sequence = original_sequence
    
    # For shorter sequences, pad to 1001; for longer sequences, truncate
    TARGET_LENGTH = 1001
    seq_len = len(sequence)
    
    # Track padding/trimming for index remapping
    left_padding = 0
    left_trimming = 0
    
    if seq_len != TARGET_LENGTH:
        if seq_len < TARGET_LENGTH:
            padding_needed = TARGET_LENGTH - seq_len
            left_pad = padding_needed // 2
            right_pad = padding_needed - left_pad
            left_padding = left_pad
            sequence = 'N' * left_pad + sequence + 'N' * right_pad
        else:
            excess = seq_len - TARGET_LENGTH
            left_trim = excess // 2
            right_trim = excess - left_trim
            left_trimming = left_trim
            sequence = sequence[left_trim:seq_len - right_trim]
    
    # Map original target node index to model coordinates
    model_target_idx = target_node_idx + left_padding - left_trimming
    
    # Validate model target index
    if not (0 <= model_target_idx < len(sequence)):
        return jsonify({"error": f"Target node index out of bounds after padding/trimming"}), 400
    
    try:
        # Step 1: Call LinearFold to get secondary structure
        structures = run_linearfold([sequence])
        structure = structures[0]
        
        # Step 2: Build edge index from structure
        edge_index = build_edge_index_from_structure(sequence, structure)
        
        # Step 3: Prepare model input
        x = one_hot_encode_sequence(sequence)
        x = torch.FloatTensor(x)  # Shape: [1001, 4]
        
        # Create batch tensor (single sample)
        batch = torch.zeros(len(sequence), dtype=torch.long)
        
        # Step 4: Create Batch object
        data_batch = Batch(x=x, edge_index=edge_index, batch=batch)
        data_batch = data_batch.to(device)
        
        # Step 5: Run model with aggregation details
        with torch.no_grad():
            output, aggregation_details = model(
                data_batch.x,
                data_batch.edge_index,
                data_batch.batch,
                return_aggregation_details=True,
                target_node_idx=model_target_idx
            )
        
        # Step 6: Process aggregation details
        # Map model indices back to original indices
        processed_aggregation = []
        for layer_data in aggregation_details:
            processed_layer = {
                "layer": layer_data["layer"],
                "messages": []
            }
            
            for msg in layer_data["messages"]:
                # Map model index to original index
                model_from_idx = msg["from"]
                orig_from_idx = model_from_idx - left_padding + left_trimming
                
                # Only include messages within original sequence bounds
                if 0 <= orig_from_idx < len(original_sequence):
                    processed_layer["messages"].append({
                        "from": orig_from_idx,
                        "strength": msg["strength"]
                    })
            
            processed_aggregation.append(processed_layer)
        
        # Step 7: Build graph structure for visualization
        edge_index_np = edge_index.cpu().numpy()
        edges = []
        
        # Calculate the valid range in model coordinates
        valid_start = left_padding
        valid_end = left_padding + len(original_sequence)
        
        # Process all edges
        for i in range(int(edge_index_np.shape[1])):
            source = int(edge_index_np[0, i])
            target = int(edge_index_np[1, i])
            
            # Only process edges within valid range
            if not (valid_start <= source < valid_end and valid_start <= target < valid_end):
                continue
            
            # Map model indices to original indices
            orig_source = source - left_padding + left_trimming
            orig_target = target - left_padding + left_trimming
            
            # Only include edges within original sequence bounds
            if not (0 <= orig_source < len(original_sequence) and 0 <= orig_target < len(original_sequence)):
                continue
            
            # Only keep one direction (source < target) to avoid duplicates
            if orig_source >= orig_target:
                continue
            
            nuc_source = original_sequence[orig_source]
            nuc_target = original_sequence[orig_target]
            edges.append({
                "source": f"{nuc_source}{orig_source}",
                "target": f"{nuc_target}{orig_target}"
            })
        
        # Create nodes
        nodes = []
        for i in range(len(original_sequence)):
            nuc = original_sequence[i]
            nodes.append({
                "id": f"{nuc}{i}",
                "label": f"位置{i}: {nuc}",
                "data": {
                    "index": i,
                    "type": nuc,
                    "name": f"{'腺嘌呤' if nuc == 'A' else '胞嘧啶' if nuc == 'C' else '鸟嘌呤' if nuc == 'G' else '尿嘧啶'}"
                }
            })
        
        logger.info(f"GCN Aggregation: 节点数={len(nodes)}, 边数={len(edges)}, 层数={len(processed_aggregation)}")
        
        # Return response
        return jsonify({
            "targetNode": target_node_idx,
            "nodes": nodes,
            "edges": edges,
            "aggregationData": processed_aggregation
        }), 200
        
    except Exception as e:
        import traceback
        error_msg = f"GCN Aggregation error: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500


# ============================================================================
# Model Comparison Endpoint
# ============================================================================

METRIC_COLUMNS = ['Acc', 'AUC', 'AUPRC', 'Precision', 'Recall', 'F1', 'MCC']

MODEL_COMPARISON_XLSX = os.path.join(os.path.dirname(__file__), 'data', 'DCPRES_cls_comp.xlsx')


@app.route('/api/v1/model-comparison', methods=['GET'])
def get_model_comparison():
    """
    Get model comparison data from DCPRES_cls_comp.xlsx.
    Returns metrics for each model.
    """
    try:
        import openpyxl

        if not os.path.exists(MODEL_COMPARISON_XLSX):
            logger.error(f"Model comparison Excel file not found: {MODEL_COMPARISON_XLSX}")
            return jsonify({
                "error": "Model comparison Excel file not found",
                "detail": f"File {MODEL_COMPARISON_XLSX} does not exist"
            }), 404

        wb = openpyxl.load_workbook(MODEL_COMPARISON_XLSX, data_only=True)
        ws = wb['Sheet1']

        # Read values by header name so that adding/removing or reordering metric
        # columns in the workbook does not leave stale positional indexes here.
        header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
        column_indexes = {
            str(header).strip(): index
            for index, header in enumerate(header_row)
            if header is not None
        }
        missing_metrics = [
            metric_name
            for metric_name in METRIC_COLUMNS
            if metric_name not in column_indexes
        ]
        if missing_metrics:
            raise ValueError(
                "Model comparison Excel file is missing metric columns: "
                + ", ".join(missing_metrics)
            )

        models = []
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True):
            model_name = row[0]
            if model_name is None:
                continue
            model_name = str(model_name)
            metrics = {}
            for metric_name in METRIC_COLUMNS:
                column_index = column_indexes[metric_name]
                cell_val = row[column_index] if column_index < len(row) else None
                if cell_val is not None:
                    try:
                        metrics[metric_name] = float(cell_val)
                    except (ValueError, TypeError):
                        metrics[metric_name] = 0.0
                else:
                    metrics[metric_name] = 0.0
            models.append({
                "name": model_name,
                "display_name": model_name,
                "metrics": metrics
            })
            logger.info(f"Loaded model comparison data for {model_name}")

        wb.close()

        response = {
            "models": models,
            "metric_names": METRIC_COLUMNS
        }

        logger.info(f"Model comparison data returned for {len(models)} models")
        return jsonify(response), 200

    except Exception as e:
        import traceback
        error_msg = f"Model comparison error: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500


# ============================================================================
# DCPRES Classification Heatmap Endpoint
# ============================================================================

@app.route('/api/v1/rgcnformer-classification-heatmap', methods=['GET'])
def get_rgcnformer_classification_heatmap():
    """
    Get DCPRES per-class classification performance data for heatmap visualization.
    Returns the full 12-class classification metrics from DCPRES_cls.csv.
    """
    try:
        csv_path = os.path.join(
            config.MODEL_COMPARISON_CSV_DIR,
            config.MODEL_COMPARISON_FILES.get('DCPRES', 'DCPRES_cls.csv')
        )

        if not os.path.exists(csv_path):
            logger.error(f"DCPRES CSV file not found: {csv_path}")
            return jsonify({
                "error": "DCPRES CSV file not found",
                "detail": f"File {csv_path} does not exist"
            }), 404

        classes = []
        heatmap_data = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_name = row.get('Class', '')
                classes.append(class_name)
                row_data = {"class": class_name}
                for col in METRIC_COLUMNS:
                    if col in row and row[col]:
                        try:
                            row_data[col] = float(row[col])
                        except ValueError:
                            row_data[col] = 0.0
                heatmap_data.append(row_data)

        response = {
            "model_name": "DCPRES",
            "classes": classes,
            "metric_names": METRIC_COLUMNS,
            "data": heatmap_data
        }

        logger.info(f"DCPRES heatmap data returned: {len(classes)} classes, {len(METRIC_COLUMNS)} metrics")
        return jsonify(response), 200

    except Exception as e:
        import traceback
        error_msg = f"DCPRES heatmap error: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500


# ============================================================================
# Dataset Comparison Heatmap Endpoint
# ============================================================================

@app.route('/api/v1/dataset-comparison-heatmap', methods=['GET'])
def get_dataset_comparison_heatmap():
    """
    Get multi-dataset comparison performance data for heatmap visualization.
    Reads from CORA-CITE-AMAP-BAT-EAT.xlsx with 6 datasets x 4 metrics x 9 models.
    """
    try:
        import re
        import openpyxl

        xlsx_path = config.DATASET_COMPARISON_XLSX
        if not os.path.exists(xlsx_path):
            logger.error(f"Dataset comparison Excel file not found: {xlsx_path}")
            return jsonify({
                "error": "Excel file not found",
                "detail": f"File {xlsx_path} does not exist"
            }), 404

        wb = openpyxl.load_workbook(xlsx_path, data_only=True)
        ws = wb['Sheet1']

        model_names = [cell.value for cell in ws[1][2:]]  # DAEGC, DyFSS, ..., DCPRES
        dataset_names = []
        metric_names = ['NMI', 'ACC', 'ARI', 'F1']

        rows_data = []
        current_dataset = None

        for row in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True):
            dataset_val = row[0]
            metric_val = row[1]

            if dataset_val is not None and dataset_val != 'None':
                current_dataset = dataset_val
                dataset_names.append(current_dataset)
            elif dataset_val == 'None' and current_dataset is not None:
                pass
            else:
                current_dataset = None

            row_label = f"{current_dataset}-{metric_val}"
            row_dict = {"row": row_label}

            for col_idx, model_name in enumerate(model_names):
                cell_val = row[col_idx + 2]
                if cell_val == '—' or cell_val is None:
                    row_dict[model_name] = None
                else:
                    match = re.match(r"([\d.]+)", str(cell_val))
                    if match:
                        row_dict[model_name] = float(match.group(1))
                    else:
                        row_dict[model_name] = None

            rows_data.append(row_dict)

        response = {
            "dataset_names": dataset_names,
            "metric_names": metric_names,
            "model_names": model_names,
            "row_labels": [r["row"] for r in rows_data],
            "data": rows_data
        }

        logger.info(f"Dataset comparison heatmap returned: {len(dataset_names)} datasets, {len(metric_names)} metrics, {len(model_names)} models")
        return jsonify(response), 200

    except Exception as e:
        import traceback
        error_msg = f"Dataset comparison heatmap error: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500


# ============================================================================
# RGCNFormer Localization Performance Endpoint
# ============================================================================

LOC_COMPARISON_FILES = {
    'DCPRES': 'DCPRES_loc.csv',
    'SCDGC': 'SCDGC_loc.csv',
    'DSCPS': 'DSCPS_loc.csv',
}

K_VALUE_COLUMNS = ['Top-1', 'Top-3', 'Top-5', 'Top-7', 'Top-10', 'Top-20', 'Top-50']
K_VALUES = [1, 3, 5, 7, 10, 20, 50]


@app.route('/api/v1/rgcnformer-localization', methods=['GET'])
def get_rgcnformer_localization():
    """
    Get RGCNFormer localization performance data.
    Returns donut chart data (12 classes x 7 Top-K values) and per-class statistics.
    """
    try:
        loc_csv_path = os.path.join(config.MODEL_COMPARISON_CSV_DIR, 'DCPRES_loc.csv')
        stat_csv_path = os.path.join(config.MODEL_COMPARISON_CSV_DIR, 'statistic_loc.csv')

        if not os.path.exists(loc_csv_path):
            logger.error(f"Localization CSV not found: {loc_csv_path}")
            return jsonify({
                "error": "Localization CSV file not found",
                "detail": f"File {loc_csv_path} does not exist"
            }), 404

        classes = []
        class_names = []
        heatmap = []

        with open(loc_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_id = row.get('Class', '')
                name = row.get('Name', '')
                classes.append(class_id)
                class_names.append(name)
                row_values = []
                for col in K_VALUE_COLUMNS:
                    try:
                        row_values.append(float(row.get(col, 0)))
                    except (ValueError, TypeError):
                        row_values.append(0.0)
                heatmap.append(row_values)

        statistics = []
        if os.path.exists(stat_csv_path):
            with open(stat_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    statistics.append({
                        "class": row.get('Class', ''),
                        "Mean": float(row.get('Mean', 0)),
                        "Median": float(row.get('Median', 0)),
                        "Mode": int(float(row.get('Mode', 0))),
                        "Mode_Ratio": float(row.get('Mode_Ratio(%)', 0)),
                        "Sequence_Count": int(float(row.get('Sequence_Count', 0))),
                        "Min_Value": int(float(row.get('Min_Value', 0))),
                        "Max_Value": int(float(row.get('Max_Value', 0))),
                        "Standard_Deviation": float(row.get('Standard_Deviation', 0)),
                    })

        response = {
            "model_name": "DCPRES",
            "classes": classes,
            "class_names": class_names,
            "k_labels": K_VALUE_COLUMNS,
            "k_values": K_VALUES,
            "heatmap": heatmap,
            "statistics": statistics
        }

        logger.info(f"Localization data returned: {len(classes)} classes, {len(K_VALUE_COLUMNS)} Top-K values")
        return jsonify(response), 200

    except Exception as e:
        import traceback
        error_msg = f"Localization error: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500


# ============================================================================
# RGCNFormer Localization Model Comparison Endpoint
# ============================================================================

LOC_COMPARISON_XLSX = os.path.join(os.path.dirname(__file__), 'data', 'DCPRES_loc_comp.xlsx')


@app.route('/api/v1/rgcnformer-loc-comparison', methods=['GET'])
def get_rgcnformer_loc_comparison():
    """
    Get localization model comparison data from DCPRES_loc_comp.xlsx.
    Returns per-model Top-K performance for bubble chart visualization.
    """
    try:
        import openpyxl

        if not os.path.exists(LOC_COMPARISON_XLSX):
            logger.error(f"Loc comparison Excel file not found: {LOC_COMPARISON_XLSX}")
            return jsonify({
                "error": "Loc comparison Excel file not found",
                "detail": f"File {LOC_COMPARISON_XLSX} does not exist"
            }), 404

        wb = openpyxl.load_workbook(LOC_COMPARISON_XLSX, data_only=True)
        ws = wb['Sheet1']

        # Row 1: headers [None, 'Top-1', 'Top-3', ...]
        # Row 2+: [ModelName, value, value, ...]
        model_names = []
        heatmap = []

        for row in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True):
            model_name = row[0]
            if model_name is None:
                continue
            model_name = str(model_name)
            model_names.append(model_name)
            row_values = []
            for col_idx in range(len(K_VALUE_COLUMNS)):
                cell_val = row[col_idx + 1]
                if cell_val is not None:
                    try:
                        row_values.append(float(cell_val))
                    except (ValueError, TypeError):
                        row_values.append(0.0)
                else:
                    row_values.append(0.0)
            heatmap.append(row_values)
            logger.info(f"Loaded loc comparison data for {model_name}")

        wb.close()

        response = {
            "model_names": model_names,
            "k_labels": K_VALUE_COLUMNS,
            "k_values": K_VALUES,
            "heatmap": heatmap
        }

        logger.info(f"Loc comparison data returned: {len(model_names)} models, {len(K_VALUE_COLUMNS)} Top-K values")
        return jsonify(response), 200

    except Exception as e:
        import traceback
        error_msg = f"Loc comparison error: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500


# ============================================================================
# UMAP Visualization Data Endpoint
# ============================================================================

@app.route('/api/v1/umap-data', methods=['GET'])
def get_umap_data():
    """
    Return pre-computed UMAP coordinates and metadata for visualization.
    Optional query param 'n' to subsample points for faster loading.
    """
    try:
        umap_path = config.UMAP_DATA_PATH

        if not os.path.exists(umap_path):
            logger.error(f"UMAP data file not found: {umap_path}")
            return jsonify({
                "error": "UMAP data not found",
                "detail": f"File {umap_path} does not exist"
            }), 404

        with open(umap_path, 'r') as f:
            data = json.load(f)

        n_param = request.args.get('n', type=int)
        if n_param and n_param < len(data.get('points', [])):
            import random
            points = data['points']
            label_groups = {}
            for p in points:
                label_groups.setdefault(p['label'], []).append(p)

            sampled = []
            per_label = max(1, n_param // len(label_groups))
            for label, pts in label_groups.items():
                n_take = min(per_label, len(pts))
                sampled.extend(random.sample(pts, n_take))

            data['points'] = sampled
            data['metadata'] = data.get('metadata', {})
            data['metadata']['total_points'] = len(sampled)
            data['metadata']['subsampled'] = True
            logger.info(f"UMAP subsampled to {len(sampled)} points")

        logger.info(f"UMAP data served: {len(data.get('points', []))} points")
        return jsonify(data), 200

    except Exception as e:
        import traceback
        error_msg = f"UMAP data error: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500


# ============================================================================
# CORA UMAP Visualization Data Endpoint
# ============================================================================

@app.route('/api/v1/umap-cora-data', methods=['GET'])
def get_umap_cora_data():
    """
    Return pre-computed CORA UMAP coordinates and metadata for visualization.
    Optional query param 'n' to subsample points for faster loading.
    """
    try:
        umap_path = config.UMAP_CORA_DATA_PATH

        if not os.path.exists(umap_path):
            logger.error(f"CORA UMAP data file not found: {umap_path}")
            return jsonify({
                "error": "CORA UMAP data not found",
                "detail": f"File {umap_path} does not exist"
            }), 404

        with open(umap_path, 'r') as f:
            data = json.load(f)

        n_param = request.args.get('n', type=int)
        if n_param and n_param < len(data.get('points', [])):
            import random
            points = data['points']
            label_groups = {}
            for p in points:
                label_groups.setdefault(p['label'], []).append(p)

            sampled = []
            per_label = max(1, n_param // len(label_groups))
            for label, pts in label_groups.items():
                n_take = min(per_label, len(pts))
                sampled.extend(random.sample(pts, n_take))

            data['points'] = sampled
            data['metadata'] = data.get('metadata', {})
            data['metadata']['total_points'] = len(sampled)
            data['metadata']['subsampled'] = True
            logger.info(f"CORA UMAP subsampled to {len(sampled)} points")

        logger.info(f"CORA UMAP data served: {len(data.get('points', []))} points")
        return jsonify(data), 200

    except Exception as e:
        import traceback
        error_msg = f"CORA UMAP data error: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500


# ============================================================================
# Attention Comparison Endpoint (Pre-computed from .npz files)
# ============================================================================

ATTENTION_NPZ_FILES = {
    'DCPRES': os.path.join(os.path.dirname(__file__), 'data', 'DCPRES_atten.npz'),
    'SCDGC': os.path.join(os.path.dirname(__file__), 'data', 'SCDGC_atten.npz'),
    'DSCPS': os.path.join(os.path.dirname(__file__), 'data', 'DSCPS_atten.npz'),
    'GCN': os.path.join(os.path.dirname(__file__), 'data', 'GCN_atten.npz'),
}

CLASS_NAMES = ['Am', 'Atol', 'Cm', 'Gm', 'Tm', 'Y', 'ac4C', 'm1A', 'm5C', 'm6A', 'm6Am', 'm7G']


@app.route('/api/v1/attention-comparison', methods=['GET'])
def get_attention_comparison():
    """
    Get pre-computed attention comparison data for 4 models.
    Randomly selects 5 samples, or selects one sample by the sequence_id query
    parameter, and returns attention weights for classes with label=1.
    """
    try:
        import numpy as np

        # Check all files exist
        for model_name, path in ATTENTION_NPZ_FILES.items():
            if not os.path.exists(path):
                return jsonify({
                    "error": f"Attention data file not found for {model_name}",
                    "detail": f"File {path} does not exist"
                }), 404

        # Load all models data
        models_data = {}
        for model_name, path in ATTENTION_NPZ_FILES.items():
            data = np.load(path, allow_pickle=True)
            models_data[model_name] = {
                'attn_weights': data['attn_weights'],  # [200, 12, 1001]
                'labels': data['labels'],               # [200, 12]
                'sites': data['sites'],                 # [200, 1001]
                'indices': data['indices'],             # [200]
            }

        # Use first model to determine sample indices (same across all models)
        num_samples = models_data['DCPRES']['attn_weights'].shape[0]

        sequence_id = request.args.get('sequence_id', type=str)
        if sequence_id is not None:
            try:
                requested_sequence_id = int(sequence_id)
            except ValueError:
                return jsonify({
                    "error": "Invalid sequence ID",
                    "detail": "sequence_id must be an integer"
                }), 400

            sequence_ids = models_data['DCPRES']['indices']
            matches = np.where(sequence_ids == requested_sequence_id)[0]
            if matches.size == 0:
                return jsonify({
                    "error": "Sequence ID not found",
                    "detail": f"Sequence ID {requested_sequence_id} is not available"
                }), 404
            selected_indices = [int(matches[0])]
        else:
            # Randomly select 5 samples
            np.random.seed(None)  # True random each time
            selected_indices = np.random.choice(
                num_samples,
                size=min(5, num_samples),
                replace=False
            )

        samples = []
        for idx in selected_indices:
            idx = int(idx)
            sample_data = {
                'index': int(models_data['DCPRES']['indices'][idx]),
                'models': {}
            }

            for model_name, mdata in models_data.items():
                labels = mdata['labels'][idx]  # [12]
                attn = mdata['attn_weights'][idx]  # [12, 1001]
                sites = mdata['sites'][idx]  # [1001]

                # Find classes with label=1
                active_class_indices = np.where(labels == 1)[0].tolist()

                # Get attention only for active classes
                active_attention = attn[active_class_indices].tolist()

                # Get true sites (positions where sites > 0)
                true_sites = np.where(sites > 0)[0].tolist()

                sample_data['models'][model_name] = {
                    'attention': active_attention,
                    'class_indices': active_class_indices,
                    'class_names': [CLASS_NAMES[i] for i in active_class_indices],
                    'true_sites': true_sites,
                }

            samples.append(sample_data)

        response = {
            'samples': samples,
            'class_names': CLASS_NAMES,
            'model_names': list(ATTENTION_NPZ_FILES.keys()),
            'available_sequence_ids': [
                int(sequence_id)
                for sequence_id in models_data['DCPRES']['indices']
            ],
        }

        logger.info(f"Attention comparison data returned: {len(samples)} samples, {len(ATTENTION_NPZ_FILES)} models")
        return jsonify(response), 200

    except Exception as e:
        import traceback
        error_msg = f"Attention comparison error: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500


@app.route('/api/v1/attention-visualization', methods=['POST'])
def get_attention_visualization():
    """
    Get attention visualization for a user-submitted sequence.
    Runs model inference and returns attention weights for all 12 classes.
    """
    try:
        data = request.get_json()
        original_sequence = data.get('rnaSequence', '')

        if not original_sequence:
            return jsonify({"error": "No sequence provided"}), 400

        logger.info(f"Attention visualization: sequence length={len(original_sequence)}")

        # Store original sequence for response
        sequence = original_sequence

        # For shorter sequences, pad to 1001; for longer sequences, truncate
        TARGET_LENGTH = 1001
        seq_len = len(sequence)

        # Track padding/trimming for index remapping
        left_padding = 0

        if seq_len != TARGET_LENGTH:
            if seq_len < TARGET_LENGTH:
                padding_needed = TARGET_LENGTH - seq_len
                left_pad = padding_needed // 2
                left_padding = left_pad
                sequence = 'N' * left_pad + sequence + 'N' * (padding_needed - left_pad)
            else:
                excess = seq_len - TARGET_LENGTH
                left_trim = excess // 2
                sequence = sequence[left_trim:seq_len - (excess - left_trim)]

        # Call LinearFold to get secondary structure
        structures = run_linearfold([sequence])
        structure = structures[0]

        # Build edge index from structure
        edge_index = build_edge_index_from_structure(sequence, structure)

        # Prepare model input
        x = one_hot_encode_sequence(sequence)
        x = torch.FloatTensor(x)

        # Create batch tensor
        batch = torch.zeros(len(sequence), dtype=torch.long)

        # Create Batch object
        data_batch = Batch(x=x, edge_index=edge_index, batch=batch)
        data_batch = data_batch.to(device)

        # Run model inference
        with torch.no_grad():
            output = model(
                data_batch.x,
                data_batch.edge_index,
                data_batch.batch,
                return_attention=True
            )

        # Extract attention weights based on model type
        if model_cfg['use_hierarchical']:
            # Hierarchical model returns (logits_12, logits_4, attn_weights_12)
            logits_12 = output[0]  # [1, 12]
            attn_weights_12 = output[2]  # [1, 12, 1001]

            # Convert to numpy
            probs = torch.sigmoid(logits_12).cpu().numpy()[0]  # [12]
            attn = attn_weights_12.cpu().numpy()[0]  # [12, 1001]
        else:
            # Simple pooling model returns (logits, attn_weights)
            logits = output[0]
            attn_weights = output[1]

            probs = torch.sigmoid(logits).cpu().numpy()[0]  # [12]
            attn = attn_weights.cpu().numpy()[0]  # [12, 1001]

        # Build response - all 12 classes
        classes_data = []
        for i in range(12):
            # Normalize attention weights
            attn_i = attn[i]
            attn_sum = attn_i.sum()
            if attn_sum > 0:
                attn_normalized = (attn_i / attn_sum).tolist()
            else:
                attn_normalized = attn_i.tolist()

            classes_data.append({
                'index': i,
                'name': CLASS_NAMES[i],
                'probability': float(probs[i]),
                'attention': attn_normalized,
            })

        response = {
            'sequence_length': len(original_sequence),
            'left_padding': left_padding,
            'classes': classes_data,
            'class_names': CLASS_NAMES,
        }

        logger.info(f"Attention visualization returned for {len(CLASS_NAMES)} classes")
        return jsonify(response), 200

    except Exception as e:
        import traceback
        error_msg = f"Attention visualization error: {str(e)}"
        logger.error(f"ERROR: {error_msg}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500


if __name__ == '__main__':
    # 运行服务器（与 Nginx 配置一致）
    logger.info(f"Starting Flask server on {config.FLASK_HOST}:{config.FLASK_PORT}")
    app.run(debug=config.FLASK_DEBUG, host=config.FLASK_HOST, port=config.FLASK_PORT)
