"""
中文：预测任务提交与结果检索 API 端点。
English: Prediction task submission and result retrieval API endpoints.
"""
import json
import hashlib
import uuid
import time
from flask import Blueprint, jsonify, request, current_app
from celery.result import AsyncResult

from mrmodn_backend.core.config import config, get_logger
from mrmodn_backend.workers.tasks import celery_app, run_prediction_task, process_sequence_in_batch

prediction_bp = Blueprint('prediction', __name__)
logger = get_logger('api.prediction')


@prediction_bp.route('/mrmodn/api/v1/submit-task', methods=['POST'])
def submit_task():
    """
    中文：提交异步预测任务。立即返回 job_id 和 pending 状态，实际预测通过 Celery 后台执行。
    English: Submit an async prediction task. Returns immediately with job_id and pending status; actual prediction runs in background via Celery.
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

    redis_client = current_app.config.get('REDIS_CLIENT')

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


@prediction_bp.route('/mrmodn/api/v1/results/<job_id>', methods=['GET'])
def get_result(job_id):
    """
    中文：根据 job_id 查询预测结果。优先查 Redis 缓存，未命中则查 Celery 任务状态。
    English: Retrieve prediction result by job_id. Checks Redis cache first, then falls back to Celery task status.
    """
    redis_client = current_app.config.get('REDIS_CLIENT')

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


@prediction_bp.route('/mrmodn/api/v1/wx-submit-task', methods=['POST'])
def wx_submit_task():
    """
    中文：微信批量提交预测任务（最多 5 条序列）。返回批量 job_id 用于进度追踪，空序列将被跳过。
    English: Submit up to 5 prediction tasks for WeChat Mini Program. Returns a batch job_id for progress tracking. Empty sequences are skipped.
    """
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

    redis_client = current_app.config.get('REDIS_CLIENT')

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


@prediction_bp.route('/mrmodn/api/v1/wx-task-progress/<job_id>', methods=['GET'])
def wx_task_progress(job_id):
    """
    中文：查询批量任务进度。返回状态、总序列数、已完成数和已完成结果列表。
    English: Query batch task progress. Returns status, total sequences, completed count, and completed results.
    """
    redis_client = current_app.config.get('REDIS_CLIENT')

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
