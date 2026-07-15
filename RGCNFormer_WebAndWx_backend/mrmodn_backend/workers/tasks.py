"""
中文：Celery 异步任务模块。定义 RNA 预测任务（run_prediction_task）和微信批量处理任务（process_sequence_in_batch）。
English: Celery async task module. Defines the RNA prediction task (run_prediction_task) and WeChat batch processing task (process_sequence_in_batch).
"""
from celery import Celery
import redis
import json
import hashlib
import torch
import numpy as np
from torch_geometric.data import Batch

from mrmodn_backend.models.mrmodn import RNA_ClassQuery_Model
from mrmodn_backend.services.rna_structure import run_linearfold, build_edge_index_from_structure
from mrmodn_backend.core.config import config, get_logger
from mrmodn_backend.core.constants import MOD_NAMES, INDEX_TO_NUCLEOTIDE

# ============================================================================
# Celery Application Configuration
# ============================================================================

celery_app = Celery(
    'rna_prediction_tasks',
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=config.CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=config.CELERY_TASK_SOFT_TIME_LIMIT,
)

# ============================================================================
# Redis Connection (for caching results)
# ============================================================================

logger = get_logger('tasks')

try:
    redis_client = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        db=config.REDIS_DB,
        decode_responses=True
    )
    redis_client.ping()
    logger.info("Connected to Redis successfully")
except Exception as e:
    logger.warning(f"Could not connect to Redis: {e}")
    redis_client = None

# ============================================================================
# Model Loading (Load once at worker startup)
# ============================================================================

logger.info("Loading model and configuration...")

with open(config.MODEL_CONFIG_PATH, 'r') as f:
    model_config_file = json.load(f)

model_cfg = model_config_file['model']

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

checkpoint_path = config.MODEL_CHECKPOINT_PATH
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

logger.info(f"Model loaded successfully from {checkpoint_path}")
logger.info("Model is ready for predictions!")

# ============================================================================
# Helper Functions
# ============================================================================

def one_hot_encode_sequence(sequence: str) -> np.ndarray:
    """
    中文：将 RNA 序列字符串转换为 one-hot 编码。
    English: Convert RNA sequence string to one-hot encoding.

    Args:
        sequence: RNA 序列字符串 (A, C, G, U) / RNA sequence string (A, C, G, U)

    Returns:
        shape 为 (len(sequence), 4) 的 one-hot 数组 / One-hot encoded array of shape (len(sequence), 4)
    """
    one_hot_mapping = {
        'A': [1., 0., 0., 0.],
        'C': [0., 1., 0., 0.],
        'G': [0., 0., 1., 0.],
        'U': [0., 0., 0., 1.],
        'T': [0., 0., 0., 1.],
        'N': [0., 0., 0., 0.]
    }

    one_hot = np.zeros((len(sequence), 4), dtype=np.float32)
    for i, nucleotide in enumerate(sequence.upper()):
        if nucleotide in one_hot_mapping:
            one_hot[i] = one_hot_mapping[nucleotide]
        else:
            one_hot[i] = [0., 0., 0., 0.]

    return one_hot

# ============================================================================
# Celery Task Definition
# ============================================================================

@celery_app.task(name='tasks.run_prediction_task', bind=True)
def run_prediction_task(self, original_sequence, target_class_id=None, top_k=None):
    """
    中文：RNA 预测 Celery 任务。执行预处理、LinearFold、模型推理、层级剪枝、注意力提取和 GCN 图构建，并将结果缓存到 Redis。
    English: RNA prediction Celery task. Performs preprocessing, LinearFold, model inference, hierarchical pruning, attention extraction, and GCN graph construction, then caches results in Redis.
    """
    logger.info(f"Task {self.request.id}: Starting prediction for sequence length: {len(original_sequence)}")

    try:
        job_id = hashlib.sha256(original_sequence.encode('utf-8')).hexdigest()
        logger.info(f"Task {self.request.id}: Generated job_id: {job_id}")

        sequence = original_sequence

        TARGET_LENGTH = 1001
        seq_len = len(sequence)

        left_padding = 0
        left_trimming = 0

        if seq_len != TARGET_LENGTH:
            if seq_len < TARGET_LENGTH:
                padding_needed = TARGET_LENGTH - seq_len
                left_pad = padding_needed // 2
                right_pad = padding_needed - left_pad
                left_padding = left_pad
                sequence = 'N' * left_pad + sequence + 'N' * right_pad
                logger.info(f"Task {self.request.id}: Padded sequence to {TARGET_LENGTH} with {padding_needed} 'N's (left: {left_pad}, right: {right_pad})")
            else:
                excess = seq_len - TARGET_LENGTH
                left_trim = excess // 2
                right_trim = excess - left_trim
                left_trimming = left_trim
                sequence = sequence[left_trim:seq_len - right_trim]
                logger.info(f"Task {self.request.id}: Trimmed sequence from {seq_len} to {TARGET_LENGTH} (left: {left_trim}, right: {right_trim})")

        logger.info(f"Task {self.request.id}: Running LinearFold...")
        try:
            structures = run_linearfold([sequence])
            if not structures or len(structures) == 0:
                raise ValueError("LinearFold returned empty results")
            structure = structures[0]
            if not structure:
                raise ValueError("LinearFold returned empty structure for sequence")
            logger.info(f"Task {self.request.id}: LinearFold completed")
        except Exception as e:
            error_detail = {
                "step": "linearfold",
                "error_type": type(e).__name__,
                "message": str(e),
                "sequence_length": len(sequence)
            }
            logger.error(f"Task {self.request.id}: LinearFold failed: {error_detail}")
            raise type(e)(f"LinearFold execution failed: {str(e)}") from e

        try:
            edge_index = build_edge_index_from_structure(sequence, structure)
            if edge_index is None or edge_index.numel() == 0:
                raise ValueError("Failed to build edge index from structure")
        except Exception as e:
            error_detail = {
                "step": "edge_index_building",
                "error_type": type(e).__name__,
                "message": str(e),
                "sequence_length": len(sequence),
                "structure_length": len(structure) if structure else 0
            }
            logger.error(f"Task {self.request.id}: Edge index building failed: {error_detail}")
            raise type(e)(f"Edge index building failed: {str(e)}") from e

        try:
            x = one_hot_encode_sequence(sequence)
            if x.shape != (len(sequence), 4):
                raise ValueError(f"Invalid one-hot encoding shape: expected ({len(sequence)}, 4), got {x.shape}")
            x = torch.FloatTensor(x)

            batch = torch.zeros(len(sequence), dtype=torch.long)
        except Exception as e:
            error_detail = {
                "step": "sequence_encoding",
                "error_type": type(e).__name__,
                "message": str(e),
                "sequence_length": len(sequence)
            }
            logger.error(f"Task {self.request.id}: Sequence encoding failed: {error_detail}")
            raise type(e)(f"Sequence encoding failed: {str(e)}") from e

        logger.info(f"Task {self.request.id}: Running model inference...")
        try:
            with torch.no_grad():
                data_batch = Batch(x=x, edge_index=edge_index, batch=batch)
                data_batch = data_batch.to(device)

                if model_cfg['use_hierarchical']:
                    logits_12class, logits_4class, attn_weights = model(
                        data_batch.x,
                        data_batch.edge_index,
                        data_batch.batch,
                        return_attention=True
                    )
                else:
                    logits_12class = model(data_batch)
                    attn_weights = None

            if logits_12class is None or logits_12class.shape[1] != 12:
                raise ValueError(f"Invalid model output shape: expected [batch, 12], got {logits_12class.shape if logits_12class is not None else 'None'}")
            if model_cfg['use_hierarchical'] and (logits_4class is None or logits_4class.shape[1] != 4):
                raise ValueError(f"Invalid 4-class output shape: expected [batch, 4], got {logits_4class.shape if logits_4class is not None else 'None'}")

            logger.info(f"Task {self.request.id}: Model inference completed")
        except Exception as e:
            error_detail = {
                "step": "model_inference",
                "error_type": type(e).__name__,
                "message": str(e),
                "device": str(device),
                "is_hierarchical": model_cfg['use_hierarchical'],
                "sequence_length": len(sequence)
            }
            logger.error(f"Task {self.request.id}: Model inference failed: {error_detail}")
            raise type(e)(f"Model inference failed: {str(e)}") from e

        logits_12class = logits_12class.cpu().numpy()[0]
        probs_12class = 1 / (1 + np.exp(-logits_12class))

        if model_cfg['use_hierarchical']:
            logits_4class = logits_4class.cpu().numpy()[0]
            probs_4class = 1 / (1 + np.exp(-logits_4class))
            attn_weights = attn_weights.cpu().numpy()[0]

        group_names = ['A', 'C', 'G', 'U']
        group_to_classes = {
            'A': [0, 1, 7, 9, 10],
            'C': [2, 6, 8],
            'G': [3, 11],
            'U': [4, 5]
        }

        class_to_group = {}
        for group_name, class_indices in group_to_classes.items():
            for class_idx in class_indices:
                class_to_group[class_idx] = group_name

        thresholds_12class = config.THRESHOLDS_12_CLASS
        thresholds_4class = config.THRESHOLDS_4_CLASS

        predictions_12class = {}
        for class_idx in range(12):
            class_prob = probs_12class[class_idx]
            class_threshold = thresholds_12class[class_idx]
            predictions_12class[class_idx] = bool(class_prob > class_threshold)

        predictions_4class = {}
        if model_cfg['use_hierarchical']:
            for group_idx in range(4):
                group_prob = probs_4class[group_idx]
                group_threshold = thresholds_4class[group_idx]
                predictions_4class[group_idx] = bool(group_prob > group_threshold)
        else:
            for group_idx in range(4):
                predictions_4class[group_idx] = False

        for group_idx, group_name in enumerate(group_names):
            child_indices = group_to_classes[group_name]
            has_any_child = any(predictions_12class[class_idx] for class_idx in child_indices)
            if not has_any_child:
                predictions_4class[group_idx] = False

        for group_idx, group_name in enumerate(group_names):
            if not predictions_4class[group_idx]:
                child_indices = group_to_classes[group_name]
                for class_idx in child_indices:
                    predictions_12class[class_idx] = False

        classification = {
            "name": "RNA Sequence",
            "isPredicted": True,
            "children": []
        }

        for group_idx, group_name in enumerate(group_names):
            group_predicted = predictions_4class[group_idx]

            children = []
            for class_idx in group_to_classes[group_name]:
                class_predicted = predictions_12class[class_idx]
                children.append({
                    "name": MOD_NAMES.get(class_idx, f"Class{class_idx}"),
                    "isPredicted": class_predicted
                })

            classification["children"].append({
                "name": f"Group {group_name}",
                "isPredicted": group_predicted,
                "children": children
            })

        active_groups = set()
        for group_idx, is_predicted in predictions_4class.items():
            if is_predicted:
                active_groups.add(group_names[group_idx])

        logger.info(f"Task {self.request.id}: Active groups after pruning: {active_groups}")

        attention_data = {
            "sequence": original_sequence,
            "weights": []
        }

        if attn_weights is not None and active_groups:
            if target_class_id is not None and 0 <= target_class_id < 12:
                combined_attention = attn_weights[target_class_id]
                predicted_class_indices = [target_class_id]
                logger.info(f"Task {self.request.id}: Using attention weights for class {target_class_id}")

                target_nucleotide = INDEX_TO_NUCLEOTIDE.get(target_class_id)
                if target_nucleotide:
                    active_groups = {target_nucleotide}
                    logger.info(f"Task {self.request.id}: Filtering to nucleotide group: {target_nucleotide} for class {target_class_id}")
                else:
                    logger.warning(f"Task {self.request.id}: Warning: No nucleotide mapping found for class {target_class_id}")
                    active_groups = set()
            else:
                predicted_class_indices = [idx for idx, pred in predictions_12class.items() if pred]
                combined_attention = np.zeros(len(sequence))
                if predicted_class_indices:
                    for class_idx in predicted_class_indices:
                        combined_attention += attn_weights[class_idx]
                    combined_attention /= len(predicted_class_indices)

            K = top_k if top_k is not None else 3
            all_top_sites = []

            for group in active_groups:
                if group == 'U':
                    group_mask = np.array([1.0 if (nucleotide == 'U' or nucleotide == 'T') else 0.0 for nucleotide in sequence.upper()])
                else:
                    group_mask = np.array([1.0 if (nucleotide == group and nucleotide != 'T') else 0.0 for nucleotide in sequence.upper()])

                logger.info(f"Task {self.request.id}: Processing group '{group}':")
                logger.info(f"Task {self.request.id}:   - Nucleotides in sequence: {np.sum(group_mask)}")
                logger.info(f"Task {self.request.id}:   - Combined attention shape: {combined_attention.shape}")

                num_available_sites = int(group_mask.sum())
                top_k_for_group = min(K, num_available_sites)

                logger.info(f"Task {self.request.id}:   - Top K for group: {top_k_for_group}")

                if top_k_for_group <= 0:
                    logger.info(f"Task {self.request.id}:   - Skipping group '{group}': no nucleotides found in sequence")
                    continue

                group_attention = combined_attention * group_mask
                group_attention[group_mask == 0] = -np.inf

                logger.info(f"Task {self.request.id}:   - Max attention in group: {np.max(group_attention[group_mask == 1])}")

                if top_k_for_group > 0:
                    top_indices_for_group = np.argsort(group_attention)[-top_k_for_group:][::-1]

                    for pos in top_indices_for_group:
                        pos_int = int(pos)
                        score_float = float(combined_attention[pos])
                        original_index = pos_int - left_padding + left_trimming

                        if 0 <= original_index < len(original_sequence):
                            all_top_sites.append({
                                "index": original_index,
                                "type": group,
                                "score": score_float
                            })

            all_top_sites.sort(key=lambda x: x["score"], reverse=True)
            attention_data["weights"] = all_top_sites

        edge_index_np = edge_index.cpu().numpy()
        edges = []

        valid_start = left_padding
        valid_end = left_padding + len(original_sequence)

        for i in range(int(edge_index_np.shape[1])):
            source = int(edge_index_np[0, i])
            target = int(edge_index_np[1, i])

            if not (0 <= source < len(sequence) and 0 <= target < len(sequence)):
                continue

            orig_source = source - left_padding + left_trimming
            orig_target = target - left_padding + left_trimming

            if not (0 <= orig_source < len(original_sequence) and 0 <= orig_target < len(original_sequence)):
                continue

            if source >= target:
                continue

            nuc_source = original_sequence[orig_source]
            nuc_target = original_sequence[orig_target]
            edges.append({
                "source": f"{nuc_source}{orig_source}",
                "target": f"{nuc_target}{orig_target}"
            })

        logger.info(f"Task {self.request.id}: GCN visualization stats: total edges={len(edges)}")

        nodes = []
        for i in range(len(original_sequence)):
            nuc = original_sequence[i]
            nodes.append({
                "id": f"{nuc}{i}",
                "label": f"位置{i}: {nuc}",
                "data": {"index": i, "type": nuc, "name": f"{'腺嘌呤' if nuc == 'A' else '胞嘧啶' if nuc == 'C' else '鸟嘌呤' if nuc == 'G' else '尿嘧啶'}"}
            })

        valid_node_ids = {node["id"] for node in nodes}

        valid_edges = [
            edge for edge in edges
            if edge["source"] in valid_node_ids and edge["target"] in valid_node_ids
        ]

        logger.info(f"Task {self.request.id}: GCN visualization stats: nodes={len(nodes)}, valid edges={len(valid_edges)}")

        gcn_data = {
            "nodes": nodes,
            "edges": valid_edges
        }

        response = {
            "jobId": job_id,
            "status": "completed",
            "classification": classification,
            "attention": attention_data,
            "gcn": gcn_data
        }

        if redis_client:
            try:
                response_json = json.dumps(response, ensure_ascii=False)
                redis_client.setex(f"task:{job_id}", config.REDIS_CACHE_TTL, response_json)
                logger.info(f"Task {self.request.id}: Result cached in Redis for job_id: {job_id}")
            except Exception as e:
                error_detail = {
                    "step": "redis_caching",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "job_id": job_id
                }
                logger.error(f"Task {self.request.id}: Failed to cache result in Redis: {error_detail}")

        logger.info(f"Task {self.request.id}: Prediction completed successfully")
        return response

    except Exception as e:
        import traceback
        error_info = {
            "task_id": self.request.id,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "sequence_length": len(original_sequence) if original_sequence else 0
        }

        logger.error(f"Task {self.request.id}: Task failed with error: {error_info['error_type']} - {error_info['error_message']}")
        logger.debug(f"Task {self.request.id}: Full traceback:\n{error_info['traceback']}")

        class TaskError(Exception):
            def __init__(self, message, error_info):
                super().__init__(message)
                self.error_info = error_info


# ============================================================================
# Batch Processing Task for WeChat Mini Program
# ============================================================================

@celery_app.task(name='tasks.process_sequence_in_batch', bind=True)
def process_sequence_in_batch(self, job_id, sequence_data, index, target_class_id=None, top_k=None):
    """
    中文：批量处理中的单条序列 Celery 任务。处理完成后更新 Redis 中的批量任务状态。
    English: Celery task for processing a single sequence within a batch. Updates Redis batch job state after completion.
    """
    logger.info(f"Task {self.request.id}: Processing sequence {index} for batch {job_id}")

    try:
        result = run_prediction_task(sequence_data, target_class_id, top_k)

        if redis_client:
            try:
                results_json = redis_client.hget(f'batch_job:{job_id}', 'results')
                results = json.loads(results_json) if results_json else []

                result_with_index = {
                    "index": index,
                    "jobId": result.get("jobId"),
                    "sequence": sequence_data,
                    "status": "completed",
                    "classification": result.get("classification"),
                    "attention": result.get("attention"),
                    "gcn": result.get("gcn")
                }
                results.append(result_with_index)

                redis_client.hset(f'batch_job:{job_id}', 'results', json.dumps(results, ensure_ascii=False))

                completed = redis_client.hincrby(f'batch_job:{job_id}', 'completed_sequences', 1)
                logger.info(f"Task {self.request.id}: Sequence {index} completed. Total completed: {completed}")

                total_sequences = int(redis_client.hget(f'batch_job:{job_id}', 'total_sequences') or 0)
                if completed >= total_sequences:
                    redis_client.hset(f'batch_job:{job_id}', 'status', 'COMPLETED')
                    logger.info(f"Task {self.request.id}: Batch {job_id} fully completed ({completed}/{total_sequences} sequences)")

            except Exception as e:
                logger.error(f"Task {self.request.id}: Failed to update Redis state for batch {job_id}: {e}")

        return result

    except Exception as e:
        if redis_client:
            try:
                results_json = redis_client.hget(f'batch_job:{job_id}', 'results')
                results = json.loads(results_json) if results_json else []

                result_with_index = {
                    "index": index,
                    "sequence": sequence_data,
                    "status": "failed",
                    "error": str(e)
                }
                results.append(result_with_index)

                redis_client.hset(f'batch_job:{job_id}', 'results', json.dumps(results, ensure_ascii=False))

                completed = redis_client.hincrby(f'batch_job:{job_id}', 'completed_sequences', 1)

                total_sequences = int(redis_client.hget(f'batch_job:{job_id}', 'total_sequences') or 0)
                if completed >= total_sequences:
                    redis_client.hset(f'batch_job:{job_id}', 'status', 'COMPLETED')
                    logger.warning(f"Task {self.request.id}: Batch {job_id} completed with {completed} sequences (some may have failed)")

            except Exception as redis_error:
                logger.error(f"Task {self.request.id}: Failed to update Redis error state: {redis_error}")

        raise
