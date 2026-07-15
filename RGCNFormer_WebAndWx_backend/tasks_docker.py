"""
tasks_docker.py - Celery 异步任务(Docker 环境) / Celery async tasks (Docker)

与 tasks.py 结构相同,但使用 config_docker.py(REDIS_HOST 默认指向 docker 服务名
'redis',而 'localhost')。在 docker-compose 容器中跑 Celery worker 时使用。 / Same
as tasks.py but uses config_docker.py (REDIS_HOST defaults to docker service
'redis' instead of 'localhost'). Use when running the Celery worker in the
docker-compose stack.

功能模块 / Modules:
- celery_app: Celery 实例,broker/backend 指向 docker 'redis' / Celery app, points to docker 'redis'
- run_prediction_task(job_id, sequence): 异步推理任务 / Async inference task

输入 / Inputs:
- job_id: str - 任务 ID / Task ID
- sequence: str - RNA 序列 / RNA sequence

输出 / Outputs:
- Redis 缓存 + 任务状态 / Redis cache + task state

数据流 / Data Flow:
1. 容器内 server.py 投递任务 / server.py dispatches (inside container)
2. celery worker(可能同一容器,可能独立容器)消费任务 / Celery worker consumes
3. 写结果到 'redis' 服务的 Redis / Write to 'redis' service
4. 轮询接口通过 docker DNS 读取结果 / Polling reads via docker DNS

相关文件 / Related Files:
- 调用 / Calls: main_model、human、common、config_docker
- 被调用 / Called by: server.py(在 docker 容器中运行时) / server.py (in Docker)

使用示例 / Usage Example:
    # docker-compose up 后,Celery worker 自动启动
    docker-compose logs -f celery

作者 / Author: 项目组 / Project Team
版本 / Version: 1.0
"""
from celery import Celery
import redis
import json
import hashlib
import uuid
import torch
import numpy as np
from torch_geometric.data import Batch

# Import local modules
from main_model import RNA_ClassQuery_Model
from human import run_linearfold, build_edge_index_from_structure, MOD_NAMES
from common import INDEX_TO_NUCLEOTIDE
from attention_distribution import (
    attention_distribution_cache_key,
    build_attention_distribution,
)
from config_docker import config, get_logger

# ============================================================================
# Celery Application Configuration
# ============================================================================

# Initialize Celery app with Redis as both broker and backend
celery_app = Celery(
    'rna_prediction_tasks_docker', 
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND
)

# Optional: Configure Celery settings
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
    # Test connection
    redis_client.ping()
    logger.info("Connected to Redis successfully")
except Exception as e:
    logger.warning(f"Could not connect to Redis: {e}")
    redis_client = None

# ============================================================================
# Model Loading (Load once at worker startup)
# ============================================================================

logger.info("Loading model and configuration...")

# Load configuration
with open(config.MODEL_CONFIG_PATH, 'r') as f:
    model_config_file = json.load(f)

model_cfg = model_config_file['model']

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

# ============================================================================
# Celery Task Definition
# ============================================================================

@celery_app.task(name='tasks.run_prediction_task', bind=True)
def run_prediction_task(self, original_sequence, target_class_id=None, top_k=None):
    """
    Celery task for running RNA prediction in the background.

    This task performs the following steps:
    1. Preprocesses the RNA sequence (pad/truncate to 1001)
    2. Runs LinearFold to get secondary structure
    3. Builds edge index from structure
    4. Runs model inference
    5. Applies hierarchical pruning
    6. Generates classification tree
    7. Extracts attention weights for top-K sites
    8. Builds GCN graph data
    9. Stores result in Redis cache
    10. Returns the complete prediction result

    Args:
        original_sequence (str): The original RNA sequence
        target_class_id (int, optional): Specific class ID for attention visualization
        top_k (int, optional): Number of top sites to display (default: 3)

    Returns:
        dict: Complete prediction result containing:
            - jobId (str): SHA256 hash of the sequence
            - status (str): "completed"
            - classification (dict): Hierarchical classification tree
            - attention (dict): Attention weights for top-K sites
            - gcn (dict): GCN graph nodes and edges
    """
    logger.info(f"Task {self.request.id}: Starting prediction for sequence length: {len(original_sequence)}")

    try:
        # Generate job_id from sequence (SHA256 hash)
        job_id = hashlib.sha256(original_sequence.encode('utf-8')).hexdigest()
        logger.info(f"Task {self.request.id}: Generated job_id: {job_id}")

        # Store original sequence for response
        sequence = original_sequence

        # For shorter sequences, pad to 1001; for longer sequences, truncate
        # The model expects exactly 1001 nucleotides
        TARGET_LENGTH = 1001
        seq_len = len(sequence)

        # Track padding/trimming for index remapping
        left_padding = 0
        left_trimming = 0

        if seq_len != TARGET_LENGTH:
            if seq_len < TARGET_LENGTH:
                # Pad with 'N' (unknown nucleotide) at the center
                padding_needed = TARGET_LENGTH - seq_len
                left_pad = padding_needed // 2
                right_pad = padding_needed - left_pad
                left_padding = left_pad
                sequence = 'N' * left_pad + sequence + 'N' * right_pad
                logger.info(f"Task {self.request.id}: Padded sequence to {TARGET_LENGTH} with {padding_needed} 'N's (left: {left_pad}, right: {right_pad})")
            else:
                # Truncate from both sides (center the sequence)
                excess = seq_len - TARGET_LENGTH
                left_trim = excess // 2
                right_trim = excess - left_trim
                left_trimming = left_trim
                sequence = sequence[left_trim:seq_len - right_trim]
                logger.info(f"Task {self.request.id}: Trimmed sequence from {seq_len} to {TARGET_LENGTH} (left: {left_trim}, right: {right_trim})")

        # Step 1: Call LinearFold to get secondary structure
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

        # Step 2: Build edge index from structure
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

        # Step 3: Prepare model input
        # Convert sequence to one-hot encoding
        try:
            x = one_hot_encode_sequence(sequence)
            if x.shape != (len(sequence), 4):
                raise ValueError(f"Invalid one-hot encoding shape: expected ({len(sequence)}, 4), got {x.shape}")
            x = torch.FloatTensor(x)  # Shape: [1001, 4]

            # Create batch tensor (single sample)
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

        # Step 4: Run model inference
        logger.info(f"Task {self.request.id}: Running model inference...")
        try:
            with torch.no_grad():
                # Create Batch object
                data_batch = Batch(x=x, edge_index=edge_index, batch=batch)

                # Move to device
                data_batch = data_batch.to(device)

                # Get predictions
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

            # Validate outputs
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

        # Step 5: Process and format output
        # Move to CPU and convert to numpy
        logits_12class = logits_12class.cpu().numpy()[0]  # [12]
        probs_12class = 1 / (1 + np.exp(-logits_12class))  # Sigmoid

        if model_cfg['use_hierarchical']:
            logits_4class = logits_4class.cpu().numpy()[0]  # [4]
            probs_4class = 1 / (1 + np.exp(-logits_4class))  # Sigmoid
            attn_weights = attn_weights.cpu().numpy()[0]  # [12, 1001]

        # Build classification tree
        group_names = ['A', 'C', 'G', 'U']
        group_to_classes = {
            'A': [0, 1, 7, 9, 10],
            'C': [2, 6, 8],
            'G': [3, 11],
            'U': [4, 5]
        }

        # Reverse mapping: class index to group name
        class_to_group = {}
        for group_name, class_indices in group_to_classes.items():
            for class_idx in class_indices:
                class_to_group[class_idx] = group_name

        # Use thresholds from configuration
        thresholds_12class = config.THRESHOLDS_12_CLASS
        thresholds_4class = config.THRESHOLDS_4_CLASS

        # Step 5.1: Get initial predictions for 12-class and 4-class
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
            # If not hierarchical, all groups are False by default
            for group_idx in range(4):
                predictions_4class[group_idx] = False

        # Step 5.2: Apply hierarchical pruning (bottom-up)
        # Rule 1: If all children of a group are False, set the group to False
        for group_idx, group_name in enumerate(group_names):
            child_indices = group_to_classes[group_name]
            # Check if any child is predicted True
            has_any_child = any(predictions_12class[class_idx] for class_idx in child_indices)
            if not has_any_child:
                predictions_4class[group_idx] = False

        # Step 5.3: Apply hierarchical pruning (top-down)
        # Rule 2: If a group is False, set all its children to False
        for group_idx, group_name in enumerate(group_names):
            if not predictions_4class[group_idx]:
                child_indices = group_to_classes[group_name]
                for class_idx in child_indices:
                    predictions_12class[class_idx] = False

        # Step 5.4: Build classification tree with pruned predictions
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
                # Add all modification child nodes, regardless of prediction
                children.append({
                    "name": MOD_NAMES.get(class_idx, f"Class{class_idx}"),
                    "isPredicted": class_predicted
                })

            classification["children"].append({
                "name": f"Group {group_name}",
                "isPredicted": group_predicted,
                "children": children
            })

        # Step 5.5: Identify active nucleotide groups based on pruned predictions
        # Find which nucleotide groups have at least one predicted modification
        active_groups = set()
        for group_idx, is_predicted in predictions_4class.items():
            if is_predicted:
                active_groups.add(group_names[group_idx])

        logger.info(f"Task {self.request.id}: Active groups after pruning: {active_groups}")

        # Build attention data
        attention_data = {
            "sequence": original_sequence,  # Return original sequence
            "weights": []
        }

        if attn_weights is not None and active_groups:
            # Step 5.6: Calculate combined attention for predicted classes
            # If target_class_id is specified, use only that class's attention
            # Otherwise, use combined attention for all predicted classes
            if target_class_id is not None and 0 <= target_class_id < 12:
                # Use attention weights for the specific target class
                combined_attention = attn_weights[target_class_id]
                predicted_class_indices = [target_class_id]
                logger.info(f"Task {self.request.id}: Using attention weights for class {target_class_id}")

                # Get the nucleotide group for the target class
                target_nucleotide = INDEX_TO_NUCLEOTIDE.get(target_class_id)
                if target_nucleotide:
                    active_groups = {target_nucleotide}
                    logger.info(f"Task {self.request.id}: Filtering to nucleotide group: {target_nucleotide} for class {target_class_id}")
                else:
                    logger.warning(f"Task {self.request.id}: Warning: No nucleotide mapping found for class {target_class_id}")
                    active_groups = set()
            else:
                # Default: combine attention for all predicted classes
                predicted_class_indices = [idx for idx, pred in predictions_12class.items() if pred]
                combined_attention = np.zeros(len(sequence))
                if predicted_class_indices:
                    for class_idx in predicted_class_indices:
                        combined_attention += attn_weights[class_idx]
                    combined_attention /= len(predicted_class_indices)

            # Step 5.7: Per-group Top-K selection
            # Use top_k from request if provided, otherwise default to 3
            K = top_k if top_k is not None else 3
            all_top_sites = []

            for group in active_groups:
                # Create a mask specifically for the current group's nucleotides
                # Note: T and U are equivalent in RNA
                if group == 'U':
                    # For U group, include both U and T nucleotides
                    group_mask = np.array([1.0 if (nucleotide == 'U' or nucleotide == 'T') else 0.0 for nucleotide in sequence.upper()])
                else:
                    # For other groups, exclude T (since T belongs to U group)
                    group_mask = np.array([1.0 if (nucleotide == group and nucleotide != 'T') else 0.0 for nucleotide in sequence.upper()])

                logger.info(f"Task {self.request.id}: Processing group '{group}':")
                logger.info(f"Task {self.request.id}:   - Nucleotides in sequence: {np.sum(group_mask)}")
                logger.info(f"Task {self.request.id}:   - Combined attention shape: {combined_attention.shape}")

                # Determine how many sites to get for this group (min of K and available sites)
                num_available_sites = int(group_mask.sum())
                top_k_for_group = min(K, num_available_sites)

                logger.info(f"Task {self.request.id}:   - Top K for group: {top_k_for_group}")

                # Skip if no nucleotides of this type in sequence
                if top_k_for_group <= 0:
                    logger.info(f"Task {self.request.id}:   - Skipping group '{group}': no nucleotides found in sequence")
                    continue

                # Apply the group mask to the attention weights
                group_attention = combined_attention * group_mask

                # Set non-group positions to a very low value so they are not selected
                group_attention[group_mask == 0] = -np.inf

                logger.info(f"Task {self.request.id}:   - Max attention in group: {np.max(group_attention[group_mask == 1])}")

                if top_k_for_group > 0:
                    # Get the indices of the top-k highest scores for this group
                    top_indices_for_group = np.argsort(group_attention)[-top_k_for_group:][::-1]

                    for pos in top_indices_for_group:
                        pos_int = int(pos)
                        score_float = float(combined_attention[pos])  # Use original combined attention for score
                        original_index = pos_int - left_padding + left_trimming

                        # Ensure the site is within the original sequence bounds
                        if 0 <= original_index < len(original_sequence):
                            all_top_sites.append({
                                "index": original_index,
                                "type": group,  # Label with the group name
                                "score": score_float
                            })

            # Sort all collected sites by score in descending order
            all_top_sites.sort(key=lambda x: x["score"], reverse=True)
            attention_data["weights"] = all_top_sites

        # Build GCN graph data
        # Limit to original sequence for visualization
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

            # Only keep one direction (source < target) to avoid duplicates in visualization
            if source >= target:
                continue

            nuc_source = original_sequence[orig_source]
            nuc_target = original_sequence[orig_target]
            edges.append({
                "source": f"{nuc_source}{orig_source}",
                "target": f"{nuc_target}{orig_target}"
            })

        logger.info(f"Task {self.request.id}: GCN visualization stats: total edges={len(edges)}")

        # Create nodes (use all nodes from original sequence)
        nodes = []
        for i in range(len(original_sequence)):
            nuc = original_sequence[i]
            nodes.append({
                "id": f"{nuc}{i}",
                "label": f"位置{i}: {nuc}",
                "data": {"index": i, "type": nuc, "name": f"{'腺嘌呤' if nuc == 'A' else '胞嘧啶' if nuc == 'C' else '鸟嘌呤' if nuc == 'G' else '尿嘧啶'}"}
            })

        # Create a set of valid node IDs for filtering edges
        valid_node_ids = {node["id"] for node in nodes}

        # Filter edges to only include those that reference valid nodes
        valid_edges = [
            edge for edge in edges
            if edge["source"] in valid_node_ids and edge["target"] in valid_node_ids
        ]

        logger.info(f"Task {self.request.id}: GCN visualization stats: nodes={len(nodes)}, valid edges={len(valid_edges)}")

        gcn_data = {
            "nodes": nodes,
            "edges": valid_edges
        }

        # Build final response
        response = {
            "jobId": job_id,
            "status": "completed",
            "classification": classification,
            "attention": attention_data,
            "gcn": gcn_data
        }

        # Store full attention separately so normal task responses stay compact.
        if redis_client:
            if attn_weights is not None:
                try:
                    attention_distribution = build_attention_distribution(
                        original_sequence=original_sequence,
                        attn_weights=attn_weights,
                        probs_12class=probs_12class,
                        predictions_12class=predictions_12class,
                        thresholds_12class=thresholds_12class,
                        class_names=[MOD_NAMES.get(i, f"Class{i}") for i in range(12)],
                        left_padding=left_padding,
                        left_trimming=left_trimming,
                    )
                    attention_distribution["job_id"] = job_id
                    redis_client.setex(
                        attention_distribution_cache_key(job_id),
                        config.REDIS_CACHE_TTL,
                        json.dumps(attention_distribution, ensure_ascii=False),
                    )
                except Exception as e:
                    logger.error(
                        f"Task {self.request.id}: Failed to cache attention distribution "
                        f"for job_id {job_id}: {e}"
                    )

            # Store the existing compact task result independently.
            try:
                # Serialize response to JSON
                response_json = json.dumps(response, ensure_ascii=False)
                # Store in Redis with job_id as key
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
                # Don't fail the task if caching fails, just log the error

        logger.info(f"Task {self.request.id}: Prediction completed successfully")
        return response

    except Exception as e:
        import traceback
        # Build structured error information
        error_info = {
            "task_id": self.request.id,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "sequence_length": len(original_sequence) if original_sequence else 0
        }

        # Log the full error details
        logger.error(f"Task {self.request.id}: Task failed with error: {error_info['error_type']} - {error_info['error_message']}")
        logger.debug(f"Task {self.request.id}: Full traceback:\n{error_info['traceback']}")

        # Create a custom exception with structured information
        # This will be available in Celery's result backend
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
    Celery task for processing a single sequence within a batch.
    Updates Redis state after completion.
    
    Args:
        job_id (str): Batch job ID
        sequence_data (str): RNA sequence to process
        index (int): Index of this sequence in the batch (0-4)
        target_class_id (int, optional): Specific class ID for attention visualization
        top_k (int, optional): Number of top sites to display
    
    Returns:
        dict: Complete prediction result for this sequence
    """
    logger.info(f"Task {self.request.id}: Processing sequence {index} for batch {job_id}")
    
    try:
        # Call the existing prediction task
        result = run_prediction_task(sequence_data, target_class_id, top_k)
        
        # Update Redis state for this batch
        if redis_client:
            try:
                # Get current results list
                results_json = redis_client.hget(f'batch_job:{job_id}', 'results')
                results = json.loads(results_json) if results_json else []
                
                # Append result with index
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
                
                # Update results in Redis
                redis_client.hset(f'batch_job:{job_id}', 'results', json.dumps(results, ensure_ascii=False))
                
                # Increment completed_sequences counter atomically
                completed = redis_client.hincrby(f'batch_job:{job_id}', 'completed_sequences', 1)
                logger.info(f"Task {self.request.id}: Sequence {index} completed. Total completed: {completed}")
                
                # Check if all sequences are done
                total_sequences = int(redis_client.hget(f'batch_job:{job_id}', 'total_sequences') or 0)
                if completed >= total_sequences:
                    # Update status to COMPLETED
                    redis_client.hset(f'batch_job:{job_id}', 'status', 'COMPLETED')
                    logger.info(f"Task {self.request.id}: Batch {job_id} fully completed ({completed}/{total_sequences} sequences)")
                
            except Exception as e:
                logger.error(f"Task {self.request.id}: Failed to update Redis state for batch {job_id}: {e}")
                # Don't fail the task if Redis update fails
        
        return result
        
    except Exception as e:
        # Handle failure - still update Redis to mark this sequence as failed
        if redis_client:
            try:
                results_json = redis_client.hget(f'batch_job:{job_id}', 'results')
                results = json.loads(results_json) if results_json else []
                
                # Append error result
                result_with_index = {
                    "index": index,
                    "sequence": sequence_data,
                    "status": "failed",
                    "error": str(e)
                }
                results.append(result_with_index)
                
                # Update results in Redis
                redis_client.hset(f'batch_job:{job_id}', 'results', json.dumps(results, ensure_ascii=False))
                
                # Increment completed_sequences counter (even if failed)
                completed = redis_client.hincrby(f'batch_job:{job_id}', 'completed_sequences', 1)
                
                # Check if all sequences are done (including failures)
                total_sequences = int(redis_client.hget(f'batch_job:{job_id}', 'total_sequences') or 0)
                if completed >= total_sequences:
                    # Update status to COMPLETED (even with some failures)
                    redis_client.hset(f'batch_job:{job_id}', 'status', 'COMPLETED')
                    logger.warning(f"Task {self.request.id}: Batch {job_id} completed with {completed} sequences (some may have failed)")
                
            except Exception as redis_error:
                logger.error(f"Task {self.request.id}: Failed to update Redis error state: {redis_error}")
        
        # Re-raise the exception
        raise
