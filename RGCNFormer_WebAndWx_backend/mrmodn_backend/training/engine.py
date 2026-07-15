"""
中文：训练与评估引擎。提供 train_epoch()、test_epoch() 和注意力监督损失计算。
English: Training and evaluation engine. Provides train_epoch(), test_epoch(), and attention supervision loss computation.
"""
import torch
import logging
from typing import Optional
from torch_geometric.loader import DataLoader

from mrmodn_backend.core.constants import GROUP_TO_CLASS_INDICES, LABEL_MAPPING


def compute_attention_supervision_loss(
    attn_weights: torch.Tensor,
    y_site: torch.Tensor,
    num_classes: int = 12,
    seq_len: int = 1001
) -> torch.Tensor:
    """
    中文：使用 KL 散度计算注意力监督损失。引导模型关注实际发生修饰的位点。
    English: Compute attention supervision loss using KL divergence. Encourages the model to attend to positions where modifications actually occur.
    """
    import torch.nn.functional as F

    batch_size = attn_weights.size(0)
    device = attn_weights.device

    y_site_reshaped = y_site.view(batch_size, seq_len)

    loss_attn = 0.0
    num_valid_classes = 0

    for class_idx in range(num_classes):
        original_label_id = None
        for k, v in LABEL_MAPPING.items():
            if v == class_idx:
                original_label_id = k
                break

        if original_label_id is None:
            continue

        target_mask = (y_site_reshaped == original_label_id).float()

        has_mod_mask = target_mask.sum(dim=1) > 0

        if has_mod_mask.sum() > 0:
            num_valid_classes += 1

            pred_attn = attn_weights[has_mod_mask, class_idx, :]
            target_mask_filtered = target_mask[has_mod_mask]

            target_dist = target_mask_filtered / (target_mask_filtered.sum(dim=1, keepdim=True) + 1e-10)

            pred_log_dist = torch.log(pred_attn + 1e-10)

            loss_kl = F.kl_div(pred_log_dist, target_dist, reduction='batchmean')

            loss_attn += loss_kl

    if num_valid_classes > 0:
        loss_attn = loss_attn / num_valid_classes
    else:
        loss_attn = torch.tensor(0.0, device=device)

    return loss_attn


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    use_hierarchical: bool = False,
    use_amp: bool = False,
    use_attention_supervision: bool = False,
    attention_lambda: float = 1.0
) -> float:
    """
    中文：单轮训练。支持分层多任务学习、混合精度训练和注意力监督损失。
    English: Single training epoch. Supports hierarchical multi-task learning, automatic mixed precision, and attention supervision loss.
    """
    from tqdm import tqdm
    import torch.nn.functional as F

    model.train()
    total_loss = 0.0
    total_loss_12 = 0.0
    total_loss_4 = 0.0
    total_loss_attn = 0.0
    num_batches = 0

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    pbar = tqdm(dataloader, desc="Training", leave=True)
    for batch in pbar:
        if not isinstance(batch.y, torch.Tensor):
            batch.y = torch.tensor(batch.y, dtype=torch.float32)

        batch = batch.to(device)
        batch.y = batch.y.to(device)

        optimizer.zero_grad()
        
        should_return_attention = use_attention_supervision and hasattr(batch, 'y_site')

        if use_amp:
            with torch.cuda.amp.autocast():
                if use_hierarchical:
                    if should_return_attention:
                        logits_12, logits_4, attn_weights = model(
                            batch.x, batch.edge_index, batch.batch, return_attention=True
                        )
                    else:
                        logits_12, logits_4 = model(
                            batch.x, batch.edge_index, batch.batch, return_attention=False
                        )

                    y_12 = batch.y
                    y_4 = torch.zeros(y_12.size(0), 4, device=y_12.device)

                    for group_idx, group_name in enumerate(['A', 'C', 'G', 'U']):
                        class_indices = GROUP_TO_CLASS_INDICES[group_name]
                        y_4[:, group_idx] = y_12[:, class_indices].max(dim=1)[0]

                    loss_12 = criterion(logits_12, y_12)
                    loss_4 = F.binary_cross_entropy_with_logits(logits_4, y_4)
                    
                    loss_attn = torch.tensor(0.0, device=device)
                    if should_return_attention:
                        loss_attn = compute_attention_supervision_loss(attn_weights, batch.y_site)
                        total_loss_attn += loss_attn.item()

                    loss = loss_12 + loss_4 + (attention_lambda * loss_attn)

                    total_loss_12 += loss_12.item()
                    total_loss_4 += loss_4.item()
                    
                else:
                    if should_return_attention:
                        logits, attn_weights = model(batch.x, batch.edge_index, batch.batch, return_attention=True)
                        loss_cls = criterion(logits, batch.y)
                        loss_attn = compute_attention_supervision_loss(attn_weights, batch.y_site)
                        loss = loss_cls + attention_lambda * loss_attn
                        total_loss_attn += loss_attn.item()
                    else:
                        logits = model(batch.x, batch.edge_index, batch.batch)
                        loss = criterion(logits, batch.y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        else:
            if use_hierarchical:
                if should_return_attention:
                    logits_12, logits_4, attn_weights = model(
                        batch.x, batch.edge_index, batch.batch, return_attention=True
                    )
                else:
                    logits_12, logits_4 = model(
                        batch.x, batch.edge_index, batch.batch, return_attention=False
                    )

                y_12 = batch.y
                y_4 = torch.zeros(y_12.size(0), 4, device=y_12.device)

                for group_idx, group_name in enumerate(['A', 'C', 'G', 'U']):
                    class_indices = GROUP_TO_CLASS_INDICES[group_name]
                    y_4[:, group_idx] = y_12[:, class_indices].max(dim=1)[0]

                loss_12 = criterion(logits_12, y_12)
                loss_4 = F.binary_cross_entropy_with_logits(logits_4, y_4)
                
                loss_attn = torch.tensor(0.0, device=device)
                if should_return_attention:
                    loss_attn = compute_attention_supervision_loss(attn_weights, batch.y_site)
                    total_loss_attn += loss_attn.item()

                loss = loss_12 + loss_4 + (attention_lambda * loss_attn)

                total_loss_12 += loss_12.item()
                total_loss_4 += loss_4.item()
                
            else:
                if should_return_attention:
                    logits, attn_weights = model(batch.x, batch.edge_index, batch.batch, return_attention=True)
                    loss_cls = criterion(logits, batch.y)
                    loss_attn = compute_attention_supervision_loss(attn_weights, batch.y_site)
                    loss = loss_cls + attention_lambda * loss_attn
                    total_loss_attn += loss_attn.item()
                else:
                    logits = model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(logits, batch.y)

            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        postfix_dict = {"loss": f"{loss.item():.4f}"}
        
        if use_hierarchical:
            postfix_dict["l12"] = f"{loss_12.item():.4f}"
            postfix_dict["l4"] = f"{loss_4.item():.4f}"
            
        if should_return_attention and total_loss_attn > 0:
            postfix_dict["lattn"] = f"{loss_attn.item():.4f}"
            
        pbar.set_postfix(postfix_dict)

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / num_batches
    msg_parts = [f"Train loss: {avg_loss:.4f}"]
    
    if use_hierarchical:
        avg_loss_12 = total_loss_12 / num_batches
        avg_loss_4 = total_loss_4 / num_batches
        msg_parts.append(f"12-class: {avg_loss_12:.4f}")
        msg_parts.append(f"4-class: {avg_loss_4:.4f}")
        
    if use_attention_supervision and total_loss_attn > 0:
        avg_loss_attn = total_loss_attn / num_batches
        msg_parts.append(f"attn: {avg_loss_attn:.4f}")
        
    msg = ", ".join(msg_parts)

    if logger:
        logger.info(msg)
    else:
        print(msg)

    return avg_loss


def test_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    phase: str = "test",
    logger: Optional[logging.Logger] = None,
    use_hierarchical: bool = False,
    use_amp: bool = False
) -> float:
    """
    中文：单轮评估。在无梯度模式下运行模型，计算验证/测试集损失。
    English: Single evaluation epoch. Runs the model in no-grad mode and computes validation/test loss.
    """
    from tqdm import tqdm
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    total_loss_12 = 0.0
    total_loss_4 = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"{phase.capitalize()}", leave=True)
    if use_amp:
        with torch.no_grad():
            for batch in pbar:
                if not isinstance(batch.y, torch.Tensor):
                    batch.y = torch.tensor(batch.y, dtype=torch.float32)

                batch = batch.to(device)
                batch.y = batch.y.to(device)

                with torch.cuda.amp.autocast():
                    if use_hierarchical:
                        logits_12, logits_4, _ = model(batch.x, batch.edge_index, batch.batch)

                        y_12 = batch.y
                        y_4 = torch.zeros(y_12.size(0), 4, device=y_12.device)

                        for group_idx, group_name in enumerate(['A', 'C', 'G', 'U']):
                            class_indices = GROUP_TO_CLASS_INDICES[group_name]
                            y_4[:, group_idx] = y_12[:, class_indices].max(dim=1)[0]

                        loss_12 = criterion(logits_12, y_12)
                        loss_4 = F.binary_cross_entropy_with_logits(logits_4, y_4)
                        loss = loss_12 + loss_4

                        total_loss_12 += loss_12.item()
                        total_loss_4 += loss_4.item()
                    else:
                        logits = model(batch.x, batch.edge_index, batch.batch)
                        loss = criterion(logits, batch.y)

                total_loss += loss.item()
                num_batches += 1

                if use_hierarchical:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "loss_12": f"{loss_12.item():.4f}", "loss_4": f"{loss_4.item():.4f}"})
                else:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    else:
        with torch.no_grad():
            for batch in pbar:
                if not isinstance(batch.y, torch.Tensor):
                    batch.y = torch.tensor(batch.y, dtype=torch.float32)

                batch = batch.to(device)
                batch.y = batch.y.to(device)

                if use_hierarchical:
                    logits_12, logits_4 = model(batch.x, batch.edge_index, batch.batch)

                    y_12 = batch.y
                    y_4 = torch.zeros(y_12.size(0), 4, device=y_12.device)

                    for group_idx, group_name in enumerate(['A', 'C', 'G', 'U']):
                        class_indices = GROUP_TO_CLASS_INDICES[group_name]
                        y_4[:, group_idx] = y_12[:, class_indices].max(dim=1)[0]

                    loss_12 = criterion(logits_12, y_12)
                    loss_4 = F.binary_cross_entropy_with_logits(logits_4, y_4)
                    loss = loss_12 + loss_4

                    total_loss_12 += loss_12.item()
                    total_loss_4 += loss_4.item()
                else:
                    logits = model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(logits, batch.y)

                total_loss += loss.item()
                num_batches += 1

                if use_hierarchical:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "loss_12": f"{loss_12.item():.4f}", "loss_4": f"{loss_4.item():.4f}"})
                else:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    if use_hierarchical:
        avg_loss_12 = total_loss_12 / num_batches
        avg_loss_4 = total_loss_4 / num_batches
        msg = f"{phase.capitalize()} loss: {avg_loss:.4f} (12-class: {avg_loss_12:.4f}, 4-class: {avg_loss_4:.4f})"
    else:
        msg = f"{phase.capitalize()} loss: {avg_loss:.4f}"

    if logger:
        logger.info(msg)
    else:
        print(msg)

    return avg_loss
