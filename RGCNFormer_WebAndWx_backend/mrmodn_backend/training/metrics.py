"""
中文：RNA 位点定位评估指标。提供 Top-K 召回率、mAP、MRR、R-Precision、NDCG 和 MDE 等综合评估指标。
English: RNA site localization evaluation metrics. Provides Top-K recall, mAP, MRR, R-Precision, NDCG, and MDE comprehensive metrics.
"""
import torch
import numpy as np

from mrmodn_backend.core.constants import LABEL_MAPPING, MOD_NAMES


def calculate_topk_recall(
    attn_weights: torch.Tensor,
    y_site: torch.Tensor,
    k_list: list = [1, 5, 10, 20, 50],
    num_classes: int = 12,
    seq_len: int = 1001
) -> dict:
    """
    Calculate Top-K site recall based on attention weights.

    For each class, compute the recall rate: how many of the true modification
    sites appear in the top-K positions ranked by attention weights.

    Args:
        attn_weights: Attention weights [N, Num_Classes, Seq_Len]
        y_site: Site-level labels [N * Seq_Len] or [N, Seq_Len]
        k_list: List of K values for top-K recall
        num_classes: Number of classes (default 12)
        seq_len: Sequence length (default 1001)

    Returns:
        dict: {class_idx: {k: recall_value}}
    """
    attn_weights = attn_weights.detach().cpu().numpy()
    y_site = y_site.detach().cpu().numpy()

    if y_site.ndim == 1:
        batch_size = attn_weights.shape[0]
        y_site = y_site.reshape(batch_size, seq_len)

    results = {}
    max_k = max(k_list)

    for class_idx in range(num_classes):
        original_label_id = None
        for k, v in LABEL_MAPPING.items():
            if v == class_idx:
                original_label_id = k
                break

        if original_label_id is None:
            results[class_idx] = {k: 0.0 for k in k_list}
            continue

        has_mod_samples = np.any(y_site == original_label_id, axis=1)

        if np.sum(has_mod_samples) == 0:
            results[class_idx] = {k: 0.0 for k in k_list}
            continue

        target_attn = attn_weights[has_mod_samples, class_idx, :]
        target_labels = y_site[has_mod_samples]

        class_recalls = {k: [] for k in k_list}

        for i in range(len(target_attn)):
            true_indices = np.where(target_labels[i] == original_label_id)[0]
            num_true = len(true_indices)

            if num_true == 0:
                continue

            pred_indices = np.argsort(-target_attn[i])[:max_k]

            for k in k_list:
                topk_pred = pred_indices[:k]
                hit_count = len(np.intersect1d(topk_pred, true_indices))
                recall = hit_count / num_true
                class_recalls[k].append(recall)

        results[class_idx] = {}
        for k in k_list:
            if len(class_recalls[k]) > 0:
                results[class_idx][k] = np.mean(class_recalls[k])
            else:
                results[class_idx][k] = 0.0

    return results


def print_topk_table(
    topk_results: dict,
    k_list: list = [1, 5, 10, 20, 50],
    logger=None
):
    """
    Print Top-K site recall results in a formatted table.

    Args:
        topk_results: Results from calculate_topk_recall
        k_list: List of K values
        logger: Optional logger instance
    """
    from prettytable import PrettyTable

    output = f"\n{'='*80}\n"
    output += f"Top-K Site Localization Recall (Attention Analysis - Macro-Average)\n"
    output += f"{'='*80}\n"

    table = PrettyTable()
    field_names = ["Class", "Name"] + [f"Top-{k}" for k in k_list]
    table.field_names = field_names
    table.align = "r"
    table.align["Class"] = "l"
    table.align["Name"] = "l"

    for c in range(12):
        row = [c, MOD_NAMES.get(c, str(c))]
        metrics = topk_results.get(c, {})
        for k in k_list:
            rec = metrics.get(k, 0.0)
            row.append(f"{rec:.4f}")
        table.add_row(row)

    output += str(table) + "\n"

    print(output)
    if logger:
        logger.info(output)


def calculate_comprehensive_localization_metrics(
    attn_weights: torch.Tensor,
    y_site: torch.Tensor,
    k_list: list = [1, 3, 5, 10],
    num_classes: int = 12,
    seq_len: int = 1001
) -> dict:
    """
    Calculate comprehensive localization metrics based on attention weights.
    This function computes multiple evaluation metrics:
    1.  Global Recall@K (Micro-Average): Sum of all hits / Sum of all true sites
    2.  Mean Average Precision (mAP): Average AP across all valid samples
    3.  Mean Reciprocal Rank (MRR): Average of 1/Rank for first correct prediction
    4.  R-Precision: Precision at R, where R is the number of true sites for the sample.
    5.  NDCG@K: Normalized Discounted Cumulative Gain, measuring ranking quality.
    6.  MDE (Mean Distance Error): Average distance of Top-1 false positives to the nearest true site.
    Args:
        attn_weights: Attention weights [N, Num_Classes, Seq_Len]
        y_site: Site-level labels [N * Seq_Len] or [N, Seq_Len]
        k_list: List of K values for top-K recall/precision
        num_classes: Number of classes (default 12)
        seq_len: Sequence length (default 1001)
    Returns:
        dict: {class_idx: {'mAP': float, 'MRR': float, 'R@K': float, 'R-Precision': float, 'NDCG@K': float, 'MDE': float}}
    """
    attn_weights = attn_weights.detach().cpu().numpy()
    y_site = y_site.detach().cpu().numpy()

    if y_site.ndim == 1:
        batch_size = attn_weights.shape[0]
        y_site = y_site.reshape(batch_size, seq_len)

    results = {}

    for class_idx in range(num_classes):
        original_label_id = None
        for k, v in LABEL_MAPPING.items():
            if v == class_idx:
                original_label_id = k
                break

        if original_label_id is None:
            results[class_idx] = {
                'mAP': 0.0, 'MRR': 0.0, 'R-Precision': 0.0, 'MDE': 0.0
            }
            for k in k_list:
                results[class_idx][f'R@{k}'] = 0.0
                results[class_idx][f'NDCG@{k}'] = 0.0
            continue

        has_mod_samples = np.any(y_site == original_label_id, axis=1)

        if np.sum(has_mod_samples) == 0:
            results[class_idx] = {
                'mAP': 0.0, 'MRR': 0.0, 'R-Precision': 0.0, 'MDE': 0.0
            }
            for k in k_list:
                results[class_idx][f'R@{k}'] = 0.0
                results[class_idx][f'NDCG@{k}'] = 0.0
            continue

        target_attn = attn_weights[has_mod_samples, class_idx, :]
        target_labels = y_site[has_mod_samples]

        ap_list, mrr_list, r_precision_list, mde_list = [], [], [], []
        ndcg_scores = {k: [] for k in k_list}
        global_hits = {k: 0 for k in k_list}
        global_true_count = 0

        for i in range(len(target_attn)):
            true_indices = np.where(target_labels[i] == original_label_id)[0]
            num_true = len(true_indices)

            if num_true == 0:
                continue

            global_true_count += num_true
            pred_ranks = np.argsort(-target_attn[i])

            r = num_true
            top_r_preds = pred_ranks[:r]
            r_precision_hits = np.sum(np.isin(top_r_preds, true_indices))
            r_precision_list.append(r_precision_hits / r)

            relevance = np.zeros_like(target_attn[i])
            relevance[true_indices] = 1
            
            def dcg_at_k(r, k):
                r = np.asarray(r)[:k]
                if r.size:
                    return np.sum(r / np.log2(np.arange(2, r.size + 2)))
                return 0.

            for k in k_list:
                pred_rel_at_k = relevance[pred_ranks]
                dcg_val = dcg_at_k(pred_rel_at_k, k)
                
                ideal_rel_at_k = np.sort(relevance)[::-1]
                idcg_val = dcg_at_k(ideal_rel_at_k, k)

                if idcg_val > 0:
                    ndcg_scores[k].append(dcg_val / idcg_val)

            top_1_pred = pred_ranks[0]
            if top_1_pred not in true_indices:
                distances = np.abs(true_indices - top_1_pred)
                mde_list.append(np.min(distances))

            precisions_at_k = []
            for rank_idx, pred_pos in enumerate(pred_ranks):
                if pred_pos in true_indices:
                    hit_count = np.sum(np.isin(pred_ranks[:rank_idx+1], true_indices))
                    precisions_at_k.append(hit_count / (rank_idx + 1))
            
            if precisions_at_k:
                ap_list.append(np.mean(precisions_at_k))

            for rank, pos_idx in enumerate(pred_ranks):
                if pos_idx in true_indices:
                    mrr_list.append(1.0 / (rank + 1))
                    break
            
            for k in k_list:
                hit_count = len(np.intersect1d(pred_ranks[:k], true_indices))
                global_hits[k] += hit_count

        class_results = {}
        class_results['mAP'] = np.mean(ap_list) if ap_list else 0.0
        class_results['MRR'] = np.mean(mrr_list) if mrr_list else 0.0
        class_results['R-Precision'] = np.mean(r_precision_list) if r_precision_list else 0.0
        class_results['MDE'] = np.mean(mde_list) if mde_list else 0.0

        for k in k_list:
            class_results[f'R@{k}'] = global_hits[k] / global_true_count if global_true_count > 0 else 0.0
            class_results[f'NDCG@{k}'] = np.mean(ndcg_scores[k]) if ndcg_scores.get(k) else 0.0

        results[class_idx] = class_results

    return results


def print_comprehensive_table(
    comprehensive_results: dict,
    k_list: list = [1, 3, 5, 10],
    logger=None
):
    """
    Print comprehensive localization metrics in multiple formatted tables.
    - Table A: General Accuracy (mAP, MRR, R-Precision)
    - Table B: Recall Analysis (R@K)
    - Table C: Ranking Quality (NDCG@K)
    - Table D: Error Analysis (MDE)
    """
    from prettytable import PrettyTable

    def _print_table(title, table, note=""):
        output = f"\n{'='*80}\n"
        output += f"=== {title} ===\n"
        output += f"{'='*80}\n"
        output += str(table) + "\n"
        if note:
            output += f"Note: {note}\n"
        print(output)
        if logger:
            logger.info(output)

    table_a = PrettyTable()
    table_a.field_names = ["Class", "Name", "mAP", "MRR", "R-Prec"]
    table_a.align = "r"
    table_a.align["Class"] = "l"
    table_a.align["Name"] = "l"
    for c in range(12):
        metrics = comprehensive_results.get(c, {})
        table_a.add_row([
            c, MOD_NAMES.get(c, str(c)),
            f"{metrics.get('mAP', 0.0):.4f}",
            f"{metrics.get('MRR', 0.0):.4f}",
            f"{metrics.get('R-Precision', 0.0):.4f}"
        ])
    _print_table("Table A: Localization Accuracy (mAP, MRR, R-Precision)", table_a, 
                 "mAP: Mean Avg Precision, MRR: Mean Reciprocal Rank, R-Prec: R-Precision")

    table_b = PrettyTable()
    table_b.field_names = ["Class", "Name"] + [f"R@{k}" for k in k_list]
    table_b.align = "r"
    table_b.align["Class"] = "l"
    table_b.align["Name"] = "l"
    for c in range(12):
        metrics = comprehensive_results.get(c, {})
        row = [c, MOD_NAMES.get(c, str(c))]
        row.extend([f"{metrics.get(f'R@{k}', 0.0):.4f}" for k in k_list])
        table_b.add_row(row)
    _print_table("Table B: Recall Analysis (R@K)", table_b, "R@K = Global Recall (Micro-Average)")

    table_c = PrettyTable()
    table_c.field_names = ["Class", "Name"] + [f"NDCG@{k}" for k in k_list]
    table_c.align = "r"
    table_c.align["Class"] = "l"
    table_c.align["Name"] = "l"
    for c in range(12):
        metrics = comprehensive_results.get(c, {})
        row = [c, MOD_NAMES.get(c, str(c))]
        row.extend([f"{metrics.get(f'NDCG@{k}', 0.0):.4f}" for k in k_list])
        table_c.add_row(row)
    _print_table("Table C: Ranking Quality (NDCG@K)", table_c, "NDCG = Normalized Discounted Cumulative Gain")

    table_d = PrettyTable()
    table_d.field_names = ["Class", "Name", "MDE (bp)"]
    table_d.align = "r"
    table_d.align["Class"] = "l"
    table_d.align["Name"] = "l"
    for c in range(12):
        metrics = comprehensive_results.get(c, {})
        table_d.add_row([
            c, MOD_NAMES.get(c, str(c)),
            f"{metrics.get('MDE', 0.0):.2f}"
        ])
    _print_table("Table D: Error Analysis", table_d, "MDE = Mean Distance Error for Top-1 False Positives (in base pairs)")
