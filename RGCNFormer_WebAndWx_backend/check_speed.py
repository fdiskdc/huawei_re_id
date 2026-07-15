#!/usr/bin/env python3
"""
check_speed.py - CPU 推理速度基准 / CPU inference speed benchmark

基准测试:在 CPU 上运行 submit_task 触发的实际推理,分解各步骤耗时(LinearFold
二级结构、构图、前向传播、IG、UMAP)。 / Benchmark: run actual inference on CPU,
breaking down the cost of each step (LinearFold structure, graph construction,
forward, IG, UMAP).

功能模块 / Modules:
- one_hot_encode_sequence: ACGT/U → 4-d one-hot / one-hot encode
- 基准测试:多序列,多步骤计时 / benchmark over multiple sequences and steps

输入 / Inputs:
- 命令行:--num-sequences, --seq-length / CLI args

输出 / Outputs:
- 各步骤平均耗时(ms)/ Avg per-step time (ms)

数据流 / Data Flow:
1. 加载模型 / Load model
2. 对每条序列:fold → encode → forward → 计时 / Per-sequence: fold, encode, forward, time
3. 汇总报告 / Aggregate report

相关文件 / Related Files:
- 调用 / Calls: human.run_linearfold、main_model.RNA_ClassQuery_Model
- 被调用 / Called by: 开发/性能调优期手动运行 / Manual run during perf tuning

使用示例 / Usage Example:
    python check_speed.py --num-sequences 10 --seq-length 1001

作者 / Author: 项目组 / Project Team
版本 / Version: 1.0
"""
#!/usr/bin/env python3
"""
检查human.py和相关模块的推理速度
测试submit_task()触发的实际推理任务的各个步骤性能
使用CPU推理
"""

import time
import numpy as np
import torch
from human import run_linearfold, build_edge_index_from_structure
from main_model import RNA_ClassQuery_Model
import json
from config import config
import argparse

# 全局变量
model = None
device = None

def one_hot_encode_sequence(sequence: str) -> np.ndarray:
    """
    Convert RNA sequence string to one-hot encoding.
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

def load_model():
    """加载模型（使用CPU）"""
    global model, device
    
    print("=" * 60)
    print("加载模型...")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load configuration
    with open(config.MODEL_CONFIG_PATH, 'r') as f:
        model_config_file = json.load(f)
    
    model_cfg = model_config_file['model']
    
    # 强制使用CPU
    device = torch.device('cpu')
    print(f"使用设备: {device} (CPU)")
    
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
    
    load_time = time.time() - start_time
    print(f"✓ 模型加载完成 (耗时: {load_time:.3f}s)")
    print("=" * 60)
    
    return model, device

def preprocess_sequence(original_sequence):
    """
    预处理序列（填充/截断到1001）
    返回: (processed_sequence, left_padding, left_trimming)
    """
    TARGET_LENGTH = 1001
    seq_len = len(original_sequence)
    
    left_padding = 0
    left_trimming = 0
    sequence = original_sequence
    
    if seq_len != TARGET_LENGTH:
        if seq_len < TARGET_LENGTH:
            # Pad with 'N' at the center
            padding_needed = TARGET_LENGTH - seq_len
            left_pad = padding_needed // 2
            right_pad = padding_needed - left_pad
            left_padding = left_pad
            sequence = 'N' * left_pad + sequence + 'N' * right_pad
        else:
            # Truncate from both sides (center the sequence)
            excess = seq_len - TARGET_LENGTH
            left_trim = excess // 2
            right_trim = excess - left_trim
            left_trimming = left_trim
            sequence = sequence[left_trim:seq_len - right_trim]
    
    return sequence, left_padding, left_trimming

def check_inference_speed(sequence: str, warmup_runs: int = 2, test_runs: int = 5):
    """
    检查推理速度，测量各个步骤的耗时（CPU推理）
    
    Args:
        sequence: RNA序列
        warmup_runs: 预热运行次数（不计入统计）
        test_runs: 测试运行次数
    """
    global model, device
    
    print("\n" + "=" * 60)
    print(f"测试序列长度: {len(sequence)}")
    print(f"预热运行: {warmup_runs} 次")
    print(f"测试运行: {test_runs} 次")
    print(f"推理设备: CPU")
    print("=" * 60 + "\n")
    
    # 存储每次运行的时间
    all_timings = []
    
    for run_idx in range(warmup_runs + test_runs):
        run_number = run_idx + 1
        is_warmup = run_idx < warmup_runs
        run_type = "预热" if is_warmup else "测试"
        
        print(f"\n[{run_type}运行 #{run_number}]")
        print("-" * 60)
        
        timings = {}
        start_total = time.time()
        
        # 步骤1: 序列预处理
        step_start = time.time()
        processed_seq, left_pad, left_trim = preprocess_sequence(sequence)
        timings['preprocess'] = time.time() - step_start
        print(f"1. 序列预处理: {timings['preprocess']*1000:.2f}ms")
        
        # 步骤2: LinearFold计算二级结构
        step_start = time.time()
        structures = run_linearfold([processed_seq])
        structure = structures[0]
        timings['linearfold'] = time.time() - step_start
        print(f"2. LinearFold二级结构: {timings['linearfold']*1000:.2f}ms")
        
        # 步骤3: 构建边索引
        step_start = time.time()
        edge_index = build_edge_index_from_structure(processed_seq, structure)
        timings['edge_index'] = time.time() - step_start
        print(f"3. 构建边索引: {timings['edge_index']*1000:.2f}ms")
        
        # 步骤4: One-hot编码
        step_start = time.time()
        x = one_hot_encode_sequence(processed_seq)
        x = torch.FloatTensor(x)  # Shape: [1001, 4]
        batch = torch.zeros(len(processed_seq), dtype=torch.long)
        timings['encoding'] = time.time() - step_start
        print(f"4. One-hot编码: {timings['encoding']*1000:.2f}ms")
        
        # 步骤5: 数据准备（CPU，无需传输）
        step_start = time.time()
        from torch_geometric.data import Batch
        data_batch = Batch(x=x, edge_index=edge_index, batch=batch)
        data_batch = data_batch.to(device)
        timings['prepare_data'] = time.time() - step_start
        print(f"5. 数据准备: {timings['prepare_data']*1000:.2f}ms")
        
        # 步骤6: 模型推理（包含前向传播和注意力计算）
        step_start = time.time()
        with torch.no_grad():
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
        timings['inference'] = time.time() - step_start
        print(f"6. 模型推理 (CPU): {timings['inference']*1000:.2f}ms")
        
        # 步骤7: 后处理（移动到CPU、转换格式）
        step_start = time.time()
        logits_12class = logits_12class.cpu().numpy()[0]
        if model_cfg['use_hierarchical']:
            logits_4class = logits_4class.cpu().numpy()[0]
            attn_weights = attn_weights.cpu().numpy()[0]
        timings['postprocess'] = time.time() - step_start
        print(f"7. 后处理: {timings['postprocess']*1000:.2f}ms")
        
        # 总时间
        timings['total'] = time.time() - start_total
        print(f"\n总耗时: {timings['total']*1000:.2f}ms ({timings['total']:.3f}s)")
        
        if not is_warmup:
            all_timings.append(timings)
    
    # 统计结果
    if all_timings:
        print("\n" + "=" * 60)
        print("性能统计 (基于测试运行)")
        print("=" * 60)
        
        # 计算平均值和标准差
        avg_timings = {}
        std_timings = {}
        
        for key in all_timings[0].keys():
            values = [t[key] for t in all_timings]
            avg_timings[key] = np.mean(values)
            std_timings[key] = np.std(values)
        
        # 打印统计表格
        print(f"\n{'步骤':<20} {'平均时间(ms)':<15} {'标准差(ms)':<15} {'占比(%)':<15}")
        print("-" * 65)
        
        total_avg = avg_timings['total']
        
        step_names = {
            'preprocess': '序列预处理',
            'linearfold': 'LinearFold',
            'edge_index': '构建边索引',
            'encoding': 'One-hot编码',
            'prepare_data': '数据准备',
            'inference': '模型推理',
            'postprocess': '后处理',
            'total': '总计'
        }
        
        for key in ['preprocess', 'linearfold', 'edge_index', 'encoding', 'prepare_data', 'inference', 'postprocess']:
            avg_ms = avg_timings[key] * 1000
            std_ms = std_timings[key] * 1000
            percentage = (avg_timings[key] / total_avg) * 100
            print(f"{step_names[key]:<20} {avg_ms:>10.2f}     {std_ms:>10.2f}     {percentage:>12.1f}")
        
        print("-" * 65)
        total_ms = avg_timings['total'] * 1000
        total_std = std_timings['total'] * 1000
        print(f"{'总计':<20} {total_ms:>10.2f}     {total_std:>10.2f}     {100:>12.1f}")
        
        # 关键指标
        print("\n" + "=" * 60)
        print("关键性能指标")
        print("=" * 60)
        print(f"• 平均总推理时间: {avg_timings['total']:.3f}s (±{std_timings['total']:.3f}s)")
        print(f"• 模型推理占比: {(avg_timings['inference']/total_avg)*100:.1f}%")
        print(f"• LinearFold占比: {(avg_timings['linearfold']/total_avg)*100:.1f}%")
        print(f"• 预处理占比: {(avg_timings['preprocess']/total_avg)*100:.1f}%")
        print(f"• 吞吐量 (预测数/秒): {1/avg_timings['total']:.2f}")
        
        return avg_timings, std_timings
    else:
        print("\n警告: 没有收集到测试数据")
        return None, None

def compare_sequence_lengths():
    """比较不同长度序列的推理速度"""
    print("\n" + "=" * 60)
    print("不同长度序列性能对比")
    print("=" * 60)
    
    # 测试不同长度的序列
    test_sequences = [
        ("短序列 (200nt)", "A" * 100 + "C" * 50 + "G" * 50),
        ("中序列 (500nt)", "A" * 200 + "C" * 150 + "G" * 150),
        ("长序列 (800nt)", "A" * 400 + "C" * 200 + "G" * 200),
        ("标准长度 (1001nt)", "A" * 500 + "C" * 250 + "G" * 250 + "U" * 1),
    ]
    
    results = []
    
    for name, seq in test_sequences:
        print(f"\n测试: {name}")
        print("-" * 60)
        
        # 只运行1次测试
        avg_timings, _ = check_inference_speed(seq, warmup_runs=1, test_runs=1)
        
        if avg_timings:
            results.append({
                'name': name,
                'length': len(seq),
                'total_time': avg_timings['total'],
                'linearfold_time': avg_timings['linearfold'],
                'inference_time': avg_timings['inference']
            })
    
    # 打印对比表格
    if results:
        print("\n" + "=" * 60)
        print("性能对比总结")
        print("=" * 60)
        print(f"\n{'序列':<20} {'长度':<10} {'总时间(s)':<12} {'LinearFold(s)':<15} {'推理(s)':<12}")
        print("-" * 70)
        
        for r in results:
            print(f"{r['name']:<20} {r['length']:<10} {r['total_time']:<12.3f} "
                  f"{r['linearfold_time']:<15.3f} {r['inference_time']:<12.3f}")

def main():
    parser = argparse.ArgumentParser(description='检查RNA预测任务的推理速度（CPU）')
    parser.add_argument('--sequence', type=str, help='自定义RNA序列')
    parser.add_argument('--compare', action='store_true', help='比较不同长度序列的性能')
    parser.add_argument('--warmup', type=int, default=2, help='预热运行次数')
    parser.add_argument('--runs', type=int, default=5, help='测试运行次数')
    
    args = parser.parse_args()
    
    # 加载模型
    load_model()
    
    # 获取模型配置
    global model_cfg
    with open(config.MODEL_CONFIG_PATH, 'r') as f:
        model_config_file = json.load(f)
    model_cfg = model_config_file['model']
    
    if args.compare:
        # 比较不同长度序列
        compare_sequence_lengths()
    elif args.sequence:
        # 测试自定义序列
        print(f"\n测试自定义序列 (长度: {len(args.sequence)})")
        check_inference_speed(args.sequence, warmup_runs=args.warmup, test_runs=args.runs)
    else:
        # 默认测试：使用标准长度序列
        default_sequence = "A" * 500 + "C" * 250 + "G" * 250 + "U" * 1
        print(f"\n测试默认序列 (长度: {len(default_sequence)})")
        check_inference_speed(default_sequence, warmup_runs=args.warmup, test_runs=args.runs)

if __name__ == "__main__":
    main()