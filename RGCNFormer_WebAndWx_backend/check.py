#!/usr/bin/env python3
"""
check.py - LinearFold 边构建诊断 / LinearFold edge construction diagnostic

诊断脚本:验证 run_linearfold + build_edge_index_from_structure 是否真的生成了
非顺序边(碱基配对边),而非所有.扁平结构。运行后输出边数与配对边占比。 / Diagnostic
script: verifies that run_linearfold + build_edge_index_from_structure actually
produces non-sequential (base-pairing) edges, not just sequential edges from an
all-flat structure. Prints edge counts and the base-pair ratio.

功能模块 / Modules:
- analyze_edge_index(edge_index, seq_len): 统计顺序边 / 配对边 / Analyze edge index
- 主流程:加载测试序列 → run_linearfold → build_edge_index → analyze / Main flow

输入 / Inputs:
- 内置测试序列 / Built-in test sequences
- 命令行可选:具体序列 / CLI: optional sequence arg

输出 / Outputs:
- 标准输出:边数统计、结论 / stdout: edge statistics, verdict

数据流 / Data Flow:
1. 取测试序列 / Take test sequence
2. run_linearfold → 点括号结构 / Run LinearFold
3. build_edge_index_from_structure → [2, E] / Build edges
4. analyze_edge_index → 报告 / Analyze & report

相关文件 / Related Files:
- 调用 / Calls: human.run_linearfold、human.build_edge_index_from_structure
- 被调用 / Called by: 开发期手动运行 / Manual run during dev

使用示例 / Usage Example:
    python check.py
    python check.py "ACGUACGU..."

作者 / Author: 项目组 / Project Team
版本 / Version: 1.0
"""
#!/usr/bin/env python3
"""
诊断脚本：验证 run_linearfold 和 build_edge_index_from_structure
是否成功生成碱基配对边（非顺序边）

目的：验证用户的猜想 - LinearFold 是否只返回了扁平结构（全.），
导致没有生成任何真实的碱基配对边。
"""

import sys
import numpy as np
import torch

# 从 human.py 导入需要测试的函数
from human import run_linearfold, build_edge_index_from_structure, LINEARFOLD_PATH

def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def analyze_edge_index(edge_index, sequence_length):
    """
    分析 edge_index，统计顺序边和非顺序边（碱基配对边）

    Args:
        edge_index: torch.Tensor, 形状为 [2, E]
        sequence_length: int, 序列长度

    Returns:
        dict: 包含统计信息的字典
    """
    if edge_index.numel() == 0:
        return {
            'total_edges': 0,
            'sequential_edges': 0,
            'pairing_edges': 0,
            'theoretical_sequential': 2 * (sequence_length - 1)
        }

    # 获取所有边
    edges = edge_index.t().cpu().numpy()  # [E, 2]

    # 统计顺序边和非顺序边
    sequential_count = 0
    pairing_count = 0

    for u, v in edges:
        if abs(u - v) == 1:
            sequential_count += 1
        else:
            pairing_count += 1

    return {
        'total_edges': len(edges),
        'sequential_edges': sequential_count,
        'pairing_edges': pairing_count,
        'theoretical_sequential': 2 * (sequence_length - 1)
    }

def main():
    print_section("LinearFold 边生成诊断工具")

    # 检查 LinearFold 路径
    print(f"\n[1] LinearFold 配置检查")
    print(f"    LINEARFOLD_PATH = {LINEARFOLD_PATH}")

    import os
    if os.path.exists(LINEARFOLD_PATH):
        print(f"    状态: 文件存在")
    else:
        print(f"    状态: 文件不存在!")
        print(f"\n    错误: LinearFold 可执行文件未找到!")
        print(f"    请检查路径配置是否正确。")
        return

    # 测试序列：一个已知能够形成稳定二级结构的发夹环序列
    test_sequence = "TCAGGAGTTCGAGACCAGCCTGATCAACATGACGAAACCCTATCTCTACTAAAAATACAAAAATTAGCCGGGCGTGGTGGCATGCGCCTGTAGTCTCAGCTACTTGGGAGGCTGAAGCAGGAGAATCGTTTGAACCCAGGAGGCAGAGGTTGCAGTGAGCCGAGATCGTGCCACTGCACTCCAGCCTGGGTGACACAGCGAGACTCTGTCTCAAAAAAATAAAAATAAAAAAATAAATAAATAACCTTTAATTTAGTGAGACTTCATATAGAATTGTTTTAATGTTTAATATAGACCATTTGTTTTAGGTGAATTTAACAATTTCATACTGTGATTAAGATTAATTTCTTTTTCTGACTTCTACCAGAAAGCAGGAATTATGTTTCAAATGGACAATCATTTACCAAACCTTGTTAATCTGAATGAAGATCCACAACTATCTGAGATGCTGCTATATATGATAAAAGAAGGAACAACTACAGTTGGAAAGTATAAACCAAACTCAAGCCATGATATTCAGTTATCTGGGGTGCTGATTGCTGATGATCATTGGTATGTTAATCCTCTAAAAAAAAAGAAAAGGCACCTGTTCTATATCTTGATAACATGTGGTTTCCTTCATATGGCATATTCGTTGATACTGATCGTTTGGTAGAATTCTTCAAACCCATTGTTTAGTCAGGAAAAACATACATTCTGAGTGTGTTATAAGGATGATAGGTCAGTTACTCTCAATATAAAGTACAGTGTAATGCTCTCTCTGTTTTTGTTTTGGCATACTTGATCTGTTGATTGAAGAATAATTTATTTTCTTGCAATTATAATGATGCACATGCAAGTAAACTATCTATCTTACATAACAGAATTTTTGGTTGGATTGACCAATTTAAAAATGTTACTTTATGTGAATTTTGTTCATATGAATGGAATACTTGTATATATTGTTGGAATGATAGCGTATGTAAACTTTTTTGACTCTGCATTGTGTTTCCAAGATTTGT"

    print(f"\n[2] 测试序列")
    print(f"    序列: {test_sequence}")
    print(f"    长度: {len(test_sequence)} nt")
    print(f"    预期: 应该形成一个发夹环 (hairpin) 结构")

    try:
        # 步骤 1: 调用 run_linearfold
        print_section("步骤 1: 调用 run_linearfold")
        structures = run_linearfold([test_sequence])

        if not structures or len(structures) == 0:
            print("\n错误: run_linearfold 返回空列表!")
            return

        structure = structures[0]

        # 打印关键的二级结构字符串（最重要的证据）
        print(f"\n[关键证据] LinearFold 返回的二级结构 (点括号表示法):")
        print(f"    {structure}")
        print(f"\n    解析:")
        print(f"    - '.' (点): 未配对的核苷酸")
        print(f"    - '(': 左括号: 配对的起始位置")
        print(f"    - ')': 右括号: 配对的结束位置")

        # 统计括号
        dot_count = structure.count('.')
        open_paren_count = structure.count('(')
        close_paren_count = structure.count(')')

        print(f"\n    结构统计:")
        print(f"    - 未配对核苷酸 ('.'): {dot_count}")
        print(f"    - 配对起始 ('('):      {open_paren_count}")
        print(f"    - 配对结束 (')'):      {close_paren_count}")
        print(f"    - 总配对数:            {open_paren_count} (应该等于 {close_paren_count})")

        # 步骤 2: 调用 build_edge_index_from_structure
        print_section("步骤 2: 调用 build_edge_index_from_structure")
        edge_index = build_edge_index_from_structure(test_sequence, structure)

        print(f"\n    edge_index 形状: {edge_index.shape}")
        print(f"    [2, E] 其中 E = 边的总数")

        # 步骤 3: 分析边
        print_section("步骤 3: 边分析")

        stats = analyze_edge_index(edge_index, len(test_sequence))

        print(f"\n    边统计:")
        print(f"    - 总边数:                    {stats['total_edges']}")
        print(f"    - 理论顺序边数 (双向):       {stats['theoretical_sequential']}")
        print(f"      (公式: 2 × (序列长度 - 1))")
        print(f"    - 实际顺序边数:              {stats['sequential_edges']}")
        print(f"    - 非顺序边数 (碱基配对边):   {stats['pairing_edges']}")

        # 验证顺序边是否正确
        if stats['sequential_edges'] == stats['theoretical_sequential']:
            print(f"\n    ✓ 顺序边数量正确")
        else:
            print(f"\n    ✗ 顺序边数量异常!")

        # 打印一些非顺序边的示例（如果有的话）
        if stats['pairing_edges'] > 0:
            print(f"\n    非顺序边 (碱基配对边) 示例 (前5条):")
            edges = edge_index.t().cpu().numpy()
            pairing_edges = [(int(u), int(v)) for u, v in edges if abs(u - v) > 1]
            for i, (u, v) in enumerate(pairing_edges[:5]):
                print(f"      {i+1}. 边 ({u}, {v}): 距离 = {abs(u - v)}")

        # 步骤 4: 最终结论
        print_section("步骤 4: 最终结论")

        if stats['pairing_edges'] > 0:
            print(f"\n    结论: 猜想不成立。")
            print(f"    脚本成功生成了碱基配对边。")
            print(f"\n    说明:")
            print(f"    - LinearFold 正常工作，预测出了二级结构")
            print(f"    - build_edge_index_from_structure 正确解析了结构")
            print(f"    - 共找到 {stats['pairing_edges']} 条碱基配对边")
        else:
            print(f"\n    结论: 猜想成立。")
            print(f"    脚本未能生成任何碱基配对边，只包含了顺序边。")
            print(f"\n    可能的原因:")
            print(f"    1. LinearFold 返回的结构全为 '.' (扁平结构)")
            print(f"    2. LINEARFOLD_PATH 配置错误，调用了错误的可执行文件")
            print(f"    3. LinearFold 参数配置问题，没有正确预测结构")

    except FileNotFoundError as e:
        print(f"\n错误: LinearFold 可执行文件未找到!")
        print(f"详细信息: {e}")
    except subprocess.TimeoutExpired as e:
        print(f"\n错误: LinearFold 执行超时!")
        print(f"超时时间: {e.timeout} 秒")
    except Exception as e:
        print(f"\n错误: 执行过程中发生异常!")
        print(f"异常类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("诊断结束")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
