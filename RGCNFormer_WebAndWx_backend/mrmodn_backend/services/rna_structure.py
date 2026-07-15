"""
中文：RNA 二级结构计算模块。调用 LinearFold 预测二级结构，并从点括号表示构建图边索引。
English: RNA secondary structure computation using LinearFold. Predicts secondary structures and builds graph edge indices from dot-bracket notation.
"""
import os
import subprocess
import torch

from mrmodn_backend.core.paths import LINEARFOLD_PATH


def run_linearfold(sequences, timeout_seconds=1800):
    """
    Use LinearFold to predict RNA secondary structures (thread-safe).

    Args:
        sequences (list): List of RNA sequence strings
        timeout_seconds (int): Timeout in seconds

    Returns:
        list: List of secondary structure strings

    Raises:
        RuntimeError: If LinearFold execution fails
        FileNotFoundError: If LinearFold executable does not exist
        subprocess.TimeoutExpired: If execution times out
    """
    if not sequences:
        return []
    
    fasta_input = '\n'.join([f'>seq_{i}\n{seq}' for i, seq in enumerate(sequences)])
    
    structures = []
    
    try:
        if not os.path.exists(LINEARFOLD_PATH):
            raise FileNotFoundError(f"LinearFold executable not found: {LINEARFOLD_PATH}")
        
        process = subprocess.Popen(
            [LINEARFOLD_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        
        stdout_data, stderr_data = process.communicate(input=fasta_input, timeout=timeout_seconds)
        
        if process.returncode != 0:
            raise RuntimeError(
                f"LinearFold failed with return code {process.returncode}."
                f"Error: {stderr_data[:500]}"
            )
        
        if len(stdout_data.strip()) == 0:
            print(f"LinearFold debug info:")
            print(f"  Return code: {process.returncode}")
            print(f"  stdout length: {len(stdout_data)}")
            print(f"  stderr length: {len(stderr_data)}")
            if stderr_data:
                print(f"  stderr: {stderr_data[:500]}")
            print(f"  Input sequences: {len(sequences)}")
            print(f"  Input first 200 chars: {fasta_input[:200]}")
            
            raise RuntimeError("LinearFold returned no output")
        
        lines = stdout_data.strip().split('\n')
        structures = []
        
        lines = [line.strip() for line in lines if line.strip()]
        
        for i in range(2, len(lines), 3):
            line = lines[i]
            if not line:
                continue
            structure = line.split()[0]
            structures.append(structure)
        
        if len(structures) != len(sequences):
            print(f"LinearFold output debug info:")
            print(f"  Input sequences: {len(sequences)}")
            print(f"  Output lines: {len(lines)}")
            print(f"  Parsed structures: {len(structures)}")
            print(f"  First 10 output lines:")
            for i, line in enumerate(lines[:10]):
                print(f"    [{i}]: {line}")
            
            raise RuntimeError(
                f"LinearFold returned {len(structures)} structures but expected {len(sequences)}"
            )
        
    except FileNotFoundError:
        raise
    except subprocess.TimeoutExpired as e:
        raise subprocess.TimeoutExpired(e.cmd, e.timeout, output=e.output, stderr=e.stderr)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"LinearFold execution failed: {e}")
    except Exception as e:
        raise RuntimeError(f"LinearFold unknown error: {e}")
    
    return structures


def build_edge_index_from_structure(sequence, structure):
    """
    Build edge index from RNA secondary structure.

    Args:
        sequence (str): RNA sequence
        structure (str): Secondary structure (dot-bracket notation)

    Returns:
        torch.Tensor: Edge index, shape [2, E]
    """
    if not structure:
        return build_sequential_edge_index(sequence)
    
    stack = []
    pairs = {}
    
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')' and stack:
            j = stack.pop()
            pairs[j] = i
            pairs[i] = j
    
    edge_list = []
    
    for i in range(len(sequence) - 1):
        edge_list.extend([(i, i + 1), (i + 1, i)])
    
    for i, j in pairs.items():
        if i < j:
            edge_list.extend([(i, j), (j, i)])
    
    if edge_list:
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        return torch.empty((2, 0), dtype=torch.long)


def build_sequential_edge_index(sequence):
    """
    Build sequential edges only (i, i+1).

    Args:
        sequence (str): RNA sequence

    Returns:
        torch.Tensor: Edge index, shape [2, E]
    """
    edge_list = []
    for i in range(len(sequence) - 1):
        edge_list.extend([(i, i + 1), (i + 1, i)])
    
    if edge_list:
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        return torch.empty((2, 0), dtype=torch.long)
