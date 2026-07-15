"""
中文：人类 RNA 数据集模块。提供 Mer100Dataset（PyG 兼容数据集）用于人类 RNA 修饰预测训练和评估。
English: Human RNA dataset module. Provides Mer100Dataset (PyG-compatible dataset) for human RNA modification prediction training and evaluation.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import os
import subprocess
import hashlib
import pickle

from mrmodn_backend.core.constants import LABEL_MAPPING, INDEX_TO_NUCLEOTIDE
from mrmodn_backend.core.paths import LINEARFOLD_PATH
from mrmodn_backend.services.rna_structure import run_linearfold, build_edge_index_from_structure


ONE_HOT_MAPPING = {
    'A': [1., 0., 0., 0.], 'C': [0., 1., 0., 0.],
    'G': [0., 0., 1., 0.], 'T': [0., 0., 0., 1.],
    'U': [0., 0., 0., 1.], 'N': [0., 0., 0., 0.]
}

DEFAULT_ONE_HOT = [0., 0., 0., 0.]

TARGET_LENGTH = 1001

BATCH_CACHE_FILE = 'structures_cache.npz'


def _create_byte_to_onehot_mapping():
    """
    Create byte-to-one-hot mapping table for fast conversion.

    Returns:
        np.array: Shape (256, 4) mapping table
    """
    mapping = np.zeros((256, 4), dtype=np.float32)
    
    mapping[65] = [1., 0., 0., 0.]
    mapping[97] = [1., 0., 0., 0.]
    
    mapping[67] = [0., 1., 0., 0.]
    mapping[99] = [0., 1., 0., 0.]
    
    mapping[71] = [0., 0., 1., 0.]
    mapping[103] = [0., 0., 1., 0.]
    
    mapping[84] = [0., 0., 0., 1.]
    mapping[116] = [0., 0., 0., 1.]
    
    mapping[85] = [0., 0., 0., 1.]
    mapping[117] = [0., 0., 0., 1.]
    
    mapping[78] = [0., 0., 0., 0.]
    mapping[110] = [0., 0., 0., 0.]
    
    return mapping


_BYTE_TO_ONEHOT_MAPPING = _create_byte_to_onehot_mapping()


def one_hot_to_sequence(one_hot_array):
    """
    Convert one-hot encoded RNA sequence back to character sequence.

    Args:
        one_hot_array (np.array): One-hot array, shape (N, 4)

    Returns:
        str: RNA sequence string
    """
    reverse_mapping = {
        tuple([1., 0., 0., 0.]): 'A',
        tuple([0., 1., 0., 0.]): 'C',
        tuple([0., 0., 1., 0.]): 'G',
        tuple([0., 0., 0., 1.]): 'U',
        tuple([0., 0., 0., 0.]): 'N'
    }
    
    sequence = []
    for one_hot in one_hot_array:
        key = tuple(one_hot)
        nucleotide = reverse_mapping.get(key, 'N')
        sequence.append(nucleotide)
    
    return ''.join(sequence)


def _worker_process_batch(args):
    """
    Worker function: compute secondary structures for a batch of sequences.

    Args:
        args: tuple (batch_indices, sequences_bytes_array, linearfold_path)

    Returns:
        list: [(idx, edge_index_numpy), ...] or [(idx, None, error_msg), ...]
    """
    batch_indices, sequences_bytes_array, linearfold_path = args

    sequences_str = []
    for idx in batch_indices:
        sequence_bytes = sequences_bytes_array[idx].copy()
        sequence_str = sequence_bytes.tobytes().decode('ascii', errors='ignore')
        sequences_str.append(sequence_str)

    results = []
    try:
        structures = run_linearfold(sequences_str)

        for i, (idx, structure) in enumerate(zip(batch_indices, structures)):
            edge_index = build_edge_index_from_structure(sequences_str[i], structure)
            edge_index_numpy = edge_index.cpu().numpy()
            results.append((idx, edge_index_numpy, None))

    except Exception as e:
        for idx in batch_indices:
            results.append((idx, None, str(e)))

    return results


class Mer100Dataset(Dataset):
    """
    中文：人类 RNA 数据集，支持 12 类细粒度分类和注意力监督。使用内存映射加载，兼容多线程 DataLoader。
    English: Human RNA dataset for 12-class fine-grained classification with attention supervision. Uses memory-mapped loading and supports multi-threaded DataLoader.
    """

    def __init__(self, mode='train', data_dir='../npy', cache_dir=None, use_human3=True, use_cache=True, preload_cache=True):
        """
        Initialize dataset (supports memory mapping and multi-threading).

        Args:
            mode (str): 'train' or 'test'
            data_dir (str): Data file directory path
            cache_dir (str): Cache directory path (None for default)
            use_human3 (bool): Whether to use human3 directory data (default True)
            use_cache (bool): Whether to enable structure caching (default True)
            preload_cache (bool): Whether to load all edge indices into memory at init (default True)
        """
        self.mode = mode
        self.data_dir = data_dir
        self.use_cache = use_cache
        self._batch_cache = None
        self._edge_indices = None

        if cache_dir is None:
            self.CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
        else:
            self.CACHE_DIR = cache_dir

        if self.use_cache:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
        
        if use_human3:
            if os.path.exists('human3'):
                human3_dir = 'human3'
            elif os.path.exists('../human3'):
                human3_dir = '../human3'
            else:
                human3_dir = '/home/dc/vscode/vscode20251230/human_and_plant/human3'
            
            print(f"Loading human3 data via memory mapping: {human3_dir}")
            
            self.sequences = np.load(f'{human3_dir}/seq.npy', mmap_mode='r')
            self.full_labels = np.load(f'{human3_dir}/1001loc.npy', mmap_mode='r')
            self.y_12class = np.load(f'{human3_dir}/12loc.npy', mmap_mode='r')
            self.y_4class = np.load(f'{human3_dir}/4loc.npy', mmap_mode='r')
            
            print(f"Dataset initialized (mode={mode}):")
            print(f"  Total samples: {len(self.sequences)}")
            print(f"  Sequence shape: {self.sequences.shape}, dtype: {self.sequences.dtype}")
            print(f"  1001loc shape: {self.full_labels.shape}, dtype: {self.full_labels.dtype}")
            print(f"  12loc shape: {self.y_12class.shape}, dtype: {self.y_12class.dtype}")
            print(f"  4loc shape: {self.y_4class.shape}, dtype: {self.y_4class.dtype}")

            if self.use_cache and preload_cache:
                self._load_batch_cache()
        else:
            self._load_legacy_data(mode, data_dir)
    
    def _load_legacy_data(self, mode, data_dir):
        """
        Load legacy npy data (backward compatibility).

        Args:
            mode (str): 'train' or 'test'
            data_dir (str): Data file directory path
        """
        print(f"Loading legacy data: {data_dir}")
        
        nucleotides = ['A', 'C', 'G', 'U']
        suffix = '_train.npy' if mode == 'train' else '_test.npy'
        
        all_sequences = []
        all_full_labels = []
        all_y_12class = []
        all_y_4class = []
        
        for nuc in nucleotides:
            file_path = f'{data_dir}/{nuc}_expert{suffix}'
            try:
                data = np.load(file_path, allow_pickle=True)
                
                sequences = data['seq']
                full_labels = data['full_label']
                
                y_12class = np.zeros((len(sequences), 12), dtype=np.int8)
                y_4class = np.zeros((len(sequences), 4), dtype=np.int8)
                
                for i, label in enumerate(full_labels):
                    for label_id in np.unique(label):
                        if label_id == 0:
                            continue
                        if label_id in LABEL_MAPPING:
                            idx = LABEL_MAPPING[label_id]
                            y_12class[i, idx] = 1
                            nuc_group = INDEX_TO_NUCLEOTIDE[idx]
                            group_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}[nuc_group]
                            y_4class[i, group_idx] = 1
                
                all_sequences.append(sequences)
                all_full_labels.append(full_labels)
                all_y_12class.append(y_12class)
                all_y_4class.append(y_4class)
                
                print(f"Loaded {file_path}: {len(sequences)} samples")
                
            except Exception as e:
                raise RuntimeError(f"Error loading {file_path}: {e}")
        
        self.sequences = np.concatenate(all_sequences, axis=0)
        self.full_labels = np.concatenate(all_full_labels, axis=0)
        self.y_12class = np.concatenate(all_y_12class, axis=0)
        self.y_4class = np.concatenate(all_y_4class, axis=0)
        
        print(f"Dataset initialized (mode={mode}):")
        print(f"  Total samples: {len(self.sequences)}")
        print(f"  full_label shape: {self.full_labels.shape}")
        print(f"  12loc shape: {self.y_12class.shape}")
        print(f"  4loc shape: {self.y_4class.shape}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a single data sample as a PyG Data object.

        Args:
            idx (int): Sample index

        Returns:
            Data: PyG Data object with x, edge_index, y, y_4class, y_site, attention masks
        """
        sequence_bytes = self.sequences[idx].copy()
        full_label = self.full_labels[idx].copy()
        y_12class = self.y_12class[idx].copy()
        y_4class = self.y_4class[idx].copy()

        one_hot_seq = self._one_hot_encode_optimized(sequence_bytes)

        sequence_str = sequence_bytes.tobytes().decode('ascii', errors='ignore')

        edge_index = self._get_or_compute_edge_index(sequence_str, idx)

        node_features = torch.FloatTensor(one_hot_seq)

        attn_masks = self._extract_attention_masks(full_label)

        attn_mask_N = self._extract_attention_masks_N(sequence_bytes)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=torch.FloatTensor(y_12class).unsqueeze(0),
            y_4class=torch.FloatTensor(y_4class).unsqueeze(0),
            y_site=torch.LongTensor(full_label)
        )

        for nuc, mask in attn_masks.items():
            setattr(data, f'attn_mask_{nuc}', mask)

        setattr(data, 'attn_mask_N', attn_mask_N)

        return data
    
    def _extract_attention_masks(self, full_label):
        attn_masks = {}
        for nuc in ['A', 'C', 'G', 'U']:
            attn_masks[nuc] = np.zeros(1001, dtype=np.float32)
        
        for i, label_id in enumerate(full_label):
            if label_id == 0:
                continue
                
            if label_id in LABEL_MAPPING:
                model_idx = LABEL_MAPPING[label_id]
                nucleotide = INDEX_TO_NUCLEOTIDE[model_idx]
                attn_masks[nucleotide][i] = 1.0
        
        for nuc in ['A', 'C', 'G', 'U']:
            mask = attn_masks[nuc]
            if np.sum(mask) > 0:
                attn_masks[nuc] = mask / np.sum(mask)
            else:
                attn_masks[nuc] = np.ones(1001, dtype=np.float32) / 1001
        
        for nuc in ['A', 'C', 'G', 'U']:
            attn_masks[nuc] = torch.FloatTensor(attn_masks[nuc])
        
        return attn_masks
    
    def _extract_attention_masks_N(self, sequence_bytes):
        byte_array = sequence_bytes.view(np.uint8)
        mask = np.zeros(1001, dtype=np.float32)
        mask[(byte_array == 78) | (byte_array == 110)] = 1.0
        return torch.FloatTensor(mask)

    def _get_batch_cache_path(self):
        return os.path.join(self.CACHE_DIR, f"human_{self.mode}_{BATCH_CACHE_FILE}")

    def _load_batch_cache(self):
        cache_path = self._get_batch_cache_path()

        if os.path.exists(cache_path):
            print(f"Loading batch cache: {cache_path}")
            try:
                self._batch_cache = np.load(cache_path, allow_pickle=True)
                self._edge_indices = self._batch_cache['edge_indices']
                print(f"  Loaded {len(self._edge_indices)} edge indices")
                print(f"  Cache file size: {os.path.getsize(cache_path) / (1024**2):.2f} MB")
            except Exception as e:
                print(f"  Warning: Failed to load batch cache: {e}")
                self._batch_cache = None
                self._edge_indices = None
        else:
            print(f"Batch cache not found: {cache_path}")
            print(f"  Hint: Call dataset.precompute_all_structures() to generate cache")

    def precompute_all_structures(self, batch_size=100, num_workers=None, show_progress=True):
        """
        Precompute all structures and save to batch cache (supports multiprocessing).

        Args:
            batch_size (int): Sequences per LinearFold call
            num_workers (int): Worker count, None for CPU count
            show_progress (bool): Show progress bar

        Returns:
            dict: Statistics
        """
        from tqdm import tqdm
        from multiprocessing import Pool, cpu_count

        num_samples = len(self.sequences)
        cache_path = self._get_batch_cache_path()

        if num_workers is None:
            num_workers = cpu_count()

        use_multiprocessing = num_workers > 1

        print(f"\n{'='*60}")
        print(f"Precomputing all structures...")
        print(f"  Total samples: {num_samples}")
        print(f"  Batch size: {batch_size}")
        print(f"  Workers: {num_workers if use_multiprocessing else 1} ({'multiprocessing' if use_multiprocessing else 'single'})")
        print(f"  Cache path: {cache_path}")
        print(f"{'='*60}\n")

        edge_indices = np.empty(num_samples, dtype=object)
        stats = {
            'total': num_samples,
            'computed': 0,
            'failed': 0
        }

        batch_tasks = []
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = list(range(start_idx, end_idx))
            batch_tasks.append((batch_indices, self.sequences, LINEARFOLD_PATH))

        total_batches = len(batch_tasks)

        if use_multiprocessing:
            print(f"Using {num_workers} processes for {total_batches} batches...\n")

            with Pool(processes=num_workers) as pool:
                results_iter = pool.imap_unordered(_worker_process_batch, batch_tasks)

                if show_progress:
                    results_iter = tqdm(results_iter, total=total_batches, desc="Precomputing structures")

                for batch_results in results_iter:
                    for idx, edge_index_numpy, error in batch_results:
                        if error is None:
                            edge_indices[idx] = edge_index_numpy
                            stats['computed'] += 1
                        else:
                            stats['failed'] += 1
                            if stats['failed'] <= 5:
                                print(f"  Warning: Index {idx} failed: {error}")

        else:
            iterator = range(0, num_samples, batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="Precomputing structures")

            for start_idx in iterator:
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = list(range(start_idx, end_idx))

                sequences_str = []
                for idx in batch_indices:
                    sequence_bytes = self.sequences[idx].copy()
                    sequence_str = sequence_bytes.tobytes().decode('ascii', errors='ignore')
                    sequences_str.append(sequence_str)

                try:
                    structures = run_linearfold(sequences_str)

                    for i, (idx, structure) in enumerate(zip(batch_indices, structures)):
                        edge_index = build_edge_index_from_structure(sequences_str[i], structure)
                        edge_indices[idx] = edge_index.cpu().numpy()
                        stats['computed'] += 1

                except Exception as e:
                    print(f"\nWarning: Batch failed (index {start_idx}-{end_idx}): {e}")
                    for idx in batch_indices:
                        stats['failed'] += 1

        print(f"\nSaving batch cache to: {cache_path}")
        try:
            np.savez_compressed(
                cache_path,
                edge_indices=edge_indices,
                mode=self.mode,
                num_samples=num_samples
            )

            file_size_mb = os.path.getsize(cache_path) / (1024**2)
            print(f"  Cache saved, size: {file_size_mb:.2f} MB")

        except Exception as e:
            print(f"  Error: Failed to save batch cache: {e}")
            return

        self._batch_cache = np.load(cache_path, allow_pickle=True)
        self._edge_indices = self._batch_cache['edge_indices']

        print(f"\n{'='*60}")
        print(f"Precomputation complete!")
        print(f"  Total samples: {stats['total']}")
        print(f"  Computed: {stats['computed']}")
        print(f"  Failed: {stats['failed']}")
        print(f"{'='*60}\n")

    def _get_cache_key(self, sequence_str, idx):
        sequence_hash = hashlib.md5(sequence_str.encode('utf-8')).hexdigest()
        cache_key = f"{self.mode}_{idx}_{sequence_hash}"
        return cache_key

    def _get_cache_path(self, cache_key):
        return os.path.join(self.CACHE_DIR, f"{cache_key}.pkl")

    def _load_from_cache(self, cache_key):
        if not self.use_cache:
            return None

        cache_path = self._get_cache_path(cache_key)

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                return cached_data['edge_index']
            except (pickle.PickleError, EOFError, KeyError):
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
                return None

        return None

    def _save_to_cache(self, cache_key, edge_index):
        if not self.use_cache:
            return

        cache_path = self._get_cache_path(cache_key)

        try:
            cached_data = {
                'edge_index': edge_index,
                'mode': self.mode
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
        except (pickle.PickleError, OSError):
            pass

    def _get_or_compute_edge_index(self, sequence_str, idx):
        if self._edge_indices is not None and idx < len(self._edge_indices):
            cached = self._edge_indices[idx]
            if cached is not None:
                return torch.from_numpy(cached)

        cache_key = self._get_cache_key(sequence_str, idx)
        cached_edge_index = self._load_from_cache(cache_key)
        if cached_edge_index is not None:
            return cached_edge_index

        try:
            structures = run_linearfold([sequence_str])
            structure = structures[0]
            edge_index = build_edge_index_from_structure(sequence_str, structure)

            self._save_to_cache(cache_key, edge_index)

            return edge_index
        except (FileNotFoundError, subprocess.TimeoutExpired, RuntimeError):
            raise

    def clear_cache(self):
        if not os.path.exists(self.CACHE_DIR):
            return

        removed_count = 0
        prefix = f"{self.mode}_"

        for filename in os.listdir(self.CACHE_DIR):
            if filename.startswith(prefix) and filename.endswith('.pkl'):
                cache_path = os.path.join(self.CACHE_DIR, filename)
                try:
                    os.remove(cache_path)
                    removed_count += 1
                except OSError:
                    pass

        print(f"Cleared {removed_count} single-file caches (mode={self.mode})")

        batch_cache_path = self._get_batch_cache_path()
        if os.path.exists(batch_cache_path):
            try:
                os.remove(batch_cache_path)
                print(f"Cleared batch cache: {batch_cache_path}")
            except OSError as e:
                print(f"Failed to clear batch cache: {e}")

        self._batch_cache = None
        self._edge_indices = None

    def get_cache_stats(self):
        stats = {
            'cache_dir': self.CACHE_DIR,
            'batch_cache': None,
            'single_file_cache': {
                'total_files': 0,
                'total_size_mb': 0.0
            }
        }

        if not os.path.exists(self.CACHE_DIR):
            return stats

        prefix = f"{self.mode}_"
        mode_files = []
        total_size = 0

        for filename in os.path.listdir(self.CACHE_DIR):
            if filename.startswith(prefix) and filename.endswith('.pkl'):
                cache_path = os.path.join(self.CACHE_DIR, filename)
                try:
                    file_size = os.path.getsize(cache_path)
                    mode_files.append(filename)
                    total_size += file_size
                except OSError:
                    pass

        stats['single_file_cache']['total_files'] = len(mode_files)
        stats['single_file_cache']['total_size_mb'] = total_size / (1024 * 1024)

        batch_cache_path = self._get_batch_cache_path()
        if os.path.exists(batch_cache_path):
            try:
                batch_size_mb = os.path.getsize(batch_cache_path) / (1024 * 1024)
                stats['batch_cache'] = {
                    'exists': True,
                    'path': batch_cache_path,
                    'size_mb': batch_size_mb,
                    'loaded_in_memory': self._edge_indices is not None
                }
            except OSError:
                stats['batch_cache'] = {'exists': False}
        else:
            stats['batch_cache'] = {'exists': False}

        return stats
    
    def _one_hot_encode_optimized(self, sequence_bytes):
        byte_array = sequence_bytes.view(np.uint8)
        one_hot = _BYTE_TO_ONEHOT_MAPPING[byte_array]
        return one_hot.astype(np.float32)
    
    def _one_hot_encode(self, seq):
        if isinstance(seq, np.ndarray) and seq.ndim == 2 and seq.shape[1] == 4:
            return seq
            
        if isinstance(seq, np.ndarray) and seq.dtype == np.dtype('|S1'):
            return self._one_hot_encode_optimized(seq)
        
        if isinstance(seq, str):
            one_hot = np.zeros((len(seq), 4), dtype=np.float32)
            
            for i, nucleotide in enumerate(seq):
                nuc_str = nucleotide.upper()
                if nuc_str in ONE_HOT_MAPPING:
                    one_hot[i] = ONE_HOT_MAPPING[nuc_str]
                else:
                    one_hot[i] = [0., 0., 0., 0.]
            
            return one_hot
        
        if isinstance(seq, np.ndarray):
            one_hot = np.zeros((len(seq), 4), dtype=np.float32)
            
            for i, nucleotide in enumerate(seq):
                nuc_str = str(nucleotide).upper()
                if nuc_str in ONE_HOT_MAPPING:
                    one_hot[i] = ONE_HOT_MAPPING[nuc_str]
                else:
                    one_hot[i] = [0., 0., 0., 0.]
            
            return one_hot
        
        raise ValueError(f"Unsupported sequence type: {type(seq)}")
