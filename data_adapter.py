"""
Shard-Streaming Data Adapter v3 (Fixed)
=======================================
1. Memory-efficient: Loads one shard at a time
2. Global Shuffle Buffer: Prevents Catastrophic Forgetting
3. Multi-worker compatible via IterableDataset
"""

import pickle
import os
import gc
import random
from glob import glob
import torch
from torch.utils.data import IterableDataset

TRAIN_SAMPLES_COUNT = 18816594

class ShardStreamingDataset(IterableDataset):
    def __init__(self, metadata_path, seed=42, buffer_size=30000):
        self.metadata_path = metadata_path
        self.base_dir = os.path.dirname(metadata_path)
        self.seed = seed
        self.buffer_size = buffer_size
        
        with open(metadata_path, 'rb') as f:
            meta = pickle.load(f)
        
        if 'train_shard_paths' in meta:
            self.shard_paths = [
                p if os.path.isabs(p) else os.path.join(self.base_dir, os.path.basename(p))
                for p in meta['train_shard_paths']
            ]
        else:
            shard_dir = metadata_path.replace('.pkl', '_shards')
            self.shard_paths = sorted(glob(os.path.join(shard_dir, '*.pkl')))
        
        self.item_to_idx = meta['item_to_idx']
        self.idx_to_item = meta['idx_to_item']
        self.item_vocab_size = meta['item_vocab_size']
        
        self.test_samples = []
        for s in meta.get('test_samples', []):
            mapped = {
                'seq_idx': s.get('pre_items', s.get('seq_idx', [])),
                'target': s.get('target_item', s.get('target')),
                'target_idx': s.get('target_item', s.get('target_idx')),
                'gap_days': s.get('gap_days'),
                'bucket': s.get('bucket'),
                'intervals': s.get('intra_intervals', s.get('intervals', [])),
                'user_id': s.get('user_id')
            }
            self.test_samples.append(mapped)
        
        self._total_samples = meta.get('train_samples_count', TRAIN_SAMPLES_COUNT)
        print(f"[ShardStreaming] {len(self.shard_paths)} shards, {self._total_samples:,} samples")
    
    def __len__(self):
        return self._total_samples
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Assign different shards to different workers to avoid duplicate reads.
        worker_shards = [p for i, p in enumerate(self.shard_paths) if i % num_workers == worker_id]

        # Use a different random seed per worker.
        rng = random.Random(self.seed + worker_id)
        rng.shuffle(worker_shards)
        
        buffer = []
        
        for path in worker_shards:
            with open(path, 'rb') as f:
                shard = pickle.load(f)
            
            samples = shard.get('train_samples', shard.get('train', []))
            rng.shuffle(samples)
            
            for s in samples:
                item = {
                    'seq_idx': s.get('pre_items', s.get('seq_idx', [])),
                    'target': s.get('target_item', s.get('target')),
                    'target_idx': s.get('target_item', s.get('target_idx')),
                    'gap_days': s.get('gap_days'),
                    'bucket': s.get('bucket'),
                    'intervals': s.get('intra_intervals', s.get('intervals', [])),
                    'user_id': s.get('user_id')
                }
                
                buffer.append(item)
                
                # Pop a random sample once the buffer is full for approximate global shuffling.
                if len(buffer) >= self.buffer_size:
                    idx = rng.randrange(len(buffer))
                    yield buffer.pop(idx)
            
            del shard
            gc.collect()
            
        # Flush the remaining buffer.
        rng.shuffle(buffer)
        for item in buffer:
            yield item

def load_data(file_path, seed=42):
    print(f"[DataAdapter] Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    if 'train_shard_paths' in data:
        print("[DataAdapter] Detected Sharded Architecture. Creating streaming dataset...")
        return ShardStreamingDataset(file_path, seed=seed)
    
    # Legacy monolithic-dict path.
    print("[DataAdapter] Detected Monolithic Architecture.")
    train_samples = data.get('train_samples', data.get('train_data', []))
    test_samples = data.get('test_samples', data.get('test_data', []))
    for s in train_samples + test_samples:
        s['seq_idx'] = s.get('pre_items', s.get('seq_idx', []))
        s['target'] = s.get('target_item', s.get('target'))
        s['target_idx'] = s.get('target_item', s.get('target_idx'))
        s['intervals'] = s.get('intra_intervals', s.get('intervals', []))
    
    return {
        'train_samples': train_samples,
        'test_samples': test_samples,
        'item_to_idx': data['item_to_idx'],
        'idx_to_item': data['idx_to_item'],
        'item_vocab_size': data.get('item_vocab_size', data.get('vocab_size'))
    }
