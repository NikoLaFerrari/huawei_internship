from datasketch import MinHash, MinHashLSH
from dataclasses import dataclass
from typing import Union, Dict, Sequence, Dict, Any, List, Optional
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import math
import time
from functools import wraps


class SkyLadderConfig:
    """Configuration for SkyLadder training"""
    def __init__(self):
        # Context Window
        self.min_ctx_len = 512
        self.max_ctx_len = 8192
        
        # Training
        self.total_steps = 10000
        self.warmup_steps = 1000
        self.base_batch_size = 4
        
        # Scheduler
        self.schedule_type = "cosine"  # "linear" or "cosine"

# Fix the ContextWindowScheduler reference (line ~126)
class ContextWindowScheduler:
    """Dynamically adjusts context length during training"""
    def __init__(self, config):  # Remove type hint to avoid circular reference
        self.config = config
    
    def get_ctx_len(self, step: int) -> int:
        if step < self.config.warmup_steps:
            return self.config.min_ctx_len
        
        progress = min(1.0, (step - self.config.warmup_steps) / 
                  (self.config.total_steps - self.config.warmup_steps))
        
        if self.config.schedule_type == "linear":
            return int(self.config.min_ctx_len + 
                     (self.config.max_ctx_len - self.config.min_ctx_len) * progress)
        return int(self.config.min_ctx_len + 
                 0.5 * (self.config.max_ctx_len - self.config.min_ctx_len) * 
                 (1 - math.cos(math.pi * progress)))

class SkyLadder(DataLoader):
    """Optimized DataLoader with dynamic context windows"""
    
    class Config:
        def __init__(self):
            self.min_ctx_len = 512
            self.max_ctx_len = 8192
            self.total_steps = 10000
            self.warmup_steps = 1000
            self.schedule_type = "cosine"
            self.memory_safety_factor = 0.8

    def __init__(self,
                 #dataset: Union[str, Dataset],
                 global_batch_size: int,
                 num_workers: int = 0,
                 seed: int = 42,
                 dataset_additional_keys: List[str] = None,
                 no_shuffle: bool = False,
                 **kwargs):
        
        # Initialize dataset
        '''
        if isinstance(dataset, str):
            self.dataset = PackedBinaryDataset(dataset)
        else:
            self.dataset = dataset
        '''
        #Temporary fix of data_path=None:
        dataset = '/data/jagan/qwen3_32b_dataset/qwen3_32b/shutong_packed'
        self.dataset = PackedBinaryDataset(dataset)
        print(f'WARNING: Using hardcoded dataset path: {dataset}')

        # Internal config
        self.config = self.Config()
        self.scheduler = ContextWindowScheduler(self.config)
        self.current_step = 0
        self.additional_keys = dataset_additional_keys or []

        # Initialize DataLoader
        sampler = self._create_sampler(no_shuffle, seed)
        super().__init__(
            dataset=self.dataset,
            batch_size=self._calculate_batch_size(global_batch_size),
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=self._dynamic_collate,
            pin_memory=True,
            drop_last=True,
            **kwargs
        )

    def _create_sampler(self, no_shuffle: bool, seed: int):
        return (SequentialSampler(self.dataset) if no_shuffle else
                RandomSampler(self.dataset, generator=torch.Generator().manual_seed(seed)))

    def _calculate_batch_size(self, global_bs: int) -> int:
        ctx_len = self.scheduler.get_ctx_len(self.current_step)
        scale_factor = self.config.min_ctx_len / ctx_len
        return max(1, int(global_bs * scale_factor))

    def _dynamic_collate(self, batch: List[Dict]) -> Dict:
        ctx_len = self.scheduler.get_ctx_len(self.current_step)
        result = {
            'input_ids': torch.stack([x['input_ids'][:ctx_len] for x in batch]),
            'attention_mask': torch.stack([x['attention_mask'][:ctx_len] for x in batch]),
            'labels': torch.stack([x['labels'][:ctx_len] for x in batch])
        }
        for key in self.additional_keys:
            if key in batch[0]:
                result[key] = torch.stack([torch.tensor(x[key]) for x in batch])
        return result

    def __iter__(self):
        self.current_step = 0
        for batch in super().__iter__():
            yield batch
            self.current_step += 1

