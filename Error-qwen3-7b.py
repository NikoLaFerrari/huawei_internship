'''
Error executing job with overrides: []
Traceback (most recent call last):
  File "/models/k0050048751/MindSpeed-RL-master/cli/train_grpo.py", line 574, in <module>
    main()
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/hydra/main.py", line 94, in decorated_main
    _run_hydra(
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/hydra/_internal/utils.py", line 394, in _run_hydra
    _run_app(
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/hydra/_internal/utils.py", line 457, in _run_app
    run_and_report(
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/hydra/_internal/utils.py", line 223, in run_and_report
    raise ex
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
    return func()
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
    lambda: hydra.run(
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/hydra/_internal/hydra.py", line 132, in run
    _ = ret.return_value
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/hydra/core/utils.py", line 260, in return_value
    raise self._return_value
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/hydra/core/utils.py", line 186, in run_job
    ret.return_value = task_function(task_cfg)
  File "/models/k0050048751/MindSpeed-RL-master/cli/train_grpo.py", line 570, in main
    ray.get(train.remote(config))
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/worker.py", line 2772, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/worker.py", line 919, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(TypeError): ray::train() (pid=6382, ip=172.16.4.110)
  File "/models/k0050048751/MindSpeed-RL-master/cli/train_grpo.py", line 163, in train
    data_loader = SkyLadder(
  File "/models/k0050048751/MindSpeed-RL-master/mindspeed_rl/datasets/dataloader.py", line 156, in __init__
    sampler = self._create_sampler(no_shuffle, seed)
  File "/models/k0050048751/MindSpeed-RL-master/mindspeed_rl/datasets/dataloader.py", line 172, in _create_sampler
    return RandomSampler(self.dataset, generator=torch.Generator().manual_seed(seed))
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/torch/utils/data/sampler.py", line 163, in __init__
    if not isinstance(self.num_samples, int) or self.num_samples <= 0:
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/torch/utils/data/sampler.py", line 172, in num_samples
    return len(self.data_source)
TypeError: object of type 'NoneType' has no len()
/usr/local/python3.10.16/lib/python3.10/tempfile.py:869: ResourceWarning: Implicitly cleaning up <TemporaryDirectory '/tmp/tmpnozpaglu'>
  _warnings.warn(warn_message, ResourceWarning)

'''

# coding=utf-8
# Copyright (c) 2025, HUAMEI CORPORATION. All rights reserved.
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


__all__ = [
    'PromptDataLoader',
    'PackedBinaryDataset',
    'SkyLadder',
    'MultiModalDataLoader'
]



class PromptDataLoader(DataLoader):
    """PromptDataLoader.

    Args:
        dataset: An Prompt Implementation of BaseDataset
        consumed_samples: the number of consumed samples for continue training
        global_batch_size: global batch size for loader
        num_workers: workers of dataloader
        seed: random seed
        dataset_additional_keys: extra keys for data loading
    """
    def __init__(self,
                 dataset,
                 global_batch_size,
                 num_workers,
                 seed,
                 dataset_additional_keys,
                 no_shuffle):
        def collator(features, return_tensors=None):
            features_dict = {}

            features_dict["prompts"] = [torch.tensor(value['input_ids']) for value in features]

            for add_key in dataset_additional_keys:
                features_dict[add_key] = [torch.tensor(value[add_key]) for value in features]

            return features_dict

        if not no_shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(seed)
            sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=dataset)

        super().__init__(dataset,
                        num_workers=num_workers,
                        generator=torch.Generator().manual_seed(seed),
                        collate_fn=collator,
                        pin_memory=True,
                        sampler=sampler,
                        batch_size=global_batch_size,
                        drop_last=True)




class PackedBinaryDataset(Dataset):
    def __init__(self, base_path: str, index_map_path: Optional[str] = None):
        self.input_ids = np.memmap(f"{base_path}_input_ids_document.bin", dtype=np.int64, mode='r')
        self.attention_mask = np.memmap(f"{base_path}_attention_mask_document.bin", dtype=np.int64, mode='r')
        self.labels = np.memmap(f"{base_path}_labels_document.bin", dtype=np.int64, mode='r')
        self.index_map = np.load(index_map_path) if index_map_path else None

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.index_map is not None:
            safe_idx = idx % len(self.index_map)
            idx = self.index_map[safe_idx]
        
        return {
            'input_ids': torch.tensor(self.input_ids[idx]),
            'attention_mask': torch.tensor(self.attention_mask[idx]),
            'labels': torch.tensor(self.labels[idx])
        }


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
        self.memory_safety_factor = 0.8

class ContextWindowScheduler:
    """Dynamically adjusts context length during training"""
    def __init__(self, config: SkyLadderConfig):
        self.config = config
    
    def get_ctx_len(self, step: int) -> int:
        if step < self.config.warmup_steps:
            return self.config.min_ctx_len
        
        progress = min(1.0, (step - self.config.warmup_steps) / 
                     (self.config.total_steps - self.config.warmup_steps))
        
        if self.config.schedule_type == "linear":
            ctx_len = self.config.min_ctx_len + (self.config.max_ctx_len - self.config.min_ctx_len) * progress
        else:  # cosine
            ctx_len = self.config.min_ctx_len + 0.5 * (self.config.max_ctx_len - self.config.min_ctx_len) * (1 - math.cos(math.pi * progress))
        
        return int(ctx_len)

class SkyLadder(DataLoader):
    """Optimized DataLoader with dynamic context windows"""
    
    def __init__(self,
                 dataset: Union[str, Dataset],
                 global_batch_size: int,
                 num_workers: int = 0,
                 seed: int = 42,
                 dataset_additional_keys: List[str] = None,
                 no_shuffle: bool = False,
                 **kwargs):
        
        # Initialize configuration
        self.config = SkyLadderConfig()
        self.scheduler = ContextWindowScheduler(self.config)
        self.current_step = 0
        self.additional_keys = dataset_additional_keys or []
        self.global_batch_size = global_batch_size

        # Initialize dataset
        if isinstance(dataset, str):
            if not os.path.exists(dataset):
                raise FileNotFoundError(f"Dataset path does not exist: {dataset}")
            self.dataset = PackedBinaryDataset(dataset)
        else:
            self.dataset = dataset

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
        """Create appropriate sampler based on shuffle setting"""
        if no_shuffle:
            return SequentialSampler(self.dataset)
        return RandomSampler(self.dataset, generator=torch.Generator().manual_seed(seed))

    def _calculate_batch_size(self, global_bs: int) -> int:
        """Calculate batch size based on current context length"""
        ctx_len = self.scheduler.get_ctx_len(self.current_step)
        scale_factor = self.config.min_ctx_len / ctx_len
        adjusted_bs = max(1, int(global_bs * scale_factor * self.config.memory_safety_factor))
        return adjusted_bs

    def _dynamic_collate(self, batch: List[Dict]) -> Dict:
        """Dynamic batching with current context length"""
        ctx_len = self.scheduler.get_ctx_len(self.current_step)
        
        # Base fields
        result = {
            'input_ids': torch.stack([x['input_ids'][:ctx_len] for x in batch]),
            'attention_mask': torch.stack([x['attention_mask'][:ctx_len] for x in batch]),
            'labels': torch.stack([x['labels'][:ctx_len] for x in batch])
        }
        
        # Additional fields
        for key in self.additional_keys:
            if key in batch[0]:
                result[key] = torch.stack([x[key][:ctx_len] if isinstance(x[key], torch.Tensor) 
                              else torch.tensor(x[key]) for x in batch])
        
        return result

    def __iter__(self):
        """Custom iterator that tracks training steps"""
        self.current_step = 0
        for batch in super().__iter__():
            yield batch
            self.current_step += 1

    def set_step(self, step: int):
        """Manually set current training step"""
        self.current_step = step
        # Update batch size for next iteration
        self.batch_sampler.batch_size = self._calculate_batch_size(self.global_batch_size)



