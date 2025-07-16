# coding=utf-8
# Copyright (c) 2025, HUAMEI CORPORATION. All rights reserved.

from __future__ import annotations
#from datasketch import MinHash, MinHashLSH
from dataclasses import dataclass
from typing import Union, Dict, Sequence, Dict, Any, List, Optional
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import math
import time
from functools import wraps
from .skyladder_mask_cache import get_causal_mask


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
        self.input_ids = np.memmap(f"{base_path}_input_ids_document.bin", dtype=np.int32, mode='r')
        self.attention_mask = np.memmap(f"{base_path}_attention_mask_document.bin", dtype=np.int32, mode='r')
        self.labels = np.memmap(f"{base_path}_labels_document.bin", dtype=np.int32, mode='r')
        self.index_map = np.load(index_map_path) if index_map_path else None

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.index_map is not None:
            safe_idx = idx % len(self.index_map)
            idx = self.index_map[safe_idx]
        
        return {
            'input_ids': torch.tensor(self.input_ids[idx],dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx]),
            'labels': torch.tensor(self.labels[idx])
        }



#=============================================================================================================================



# --------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------
@dataclass
class SkyLadderCfg:
    min_ctx_len: int = 512
    max_ctx_len: int = 8192
    warmup_steps: int = 100
    total_steps: int = 100000
    schedule_type: str = "cosine"           # "linear" | "cosine" | "sin" | "exp" | "log" | "inv"
    memory_safety_factor: float = 0.8       # multiply batch-size by this
    alpha: int = 1                          # 1/alpha is the rate for some schedules
    changing_point: Optional[int] = 10      # Override automatic calculation

# --------------------------------------------------------------------
# 2. Scheduler
# --------------------------------------------------------------------

class AdaptiveContextScheduler:
    """
    Expands context fast when loss drops, slows when it plateaus.
    Call like a function: ctx_len = scheduler(step, loss)   # callable wrapper
    """

    def __init__(self,
                 min_ctx_len: int,
                 max_ctx_len: int,
                 total_steps: int = 100_000,
                 ema_alpha: float = 0.10,
                 k: float = 300.0,
                 v_min: float = 0.1,
                 v_max: float = 3.0):
        self.min_ctx = min_ctx_len
        self.max_ctx = max_ctx_len
        self.T       = total_steps

        self.ema_a   = ema_alpha
        self.k       = k
        self.v_min   = v_min
        self.v_max   = v_max

        self.loss_ema  = None
        self.loss_prev = None
        self.v_step    = 0.0

    # -------- public API -------- #
    def update(self, loss: float) -> int:
        self._update_ema(loss)
        self.v_step = min(self.v_step + self._velocity(), self.T)
        return self._ctx_from_vstep(self.v_step)

    # make the object itself callable so dataloader can use self._schedule(step)
    def __call__(self, step: int) -> int:
        r = step / self.T
        f = 0.5 * (1 - np.cos(np.pi * r))
        return int(self.min_ctx + f * (self.max_ctx - self.min_ctx))

    # -------- internal helpers -------- #
    def _update_ema(self, loss):
        self.loss_ema = loss if self.loss_ema is None \
                        else self.ema_a * loss + (1 - self.ema_a) * self.loss_ema

    def _velocity(self):
        progress = max(self.loss_prev - self.loss_ema, 0.0) if self.loss_prev is not None else 0.0
        self.loss_prev = self.loss_ema
        return float(np.clip(self.k * progress, self.v_min, self.v_max))

    def _ctx_from_vstep(self, v):
        r = v / self.T
        f = 0.5 * (1 - np.cos(np.pi * r))
        return int(self.min_ctx + f * (self.max_ctx - self.min_ctx))


class ContextWindowScheduler:
    def __init__(self, cfg: SkyLadderCfg):
        self.cfg = cfg
        self.init_mask_length = cfg.min_ctx_len
        self.final_mask_length = cfg.max_ctx_len

        # Calculate changing point if not provided
        if cfg.changing_point is not None:
            self.changing_point = cfg.changing_point
        else:
            self.changing_point = self.final_mask_length * cfg.alpha

        # Map schedule types to methods
        self.schedule_methods = {
            'linear': self.linear,
            'cosine': self.cosine,
            'sin': self.sine,
            'exp': self.exp,
            'log': self.log,
            'inv': self.inv
        }

    def __call__(self, step: int) -> int:
        """Main scheduler interface - calculates context length for given step"""
        if step < self.cfg.warmup_steps:
            return self.cfg.min_ctx_len

        if self.cfg.schedule_type in self.schedule_methods:
            return self.schedule_methods[self.cfg.schedule_type](step - self.cfg.warmup_steps)
        else:
            return self._original_cosine_schedule(step)

    def _original_cosine_schedule(self, step: int) -> int:
        """Original cosine schedule"""
        theta = (3/2)*np.pi + step/self.cfg.total_steps
        cos = np.cos(theta)
        return cos*total_steps

    def __call__(self, step: int) -> int:  
        """Main scheduler interface - calculates context length for given step"""  
        if step < self.cfg.warmup_steps:  
            return self.cfg.min_ctx_len  
          
        if self.cfg.schedule_type in self.schedule_methods:  
            return self.schedule_methods[self.cfg.schedule_type](step - self.cfg.warmup_steps)  
        else:  
            return self._original_cosine_schedule(step)  
  
    def linear(self, curr_iter_num: int) -> int:  
        """Linear scheduling from original SkyLadder"""  
        if curr_iter_num >= self.changing_point:  
            return self.final_mask_length  
        else:  
            curr_mask_length = self.init_mask_length + int(  
                (self.final_mask_length - self.init_mask_length) * (curr_iter_num / self.changing_point))  
            return curr_mask_length  
  
    def cosine(self, curr_iter_num: int) -> int:  
        """Cosine scheduling"""  
        if curr_iter_num >= self.changing_point:  
            return self.final_mask_length  
        else:  
            theta = (3/2)*np.pi + step/self.cfg.total_steps
            cos = np.cos(theta)
            return int(cos*total_steps)+1

            '''
            progress = curr_iter_num / self.changing_point  
            curr_mask_length = self.init_mask_length + int(  
                (self.final_mask_length - self.init_mask_length) * (1 - math.cos(math.pi * progress)))  
            return curr_mask_length  
            '''

    def sine(self, curr_iter_num: int) -> int:  
        """Sinusoidal scheduling"""  
        if curr_iter_num >= self.changing_point:  
            return self.final_mask_length  
        else:  
            curr_mask_length = self.init_mask_length + int((self.final_mask_length - self.init_mask_length) * np.sin(  
                (np.pi / 2) * (curr_iter_num / self.changing_point)))  
            return curr_mask_length  
  
    def exp(self, curr_iter_num: int) -> int:  
        """Exponential scheduling"""  
        if curr_iter_num >= self.changing_point:  
            return self.final_mask_length  
        else:  
            curr_mask_length = int(self.init_mask_length * (self.final_mask_length / self.init_mask_length) ** (  
                        curr_iter_num / self.changing_point))  
            return curr_mask_length  
  
    def log(self, curr_iter_num: int) -> int:  
        """Logarithmic scheduling"""  
        if curr_iter_num >= self.changing_point:  
            return self.final_mask_length  
        else:  
            curr_mask_length = int(self.init_mask_length * (self.final_mask_length / self.init_mask_length) ** (  
                        np.log(1 + curr_iter_num) / np.log(1 + self.changing_point)))  
            return curr_mask_length  
  
    def inv(self, curr_iter_num: int) -> int:  
        """Inverse linear scheduling"""  
        if curr_iter_num >= self.changing_point:  
            return self.final_mask_length  
        else:  
            linear_curr_mask_length = self.calculate_linear_schedule(curr_iter_num)  
            curr_mask_length = self.final_mask_length + self.init_mask_length - linear_curr_mask_length  
            return curr_mask_length  
  
    def calculate_linear_schedule(self, curr_iter_num: int) -> int:  
        """Basic linear schedule helper"""  
        curr_mask_length = self.init_mask_length + (curr_iter_num // 1)  
        return curr_mask_length  
  


# --------------------------------------------------------------------
# 3. SkyLadder DataLoader with Cached Scheduling
# --------------------------------------------------------------------
class SkyLadder(DataLoader):
    """Memory-optimized SkyLadder DataLoader with cached scheduling calculations"""

    def __init__(
        self,
        dataset: Union[torch.utils.data.Dataset, str],
        global_batch_size: int,
        num_workers: int,
        seed: int,
        dataset_additional_keys: Sequence[str],
        no_shuffle: bool,
        cfg: SkyLadderCfg | None = None,
        **kwargs,
    ):
        self.dataset = dataset
        self.cfg = cfg or SkyLadderCfg()
        self._schedule = AdaptiveContextScheduler(
                    min_ctx_len=self.cfg.min_ctx_len,
                    max_ctx_len=self.cfg.max_ctx_len,
                    total_steps=self.cfg.total_steps
                )  
        #ContextWindowScheduler(self.cfg)
        self._global_bs = global_batch_size
        self._step = 0
        self._extra_keys = list(dataset_additional_keys or [])

        # OPTIMIZATION 1: Cache scheduling calculations
        self._cached_step = -1
        self._cached_context_length = None

        sampler = (
            SequentialSampler(self.dataset)
            if no_shuffle
            else RandomSampler(self.dataset, generator=torch.Generator().manual_seed(seed))
        )

        super().__init__(
            dataset=self.dataset,
            batch_size=global_batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=self._collate_no_padding,
            pin_memory=True,
            drop_last=True,
            **kwargs,
        )

    def set_current_ctx(self,ctx_len):
        self._cached_context_length=ctx_len
        self._cached_step=-1

    def _get_current_context_length(self) -> int:
        """OPTIMIZATION 1: Get current context length with caching"""
        if self._cached_step != self._step:
            self._cached_context_length = self._schedule(self._step)
            self._cached_step = self._step
        return self._cached_context_length

    def dynamic_context_length(self):
        """Calculates the current context window size based on the training step."""
        current_step = self._step
        max_length, min_length, alpha = self.cfg.max_ctx_len, self.cfg.min_ctx_len, self.cfg.alpha
        total_steps = self.cfg.total_steps
        current_ctx_length = min(max_length, min_length + int(alpha * current_step))
        return current_ctx_length

    def _collate_no_padding(self, rows):
        """Collate function with NO PADDING and cached scheduling"""
        # OPTIMIZATION 1: Use cached context length calculation
        current_ctx = self._get_current_context_length()

        def _slice_to_current_no_padding(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
            """Slice tensors to current context length WITHOUT padding"""
            out = []
            for t in tensors:
                t = t[:current_ctx]  # Only slice to current context
                out.append(t)
            return out  # Return list instead of stacked tensor

        # Handle both "input_ids" and "prompts" keys
        input_key = "prompts" if "prompts" in rows[0] else "input_ids"

        # Process input tensors with dynamic sizing, NO PADDING
        input_ids_list = _slice_to_current_no_padding([torch.as_tensor(r[input_key]) for r in rows])
        labels_list = _slice_to_current_no_padding([torch.as_tensor(r["labels"]) for r in rows])

        # Create attention masks for each sequence individually (no padding)
        attention_masks = []
        for input_ids in input_ids_list:
            seq_len = input_ids.size(0)
            # Create causal mask for this specific sequence length
            mask = get_causal_mask(seq_len, device=input_ids.device)
            attention_masks.append(mask)

        # Debugging: Log the shapes and sizes of input tensors

        batch: Dict[str, Any] = {
            "prompts": input_ids_list,  # List of tensors with different lengths
            "attention_mask": attention_masks,  # List of 2D masks
            "labels": labels_list,  # List of tensors with different lengths
            "current_ctx_len": torch.tensor(current_ctx, dtype=torch.int32),
            "orig_ctx_len": torch.tensor([len(ids) for ids in input_ids_list], dtype=torch.int32),
        }

        # Handle additional keys without padding
        for k in self._extra_keys:
            if k not in rows[0]:
                continue
            try:
                batch[k] = _slice_to_current_no_padding([torch.as_tensor(r[k]) for r in rows])
            except Exception:
                batch[k] = [r[k] for r in rows]

        if self._step % 50 == 0: print(f"[SkyLAdder] step {self._step:5d} ctx_len={current_ctx}")
        return batch

    def __iter__(self):
        """Independent iterator implementation with cached scheduling"""
        self._step = 0
        # Reset cache when starting new epoch
        self._cached_step = -1
        self._cached_context_length = None

        if hasattr(self.sampler, '__iter__'):
            indices = iter(self.sampler)
        else:
            indices = iter(range(len(self.dataset)))

        batch = []
        for idx in indices:
            try:
                sample = self.dataset[idx]
                batch.append(sample)

                if len(batch) >= self._global_bs:
                    processed_batch = self._collate_no_padding(batch)
                    yield processed_batch
                    self._step += 1
                    batch = []

            except (StopIteration, IndexError):
                break

        if batch:
            processed_batch = self._collate_no_padding(batch)
            yield processed_batch
            self._step += 1

    def get_current_context_length(self) -> int:
        """Public method to get current context length (uses cache)"""
        return self._get_current_context_length()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        current_ctx = self._get_current_context_length()

        return {
            "current_context_length": current_ctx,
            "step": self._step,
            "memory_safety_factor": self.cfg.memory_safety_factor,
            "schedule_type": self.cfg.schedule_type,
            "min_ctx_len": self.cfg.min_ctx_len,
            "max_ctx_len": self.cfg.max_ctx_len,
            "memory_efficiency_ratio": self.cfg.min_ctx_len / current_ctx,
            "cache_hit": self._cached_step == self._step  # Shows if cache was used
        }






'''
@dataclass
class SkyLadderCfg:
    min_ctx_len: int = 64
    max_ctx_len: int = 8192
    warmup_steps: int = 10
    total_steps: int = 10000
    schedule_type: str = "linear"
    memory_safety_factor: float = 0.80
    alpha: int = 1
    changing_point: Optional[int] = None

class ContextWindowScheduler:
    def __init__(self, cfg: SkyLadderCfg):
        self.cfg = cfg
        self.init_mask_length = cfg.min_ctx_len
        self.final_mask_length = cfg.max_ctx_len
        self.changing_point = cfg.changing_point if cfg.changing_point is not None else self.final_mask_length * cfg.alpha

    def __call__(self, step: int) -> int:
        if step < self.cfg.warmup_steps:
            return self.cfg.min_ctx_len

        curr_iter_num = step - self.cfg.warmup_steps
        if curr_iter_num >= self.changing_point:
            return self.final_mask_length
        else:
            curr_mask_length = self.init_mask_length + int(
                (self.final_mask_length - self.init_mask_length) * (curr_iter_num / self.changing_point))
            return curr_mask_length

class SkyLadder(DataLoader):
    """SkyLadder DataLoader with NO PADDING - returns variable-length tensors"""

    def __init__(
        self,
        dataset: Union[torch.utils.data.Dataset, str],
        global_batch_size: int,
        num_workers: int,
        seed: int,
        dataset_additional_keys: Sequence[str],
        no_shuffle: bool,
    ):
        self.dataset = dataset
        self.cfg = SkyLadderCfg()
        self._schedule = ContextWindowScheduler(self.cfg)
        self._global_bs = global_batch_size
        self._step = 0
        self._extra_keys = list(dataset_additional_keys or [])

        sampler = (
            SequentialSampler(self.dataset)
            if no_shuffle
            else RandomSampler(self.dataset, generator=torch.Generator().manual_seed(seed))
        )

        super().__init__(
            dataset=self.dataset,
            batch_size=global_batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=self._collate_no_padding,
            pin_memory=True,
            drop_last=True,
        )

    def _collate_no_padding(self, rows: List[Dict]) -> Dict[str, Any]:
        """Collate function with NO PADDING - returns lists of variable-length tensors"""
        current_ctx = self._schedule(self._step)

        def _slice_to_current_no_padding(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
            """Slice tensors to current context length WITHOUT padding"""
            out = []
            for t in tensors:
                t = t[:current_ctx]  # Only slice to current context
                out.append(t)
            return out  # Return list instead of stacked tensor

        # Handle both "input_ids" and "prompts" keys
        input_key = "prompts" if "prompts" in rows[0] else "input_ids"

        # Process input tensors with dynamic sizing, NO PADDING
        input_ids_list = _slice_to_current_no_padding([torch.as_tensor(r[input_key]) for r in rows])
        labels_list = _slice_to_current_no_padding([torch.as_tensor(r["labels"]) for r in rows])

        # Create attention masks for each sequence individually (no padding)
        attention_masks = []
        for input_ids in input_ids_list:
            seq_len = len(input_ids)
            # Create causal mask for this specific sequence length
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            attention_masks.append(mask)

        batch: Dict[str, Any] = {
            "prompts": input_ids_list,  # List of tensors with different lengths
            "attention_mask": attention_masks,  # List of 2D masks
            "labels": labels_list,  # List of tensors with different lengths
            "current_ctx_len": torch.tensor(current_ctx, dtype=torch.int32),
            "orig_ctx_len": torch.tensor([len(ids) for ids in input_ids_list], dtype=torch.int32),
        }

        # Handle additional keys without padding
        for k in self._extra_keys:
            if k not in rows[0]:
                continue
            try:
                batch[k] = _slice_to_current_no_padding([torch.as_tensor(r[k]) for r in rows])
            except Exception:
                batch[k] = [r[k] for r in rows]

        return batch

    def __iter__(self):
        """Independent iterator implementation"""
        self._step = 0

        if hasattr(self.sampler, '__iter__'):
            indices = iter(self.sampler)
        else:
            indices = iter(range(len(self.dataset)))

        batch = []
        for idx in indices:
            try:
                sample = self.dataset[idx]
                batch.append(sample)

                if len(batch) >= self._global_bs:
                    processed_batch = self._collate_no_padding(batch)
                    yield processed_batch
                    self._step += 1
                    batch = []

            except (StopIteration, IndexError):
                break

        if batch:
            processed_batch = self._collate_no_padding(batch)
            yield processed_batch
            self._step += 1

    def get_current_context_length(self) -> int:
        return self._schedule(self._step)
'''
