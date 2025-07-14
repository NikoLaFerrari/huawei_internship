from __future__ import annotations  
from dataclasses import dataclass  
from typing import Union, Dict, Sequence, Any, List, Optional  
import torch  
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler  
import numpy as np  
import math  
  
# --------------------------------------------------------------------  
# 1. Configuration  
# --------------------------------------------------------------------  
@dataclass  
class SkyLadderCfg:  
    min_ctx_len: int = 512  
    max_ctx_len: int = 8192  
    warmup_steps: int = 1_000  
    total_steps: int = 10_000  
    schedule_type: str = "cosine"           # "linear" | "cosine" | "sin" | "exp" | "log" | "inv"  
    memory_safety_factor: float = 0.80      # multiply batch-size by this  
    alpha: int = 8                          # 1/alpha is the rate for some schedules  
    changing_point: Optional[int] = None    # Override automatic calculation  
  
# --------------------------------------------------------------------  
# 2. Context Window Scheduler  
# --------------------------------------------------------------------  
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
            'linear': self.calculate_mask_length_linear_schedule,  
            'cosine': self.calculate_mask_length_cosine_schedule,  
            'sin': self.calculate_mask_length_sin_schedule,  
            'exp': self.calculate_mask_length_exp_schedule,  
            'log': self.calculate_mask_length_log_schedule,  
            'inv': self.calculate_mask_length_inverse_linear_schedule  
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
        prog = (step - self.cfg.warmup_steps) / max(1, self.cfg.total_steps - self.cfg.warmup_steps)  
        prog = min(prog, 1.0)  
        ctx = self.cfg.min_ctx_len + 0.5 * (self.cfg.max_ctx_len - self.cfg.min_ctx_len) * (1 - math.cos(math.pi * prog))  
        return int(ctx)  
  
    def calculate_mask_length_linear_schedule(self, curr_iter_num: int) -> int:  
        """Linear scheduling from original SkyLadder"""  
        if curr_iter_num >= self.changing_point:  
            return self.final_mask_length  
        else:  
            curr_mask_length = self.init_mask_length + int(  
                (self.final_mask_length - self.init_mask_length) * (curr_iter_num / self.changing_point))  
            return curr_mask_length  
  
    def calculate_mask_length_cosine_schedule(self, curr_iter_num: int) -> int:  
        """Cosine scheduling"""  
        if curr_iter_num >= self.changing_point:  
            return self.final_mask_length  
        else:  
            progress = curr_iter_num / self.changing_point  
            curr_mask_length = self.init_mask_length + int(  
                (self.final_mask_length - self.init_mask_length) * (1 - math.cos(math.pi * progress)))  
            return curr_mask_length  
  
    def calculate_mask_length_sin_schedule(self, curr_iter_num: int) -> int:  
        """Sinusoidal scheduling"""  
        if curr_iter_num >= self.changing_point:  
            return self.final_mask_length  
        else:  
            curr_mask_length = self.init_mask_length + int((self.final_mask_length - self.init_mask_length) * np.sin(  
                (np.pi / 2) * (curr_iter_num / self.changing_point)))  
            return curr_mask_length  
  
    def calculate_mask_length_exp_schedule(self, curr_iter_num: int) -> int:  
        """Exponential scheduling"""  
        if curr_iter_num >= self.changing_point:  
            return self.final_mask_length  
        else:  
            curr_mask_length = int(self.init_mask_length * (self.final_mask_length / self.init_mask_length) ** (  
                        curr_iter_num / self.changing_point))  
            return curr_mask_length  
  
    def calculate_mask_length_log_schedule(self, curr_iter_num: int) -> int:  
        """Logarithmic scheduling"""  
        if curr_iter_num >= self.changing_point:  
            return self.final_mask_length  
        else:  
            curr_mask_length = int(self.init_mask_length * (self.final_mask_length / self.init_mask_length) ** (  
                        np.log(1 + curr_iter_num) / np.log(1 + self.changing_point)))  
            return curr_mask_length  
  
    def calculate_mask_length_inverse_linear_schedule(self, curr_iter_num: int) -> int:  
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
    """  
    Memory-optimized SkyLadder DataLoader with cached scheduling calculations  
    """  
  
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
        self._schedule = ContextWindowScheduler(self.cfg)  
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
  
    def _get_current_context_length(self) -> int:  
        """OPTIMIZATION 1: Get current context length with caching"""  
        if self._cached_step != self._step:  
            # Only recalculate when step changes  
            self._cached_context_length = self._schedule(self._step)  
            self._cached_step = self._step  
        return self._cached_context_length  
  
    def get_fragment_lens_fixed_length(self, sequence_length: int, fixed_length: int):  
        """Calculate fragment lengths for variable-length attention"""  
        if fixed_length >= sequence_length:  
            return [sequence_length], 1  
          
        filtered_indices = np.arange(fixed_length, sequence_length, fixed_length) - 1  
          
        if filtered_indices.size > 0:  
            fragment_lengths = []  
            prev = 0  
            for idx in filtered_indices:  
                fragment_lengths.append(int(idx - prev + 1))  
                prev = idx + 1  
            if prev < sequence_length:  
                fragment_lengths.append(sequence_length - prev)  
        else:  
            fragment_lengths = [sequence_length]  
          
        return fragment_lengths, len(fragment_lengths)  
  
    def _collate_no_padding(self, rows: List[Dict]) -> Dict[str, Any]:  
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
            seq_len = len(input_ids)  
            # Create causal mask for this specific sequence length  
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))  
            attention_masks.append(mask)  
  
        batch: Dict[str, Any] = {  
            "input_ids": input_ids_list,  # List of tensors with different lengths  
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
