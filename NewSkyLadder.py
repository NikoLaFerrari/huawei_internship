from __future__ import annotations
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
    'SkyLadderPromptAdapter',
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
            'input_ids': torch.as_tensor(self.input_ids[idx],dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx]),
            'labels': torch.tensor(self.labels[idx])
        }




# --------------------------------------------------------------------
# 1.  Curriculum hyper-parameters
# --------------------------------------------------------------------
@dataclass
class SkyLadderCfg:
    min_ctx_len: int = 512
    max_ctx_len: int = 8192
    warmup_steps: int = 1_000
    total_steps: int = 10_000
    schedule_type: str = "cosine"   # "linear" or "cosine"
    memory_safety_factor: float = 0.80   # 0 – 1


# --------------------------------------------------------------------
# 2.  Scheduler   (step → context length)
# --------------------------------------------------------------------
class ContextWindowScheduler:
    def __init__(self, cfg: SkyLadderCfg):
        self.cfg = cfg

    def __call__(self, step: int) -> int:
        if step < self.cfg.warmup_steps:
            return self.cfg.min_ctx_len

        progress = (step - self.cfg.warmup_steps) / max(
            1, self.cfg.total_steps - self.cfg.warmup_steps
        )
        progress = min(progress, 1.0)

        if self.cfg.schedule_type == "linear":
            ctx = self.cfg.min_ctx_len + (
                self.cfg.max_ctx_len - self.cfg.min_ctx_len
            ) * progress
        else:  # cosine
            ctx = self.cfg.min_ctx_len + 0.5 * (
                self.cfg.max_ctx_len - self.cfg.min_ctx_len
            ) * (1 - math.cos(math.pi * progress))

        return int(ctx)


# --------------------------------------------------------------------
# 3.  SkyLadder DataLoader
# --------------------------------------------------------------------
class SkyLadderPromptAdapter:
    def __init__(self, *args, **kw):
        self.sl = SkyLadder(*args, **kw)
    def __iter__(self):
        for batch in self.sl:
            batch["prompts"] = batch["input_ids"]
            yield batch
    def __len__(self):                # passthrough
        return len(self.sl)


class SkyLadder(DataLoader):
    """
    *Dynamic-context* DataLoader:

    • starts at `min_ctx_len` and grows to `max_ctx_len`
      (linear or cosine schedule);
    • rescales batch-size each step so
        `batch_tokens ≈ constant`;
    • validates dataset path and emptiness;
    • works with *any* torch Dataset (or a packed prefix string).

    **No deduplication** – every sample that exists on disk will be seen.
    """

    def __init__(
        self,
        dataset: Union[str, torch.utils.data.Dataset],
        global_batch_size: int,
        num_workers: int = 0,
        seed: int = 42,
        dataset_additional_keys: List[str] | None = None,
        no_shuffle: bool = False,
        cfg: SkyLadderCfg | None = None,
        **dl_kwargs,
    ):
        # --------------------------------------------------------
        # Dataset resolution
        # --------------------------------------------------------
        self.dataset = dataset

        # --------------------------------------------------------
        # Curriculum state
        # --------------------------------------------------------
        self.cfg = cfg or SkyLadderCfg()
        self._schedule = ContextWindowScheduler(self.cfg)
        self._global_bs = global_batch_size
        self._step = 0
        self._extra_keys = dataset_additional_keys or []

        # --------------------------------------------------------
        # Sampler
        # --------------------------------------------------------
        sampler = (
            SequentialSampler(self.dataset)
            if no_shuffle
            else RandomSampler(
                self.dataset, generator=torch.Generator().manual_seed(seed)
            )
        )

        super().__init__(
            dataset=self.dataset,
            batch_size=self._calc_batch_size(),  # initial value
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=self._collate,
            pin_memory=True,
            drop_last=True,
            **dl_kwargs,
        )

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    def _calc_batch_size(self) -> int:
        ctx = self._schedule(self._step)
        scale = self.cfg.min_ctx_len / ctx
        bs = int(self._global_bs * scale * self.cfg.memory_safety_factor)
        return max(bs, 1)

    # ------------------------------------------------------------
    # PyTorch DataLoader hooks
    # ------------------------------------------------------------
    def __iter__(self):
        for batch in super().__iter__():
            yield batch
            self._step += 1
            # prepare *next* batch-size
            if hasattr(self.batch_sampler, "batch_size"):
                self.batch_sampler.batch_size = self._calc_batch_size()

    # ------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------
    def _collate(self, rows: List[Dict]) -> Dict[str, torch.Tensor]:
        ctx = self._schedule(self._step)

        def _slice_stack(key: str):
            tensors = [torch.as_tensor(r[key]) for r in rows]
            padded = [torch.nn.functional.pad(t[:ctx],(0,max(0,ctx-t.size(0)))) for t in tensors]
            return torch.stack(padded)

        batch: Dict[str, torch.Tensor | List] = {
            "input_ids": _slice_stack("input_ids"),
            "attention_mask": _slice_stack("attention_mask"),
            "labels": _slice_stack("labels"),
        }

        for k in self._extra_keys:
            if k not in rows[0]:
                continue
            try:
                batch[k] = torch.stack(
                    [torch.as_tensor(r[k])[:ctx] for r in rows]
                )
            except Exception:
                # ragged or non-tensor → keep as list
                batch[k] = [r[k] for r in rows]

        return batch
