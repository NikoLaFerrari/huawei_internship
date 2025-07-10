# ----------------------------------------------------------------------
# 0. imports  (unchanged – keep the ones you already have)
# ----------------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Union

import math
import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)

# ----------------------------------------------------------------------
# 1. curriculum hyper-parameters  (unchanged)
# ----------------------------------------------------------------------
@dataclass
class SkyLadderCfg:
    min_ctx_len: int = 512
    max_ctx_len: int = 8192
    warmup_steps: int = 1_000
    total_steps: int = 10_000
    schedule_type: str = "cosine"          # "linear" | "cosine"
    memory_safety_factor: float = 0.80     # 0–1

# ----------------------------------------------------------------------
# 2. scheduler  (unchanged)
# ----------------------------------------------------------------------
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
        else:                                   # cosine
            ctx = self.cfg.min_ctx_len + 0.5 * (
                self.cfg.max_ctx_len - self.cfg.min_ctx_len
            ) * (1 - math.cos(math.pi * progress))

        return int(ctx)

# ----------------------------------------------------------------------
# 3.   SkyLadder  – same logic, PromptDataLoader-style signature
# ----------------------------------------------------------------------
class SkyLadder(DataLoader):
    """
    Dynamic-context DataLoader:
    – identical positional arguments to PromptDataLoader
    – no dataset mutation
    """

    # ←–––– the first six positional args match PromptDataLoader ––––→
    def __init__(
        self,
        dataset: Union[str, torch.utils.data.Dataset],
        global_batch_size: int,
        num_workers: int,
        seed: int,
        dataset_additional_keys: List[str],
        no_shuffle: bool,
        *,
        cfg: SkyLadderCfg | None = None,
        **dl_kwargs,
    ):
        self.dataset = dataset
        self.cfg = cfg or SkyLadderCfg()
        self._schedule = ContextWindowScheduler(self.cfg)
        self._global_bs = global_batch_size
        self._step = 0
        self._extra_keys = dataset_additional_keys or []

        sampler = (
            SequentialSampler(self.dataset)
            if no_shuffle
            else RandomSampler(
                self.dataset, generator=torch.Generator().manual_seed(seed)
            )
        )

        super().__init__(
            dataset=self.dataset,
            batch_size=self._calc_batch_size(),   # first batch
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=self._collate,
            pin_memory=True,
            drop_last=True,
            **dl_kwargs,
        )

    # ––––– helpers –––––
    def _calc_batch_size(self) -> int:
        ctx = self._schedule(self._step)
        scale = self.cfg.min_ctx_len / ctx
        bs = int(self._global_bs * scale * self.cfg.memory_safety_factor)
        return max(bs, 1)

    # ––––– DataLoader protocol –––––
    def __iter__(self):
        for batch in super().__iter__():
            yield batch
            self._step += 1
            if hasattr(self.batch_sampler, "batch_size"):
                self.batch_sampler.batch_size = self._calc_batch_size()

    # ––––– collate –––––
    def _collate(self, rows: List[Dict]) -> Dict[str, torch.Tensor]:
        ctx = self._schedule(self._step)

        def _slice_pad(t: torch.Tensor) -> torch.Tensor:
            t = t[:ctx]
            pad = ctx - t.size(0)
            return torch.nn.functional.pad(t, (0, pad))

        def _stack(key: str):
            return torch.stack([_slice_pad(torch.as_tensor(r[key])) for r in rows])

        batch = {
            "input_ids": _stack("input_ids"),
            "attention_mask": _stack("attention_mask"),
            "labels": _stack("labels"),
        }

        for k in self._extra_keys:
            if k in rows[0]:
                try:
                    batch[k] = torch.stack(
                        [_slice_pad(torch.as_tensor(r[k])) for r in rows]
                    )
                except Exception:
                    batch[k] = [r[k] for r in rows]

        return batch

# ----------------------------------------------------------------------
# 4.  adapter if the trainer needs "prompts"
# ----------------------------------------------------------------------
class SkyLadderPromptAdapter:
    """Adds `prompts` = `input_ids` when your trainer expects that key."""
    def __init__(self, *args, **kwargs):
        self._sl = SkyLadder(*args, **kwargs)

    def __iter__(self):
        for batch in self._sl:
            batch["prompts"] = batch["input_ids"]
            yield batch

    def __len__(self):
        return len(self._sl)
