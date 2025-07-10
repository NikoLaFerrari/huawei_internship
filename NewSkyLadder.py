# skyladder.py
# Fully-robust SkyLadder DataLoader
# -----------------------------------------------------------
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


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
        *,
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
        if isinstance(dataset, str):
            prefix = dataset
            if not os.path.exists(f"{prefix}_input_ids_document.bin"):
                raise FileNotFoundError(
                    f"No packed dataset found at '{prefix}_input_ids_document.bin'"
                )
            # local import avoids circular refs
            from .datasets import PackedBinaryDataset

            self.dataset = PackedBinaryDataset(prefix)
        else:
            self.dataset = dataset

        if len(self.dataset) == 0:
            raise ValueError("Dataset contains zero samples. Check data path / filters.")

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
            return torch.stack([torch.as_tensor(r[key])[:ctx] for r in rows])

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
