# skyladder.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Union, Sequence

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
    schedule_type: str = "cosine"           # "linear" | "cosine"
    memory_safety_factor: float = 0.80      # multiply batch-size by this


# --------------------------------------------------------------------
# 2.  step ➜ ctx_len scheduler
# --------------------------------------------------------------------
class _CtxScheduler:
    def __init__(self, cfg: SkyLadderCfg):
        self.cfg = cfg

    def __call__(self, step: int) -> int:
        if step < self.cfg.warmup_steps:
            return self.cfg.min_ctx_len

        prog = (step - self.cfg.warmup_steps) / max(
            1, self.cfg.total_steps - self.cfg.warmup_steps
        )
        prog = min(prog, 1.0)

        if self.cfg.schedule_type == "linear":
            ctx = self.cfg.min_ctx_len + (self.cfg.max_ctx_len - self.cfg.min_ctx_len) * prog
        else:  # cosine
            ctx = self.cfg.min_ctx_len + 0.5 * (self.cfg.max_ctx_len - self.cfg.min_ctx_len) * (
                1 - math.cos(math.pi * prog)
            )
        return int(ctx)


# --------------------------------------------------------------------
# 3.  SkyLadder DataLoader
# --------------------------------------------------------------------
class SkyLadder(DataLoader):
    """
    Dynamic-context DataLoader implementing the “SkyLadder” curriculum.
    *   Early steps train on `min_ctx_len` tokens, then length grows to `max_ctx_len`.
    *   Batch-size shrinks automatically to keep per-batch token count ≈ constant.
    *   Nothing is removed from the dataset; we only slice / pad **within** each batch.
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
        self._schedule = _CtxScheduler(self.cfg)
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
            batch_size=self._calc_batch_size(),      # initial guess
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=self._collate,
            pin_memory=True,
            drop_last=True,
            **kwargs,
        )

    # ------------------------------------------------------------
    # curriculum helpers
    # ------------------------------------------------------------
    def _calc_batch_size(self) -> int:
        ctx = self._schedule(self._step)
        scale = self.cfg.min_ctx_len / ctx
        bs = int(self._global_bs * scale * self.cfg.memory_safety_factor)
        return max(bs, 1)

    # ------------------------------------------------------------
    # DataLoader protocol
    # ------------------------------------------------------------
    def __iter__(self):
        for batch in super().__iter__():
            yield batch
            self._step += 1
            if hasattr(self.batch_sampler, "batch_size"):
                self.batch_sampler.batch_size = self._calc_batch_size()

    # ------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------
    def _collate(self, rows: List[Dict]) -> Dict[str, torch.Tensor]:
        ctx = self._schedule(self._step)

        def _slice_pad(tensors: List[torch.Tensor]) -> torch.Tensor:
            # ↓ slice then pad-right with zeros to size [ctx]
            out = []
            for t in tensors:
                t = t[:ctx]
                pad = ctx - t.size(0)
                if pad:
                    t = torch.nn.functional.pad(t, (0, pad))
                out.append(t)
            return torch.stack(out)

        batch: Dict[str, torch.Tensor] = {
            "input_ids": _slice_pad([torch.as_tensor(r["input_ids"]) for r in rows]),
            "attention_mask": _slice_pad([torch.as_tensor(r["attention_mask"]) for r in rows]),
            "labels": _slice_pad([torch.as_tensor(r["labels"]) for r in rows]),
            # keep track of how many *real* tokens survived
            "orig_ctx_len": torch.tensor(
                [min(len(r["input_ids"]), ctx) for r in rows], dtype=torch.int32
            ),
        }

        for k in self._extra_keys:
            if k not in rows[0]:
                continue
            try:
                batch[k] = _slice_pad([torch.as_tensor(r[k]) for r in rows])
            except Exception:
                # ragged or non-tensor field → leave as list (unchanged)
                batch[k] = [r[k] for r in rows]

        return batch
