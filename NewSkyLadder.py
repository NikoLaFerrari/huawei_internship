"""SkyLadder – curriculum‑based context window scheduler & DataLoader
-------------------------------------------------------------------
A faithful, **from‑scratch** re‑implementation inspired by
https://github.com/sail-sg/SkyLadder .

Highlights
~~~~~~~~~~
* 👉  drop‑in replacement for any PyTorch `Dataset` that provides
  `input_ids`, `attention_mask`, `labels` (or a subset).
* 👉  grows context length from `min_ctx_len` → `max_ctx_len` following a
  *linear* or *cosine* schedule – exactly the curriculum described in
  the original paper.
* 👉  automatically rescales batch‑size each step so the approximate
  *total* number of tokens per batch stays constant.
* 👉  **no dataset mutation**: every sample on disk is presented;
  deduplication & filtering are deliberately left to upstream tooling.

Usage example
-------------
```python
from skyladder import SkyLadder, SkyLadderPromptAdapter

loader = SkyLadder(
    dataset=train_dataset,
    global_batch_size=32,
    num_workers=4,
    seed=42,
    dataset_additional_keys=["source", "id"],
    no_shuffle=False,
)

for step, batch in enumerate(loader):
    # batch is already length‑padded to current context length
    ...
```

If your trainer expects a `prompts` key that mirrors `input_ids`:
```python
loader = SkyLadderPromptAdapter(...)
```
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# ---------------------------------------------------------------------
# 1.  Hyper‑parameters & scheduler
# ---------------------------------------------------------------------
@dataclass
class SkyLadderConfig:
    """Curriculum configuration."""

    min_ctx_len: int = 512
    max_ctx_len: int = 8192
    warmup_steps: int = 1_000
    total_steps: int = 10_000
    schedule: str = "cosine"  # "linear" | "cosine"
    memory_safety_factor: float = 0.8  # scale down batch to avoid OOM

    def __post_init__(self):
        assert self.min_ctx_len > 0 and self.max_ctx_len >= self.min_ctx_len
        assert self.schedule in {"linear", "cosine"}


class ContextWindowScheduler:
    """step → context length according to SkyLadder curriculum"""

    def __init__(self, cfg: SkyLadderConfig):
        self.cfg = cfg

    def __call__(self, step: int) -> int:
        if step < self.cfg.warmup_steps:
            return self.cfg.min_ctx_len

        frac = (step - self.cfg.warmup_steps) / max(
            1, self.cfg.total_steps - self.cfg.warmup_steps
        )
        frac = min(frac, 1.0)

        if self.cfg.schedule == "linear":
            ctx = self.cfg.min_ctx_len + (
                self.cfg.max_ctx_len - self.cfg.min_ctx_len
            ) * frac
        else:  # cosine
            ctx = self.cfg.min_ctx_len + 0.5 * (
                self.cfg.max_ctx_len - self.cfg.min_ctx_len
            ) * (1 - math.cos(math.pi * frac))

        return int(ctx)


# ---------------------------------------------------------------------
# 2.  Main DataLoader
# ---------------------------------------------------------------------
class SkyLadder(DataLoader):
    """Dynamic‑context DataLoader implementing the SkyLadder curriculum."""

    def __init__(
        self,
        dataset: Union[torch.utils.data.Dataset, Sequence[Dict[str, torch.Tensor]]],
        global_batch_size: int,
        num_workers: int,
        seed: int,
        dataset_additional_keys: List[str],
        no_shuffle: bool,
        *,
        cfg: SkyLadderConfig | None = None,
        **dataloader_kwargs,
    ) -> None:
        self.dataset = dataset
        self.cfg = cfg or SkyLadderConfig()
        self.schedule = ContextWindowScheduler(self.cfg)
        self.global_bs = global_batch_size
        self.step = 0
        self.extra_keys = dataset_additional_keys or []

        sampler = (
            SequentialSampler(self.dataset)
            if no_shuffle
            else RandomSampler(
                self.dataset, generator=torch.Generator().manual_seed(seed)
            )
        )

        super().__init__(
            dataset=self.dataset,
            batch_size=self._dynamic_batch_size(),
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            drop_last=True,
            **dataloader_kwargs,
        )

    # ---------------------------------------------------------- helpers
    def _dynamic_batch_size(self) -> int:
        ctx = self.schedule(self.step)
        scale = self.cfg.min_ctx_len / ctx
        bs = int(self.global_bs * scale * self.cfg.memory_safety_factor)
        return max(bs, 1)

    # ---------------------------------------------------------- DataLoader hooks
    def __iter__(self):
        for batch in super().__iter__():
            yield batch
            self.step += 1
            if hasattr(self.batch_sampler, "batch_size"):
                self.batch_sampler.batch_size = self._dynamic_batch_size()

    # ---------------------------------------------------------- collate
    def _collate_fn(self, rows: List[Dict]) -> Dict[str, torch.Tensor]:
        ctx = self.schedule(self.step)

        def _slice_pad(t: torch.Tensor) -> torch.Tensor:
            t = t[:ctx]
            pad_len = ctx - t.size(0)
            return torch.nn.functional.pad(t, (0, pad_len))

        def _stack(key: str):
            return torch.stack([_slice_pad(torch.as_tensor(r[key])) for r in rows])

        batch: Dict[str, torch.Tensor | List] = {
            "input_ids": _stack("input_ids"),
            "attention_mask": _stack("attention_mask"),
            "labels": _stack("labels"),
        }

        for k in self.extra_keys:
            if k not in rows[0]:
                continue
            try:
                batch[k] = torch.stack([_slice_pad(torch.as_tensor(r[k])) for r in rows])
            except Exception:
                batch[k] = [r[k] for r in rows]

        return batch


# ---------------------------------------------------------------------
# 3.  Optional adapter (input_ids → prompts)
# ---------------------------------------------------------------------
class SkyLadderPromptAdapter:
    """Wrapper that duplicates `input_ids` under a `prompts` key."""

    def __init__(self, *args, **kwargs):
        self.loader = SkyLadder(*args, **kwargs)

    def __iter__(self):
        for batch in self.loader:
            batch["prompts"] = batch["input_ids"]
            yield batch

    def __len__(self):
        return len(self.loader)
