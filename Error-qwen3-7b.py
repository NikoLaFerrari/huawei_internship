# skyladder_minimal.py
from __future__ import annotations
from typing import List, Dict, Optional, Union
import math
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

class SkyLadder(DataLoader):
    """A _tiny_ SkyLadder curriculum DataLoader.

    • Starts at `min_ctx_len`, grows to `max_ctx_len` over `total_steps`.
    • Keeps token-budget ~constant by shrinking the batch each step.
    • Works with any Dataset that yields dicts containing
      'input_ids' (and optionally 'attention_mask', 'labels', …).

    Parameters
    ----------
    dataset : str | Dataset
        A torch Dataset instance **or** a string path to torch.load-able file.
    global_batch_size : int
        Desired batch when ctx == min_ctx_len.
    min_ctx_len / max_ctx_len : int
        Curriculum range in tokens.
    warmup_steps : int
        Flat phase before growth starts.
    total_steps : int
        Step at which ctx_len == max_ctx_len.
    schedule : {"linear","cosine"}
        Shape of the growth curve.
    safety : float
        0-1 factor to avoid running out of memory at long contexts.
    shuffle : bool
        Use RandomSampler if True.
    """

    def __init__(
        self,
        dataset: Union[str, Dataset],
        global_batch_size: int,
        *,
        min_ctx_len: int = 512,
        max_ctx_len: int = 8_192,
        warmup_steps: int = 1_000,
        total_steps: int = 10_000,
        schedule: str = "cosine",
        safety: float = 0.8,
        shuffle: bool = True,
        seed: int = 42,
        **dl_kwargs,
    ):
        # --------------------------------------------------
        # Dataset
        # --------------------------------------------------
        if isinstance(dataset, str):
            dataset = torch.load(dataset)
        self.dataset = dataset

        # --------------------------------------------------
        # Curriculum hyper-parameters
        # --------------------------------------------------
        self.min_ctx = min_ctx_len
        self.max_ctx = max_ctx_len
        self.warmup = warmup_steps
        self.total = total_steps
        self.schedule = schedule
        self.global_bs = global_batch_size
        self.safety = safety
        self.step = 0  # keeps track of iterations

        # --------------------------------------------------
        # Sampler
        # --------------------------------------------------
        sampler_cls = RandomSampler if shuffle else SequentialSampler
        sampler = sampler_cls(
            dataset,
            generator=torch.Generator().manual_seed(seed) if shuffle else None,
        )

        super().__init__(
            dataset,
            batch_size=self._calc_batch_size(),
            sampler=sampler,
            collate_fn=self._collate,
            drop_last=True,
            pin_memory=True,
            **dl_kwargs,
        )

    # ---------- curriculum -------------------------------------------------
    def _ctx_len(self, step: int) -> int:
        if step < self.warmup:
            return self.min_ctx

        p = min(1.0, (step - self.warmup) / max(1, self.total - self.warmup))
        if self.schedule == "linear":
            ctx = self.min_ctx + (self.max_ctx - self.min_ctx) * p
        else:  # cosine
            ctx = self.min_ctx + 0.5 * (self.max_ctx - self.min_ctx) * (1 - math.cos(math.pi * p))
        return int(ctx)

    def _calc_batch_size(self) -> int:
        ctx = self._ctx_len(self.step)
        scale = self.min_ctx / ctx
        return max(1, int(self.global_bs * scale * self.safety))

    # ---------- DataLoader hooks -------------------------------------------
    def __iter__(self):
        for batch in super().__iter__():
            yield batch
            self.step += 1
            # update next batch-size (works because PyTorch keeps a reference)
            if hasattr(self.batch_sampler, "batch_size"):
                self.batch_sampler.batch_size = self._calc_batch_size()

    # ---------- collate ----------------------------------------------------
    def _collate(self, rows: List[Dict]) -> Dict[str, torch.Tensor]:
        ctx = self._ctx_len(self.step)

        def pad_and_stack(key: str):
            ts = [torch.as_tensor(r[key]) for r in rows]
            ts = [t[:ctx] for t in ts]                     # slice
            ts = [torch.nn.functional.pad(t, (0, ctx - t.size(0))) for t in ts]
            return torch.stack(ts)

        batch = {"input_ids": pad_and_stack("input_ids")}

        for extra_key in ("attention_mask", "labels"):
            if extra_key in rows[0]:
                batch[extra_key] = pad_and_stack(extra_key)

        return batch
