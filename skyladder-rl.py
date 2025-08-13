[root@train-qwen3235b-test4n-v13-task-1 mindspeed_rl]# cat datasets/dataloader.py 
# coding=utf-8
# SkyLadder + EnhancedSkyLadder (keeps your original SkyLadderCfg fields)
# No new YAML keys. Minimal, surgical fixes only.

from __future__ import annotations

from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import math
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# Optional mask import; fallback if not available
try:
    from mindspeed_rl.datasets.skyladder_mask_cache import get_causal_mask  # noqa: F401
except Exception:
    def get_causal_mask(seq_len: int, device=None) -> torch.Tensor:
        # Not used by default (we avoid LxL masks), kept for API compatibility
        return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))


class PromptDataLoader(torch.utils.data.DataLoader):
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


class MultiModalDataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 dataset,
                 global_batch_size,
                 num_workers,
                 seed,
                 dataset_additional_keys,
                 no_shuffle):

        def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
            batch_dict = defaultdict(list)
            for feature in features:
                for key, value in feature.items():
                    batch_dict[key].append(value)

            batch_dict['prompts'] = [torch.tensor(i) for i in batch_dict['input_ids']]

            return batch_dict

        if not no_shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(seed)
            sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=dataset)

        super().__init__(dataset,
                        num_workers=num_workers,
                        generator=torch.Generator().manual_seed(seed),
                        collate_fn=collate_fn,
                        pin_memory=True,
                        sampler=sampler,
                        batch_size=global_batch_size,
                        drop_last=True)




# ============================================================
# Your original config (intact): context + generation + extras
# ============================================================
@dataclass
class SkyLadderCfg:
    # Context length parameters
    min_ctx_len: int = 1024
    max_ctx_len: int = 8192
    warmup_steps: int = 0
    total_steps: int = 12
    schedule_type: str = "cosine"
    memory_safety_factor: float = 1.0
    alpha: int = 1
    changing_point: int | None = 12

    # Generation length parameters (kept as before)
    min_gen_len: int = 1024
    max_gen_len: int = 8192
    gen_schedule_type: str = "follow_context"  # "follow_context", "independent", "fixed"
    gen_ratio: float = 1.0
    min_gen_ratio: float = 1
    max_gen_ratio: float = 1.0

    # Misc (kept)
    ema_alpha: float = 0.1
    k: float = 300.0
    v_min: float = 0.1
    v_max: float = 3.0

    # Dynamic batching (kept)
    dynamic_batch: bool = False
    target_tokens_per_rank: int | None = None
    min_batch_size: int = 1
    max_batch_size: int | None = None
    round_batch_to: int = 1


# ===========================================
# Plain context scheduler for base SkyLadder
# ===========================================
class ContextWindowScheduler:
    """Return ONLY context length for a given step (base SkyLadder)."""

    def __init__(self, cfg: SkyLadderCfg):
        self.cfg = cfg
        self.init_mask_length = int(cfg.min_ctx_len)
        self.final_mask_length = int(cfg.max_ctx_len)
        self.changing_point = int(cfg.changing_point) if cfg.changing_point is not None else int(cfg.total_steps)

        self._fn = {
            "linear": self._linear,
            "cosine": self._cosine,
            "sin": self._sine,
            "exp": self._exp,
            "log": self._log,
            "inv": self._inv,
        }.get(cfg.schedule_type, self._cosine)

    def __call__(self, step: int) -> int:
        if step < self.cfg.warmup_steps:
            return int(self.cfg.min_ctx_len)
        return int(self._fn(max(0, step - self.cfg.warmup_steps)))

    def _p(self, s: int) -> float:
        if self.changing_point <= 0:
            return 1.0
        return min(1.0, max(0.0, s / self.changing_point))

    def _linear(self, s: int) -> int:
        p = self._p(s)
        return int(self.init_mask_length + (self.final_mask_length - self.init_mask_length) * p)

    def _cosine(self, s: int) -> int:
        p = self._p(s)
        return int(self.init_mask_length
                   + (self.final_mask_length - self.init_mask_length) * (1 - math.cos(math.pi * p)) / 2)

    def _sine(self, s: int) -> int:
        p = self._p(s)
        return int(self.init_mask_length
                   + (self.final_mask_length - self.init_mask_length) * math.sin((math.pi / 2) * p))

    def _exp(self, s: int) -> int:
        if self.changing_point <= 0:
            return self.final_mask_length
        base = (self.final_mask_length / max(1, self.init_mask_length))
        return int(self.init_mask_length * (base ** (s / self.changing_point)))

    def _log(self, s: int) -> int:
        if self.changing_point <= 0:
            return self.final_mask_length
        return int(self.init_mask_length
                   * (self.final_mask_length / max(1, self.init_mask_length))
                   ** (math.log1p(s) / math.log1p(self.changing_point)))

    def _inv(self, s: int) -> int:
        lin = self._linear(s)
        return int(self.final_mask_length + self.init_mask_length - lin)


# ===================================================
# Enhanced scheduler (ctx + gen) for EnhancedSkyLadder
# ===================================================
class EnhancedContextWindowScheduler:
    """Return (context_length, generation_length) for a given step."""

    def __init__(self, cfg: SkyLadderCfg):
        self.cfg = cfg
        self.init_mask_length = int(cfg.min_ctx_len)
        self.final_mask_length = int(cfg.max_ctx_len)

        self.min_gen_length = int(cfg.min_gen_len)
        self.max_gen_length = int(cfg.max_gen_len)

        self.changing_point = int(cfg.changing_point) if cfg.changing_point is not None else int(cfg.total_steps)

        self._ctx_fn = {
            "linear": self._linear,
            "cosine": self._cosine,
            "sin": self._sine,
            "exp": self._exp,
            "log": self._log,
            "inv": self._inv,
        }.get(cfg.schedule_type, self._cosine)

    def __call__(self, step: int) -> tuple[int, int]:
        ctx = self.get_context_length(step)
        gen = self.get_generation_length(step, ctx)
        return int(ctx), int(gen)

    # ---------- context ----------
    def get_context_length(self, step: int) -> int:
        if step < self.cfg.warmup_steps:
            return int(self.cfg.min_ctx_len)
        return int(self._ctx_fn(max(0, step - self.cfg.warmup_steps)))

    def _p(self, s: int) -> float:
        if self.changing_point <= 0:
            return 1.0
        return min(1.0, max(0.0, s / self.changing_point))

    def _linear(self, s: int) -> int:
        p = self._p(s)
        return int(self.init_mask_length + (self.final_mask_length - self.init_mask_length) * p)

    def _cosine(self, s: int) -> int:
        p = self._p(s)
        return int(self.init_mask_length
                   + (self.final_mask_length - self.init_mask_length) * (1 - math.cos(math.pi * p)) / 2)

    def _sine(self, s: int) -> int:
        p = self._p(s)
        return int(self.init_mask_length
                   + (self.final_mask_length - self.init_mask_length) * math.sin((math.pi / 2) * p))

    def _exp(self, s: int) -> int:
        if self.changing_point <= 0:
            return self.final_mask_length
        base = (self.final_mask_length / max(1, self.init_mask_length))
        return int(self.init_mask_length * (base ** (s / self.changing_point)))

    def _log(self, s: int) -> int:
        if self.changing_point <= 0:
            return self.final_mask_length
        return int(self.init_mask_length
                   * (self.final_mask_length / max(1, self.init_mask_length))
                   ** (math.log1p(s) / math.log1p(self.changing_point)))

    def _inv(self, s: int) -> int:
        lin = self._linear(s)
        return int(self.final_mask_length + self.init_mask_length - lin)

    # ---------- generation ----------
    def get_generation_length(self, step: int, current_ctx_len: int | None = None) -> int:
        t = max(0, step - self.cfg.warmup_steps)

        if self.cfg.gen_schedule_type == "fixed":
            # Always clamp to configured [min_gen_len, max_gen_len]
            return int(max(self.min_gen_length, min(self.max_gen_length, self.max_gen_length)))

        if self.cfg.gen_schedule_type == "follow_context":
            # Follow ctx via ratio + clamp
            if current_ctx_len is None:
                current_ctx_len = self.get_context_length(step)
            base = int(round(float(self.cfg.gen_ratio) * int(current_ctx_len)))
            # clamp by ratio bounds relative to ctx
            min_allowed = int(round(float(self.cfg.min_gen_ratio) * int(current_ctx_len)))
            max_allowed = int(round(float(self.cfg.max_gen_ratio) * int(current_ctx_len)))
            # absolute clamp
            gen = max(self.min_gen_length, min(base, self.max_gen_length))
            # ratio clamp
            gen = max(min_allowed, min(gen, max_allowed))
            print(f"[EnhancedSkyLadder dataloader.py] current gen_len inside SkyLadder: {int(gen)}")
            return int(gen)

        if self.cfg.gen_schedule_type == "independent":
            # Same shape as ctx schedule but over [min_gen_len, max_gen_len]
            p = self._p(t)
            return int(self.min_gen_length
                       + (self.max_gen_length - self.min_gen_length) * (1 - math.cos(math.pi * p)) / 2)

        raise ValueError(f"Unknown gen_schedule_type: {self.cfg.gen_schedule_type}")

# =============================
# Base SkyLadder (context only)
# =============================
class SkyLadder(DataLoader):
    """SkyLadder with cached context scheduling (no generation logic here)."""

    def __init__(
        self,
        dataset: Union[torch.utils.data.Dataset, str],
        global_batch_size: int,
        num_workers: int,
        seed: int,
        dataset_additional_keys: Sequence[str],
        no_shuffle: bool,
        cfg: SkyLadderCfg | Mapping | Any = None,
        **kwargs,
    ):
        self._drop_last = bool(kwargs.get("drop_last", True))
        self.cfg = self._normalize_cfg(cfg)

        self.dataset = dataset
        # IMPORTANT: base SkyLadder uses the plain context scheduler
        self._schedule = EnhancedContextWindowScheduler(self.cfg)

        self._baseline_bs = max(1, int(global_batch_size))
        self._step = 0
        self._extra_keys = list(dataset_additional_keys or [])

        self._cached_step = -1
        self._cached_context_length: Optional[int] = None
        self._last_batch_size: int = self._baseline_bs

        sampler = (
            SequentialSampler(dataset)
            if no_shuffle
            else RandomSampler(dataset, generator=torch.Generator().manual_seed(seed))
        )

        super().__init__(
            dataset=dataset,
            batch_size=global_batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=self._collate_no_padding,
            pin_memory=kwargs.get("pin_memory", True),
            drop_last=self._drop_last,
            **{k: v for k, v in kwargs.items() if k not in ("collate_fn", "pin_memory", "drop_last")},
        )
        print("[SkyLadder] Successfully passed config vars")

    # ---- config normalize (keeps only known fields) ----
    @staticmethod
    def _normalize_cfg(cfg_input) -> SkyLadderCfg:
        if cfg_input is None:
            return SkyLadderCfg()
        if isinstance(cfg_input, SkyLadderCfg):
            return cfg_input

        if isinstance(cfg_input, Mapping):
            d = dict(cfg_input)
        elif hasattr(cfg_input, "dict") and callable(getattr(cfg_input, "dict")):
            d = dict(cfg_input.dict())
        elif isinstance(cfg_input, SimpleNamespace):
            d = dict(cfg_input.__dict__)
        elif hasattr(cfg_input, "__dict__"):
            d = {k: v for k, v in vars(cfg_input).items() if not k.startswith("_")}
        else:
            print("[SkyLadder] returning cfg_input instead of SkyLadderCfg()")
            return cfg_input

        allowed = {f.name for f in fields(SkyLadderCfg)}
        filtered = {k: v for k, v in d.items() if k in allowed}
        return SkyLadderCfg(**filtered)

    # ---- public helpers ----
    def set_current_ctx(self, ctx_len: int):
        self._cached_context_length = int(ctx_len)
        self._cached_step = -1

    def get_current_context_length(self) -> int:
        return int(self._get_current_context_length())

    def get_last_batch_size(self) -> int:
        return int(self._last_batch_size)

    def get_memory_stats(self) -> Dict[str, Any]:
        current_ctx = self._get_current_context_length()
        print(
            f"current_context_length: {int(current_ctx)}",
            f"step: {int(self._step)}",
            f"memory_safety_factor: {float(self.cfg.memory_safety_factor)}",
            f"schedule_type: {str(self.cfg.schedule_type)}",
            f"min_ctx_len: {int(self.cfg.min_ctx_len)}",
            f"max_ctx_len: {int(self.cfg.max_ctx_len)}",
            f"cache_hit: {bool(self._cached_step == self._step)}",
            f"current_batch_size: {int(self._last_batch_size)}",
            f"tokens_per_rank_step: {int(self._last_batch_size) * max(1, int(current_ctx))}",
            f"dynamic_batch_enabled: {bool(self.cfg.dynamic_batch)}",
            f"target_tokens_per_rank: {int(self.cfg.target_tokens_per_rank or 0)}"
        )

        return {
            "current_context_length": int(current_ctx),
            "step": int(self._step),
            "memory_safety_factor": float(self.cfg.memory_safety_factor),
            "schedule_type": str(self.cfg.schedule_type),
            "min_ctx_len": int(self.cfg.min_ctx_len),
            "max_ctx_len": int(self.cfg.max_ctx_len),
            "cache_hit": bool(self._cached_step == self._step),
            "current_batch_size": int(self._last_batch_size),
            "tokens_per_rank_step": int(self._last_batch_size) * max(1, int(current_ctx)),
            "dynamic_batch_enabled": bool(self.cfg.dynamic_batch),
            "target_tokens_per_rank": int(self.cfg.target_tokens_per_rank or 0),
        }

    # ---- internals ----
    def _get_current_context_length(self) -> int:
        if self._cached_step != self._step:
            self._cached_context_length, _ = self._schedule(self._step)
            self._cached_context_length = int(self._cached_context_length)
            self._cached_step = self._step
        return int(self._cached_context_length)

    def _get_desired_batch_size(self, current_ctx: int) -> int:
        if not bool(self.cfg.dynamic_batch) or not self.cfg.target_tokens_per_rank:
            return int(self._baseline_bs)

        eff_target = max(1, int(self.cfg.target_tokens_per_rank * float(self.cfg.memory_safety_factor)))
        ctx = max(1, int(current_ctx))
        desired = max(int(self.cfg.min_batch_size), eff_target // ctx)

        if self.cfg.max_batch_size is not None:
            desired = min(int(desired), int(self.cfg.max_batch_size))

        r = int(self.cfg.round_batch_to or 1)
        if r > 1:
            desired = max(int(self.cfg.min_batch_size), (desired // r) * r)

        return max(1, int(desired))

    def _collate_no_padding(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        current_ctx = self._get_current_context_length()

        def _slice_to_current_no_padding(tensors: List[Any]) -> List[torch.Tensor]:
            out: List[torch.Tensor] = []
            for t in tensors:
                tt = torch.as_tensor(t)[:current_ctx]
                out.append(tt)
            return out

        input_key = "prompts" if "prompts" in rows[0] else "input_ids"

        input_ids_list = _slice_to_current_no_padding([r[input_key] for r in rows])
        labels_list = _slice_to_current_no_padding([r.get("labels", r[input_key]) for r in rows])

        attention_masks = None  # causal kernels handle masks

        batch: Dict[str, Any] = {
            "prompts": input_ids_list,
            "attention_mask": attention_masks,
            "labels": labels_list,
            "current_ctx_len": torch.tensor(current_ctx, dtype=torch.int32),
            "orig_ctx_len": torch.tensor([len(ids) for ids in input_ids_list], dtype=torch.int32),
            "current_batch_size": torch.tensor(len(input_ids_list), dtype=torch.int32),
        }

        for k in self._extra_keys:
            if k not in rows[0]:
                continue
            try:
                batch[k] = _slice_to_current_no_padding([r[k] for r in rows])
            except Exception:
                batch[k] = [r[k] for r in rows]

        if self._step % 50 == 0:
            tokens_rank = int(len(input_ids_list)) * max(1, int(current_ctx))
            print(f"[SkyLadder] step {self._step:5d} ctx_len={current_ctx} "
                  f"batch={len(input_ids_list)} tokens/rank={tokens_rank}")
        return batch

    def __iter__(self):
        self._step = 0
        self._cached_step = -1
        self._cached_context_length = None

        if hasattr(self.sampler, "__iter__"):
            indices = iter(self.sampler)
        else:
            indices = iter(range(len(self.dataset)))

        batch: List[Any] = []
        desired_bs_for_step: Optional[int] = None

        for idx in indices:
            try:
                if len(batch) == 0:
                    current_ctx = self._get_current_context_length()
                    desired_bs_for_step = int(self._get_desired_batch_size(current_ctx))
                    self._last_batch_size = int(desired_bs_for_step)

                sample = self.dataset[idx]
                batch.append(sample)

                if len(batch) >= int(desired_bs_for_step):
                    processed_batch = self._collate_no_padding(batch)
                    yield processed_batch
                    self._step += 1
                    batch = []
                    desired_bs_for_step = None

            except (StopIteration, IndexError):
                break

        if len(batch) > 0 and not self._drop_last:
            processed_batch = self._collate_no_padding(batch)
            yield processed_batch
            self._step += 1


# ======================================
# Enhanced SkyLadder (ctx + gen lengths)
# ======================================
class EnhancedSkyLadder(SkyLadder):
    """Enhanced SkyLadder with generation length scheduling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # swap scheduler for enhanced version
        self._schedule = EnhancedContextWindowScheduler(self.cfg)
        self._cached_generation_length: Optional[int] = None

    # handy getters
    def get_current_generation_length(self) -> int:
        if self._cached_step != self._step:
            ctx_len, gen_len = self._schedule(self._step)
            self._cached_context_length = int(ctx_len)
            self._cached_generation_length = int(gen_len)
            self._cached_step = self._step
        return int(self._cached_generation_length)

    def get_current_lengths(self) -> tuple[int, int]:
        if self._cached_step != self._step:
            ctx_len, gen_len = self._schedule(self._step)
            self._cached_context_length = int(ctx_len)
            self._cached_generation_length = int(gen_len)
            self._cached_step = self._step
        return int(self._cached_context_length), int(self._cached_generation_length)

    # base SkyLadder expects _get_current_context_length to return int
    def _get_current_context_length(self) -> int:
        ctx_len, _ = self.get_current_lengths()
        return int(ctx_len)

    def get_memory_stats(self) -> Dict[str, Any]:
        stats = super().get_memory_stats()
        ctx_len, gen_len = self.get_current_lengths()
        stats.update({
            "current_generation_length": int(gen_len),
            "generation_schedule_type": str(self.cfg.gen_schedule_type),
            "min_gen_len": int(self.cfg.min_gen_len),
            "max_gen_len": int(self.cfg.max_gen_len),
            "gen_ratio": float(self.cfg.gen_ratio),
            "total_seq_len_per_sample": int(ctx_len + gen_len),
        })
        return stats

    def _collate_no_padding(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        # reuse base collation
        batch = super()._collate_no_padding(rows)
        _, gen_len = self.get_current_lengths()
        batch["current_gen_len"] = torch.tensor(int(gen_len), dtype=torch.int32)

        if self._step % 50 == 0:
            ctx_len = int(batch["current_ctx_len"])
            tokens_rank = int(len(batch["prompts"])) * max(1, int(ctx_len))
            print(f"[EnhancedSkyLadder] step {self._step:5d} "
                  f"ctx_len={ctx_len} gen_len={int(gen_len)} "
                  f"batch={len(batch['prompts'])} tokens/rank={tokens_rank}")
        return batch
