# coding=utf-8
# SkyLadder + EnhancedSkyLadder (keeps your original SkyLadderCfg fields)
# No new YAML keys. Minimal, surgical fixes only.

from __future__ import annotations
from collections import defaultdict
import math, random
from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
#

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


# =========================
# Config
# =========================

@dataclass
class SkyLadderCfg:
    # ---- Context schedule ----
    min_ctx_len: int = 1024
    max_ctx_len: int = 8192
    total_steps: int = 30           # total training steps (for schedule)
    warmup_steps: int = 0           # optional low-L hold at the start
    changing_point: Optional[int] = 20  # ramp length; rest is plateau at max
    schedule_type: str = "cosine"   # linear|cosine|sin|exp|log|inv

    # ---- Generation schedule (follow context) ----
    gen_schedule_type: str = "follow_context"   # follow_context|independent|fixed
    gen_ratio: float = 1.0                      # base ρ
    min_gen_ratio: float = 0.7
    max_gen_ratio: float = 1.2
    min_gen_len: int = 1024
    max_gen_len: int = 8192
    fixed_gen_len: Optional[int] = None         # used when gen_schedule_type == "fixed"
    gen_jitter_pct: float = 0.10                # ± jitter on G to avoid long-tail stalls
    align_multiple: int = 64                    # align L and G to multiples (kernel-friendly)

    # ---- Global caps ----
    max_total_len: Optional[int] = 16384        # enforce L + G ≤ this (e.g., vLLM max_model_len)

    # ---- Dynamic batching: batch on (L + G) to hit token cap ----
    dynamic_batch: bool = True
    target_tokens_per_rank: int = 65536         # set to engine's max_num_batched_tokens
    memory_safety_factor: float = 1.0           # keep 1.0 to actually hit the cap
    min_batch_size: int = 1
    max_batch_size: int = 16
    round_batch_to: int = 2                     # round batch to granule (nice for PP)

    # ---- Misc ----
    no_shuffle: bool = False
    print_every: int = 1                       # debug cadence

    # Validate / clamp
    def __post_init__(self):
        if self.total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if self.changing_point is None:
            self.changing_point = self.total_steps
        if self.changing_point <= 0:
            raise ValueError("changing_point must be > 0")
        if not (0 <= self.gen_jitter_pct <= 0.5):
            raise ValueError("gen_jitter_pct out of range [0, 0.5]")
        self.gen_ratio = float(self.gen_ratio)
        self.min_gen_ratio = float(self.min_gen_ratio)
        self.max_gen_ratio = float(self.max_gen_ratio)
        if self.min_gen_ratio > self.max_gen_ratio:
            self.min_gen_ratio, self.max_gen_ratio = self.max_gen_ratio, self.min_gen_ratio
        if self.align_multiple <= 0:
            self.align_multiple = 1

# =========================
# Schedulers
# =========================

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
        # delegate to the unified scheduler to avoid drift
        L, G = _CtxGenScheduler(self.cfg).lengths(step)
        return int(L), int(G)

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
            tgt = int(self.cfg.fixed_gen_len or self.max_gen_length)
            gen = max(self.min_gen_length, min(tgt, self.max_gen_length))
            # (optional) enforce L+G cap if you keep this class
            if self.cfg.max_total_len is not None and current_ctx_len is not None:
                gen = min(gen, int(self.cfg.max_total_len) - int(current_ctx_len))
                gen = max(self.min_gen_length, gen)
            return int(gen)

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



def _align_down(x: int, m: int) -> int:
    if m <= 1: return int(x)
    return int(max(1, (x // m) * m))

class _CtxGenScheduler:
    """Returns (L, G) lengths for a step, then batching uses (L+G)."""
    def __init__(self, cfg: SkyLadderCfg):
        self.cfg = cfg
        self._ctx_fn = {
            "linear": self._linear,
            "cosine": self._cosine,
            "sin":    self._sine,
            "exp":    self._exp,
            "log":    self._log,
            "inv":    self._inv,
        }.get(cfg.schedule_type, self._cosine)

    # ---- public ----
    def lengths(self, step: int) -> Tuple[int, int]:
        L = self._ctx(step)
        G = self._gen(step, L)
        # enforce L + G cap (after ratio/jitter), and align both
        if self.cfg.max_total_len is not None:
            G = min(G, int(self.cfg.max_total_len) - int(L))
        L = _align_down(max(self.cfg.min_ctx_len, L), self.cfg.align_multiple)
        G = _align_down(max(self.cfg.min_gen_len, G), self.cfg.align_multiple)
        # respect absolute max caps
        L = min(L, int(self.cfg.max_ctx_len))
        G = min(G, int(self.cfg.max_gen_len))
        # final safety: don't let G go negative due to caps
        G = max(self.cfg.min_gen_len, G)
        return int(L), int(G)

    # ---- context ----
    def _ctx(self, step: int) -> int:
        if step < self.cfg.warmup_steps:
            return int(self.cfg.min_ctx_len)
        s = max(0, step - self.cfg.warmup_steps)
        p = min(1.0, s / max(1, self.cfg.changing_point))
        return int(self._ctx_fn(p))

    def _linear(self, p: float) -> int:
        return int(self.cfg.min_ctx_len + (self.cfg.max_ctx_len - self.cfg.min_ctx_len) * p)

    def _cosine(self, p: float) -> int:
        return int(self.cfg.min_ctx_len + (self.cfg.max_ctx_len - self.cfg.min_ctx_len) * (1 - math.cos(math.pi * p)) / 2)

    def _sine(self, p: float) -> int:
        return int(self.cfg.min_ctx_len + (self.cfg.max_ctx_len - self.cfg.min_ctx_len) * math.sin((math.pi / 2) * p))

    def _exp(self, p: float) -> int:
        base = (self.cfg.max_ctx_len / max(1, self.cfg.min_ctx_len))
        return int(self.cfg.min_ctx_len * (base ** p))

    def _log(self, p: float) -> int:
        return int(self.cfg.min_ctx_len * (self.cfg.max_ctx_len / max(1, self.cfg.min_ctx_len)) ** (math.log1p(p) / math.log1p(1.0)))

    def _inv(self, p: float) -> int:
        # inverse linear around center
        lin = self._linear(p)
        return int(self.cfg.max_ctx_len + self.cfg.min_ctx_len - lin)

    # ---- generation ----
    def _gen(self, step: int, L: int) -> int:
        t = max(0, step - self.cfg.warmup_steps)
        if self.cfg.gen_schedule_type == "fixed":
            G = int(self.cfg.fixed_gen_len or self.cfg.max_gen_len)
            return int(G)

        if self.cfg.gen_schedule_type == "independent":
            # same shape as ctx over [min_gen_len, max_gen_len]
            p = min(1.0, t / max(1, self.cfg.changing_point))
            base = self.cfg.min_gen_len + (self.cfg.max_gen_len - self.cfg.min_gen_len) * (1 - math.cos(math.pi * p)) / 2
            return int(base)

        # follow_context (default)
        rho = float(self.cfg.gen_ratio)
        rho = max(self.cfg.min_gen_ratio, min(rho, self.cfg.max_gen_ratio))
        target = int(round(rho * int(L)))
        # jitter to avoid long-tail decode stalls
        if self.cfg.gen_jitter_pct > 0:
            jitter = 1.0 + random.uniform(-self.cfg.gen_jitter_pct, self.cfg.gen_jitter_pct)
            target = int(round(target * jitter))
        # absolute clamp
        G = max(self.cfg.min_gen_len, min(target, self.cfg.max_gen_len))
        return int(G)

# =========================
# DataLoader (variable batch on L+G)
# =========================

class SkyLadder(DataLoader):
    """
    Throughput-optimal RL dataloader:
      - computes (L, G) per step (G follows L),
      - batches on (L + G) to hit the token cap,
      - truncates inputs to L (decoder will use G),
      - yields metadata for logging/inference engine.
    """

    def __init__(
        self,
        dataset: Union[torch.utils.data.Dataset, str],
        global_batch_size: int,                 # baseline when dynamic batching is off
        num_workers: int,
        seed: int,
        dataset_additional_keys: Sequence[str],
        cfg: SkyLadderCfg | Mapping | Any,
        drop_last: bool = True,
        pin_memory: bool = True,
    ):
        self.cfg = self._normalize_cfg(cfg)
        self.dataset = dataset
        self._schedule = _CtxGenScheduler(self.cfg)

        self._baseline_bs = max(1, int(global_batch_size))
        self._step = 0
        self._drop_last = bool(drop_last)
        self._extra_keys = list(dataset_additional_keys or [])

        sampler = (
            SequentialSampler(dataset)
            if bool(self.cfg.no_shuffle)
            else RandomSampler(dataset, generator=torch.Generator().manual_seed(seed))
        )

        # stash last sizes for stats
        self._last_ctx = int(self.cfg.min_ctx_len)
        self._last_gen = int(self.cfg.min_gen_len)
        self._last_batch_size = int(self._baseline_bs)

        super().__init__(
            dataset=dataset,
            batch_size=global_batch_size,
            sampler=sampler,
            num_workers=int(num_workers),
            collate_fn=self._collate_no_padding,
            pin_memory=bool(pin_memory),
            drop_last=self._drop_last,
        )

    # ---- config normalize ----
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
            return SkyLadderCfg()

        valid = {f.name for f in fields(SkyLadderCfg)}
        filtered = {k: d[k] for k in d.keys() if k in valid}
        return SkyLadderCfg(**filtered)

    # ---- public helpers ----
    def get_current_lengths(self) -> Tuple[int, int]:
        return int(self._last_ctx), int(self._last_gen)

    def get_last_batch_size(self) -> int:
        return int(self._last_batch_size)

    def get_memory_stats(self) -> Dict[str, Any]:
        total = int(self._last_ctx) + int(self._last_gen)
        return {
            "step": int(self._step),
            "current_ctx_len": int(self._last_ctx),
            "current_gen_len": int(self._last_gen),
            "total_seq_len": int(total),
            "current_batch_size": int(self._last_batch_size),
            "tokens_per_rank_step": int(self._last_batch_size) * int(total),
            "target_tokens_per_rank": int(self.cfg.target_tokens_per_rank),
            "dynamic_batch_enabled": bool(self.cfg.dynamic_batch),
        }

    # ---- internals ----
    def _desired_batch_size(self, L: int, G: int) -> int:
        if not bool(self.cfg.dynamic_batch) or not self.cfg.target_tokens_per_rank:
            return int(self._baseline_bs)

        total = max(1, int(L) + int(G))
        eff_target = max(1, int(self.cfg.target_tokens_per_rank * float(self.cfg.memory_safety_factor)))
        desired = min(int(self.cfg.max_batch_size), eff_target // total)
        print(f"[SkyLadder dataloader.py] desired batch_size inside SkyLadder: min: {self.cfg.min_batch_size}, max: {self.cfg.max_batch_size}, current: {int(desired)}")

        if self.cfg.max_batch_size is not None:
            desired = min(int(desired), int(self.cfg.max_batch_size))
        # optional parallelism granule
        r = int(self.cfg.round_batch_to or 1)
        if r > 1:
            desired = max(int(self.cfg.min_batch_size), (desired // r) * r)
        return max(1, int(desired))

    def _collate_no_padding(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        L, G = int(self._last_ctx), int(self._last_gen)

        def _slice_to_L(seq_list: List[Any]) -> List[torch.Tensor]:
            out: List[torch.Tensor] = []
            for t in seq_list:
                tt = torch.as_tensor(t)[:L]
                out.append(tt)
            return out

        input_key = "prompts" if "prompts" in rows[0] else "input_ids"
        input_ids_list = _slice_to_L([r[input_key] for r in rows])
        labels_list    = _slice_to_L([r.get("labels", r[input_key]) for r in rows])

        batch: Dict[str, Any] = {
            "prompts": input_ids_list,
            "attention_mask": None,
            "labels": labels_list,
            "current_ctx_len": torch.tensor(L, dtype=torch.int32),
            "current_gen_len": torch.tensor(G, dtype=torch.int32),
            "current_batch_size": torch.tensor(len(input_ids_list), dtype=torch.int32),
            "total_seq_len": torch.tensor(L + G, dtype=torch.int32),
        }

        for k in self._extra_keys:
            if k not in rows[0]:
                continue
            try:
                batch[k] = _slice_to_L([r[k] for r in rows])
            except Exception:
                batch[k] = [r[k] for r in rows]

        if self.cfg.print_every and (self._step % max(1, int(self.cfg.print_every)) == 0):
            tokens_rank = int(len(input_ids_list)) * max(1, int(L + G))
            print(f"[SkyLadder-RL] step {self._step:5d} L={L} G={G} "
                  f"B={len(input_ids_list)} tokens/rank={tokens_rank}")
        return batch

    def __iter__(self):
        self._step = 0
        if hasattr(self.sampler, "__iter__"):
            indices = iter(self.sampler)
        else:
            indices = iter(range(len(self.dataset)))

        batch_rows: List[Any] = []
        desired_bs: Optional[int] = None

        for idx in indices:
            try:
                # start of a new step: compute L,G then B*(L+G)
                if len(batch_rows) == 0:
                    L, G = self._schedule.lengths(self._step)
                    # enforce L + G <= max_total_len (again, safety)
                    if self.cfg.max_total_len is not None:
                        G = min(int(G), int(self.cfg.max_total_len) - int(L))
                        G = max(self.cfg.min_gen_len, int(G))

                    B = self._desired_batch_size(L, G)

                    self._last_ctx = int(L)
                    self._last_gen = int(G)
                    self._last_batch_size = int(B)
                    desired_bs = int(B)

                sample = self.dataset[idx]
                batch_rows.append(sample)

                if len(batch_rows) >= int(desired_bs):
                    processed = self._collate_no_padding(batch_rows)
                    yield processed
                    self._step += 1
                    batch_rows = []
                    desired_bs = None

                    # end of run guard
                    if self._step >= self.cfg.total_steps:
                        break

            except (StopIteration, IndexError):
                break

        if len(batch_rows) > 0 and not self._drop_last and self._step < self.cfg.total_steps:
            processed = self._collate_no_padding(batch_rows)
            yield processed
            self._step += 1

def _normalize_cfg_like(cfg_like) -> SkyLadderCfg:
    if isinstance(cfg_like, SkyLadderCfg):
        return cfg_like
    if isinstance(cfg_like, Mapping):
        valid = {f.name for f in fields(SkyLadderCfg)}
        return SkyLadderCfg(**{k: cfg_like[k] for k in cfg_like if k in valid})
    return SkyLadderCfg()

def make_scheduler(cfg_like) -> _CtxGenScheduler:
    return _CtxGenScheduler(_normalize_cfg_like(cfg_like))

def lengths_for_step(cfg_like, step: int) -> Tuple[int, int]:
    sched = make_scheduler(cfg_like)
    return sched.lengths(int(step))

def schedule_table(cfg_like, total_steps: int | None = None):
    cfg = _normalize_cfg_like(cfg_like)
    T = int(total_steps or cfg.total_steps)
    sch = _CtxGenScheduler(cfg)
    return [sch.lengths(t) for t in range(T)]

__all__ = [
    "SkyLadderCfg",
    "SkyLadder",
    "EnhancedContextWindowScheduler",
    "_CtxGenScheduler",
    "make_scheduler",
    "lengths_for_step",
    "schedule_table",
]
