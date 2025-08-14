
from __future__ import annotations
from collections import defaultdict
import math, random
from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# Optional mask import; fallback if not available
try:
    from mindspeed_rl.datasets.skyladder_mask_cache import get_causal_mask  # noqa: F401
except Exception:
    def get_causal_mask(seq_len: int, device=None) -> torch.Tensor:
        # Not used by default (we avoid LxL masks), kept for API compatibility
        return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))


# -----------------------------------------------------------------------------
# Simple prompt dataloaders (unchanged)
# -----------------------------------------------------------------------------
class PromptDataLoader(torch.utils.data.DataLoader):
    """PromptDataLoader.

    Args:
        dataset: An Prompt Implementation of BaseDataset
        global_batch_size: global batch size for loader
        num_workers: workers of dataloader
        seed: random seed
        dataset_additional_keys: extra keys for data loading
        no_shuffle: disable random sampling
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


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class SkyLadderCfg:
    # ---- Context schedule ----
    min_ctx_len: int = 1024
    max_ctx_len: int = 8192
    total_steps: int = 30            # total training steps (for schedule)
    warmup_steps: int = 0            # optional low-L hold at the start
    changing_point: Optional[int] = 20   # ramp length; rest is plateau at max
    schedule_type: str = "cosine"    # linear|cosine|sin|exp|log|inv

    # ---- Generation schedule ----
    gen_schedule_type: str = "follow_context"  # follow_context|independent|fixed
    gen_ratio: float = 1.0                     # base ρ
    min_gen_ratio: float = 1
    max_gen_ratio: float = 1
    min_gen_len: int = 1024
    max_gen_len: int = 8192
    fixed_gen_len: Optional[int] = None        # used when gen_schedule_type == "fixed"
    gen_jitter_pct: float = 0               # ± jitter on G; deterministic per step
    align_multiple: int = 8                   # align L and G to multiples (kernel-friendly)

    # ---- Global caps ----
    max_total_len: Optional[int] = 16384       # enforce L + G ≤ this (e.g., vLLM max_model_len)

    # ---- Dynamic batching (ctx-only in this deterministic version) ----
    dynamic_batch: bool = False
    target_tokens_per_rank: int = 32768        # often == engine's max_num_batched_tokens
    memory_safety_factor: float = 0.88
    min_batch_size: int = 1
    max_batch_size: int = 16
    round_batch_to: int = 1                    # round B to a granule (nice for PP)

    # ---- Misc ----
    no_shuffle: bool = False
    print_every: int = 1                        # debug cadence

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


# -----------------------------------------------------------------------------
# Legacy context-only scheduler (kept for API compatibility)
# -----------------------------------------------------------------------------
class ContextWindowScheduler:
    """Return ONLY context length for a given step (base SkyLadder pre-RL)."""

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


# -----------------------------------------------------------------------------
# Enhanced scheduler (ctx + gen), front-end compatible with prior code
# -----------------------------------------------------------------------------
class EnhancedContextWindowScheduler:
    """Return (context_length, generation_length) for a given step.
       Internally delegates to the unified _CtxGenScheduler to avoid drift.
    """
    def __init__(self, cfg: SkyLadderCfg):
        self.cfg = cfg
        self.init_mask_length = int(cfg.min_ctx_len)
        self.final_mask_length = int(cfg.max_ctx_len)
        self.min_gen_length = int(cfg.min_gen_len)
        self.max_gen_length = int(cfg.max_gen_len)
        self.changing_point = int(cfg.changing_point) if cfg.changing_point is not None else int(cfg.total_steps)
        self._ctx_fn = {
            "linear": self._linear, "cosine": self._cosine, "sin": self._sine,
            "exp": self._exp, "log": self._log, "inv": self._inv,
        }.get(cfg.schedule_type, self._cosine)

    def __call__(self, step: int) -> tuple[int, int]:
        L, G = _CtxGenScheduler(self.cfg).lengths(step)
        return int(L), int(G)

    # --- Legacy helpers (kept for API compatibility if some code calls these) ---
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


# -----------------------------------------------------------------------------
# Unified ctx+gen scheduler used by the loaders
# -----------------------------------------------------------------------------
def _align_down(x: int, m: int) -> int:
    if m <= 1:
        return int(x)
    return int(max(1, (int(x) // int(m)) * int(m)))

class _CtxGenScheduler:
    """Return (L, G) for a step. Deterministic: G≈ρ·L (ratio-clamped) + deterministic jitter per step.
       Enforces L+G≤max_total_len by shrinking L if needed to preserve G≥min_gen_len.
    """
    def __init__(self, cfg: SkyLadderCfg):
        self.cfg = cfg
        self._ctx_fn = {
            "linear": self._linear, "cosine": self._cosine, "sin": self._sine,
            "exp": self._exp, "log": self._log, "inv": self._inv,
        }.get(cfg.schedule_type, self._cosine)

    # ---- public ----
    def lengths(self, step: int) -> Tuple[int, int]:
        L = self._ctx(step)
        G = self._gen(step, L)

        # Primary cap: enforce L + G ≤ max_total_len (if set).
        if self.cfg.max_total_len is not None:
            cap = int(self.cfg.max_total_len)
            # First try to clamp G down if needed
            if L + G > cap:
                G = max(self.cfg.min_gen_len, min(G, cap - L))
            # If still violating (cap - L < min_gen_len), shrink L to make room for min_gen_len
            if L + G > cap:
                L = max(self.cfg.min_ctx_len, cap - max(self.cfg.min_gen_len, 1))
                L = _align_down(L, self.cfg.align_multiple)
                G = max(self.cfg.min_gen_len, min(G, cap - L))

        # Align down to kernel-friendly multiples
        L = _align_down(max(self.cfg.min_ctx_len, L), self.cfg.align_multiple)
        G = _align_down(max(self.cfg.min_gen_len, G), self.cfg.align_multiple)

        # Absolute max caps
        L = min(L, int(self.cfg.max_ctx_len))
        G = min(G, int(self.cfg.max_gen_len))

        # Final guard: never negative and try to keep L+G<=cap if provided
        if self.cfg.max_total_len is not None and (L + G > int(self.cfg.max_total_len)):
            G = max(self.cfg.min_gen_len, int(self.cfg.max_total_len) - L)
            if G < self.cfg.min_gen_len:
                # As a last resort make L just big enough for min_gen_len
                L = max(self.cfg.min_ctx_len, int(self.cfg.max_total_len) - self.cfg.min_gen_len)
                L = _align_down(L, self.cfg.align_multiple)
                G = self.cfg.min_gen_len

        return int(L), int(G)

    # ---- context ----
    def _ctx(self, step: int) -> int:
        if step < self.cfg.warmup_steps:
            return int(self.cfg.min_ctx_len)
        s = max(0, int(step) - int(self.cfg.warmup_steps))
        p = min(1.0, s / max(1, int(self.cfg.changing_point)))
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
        # map p∈[0,1] via log1p for a gentle early growth
        return int(self.cfg.min_ctx_len * (self.cfg.max_ctx_len / max(1, self.cfg.min_ctx_len)) ** (math.log1p(p) / math.log1p(1.0)))

    def _inv(self, p: float) -> int:
        lin = self._linear(p)
        return int(self.cfg.max_ctx_len + self.cfg.min_ctx_len - lin)

    # ---- generation (DETERMINISTIC) ----
    def _gen(self, step: int, L: int) -> int:
        t = max(0, int(step) - int(self.cfg.warmup_steps))

        if self.cfg.gen_schedule_type == "fixed":
            G = int(self.cfg.fixed_gen_len if self.cfg.fixed_gen_len is not None else self.cfg.max_gen_len)
            return int(max(self.cfg.min_gen_len, min(G, self.cfg.max_gen_len)))

        if self.cfg.gen_schedule_type == "independent":
            cp = max(1, int(self.cfg.changing_point))
            p  = min(1.0, t / cp)
            base = self.cfg.min_gen_len + (self.cfg.max_gen_len - self.cfg.min_gen_len) * (1 - math.cos(math.pi * p)) / 2
            return int(base)

        # follow_context (default): G ≈ ρ·L with ratio clamps
        rho = float(self.cfg.gen_ratio)
        rho = max(self.cfg.min_gen_ratio, min(rho, self.cfg.max_gen_ratio))
        target = int(round(rho * int(L)))

        # Deterministic jitter per step (set gen_jitter_pct=0.0 to disable)
        if self.cfg.gen_jitter_pct > 0.0:
            rng = random.Random(int(step))  # seed by step for determinism across ranks
            jitter = 1.0 + rng.uniform(-self.cfg.gen_jitter_pct, self.cfg.gen_jitter_pct)
            target = int(round(target * jitter))

        # Absolute clamp
        G = max(self.cfg.min_gen_len, min(target, self.cfg.max_gen_len))
        return int(G)


# -----------------------------------------------------------------------------
# SkyLadder DataLoader (ctx+gen) with ctx-only dynamic batching (deterministic)
# -----------------------------------------------------------------------------
class SkyLadder(DataLoader):
    """
    Deterministic SkyLadder for RL:
      - computes (L, G) per step (G follows L deterministically),
      - batches on CONTEXT LENGTH ONLY (ctx-only) to match prior high TPS,
      - truncates inputs to L (decoder uses G),
      - yields metadata for logging/inference engine.
    """

    def __init__(
        self,
        dataset: Union[torch.utils.data.Dataset, str],
        global_batch_size: int,         # baseline when dynamic batching is off
        num_workers: int,
        seed: int,
        dataset_additional_keys: Sequence[str],
        no_shuffle: bool,
        cfg: SkyLadderCfg | Mapping | Any = None,
        **kwargs,
    ):
        self.cfg = self._normalize_cfg(cfg)
        self.dataset = dataset
        self._schedule = _CtxGenScheduler(self.cfg)

        self._baseline_bs = max(1, int(global_batch_size))
        self._step = 0
        self._drop_last = bool(kwargs.get("drop_last", True))
        self._extra_keys = list(dataset_additional_keys or [])

        sampler = (
            SequentialSampler(dataset)
            if bool(no_shuffle or self.cfg.no_shuffle)
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
            pin_memory=bool(kwargs.get("pin_memory", True)),
            drop_last=self._drop_last,
            **{k: v for k, v in kwargs.items() if k not in ("collate_fn", "pin_memory", "drop_last")}
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
        # Accept alias 'rho' -> gen_ratio
        if "rho" in d and "gen_ratio" not in filtered:
            filtered["gen_ratio"] = d["rho"]
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
        """
        Deterministic 'v1' behavior: batch-size based on CONTEXT length ONLY.
        B ≈ floor(target_tokens_per_rank / L), clamped and rounded.
        This matches the earlier runs that achieved ~55 TPS.
        """
        if not bool(self.cfg.dynamic_batch) or not self.cfg.target_tokens_per_rank:
            return self.cfg._baseline_bs
        
        if G < 3000: return self.cfg._baseline_bs*2
        ctx = max(1, int(L))
        eff_target = max(1, int(self.cfg.target_tokens_per_rank * float(self.cfg.memory_safety_factor)))
        desired = eff_target // ctx

        # Clamp to [min_batch_size, max_batch_size]
        
        desired = max(int(self.cfg.min_batch_size), min(int(desired), int(self.cfg.max_batch_size)))

        # Optional pipeline granule rounding
        gran = int(self.cfg.round_batch_to or 1)
        if gran > 1:
            desired = max(int(self.cfg.min_batch_size), (desired // gran) * gran)

        if desired < 1:
            desired = 1

        # Lightweight debug (helps confirm B isn't collapsing)
        step = getattr(self, "_step", -1)
        if self.cfg.print_every and (step % max(1, int(self.cfg.print_every)) == 0):
            print(f"[SkyLadder-RL/B(ctx-only)] step={step} L={L} G={G} "
                  f"ctx={ctx} target_tokens_per_rank={self.cfg.target_tokens_per_rank} "
                  f"safety={self.cfg.memory_safety_factor} -> desired={desired} "
                  f"(min={self.cfg.min_batch_size}, max={self.cfg.max_batch_size}, gran={gran})")
    
        print(f"[SkyLadder] batch_size: max,min,current = {self.cfg.max_batch_size},{self.cfg.min_batch_size},{desired}")
        return int(desired)

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
            print(f"[SkyLadder-RL(det)] step {self._step:5d} L={L} G={G} "
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
                # start of a new step: compute (L, G), then choose B (ctx-only)
                if len(batch_rows) == 0:
                    L, G = self._schedule.lengths(self._step)
                    # Final safety: ensure L + G cap again (consistent with schedule)
                    if self.cfg.max_total_len is not None and (L + G > int(self.cfg.max_total_len)):
                        G = max(self.cfg.min_gen_len, int(self.cfg.max_total_len) - L)
                        if G < self.cfg.min_gen_len:
                            L = max(self.cfg.min_ctx_len, int(self.cfg.max_total_len) - self.cfg.min_gen_len)
                            L = _align_down(L, self.cfg.align_multiple)
                            G = self.cfg.min_gen_len

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

                    if self._step >= int(self.cfg.total_steps):
                        break

            except (StopIteration, IndexError):
                break

        if len(batch_rows) > 0 and not self._drop_last and self._step < int(self.cfg.total_steps):
            processed = self._collate_no_padding(batch_rows)
            yield processed
            self._step += 1


# -----------------------------------------------------------------------------
# Enhanced SkyLadder (subclass) — keeps the class name for compatibility
# -----------------------------------------------------------------------------
class EnhancedSkyLadder(SkyLadder):
    """Enhanced SkyLadder retaining the same public surface; uses EnhancedContextWindowScheduler if needed.
       (By default we already produce (L,G) in SkyLadder; this class preserves prior imports.)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Swap scheduler to the front-end wrapper (internally uses _CtxGenScheduler)
        self._schedule = _CtxGenScheduler(self.cfg)  # or EnhancedContextWindowScheduler(self.cfg)
        self._cached_generation_length: Optional[int] = None

    def get_current_generation_length(self) -> int:
        # Provided for compatibility with any code relying on this accessor.
        return int(self._last_gen)

    def get_current_lengths(self) -> tuple[int, int]:
        return int(self._last_ctx), int(self._last_gen)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _normalize_cfg_like(cfg_like) -> SkyLadderCfg:
    if isinstance(cfg_like, SkyLadderCfg):
        return cfg_like
    if isinstance(cfg_like, Mapping):
        valid = {f.name for f in fields(SkyLadderCfg)}
        d = {k: cfg_like[k] for k in cfg_like if k in valid}
        if "rho" in cfg_like and "gen_ratio" not in d:
            d["gen_ratio"] = cfg_like["rho"]
        return SkyLadderCfg(**d)
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
    "PromptDataLoader",
    "MultiModalDataLoader",
    "SkyLadderCfg",
    "ContextWindowScheduler",
    "EnhancedContextWindowScheduler",
    "_CtxGenScheduler",
    "SkyLadder",
    "EnhancedSkyLadder",
    "make_scheduler",
    "lengths_for_step",
    "schedule_table",
]
