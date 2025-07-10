# --- put this inside your SkyLadder class ---------------------------------
def _collate(self, rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate a list of samples into a single batch **with globally-consistent
    shape**.

    1.  Rank-0 decides the target context length (`ctx_len`).
    2.  All ranks broadcast that scalar so they use the same number.
    3.  We slice each sample to `ctx_len` and pad on the **right** if needed.
    """
    # ------------------------------------------------------------------
    # 1. rank-0 decides ctx_len for this step
    # ------------------------------------------------------------------
    local_ctx = self._schedule(self._step)

    # broadcast  -->  every rank now has identical ctx_len (int on CPU)
    ctx_len_tensor = torch.tensor([local_ctx], dtype=torch.int32, device="cpu")
    if torch.distributed.is_initialized():
        torch.distributed.broadcast(ctx_len_tensor, src=0)
    ctx_len = int(ctx_len_tensor.item())

    # ------------------------------------------------------------------
    # 2. helper that slices + pads a 1-D tensor to ctx_len
    # ------------------------------------------------------------------
    def _slice_pad(t: torch.Tensor) -> torch.Tensor:
        t = t[:ctx_len]                                         # slice
        if t.numel() < ctx_len:                                 # right-pad
            pad_amt = ctx_len - t.numel()
            t = torch.nn.functional.pad(t, (0, pad_amt))
        return t

    # ------------------------------------------------------------------
    # 3. build the batch dict
    # ------------------------------------------------------------------
    batch = {
        "input_ids":     torch.stack([_slice_pad(r["input_ids"])     for r in rows]),
        "attention_mask":torch.stack([_slice_pad(r["attention_mask"])for r in rows]),
        "labels":        torch.stack([_slice_pad(r["labels"])        for r in rows]),
    }

    # any extra keys – try to slice/pad if tensor-like, else keep list
    for k in self._extra_keys:
        if k not in rows[0]:
            continue
        sample0 = rows[0][k]
        if torch.is_tensor(sample0) or isinstance(sample0, np.ndarray):
            batch[k] = torch.stack([_slice_pad(torch.as_tensor(r[k])) for r in rows])
        else:
            batch[k] = [r[k] for r in rows]

    return batch
# ---------------------------------------------------------------------------
