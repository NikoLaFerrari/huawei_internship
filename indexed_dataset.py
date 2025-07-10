# ----------------------------------------------------------------------
# Robust replacement for get_packed_indexed_dataset
# ----------------------------------------------------------------------
def get_packed_indexed_dataset(data_prefix: str,
                               filter_length: Optional[int] = None):
    """
    Build a CombinedDataset from packed binary shards.

    Expected files:
        {data_prefix}_packed_<field>_document.bin
        {data_prefix}_packed_<field>_document.idx
    where <field> includes at least "input_ids".

    Parameters
    ----------
    data_prefix : str
        Path prefix up to (and including) the dataset name, *without* the
        "_packed_<field>_document" suffix.
    filter_length : Optional[int]
        If given, samples whose `input_ids` length > filter_length are removed
        (same mask applied to every field).

    Raises
    ------
    FileNotFoundError
        If no packed files are found, or the required *input_ids* field is
        missing.
    ValueError
        If filtering removes every sample in the dataset.
    """
    pattern_glob = f"{data_prefix}_packed_*_document*"
    paths = glob.glob(pattern_glob)
    if not paths:
        raise FileNotFoundError(
            f"No packed shards match '{pattern_glob}'. "
            "Check `data_path` or run the packing script first."
        )

    # Regex that works on *base names* so directories don't break the match.
    rex = re.compile(rf"{re.escape(os.path.basename(data_prefix))}_packed_(.*?)_document")
    fields: set[str] = set()

    for p in paths:
        m = rex.search(os.path.basename(p))
        if m:
            fields.add(m.group(1))

    if "input_ids" not in fields:
        raise FileNotFoundError(
            "Required shard '*_packed_input_ids_document.*' is missing for "
            f"prefix '{data_prefix}'. Found fields: {', '.join(sorted(fields)) or 'NONE'}"
        )

    packed_dataset: dict[str, IndexedDataset] = {}
    for field in sorted(fields):
        max_len = filter_length if (filter_length and field == "input_ids") else None
        packed_dataset[field] = IndexedDataset(
            f"{data_prefix}_packed_{field}_document",
            max_len=max_len,
        )

    # Optional length-based filter
    if filter_length:
        mask = packed_dataset["input_ids"].get_filter_mask()
        for ds in packed_dataset.values():
            ds.do_filter(mask)

        if len(packed_dataset["input_ids"]) == 0:
            raise ValueError(
                f"All samples were filtered out by `filter_length={filter_length}` "
                f"for prefix '{data_prefix}'. Reduce filter_length or repack."
            )

    return CombinedDataset(packed_dataset)
