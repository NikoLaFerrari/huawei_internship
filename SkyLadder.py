from torch.utils.data import DataLoader

class SkyLadder(DataLoader):
    """Enhanced DataLoader with Skyladder optimizations for packed binary datasets"""

    SHINGLE_SIZE = 5  # Class constant for shingling

    def __init__(self,
                 dataset: Dataset ,
                 global_batch_size: int,
                 num_workers: int,
                 seed: int,
                 dataset_additional_keys: List[str],
                 no_shuffle: bool,
                 skyladder_optimizations: bool = True):
        """
        Args matching PromptDataLoader:
            dataset: Input dataset (PackedBinaryDataset or similar)
            global_batch_size: Batch size for processing
            num_workers: Parallel data loading workers
            seed: Random seed for reproducibility
            dataset_additional_keys: Extra keys to include in batches
            no_shuffle: Disable shuffling if True
            
        Skyladder-specific:
            skyladder_optimizations: Enable/disable optimizations
        """
        self.shingle_cache = {}
        self.minhash_cache = {}
        self.batch_optimizations = skyladder_optimizations
        self.collator = self._create_collator(dataset_additional_keys)

        if not no_shuffle:
            sampler = torch.utils.data.RandomSampler(
                dataset,
                generator=torch.Generator().manual_seed(seed)
            )
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)


      super().__init__(
            dataset=dataset,
            batch_size=32 if global_batch_size is None else int(global_batch_size),
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=self.collator,
            pin_memory=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(seed)
        )

        if skyladder_optimizations:
            self._init_skyladder()

    def _create_collator(self, additional_keys: List[str]):
        def collate(features: List[Dict]) -> Dict:
            batch = {
                'input_ids': torch.stack([torch.as_tensor(f['input_ids']) for f in features]),
                'attention_mask': torch.stack([torch.as_tensor(f['attention_mask']) for f in features]),
                'labels': torch.stack([torch.as_tensor(f['labels']) for f in features])
            }

            for key in additional_keys:
                if key in features[0]:
                    batch[key] = torch.stack([torch.tensor(f[key]) for f in features])
            return batch
        return collate

    def _init_skyladder(self):
        """Initialize Skyladder optimizations"""
        self.prefetch_factor = 2
        self._read_ahead_buffer = []
        self._buffer_size = 4  # Number of batches to buffer
        self.batch_times = []
        self.throughput = 0.0

    def shingle_text(self, text: str, k: int = SHINGLE_SIZE) -> List[str]:
        cache_key = xxhash.xxh64_intdigest(text.encode('utf-8'))
        if cache_key in self.shingle_cache:
            return self.shingle_cache[cache_key]

        shingles = [text[i:i+k] for i in range(len(text)-k+1)]
        self.shingle_cache[cache_key] = shingles
        return shingles

    def create_minhash(self, shingles: List[str]) -> 'MinHash':
        cache_key = xxhash.xxh64_intdigest(str(shingles).encode('utf-8'))
        if cache_key in self.minhash_cache:
            return self.minhash_cache[cache_key].copy()

        mh = MinHash(num_perm=128)  # Assuming MinHash is imported
        for shingle in shingles:
            mh.update(shingle.encode('utf-8'))
        self.minhash_cache[cache_key] = mh.copy()
        return mh
