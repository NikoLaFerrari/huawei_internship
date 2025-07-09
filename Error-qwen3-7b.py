'''
Error executing job with overrides: []
Traceback (most recent call last):
  File "/models/k0050048751/MindSpeed-RL-master/cli/train_grpo.py", line 574, in <module>
    main()
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/hydra/main.py", line 94, in decorated_main
    _run_hydra(
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/hydra/_internal/utils.py", line 394, in _run_hydra
    _run_app(
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/hydra/_internal/utils.py", line 457, in _run_app
    run_and_report(
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/hydra/_internal/utils.py", line 223, in run_and_report
    raise ex
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
    return func()
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
    lambda: hydra.run(
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/hydra/_internal/hydra.py", line 132, in run
    _ = ret.return_value
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/hydra/core/utils.py", line 260, in return_value
    raise self._return_value
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/hydra/core/utils.py", line 186, in run_job
    ret.return_value = task_function(task_cfg)
  File "/models/k0050048751/MindSpeed-RL-master/cli/train_grpo.py", line 570, in main
    ray.get(train.remote(config))
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/worker.py", line 2772, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/worker.py", line 919, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(KeyError): ray::train() (pid=2182, ip=172.16.4.110)
  File "/models/k0050048751/MindSpeed-RL-master/cli/train_grpo.py", line 146, in train
    train_ds, _, _ = build_train_valid_test_datasets(
  File "/models/k0050048751/MindSpeed-RL-master/mindspeed_rl/datasets/build_dataset.py", line 70, in build_train_valid_test_datasets
    all_train_datasets, all_valid_datasets, all_test_datasets = _build_train_valid_test_datasets(
  File "/models/k0050048751/MindSpeed-RL-master/mindspeed_rl/datasets/build_dataset.py", line 116, in _build_train_valid_test_datasets
    packed_indexed_dataset = get_packed_indexed_dataset(data_prefix=data_prefix,
  File "/models/k0050048751/MindSpeed-RL-master/mindspeed_rl/datasets/indexed_dataset.py", line 39, in get_packed_indexed_dataset
    filter_mask = packed_dataset['input_ids'].get_filter_mask()
KeyError: 'input_ids'
/usr/local/python3.10.16/lib/python3.10/tempfile.py:869: ResourceWarning: Implicitly cleaning up <TemporaryDirectory '/tmp/tmpv40n0lke'>
  _warnings.warn(warn_message, ResourceWarning)




import os
import shutil
import struct
import glob
import re
from enum import Enum
from functools import lru_cache
from abc import ABC, abstractmethod
from itertools import accumulate
from types import TracebackType
from typing import Optional, Tuple, Type, Union, List

import torch
import numpy

_INDEX_HEADER = b"MMIDIDX\x00\x00"
'''

def get_packed_indexed_dataset(data_prefix: str, filter_length: Optional[int] = None):
    index_dataset_name = f"{data_prefix}_packed_*_document*"
    names = glob.glob(index_dataset_name)
    template = f"{data_prefix}_packed_(.*)_document(.*)"
    all_field = set()
    for name in names:
        fields = re.match(template, name)
        all_field.add(fields.group(1))
    packed_dataset = dict()

    for field in all_field:
        # We only do filter for input_ids when filter_length is specified
        max_len = filter_length if filter_length and field == 'input_ids' else None
        packed_dataset[field] = IndexedDataset(f"{data_prefix}_packed_{field}_document", max_len=max_len)

    if filter_length:
        filter_mask = packed_dataset['input_ids'].get_filter_mask()
        for field in packed_dataset:
            packed_dataset[field].do_filter(filter_mask)

    combine_dataset = CombinedDataset(packed_dataset)
    return combine_dataset



