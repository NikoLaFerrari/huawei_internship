Error executing job with overrides: []
Traceback (most recent call last):
  File "/models/k50048751/MindSpeed-RL-master/cli/train_grpo.py", line 574, in <module>
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
  File "/models/k50048751/MindSpeed-RL-master/cli/train_grpo.py", line 570, in main
    ray.get(train.remote(config))
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/worker.py", line 2772, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/worker.py", line 919, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(KeyError): ray::train() (pid=11155, ip=172.16.4.104)
  File "/models/k50048751/MindSpeed-RL-master/cli/train_grpo.py", line 146, in train
    train_ds, _, _ = build_train_valid_test_datasets(
  File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/datasets/build_dataset.py", line 70, in build_train_valid_test_datasets
    all_train_datasets, all_valid_datasets, all_test_datasets = _build_train_valid_test_datasets(
  File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/datasets/build_dataset.py", line 116, in _build_train_valid_test_datasets
    packed_indexed_dataset = get_packed_indexed_dataset(data_prefix=data_prefix,
  File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/datasets/indexed_dataset.py", line 39, in get_packed_indexed_dataset
    filter_mask = packed_dataset['input_ids'].get_filter_mask()
KeyError: 'input_ids'
/usr/local/python3.10.16/lib/python3.10/tempfile.py:869: ResourceWarning: Implicitly cleaning up <TemporaryDirectory '/tmp/tmp3ea0fiba'>
  _warnings.warn(warn_message, ResourceWarning)
(train pid=11155) 2025-07-10 09:48:11,385 - INFO - [2025-07-10 01:48:11] INFO  > datasets target sizes (minimum size):
(train pid=11155) 2025-07-10 09:48:11,385 - INFO - [2025-07-10 01:48:11] INFO     train:      8000
(train pid=11155) 2025-07-10 09:48:11,385 - INFO - [2025-07-10 01:48:11] INFO     validation: 0
(train pid=11155) 2025-07-10 09:48:11,385 - INFO - [2025-07-10 01:48:11] INFO     test:       0
(train pid=11155) 2025-07-10 09:48:11,385 - INFO - [2025-07-10 01:48:11] INFO  > datasets target sizes (minimum size):
(train pid=11155) 2025-07-10 09:48:11,385 - INFO - [2025-07-10 01:48:11] INFO     train:      8000
(train pid=11155) 2025-07-10 09:48:11,385 - INFO - [2025-07-10 01:48:11] INFO     validation: 0
(train pid=11155) 2025-07-10 09:48:11,385 - INFO - [2025-07-10 01:48:11] INFO     test:       0
