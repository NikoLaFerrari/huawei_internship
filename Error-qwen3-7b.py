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
ray.exceptions.RayTaskError(RuntimeError): ray::train() (pid=48659, ip=172.16.4.104)
  File "/models/k50048751/MindSpeed-RL-master/cli/train_grpo.py", line 197, in train
    trainer.fit(data_iters)
  File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/trainer/grpo_trainer_hybrid.py", line 137, in fit
    batch = next(data_iters)
  File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/datasets/dataloader.py", line 229, in __iter__
    for batch in super().__iter__():
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/profiler/_add_mstx_patch.py", line 28, in wrapper
    out = func(*args, **kwargs)
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 701, in __next__
    data = self._next_data()
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1465, in _next_data
    return self._process_data(data)
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1491, in _process_data
    data.reraise()
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/torch/_utils.py", line 715, in reraise
    raise exception
RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/torch/utils/data/_utils/worker.py", line 351, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
  File "/usr/local/python3.10.16/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
  File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/datasets/dataloader.py", line 248, in _collate
    "labels": _slice_stack("labels"),
  File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/datasets/dataloader.py", line 243, in _slice_stack
    return torch.stack([torch.as_tensor(r[key])[:ctx] for r in rows])
RuntimeError: stack expects each tensor to be equal size, but got [163] at entry 0 and [187] at entry 3
/usr/local/python3.10.16/lib/python3.10/tempfile.py:869: ResourceWarning: Implicitly cleaning up <TemporaryDirectory '/tmp/tmp4zhzqp8w'>
  _warnings.warn(warn_message, ResourceWarning)
