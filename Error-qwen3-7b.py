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
ray.exceptions.RayTaskError(TypeError): ray::train() (pid=43617, ip=172.16.4.104)
  File "/models/k50048751/MindSpeed-RL-master/cli/train_grpo.py", line 163, in train
    data_loader = SkyLadder(
TypeError: SkyLadder.__init__() takes 3 positional arguments but 7 were given
/usr/local/python3.10.16/lib/python3.10/tempfile.py:869: ResourceWarning: Implicitly cleaning up <TemporaryDirectory '/tmp/tmpm_wohb_m'>
  _warnings.warn(warn_message, ResourceWarning)


    data_loader = SkyLadder(
        train_ds, actor_config.global_batch_size, actor_config.num_workers, actor_config.seed,
        actor_config.dataset_additional_keys, actor_config.no_shuffle
    )
    '''
    data_loader = PromptDataLoader(
        train_ds, actor_config.global_batch_size,
        actor_config.num_workers, actor_config.seed, actor_config.dataset_additional_keys,
        actor_config.no_shuffle
    )
    '''






