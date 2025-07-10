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
ray.exceptions.RayTaskError(KeyError): ray::train() (pid=60911, ip=172.16.4.104)
  File "/models/k50048751/MindSpeed-RL-master/cli/train_grpo.py", line 197, in train
    trainer.fit(data_iters)
  File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/trainer/grpo_trainer_hybrid.py", line 138, in fit
    ray.get(self.transfer_dock.put_prompts_experience.remote(batch, self.dataset_additional_keys))
ray.exceptions.RayTaskError(KeyError): ray::GRPOTransferDock.put_prompts_experience() (pid=445, ip=172.16.4.104, actor_id=82f3a5dba2eb4b618d27ecac14000000, repr=<mindspeed_rl.trainer.utils.transfer_dock.GRPOTransferDock object at 0xffd046e1d9c0>)
  File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/trainer/utils/transfer_dock.py", line 480, in put_prompts_experience
    prompts = batch["prompts"]
KeyError: 'prompts'
/usr/local/python3.10.16/lib/python3.10/tempfile.py:869: ResourceWarning: Implicitly cleaning up <TemporaryDirectory '/tmp/tmp6fkb6y64'>
  _warnings.warn(warn_message, ResourceWarning)
(IntegratedWorker pid=61540) [ERROR] TBE Subprocess[task_distribute] raise error[], main process disappeared! [repeated 120x across cluster]
[ERROR] 2025-07-10-10:59:49 (PID:60711, Device:-1, RankID:-1) ERR99999 UNKNOWN applicaiton exception
[root@train-qwen3-235btest4n2-v11-task-2 MindSpeed-RL-master]# 





