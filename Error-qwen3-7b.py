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
ray.exceptions.RayTaskError(ValueError): ray::train() (pid=34140, ip=172.16.4.104)
  File "/models/k50048751/MindSpeed-RL-master/cli/train_grpo.py", line 159, in train
    actor_worker.wait_all_ref_objs_run_over()
  File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/workers/scheduler/launcher.py", line 268, in wait_all_ref_objs_run_over
    ray.get(self.temp_actor_ref_objs)
ray.exceptions.RayTaskError(ValueError): ray::IntegratedWorker.initialize() (pid=34769, ip=172.16.4.104, actor_id=7ff4c23b1ff0a604e4da7e890f000000, repr=<mindspeed_rl.workers.integrated_worker.IntegratedWorker object at 0xffd06d075d20>)
  File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/workers/integrated_worker.py", line 84, in initialize
    ActorHybridWorkerBase.initialize(self)
  File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/workers/actor_hybrid_worker.py", line 98, in initialize
    self.inference_model = self._build_rollout()
  File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/workers/actor_hybrid_worker.py", line 460, in _build_rollout
    rollout = VLLMInferEngine(
  File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/models/rollout/vllm_engine.py", line 157, in __init__
    initialize_parallel_state(
  File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/models/rollout/vllm_adapter/vllm_parallel_state.py", line 89, in initialize_parallel_state
    initialize_model_parallel_for_vllm(
  File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/models/rollout/vllm_adapter/vllm_parallel_state.py", line 201, in initialize_model_parallel_for_vllm
    tp_group_ranks = get_tp_group_ranks()
  File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/models/rollout/vllm_adapter/vllm_parallel_state.py", line 194, in get_tp_group_ranks
    tp_group_ranks = get_split_tp_group_ranks()
  File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/models/rollout/vllm_adapter/vllm_parallel_state.py", line 154, in get_split_tp_group_ranks
    raise ValueError(
ValueError: Can't split train tp size 4 to infer tp size 8 with train dp size 1.
/usr/local/python3.10.16/lib/python3.10/tempfile.py:869: ResourceWarning: Implicitly cleaning up <TemporaryDirectory '/tmp/tmp88oooon8'>
  _warnings.warn(warn_message, ResourceWarning)
