Traceback for thread 32466 (ray::Integrated) [] (most recent call last):
    (Python) File "/usr/local/python3.10.16/lib/python3.10/threading.py", line 973, in _bootstrap
        self._bootstrap_inner()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
        self.run()
    (Python) File "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/repository_manager/utils/multiprocess_util.py", line 91, in run
        key, func, args, kwargs = self.task_q.get(timeout=TIMEOUT)
    (Python) File "<string>", line 2, in get
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/managers.py", line 818, in _callmethod
        kind, result = conn.recv()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 250, in recv
        buf = self._recv_bytes()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 414, in _recv_bytes
        buf = self._recv(4)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 379, in _recv
        chunk = read(handle, remaining)

Traceback for thread 31444 (ray::Integrated) [] (most recent call last):
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/workers/default_worker.py", line 297, in <module>
        worker.main_loop()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/worker.py", line 935, in main_loop
        self.core_worker.run_task_loop()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/function_manager.py", line 696, in actor_method_executor
        return method(__ray_actor, *args, **kwargs)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/util/tracing/tracing_helper.py", line 463, in _resume_span
        return method(self, *_args, **_kwargs)
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/utils/utils.py", line 553, in wrapper
        result = func(*args, **kwargs)
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/workers/actor_hybrid_worker.py", line 306, in generate_sequences
        while self.all_consumed(experience_consumer_stage, sorted_indexes, use_vllm=True) > 0:

[root@train-qwen3-235btest4n2-v11-task-2 /]# pystack remote 31444
Traceback for thread 32466 (ray::Integrated) [] (most recent call last):
    (Python) File "/usr/local/python3.10.16/lib/python3.10/threading.py", line 973, in _bootstrap
        self._bootstrap_inner()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
        self.run()
    (Python) File "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/repository_manager/utils/multiprocess_util.py", line 91, in run
        key, func, args, kwargs = self.task_q.get(timeout=TIMEOUT)
    (Python) File "<string>", line 2, in get
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/managers.py", line 818, in _callmethod
        kind, result = conn.recv()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 250, in recv
        buf = self._recv_bytes()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 414, in _recv_bytes
        buf = self._recv(4)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 379, in _recv
        chunk = read(handle, remaining)

Traceback for thread 31444 (ray::Integrated) [] (most recent call last):
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/workers/default_worker.py", line 297, in <module>
        worker.main_loop()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/worker.py", line 935, in main_loop
        self.core_worker.run_task_loop()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/function_manager.py", line 696, in actor_method_executor
        return method(__ray_actor, *args, **kwargs)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/util/tracing/tracing_helper.py", line 463, in _resume_span
        return method(self, *_args, **_kwargs)
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/utils/utils.py", line 553, in wrapper
        result = func(*args, **kwargs)
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/workers/actor_hybrid_worker.py", line 306, in generate_sequences
        while self.all_consumed(experience_consumer_stage, sorted_indexes, use_vllm=True) > 0:

[root@train-qwen3-235btest4n2-v11-task-2 /]# pystack remote 31444
Traceback for thread 32466 (ray::Integrated) [] (most recent call last):
    (Python) File "/usr/local/python3.10.16/lib/python3.10/threading.py", line 973, in _bootstrap
        self._bootstrap_inner()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
        self.run()
    (Python) File "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/repository_manager/utils/multiprocess_util.py", line 91, in run
        key, func, args, kwargs = self.task_q.get(timeout=TIMEOUT)
    (Python) File "<string>", line 2, in get
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/managers.py", line 818, in _callmethod
        kind, result = conn.recv()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 250, in recv
        buf = self._recv_bytes()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 414, in _recv_bytes
        buf = self._recv(4)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 379, in _recv
        chunk = read(handle, remaining)

Traceback for thread 31444 (ray::Integrated) [] (most recent call last):
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/workers/default_worker.py", line 297, in <module>
        worker.main_loop()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/worker.py", line 935, in main_loop
        self.core_worker.run_task_loop()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/function_manager.py", line 696, in actor_method_executor
        return method(__ray_actor, *args, **kwargs)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/util/tracing/tracing_helper.py", line 463, in _resume_span
        return method(self, *_args, **_kwargs)
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/utils/utils.py", line 553, in wrapper
        result = func(*args, **kwargs)
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/workers/actor_hybrid_worker.py", line 306, in generate_sequences
        while self.all_consumed(experience_consumer_stage, sorted_indexes, use_vllm=True) > 0:

[root@train-qwen3-235btest4n2-v11-task-2 /]# pystack remote 31444
Traceback for thread 32466 (ray::Integrated) [] (most recent call last):
    (Python) File "/usr/local/python3.10.16/lib/python3.10/threading.py", line 973, in _bootstrap
        self._bootstrap_inner()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
        self.run()
    (Python) File "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/repository_manager/utils/multiprocess_util.py", line 91, in run
        key, func, args, kwargs = self.task_q.get(timeout=TIMEOUT)
    (Python) File "<string>", line 2, in get
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/managers.py", line 818, in _callmethod
        kind, result = conn.recv()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 250, in recv
        buf = self._recv_bytes()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 414, in _recv_bytes
        buf = self._recv(4)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 379, in _recv
        chunk = read(handle, remaining)

Traceback for thread 31444 (ray::Integrated) [] (most recent call last):
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/workers/default_worker.py", line 297, in <module>
        worker.main_loop()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/worker.py", line 935, in main_loop
        self.core_worker.run_task_loop()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/function_manager.py", line 696, in actor_method_executor
        return method(__ray_actor, *args, **kwargs)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/util/tracing/tracing_helper.py", line 463, in _resume_span
        return method(self, *_args, **_kwargs)
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/utils/utils.py", line 553, in wrapper
        result = func(*args, **kwargs)
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/workers/actor_hybrid_worker.py", line 307, in generate_sequences
        batch_data, index = self.dispatch_transfer_dock_data(
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/util/tracing/tracing_helper.py", line 463, in _resume_span
        return method(self, *_args, **_kwargs)
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/utils/utils.py", line 553, in wrapper
        result = func(*args, **kwargs)
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/workers/base_worker.py", line 313, in dispatch_transfer_dock_data
        torch.distributed.broadcast(
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/torch/distributed/c10d_logger.py", line 83, in wrapper
        return func(*args, **kwargs)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py", line 2421, in broadcast
        work = group.broadcast([tensor], opts)

[root@train-qwen3-235btest4n2-v11-task-2 /]# pystack remote 31444
Traceback for thread 32466 (ray::Integrated) [] (most recent call last):
    (Python) File "/usr/local/python3.10.16/lib/python3.10/threading.py", line 973, in _bootstrap
        self._bootstrap_inner()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
        self.run()
    (Python) File "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/repository_manager/utils/multiprocess_util.py", line 91, in run
        key, func, args, kwargs = self.task_q.get(timeout=TIMEOUT)
    (Python) File "<string>", line 2, in get
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/managers.py", line 818, in _callmethod
        kind, result = conn.recv()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 250, in recv
        buf = self._recv_bytes()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 414, in _recv_bytes
        buf = self._recv(4)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 379, in _recv
        chunk = read(handle, remaining)

Traceback for thread 31444 (ray::Integrated) [] (most recent call last):
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/workers/default_worker.py", line 297, in <module>
        worker.main_loop()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/worker.py", line 935, in main_loop
        self.core_worker.run_task_loop()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/function_manager.py", line 696, in actor_method_executor
        return method(__ray_actor, *args, **kwargs)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/util/tracing/tracing_helper.py", line 463, in _resume_span
        return method(self, *_args, **_kwargs)
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/utils/utils.py", line 553, in wrapper
        result = func(*args, **kwargs)
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/workers/actor_hybrid_worker.py", line 306, in generate_sequences
        while self.all_consumed(experience_consumer_stage, sorted_indexes, use_vllm=True) > 0:

[root@train-qwen3-235btest4n2-v11-task-2 /]# pystack remote 31444
Traceback for thread 32466 (ray::Integrated) [] (most recent call last):
    (Python) File "/usr/local/python3.10.16/lib/python3.10/threading.py", line 973, in _bootstrap
        self._bootstrap_inner()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
        self.run()
    (Python) File "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/repository_manager/utils/multiprocess_util.py", line 91, in run
        key, func, args, kwargs = self.task_q.get(timeout=TIMEOUT)
    (Python) File "<string>", line 2, in get
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/managers.py", line 818, in _callmethod
        kind, result = conn.recv()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 250, in recv
        buf = self._recv_bytes()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 414, in _recv_bytes
        buf = self._recv(4)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 379, in _recv
        chunk = read(handle, remaining)

Traceback for thread 31444 (ray::Integrated) [] (most recent call last):
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/workers/default_worker.py", line 297, in <module>
        worker.main_loop()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/worker.py", line 935, in main_loop
        self.core_worker.run_task_loop()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/function_manager.py", line 696, in actor_method_executor
        return method(__ray_actor, *args, **kwargs)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/util/tracing/tracing_helper.py", line 463, in _resume_span
        return method(self, *_args, **_kwargs)
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/utils/utils.py", line 553, in wrapper
        result = func(*args, **kwargs)
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/workers/actor_hybrid_worker.py", line 306, in generate_sequences
        while self.all_consumed(experience_consumer_stage, sorted_indexes, use_vllm=True) > 0:

[root@train-qwen3-235btest4n2-v11-task-2 /]# pystack remote 31444
Traceback for thread 32466 (ray::Integrated) [] (most recent call last):
    (Python) File "/usr/local/python3.10.16/lib/python3.10/threading.py", line 973, in _bootstrap
        self._bootstrap_inner()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
        self.run()
    (Python) File "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/repository_manager/utils/multiprocess_util.py", line 91, in run
        key, func, args, kwargs = self.task_q.get(timeout=TIMEOUT)
    (Python) File "<string>", line 2, in get
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/managers.py", line 818, in _callmethod
        kind, result = conn.recv()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 250, in recv
        buf = self._recv_bytes()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 414, in _recv_bytes
        buf = self._recv(4)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 379, in _recv
        chunk = read(handle, remaining)

Traceback for thread 31444 (ray::Integrated) [] (most recent call last):
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/workers/default_worker.py", line 297, in <module>
        worker.main_loop()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/worker.py", line 935, in main_loop
        self.core_worker.run_task_loop()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/function_manager.py", line 696, in actor_method_executor
        return method(__ray_actor, *args, **kwargs)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/util/tracing/tracing_helper.py", line 463, in _resume_span
        return method(self, *_args, **_kwargs)
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/utils/utils.py", line 553, in wrapper
        result = func(*args, **kwargs)
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/workers/actor_hybrid_worker.py", line 306, in generate_sequences
        while self.all_consumed(experience_consumer_stage, sorted_indexes, use_vllm=True) > 0:

[root@train-qwen3-235btest4n2-v11-task-2 /]# pystack remote 31444
Traceback for thread 32466 (ray::Integrated) [] (most recent call last):
    (Python) File "/usr/local/python3.10.16/lib/python3.10/threading.py", line 973, in _bootstrap
        self._bootstrap_inner()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
        self.run()
    (Python) File "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/common/repository_manager/utils/multiprocess_util.py", line 91, in run
        key, func, args, kwargs = self.task_q.get(timeout=TIMEOUT)
    (Python) File "<string>", line 2, in get
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/managers.py", line 818, in _callmethod
        kind, result = conn.recv()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 250, in recv
        buf = self._recv_bytes()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 414, in _recv_bytes
        buf = self._recv(4)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/multiprocessing/connection.py", line 379, in _recv
        chunk = read(handle, remaining)

Traceback for thread 31444 (ray::Integrated) [Has the GIL] (most recent call last):
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/workers/default_worker.py", line 297, in <module>
        worker.main_loop()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/worker.py", line 935, in main_loop
        self.core_worker.run_task_loop()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/function_manager.py", line 696, in actor_method_executor
        return method(__ray_actor, *args, **kwargs)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/util/tracing/tracing_helper.py", line 463, in _resume_span
        return method(self, *_args, **_kwargs)
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/utils/utils.py", line 553, in wrapper
        result = func(*args, **kwargs)
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/workers/actor_hybrid_worker.py", line 307, in generate_sequences
        batch_data, index = self.dispatch_transfer_dock_data(
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/util/tracing/tracing_helper.py", line 463, in _resume_span
        return method(self, *_args, **_kwargs)
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/utils/utils.py", line 553, in wrapper
        result = func(*args, **kwargs)
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/workers/base_worker.py", line 271, in dispatch_transfer_dock_data
        if is_multimodal():
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/utils/utils.py", line 583, in is_multimodal
        return eval(os.getenv("IS_MULTIMODAL", "False"))

