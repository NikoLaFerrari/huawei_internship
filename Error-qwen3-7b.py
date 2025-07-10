[root@train-qwen3-235btest4n2-v11-task-2 /]# pystack remote 63627
Traceback for thread 65014 (ray::Integrated) [] (most recent call last):
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

Traceback for thread 63627 (ray::Integrated) [] (most recent call last):
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
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/workers/base_worker.py", line 299, in dispatch_transfer_dock_data
        batch_data, index = ray.get(
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
        return fn(*args, **kwargs)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
        return func(*args, **kwargs)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/worker.py", line 2772, in get
        values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/worker.py", line 893, in get_objects
        ] = self.core_worker.get_objects(

[root@train-qwen3-235btest4n2-v11-task-2 /]# pystack remote 63627
Traceback for thread 65014 (ray::Integrated) [] (most recent call last):
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

Traceback for thread 63627 (ray::Integrated) [] (most recent call last):
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
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/workers/base_worker.py", line 299, in dispatch_transfer_dock_data
        batch_data, index = ray.get(
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
        return fn(*args, **kwargs)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
        return func(*args, **kwargs)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/worker.py", line 2772, in get
        values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/ray/_private/worker.py", line 893, in get_objects
        ] = self.core_worker.get_objects(

[root@train-qwen3-235btest4n2-v11-task-2 /]# pystack remote 63627
Traceback for thread 65014 (ray::Integrated) [] (most recent call last):
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

Traceback for thread 63627 (ray::Integrated) [] (most recent call last):
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
    (Python) File "/models/k50048751/MindSpeed-RL-master/mindspeed_rl/workers/base_worker.py", line 309, in dispatch_transfer_dock_data
        index = torch.tensor(index + ([-1] * (experience_count - len(index)))).cuda()
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/torch/utils/backend_registration.py", line 148, in wrap_tensor_to
        return self.to(device=torch.device(f'{custom_backend_name}:{device_idx}'), non_blocking=non_blocking, **kwargs)
    (Python) File "/usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/contrib/transfer_to_npu.py", line 153, in decorated
        return fn(*args, **kwargs)

----------------------------------------------------------------------------------------------------------------------------------------------------

(pid=2737) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/contrib/transfer_to_npu.py:292: ImportWarning: 
(pid=2737)     *************************************************************************************************************
(pid=2737)     The torch.Tensor.cuda and torch.nn.Module.cuda are replaced with torch.Tensor.npu and torch.nn.Module.npu now..
(pid=2737)     The torch.cuda.DoubleTensor is replaced with torch.npu.FloatTensor cause the double type is not supported now..
(pid=2737)     The backend in torch.distributed.init_process_group set to hccl now..
(pid=2737)     The torch.cuda.* and torch.cuda.amp.* are replaced with torch.npu.* and torch.npu.amp.* now..
(pid=2737)     The device parameters have been replaced with npu in the function below:
(pid=2737)     torch.logspace, torch.randint, torch.hann_window, torch.rand, torch.full_like, torch.ones_like, torch.rand_like, torch.randperm, torch.arange, torch.frombuffer, torch.normal, torch._empty_per_channel_affine_quantized, torch.empty_strided, torch.empty_like, torch.scalar_tensor, torch.tril_indices, torch.bartlett_window, torch.ones, torch.sparse_coo_tensor, torch.randn, torch.kaiser_window, torch.tensor, torch.triu_indices, torch.as_tensor, torch.zeros, torch.randint_like, torch.full, torch.eye, torch._sparse_csr_tensor_unsafe, torch.empty, torch._sparse_coo_tensor_unsafe, torch.blackman_window, torch.zeros_like, torch.range, torch.sparse_csr_tensor, torch.randn_like, torch.from_file, torch._cudnn_init_dropout_state, torch._empty_affine_quantized, torch.linspace, torch.hamming_window, torch.empty_quantized, torch._pin_memory, torch.autocast, torch.load, torch.Generator, torch.set_default_device, torch.Tensor.new_empty, torch.Tensor.new_empty_strided, torch.Tensor.new_full, torch.Tensor.new_ones, torch.Tensor.new_tensor, torch.Tensor.new_zeros, torch.Tensor.to, torch.Tensor.pin_memory, torch.nn.Module.to, torch.nn.Module.to_empty
(pid=2737)     *************************************************************************************************************
(pid=2737)     
(pid=2737)   warnings.warn(msg, ImportWarning)
(pid=2737) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/contrib/transfer_to_npu.py:247: RuntimeWarning: torch.jit.script and torch.jit.script_method will be disabled by transfer_to_npu, which currently does not support them, if you need to enable them, please do not use transfer_to_npu.
(pid=2737)   warnings.warn(msg, RuntimeWarning)
(pid=2737) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/contrib/transfer_to_npu.py:292: ImportWarning: 
(pid=2737)     *************************************************************************************************************
(pid=2737)     The torch.Tensor.cuda and torch.nn.Module.cuda are replaced with torch.Tensor.npu and torch.nn.Module.npu now..
(pid=2737)     The torch.cuda.DoubleTensor is replaced with torch.npu.FloatTensor cause the double type is not supported now..
(pid=2737)     The backend in torch.distributed.init_process_group set to hccl now..
(pid=2737)     The torch.cuda.* and torch.cuda.amp.* are replaced with torch.npu.* and torch.npu.amp.* now..
(pid=2737)     The device parameters have been replaced with npu in the function below:
(pid=2737)     torch.logspace, torch.randint, torch.hann_window, torch.rand, torch.full_like, torch.ones_like, torch.rand_like, torch.randperm, torch.arange, torch.frombuffer, torch.normal, torch._empty_per_channel_affine_quantized, torch.empty_strided, torch.empty_like, torch.scalar_tensor, torch.tril_indices, torch.bartlett_window, torch.ones, torch.sparse_coo_tensor, torch.randn, torch.kaiser_window, torch.tensor, torch.triu_indices, torch.as_tensor, torch.zeros, torch.randint_like, torch.full, torch.eye, torch._sparse_csr_tensor_unsafe, torch.empty, torch._sparse_coo_tensor_unsafe, torch.blackman_window, torch.zeros_like, torch.range, torch.sparse_csr_tensor, torch.randn_like, torch.from_file, torch._cudnn_init_dropout_state, torch._empty_affine_quantized, torch.linspace, torch.hamming_window, torch.empty_quantized, torch._pin_memory, torch.autocast, torch.load, torch.Generator, torch.set_default_device, torch.Tensor.new_empty, torch.Tensor.new_empty_strided, torch.Tensor.new_full, torch.Tensor.new_ones, torch.Tensor.new_tensor, torch.Tensor.new_zeros, torch.Tensor.to, torch.Tensor.pin_memory, torch.nn.Module.to, torch.nn.Module.to_empty
(pid=2737)     *************************************************************************************************************
(pid=2737)     
(pid=2737)   warnings.warn(msg, ImportWarning)
(pid=2737) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/contrib/transfer_to_npu.py:247: RuntimeWarning: torch.jit.script and torch.jit.script_method will be disabled by transfer_to_npu, which currently does not support them, if you need to enable them, please do not use transfer_to_npu.
(pid=2737)   warnings.warn(msg, RuntimeWarning)
(pid=2737) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/dynamo/torchair/__init__.py:8: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
(pid=2737)   import pkg_resources
(pid=2737) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/dynamo/torchair/__init__.py:8: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
(pid=2737)   import pkg_resources
(pid=2737) Initialized ProcessPoolExecutor with 16 workers
(pid=2737) Initialized ProcessPoolExecutor with 16 workers
(train pid=62706) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/contrib/transfer_to_npu.py:153: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:206.)
(train pid=62706)   return fn(*args, **kwargs)
(train pid=62706) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/contrib/transfer_to_npu.py:153: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:206.)
(train pid=62706)   return fn(*args, **kwargs)
(train pid=62706) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/contrib/transfer_to_npu.py:153: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:206.)
(train pid=62706)   return fn(*args, **kwargs)
(train pid=62706) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/contrib/transfer_to_npu.py:153: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:206.)
(train pid=62706)   return fn(*args, **kwargs)
(GRPOTransferDock pid=2737) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/utils/storage.py:38: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
(GRPOTransferDock pid=2737)   if self.device.type != 'cpu':
(GRPOTransferDock pid=2737) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/utils/storage.py:38: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
(GRPOTransferDock pid=2737)   if self.device.type != 'cpu':
(GRPOTransferDock pid=2737) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/contrib/transfer_to_npu.py:153: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
(GRPOTransferDock pid=2737) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/contrib/transfer_to_npu.py:153: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
(IntegratedWorker pid=63627) /usr/local/python3.10.16/lib/python3.10/site-packages/torch/utils/_contextlib.py:116: DeprecationWarning: The keyword arguments {'prompt_token_ids'} are deprecated and will be removed in a future update. Please use the 'prompts' parameter instead.
(IntegratedWorker pid=63627)   return func(*args, **kwargs)
(IntegratedWorker pid=63627) /usr/local/python3.10.16/lib/python3.10/site-packages/torch/utils/_contextlib.py:116: DeprecationWarning: The keyword arguments {'prompt_token_ids'} are deprecated and will be removed in a future update. Please use the 'prompts' parameter instead.
(IntegratedWorker pid=63627)   return func(*args, **kwargs)

