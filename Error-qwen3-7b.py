I thought the skyladder only re-arrange the tokens, but the result seems like it also change the data? You can print out the output of skyladder loader and check. If you think the skyladder modify the data too much, making the process stuck and hard to handle, we can discuss with Zeng Li tomorrow and see if we need to skip it and try next techniqueI thought the skyladder only re-arrange the tokens, but the result seems like it also change the data? You can print out the output of skyladder loader and check. If you think the skyladder modify the data too much, making the process stuck and hard to handle, we can discuss with Zeng Li tomorrow and see if we need to skip it and try next technique                                                        
def context_decorator(ctx, func):
    """
    Like contextlib.ContextDecorator.

    But with the following differences:
    1. Is done by wrapping, rather than inheritance, so it works with context
       managers that are implemented from C and thus cannot easily inherit from
       Python classes
    2. Wraps generators in the intuitive way (c.f. https://bugs.python.org/issue37743)
    3. Errors out if you try to wrap a class, because it is ambiguous whether
       or not you intended to wrap only the constructor

    The input argument can either be a context manager (in which case it must
    be a multi-shot context manager that can be directly invoked multiple times)
    or a callable that produces a context manager.
    """
    assert not (callable(ctx) and hasattr(ctx, '__enter__')), (
        f"Passed in {ctx} is both callable and also a valid context manager "
        "(has __enter__), making it ambiguous which interface to use.  If you "
        "intended to pass a context manager factory, rewrite your call as "
        "context_decorator(lambda: ctx()); if you intended to pass a context "
        "manager directly, rewrite your call as context_decorator(lambda: ctx)"
    )

    if not callable(ctx):
        def ctx_factory():
            return ctx
    else:
        ctx_factory = ctx

    if inspect.isclass(func):
        raise RuntimeError(
            "Cannot decorate classes; it is ambiguous whether or not only the "
            "constructor or all methods should have the context manager applied; "
            "additionally, decorating a class at definition-site will prevent "
            "use of the identifier as a conventional type.  "
            "To specify which methods to decorate, decorate each of them "
            "individually."
        )

    if inspect.isgeneratorfunction(func):
        return _wrap_generator(ctx_factory, func)

    @functools.wraps(func)
    def decorate_context(*args, **kwargs):
        with ctx_factory():
            return func(*args, **kwargs)

    return decorate_context
==============================================================================================

(pid=21628)     *************************************************************************************************************
(pid=21628)     
(pid=21628)   warnings.warn(msg, ImportWarning)
(pid=21628) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/contrib/transfer_to_npu.py:247: RuntimeWarning: torch.jit.script and torch.jit.script_method will be disabled by transfer_to_npu, which currently does not support them, if you need to enable them, please do not use transfer_to_npu.
(pid=21628)   warnings.warn(msg, RuntimeWarning)
(pid=21628) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/dynamo/torchair/__init__.py:8: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
(pid=21628)   import pkg_resources
(pid=21628) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/dynamo/torchair/__init__.py:8: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
(pid=21628)   import pkg_resources
(pid=21628) Initialized ProcessPoolExecutor with 16 workers
(pid=21628) Initialized ProcessPoolExecutor with 16 workers
(train pid=16615) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/contrib/transfer_to_npu.py:153: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:206.)
(train pid=16615)   return fn(*args, **kwargs)
(train pid=16615) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/contrib/transfer_to_npu.py:153: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:206.)
(train pid=16615)   return fn(*args, **kwargs)
(train pid=16615) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/contrib/transfer_to_npu.py:153: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:206.)
(train pid=16615)   return fn(*args, **kwargs)
(train pid=16615) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/contrib/transfer_to_npu.py:153: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:206.)
(train pid=16615)   return fn(*args, **kwargs)
(GRPOTransferDock pid=21628) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/utils/storage.py:38: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
(GRPOTransferDock pid=21628)   if self.device.type != 'cpu':
(GRPOTransferDock pid=21628) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/utils/storage.py:38: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
(GRPOTransferDock pid=21628)   if self.device.type != 'cpu':
(GRPOTransferDock pid=21628) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/contrib/transfer_to_npu.py:153: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
(GRPOTransferDock pid=21628) /usr/local/python3.10.16/lib/python3.10/site-packages/torch_npu/contrib/transfer_to_npu.py:153: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
(IntegratedWorker pid=17236) /usr/local/python3.10.16/lib/python3.10/site-packages/torch/utils/_contextlib.py:116: DeprecationWarning: The keyword arguments {'prompt_token_ids'} are deprecated and will be removed in a future update. Please use the 'prompts' parameter instead.
(IntegratedWorker pid=17236)   return func(*args, **kwargs)
(IntegratedWorker pid=17236) /usr/local/python3.10.16/lib/python3.10/site-packages/torch/utils/_contextlib.py:116: DeprecationWarning: The keyword arguments {'prompt_token_ids'} are deprecated and will be removed in a future update. Please use the 'prompts' parameter instead.
(IntegratedWorker pid=17236)   return func(*args, **kwargs)

====================================================================================================
It is stuck forever at "return func(*args, **kwargs)" and its getting annoying. This is the 15th consecutive time I am dealing with this. What the fuck is the problem and how the fuck do we deal with this crap?
