# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
from collections.abc import Callable
from concurrent.futures import Future
from functools import cached_property
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.executor import abstract as executor_abstract_module
from vllm.v1.executor import multiproc_executor as multiproc_executor_module
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.executor.ray_executor import RayDistributedExecutor
from vllm.v1.executor.uniproc_executor import (
    ExecutorWithExternalLauncher,
    UniProcExecutor,
)


class Mock: ...


def test_supports_async_scheduling_base_executor():
    assert Executor.supports_async_scheduling() is False


def test_supports_async_scheduling_uniproc_executor():
    assert UniProcExecutor.supports_async_scheduling() is True


def test_supports_async_scheduling_executor_with_external_launcher():
    # ExecutorWithExternalLauncher inherits from UniProcExecutor and does not
    # override supports_async_scheduling, so it should return True.
    assert ExecutorWithExternalLauncher.supports_async_scheduling() is True


def test_supports_async_scheduling_multiproc_executor():
    assert MultiprocExecutor.supports_async_scheduling() is True


class _RayDistributedExecutorForMaxConcurrentBatches(RayDistributedExecutor):
    def __del__(self):
        pass


def _make_executor(
    executor_cls: type[Executor],
    *,
    async_scheduling: bool,
    pipeline_parallel_size: int,
    speculative_method: str | None = None,
) -> Executor:
    executor = object.__new__(executor_cls)
    executor.parallel_config = SimpleNamespace(
        pipeline_parallel_size=pipeline_parallel_size
    )
    executor.scheduler_config = SimpleNamespace(async_scheduling=async_scheduling)
    speculative_config = (
        SimpleNamespace(method=speculative_method)
        if speculative_method is not None
        else None
    )
    executor.vllm_config = SimpleNamespace(speculative_config=speculative_config)
    return executor


def _clear_max_concurrent_batches_cache(executor: Executor) -> None:
    descriptor = type(executor).__dict__.get("max_concurrent_batches")
    if isinstance(descriptor, cached_property):
        executor.__dict__.pop("max_concurrent_batches", None)


@pytest.mark.parametrize(
    "executor_cls", [MultiprocExecutor, _RayDistributedExecutorForMaxConcurrentBatches]
)
def test_sm12x_mtp_async_scheduling_uses_single_batch_queue(
    monkeypatch: pytest.MonkeyPatch,
    executor_cls: type[Executor],
):
    for module in (executor_abstract_module, multiproc_executor_module):
        monkeypatch.setattr(module.current_platform, "is_cuda", lambda: True)
        monkeypatch.setattr(
            module.current_platform,
            "is_device_capability_family",
            lambda capability: capability == 120,
        )

    executor = _make_executor(
        executor_cls,
        async_scheduling=True,
        pipeline_parallel_size=1,
        speculative_method="mtp",
    )
    _clear_max_concurrent_batches_cache(executor)

    assert executor.max_concurrent_batches == 1


@pytest.mark.parametrize(
    "executor_cls", [MultiprocExecutor, _RayDistributedExecutorForMaxConcurrentBatches]
)
def test_non_sm12x_mtp_async_scheduling_keeps_batch_queue(
    monkeypatch: pytest.MonkeyPatch,
    executor_cls: type[Executor],
):
    monkeypatch.setattr(
        executor_abstract_module.current_platform, "is_cuda", lambda: True
    )
    monkeypatch.setattr(
        executor_abstract_module.current_platform,
        "is_device_capability_family",
        lambda capability: False,
    )
    executor = _make_executor(
        executor_cls,
        async_scheduling=True,
        pipeline_parallel_size=1,
        speculative_method="mtp",
    )
    _clear_max_concurrent_batches_cache(executor)

    assert executor.max_concurrent_batches == 2


@pytest.mark.parametrize(
    "executor_cls", [MultiprocExecutor, _RayDistributedExecutorForMaxConcurrentBatches]
)
def test_non_mtp_async_scheduling_keeps_batch_queue(
    executor_cls: type[Executor],
):
    executor = _make_executor(
        executor_cls,
        async_scheduling=True,
        pipeline_parallel_size=1,
        speculative_method="eagle",
    )
    _clear_max_concurrent_batches_cache(executor)

    assert executor.max_concurrent_batches == 2


@pytest.mark.parametrize(
    "executor_cls", [MultiprocExecutor, _RayDistributedExecutorForMaxConcurrentBatches]
)
def test_pipeline_parallelism_sets_batch_queue_to_pp_size(
    executor_cls: type[Executor],
):
    executor = _make_executor(
        executor_cls,
        async_scheduling=True,
        pipeline_parallel_size=4,
        speculative_method="mtp",
    )
    _clear_max_concurrent_batches_cache(executor)

    assert executor.max_concurrent_batches == 4


class CustomMultiprocExecutor(MultiprocExecutor):
    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: KVOutputAggregator = None,
    ) -> Any | list[Any] | Future[Any | list[Any]]:
        # Drop marker to show that this was run
        with open(".marker", "w"):
            ...
        return super().collective_rpc(
            method,
            timeout,
            args,
            kwargs,
            non_block,
            unique_reply_rank,
            kv_output_aggregator,
        )


CustomMultiprocExecutorAsync = CustomMultiprocExecutor
MODEL = "Qwen/Qwen3-0.6B"


def test_custom_executor_type_checking():
    with pytest.raises(ValueError):
        engine_args = EngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=Mock,
        )
        LLMEngine.from_engine_args(engine_args)
    with pytest.raises(ValueError):
        engine_args = AsyncEngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=Mock,
        )
        AsyncLLM.from_engine_args(engine_args)


@pytest.mark.parametrize(
    "distributed_executor_backend",
    [
        CustomMultiprocExecutor,
        "tests.v1.executor.test_executor.CustomMultiprocExecutor",
    ],
)
def test_custom_executor(distributed_executor_backend, tmp_path):
    cwd = os.path.abspath(".")
    os.chdir(tmp_path)
    try:
        assert not os.path.exists(".marker")

        engine_args = EngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=distributed_executor_backend,
            enforce_eager=True,  # reduce test time
        )
        engine = LLMEngine.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        engine.add_request("0", "foo", sampling_params)
        engine.step()

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize(
    "distributed_executor_backend",
    [
        CustomMultiprocExecutorAsync,
        "tests.v1.executor.test_executor.CustomMultiprocExecutorAsync",
    ],
)
def test_custom_executor_async(distributed_executor_backend, tmp_path):
    cwd = os.path.abspath(".")
    os.chdir(tmp_path)
    try:
        assert not os.path.exists(".marker")

        engine_args = AsyncEngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=distributed_executor_backend,
            enforce_eager=True,  # reduce test time
        )
        engine = AsyncLLM.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        async def t():
            stream = engine.generate(
                request_id="0", prompt="foo", sampling_params=sampling_params
            )
            async for x in stream:
                ...

        asyncio.run(t())

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)
