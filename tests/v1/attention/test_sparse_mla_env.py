# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from vllm.envs import environment_variables
from vllm.v1.attention.backends.mla.sparse_mla_env import (
    is_sparse_mla_attention_dump_enabled,
    is_sparse_mla_reference_attention_enabled,
    sparse_mla_attention_dump_path,
    sparse_mla_reference_cudagraphs_allowed,
    sparse_mla_reference_head_block_size,
    sparse_mla_reference_query_chunk_size,
    sparse_mla_reference_topk_chunk_size,
)

_SPARSE_MLA_ENV_NAMES = (
    "VLLM_TRITON_MLA_SPARSE",
    "VLLM_TRITON_MLA_SPARSE_DUMP",
    "VLLM_TRITON_MLA_SPARSE_DUMP_PATH",
    "VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE",
    "VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE",
    "VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH",
    "VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE",
)


@contextmanager
def _patched_sparse_mla_env(**updates: str) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in _SPARSE_MLA_ENV_NAMES}
    try:
        for name in _SPARSE_MLA_ENV_NAMES:
            os.environ.pop(name, None)
        os.environ.update(updates)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_sparse_mla_reference_env_uses_new_name() -> None:
    with _patched_sparse_mla_env(VLLM_TRITON_MLA_SPARSE="0"):
        assert not is_sparse_mla_reference_attention_enabled(torch.device("cpu"))

    with _patched_sparse_mla_env(VLLM_TRITON_MLA_SPARSE="1"):
        assert is_sparse_mla_reference_attention_enabled(torch.device("cpu"))


def test_sparse_mla_dump_env_uses_new_name() -> None:
    with _patched_sparse_mla_env(VLLM_TRITON_MLA_SPARSE_DUMP="0"):
        assert not is_sparse_mla_attention_dump_enabled()

    with _patched_sparse_mla_env(VLLM_TRITON_MLA_SPARSE_DUMP="1"):
        assert is_sparse_mla_attention_dump_enabled()


def test_sparse_mla_cudagraph_env_defaults_to_allowed() -> None:
    with _patched_sparse_mla_env():
        assert sparse_mla_reference_cudagraphs_allowed()

    with _patched_sparse_mla_env(VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH="0"):
        assert not sparse_mla_reference_cudagraphs_allowed()

    with _patched_sparse_mla_env(VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH="1"):
        assert sparse_mla_reference_cudagraphs_allowed()


def test_sparse_mla_head_block_env_accepts_supported_values() -> None:
    with _patched_sparse_mla_env():
        assert sparse_mla_reference_head_block_size() is None

    with _patched_sparse_mla_env(VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE="1"):
        assert sparse_mla_reference_head_block_size() == 1

    with _patched_sparse_mla_env(VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE="2"):
        assert sparse_mla_reference_head_block_size() == 2

    with _patched_sparse_mla_env(VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE="4"):
        assert sparse_mla_reference_head_block_size() == 4


def test_sparse_mla_head_block_env_ignores_invalid_values() -> None:
    for value in ("0", "3", "invalid"):
        with _patched_sparse_mla_env(VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE=value):
            assert sparse_mla_reference_head_block_size() is None


def test_sparse_mla_head_block_env_is_registered_with_vllm_envs() -> None:
    assert "VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE" in environment_variables

    with _patched_sparse_mla_env(VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE="4"):
        assert environment_variables["VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE"]() == 4


def test_sparse_mla_chunk_env_defaults_invalid_values() -> None:
    with _patched_sparse_mla_env(
        VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE="invalid",
        VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE="-7",
    ):
        assert sparse_mla_reference_topk_chunk_size() == 512
        assert sparse_mla_reference_query_chunk_size() == 1


def test_sparse_mla_dump_path_uses_new_name() -> None:
    with _patched_sparse_mla_env(VLLM_TRITON_MLA_SPARSE_DUMP_PATH="/tmp/new.jsonl"):
        assert sparse_mla_attention_dump_path() == "/tmp/new.jsonl"
