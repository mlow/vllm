# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.envs import environment_variables
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla import flashmla_sparse, sparse_swa
from vllm.v1.attention.backends.mla.sparse_mla_env import (
    disable_triton_sparse_mla_cudagraphs_if_enabled,
    triton_sparse_mla_cudagraphs_allowed,
)


def _vllm_config(
    *,
    num_speculative_tokens: int = 0,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.FULL_AND_PIECEWISE,
):
    return SimpleNamespace(
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
            if num_speculative_tokens
            else None
        ),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE,
            compile_sizes=[1, 2],
            compile_ranges_endpoints=[(1, 8)],
            cudagraph_mode=cudagraph_mode,
            cudagraph_capture_sizes=[1, 2],
            max_cudagraph_capture_size=2,
        ),
    )


@pytest.fixture(autouse=True)
def _clear_sparse_mla_graph_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH", raising=False)


def test_sparse_mla_cudagraphs_default_to_disabled() -> None:
    assert not triton_sparse_mla_cudagraphs_allowed()


def test_sparse_mla_cudagraph_env_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH", "0")
    assert not triton_sparse_mla_cudagraphs_allowed()

    monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH", "1")
    assert triton_sparse_mla_cudagraphs_allowed()


def test_sparse_mla_cudagraph_env_is_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_value = environment_variables["VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH"]

    assert env_value() is None
    monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH", "0")
    assert env_value() is False
    monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH", "true")
    assert env_value() is True


def test_sparse_mla_graph_gate_defaults_to_off_without_mtp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.sparse_mla_env."
        "is_triton_sparse_mla_enabled_for_platform",
        lambda: True,
    )
    vllm_config = _vllm_config()

    disable_triton_sparse_mla_cudagraphs_if_enabled(vllm_config)

    assert vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert vllm_config.compilation_config.compile_sizes == [1, 2]
    assert vllm_config.compilation_config.compile_ranges_endpoints == [(1, 8)]
    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert vllm_config.compilation_config.cudagraph_capture_sizes == []
    assert vllm_config.compilation_config.max_cudagraph_capture_size == 0


def test_sparse_mla_graph_gate_can_be_forced_on_without_sm12x_downgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.sparse_mla_env."
        "is_triton_sparse_mla_enabled_for_platform",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.sparse_mla_env._is_sm12x_platform",
        lambda: False,
    )
    monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH", "1")
    vllm_config = _vllm_config()

    disable_triton_sparse_mla_cudagraphs_if_enabled(vllm_config)

    assert vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert (
        vllm_config.compilation_config.cudagraph_mode
        == CUDAGraphMode.FULL_AND_PIECEWISE
    )
    assert vllm_config.compilation_config.compile_sizes == [1, 2]
    assert vllm_config.compilation_config.cudagraph_capture_sizes == [1, 2]


def test_sparse_mla_graph_gate_forces_piecewise_on_sm12x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.sparse_mla_env."
        "is_triton_sparse_mla_enabled_for_platform",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.sparse_mla_env._is_sm12x_platform",
        lambda: True,
    )
    monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH", "1")
    vllm_config = _vllm_config()

    disable_triton_sparse_mla_cudagraphs_if_enabled(vllm_config)

    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
    assert vllm_config.compilation_config.compile_sizes == [1, 2]
    assert vllm_config.compilation_config.cudagraph_capture_sizes == [1, 2]


def test_sparse_mla_graph_gate_keeps_piecewise_when_forced_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.sparse_mla_env."
        "is_triton_sparse_mla_enabled_for_platform",
        lambda: True,
    )
    monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH", "1")
    vllm_config = _vllm_config(cudagraph_mode=CUDAGraphMode.PIECEWISE)

    disable_triton_sparse_mla_cudagraphs_if_enabled(vllm_config)

    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE


def test_sparse_mla_graph_gate_can_be_forced_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.sparse_mla_env."
        "is_triton_sparse_mla_enabled_for_platform",
        lambda: True,
    )
    monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH", "0")
    vllm_config = _vllm_config()

    disable_triton_sparse_mla_cudagraphs_if_enabled(vllm_config)

    assert vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert vllm_config.compilation_config.compile_sizes == [1, 2]
    assert vllm_config.compilation_config.compile_ranges_endpoints == [(1, 8)]
    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert vllm_config.compilation_config.cudagraph_capture_sizes == []
    assert vllm_config.compilation_config.max_cudagraph_capture_size == 0


def test_sparse_mla_graph_gate_defaults_to_off_for_mtp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.sparse_mla_env."
        "is_triton_sparse_mla_enabled_for_platform",
        lambda: True,
    )
    vllm_config = _vllm_config(num_speculative_tokens=2)

    disable_triton_sparse_mla_cudagraphs_if_enabled(vllm_config)

    assert vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert vllm_config.compilation_config.compile_sizes == [1, 2]
    assert vllm_config.compilation_config.compile_ranges_endpoints == [(1, 8)]
    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert vllm_config.compilation_config.cudagraph_capture_sizes == []
    assert vllm_config.compilation_config.max_cudagraph_capture_size == 0


def test_sparse_mla_metadata_builders_follow_graph_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        flashmla_sparse, "is_triton_sparse_mla_enabled_for_platform", lambda: True
    )
    monkeypatch.setattr(
        sparse_swa, "is_triton_sparse_mla_enabled_for_platform", lambda: True
    )
    kv_cache_spec = SimpleNamespace(model_version="deepseek_v4")

    for builder in (
        flashmla_sparse.FlashMLASparseMetadataBuilder,
        sparse_swa.DeepseekSparseSWAMetadataBuilder,
    ):
        assert (
            builder.get_cudagraph_support(_vllm_config(), kv_cache_spec)
            == AttentionCGSupport.NEVER
        )
        monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH", "1")
        assert (
            builder.get_cudagraph_support(_vllm_config(), kv_cache_spec)
            == AttentionCGSupport.UNIFORM_BATCH
        )
        monkeypatch.delenv(
            "VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH", raising=False
        )
        assert (
            builder.get_cudagraph_support(
                _vllm_config(num_speculative_tokens=2), kv_cache_spec
            )
            == AttentionCGSupport.NEVER
        )
