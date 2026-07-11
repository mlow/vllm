# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.attention.backends.mla.indexer import _uses_varlen_dspark_capacity
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec


def test_varlen_indexer_is_limited_to_active_dspark_capacity() -> None:
    def config(**overrides):
        values = {
            "use_dspark": lambda: True,
            "dspark_capacity_verification_mode": "varlen",
            "dspark_confidence_threshold": 0.0,
            "dspark_budget_frac": 1.0,
            "dspark_sps_curve": None,
        }
        values.update(overrides)
        return SimpleNamespace(speculative_config=SimpleNamespace(**values))

    assert not _uses_varlen_dspark_capacity(SimpleNamespace(speculative_config=None))
    assert not _uses_varlen_dspark_capacity(config())
    assert not _uses_varlen_dspark_capacity(
        config(use_dspark=lambda: False, dspark_budget_frac=0.5)
    )
    assert not _uses_varlen_dspark_capacity(
        config(dspark_capacity_verification_mode="mask", dspark_budget_frac=0.5)
    )
    assert _uses_varlen_dspark_capacity(config(dspark_confidence_threshold=0.5))
    assert _uses_varlen_dspark_capacity(config(dspark_budget_frac=0.5))
    assert _uses_varlen_dspark_capacity(config(dspark_sps_curve="auto"))


def test_dspark_swa_decode_threshold_matches_target_verification() -> None:
    """DSpark verifies 1 + K target tokens, not the generic 1 + 2K."""
    speculative_config = SimpleNamespace(
        num_speculative_tokens=5,
        parallel_drafting=True,
        use_dspark=lambda: True,
    )
    hf_config = SimpleNamespace(sliding_window=128, compress_ratios=[1, 4, 128])
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=4096, hf_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
    )
    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
    )

    builder = DeepseekSparseSWAMetadataBuilder(
        kv_cache_spec,
        ["placeholder"],
        vllm_config,
        torch.device("cpu"),
    )

    assert builder.decode_threshold == 6
