# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from benchmarks.kernels import benchmark_sparse_mla_splitkv as bench


def test_choose_num_kv_splits_expands_low_batch_long_context() -> None:
    assert (
        bench.choose_num_kv_splits(
            num_tokens=1,
            num_heads=64,
            num_candidates=2048,
            sm_count=120,
        )
        == 16
    )


def test_choose_num_kv_splits_keeps_saturated_batches_single_pass() -> None:
    assert (
        bench.choose_num_kv_splits(
            num_tokens=8,
            num_heads=64,
            num_candidates=2048,
            sm_count=120,
        )
        == 1
    )


def test_choose_num_kv_splits_respects_candidate_divisibility() -> None:
    assert (
        bench.choose_num_kv_splits(
            num_tokens=1,
            num_heads=64,
            num_candidates=1536,
            sm_count=120,
        )
        == 16
    )
    assert (
        bench.choose_num_kv_splits(
            num_tokens=1,
            num_heads=64,
            num_candidates=1537,
            sm_count=120,
        )
        == 1
    )
