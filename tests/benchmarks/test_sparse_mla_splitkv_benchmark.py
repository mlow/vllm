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


def test_choose_num_kv_splits_keeps_medium_batch_long_context_split() -> None:
    assert (
        bench.choose_num_kv_splits(
            num_tokens=16,
            num_heads=64,
            num_candidates=2048,
            sm_count=188,
        )
        == 8
    )


def test_choose_num_kv_splits_lifts_medium_batch_on_smaller_gpus() -> None:
    assert (
        bench.choose_num_kv_splits(
            num_tokens=16,
            num_heads=64,
            num_candidates=2048,
            sm_count=48,
        )
        == 4
    )
    assert (
        bench.choose_num_kv_splits(
            num_tokens=16,
            num_heads=64,
            num_candidates=4096,
            sm_count=48,
        )
        == 8
    )


def test_choose_num_kv_splits_does_not_oversplit_small_batches_on_smaller_gpus() -> (
    None
):
    assert (
        bench.choose_num_kv_splits(
            num_tokens=8,
            num_heads=64,
            num_candidates=2048,
            sm_count=48,
        )
        == 4
    )


def test_choose_num_kv_splits_keeps_large_saturated_batches_single_pass() -> None:
    assert (
        bench.choose_num_kv_splits(
            num_tokens=64,
            num_heads=128,
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


def test_choose_num_kv_splits_handles_empty_shapes() -> None:
    assert (
        bench.choose_num_kv_splits(
            num_tokens=0,
            num_heads=64,
            num_candidates=2048,
            sm_count=120,
        )
        == 1
    )
    assert (
        bench.choose_num_kv_splits(
            num_tokens=1,
            num_heads=0,
            num_candidates=2048,
            sm_count=120,
        )
        == 1
    )
    assert (
        bench.choose_num_kv_splits(
            num_tokens=1,
            num_heads=64,
            num_candidates=0,
            sm_count=120,
        )
        == 1
    )


def test_filter_matmul_kwargs_keeps_only_supported_optional_parameters() -> None:
    def old_kernel(required, *, num_heads=None):
        return required, num_heads

    def current_kernel(
        required,
        *,
        num_heads=None,
        score_buffer=None,
        value_block_size=None,
        candidate_block_size=None,
    ):
        return required, num_heads, score_buffer, value_block_size, candidate_block_size

    optional_kwargs = {
        "num_heads": 64,
        "score_buffer": object(),
        "value_block_size": 512,
        "candidate_block_size": 128,
    }

    assert bench._filter_matmul_kwargs(old_kernel, optional_kwargs) == {
        "num_heads": 64,
    }
    assert (
        bench._filter_matmul_kwargs(current_kernel, optional_kwargs) == optional_kwargs
    )
