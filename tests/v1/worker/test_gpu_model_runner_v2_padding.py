# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.worker.gpu.model_runner import (
    _finalize_input_padding_mask,
    _uses_input_padding_mask,
)


def test_standard_batch_ignores_stale_cudagraph_padding() -> None:
    input_batch = SimpleNamespace(
        num_tokens=3,
        num_tokens_after_padding=5,
        num_draft_tokens=0,
        is_padding=torch.ones(5, dtype=torch.bool),
    )

    _finalize_input_padding_mask(
        input_batch,
        has_capacity_manager=False,
        moe_skip_padding=False,
    )

    assert not _uses_input_padding_mask(input_batch, False, False)
    assert input_batch.is_padding.tolist() == [True, True, True, True, True]


def test_capacity_batch_preserves_active_mask_and_marks_graph_tail() -> None:
    input_batch = SimpleNamespace(
        num_tokens=3,
        num_tokens_after_padding=5,
        num_draft_tokens=2,
        is_padding=torch.tensor([False, True, False, False, False]),
    )

    _finalize_input_padding_mask(
        input_batch,
        has_capacity_manager=True,
        moe_skip_padding=True,
    )

    assert _uses_input_padding_mask(input_batch, True, True)
    assert input_batch.is_padding.tolist() == [False, True, False, True, True]
