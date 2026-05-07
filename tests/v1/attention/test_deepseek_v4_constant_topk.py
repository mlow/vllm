# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.mla.flashmla_sparse import (
    build_c128a_topk_metadata,
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    _compute_swa_indices_and_lens_kernel,
)
from vllm.v1.attention.ops.deepseek_v4_ops.cache_utils import (
    compute_global_topk_indices_and_lens,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_c4a_topk_lens_can_be_constant_for_full_cudagraph() -> None:
    device = torch.device("cuda")
    topk_indices = torch.tensor(
        [
            [0, 2, -1, -1],
            [0, -1, -1, -1],
            [0, 1, 2, 3],
        ],
        dtype=torch.int32,
        device=device,
    )
    token_to_req_indices = torch.zeros(3, dtype=torch.int32, device=device)
    block_table = torch.tensor([[10]], dtype=torch.int32, device=device)
    is_valid_token = torch.tensor([True, False, True], device=device)

    _, dynamic_lens = compute_global_topk_indices_and_lens(
        topk_indices,
        token_to_req_indices,
        block_table,
        block_size=8,
        is_valid_token=is_valid_token,
    )
    _, constant_lens = compute_global_topk_indices_and_lens(
        topk_indices,
        token_to_req_indices,
        block_table,
        block_size=8,
        is_valid_token=is_valid_token,
        needs_constant_topk=True,
    )
    torch.cuda.synchronize()

    assert dynamic_lens.cpu().tolist() == [2, 0, 4]
    assert constant_lens.cpu().tolist() == [4, 4, 4]


def test_c128a_decode_lens_can_be_constant_for_full_cudagraph() -> None:
    device = torch.device("cuda")
    max_compressed_tokens = 8
    num_decode_tokens = 3
    positions = torch.tensor([0, 127, 511, 255], dtype=torch.int64, device=device)
    token_to_req_indices = torch.zeros(4, dtype=torch.int32, device=device)
    block_table = torch.tensor([[5, 7, 11, 13]], dtype=torch.int32, device=device)
    slot_mapping = torch.tensor([0, -1, 2, 3], dtype=torch.int64, device=device)

    def build(needs_constant_topk: bool) -> tuple[torch.Tensor, torch.Tensor]:
        global_decode, decode_lens, _ = build_c128a_topk_metadata(
            positions,
            compress_ratio=128,
            num_decode_tokens=num_decode_tokens,
            token_to_req_indices=token_to_req_indices,
            block_table=block_table,
            block_size=2,
            slot_mapping=slot_mapping,
            global_decode_buffer=torch.empty(
                (num_decode_tokens, max_compressed_tokens),
                dtype=torch.int32,
                device=device,
            ),
            decode_lens_buffer=torch.empty(
                num_decode_tokens,
                dtype=torch.int32,
                device=device,
            ),
            prefill_buffer=torch.empty(
                (positions.numel() - num_decode_tokens, max_compressed_tokens),
                dtype=torch.int32,
                device=device,
            ),
            max_compressed_tokens=max_compressed_tokens,
            needs_constant_topk=needs_constant_topk,
        )
        return global_decode, decode_lens

    _, dynamic_lens = build(needs_constant_topk=False)
    constant_global, constant_lens = build(needs_constant_topk=True)
    torch.cuda.synchronize()

    assert dynamic_lens.cpu().tolist() == [0, 0, 4]
    assert constant_lens.cpu().tolist() == [8, 8, 8]
    assert constant_global[1].cpu().tolist() == [-1] * max_compressed_tokens


def test_swa_decode_lens_can_be_constant_for_full_cudagraph() -> None:
    device = torch.device("cuda")
    num_decode_tokens = 3
    window_size = 4
    block_size = 8
    token_to_req_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([1, 2, 10], dtype=torch.int32, device=device)
    is_valid_token = torch.tensor([True, True, False], device=device)
    block_table = torch.tensor(
        [[10, 11], [20, 21], [30, 31]], dtype=torch.int32, device=device
    )

    def compute(needs_constant_topk: bool) -> tuple[torch.Tensor, torch.Tensor]:
        swa_indices = torch.full(
            (num_decode_tokens, 1, window_size),
            123,
            dtype=torch.int32,
            device=device,
        )
        swa_lens = torch.empty(num_decode_tokens, dtype=torch.int32, device=device)
        _compute_swa_indices_and_lens_kernel[(num_decode_tokens,)](
            swa_indices,
            swa_indices.stride(0),
            swa_lens,
            window_size,
            query_start_loc,
            seq_lens,
            token_to_req_indices,
            is_valid_token,
            block_table,
            block_table.stride(0),
            block_size,
            TRITON_BLOCK_SIZE=4,
            CONSTANT_TOPK_LEN=needs_constant_topk,
        )
        return swa_indices, swa_lens

    _, dynamic_lens = compute(needs_constant_topk=False)
    constant_indices, constant_lens = compute(needs_constant_topk=True)
    torch.cuda.synchronize()

    assert dynamic_lens.cpu().tolist() == [1, 2, 0]
    assert constant_lens.cpu().tolist() == [4, 4, 4]
    assert constant_indices[2, 0].cpu().tolist() == [-1] * window_size
