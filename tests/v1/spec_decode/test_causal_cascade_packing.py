# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.spec_decode.causal_cascade_packing import pack_sparse_mla_rows


def _expected_pack(
    flat_cache: torch.Tensor,
    physical_slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = (physical_slots >= 0) & (physical_slots < flat_cache.shape[0])
    gather_slots = torch.where(
        valid_mask,
        physical_slots,
        torch.zeros_like(physical_slots),
    ).to(torch.long)
    packed = flat_cache.index_select(0, gather_slots.reshape(-1)).view(
        *physical_slots.shape,
        flat_cache.shape[-1],
    )
    return packed, valid_mask


def test_pack_sparse_mla_rows_cpu_masks_negative_slots() -> None:
    flat_cache = (
        torch.arange(6, dtype=torch.uint8).view(6, 1).expand(6, 13).contiguous()
    )
    physical_slots = torch.tensor(
        [[2, -1, 4], [5, 0, -1]],
        dtype=torch.int32,
    )

    packed, valid_mask = pack_sparse_mla_rows(flat_cache, physical_slots)
    expected_packed, expected_mask = _expected_pack(flat_cache, physical_slots)

    assert torch.equal(valid_mask, expected_mask)
    assert torch.equal(packed, expected_packed)
    assert torch.all(packed[0, 1] == 0)
    assert torch.all(packed[1, 2] == 0)


def test_pack_sparse_mla_rows_cpu_masks_oob_positive_slots() -> None:
    flat_cache = (
        torch.arange(6, dtype=torch.uint8).view(6, 1).expand(6, 13).contiguous()
    )
    physical_slots = torch.tensor(
        [[2, 99, 4], [5, 6, -1]],
        dtype=torch.int32,
    )

    packed, valid_mask = pack_sparse_mla_rows(flat_cache, physical_slots)
    expected_packed, expected_mask = _expected_pack(flat_cache, physical_slots)

    assert torch.equal(valid_mask, expected_mask)
    assert torch.equal(packed, expected_packed)
    assert torch.equal(
        valid_mask,
        torch.tensor([[True, False, True], [True, False, False]]),
    )
    assert torch.all(packed[0, 1] == 0)
    assert torch.all(packed[1, 1] == 0)
    assert torch.all(packed[1, 2] == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pack_sparse_mla_rows_cuda_matches_torch() -> None:
    flat_cache = torch.randint(
        0,
        256,
        (32, 656),
        device="cuda",
        dtype=torch.uint8,
    )
    physical_slots = torch.tensor(
        [[2, -1, 4, 7], [31, 32, -1, 5], [9, 10, 99, -1]],
        device="cuda",
        dtype=torch.int32,
    )

    packed, valid_mask = pack_sparse_mla_rows(flat_cache, physical_slots)
    expected_packed, expected_mask = _expected_pack(flat_cache, physical_slots)

    assert torch.equal(valid_mask, expected_mask)
    assert torch.equal(packed, expected_packed)
