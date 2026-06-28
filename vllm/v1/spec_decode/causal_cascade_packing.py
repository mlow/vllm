# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packing helpers for CausalCascade live sparse-MLA inputs."""

import torch

from vllm.triton_utils import tl, triton


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


@triton.jit
def _pack_sparse_mla_rows_kernel(
    flat_cache_ptr,
    physical_slots_ptr,
    packed_ptr,
    valid_mask_ptr,
    flat_stride_0: tl.constexpr,
    flat_stride_1: tl.constexpr,
    physical_stride_0: tl.constexpr,
    physical_stride_1: tl.constexpr,
    packed_stride_0: tl.constexpr,
    packed_stride_1: tl.constexpr,
    packed_stride_2: tl.constexpr,
    valid_stride_0: tl.constexpr,
    valid_stride_1: tl.constexpr,
    NUM_PHYSICAL_SLOTS: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
) -> None:
    row_idx = tl.program_id(0)
    topk_idx = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_WIDTH)

    slot = tl.load(
        physical_slots_ptr + row_idx * physical_stride_0 + topk_idx * physical_stride_1
    )
    valid = (slot >= 0) & (slot < NUM_PHYSICAL_SLOTS)
    safe_slot = tl.where(valid, slot, 0)

    values = tl.load(
        flat_cache_ptr + safe_slot * flat_stride_0 + offsets * flat_stride_1,
        mask=offsets < ROW_WIDTH,
        other=0,
    )
    tl.store(
        packed_ptr
        + row_idx * packed_stride_0
        + topk_idx * packed_stride_1
        + offsets * packed_stride_2,
        values,
        mask=offsets < ROW_WIDTH,
    )
    tl.store(
        valid_mask_ptr + row_idx * valid_stride_0 + topk_idx * valid_stride_1,
        valid,
    )


def pack_sparse_mla_rows(
    flat_cache: torch.Tensor,
    physical_slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather packed sparse-MLA cache rows and return a validity mask.

    Args:
        flat_cache: Packed FP8 rows shaped ``[num_physical_slots, row_width]``.
        physical_slots: Physical row IDs shaped ``[num_rows, topk]``. Negative
            entries and positive entries outside ``flat_cache`` are padding;
            they are gathered from slot 0 for deterministic bytes and marked
            invalid in the returned mask.

    Returns:
        ``(packed_rows, valid_mask)`` with shapes
        ``[num_rows, topk, row_width]`` and ``[num_rows, topk]``.
    """
    if flat_cache.ndim != 2:
        raise ValueError(
            f"flat_cache must be rank 2, got shape {tuple(flat_cache.shape)}"
        )
    if physical_slots.ndim != 2:
        raise ValueError(
            f"physical_slots must be rank 2, got shape {tuple(physical_slots.shape)}"
        )
    if flat_cache.device != physical_slots.device:
        raise ValueError(
            "flat_cache and physical_slots must be on the same device, got "
            f"{flat_cache.device} and {physical_slots.device}"
        )
    if flat_cache.shape[0] <= 0:
        raise ValueError("flat_cache must contain at least one physical row")

    num_rows = int(physical_slots.shape[0])
    topk = int(physical_slots.shape[1])
    row_width = int(flat_cache.shape[1])
    if num_rows <= 0 or topk <= 0 or row_width <= 0:
        raise ValueError(
            "Cannot pack empty sparse-MLA rows: "
            f"num_rows={num_rows}, topk={topk}, row_width={row_width}"
        )

    physical_slots = physical_slots.contiguous()
    valid_mask = torch.empty(
        (num_rows, topk),
        device=physical_slots.device,
        dtype=torch.bool,
    )
    packed_rows = torch.empty(
        (num_rows, topk, row_width),
        device=flat_cache.device,
        dtype=flat_cache.dtype,
    )

    if not flat_cache.is_cuda:
        physical_slots_long = physical_slots.to(torch.long)
        valid_mask.copy_(
            (physical_slots_long >= 0)
            & (physical_slots_long < int(flat_cache.shape[0]))
        )
        gather_slots = torch.where(
            valid_mask,
            physical_slots_long,
            torch.zeros_like(physical_slots_long),
        )
        packed_rows.copy_(
            flat_cache.index_select(0, gather_slots.reshape(-1)).view(
                num_rows,
                topk,
                row_width,
            )
        )
        return packed_rows, valid_mask

    flat_cache = flat_cache.contiguous()
    block_width = _next_power_of_2(row_width)
    _pack_sparse_mla_rows_kernel[(num_rows, topk)](
        flat_cache,
        physical_slots,
        packed_rows,
        valid_mask,
        flat_cache.stride(0),
        flat_cache.stride(1),
        physical_slots.stride(0),
        physical_slots.stride(1),
        packed_rows.stride(0),
        packed_rows.stride(1),
        packed_rows.stride(2),
        valid_mask.stride(0),
        valid_mask.stride(1),
        NUM_PHYSICAL_SLOTS=int(flat_cache.shape[0]),
        ROW_WIDTH=row_width,
        BLOCK_WIDTH=block_width,
    )
    return packed_rows, valid_mask
