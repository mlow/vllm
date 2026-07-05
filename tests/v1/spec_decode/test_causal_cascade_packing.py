# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.spec_decode.causal_cascade_packing import pack_sparse_mla_rows
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.causal_cascade.live_state import (
    CausalCascadeLiveState,
    _LayerCaptureBuffer,
)
from vllm.v1.worker.gpu.spec_decode.causal_cascade.speculator import (
    CausalCascadeSpeculator,
)


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


def test_live_state_accepts_physical_slots_without_logical_indices() -> None:
    row_width = 13
    flat_cache = torch.arange(8 * row_width, dtype=torch.uint8).reshape(
        8,
        row_width,
    )
    kv_layer = flat_cache.reshape(1, 8, row_width)
    physical_slots = torch.tensor([[2, -1, 5]], dtype=torch.int32)

    state = CausalCascadeLiveState.__new__(CausalCascadeLiveState)
    state._is_tp_rank_zero = True
    state._topk = 3
    state._expected_row_width = row_width
    state._warned_row_widths = set()
    state._debug_live_input_missing_counts = {}
    state._kv_caches_by_layer_id = {18: kv_layer}
    state._capture_buffers = {
        18: _LayerCaptureBuffer(
            page_table=physical_slots,
            topk_indices=torch.full_like(physical_slots, -1),
            nsa_lens=torch.tensor([3], dtype=torch.int32),
            cache_lens=torch.tensor([7], dtype=torch.int32),
            req_ids=torch.tensor([-1], dtype=torch.int32),
            token_ids=torch.tensor([42], dtype=torch.long),
            layer_name="layers.18",
            num_actual_tokens=1,
            topk=3,
            has_logical_indices=False,
            has_req_ids=False,
            has_token_ids=True,
        )
    }

    live_inputs = state.get_live_causal_cascade_inputs(
        [18],
        block_size=8,
        topk=3,
    )

    assert live_inputs is not None
    assert "mla_cache_topk_indices" not in live_inputs
    assert "selected_req_ids" not in live_inputs
    assert torch.equal(
        live_inputs["mla_cache_physical_slots"],
        physical_slots.reshape(1, 1, 3),
    )
    assert torch.equal(
        live_inputs["mla_cache_valid_mask"],
        torch.tensor([[[True, False, True]]]),
    )
    assert torch.equal(
        live_inputs["mla_cache_rows_packed"][0, 0, 0],
        flat_cache[2],
    )
    assert torch.equal(
        live_inputs["mla_cache_rows_packed"][0, 0, 2],
        flat_cache[5],
    )


def test_embedded_mtp_builds_own_attention_metadata(monkeypatch) -> None:
    parent_attn_groups = object()

    def set_parent_attn(self, model_state, kv_cache_config, block_tables) -> None:
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        self.attn_groups = parent_attn_groups

    monkeypatch.setattr(AutoRegressiveSpeculator, "set_attn", set_parent_attn)

    speculator = object.__new__(CausalCascadeSpeculator)
    model_state = object()
    kv_cache_config = object()
    block_tables = object()

    speculator.set_attn(model_state, kv_cache_config, block_tables)

    assert speculator.model_state is model_state
    assert speculator.kv_cache_config is kv_cache_config
    assert speculator.block_tables is block_tables
    assert speculator.attn_groups is parent_attn_groups
    assert speculator.rebuild_prefill_attn_metadata is True


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
