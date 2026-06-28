# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native live sparse-MLA state for CausalCascade serving.

This module is deliberately independent from KV-transfer connectors. It owns the
graph-stable GPU buffers used by live CausalCascade inference, records sparse
MLA metadata/cache-row pointers during target model execution, and exposes those
rows directly to the CausalCascade speculator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.v1.spec_decode.causal_cascade_packing import pack_sparse_mla_rows

logger = init_logger(__name__)

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_LIVE_STATE: CausalCascadeLiveState | None = None


@dataclass
class _LayerCaptureBuffer:
    page_table: torch.Tensor
    topk_indices: torch.Tensor
    nsa_lens: torch.Tensor
    cache_lens: torch.Tensor
    req_ids: torch.Tensor
    token_ids: torch.Tensor
    layer_name: str = ""
    num_actual_tokens: int = 0
    topk: int = 0
    has_token_ids: bool = False


def _parse_layer_id(layer_name: str) -> int | None:
    match = _LAYER_RE.search(layer_name)
    if match is None:
        return None
    return int(match.group(1))


def configure_causal_cascade_live_state(
    vllm_config: VllmConfig,
    *,
    layer_ids: list[int],
    topk: int,
    block_size: int,
    expected_row_width: int = 656,
) -> None:
    """Create the process-local native live state for CausalCascade serving."""
    global _LIVE_STATE
    _LIVE_STATE = CausalCascadeLiveState(
        vllm_config,
        layer_ids=layer_ids,
        topk=topk,
        block_size=block_size,
        expected_row_width=expected_row_width,
    )


def get_causal_cascade_live_state() -> CausalCascadeLiveState | None:
    return _LIVE_STATE


def register_causal_cascade_kv_caches(kv_caches: dict[str, torch.Tensor]) -> None:
    state = _LIVE_STATE
    if state is not None:
        state.register_kv_caches(kv_caches)


def capture_causal_cascade_sparse_mla_layer(
    layer_name: str,
    kv_layer: torch.Tensor,
    page_table_1: torch.Tensor,
    topk_indices: torch.Tensor | None,
    nsa_cache_seqlens: torch.Tensor,
    cache_seq_lens_per_token: torch.Tensor,
    req_id_per_token: torch.Tensor,
    token_ids_per_token: torch.Tensor | None,
    num_actual_tokens: int,
) -> None:
    state = _LIVE_STATE
    if state is not None:
        state.capture_sparse_mla_layer(
            layer_name,
            kv_layer,
            page_table_1,
            topk_indices,
            nsa_cache_seqlens,
            cache_seq_lens_per_token,
            req_id_per_token,
            token_ids_per_token,
            num_actual_tokens,
        )


def capture_causal_cascade_anchor_hidden_state(
    hidden_states: torch.Tensor,
    num_rows: int | None = None,
) -> None:
    state = _LIVE_STATE
    if state is not None:
        state.capture_anchor_hidden_state_tensor(hidden_states, num_rows=num_rows)


class CausalCascadeLiveState:
    """Graph-stable sparse-MLA capture buffers for live CausalCascade inference."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        layer_ids: list[int],
        topk: int,
        block_size: int,
        expected_row_width: int = 656,
    ) -> None:
        self._layer_ids = [int(layer_id) for layer_id in layer_ids]
        self._layer_id_set = set(self._layer_ids)
        self._topk = int(topk)
        self._block_size = int(block_size)
        self._expected_row_width = int(expected_row_width)
        self._max_num_batched_tokens = int(
            getattr(vllm_config.scheduler_config, "max_num_batched_tokens", 0) or 0
        )
        hf_config = getattr(vllm_config.model_config, "hf_config", None)
        self._hidden_size = int(getattr(hf_config, "hidden_size", 0) or 0)
        self._is_tp_rank_zero = get_tensor_model_parallel_rank() == 0
        self._kv_caches_by_layer_id: dict[int, torch.Tensor] = {}
        self._capture_buffers: dict[int, _LayerCaptureBuffer] = {}
        self._anchor_hidden_state_buffer: torch.Tensor | None = None
        self._warned_row_widths: set[int] = set()
        self._debug_capture_log_counts: dict[int, int] = {}
        self._debug_live_input_missing_counts: dict[str, int] = {}
        if self._topk <= 0:
            raise ValueError(f"CausalCascade sparse topk must be > 0, got {topk}")
        if self._expected_row_width != 656:
            raise ValueError(
                "CausalCascade live sparse-MLA currently expects packed "
                f"fp8_ds_mla rows of width 656, got {expected_row_width}"
            )
        logger.info(
            "CausalCascade native live state configured layers=%s topk=%d "
            "block_size=%d max_num_batched_tokens=%d hidden_size=%d tp_rank0=%s",
            self._layer_ids,
            self._topk,
            self._block_size,
            self._max_num_batched_tokens,
            self._hidden_size,
            self._is_tp_rank_zero,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if not self._is_tp_rank_zero:
            return
        self._kv_caches_by_layer_id.clear()
        for layer_name, kv_cache in kv_caches.items():
            layer_id = _parse_layer_id(layer_name)
            if layer_id is None or layer_id not in self._layer_id_set:
                continue
            self._kv_caches_by_layer_id[layer_id] = kv_cache
        if self._kv_caches_by_layer_id:
            device = next(iter(self._kv_caches_by_layer_id.values())).device
            self._allocate_capture_buffers(device)
        logger.info(
            "CausalCascade native live state registered sparse-MLA KV layers=%s",
            sorted(self._kv_caches_by_layer_id),
        )

    def capture_sparse_mla_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        page_table_1: torch.Tensor,
        topk_indices: torch.Tensor | None,
        nsa_cache_seqlens: torch.Tensor,
        cache_seq_lens_per_token: torch.Tensor,
        req_id_per_token: torch.Tensor,
        token_ids_per_token: torch.Tensor | None,
        num_actual_tokens: int,
    ) -> None:
        if not self._is_tp_rank_zero:
            return
        layer_id = _parse_layer_id(layer_name)
        if layer_id is None or layer_id not in self._layer_id_set:
            return
        if page_table_1.ndim != 2:
            logger.warning_once(
                "CausalCascade live state expected page_table_1 rank 2, got %s",
                tuple(page_table_1.shape),
            )
            return
        flat_cache = self._flat_packed_cache(kv_layer)
        if flat_cache is None:
            logger.warning_once(
                "CausalCascade live state expected packed fp8_ds_mla cache "
                "[pages,page,656], got shape=%s dtype=%s for layer %s",
                tuple(kv_layer.shape),
                kv_layer.dtype,
                layer_name,
            )
            return
        del flat_cache

        num_actual_tokens = int(num_actual_tokens)
        if num_actual_tokens <= 0:
            return
        topk = min(self._topk, int(page_table_1.shape[1]))
        if topk_indices is not None:
            topk = min(topk, int(topk_indices.shape[1]))
        buffer = self._capture_buffers.get(layer_id)
        if buffer is None:
            if self._is_cuda_graph_capturing(page_table_1):
                logger.warning_once(
                    "CausalCascade live state has no capture buffer for layer %s "
                    "during CUDA graph capture.",
                    layer_name,
                )
                return
            self._allocate_capture_buffers(page_table_1.device)
            buffer = self._capture_buffers.get(layer_id)
        if buffer is None:
            return
        if (
            num_actual_tokens > buffer.page_table.shape[0]
            or topk > buffer.page_table.shape[1]
        ):
            logger.warning_once(
                "CausalCascade live capture buffer too small for layer %s: "
                "need rows=%d topk=%d, have rows=%d topk=%d",
                layer_name,
                num_actual_tokens,
                topk,
                buffer.page_table.shape[0],
                buffer.page_table.shape[1],
            )
            return

        buffer.layer_name = layer_name
        buffer.num_actual_tokens = num_actual_tokens
        buffer.topk = topk
        buffer.has_token_ids = token_ids_per_token is not None
        buffer.page_table[:num_actual_tokens, :topk].copy_(
            page_table_1[:num_actual_tokens, :topk]
        )
        if topk_indices is None or topk_indices.ndim != 2:
            buffer.topk_indices[:num_actual_tokens, :topk].fill_(-1)
        else:
            buffer.topk_indices[:num_actual_tokens, :topk].copy_(
                topk_indices[:num_actual_tokens, :topk]
            )
        buffer.nsa_lens[:num_actual_tokens].copy_(nsa_cache_seqlens[:num_actual_tokens])
        buffer.cache_lens[:num_actual_tokens].copy_(
            cache_seq_lens_per_token[:num_actual_tokens]
        )
        buffer.req_ids[:num_actual_tokens].copy_(req_id_per_token[:num_actual_tokens])
        if token_ids_per_token is not None:
            buffer.token_ids[:num_actual_tokens].copy_(
                token_ids_per_token[:num_actual_tokens]
            )

        count = self._debug_capture_log_counts.get(layer_id, 0) + 1
        self._debug_capture_log_counts[layer_id] = count
        if count <= 3 or count & (count - 1) == 0:
            logger.info(
                "CausalCascade live capture layer=%d count=%d rows=%d topk=%d "
                "graph_capture=%s",
                layer_id,
                count,
                num_actual_tokens,
                topk,
                self._is_cuda_graph_capturing(page_table_1),
            )

    def capture_anchor_hidden_state_tensor(
        self,
        hidden_states: torch.Tensor,
        num_rows: int | None = None,
    ) -> None:
        if not self._is_tp_rank_zero:
            return
        if hidden_states.ndim != 2:
            logger.warning_once(
                "CausalCascade live state expected anchor hidden states rank 2, got %s",
                tuple(hidden_states.shape),
            )
            return
        buffer = self._anchor_hidden_state_buffer
        if buffer is None:
            if self._is_cuda_graph_capturing(hidden_states):
                logger.warning_once(
                    "CausalCascade live state anchor hidden buffer was not "
                    "allocated before CUDA graph capture; skipping capture."
                )
                return
            self._allocate_capture_buffers(hidden_states.device)
            buffer = self._anchor_hidden_state_buffer
        if buffer is None:
            return
        if hidden_states.shape[1] != buffer.shape[1]:
            logger.warning_once(
                "CausalCascade live state hidden width mismatch: got %d, expected %d",
                hidden_states.shape[1],
                buffer.shape[1],
            )
            return
        rows = int(hidden_states.shape[0] if num_rows is None else num_rows)
        rows = min(rows, int(hidden_states.shape[0]))
        if rows <= 0:
            return
        if rows > buffer.shape[0]:
            logger.warning_once(
                "CausalCascade live state anchor hidden buffer too small: "
                "need rows=%d, have rows=%d",
                rows,
                buffer.shape[0],
            )
            return
        buffer[:rows].copy_(hidden_states[:rows])

    def get_live_anchor_hidden_states(
        self,
        row_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self._is_tp_rank_zero:
            return None
        buffer = self._anchor_hidden_state_buffer
        if buffer is None:
            return None
        row_indices = row_indices.to(
            device=buffer.device,
            dtype=torch.long,
        ).reshape(-1)
        if row_indices.numel() == 0:
            return None
        if torch.any(row_indices < 0) or torch.any(row_indices >= buffer.shape[0]):
            return None
        return buffer.index_select(0, row_indices).contiguous()

    def get_live_causal_cascade_inputs(
        self,
        layer_ids: list[int],
        *,
        block_size: int,
        topk: int | None = None,
        row_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | None:
        def missing(reason: str, **details: Any) -> None:
            count = self._debug_live_input_missing_counts.get(reason, 0) + 1
            self._debug_live_input_missing_counts[reason] = count
            if count <= 8 or count & (count - 1) == 0:
                logger.warning(
                    "CausalCascade live inputs unavailable reason=%s count=%d "
                    "details=%s",
                    reason,
                    count,
                    details,
                )

        if not self._is_tp_rank_zero:
            return None
        if not layer_ids:
            missing("no_layer_ids")
            return None
        requested_topk = self._topk if topk is None else int(topk)
        if requested_topk <= 0:
            missing("bad_topk", requested_topk=requested_topk)
            return None

        if row_indices is None:
            first_buffer = self._capture_buffers.get(int(layer_ids[0]))
            if first_buffer is None:
                missing("missing_first_buffer", layer_id=int(layer_ids[0]))
                return None
            num_rows = int(first_buffer.num_actual_tokens)
            if num_rows <= 0:
                missing("no_live_decode_rows")
                return None
            row_indices = torch.arange(
                num_rows,
                device=first_buffer.page_table.device,
                dtype=torch.long,
            )
        else:
            row_indices = row_indices.to(dtype=torch.long).reshape(-1)
            if row_indices.numel() == 0:
                missing("empty_row_indices")
                return None
        num_rows = int(row_indices.numel())

        packed_layers: list[torch.Tensor] = []
        topk_index_layers: list[torch.Tensor] = []
        physical_slot_layers: list[torch.Tensor] = []
        valid_mask_layers: list[torch.Tensor] = []
        anchor_pos: torch.Tensor | None = None
        anchor_token_ids: torch.Tensor | None = None
        selected_cache_lens: torch.Tensor | None = None
        selected_nsa_lens: torch.Tensor | None = None
        selected_req_ids: torch.Tensor | None = None
        debug_cache_lens_head: torch.Tensor | None = None
        debug_nsa_lens_head: torch.Tensor | None = None
        debug_req_ids_head: torch.Tensor | None = None

        for layer_id in layer_ids:
            layer_id = int(layer_id)
            buffer = self._capture_buffers.get(layer_id)
            kv_layer = self._kv_caches_by_layer_id.get(layer_id)
            if buffer is None or kv_layer is None:
                missing(
                    "missing_layer_state",
                    layer_id=layer_id,
                    has_buffer=buffer is not None,
                    has_kv_layer=kv_layer is not None,
                )
                return None
            flat_cache = self._flat_packed_cache(kv_layer)
            if flat_cache is None:
                missing("missing_flat_cache", layer_id=layer_id)
                return None
            row_index = row_indices.to(device=buffer.page_table.device)
            if torch.any(row_index < 0) or torch.any(
                row_index >= buffer.page_table.shape[0]
            ):
                missing(
                    "row_index_oob",
                    layer_id=layer_id,
                    rows=num_rows,
                    row_min=int(row_index.min().item()),
                    row_max=int(row_index.max().item()),
                    table_rows=int(buffer.page_table.shape[0]),
                )
                return None

            layer_topk = min(
                requested_topk,
                int(buffer.page_table.shape[1]),
                int(buffer.topk_indices.shape[1]),
            )
            if layer_topk <= 0:
                missing(
                    "bad_layer_topk",
                    layer_id=layer_id,
                    requested_topk=requested_topk,
                    page_topk=int(buffer.page_table.shape[1]),
                    index_topk=int(buffer.topk_indices.shape[1]),
                )
                return None
            physical_slots = buffer.page_table.index_select(0, row_index)[
                :,
                :layer_topk,
            ].to(torch.long)
            valid_mask = (physical_slots >= 0) & (
                physical_slots < int(flat_cache.shape[0])
            )
            if not torch.all(valid_mask.any(dim=1)):
                missing(
                    "no_valid_physical_slots",
                    layer_id=layer_id,
                    rows=num_rows,
                    row_indices=row_indices.detach().cpu().tolist(),
                    valid_counts=valid_mask.sum(dim=1).detach().cpu().tolist(),
                    cache_rows=int(flat_cache.shape[0]),
                )
                return None
            topk_indices = buffer.topk_indices.index_select(0, row_index)[
                :,
                :layer_topk,
            ].to(torch.int32)

            gathered, valid_mask = pack_sparse_mla_rows(flat_cache, physical_slots)
            packed_layers.append(gathered)
            topk_index_layers.append(topk_indices.contiguous())
            physical_slot_layers.append(physical_slots.to(torch.int32).contiguous())
            valid_mask_layers.append(valid_mask.contiguous())

            if anchor_pos is None:
                cache_lens = buffer.cache_lens.index_select(
                    0,
                    row_index,
                ).to(torch.long)
                topk_anchor_pos = topk_indices.to(torch.long).amax(dim=1)
                anchor_pos = torch.where(
                    topk_anchor_pos >= 0, topk_anchor_pos, cache_lens - 1
                )
                selected_cache_lens = cache_lens.contiguous()
                selected_nsa_lens = (
                    buffer.nsa_lens.index_select(
                        0,
                        row_index,
                    )
                    .to(torch.long)
                    .contiguous()
                )
                selected_req_ids = (
                    buffer.req_ids.index_select(
                        0,
                        row_index,
                    )
                    .to(torch.long)
                    .contiguous()
                )
                debug_rows = int(buffer.num_actual_tokens)
                if debug_rows <= 0:
                    debug_rows = min(16, int(buffer.cache_lens.shape[0]))
                else:
                    debug_rows = min(16, debug_rows)
                debug_cache_lens_head = (
                    buffer.cache_lens[:debug_rows].to(torch.long).contiguous()
                )
                debug_nsa_lens_head = (
                    buffer.nsa_lens[:debug_rows].to(torch.long).contiguous()
                )
                debug_req_ids_head = (
                    buffer.req_ids[:debug_rows].to(torch.long).contiguous()
                )
                if buffer.has_token_ids:
                    anchor_token_ids = buffer.token_ids.index_select(
                        0,
                        row_index,
                    ).to(torch.long)

        assert anchor_pos is not None
        assert selected_cache_lens is not None
        assert selected_nsa_lens is not None
        assert selected_req_ids is not None
        anchor_token_ids_valid = anchor_token_ids is not None
        if anchor_token_ids is None:
            missing(
                "missing_anchor_token_ids",
                rows=num_rows,
                row_indices=row_indices.detach().cpu().tolist(),
                anchor_pos=anchor_pos.detach().cpu().tolist(),
            )
            anchor_token_ids = torch.zeros_like(anchor_pos, dtype=torch.long)
        block_offsets = torch.arange(
            block_size,
            device=anchor_pos.device,
            dtype=torch.long,
        )
        return {
            "mla_cache_rows_packed": torch.stack(packed_layers, dim=1),
            "mla_cache_topk_indices": torch.stack(topk_index_layers, dim=1),
            "mla_cache_physical_slots": torch.stack(physical_slot_layers, dim=1),
            "mla_cache_valid_mask": torch.stack(valid_mask_layers, dim=1),
            "verifier_layer_ids": torch.tensor(
                [int(layer_id) for layer_id in layer_ids],
                device=anchor_pos.device,
                dtype=torch.long,
            ),
            "anchor_pos": anchor_pos,
            "selected_cache_lens": selected_cache_lens,
            "selected_nsa_lens": selected_nsa_lens,
            "selected_req_ids": selected_req_ids,
            "debug_cache_lens_head": debug_cache_lens_head,
            "debug_nsa_lens_head": debug_nsa_lens_head,
            "debug_req_ids_head": debug_req_ids_head,
            "anchor_token_ids": anchor_token_ids,
            "anchor_token_ids_valid": torch.full(
                anchor_pos.shape,
                anchor_token_ids_valid,
                device=anchor_pos.device,
                dtype=torch.bool,
            ),
            "source_anchor_token_ids": anchor_token_ids,
            "source_anchor_token_ids_valid": torch.full(
                anchor_pos.shape,
                anchor_token_ids_valid,
                device=anchor_pos.device,
                dtype=torch.bool,
            ),
            "known_token_pos": anchor_pos + 1,
            "position_ids": anchor_pos.unsqueeze(1) + 1 + block_offsets.unsqueeze(0),
        }

    def _allocate_capture_buffers(self, device: torch.device) -> None:
        if self._max_num_batched_tokens <= 0:
            logger.warning_once(
                "CausalCascade live state cannot allocate capture buffers because "
                "max_num_batched_tokens is unset."
            )
            return
        if self._hidden_size <= 0:
            logger.warning_once(
                "CausalCascade live state cannot allocate anchor hidden buffer "
                "because hidden_size is unset."
            )
        elif (
            self._anchor_hidden_state_buffer is None
            or self._anchor_hidden_state_buffer.device != device
            or self._anchor_hidden_state_buffer.shape[0] < self._max_num_batched_tokens
            or self._anchor_hidden_state_buffer.shape[1] != self._hidden_size
        ):
            self._anchor_hidden_state_buffer = torch.empty(
                (self._max_num_batched_tokens, self._hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )

        for layer_id in self._layer_ids:
            existing = self._capture_buffers.get(layer_id)
            if (
                existing is not None
                and existing.page_table.device == device
                and existing.page_table.shape[0] >= self._max_num_batched_tokens
                and existing.page_table.shape[1] >= self._topk
            ):
                continue
            self._capture_buffers[layer_id] = _LayerCaptureBuffer(
                page_table=torch.empty(
                    (self._max_num_batched_tokens, self._topk),
                    device=device,
                    dtype=torch.int32,
                ),
                topk_indices=torch.empty(
                    (self._max_num_batched_tokens, self._topk),
                    device=device,
                    dtype=torch.int32,
                ),
                nsa_lens=torch.empty(
                    self._max_num_batched_tokens,
                    device=device,
                    dtype=torch.int32,
                ),
                cache_lens=torch.empty(
                    self._max_num_batched_tokens,
                    device=device,
                    dtype=torch.int32,
                ),
                req_ids=torch.empty(
                    self._max_num_batched_tokens,
                    device=device,
                    dtype=torch.int32,
                ),
                token_ids=torch.empty(
                    self._max_num_batched_tokens,
                    device=device,
                    dtype=torch.long,
                ),
            )

    def _flat_packed_cache(self, kv_layer: torch.Tensor) -> torch.Tensor | None:
        if kv_layer.ndim != 3:
            return None
        kv_u8 = (
            kv_layer if kv_layer.dtype == torch.uint8 else kv_layer.view(torch.uint8)
        )
        row_width = int(kv_u8.shape[-1])
        if row_width != self._expected_row_width:
            if row_width not in self._warned_row_widths:
                self._warned_row_widths.add(row_width)
                logger.warning(
                    "Skipping CausalCascade sparse-MLA rows with packed width %d; "
                    "expected %d for fp8_ds_mla.",
                    row_width,
                    self._expected_row_width,
                )
            return None
        return kv_u8.reshape(-1, row_width)

    @staticmethod
    def _is_cuda_graph_capturing(tensor: torch.Tensor) -> bool:
        return bool(tensor.is_cuda and torch.cuda.is_current_stream_capturing())
