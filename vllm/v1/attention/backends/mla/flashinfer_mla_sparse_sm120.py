# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 implementation variant for ``FLASHINFER_MLA_SPARSE_SM120``."""

from typing import TYPE_CHECKING, cast

import torch

from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_dcp_global_index_to_local_index,
    triton_convert_req_index_to_global_index,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer


def _kv_scale_format_for_model(model_type: str | None) -> str:
    if model_type is not None and model_type.startswith("glm"):
        return "arbitrary_fp32"
    return "pow2_fp32"


class FlashInferMLASparseSM120Impl(SparseMLAAttentionImpl[FlashInferMLASparseMetadata]):
    """SM120 FlashInfer sparse-MLA implementation."""

    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 does not support alibi_slopes / "
                "sliding_window / logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 only supports decoder self-attention"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        if self.kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla "
                f"KV cache layout; got kv_cache_dtype={kv_cache_dtype!r}."
            )

        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        model_type = None
        if vllm_config.model_config is not None:
            model_type = getattr(
                vllm_config.model_config.hf_text_config, "model_type", None
            )
        self.kv_scale_format = _kv_scale_format_for_model(model_type)

        topk_indices_buffer = mla_args.get("topk_indices_buffer")
        if indexer is not None:
            topk_indices_buffer = indexer.topk_indices_buffer
        if topk_indices_buffer is None:
            raise ValueError(
                "FLASHINFER_MLA_SPARSE_SM120 requires sparse-MLA top-k indices "
                "from an indexer or a shared topk_indices_buffer."
            )
        self.topk_indices_buffer: torch.Tensor = topk_indices_buffer
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's "
                "sparse MLA decode API."
            )
        assert self.topk_indices_buffer is not None

        self.supports_quant_query_input = False
        self.supports_dcp_quant_query_input = False

        parallel_config = vllm_config.parallel_config
        self.pcp_world_size = 1
        self.pcp_rank = 0
        self.dcp_world_size = int(parallel_config.decode_context_parallel_size)
        self.dcp_rank = 0
        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_dcp_group

            dcp_group = get_dcp_group()
            self.dcp_rank = dcp_group.rank_in_group
            if dcp_group.world_size != self.dcp_world_size:
                raise RuntimeError(
                    "FLASHINFER_MLA_SPARSE_SM120 DCP group size "
                    f"{dcp_group.world_size} does not match configured "
                    f"decode_context_parallel_size={self.dcp_world_size}"
                )
        self.total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        self.total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        self.cp_kv_cache_interleave_size = (
            parallel_config.cp_kv_cache_interleave_size
        )
        self.need_to_return_lse_for_decode = self.dcp_world_size > 1

        # Allocate before memory profiling so KV sizing accounts for FlashInfer's
        # TRTLLM sparse MLA scratch instead of discovering it at first decode.
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            self._workspace_buffer: torch.Tensor | None = _get_workspace_buffer(device)
        else:
            self._workspace_buffer = None

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]
        num_actual_heads = q.shape[1]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        if self.dcp_world_size > 1:
            seq_lens = torch.empty(
                num_actual_toks, dtype=torch.int32, device=q.device
            )
            topk_indices_physical, seq_lens = (
                triton_convert_dcp_global_index_to_local_index(
                    attn_metadata.req_id_per_token[:num_actual_toks],
                    attn_metadata.block_table,
                    topk_indices,
                    dcp_world_size=self.dcp_world_size,
                    dcp_rank=self.dcp_rank,
                    cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
                    BLOCK_SIZE=attn_metadata.block_size,
                    NUM_TOPK_TOKENS=topk_indices.shape[1],
                    valid_counts=seq_lens,
                )
            )
        else:
            topk_indices_physical = cast(
                torch.Tensor,
                triton_convert_req_index_to_global_index(
                    attn_metadata.req_id_per_token[:num_actual_toks],
                    attn_metadata.block_table,
                    topk_indices,
                    BLOCK_SIZE=attn_metadata.block_size,
                    NUM_TOPK_TOKENS=topk_indices.shape[1],
                ),
            )
            seq_lens = None

        output = q.new_empty(
            (num_actual_toks, num_actual_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )
        lse = (
            q.new_empty((num_actual_toks, num_actual_heads), dtype=torch.float32)
            if self.need_to_return_lse_for_decode
            else None
        )

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        ret = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=seq_lens,
            max_seq_len=attn_metadata.topk_tokens,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=attn_metadata.topk_tokens,
            kv_scale_format=self.kv_scale_format,
            lse=None if lse is None else lse.unsqueeze(1),
            return_lse=self.need_to_return_lse_for_decode,
        )
        if not self.need_to_return_lse_for_decode:
            return ret.squeeze(1), None
        if isinstance(ret, tuple):
            out, lse = ret
        else:
            out = ret
            assert lse is not None
        lse = lse.reshape(num_actual_toks, -1)[:, :num_actual_heads].contiguous()
        return out.squeeze(1), lse
