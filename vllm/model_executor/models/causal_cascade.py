# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B009, E501, F821
from typing import Any, ClassVar

import torch
import torch.nn.functional as F  # noqa: N812

# The training architecture is the source of truth; this module supplies only
# vLLM-specific loading and live inference plumbing.
from glmflash.models.dflash_sparse_mla.config import (
    DFlashSparseMLASpeculatorConfig as CanonicalSparseMLAConfig,
)
from glmflash.models.dflash_sparse_mla.core import (
    DFlashSparseMLADraftModel as CanonicalSparseMLADraftModel,
)
from glmflash.models.dflash_sparse_mla.core import (
    SparseMLACrossAttention as CanonicalSparseMLACrossAttention,
)
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm

from vllm.config import SpeculativeConfig, VllmConfig, replace
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    get_masked_input_and_mask,
)
from vllm.model_executor.models.utils import AutoWeightsLoader


class DFlashSparseMLASpeculatorConfig(PretrainedConfig):
    model_type = "causal_cascade"


class DraftVocabMixin:
    def load_vocab_mappings(
        self,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
    ) -> None:
        if t2d is not None:
            self.t2d = t2d
        if d2t is not None:
            self.d2t = d2t


class SpeculatorModel(nn.Module):
    config: PretrainedConfig

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.config = config

    @classmethod
    def register(cls, _name: str):
        def decorator(model_cls):
            return model_cls

        return decorator

    def post_init(self) -> None:
        pass


def dflash_loss_decay(
    position_ids: torch.Tensor,
    gamma: float = 4.0,
) -> torch.Tensor:
    return 1.0 / torch.pow(position_ids + 1.0, gamma)


def compute_accuracy_multi_step(
    pred_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_idx: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    correct = torch.zeros(block_size, device=pred_ids.device, dtype=torch.float32)
    total = torch.zeros(block_size, device=pred_ids.device, dtype=torch.float32)
    for pos in range(block_size):
        mask = loss_mask & (pos_idx == pos)
        total[pos] = mask.sum()
        correct[pos] = ((pred_ids == target_token_ids) & mask).sum()
    return correct, total


def load_model_layers(
    _names: list[str],
    _model_name_or_path: str,
) -> dict[str, torch.Tensor]:
    raise RuntimeError(
        "CausalCascade serving should populate frozen verifier MLA weights "
        "from the target model or an exported checkpoint, not by importing the "
        "training-side speculators loader."
    )


def _get_text_config(config: PretrainedConfig) -> PretrainedConfig:
    return config.text_config if hasattr(config, "text_config") else config


def _coerce_pretrained_config(value: Any, *, field_name: str) -> PretrainedConfig:
    if isinstance(value, PretrainedConfig):
        return value
    if isinstance(value, dict):
        return PretrainedConfig.from_dict(value)
    raise TypeError(
        f"{field_name} must be a PretrainedConfig or dict, got {type(value).__name__}"
    )


def _derive_mla_dims(verifier_name_or_path: str) -> dict[str, int]:
    verifier_config = _get_text_config(
        AutoConfig.from_pretrained(verifier_name_or_path, trust_remote_code=True)
    )
    qk_rope_head_dim = int(verifier_config.qk_rope_head_dim)
    model_type = str(getattr(verifier_config, "model_type", ""))
    head_dim = int(getattr(verifier_config, "head_dim", qk_rope_head_dim))
    if model_type.startswith("glm") and qk_rope_head_dim == head_dim == 192:
        # Match vLLM's GLM-5.2 config normalization; transformers exposes the
        # total qk head dim here, while the MLA cache row contains 64 RoPE dims.
        qk_rope_head_dim = 64

    if hasattr(verifier_config, "compress_ratios"):
        head_dim = int(verifier_config.head_dim)
        return {
            "mla_kv_lora_rank": head_dim,
            "mla_qk_rope_head_dim": qk_rope_head_dim,
            "mla_qk_nope_head_dim": max(head_dim - qk_rope_head_dim, 0),
            "mla_q_lora_rank": int(verifier_config.hidden_size),
            "mla_v_head_dim": head_dim,
            "mla_num_heads": int(verifier_config.num_attention_heads),
        }

    qk_head_dim = int(
        getattr(
            verifier_config,
            "qk_head_dim",
            getattr(verifier_config, "head_dim", qk_rope_head_dim),
        )
    )
    qk_nope_head_dim = int(
        getattr(
            verifier_config,
            "qk_nope_head_dim",
            qk_head_dim - qk_rope_head_dim,
        )
    )
    return {
        "mla_kv_lora_rank": int(verifier_config.kv_lora_rank),
        "mla_qk_rope_head_dim": qk_rope_head_dim,
        "mla_qk_nope_head_dim": qk_nope_head_dim,
        "mla_q_lora_rank": int(
            getattr(verifier_config, "q_lora_rank", verifier_config.hidden_size)
        ),
        "mla_v_head_dim": int(verifier_config.v_head_dim),
        "mla_num_heads": int(verifier_config.num_attention_heads),
    }


def dequantize_fp8_ds_mla_rows(
    packed_rows: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if packed_rows.dtype != torch.uint8:
        raise ValueError(
            f"fp8_ds_mla packed rows must be uint8, got {packed_rows.dtype}"
        )
    if packed_rows.shape[-1] != 656:
        raise ValueError(
            "fp8_ds_mla packed rows must have width 656 bytes, got "
            f"{packed_rows.shape[-1]}"
        )

    prefix = packed_rows.shape[:-1]
    latent_fp8 = packed_rows[..., :512].contiguous().view(torch.float8_e4m3fn)
    latent = latent_fp8.to(torch.float32).reshape(*prefix, 4, 128)
    scales = packed_rows[..., 512:528].contiguous().view(torch.float32)
    latent = latent * scales.unsqueeze(-1)
    rope = packed_rows[..., 528:656].contiguous().view(torch.bfloat16).to(torch.float32)
    rows = torch.cat([latent.reshape(*prefix, 512), rope], dim=-1)
    return rows.to(dtype)


class SparseMLABlockSelfAttention(nn.Module):
    def __init__(self, config: PretrainedConfig, *, non_causal: bool) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.non_causal = non_causal
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if key_value_states is None:
            key_value_states = hidden_states

        batch_size, seq_len, _ = hidden_states.shape
        kv_len = key_value_states.shape[1]
        q = self.q_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        k = self.k_proj(key_value_states).view(
            batch_size,
            kv_len,
            self.num_heads,
            self.head_dim,
        )
        v = self.v_proj(key_value_states).view(
            batch_size,
            kv_len,
            self.num_heads,
            self.head_dim,
        )
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if attn_mask is None and not self.non_causal:
            diagonal = kv_len - seq_len
            attn_mask = torch.ones(
                seq_len,
                kv_len,
                device=hidden_states.device,
                dtype=torch.bool,
            ).tril(diagonal=diagonal)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        return self.o_proj(out)


class SparseMLAQueryMLPBlock(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.norm = Qwen3RMSNorm(hidden_size, eps=eps)  # type: ignore[arg-type]
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.down_proj.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.norm(hidden_states)
        residual = F.silu(self.gate_proj(residual)) * self.up_proj(residual)
        return self.down_proj(residual)


def _tensor_rms(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float()
    if hasattr(value, "to_local"):
        value = value.to_local()
    return value.square().mean().sqrt()


class SparseMLACrossAttention(nn.Module):
    def __init__(
        self,
        config: DFlashSparseMLASpeculatorConfig,
        verifier_layer_id: int | None = None,
        active_position_start: int = 0,
    ) -> None:
        super().__init__()
        hidden_size = int(config.transformer_layer_config.hidden_size)
        eps = float(config.transformer_layer_config.rms_norm_eps)
        self.rms_norm_eps = eps
        self.num_heads = config.mla_num_heads
        self.kv_lora_rank = config.mla_kv_lora_rank
        self.qk_rope_head_dim = config.mla_qk_rope_head_dim
        self.qk_nope_head_dim = config.mla_qk_nope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.q_lora_rank = config.mla_q_lora_rank
        self.v_head_dim = config.mla_v_head_dim
        self.q_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.cross_attention_impl = config.cross_attention_impl
        if self.cross_attention_impl not in {"learned", "target_compatible"}:
            raise ValueError(
                "cross_attention_impl must be 'learned' or 'target_compatible', "
                f"got {self.cross_attention_impl!r}"
            )
        attention_scale_dim = config.mla_attention_scale_dim
        if attention_scale_dim is None:
            attention_scale_dim = (
                self.qk_head_dim
                if self.cross_attention_impl == "target_compatible"
                else self.q_head_dim
            )
        self.attention_scale_dim = int(attention_scale_dim)
        if self.attention_scale_dim <= 0:
            raise ValueError("mla_attention_scale_dim must be > 0 when set")
        self.scale = self.attention_scale_dim**-0.5
        self.num_sink_tokens = int(config.num_attention_sink_tokens)
        self.block_size = int(config.block_size)
        self.verifier_layer_id = verifier_layer_id
        self.active_position_start = int(active_position_start)
        if (
            self.active_position_start < 0
            or self.active_position_start > self.block_size
        ):
            raise ValueError(
                "active_position_start must satisfy 0 <= start <= block_size; "
                f"got {self.active_position_start} for block_size={self.block_size}"
            )
        self.active_position_slots = self.block_size - self.active_position_start
        self.query_position_adapter_rank = int(config.query_position_adapter_rank)
        self.query_position_output_adapter_rank = int(
            config.query_position_output_adapter_rank
        )
        self.query_layer_adapter_rank = int(config.query_layer_adapter_rank)
        self.query_anchor_conditioned = bool(config.query_anchor_conditioned)
        self.query_mlp_blocks = nn.ModuleList(
            [
                SparseMLAQueryMLPBlock(hidden_size, eps=eps)
                for _ in range(int(config.query_mlp_blocks))
            ]
        )
        self.rope_theta = self._derive_rope_theta(config.transformer_layer_config)
        self.rope_interleave = bool(
            getattr(config.transformer_layer_config, "rope_interleave", True)
        )
        self.query_output_size = (
            self.num_heads * self.qk_head_dim
            if self.cross_attention_impl == "target_compatible"
            else self.num_heads * self.q_head_dim
        )
        self._last_debug_metrics: dict[str, torch.Tensor] = {}

        if self.cross_attention_impl == "learned":
            self.q_proj = nn.Linear(
                hidden_size,
                self.num_heads * self.q_head_dim,
                bias=False,
            )
            self.v_up_proj = nn.Parameter(
                torch.empty(self.num_heads, self.kv_lora_rank, self.v_head_dim)
            )
            self.o_proj = nn.Linear(
                self.num_heads * self.v_head_dim,
                hidden_size,
                bias=False,
            )
            self.register_buffer("target_q_a_proj_weight", None, persistent=False)
            self.register_buffer("target_q_a_layernorm_weight", None, persistent=False)
            self.register_buffer("target_q_b_proj_weight", None, persistent=False)
            self.register_buffer("target_kv_b_proj_weight", None, persistent=False)
            self.register_buffer("target_o_proj_weight", None, persistent=False)
        else:
            self.q_proj = None
            self.register_parameter("v_up_proj", None)
            self.o_proj = None
            self.register_buffer(
                "target_q_a_proj_weight",
                torch.empty(self.q_lora_rank, hidden_size),
                persistent=False,
            )
            self.register_buffer(
                "target_q_a_layernorm_weight",
                torch.empty(self.q_lora_rank),
                persistent=False,
            )
            self.register_buffer(
                "target_q_b_proj_weight",
                torch.empty(self.num_heads * self.qk_head_dim, self.q_lora_rank),
                persistent=False,
            )
            self.register_buffer(
                "target_kv_b_proj_weight",
                torch.empty(
                    self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
                    self.kv_lora_rank,
                ),
                persistent=False,
            )
            self.register_buffer(
                "target_o_proj_weight",
                torch.empty(hidden_size, self.num_heads * self.v_head_dim),
                persistent=False,
            )
            self._reset_target_compatible_weights()
        if self.query_position_adapter_rank < 0:
            raise ValueError("query_position_adapter_rank must be >= 0")
        if self.query_position_output_adapter_rank < 0:
            raise ValueError("query_position_output_adapter_rank must be >= 0")
        if self.query_layer_adapter_rank < 0:
            raise ValueError("query_layer_adapter_rank must be >= 0")
        if self.query_position_adapter_rank:
            rank = self.query_position_adapter_rank
            self.query_position_norm = Qwen3RMSNorm(hidden_size, eps=eps)  # type: ignore[arg-type]
            self.query_position_down = nn.Parameter(
                torch.empty(self.active_position_slots, hidden_size, rank)
            )
            self.query_position_up = nn.Parameter(
                torch.empty(self.active_position_slots, rank, hidden_size)
            )
            nn.init.xavier_uniform_(self.query_position_down)
            nn.init.zeros_(self.query_position_up)
        else:
            self.query_position_norm = None
            self.register_parameter("query_position_down", None)
            self.register_parameter("query_position_up", None)
        if self.query_position_output_adapter_rank:
            rank = self.query_position_output_adapter_rank
            self.query_position_output_norm = Qwen3RMSNorm(hidden_size, eps=eps)  # type: ignore[arg-type]
            self.query_position_output_down = nn.Parameter(
                torch.empty(self.active_position_slots, hidden_size, rank)
            )
            self.query_position_output_up = nn.Parameter(
                torch.empty(self.active_position_slots, rank, self.query_output_size)
            )
            nn.init.xavier_uniform_(self.query_position_output_down)
            nn.init.zeros_(self.query_position_output_up)
        else:
            self.query_position_output_norm = None
            self.register_parameter("query_position_output_down", None)
            self.register_parameter("query_position_output_up", None)
        if self.query_layer_adapter_rank:
            rank = self.query_layer_adapter_rank
            self.query_layer_norm = Qwen3RMSNorm(hidden_size, eps=eps)  # type: ignore[arg-type]
            self.query_layer_down = nn.Linear(hidden_size, rank, bias=False)
            self.query_layer_up = nn.Linear(rank, hidden_size, bias=False)
            nn.init.zeros_(self.query_layer_up.weight)
        else:
            self.query_layer_norm = None
            self.query_layer_down = None
            self.query_layer_up = None
        if self.query_anchor_conditioned:
            self.query_anchor_norm = Qwen3RMSNorm(hidden_size, eps=eps)  # type: ignore[arg-type]
            self.query_anchor_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.zeros_(self.query_anchor_proj.weight)
        else:
            self.query_anchor_norm = None
            self.query_anchor_proj = None
        if self.num_sink_tokens < 0:
            raise ValueError("num_attention_sink_tokens must be >= 0")
        if self.num_sink_tokens:
            self.sink_k = nn.Parameter(
                torch.empty(self.num_sink_tokens, self.q_head_dim)
            )
            self.sink_v = nn.Parameter(
                torch.empty(self.num_sink_tokens, self.kv_lora_rank)
            )
        else:
            self.register_parameter("sink_k", None)
            self.register_parameter("sink_v", None)
        if self.v_up_proj is not None:
            nn.init.xavier_uniform_(self.v_up_proj)
        if self.num_sink_tokens:
            nn.init.zeros_(self.sink_k)
            nn.init.zeros_(self.sink_v)

    @staticmethod
    def _derive_rope_theta(config: PretrainedConfig) -> float:
        rope_parameters = getattr(config, "rope_parameters", None)
        if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
            return float(rope_parameters["rope_theta"])
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict) and "rope_theta" in rope_scaling:
            return float(rope_scaling["rope_theta"])
        return float(getattr(config, "rope_theta", 10000.0))

    def _reset_target_compatible_weights(self) -> None:
        assert self.target_q_a_proj_weight is not None
        assert self.target_q_a_layernorm_weight is not None
        assert self.target_q_b_proj_weight is not None
        assert self.target_kv_b_proj_weight is not None
        assert self.target_o_proj_weight is not None
        nn.init.xavier_uniform_(self.target_q_a_proj_weight)
        nn.init.ones_(self.target_q_a_layernorm_weight)
        nn.init.xavier_uniform_(self.target_q_b_proj_weight)
        nn.init.xavier_uniform_(self.target_kv_b_proj_weight)
        nn.init.xavier_uniform_(self.target_o_proj_weight)

    def load_target_compatible_weights(
        self,
        weights: dict[str, torch.Tensor],
    ) -> None:
        if self.cross_attention_impl != "target_compatible":
            return
        expected = {
            "q_a_proj.weight": self.target_q_a_proj_weight,
            "q_a_layernorm.weight": self.target_q_a_layernorm_weight,
            "q_b_proj.weight": self.target_q_b_proj_weight,
            "kv_b_proj.weight": self.target_kv_b_proj_weight,
            "o_proj.weight": self.target_o_proj_weight,
        }
        with torch.no_grad():
            for name, target in expected.items():
                assert target is not None
                source = weights[name]
                if tuple(source.shape) != tuple(target.shape):
                    raise ValueError(
                        "Target-compatible MLA weight shape mismatch for "
                        f"layer {self.verifier_layer_id} {name}: got "
                        f"{tuple(source.shape)}, expected {tuple(target.shape)}"
                    )
                target.copy_(source.to(device=target.device, dtype=target.dtype))

    def _adapter_position_slice(
        self,
        *,
        position_start: int,
        seq_len: int,
    ) -> tuple[int, int]:
        position_end = position_start + seq_len
        if position_start < 0 or position_end > self.block_size:
            raise ValueError(
                "Sparse MLA query position range is outside block_size: "
                f"start={position_start}, len={seq_len}, block_size={self.block_size}"
            )
        if position_start < self.active_position_start:
            raise ValueError(
                "Sparse MLA query position range starts before this layer's "
                f"active range: start={position_start}, "
                f"active_start={self.active_position_start}"
            )
        adapter_start = position_start - self.active_position_start
        adapter_end = adapter_start + seq_len
        if adapter_end > self.active_position_slots:
            raise ValueError(
                "Sparse MLA query position range exceeds this layer's active "
                f"adapter slots: start={position_start}, len={seq_len}, "
                f"active_start={self.active_position_start}, "
                f"active_slots={self.active_position_slots}"
            )
        return adapter_start, adapter_end

    def _build_query_input(
        self,
        hidden_states: torch.Tensor,
        *,
        anchor_hidden_state: torch.Tensor | None,
        position_start: int,
    ) -> torch.Tensor:
        query_input = hidden_states
        seq_len = hidden_states.shape[1]
        adapter_start, adapter_end = self._adapter_position_slice(
            position_start=position_start,
            seq_len=seq_len,
        )

        if self.query_position_adapter_rank:
            assert self.query_position_norm is not None
            assert self.query_position_down is not None
            assert self.query_position_up is not None
            normed = self.query_position_norm(hidden_states)
            down = self.query_position_down[adapter_start:adapter_end].to(
                dtype=normed.dtype
            )
            up = self.query_position_up[adapter_start:adapter_end].to(
                dtype=normed.dtype
            )
            delta = torch.einsum("bsh,shr->bsr", normed, down)
            delta = torch.einsum("bsr,srh->bsh", delta, up)
            query_input = query_input + delta

        if self.query_layer_adapter_rank:
            assert self.query_layer_norm is not None
            assert self.query_layer_down is not None
            assert self.query_layer_up is not None
            normed = self.query_layer_norm(hidden_states)
            query_input = query_input + self.query_layer_up(
                self.query_layer_down(normed)
            )

        for block in self.query_mlp_blocks:
            query_input = query_input + block(query_input)

        if self.query_anchor_conditioned:
            if anchor_hidden_state is None:
                raise ValueError(
                    "query_anchor_conditioned requires anchor_hidden_state"
                )
            assert self.query_anchor_norm is not None
            assert self.query_anchor_proj is not None
            anchor_delta = self.query_anchor_proj(
                self.query_anchor_norm(anchor_hidden_state)
            ).unsqueeze(1)
            query_input = query_input + anchor_delta.to(query_input.dtype)

        return query_input

    def _apply_query_output_adapter(
        self,
        q_flat: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        position_start: int,
    ) -> torch.Tensor:
        if not self.query_position_output_adapter_rank:
            return q_flat
        assert self.query_position_output_norm is not None
        assert self.query_position_output_down is not None
        assert self.query_position_output_up is not None

        seq_len = hidden_states.shape[1]
        adapter_start, adapter_end = self._adapter_position_slice(
            position_start=position_start,
            seq_len=seq_len,
        )
        normed = self.query_position_output_norm(hidden_states)
        down = self.query_position_output_down[adapter_start:adapter_end].to(
            dtype=normed.dtype
        )
        up = self.query_position_output_up[adapter_start:adapter_end].to(
            dtype=normed.dtype
        )
        adapted_positions = []
        for pos in range(seq_len):
            low_rank = F.linear(normed[:, pos], down[pos].transpose(0, 1))
            delta = F.linear(low_rank, up[pos].transpose(0, 1))
            adapted_positions.append(q_flat[:, pos] + delta.to(q_flat.dtype))
        return torch.stack(adapted_positions, dim=1)

    def _apply_query_rope(
        self,
        q_rope: torch.Tensor,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.qk_rope_head_dim == 0:
            return q_rope
        if position_ids is None:
            raise ValueError(
                "target_compatible sparse-MLA cross-attention requires position_ids"
            )
        batch_size, seq_len, _num_heads, rope_dim = q_rope.shape
        if rope_dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {rope_dim}")
        if position_ids.ndim == 1:
            if position_ids.shape[0] != seq_len:
                raise ValueError(
                    "position_ids length mismatch: "
                    f"got {position_ids.shape[0]}, expected {seq_len}"
                )
            positions = position_ids.view(1, seq_len).expand(batch_size, -1)
        elif position_ids.ndim == 2:
            if position_ids.shape != (batch_size, seq_len):
                raise ValueError(
                    "position_ids shape mismatch: "
                    f"got {tuple(position_ids.shape)}, "
                    f"expected {(batch_size, seq_len)}"
                )
            positions = position_ids
        else:
            raise ValueError(
                f"position_ids must be rank 1 or 2, got rank {position_ids.ndim}"
            )

        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(
                    0,
                    rope_dim,
                    2,
                    device=q_rope.device,
                    dtype=torch.float32,
                )
                / rope_dim
            )
        )
        freqs = positions.to(device=q_rope.device, dtype=torch.float32).unsqueeze(
            -1
        ) * inv_freq.view(1, 1, -1)
        cos = freqs.cos().unsqueeze(2).to(q_rope.dtype)
        sin = freqs.sin().unsqueeze(2).to(q_rope.dtype)
        if self.rope_interleave:
            even = q_rope[..., 0::2]
            odd = q_rope[..., 1::2]
            rotated = torch.empty_like(q_rope)
            rotated[..., 0::2] = even * cos - odd * sin
            rotated[..., 1::2] = odd * cos + even * sin
            return rotated
        first = q_rope[..., : rope_dim // 2]
        second = q_rope[..., rope_dim // 2 :]
        return torch.cat(
            [
                first * cos - second * sin,
                second * cos + first * sin,
            ],
            dim=-1,
        )

    def _target_q_a_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.target_q_a_layernorm_weight is not None
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.rms_norm_eps).to(
            hidden_states.dtype
        )
        return hidden_states * self.target_q_a_layernorm_weight.to(hidden_states.dtype)

    def _build_learned_query(
        self,
        query_input: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        position_start: int,
    ) -> torch.Tensor:
        assert self.q_proj is not None
        batch_size, seq_len, _ = hidden_states.shape
        q_flat = self.q_proj(query_input)
        q_flat = self._apply_query_output_adapter(
            q_flat,
            hidden_states,
            position_start=position_start,
        )
        q = q_flat.view(batch_size, seq_len, self.num_heads, self.q_head_dim)
        return q.transpose(1, 2)

    def _build_target_compatible_query(
        self,
        query_input: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor | None,
        position_start: int,
    ) -> torch.Tensor:
        assert self.target_q_a_proj_weight is not None
        assert self.target_q_b_proj_weight is not None
        assert self.target_kv_b_proj_weight is not None
        batch_size, seq_len, _ = hidden_states.shape
        q_low_rank = F.linear(
            query_input,
            self.target_q_a_proj_weight.to(query_input.dtype),
        )
        q_low_rank = self._target_q_a_norm(q_low_rank)
        q_flat = F.linear(
            q_low_rank,
            self.target_q_b_proj_weight.to(q_low_rank.dtype),
        )
        q_flat = self._apply_query_output_adapter(
            q_flat,
            hidden_states,
            position_start=position_start,
        )
        q_native = q_flat.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        q_nope = q_native[..., : self.qk_nope_head_dim]
        q_rope = q_native[..., self.qk_nope_head_dim :]
        q_rope = self._apply_query_rope(q_rope, position_ids)

        kv_b = self.target_kv_b_proj_weight.to(q_native.dtype).view(
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        k_up = kv_b[:, : self.qk_nope_head_dim, :]
        q_latent = torch.einsum("bshd,hdl->bshl", q_nope, k_up)
        q = torch.cat([q_latent, q_rope], dim=-1)
        return q.transpose(1, 2)

    def _expand_attended_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.cross_attention_impl == "target_compatible":
            assert self.target_kv_b_proj_weight is not None
            kv_b = self.target_kv_b_proj_weight.to(latent.dtype).view(
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
                self.kv_lora_rank,
            )
            v_up = kv_b[:, self.qk_nope_head_dim :, :]
            return torch.einsum("bhsl,hvl->bhsv", latent, v_up)
        assert self.v_up_proj is not None
        return torch.einsum(
            "bhsl,hlv->bhsv",
            latent,
            self.v_up_proj.to(latent.dtype),
        )

    def _project_cross_output(self, expanded_v: torch.Tensor) -> torch.Tensor:
        if self.cross_attention_impl == "target_compatible":
            assert self.target_o_proj_weight is not None
            return F.linear(expanded_v, self.target_o_proj_weight.to(expanded_v.dtype))
        assert self.o_proj is not None
        return self.o_proj(expanded_v)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mla_cache_rows: torch.Tensor,
        *,
        mla_cache_valid_mask: torch.Tensor | None = None,
        anchor_hidden_state: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        position_start: int = 0,
    ) -> torch.Tensor:
        # mla_cache_rows: [B, topk, kv_lora_rank + qk_rope_head_dim]
        batch_size, seq_len, _ = hidden_states.shape
        if mla_cache_rows.shape[-1] != self.q_head_dim:
            raise ValueError(
                "Sparse MLA cache row width mismatch: "
                f"got {mla_cache_rows.shape[-1]}, expected {self.q_head_dim}"
            )
        valid_mask = None
        if mla_cache_valid_mask is not None:
            if mla_cache_valid_mask.shape != mla_cache_rows.shape[:2]:
                raise ValueError(
                    "Sparse MLA cache valid mask shape mismatch: got "
                    f"{tuple(mla_cache_valid_mask.shape)}, expected "
                    f"{tuple(mla_cache_rows.shape[:2])}"
                )
            valid_mask = mla_cache_valid_mask.to(
                device=hidden_states.device,
                dtype=torch.bool,
            )
            if not torch.all(valid_mask.any(dim=-1)):
                raise ValueError("Sparse MLA cache valid mask has an empty row")

        query_input = self._build_query_input(
            hidden_states,
            anchor_hidden_state=anchor_hidden_state,
            position_start=position_start,
        )
        if self.cross_attention_impl == "target_compatible":
            q = self._build_target_compatible_query(
                query_input,
                hidden_states,
                position_ids=position_ids,
                position_start=position_start,
            )
        else:
            q = self._build_learned_query(
                query_input,
                hidden_states,
                position_start=position_start,
            )
        k = mla_cache_rows.unsqueeze(1)  # [B, 1, K, D]
        v = mla_cache_rows[..., : self.kv_lora_rank].unsqueeze(1)  # [B, 1, K, L]
        if self.num_sink_tokens:
            assert self.sink_k is not None
            assert self.sink_v is not None
            sink_k = self.sink_k.to(k.dtype).view(1, 1, self.num_sink_tokens, -1)
            sink_v = self.sink_v.to(v.dtype).view(1, 1, self.num_sink_tokens, -1)
            sink_k = sink_k.expand(batch_size, 1, -1, -1)
            sink_v = sink_v.expand(batch_size, 1, -1, -1)
            k = torch.cat([k, sink_k], dim=2)
            v = torch.cat([v, sink_v], dim=2)
            if valid_mask is not None:
                sink_mask = torch.ones(
                    batch_size,
                    self.num_sink_tokens,
                    device=valid_mask.device,
                    dtype=torch.bool,
                )
                valid_mask = torch.cat([valid_mask, sink_mask], dim=-1)

        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale
        if valid_mask is not None:
            scores = scores.masked_fill(
                ~valid_mask.view(batch_size, 1, 1, -1),
                torch.finfo(scores.dtype).min,
            )
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        latent = torch.matmul(probs, v.to(q.dtype))  # [B, H, S, L]
        expanded_v = self._expand_attended_latent(latent)
        expanded_v = expanded_v.transpose(1, 2).reshape(
            batch_size, seq_len, self.num_heads * self.v_head_dim
        )
        projected = self._project_cross_output(expanded_v)

        probs_f = probs.detach().float()
        if hasattr(probs_f, "to_local"):
            probs_f = probs_f.to_local()
        entropy = -(probs_f * probs_f.clamp_min(1e-20).log()).sum(dim=-1).mean()
        self._last_debug_metrics = {
            "attention_entropy": entropy,
            "attention_top1_prob": probs_f.max(dim=-1).values.mean(),
            "query_rms": _tensor_rms(q),
            "latent_rms": _tensor_rms(latent),
            "output_rms": _tensor_rms(projected),
        }
        return projected


def _apply_position_gate(
    hidden_states: torch.Tensor,
    gate_logits: torch.Tensor | None,
    *,
    position_start: int = 0,
) -> torch.Tensor:
    if gate_logits is None:
        return hidden_states
    position_end = position_start + hidden_states.shape[1]
    gate = torch.sigmoid(gate_logits[position_start:position_end]).view(1, -1, 1)
    return hidden_states * gate.to(hidden_states.dtype)


def _record_gate_metrics(
    metrics: dict[str, torch.Tensor],
    prefix: str,
    gate_logits: torch.Tensor | None,
) -> None:
    if gate_logits is None:
        return
    gate = torch.sigmoid(gate_logits.detach().float())
    if hasattr(gate, "to_local"):
        gate = gate.to_local()
    metrics[f"gates/{prefix}/mean"] = gate.mean()


class DFlashSparseMLADecoderLayer(nn.Module):
    def __init__(
        self,
        config: DFlashSparseMLASpeculatorConfig,
        verifier_layer_id: int | None = None,
        active_position_start: int = 0,
    ) -> None:
        super().__init__()
        tl_config = config.transformer_layer_config
        self.input_layernorm = Qwen3RMSNorm(
            tl_config.hidden_size, eps=tl_config.rms_norm_eps
        )  # type: ignore[arg-type]
        self.self_attn = SparseMLABlockSelfAttention(
            tl_config,
            non_causal=config.non_causal_block_attention,
        )
        self.cross_layernorm = Qwen3RMSNorm(
            tl_config.hidden_size, eps=tl_config.rms_norm_eps
        )  # type: ignore[arg-type]
        self.cross_attn = SparseMLACrossAttention(
            config,
            verifier_layer_id=verifier_layer_id,
            active_position_start=active_position_start,
        )
        self.use_cross_attention_gate = bool(config.cross_attention_gate)
        self.cross_attention_residual_scale = float(
            getattr(config, "cross_attention_residual_scale", 1.0)
        )
        self.mlp_residual_scale = float(getattr(config, "mlp_residual_scale", 1.0))
        if self.use_cross_attention_gate:
            self.cross_attention_gate = nn.Parameter(
                torch.full(
                    (config.block_size,),
                    float(config.cross_attention_gate_init),
                )
            )
        else:
            self.register_parameter("cross_attention_gate", None)
        self.use_residual_branch_gate = bool(config.residual_branch_gate)
        if self.use_residual_branch_gate:
            self.self_attn_residual_gate = nn.Parameter(
                torch.full(
                    (config.block_size,),
                    float(config.residual_branch_gate_init),
                )
            )
            self.cross_attn_residual_gate = nn.Parameter(
                torch.full(
                    (config.block_size,),
                    float(config.residual_branch_gate_init),
                )
            )
            self.mlp_residual_gate = nn.Parameter(
                torch.full(
                    (config.block_size,),
                    float(config.residual_branch_gate_init),
                )
            )
        else:
            self.register_parameter("self_attn_residual_gate", None)
            self.register_parameter("cross_attn_residual_gate", None)
            self.register_parameter("mlp_residual_gate", None)
        self.post_attention_layernorm = Qwen3RMSNorm(
            tl_config.hidden_size,
            eps=tl_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.mlp = Qwen3MLP(tl_config)  # type: ignore[arg-type]
        self._last_branch_metrics: dict[str, torch.Tensor] = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        mla_cache_rows: torch.Tensor,
        mla_cache_valid_mask: torch.Tensor | None = None,
        key_value_states: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        position_start: int = 0,
        position_ids: torch.Tensor | None = None,
        anchor_hidden_state: torch.Tensor | None = None,
        ablate_cross_attention: bool = False,
    ) -> torch.Tensor:
        self._last_branch_metrics = {}
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if key_value_states is not None:
            key_value_states = self.input_layernorm(key_value_states)
        self_attn_out = self.self_attn(
            hidden_states,
            key_value_states=key_value_states,
            attn_mask=attn_mask,
        )
        self_attn_out = _apply_position_gate(
            self_attn_out,
            self.self_attn_residual_gate,
            position_start=position_start,
        )
        hidden_states = residual + self_attn_out

        residual = hidden_states
        hidden_states = self.cross_layernorm(hidden_states)
        if ablate_cross_attention:
            cross_out = torch.zeros_like(residual)
            self.cross_attn._last_debug_metrics = {}
        else:
            cross_out = self.cross_attn(
                hidden_states,
                mla_cache_rows,
                mla_cache_valid_mask=mla_cache_valid_mask,
                anchor_hidden_state=anchor_hidden_state,
                position_ids=position_ids,
                position_start=position_start,
            )
        if self.use_cross_attention_gate:
            assert self.cross_attention_gate is not None
            position_end = position_start + hidden_states.shape[1]
            gate = torch.sigmoid(
                self.cross_attention_gate[position_start:position_end]
            ).view(1, -1, 1)
            cross_out = cross_out * gate.to(cross_out.dtype)
        if self.cross_attention_residual_scale != 1.0:
            cross_out = cross_out * self.cross_attention_residual_scale
        cross_out = _apply_position_gate(
            cross_out,
            self.cross_attn_residual_gate,
            position_start=position_start,
        )
        hidden_states = residual + cross_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        mlp_out = _apply_position_gate(
            mlp_out,
            self.mlp_residual_gate,
            position_start=position_start,
        )
        if self.mlp_residual_scale != 1.0:
            mlp_out = mlp_out * self.mlp_residual_scale
        self_attn_rms = _tensor_rms(self_attn_out)
        cross_attn_rms = _tensor_rms(cross_out)
        mlp_rms = _tensor_rms(mlp_out)
        self._last_branch_metrics = {
            "self_attn_rms": self_attn_rms,
            "cross_attn_rms": cross_attn_rms,
            "mlp_rms": mlp_rms,
            "cross_to_self_rms": cross_attn_rms / self_attn_rms.clamp_min(1e-12),
            "cross_to_mlp_rms": cross_attn_rms / mlp_rms.clamp_min(1e-12),
        }
        return residual + mlp_out


class DFlashSparseMLALocalDecoderLayer(nn.Module):
    def __init__(self, config: DFlashSparseMLASpeculatorConfig) -> None:
        super().__init__()
        tl_config = config.transformer_layer_config
        self.input_layernorm = Qwen3RMSNorm(
            tl_config.hidden_size, eps=tl_config.rms_norm_eps
        )  # type: ignore[arg-type]
        self.self_attn = SparseMLABlockSelfAttention(
            tl_config,
            non_causal=config.non_causal_block_attention,
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            tl_config.hidden_size,
            eps=tl_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.mlp = Qwen3MLP(tl_config)  # type: ignore[arg-type]
        self.mlp_residual_scale = float(getattr(config, "mlp_residual_scale", 1.0))
        self._last_branch_metrics: dict[str, torch.Tensor] = {}
        self.use_residual_branch_gate = bool(config.residual_branch_gate)
        if self.use_residual_branch_gate:
            self.self_attn_residual_gate = nn.Parameter(
                torch.full(
                    (config.block_size,),
                    float(config.residual_branch_gate_init),
                )
            )
            self.mlp_residual_gate = nn.Parameter(
                torch.full(
                    (config.block_size,),
                    float(config.residual_branch_gate_init),
                )
            )
        else:
            self.register_parameter("self_attn_residual_gate", None)
            self.register_parameter("mlp_residual_gate", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        position_start: int = 0,
    ) -> torch.Tensor:
        self._last_branch_metrics = {}
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if key_value_states is not None:
            key_value_states = self.input_layernorm(key_value_states)
        self_attn_out = self.self_attn(
            hidden_states,
            key_value_states=key_value_states,
            attn_mask=attn_mask,
        )
        self_attn_out = _apply_position_gate(
            self_attn_out,
            self.self_attn_residual_gate,
            position_start=position_start,
        )
        hidden_states = residual + self_attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        mlp_out = _apply_position_gate(
            mlp_out,
            self.mlp_residual_gate,
            position_start=position_start,
        )
        if self.mlp_residual_scale != 1.0:
            mlp_out = mlp_out * self.mlp_residual_scale
        self_attn_rms = _tensor_rms(self_attn_out)
        mlp_rms = _tensor_rms(mlp_out)
        self._last_branch_metrics = {
            "self_attn_rms": self_attn_rms,
            "mlp_rms": mlp_rms,
            "self_to_mlp_rms": self_attn_rms / mlp_rms.clamp_min(1e-12),
        }
        return residual + mlp_out


class SparseMLAAnchorRankBlock(nn.Module):
    def __init__(self, rank: int, *, eps: float) -> None:
        super().__init__()
        self.norm = Qwen3RMSNorm(rank, eps=eps)
        self.fc1 = nn.Linear(rank, rank * 4, bias=False)
        self.fc2 = nn.Linear(rank * 4, rank, bias=False)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.fc2(F.silu(self.fc1(hidden_states)))
        return residual + hidden_states


class SparseMLAAnchorInitializer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        block_size: int,
        rank: int,
        eps: float,
        kind: str = "low_rank",
        num_blocks: int = 0,
        block_scope: str = "shared",
    ) -> None:
        super().__init__()
        if rank < 0:
            raise ValueError("anchor_delta_rank must be >= 0")
        if kind not in {"low_rank", "dense", "dense_mlp"}:
            raise ValueError(
                "anchor_delta_kind must be 'low_rank', 'dense', or 'dense_mlp'"
            )
        if num_blocks < 0:
            raise ValueError("anchor_delta_num_blocks must be >= 0")
        if block_scope not in {"shared", "per_position"}:
            raise ValueError(
                "anchor_delta_block_scope must be 'shared' or 'per_position'"
            )
        if kind in {"dense", "dense_mlp"} and num_blocks:
            raise ValueError("dense anchor adapters do not support rank-space blocks")
        if kind == "low_rank" and rank == 0 and num_blocks:
            raise ValueError("rank-space blocks require anchor_delta_rank > 0")
        self.hidden_size = hidden_size
        self.block_size = block_size
        self.rank = rank
        self.kind = kind
        self.num_blocks = num_blocks
        self.block_scope = block_scope
        self.norm = Qwen3RMSNorm(hidden_size, eps=eps)
        if kind == "dense":
            self.dense = nn.Parameter(torch.empty(block_size, hidden_size, hidden_size))
            self.register_parameter("dense_mlp_in", None)
            self.register_parameter("dense_mlp_out", None)
            self.register_parameter("down", None)
            self.register_parameter("up", None)
        elif kind == "dense_mlp":
            self.register_parameter("dense", None)
            self.dense_mlp_in = nn.Parameter(
                torch.empty(block_size, hidden_size, hidden_size)
            )
            self.dense_mlp_out = nn.Parameter(
                torch.empty(block_size, hidden_size, hidden_size)
            )
            self.register_parameter("down", None)
            self.register_parameter("up", None)
        elif rank == 0:
            self.register_parameter("dense", None)
            self.register_parameter("dense_mlp_in", None)
            self.register_parameter("dense_mlp_out", None)
            self.register_parameter("down", None)
            self.register_parameter("up", None)
        else:
            self.register_parameter("dense", None)
            self.register_parameter("dense_mlp_in", None)
            self.register_parameter("dense_mlp_out", None)
            self.down = nn.Parameter(torch.empty(block_size, hidden_size, rank))
            self.up = nn.Parameter(torch.empty(block_size, rank, hidden_size))
        if kind == "low_rank" and rank > 0 and num_blocks:
            if block_scope == "shared":
                self.shared_blocks = nn.ModuleList(
                    [SparseMLAAnchorRankBlock(rank, eps=eps) for _ in range(num_blocks)]
                )
                self.per_position_blocks = nn.ModuleList()
            else:
                self.shared_blocks = nn.ModuleList()
                self.per_position_blocks = nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                SparseMLAAnchorRankBlock(rank, eps=eps)
                                for _ in range(num_blocks)
                            ]
                        )
                        for _ in range(block_size)
                    ]
                )
        else:
            self.shared_blocks = nn.ModuleList()
            self.per_position_blocks = nn.ModuleList()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.kind == "dense":
            assert self.dense is not None
            nn.init.zeros_(self.dense)
            return
        if self.kind == "dense_mlp":
            assert self.dense_mlp_in is not None
            assert self.dense_mlp_out is not None
            for pos in range(self.block_size):
                nn.init.xavier_uniform_(self.dense_mlp_in[pos])
                nn.init.xavier_uniform_(self.dense_mlp_out[pos])
            return
        if self.rank > 0:
            assert self.down is not None
            assert self.up is not None
            for pos in range(self.block_size):
                nn.init.xavier_uniform_(self.down[pos])
            nn.init.zeros_(self.up)
        for block in self.shared_blocks:
            block.reset_parameters()
        for position_blocks in self.per_position_blocks:
            for block in position_blocks:
                block.reset_parameters()

    def forward(self, anchor_hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_states = anchor_hidden_state.unsqueeze(1).expand(
            anchor_hidden_state.shape[0],
            self.block_size,
            self.hidden_size,
        )
        if self.kind == "dense":
            assert self.dense is not None
            normalized = self.norm(anchor_hidden_state)
            deltas = [
                F.linear(normalized, self.dense[pos]) for pos in range(self.block_size)
            ]
            return hidden_states + torch.stack(deltas, dim=1)

        if self.kind == "dense_mlp":
            assert self.dense_mlp_in is not None
            assert self.dense_mlp_out is not None
            normalized = self.norm(anchor_hidden_state)
            deltas = []
            for pos in range(self.block_size):
                intermediate = F.silu(F.linear(normalized, self.dense_mlp_in[pos]))
                deltas.append(F.linear(intermediate, self.dense_mlp_out[pos]))
            return hidden_states + torch.stack(deltas, dim=1)

        if self.rank == 0:
            return hidden_states

        normalized = self.norm(anchor_hidden_state)
        # Avoid BF16 strided-batched cuBLAS for this small per-position adapter;
        # that path has shown deterministic internal errors on Blackwell.
        deltas = []
        for pos in range(self.block_size):
            assert self.down is not None
            assert self.up is not None
            low_rank = F.linear(normalized, self.down[pos].transpose(0, 1))
            for block in self.shared_blocks:
                low_rank = block(low_rank)
            if self.per_position_blocks:
                for block in self.per_position_blocks[pos]:
                    low_rank = block(low_rank)
            deltas.append(F.linear(low_rank, self.up[pos].transpose(0, 1)))
        delta = torch.stack(deltas, dim=1)
        return hidden_states + delta


class LowRankMarkovHead(nn.Module):
    """Low-rank previous-token correction for block-parallel draft logits."""

    def __init__(self, vocab_size: int, rank: int) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size}")
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        self.embedding = nn.Embedding(vocab_size, rank)
        self.projection = nn.Linear(rank, vocab_size, bias=False)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection.weight)

    def forward(self, previous_token_ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(previous_token_ids))


def compute_sparse_mla_metrics(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
    gamma: float = 4.0,
    position_1_loss_weight: torch.Tensor | float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    vocab_size = logits.shape[-1]
    elementwise_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size).float(),
        target_token_ids.reshape(-1),
        reduction="none",
    ).reshape_as(target_token_ids)

    pos_idx = torch.arange(logits.shape[1], device=logits.device) % block_size
    pos_idx = pos_idx.unsqueeze(0).expand_as(target_token_ids)
    loss_mask_f = loss_mask.to(elementwise_loss.dtype)
    decay = dflash_loss_decay(pos_idx.to(elementwise_loss.dtype), gamma=gamma)
    position_1_loss_weight_t = torch.as_tensor(
        position_1_loss_weight,
        device=logits.device,
        dtype=elementwise_loss.dtype,
    )
    decay = decay * torch.where(
        pos_idx == 1,
        position_1_loss_weight_t,
        torch.ones((), device=logits.device, dtype=elementwise_loss.dtype),
    )
    loss = (elementwise_loss * loss_mask_f * decay).sum(dim=1) / (
        loss_mask_f.sum(dim=1) + 1e-5
    )
    loss = loss.mean()

    pred_ids = torch.argmax(logits, dim=-1)
    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_token_ids, loss_mask, pos_idx, block_size
    )
    decay_per_pos = dflash_loss_decay(
        torch.arange(block_size, device=logits.device, dtype=elementwise_loss.dtype),
        gamma=gamma,
    )

    metrics: dict[str, torch.Tensor] = {
        "loss_sum": loss.detach().clone(),
        "loss_total": torch.tensor(1.0, device=logits.device),
        "full_acc_sum": correct_per_pos[1:].sum(),
        "full_acc_total": total_per_pos[1:].sum(),
        "decayed_acc_sum": (correct_per_pos * decay_per_pos).sum(),
        "decayed_acc_total": (total_per_pos * decay_per_pos).sum(),
        "position_1_loss_weight_sum": position_1_loss_weight_t.detach().clone(),
        "position_1_loss_weight_total": torch.tensor(1.0, device=logits.device),
    }
    for pos in range(1, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]
        pos_mask = loss_mask & (pos_idx == pos)
        metrics[f"position_{pos}_ce_sum"] = elementwise_loss[pos_mask].sum()
        metrics[f"position_{pos}_ce_total"] = pos_mask.sum()
    return loss, metrics


def _linear_schedule_value(
    start: float,
    end: float,
    steps: int,
    global_step: int | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if steps <= 0:
        value = start
    elif global_step is None:
        value = end
    else:
        clamped_step = min(max(int(global_step), 0), steps)
        value = start + (end - start) * (clamped_step / steps)
    return torch.tensor(value, device=device, dtype=dtype)


@SpeculatorModel.register("dflash_sparse_mla")
class DFlashSparseMLADraftModel(DraftVocabMixin, SpeculatorModel):
    config_class: ClassVar[type[DFlashSparseMLASpeculatorConfig]] = (
        DFlashSparseMLASpeculatorConfig  # type: ignore[misc]
    )
    _no_split_modules = ["DFlashSparseMLADecoderLayer"]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "t2d",
        "d2t",
        "verifier_final_norm_weight",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "embed_tokens.weight",
        "verifier_lm_head.weight",
    ]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def __init__(self, config: DFlashSparseMLASpeculatorConfig) -> None:
        config.transformer_layer_config = _coerce_pretrained_config(
            config.transformer_layer_config,
            field_name="transformer_layer_config",
        )
        super().__init__(config=config)
        self._init_sparse_mla_vocab(config)
        self.block_size = config.block_size
        if config.local_cross_attention_layer_ids:
            if config.anchor_only:
                raise ValueError(
                    "local_cross_attention_layer_ids requires sparse-MLA cache rows; "
                    "it cannot be used with anchor_only."
                )
            if len(config.local_cross_attention_layer_ids) != config.extra_local_layers:
                raise ValueError(
                    "local_cross_attention_layer_ids must contain exactly one "
                    "verifier layer id per extra local layer; got "
                    f"{len(config.local_cross_attention_layer_ids)} ids for "
                    f"extra_local_layers={config.extra_local_layers}"
                )
        self.layers = nn.ModuleList()
        if not config.anchor_only:
            self.layers = nn.ModuleList(
                [
                    DFlashSparseMLADecoderLayer(
                        config,
                        verifier_layer_id=int(layer_id),
                    )
                    for layer_id in config.verifier_kv_layer_ids
                ]
            )
        self.local_layers = nn.ModuleList(
            [
                (
                    DFlashSparseMLADecoderLayer(
                        config,
                        verifier_layer_id=int(
                            config.local_cross_attention_layer_ids[layer_idx]
                        ),
                        active_position_start=2 + layer_idx,
                    )
                    if config.local_cross_attention_layer_ids
                    else DFlashSparseMLALocalDecoderLayer(config)
                )
                for layer_idx in range(config.extra_local_layers)
            ]
        )
        hidden_size = int(config.transformer_layer_config.hidden_size)
        self.block_position_embeddings = nn.Embedding(config.block_size, hidden_size)
        self.anchor_initializer = SparseMLAAnchorInitializer(
            hidden_size=hidden_size,
            block_size=config.block_size,
            rank=config.anchor_delta_rank,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
            kind=config.anchor_delta_kind,
            num_blocks=config.anchor_delta_num_blocks,
            block_scope=config.anchor_delta_block_scope,
        )
        self.register_buffer(
            "anchor_dropout_probs",
            self._build_anchor_dropout_probs(config),
            persistent=False,
        )
        self.register_buffer(
            "anchor_dropout_start_probs",
            self._build_anchor_dropout_probs(config, attr="anchor_dropout_start_probs"),
            persistent=False,
        )
        self.norm = Qwen3RMSNorm(
            hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.register_buffer(
            "verifier_final_norm_weight",
            torch.ones(hidden_size, dtype=torch.bfloat16),
        )
        anchor_token_conditioning = getattr(
            config,
            "anchor_token_conditioning",
            "none",
        )
        if anchor_token_conditioning == "lm_head":
            self.anchor_token_norm = Qwen3RMSNorm(
                hidden_size,
                eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
            )
        elif anchor_token_conditioning != "none":
            raise ValueError(
                "anchor_token_conditioning must be 'none' or 'lm_head', got "
                f"{anchor_token_conditioning!r}"
            )
        known_token_conditioning = getattr(
            config,
            "known_token_conditioning",
            "none",
        )
        if known_token_conditioning == "lm_head":
            if config.block_size <= 1:
                raise ValueError(
                    "known_token_conditioning requires block_size > 1 so slot "
                    "0 can be used as known context while later slots remain "
                    "draft targets"
                )
            self.known_token_norm = Qwen3RMSNorm(
                hidden_size,
                eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
            )
        elif known_token_conditioning != "none":
            raise ValueError(
                "known_token_conditioning must be 'none' or 'lm_head', got "
                f"{known_token_conditioning!r}"
            )
        markov_head_rank = int(getattr(config, "markov_head_rank", 0))
        if markov_head_rank < 0:
            raise ValueError(f"markov_head_rank must be >= 0, got {markov_head_rank}")
        if markov_head_rank > 0 and config.block_size <= 1:
            raise ValueError("markov_head_rank requires block_size > 1")
        self.markov_head = (
            LowRankMarkovHead(config.draft_vocab_size, markov_head_rank)
            if markov_head_rank > 0
            else None
        )
        self.post_init()
        self.anchor_initializer.reset_parameters()
        if self.markov_head is not None:
            self.markov_head.reset_parameters()

    @property
    def markov_head_enabled(self) -> bool:
        return self.markov_head is not None

    def apply_markov_head(
        self,
        base_logits: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.markov_head is None:
            return base_logits
        return base_logits + self.markov_head(previous_token_ids).to(base_logits.dtype)

    def _apply_markov_teacher_forcing(
        self,
        base_logits: torch.Tensor,
        target_token_ids: torch.Tensor,
        first_previous_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.markov_head is None:
            return base_logits
        if first_previous_token_ids is None:
            first_previous_token_ids = target_token_ids[:, 0]
        first_previous_token_ids = first_previous_token_ids.to(
            device=target_token_ids.device,
            dtype=torch.long,
        ).reshape(-1)
        previous_token_ids = torch.cat(
            [
                first_previous_token_ids.unsqueeze(1),
                target_token_ids[:, 1:-1],
            ],
            dim=1,
        )
        transition_bias = self.markov_head(previous_token_ids).to(base_logits.dtype)
        if bool(getattr(self.config, "slot1_verifier_head_bypass", False)):
            transition_bias = torch.cat(
                [
                    torch.zeros_like(transition_bias[:, :1]),
                    transition_bias[:, 1:],
                ],
                dim=1,
            )
        return torch.cat(
            [base_logits[:, :1], base_logits[:, 1:] + transition_bias],
            dim=1,
        )

    def _markov_greedy_rollout(
        self,
        base_logits: torch.Tensor,
        first_previous_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.markov_head is None:
            return base_logits[:, 1:].argmax(dim=-1)
        previous_token_ids = first_previous_token_ids.to(
            device=base_logits.device,
            dtype=torch.long,
        ).reshape(-1)
        selected: list[torch.Tensor] = []
        for slot in range(1, base_logits.shape[1]):
            if slot == 1 and bool(
                getattr(self.config, "slot1_verifier_head_bypass", False)
            ):
                step_logits = base_logits[:, slot]
            else:
                step_logits = self.apply_markov_head(
                    base_logits[:, slot],
                    previous_token_ids,
                )
            previous_token_ids = step_logits.argmax(dim=-1)
            selected.append(previous_token_ids)
        return torch.stack(selected, dim=1)

    @staticmethod
    def _build_anchor_dropout_probs(
        config: DFlashSparseMLASpeculatorConfig,
        *,
        attr: str = "anchor_dropout_probs",
    ) -> torch.Tensor:
        probs = list(getattr(config, attr))
        if not probs:
            return torch.empty(0, dtype=torch.float32)
        if len(probs) == 1:
            probs = probs * int(config.block_size)
        if len(probs) != int(config.block_size):
            raise ValueError(
                f"{attr} must be empty, length 1, or length "
                f"block_size={config.block_size}; got {len(probs)}"
            )
        if any(p < 0.0 or p > 1.0 for p in probs):
            raise ValueError(f"{attr} entries must satisfy 0 <= p <= 1")
        return torch.tensor(probs, dtype=torch.float32)

    def _current_anchor_dropout_probs(
        self,
        *,
        device: torch.device,
        global_step: int | None,
    ) -> torch.Tensor:
        if self.anchor_dropout_probs.numel() == 0:
            if self.anchor_dropout_start_probs.numel() == 0:
                return torch.empty(0, device=device, dtype=torch.float32)
            target = torch.zeros_like(self.anchor_dropout_start_probs)
        else:
            target = self.anchor_dropout_probs
        if self.anchor_dropout_start_probs.numel() == 0 or not self.training:
            return target.to(device=device, dtype=torch.float32)

        start = self.anchor_dropout_start_probs.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        if global_step is None:
            return target

        hold_steps = max(int(self.config.anchor_dropout_hold_steps), 0)
        ramp_steps = max(int(self.config.anchor_dropout_ramp_steps), 0)
        step = max(int(global_step), 0)
        if step < hold_steps:
            return start
        if ramp_steps == 0:
            return target
        progress = min(max((step - hold_steps) / ramp_steps, 0.0), 1.0)
        return start + (target - start) * progress

    def _apply_anchor_dropout(
        self,
        hidden_states: torch.Tensor,
        *,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        if probs.numel() == 0:
            return hidden_states
        probs = probs.to(
            device=hidden_states.device,
            dtype=torch.float32,
        )
        keep = (1.0 - probs).view(1, -1, 1)
        hard_keep = (probs < 1.0).view(1, -1, 1)
        if not self.training:
            if torch.all(hard_keep):
                return hidden_states
            return hidden_states * hard_keep.to(hidden_states.dtype)
        if torch.all(keep == 1.0):
            return hidden_states
        safe_keep = torch.where(keep > 0.0, keep, torch.ones_like(keep))
        mask = (
            torch.rand(
                hidden_states.shape,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            < keep
        )
        return (
            hidden_states
            * mask.to(hidden_states.dtype)
            / safe_keep.to(hidden_states.dtype)
        )

    def _anchor_dropout_metrics(self, probs: torch.Tensor) -> dict[str, torch.Tensor]:
        if probs.numel() == 0:
            return {}
        metrics: dict[str, torch.Tensor] = {
            "anchor_dropout/mean_sum": probs.mean().detach().clone(),
            "anchor_dropout/mean_total": torch.tensor(1.0, device=probs.device),
        }
        for pos, prob in enumerate(probs, 1):
            metrics[f"anchor_dropout/position_{pos}_sum"] = prob.detach().clone()
            metrics[f"anchor_dropout/position_{pos}_total"] = torch.tensor(
                1.0,
                device=probs.device,
            )
        return metrics

    def _uses_known_token_conditioning(self) -> bool:
        return getattr(self.config, "known_token_conditioning", "none") != "none"

    def _apply_known_token_conditioning(
        self,
        hidden_states: torch.Tensor,
        known_token_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        conditioning = getattr(self.config, "known_token_conditioning", "none")
        if conditioning == "none":
            return hidden_states
        if conditioning != "lm_head":
            raise ValueError(f"unknown known_token_conditioning mode {conditioning!r}")
        if hidden_states.shape[1] < 1:
            raise ValueError(
                "known_token_conditioning requires a block with slot 0 present"
            )
        if known_token_ids is None:
            raise ValueError(
                "known_token_ids is required when known_token_conditioning is enabled"
            )
        known_token_ids = known_token_ids.to(
            device=hidden_states.device,
            dtype=torch.long,
        ).reshape(-1)
        if known_token_ids.shape != (hidden_states.shape[0],):
            raise ValueError(
                "known_token_ids must have shape [batch] or [batch, 1], got "
                f"{tuple(known_token_ids.shape)} for batch={hidden_states.shape[0]}"
            )
        token_states = F.embedding(
            known_token_ids,
            self.lm_head.weight.detach(),
        ).to(dtype=hidden_states.dtype)
        token_states = self.known_token_norm(token_states)
        hidden_states = hidden_states.clone()
        hidden_states[:, 0] = hidden_states[:, 0] + token_states
        return hidden_states

    def _known_token_loss_mask(
        self,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self._uses_known_token_conditioning() and loss_mask.shape[1] > 0:
            loss_mask = loss_mask.clone()
            loss_mask[:, 0] = False
        return loss_mask

    def _normalize_anchor_token_ids(
        self,
        anchor_token_ids: torch.Tensor | None,
        *,
        target_token_ids: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if anchor_token_ids is None and target_token_ids is not None:
            anchor_token_ids = target_token_ids[:, 0]
        if anchor_token_ids is None:
            return None
        anchor_token_ids = anchor_token_ids.to(device=device, dtype=torch.long).reshape(
            -1
        )
        if anchor_token_ids.shape != (batch_size,):
            raise ValueError(
                "anchor_token_ids must have shape [batch] or [batch, 1], got "
                f"{tuple(anchor_token_ids.shape)} for batch={batch_size}"
            )
        if target_token_ids is not None:
            expected = target_token_ids[:, 0].to(device=device, dtype=torch.long)
            if not torch.equal(anchor_token_ids, expected):
                raise ValueError("anchor_token_ids must match target_token_ids[:, 0]")
        return anchor_token_ids

    def _apply_anchor_token_conditioning(
        self,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        conditioning = getattr(self.config, "anchor_token_conditioning", "none")
        if conditioning == "none":
            return hidden_states
        if conditioning != "lm_head":
            raise ValueError(f"unknown anchor_token_conditioning mode {conditioning!r}")
        if anchor_token_ids is None:
            raise ValueError(
                "anchor_token_ids is required when anchor_token_conditioning is enabled"
            )
        token_states = F.embedding(
            anchor_token_ids,
            self.lm_head.weight.detach(),
        ).to(dtype=hidden_states.dtype)
        token_states = self.anchor_token_norm(token_states)
        hidden_states = hidden_states.clone()
        hidden_states[:, 0] = hidden_states[:, 0] + token_states
        return hidden_states

    def load_verifier_final_norm_weight(self, weight: torch.Tensor) -> None:
        if tuple(weight.shape) != tuple(self.verifier_final_norm_weight.shape):
            raise ValueError(
                "verifier final RMSNorm shape mismatch: got "
                f"{tuple(weight.shape)}, expected "
                f"{tuple(self.verifier_final_norm_weight.shape)}"
            )
        self.verifier_final_norm_weight.copy_(
            weight.detach().to(
                device=self.verifier_final_norm_weight.device,
                dtype=self.verifier_final_norm_weight.dtype,
            )
        )

    def _verifier_final_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + float(self.config.transformer_layer_config.rms_norm_eps)
        ).to(hidden_states.dtype)
        return hidden_states * self.verifier_final_norm_weight.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    def _apply_slot1_verifier_head_bypass(
        self,
        logits: torch.Tensor,
        anchor_context: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(getattr(self.config, "slot1_verifier_head_bypass", False)):
            return logits
        if logits.shape[1] <= 1:
            raise ValueError(
                "slot1_verifier_head_bypass requires block_size >= 2, got "
                f"{logits.shape[1]}"
            )
        slot1_logits = self.lm_head(self._verifier_final_norm(anchor_context))
        logits = logits.clone()
        logits[:, 1, :] = slot1_logits
        return logits

    def _init_sparse_mla_vocab(self, config: DFlashSparseMLASpeculatorConfig) -> None:
        """Initialize only the output head sparse-MLA actually uses."""
        tl_config = config.transformer_layer_config
        self.draft_vocab_size = config.draft_vocab_size
        self.verifier_vocab_size = tl_config.vocab_size
        self.hidden_size = tl_config.hidden_size
        self.use_draft_vocab = self.draft_vocab_size != self.verifier_vocab_size
        t2d: torch.Tensor | None = None
        d2t: torch.Tensor | None = None
        if self.use_draft_vocab:
            t2d = torch.zeros((self.verifier_vocab_size,), dtype=torch.bool)
            d2t = torch.zeros((self.draft_vocab_size,), dtype=torch.long)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)

        self.lm_head = nn.Linear(
            self.hidden_size,
            self.draft_vocab_size,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.lm_head.weight.requires_grad_(False)
        torch.nn.init.constant_(self.lm_head.weight, torch.nan)
        self.lm_head._is_hf_initialized = True  # type: ignore[assignment] # noqa: SLF001

    @property
    def target_layer_ids(self) -> list[int]:
        if self.config.anchor_only:
            return []
        layer_ids: list[int] = []
        for layer_id in list(self.config.verifier_kv_layer_ids) + list(
            self.config.local_cross_attention_layer_ids
        ):
            if layer_id not in layer_ids:
                layer_ids.append(layer_id)
        return layer_ids

    @property
    def mask_token_id(self) -> int:
        if self.config.mask_token_id is None:
            raise ValueError(
                "mask_token_id is not set on the config. Pass --mask-token-id "
                "during training or save it in the config."
            )
        return self.config.mask_token_id

    def load_verifier_weights(self) -> None:
        speculators_config = getattr(
            getattr(self, "config", None), "speculators_config", None
        )
        if (
            speculators_config is None
            or speculators_config.verifier.name_or_path is None
        ):
            return

        verifier_weights = load_model_layers(
            ["lm_head.weight", "model.norm.weight"],
            speculators_config.verifier.name_or_path,
        )
        lm_head_weight = verifier_weights["lm_head.weight"]

        if self.use_draft_vocab:
            if self.t2d is None or not torch.any(self.t2d).item():  # type: ignore[arg-type]
                raise ValueError(
                    "t2d tensor hasn't been set. Please call "
                    "`.load_vocab_mappings(t2d, d2t)` before `.load_verifier_weights()`"
                )
            lm_head_weight = lm_head_weight[
                self.t2d.to(device=lm_head_weight.device, dtype=torch.bool), :  # type: ignore[union-attr,index]
            ]

        self.lm_head.load_state_dict({"weight": lm_head_weight}, strict=False)
        self.lm_head.weight.requires_grad_(False)
        if "model.norm.weight" in verifier_weights:
            self.load_verifier_final_norm_weight(verifier_weights["model.norm.weight"])

    def _iter_cross_attention_modules(self) -> list[SparseMLACrossAttention]:
        modules: list[SparseMLACrossAttention] = []
        for layer in self.layers:
            modules.append(layer.cross_attn)
        for layer in self.local_layers:
            if isinstance(layer, DFlashSparseMLADecoderLayer):
                modules.append(layer.cross_attn)
        return modules

    def load_target_compatible_mla_weights(self) -> None:
        if self.config.cross_attention_impl != "target_compatible":
            return
        speculators_config = getattr(
            getattr(self, "config", None), "speculators_config", None
        )
        if (
            speculators_config is None
            or speculators_config.verifier.name_or_path is None
        ):
            return

        modules_by_layer: dict[int, list[SparseMLACrossAttention]] = {}
        for module in self._iter_cross_attention_modules():
            if module.verifier_layer_id is None:
                raise ValueError(
                    "target_compatible sparse-MLA requires every cross-attention "
                    "module to have a verifier_layer_id"
                )
            modules_by_layer.setdefault(module.verifier_layer_id, []).append(module)

        suffixes = [
            "q_a_proj.weight",
            "q_a_layernorm.weight",
            "q_b_proj.weight",
            "kv_b_proj.weight",
            "o_proj.weight",
        ]
        requested_names: list[str] = []
        for layer_id in modules_by_layer:
            requested_names.extend(
                [f"model.layers.{layer_id}.self_attn.{suffix}" for suffix in suffixes]
            )
        requested_names.append("model.norm.weight")
        verifier_weights = load_model_layers(
            requested_names,
            speculators_config.verifier.name_or_path,
        )
        if "model.norm.weight" in verifier_weights:
            self.load_verifier_final_norm_weight(verifier_weights["model.norm.weight"])
        for layer_id, modules in modules_by_layer.items():
            layer_weights = {
                suffix: verifier_weights[f"model.layers.{layer_id}.self_attn.{suffix}"]
                for suffix in suffixes
            }
            for module in modules:
                module.load_target_compatible_weights(layer_weights)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DFlashSparseMLADraftModel":
        verifier_name_or_path = kwargs["verifier_name_or_path"]
        verifier_text_config = _get_text_config(
            AutoConfig.from_pretrained(verifier_name_or_path, trust_remote_code=True)
        )
        if kwargs["draft_vocab_size"] != verifier_text_config.vocab_size:
            raise ValueError(
                "dflash_sparse_mla currently requires the full verifier vocabulary. "
                "Use --full-draft-vocab."
            )

        verifier_kv_layer_ids = kwargs.get("verifier_kv_layer_ids")
        if verifier_kv_layer_ids is None:
            verifier_kv_layer_ids = [3, 18, 27, 42, 77]

        mla_dims = _derive_mla_dims(verifier_name_or_path)
        config = DFlashSparseMLASpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            block_size=kwargs.get("block_size", 8),
            mask_token_id=kwargs.get("mask_token_id"),
            verifier_kv_layer_ids=list(verifier_kv_layer_ids),
            sparse_topk=kwargs.get("sparse_mla_topk", 2048),
            mla_kv_lora_rank=kwargs.get("sparse_mla_kv_lora_rank")
            or mla_dims["mla_kv_lora_rank"],
            mla_qk_rope_head_dim=kwargs.get("sparse_mla_qk_rope_head_dim")
            or mla_dims["mla_qk_rope_head_dim"],
            mla_qk_nope_head_dim=mla_dims["mla_qk_nope_head_dim"],
            mla_q_lora_rank=mla_dims["mla_q_lora_rank"],
            mla_attention_scale_dim=kwargs.get("sparse_mla_attention_scale_dim"),
            mla_v_head_dim=kwargs.get("sparse_mla_v_head_dim")
            or mla_dims["mla_v_head_dim"],
            mla_num_heads=kwargs.get("sparse_mla_num_heads")
            or mla_dims["mla_num_heads"],
            cross_attention_impl=kwargs.get(
                "sparse_mla_cross_attention_impl", "learned"
            ),
            non_causal_block_attention=kwargs.get(
                "sparse_mla_non_causal_block_attention", True
            ),
            extra_local_layers=kwargs.get("sparse_mla_extra_local_layers", 0),
            local_cross_attention_layer_ids=kwargs.get(
                "sparse_mla_local_cross_layer_ids"
            )
            or [],
            anchor_only=kwargs.get("sparse_mla_anchor_only", False),
            ablate_sparse_mla_cross_attention=kwargs.get(
                "sparse_mla_ablate_cross_attention", False
            ),
            anchor_delta_rank=kwargs.get("sparse_mla_anchor_delta_rank", 0),
            anchor_delta_kind=kwargs.get("sparse_mla_anchor_delta_kind", "dense"),
            anchor_delta_num_blocks=kwargs.get("sparse_mla_anchor_delta_num_blocks", 0),
            anchor_delta_block_scope=kwargs.get(
                "sparse_mla_anchor_delta_block_scope", "shared"
            ),
            num_attention_sink_tokens=kwargs.get(
                "sparse_mla_num_attention_sink_tokens", 0
            ),
            cross_attention_gate=kwargs.get("sparse_mla_cross_attention_gate", False),
            cross_attention_gate_init=kwargs.get(
                "sparse_mla_cross_attention_gate_init", -1.0
            ),
            residual_branch_gate=kwargs.get("sparse_mla_residual_branch_gate", False),
            residual_branch_gate_init=kwargs.get(
                "sparse_mla_residual_branch_gate_init", -6.0
            ),
            position_1_loss_weight_start=kwargs.get(
                "sparse_mla_position_1_loss_weight_start", 1.0
            ),
            position_1_loss_weight_end=kwargs.get(
                "sparse_mla_position_1_loss_weight_end", 1.0
            ),
            position_1_loss_weight_steps=kwargs.get(
                "sparse_mla_position_1_loss_weight_steps", 0
            ),
            markov_head_rank=kwargs.get("sparse_mla_markov_head_rank", 0),
            anchor_dropout_probs=kwargs.get("sparse_mla_anchor_dropout_probs") or [],
            anchor_dropout_start_probs=kwargs.get(
                "sparse_mla_anchor_dropout_start_probs"
            )
            or [],
            anchor_dropout_hold_steps=kwargs.get(
                "sparse_mla_anchor_dropout_hold_steps", 0
            ),
            anchor_dropout_ramp_steps=kwargs.get(
                "sparse_mla_anchor_dropout_ramp_steps", 0
            ),
            query_position_adapter_rank=kwargs.get(
                "sparse_mla_query_position_adapter_rank", 0
            ),
            query_position_output_adapter_rank=kwargs.get(
                "sparse_mla_query_position_output_adapter_rank", 0
            ),
            query_mlp_blocks=kwargs.get("sparse_mla_query_mlp_blocks", 0),
            query_anchor_conditioned=kwargs.get(
                "sparse_mla_query_anchor_conditioned", False
            ),
            query_layer_adapter_rank=kwargs.get(
                "sparse_mla_query_layer_adapter_rank", 0
            ),
            known_token_conditioning=kwargs.get(
                "sparse_mla_known_token_conditioning", "none"
            ),
            anchor_token_conditioning=kwargs.get(
                "sparse_mla_anchor_token_conditioning", "none"
            ),
            slot1_verifier_head_bypass=kwargs.get(
                "sparse_mla_slot1_verifier_head_bypass", False
            ),
            speculators_config=SpeculatorsConfig(
                algorithm="dflash_sparse_mla",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=kwargs.get("block_size", 8) - 1,
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_text_config, name_or_path=verifier_name_or_path
                ),
            ),
        )
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            model = cls(config=config)
        finally:
            torch.set_default_dtype(original_dtype)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        model.load_target_compatible_mla_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        return {}, {}

    def _collect_gate_metrics(self) -> dict[str, torch.Tensor]:
        metrics: dict[str, torch.Tensor] = {}
        for layer_idx, layer in enumerate(self.layers):
            prefix = f"sparse_layer_{layer_idx}"
            _record_gate_metrics(
                metrics,
                f"{prefix}/self_attn_residual",
                layer.self_attn_residual_gate,
            )
            _record_gate_metrics(
                metrics,
                f"{prefix}/cross_attn_residual",
                layer.cross_attn_residual_gate,
            )
            _record_gate_metrics(
                metrics,
                f"{prefix}/mlp_residual",
                layer.mlp_residual_gate,
            )
            _record_gate_metrics(
                metrics,
                f"{prefix}/cross_attention",
                layer.cross_attention_gate,
            )

        for layer_idx, layer in enumerate(self.local_layers):
            prefix = f"local_layer_{layer_idx}"
            _record_gate_metrics(
                metrics,
                f"{prefix}/self_attn_residual",
                layer.self_attn_residual_gate,
            )
            if isinstance(layer, DFlashSparseMLADecoderLayer):
                _record_gate_metrics(
                    metrics,
                    f"{prefix}/cross_attn_residual",
                    layer.cross_attn_residual_gate,
                )
                _record_gate_metrics(
                    metrics,
                    f"{prefix}/cross_attention",
                    layer.cross_attention_gate,
                )
            _record_gate_metrics(
                metrics,
                f"{prefix}/mlp_residual",
                layer.mlp_residual_gate,
            )
        return metrics

    def _collect_branch_metrics(self) -> dict[str, torch.Tensor]:
        metrics: dict[str, torch.Tensor] = {}

        def add_layer_metrics(prefix: str, layer: nn.Module) -> None:
            branch_metrics = getattr(layer, "_last_branch_metrics", {})
            for name, value in branch_metrics.items():
                metrics[f"branches/{prefix}/{name}"] = value
            cross_attn = getattr(layer, "cross_attn", None)
            cross_metrics = getattr(cross_attn, "_last_debug_metrics", {})
            for name, value in cross_metrics.items():
                metrics[f"cross_attn/{prefix}/{name}"] = value

        for layer_idx, layer in enumerate(self.layers):
            add_layer_metrics(f"sparse_layer_{layer_idx}", layer)
        for layer_idx, layer in enumerate(self.local_layers):
            add_layer_metrics(f"local_layer_{layer_idx}", layer)
        return metrics

    def _first_refinement_position(self) -> int:
        return 1

    def _run_sparse_layers(
        self,
        hidden_states: torch.Tensor,
        sparse_mla_cache_rows: torch.Tensor,
        sparse_mla_cache_valid_mask: torch.Tensor | None = None,
        anchor_hidden_state: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        ablate_sparse_mla_cross_attention: bool = False,
    ) -> torch.Tensor:
        if not self.layers:
            return hidden_states
        if sparse_mla_cache_rows.shape[1] != len(self.layers):
            raise ValueError(
                "sparse_mla_cache_rows layer count mismatch: "
                f"got {sparse_mla_cache_rows.shape[1]}, expected "
                f"{len(self.layers)}"
            )
        if (
            sparse_mla_cache_valid_mask is not None
            and sparse_mla_cache_valid_mask.shape != sparse_mla_cache_rows.shape[:3]
        ):
            raise ValueError(
                "sparse_mla_cache_valid_mask shape mismatch: got "
                f"{tuple(sparse_mla_cache_valid_mask.shape)}, expected "
                f"{tuple(sparse_mla_cache_rows.shape[:3])}"
            )

        seq_len = hidden_states.shape[1]
        active_start = self._first_refinement_position()
        if active_start >= seq_len:
            return hidden_states

        kv_positions = torch.arange(seq_len, device=hidden_states.device)
        query_positions = torch.arange(
            active_start, seq_len, device=hidden_states.device
        )
        attn_mask = kv_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        active_position_ids = None
        if position_ids is not None:
            active_position_ids = (
                position_ids[active_start:]
                if position_ids.ndim == 1
                else position_ids[:, active_start:]
            )

        for layer_idx, layer in enumerate(self.layers):
            active_states = hidden_states[:, active_start:]
            key_value_states = torch.cat(
                [
                    hidden_states[:, :active_start].detach(),
                    active_states,
                ],
                dim=1,
            )
            refined_active = layer(
                active_states,
                sparse_mla_cache_rows[:, layer_idx],
                mla_cache_valid_mask=(
                    sparse_mla_cache_valid_mask[:, layer_idx]
                    if sparse_mla_cache_valid_mask is not None
                    else None
                ),
                key_value_states=key_value_states,
                attn_mask=attn_mask,
                position_start=active_start,
                position_ids=active_position_ids,
                anchor_hidden_state=anchor_hidden_state,
                ablate_cross_attention=ablate_sparse_mla_cross_attention,
            )
            hidden_states = torch.cat(
                [
                    hidden_states[:, :active_start],
                    refined_active,
                ],
                dim=1,
            )
        return hidden_states

    def _run_local_layers(
        self,
        hidden_states: torch.Tensor,
        local_mla_cache_rows: torch.Tensor | None = None,
        local_mla_cache_valid_mask: torch.Tensor | None = None,
        anchor_hidden_state: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        ablate_sparse_mla_cross_attention: bool = False,
    ) -> torch.Tensor:
        if not self.local_layers:
            return hidden_states
        if local_mla_cache_rows is not None and local_mla_cache_rows.shape[1] != len(
            self.local_layers
        ):
            raise ValueError(
                "local_mla_cache_rows layer count mismatch: "
                f"got {local_mla_cache_rows.shape[1]}, expected "
                f"{len(self.local_layers)}"
            )
        if (
            local_mla_cache_valid_mask is not None
            and local_mla_cache_rows is not None
            and local_mla_cache_valid_mask.shape != local_mla_cache_rows.shape[:3]
        ):
            raise ValueError(
                "local_mla_cache_valid_mask shape mismatch: got "
                f"{tuple(local_mla_cache_valid_mask.shape)}, expected "
                f"{tuple(local_mla_cache_rows.shape[:3])}"
            )
        # Position 0 is the anchor-inclusive slot. In known-token mode position
        # 1 is verifier-sampled context; otherwise it is the first drafted token
        # already produced by the sparse layer stack. Each local layer refines
        # only the remaining suffix while attending to detached prefix states.
        seq_len = hidden_states.shape[1]
        kv_positions = torch.arange(seq_len, device=hidden_states.device)
        for layer_idx, layer in enumerate(self.local_layers):
            active_start = 2 + layer_idx
            if active_start >= seq_len:
                break

            active_states = hidden_states[:, active_start:]
            key_value_states = torch.cat(
                [
                    hidden_states[:, :active_start].detach(),
                    active_states,
                ],
                dim=1,
            )
            query_positions = torch.arange(
                active_start,
                seq_len,
                device=hidden_states.device,
            )
            attn_mask = kv_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            if local_mla_cache_rows is None:
                refined_active = layer(
                    active_states,
                    key_value_states=key_value_states,
                    attn_mask=attn_mask,
                    position_start=active_start,
                )
            else:
                active_position_ids = None
                if position_ids is not None:
                    active_position_ids = (
                        position_ids[active_start:]
                        if position_ids.ndim == 1
                        else position_ids[:, active_start:]
                    )
                refined_active = layer(
                    active_states,
                    local_mla_cache_rows[:, layer_idx],
                    mla_cache_valid_mask=(
                        local_mla_cache_valid_mask[:, layer_idx]
                        if local_mla_cache_valid_mask is not None
                        else None
                    ),
                    key_value_states=key_value_states,
                    attn_mask=attn_mask,
                    position_start=active_start,
                    position_ids=active_position_ids,
                    anchor_hidden_state=anchor_hidden_state,
                    ablate_cross_attention=ablate_sparse_mla_cross_attention,
                )
            hidden_states = torch.cat(
                [
                    hidden_states[:, :active_start],
                    refined_active,
                ],
                dim=1,
            )
        return hidden_states

    def forward_logits(
        self,
        *,
        anchor_hidden_state: torch.Tensor,
        anchor_token_ids: torch.Tensor | None = None,
        mla_cache_rows: torch.Tensor | None = None,
        mla_cache_rows_packed: torch.Tensor | None = None,
        mla_cache_valid_mask: torch.Tensor | None = None,
        verifier_layer_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        known_token_ids: torch.Tensor | None = None,
        global_step: int | None = None,
        ablate_sparse_mla_cross_attention: bool = False,
    ) -> torch.Tensor:
        ablate_sparse_mla_cross_attention = ablate_sparse_mla_cross_attention or bool(
            getattr(
                self.config,
                "ablate_sparse_mla_cross_attention",
                False,
            )
        )
        if (
            not self.config.anchor_only
            and mla_cache_rows is None
            and mla_cache_rows_packed is None
        ):
            raise ValueError(
                "CausalCascade forward_logits requires mla_cache_rows or "
                "mla_cache_rows_packed unless anchor_only is enabled."
            )

        batch_size = anchor_hidden_state.shape[0]
        hidden_size = self.block_position_embeddings.embedding_dim
        if anchor_hidden_state.shape != (batch_size, hidden_size):
            raise ValueError(
                "anchor_hidden_state shape mismatch: "
                f"got {tuple(anchor_hidden_state.shape)}, "
                f"expected {(batch_size, hidden_size)}"
            )
        anchor_token_ids = self._normalize_anchor_token_ids(
            anchor_token_ids,
            target_token_ids=None,
            batch_size=batch_size,
            device=anchor_hidden_state.device,
        )
        anchor_context = anchor_hidden_state.to(
            dtype=self.block_position_embeddings.weight.dtype,
        )
        anchor_dropout_probs = self._current_anchor_dropout_probs(
            device=anchor_hidden_state.device,
            global_step=global_step,
        )
        hidden_states = self.anchor_initializer(anchor_context)
        hidden_states = self._apply_anchor_dropout(
            hidden_states,
            probs=anchor_dropout_probs,
        )
        block_positions = torch.arange(self.block_size, device=hidden_states.device)
        hidden_states = hidden_states + self.block_position_embeddings(block_positions)
        hidden_states = self._apply_anchor_token_conditioning(
            hidden_states,
            anchor_token_ids,
        )
        hidden_states = self._apply_known_token_conditioning(
            hidden_states,
            known_token_ids,
        )

        local_mla_cache_rows = None
        local_mla_cache_valid_mask = None
        if not self.config.anchor_only:
            if mla_cache_rows is None:
                assert mla_cache_rows_packed is not None
                mla_cache_rows = dequantize_fp8_ds_mla_rows(
                    mla_cache_rows_packed,
                    dtype=hidden_states.dtype,
                )
            if mla_cache_valid_mask is not None:
                if mla_cache_valid_mask.shape != mla_cache_rows.shape[:3]:
                    raise ValueError(
                        "mla_cache_valid_mask shape mismatch: got "
                        f"{tuple(mla_cache_valid_mask.shape)}, expected "
                        f"{tuple(mla_cache_rows.shape[:3])}"
                    )
                mla_cache_valid_mask = mla_cache_valid_mask.to(
                    device=mla_cache_rows.device,
                    dtype=torch.bool,
                )
            if verifier_layer_ids is not None:
                if verifier_layer_ids.ndim == 2:
                    verifier_layer_ids = verifier_layer_ids[0]
                available = {
                    int(layer_id.item()): idx
                    for idx, layer_id in enumerate(verifier_layer_ids.to("cpu"))
                }
            else:
                if (
                    mla_cache_rows.shape[1] != len(self.layers)
                    or self.config.local_cross_attention_layer_ids
                ):
                    raise ValueError(
                        f"mla_cache_rows has {mla_cache_rows.shape[1]} layers, "
                        f"expected {len(self.layers)} and no local cross-attention "
                        "layers when verifier_layer_ids is omitted"
                    )
                available = {
                    int(layer_id): idx
                    for idx, layer_id in enumerate(self.config.verifier_kv_layer_ids)
                }

            def select_layer_tensor(
                tensor: torch.Tensor,
                layer_ids: list[int],
                *,
                name: str,
            ) -> torch.Tensor:
                assert mla_cache_rows is not None
                layer_indices: list[int] = []
                for layer_id in layer_ids:
                    idx = available.get(int(layer_id))
                    if idx is None:
                        raise ValueError(
                            f"{name} does not contain requested verifier "
                            f"layer {layer_id}; available layers are "
                            f"{list(available)}"
                        )
                    layer_indices.append(idx)
                index = torch.tensor(layer_indices, device=tensor.device)
                return tensor.index_select(1, index)

            target_topk = int(self.config.sparse_topk)
            if target_topk <= 0:
                raise ValueError(f"sparse_topk must be > 0, got {target_topk}")
            if mla_cache_rows.shape[2] < target_topk:
                raise ValueError(
                    f"mla_cache_rows has topk {mla_cache_rows.shape[2]}, "
                    f"requested {target_topk}"
                )
            if mla_cache_rows.shape[2] > target_topk:
                mla_cache_rows = mla_cache_rows[:, :, :target_topk, :].contiguous()
            if mla_cache_valid_mask is not None:
                if mla_cache_valid_mask.shape[2] < target_topk:
                    raise ValueError(
                        "mla_cache_valid_mask has topk "
                        f"{mla_cache_valid_mask.shape[2]}, requested {target_topk}"
                    )
                if mla_cache_valid_mask.shape[2] > target_topk:
                    mla_cache_valid_mask = mla_cache_valid_mask[
                        :, :, :target_topk
                    ].contiguous()

            sparse_mla_cache_rows = select_layer_tensor(
                mla_cache_rows,
                list(self.config.verifier_kv_layer_ids),
                name="mla_cache_rows",
            )
            sparse_mla_cache_valid_mask = (
                select_layer_tensor(
                    mla_cache_valid_mask,
                    list(self.config.verifier_kv_layer_ids),
                    name="mla_cache_valid_mask",
                )
                if mla_cache_valid_mask is not None
                else None
            )
            if self.config.local_cross_attention_layer_ids:
                local_mla_cache_rows = select_layer_tensor(
                    mla_cache_rows,
                    list(self.config.local_cross_attention_layer_ids),
                    name="mla_cache_rows",
                )
                local_mla_cache_valid_mask = (
                    select_layer_tensor(
                        mla_cache_valid_mask,
                        list(self.config.local_cross_attention_layer_ids),
                        name="mla_cache_valid_mask",
                    )
                    if mla_cache_valid_mask is not None
                    else None
                )
            hidden_states = self._run_sparse_layers(
                hidden_states,
                sparse_mla_cache_rows,
                sparse_mla_cache_valid_mask,
                anchor_hidden_state=anchor_context,
                position_ids=position_ids,
                ablate_sparse_mla_cross_attention=ablate_sparse_mla_cross_attention,
            )
        hidden_states = self._run_local_layers(
            hidden_states,
            local_mla_cache_rows,
            local_mla_cache_valid_mask,
            anchor_hidden_state=anchor_context,
            position_ids=position_ids,
            ablate_sparse_mla_cross_attention=ablate_sparse_mla_cross_attention,
        )
        logits = self.lm_head(self.norm(hidden_states))
        return self._apply_slot1_verifier_head_bypass(logits, anchor_context)

    def forward(
        self,
        anchor_token_ids: torch.Tensor | None = None,
        anchor_hidden_state: torch.Tensor | None = None,
        target_token_ids: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        mla_cache_rows: torch.Tensor | None = None,
        mla_cache_rows_packed: torch.Tensor | None = None,
        mla_cache_valid_mask: torch.Tensor | None = None,
        verifier_layer_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        known_token_ids: torch.Tensor | None = None,
        global_step: int | None = None,
        ablate_sparse_mla_cross_attention: bool = False,
        **kwargs,
    ):
        del kwargs
        ablate_sparse_mla_cross_attention = ablate_sparse_mla_cross_attention or bool(
            getattr(
                self.config,
                "ablate_sparse_mla_cross_attention",
                False,
            )
        )
        if anchor_hidden_state is None or target_token_ids is None or loss_mask is None:
            raise ValueError(
                "Sparse-MLA DFlash forward requires anchor_hidden_state, "
                "target_token_ids, and loss_mask."
            )
        if (
            not self.config.anchor_only
            and mla_cache_rows is None
            and mla_cache_rows_packed is None
        ):
            raise ValueError(
                "Sparse-MLA DFlash forward requires mla_cache_rows or "
                "mla_cache_rows_packed unless anchor_only is enabled."
            )
        if target_token_ids.shape[1] != self.block_size:
            raise ValueError(
                f"target_token_ids block size {target_token_ids.shape[1]} does not "
                f"match config block_size {self.block_size}"
            )
        batch_size = target_token_ids.shape[0]
        hidden_size = self.block_position_embeddings.embedding_dim
        if anchor_hidden_state.shape != (batch_size, hidden_size):
            raise ValueError(
                "anchor_hidden_state shape mismatch: "
                f"got {tuple(anchor_hidden_state.shape)}, "
                f"expected {(batch_size, hidden_size)}"
            )
        anchor_token_ids = self._normalize_anchor_token_ids(
            anchor_token_ids,
            target_token_ids=target_token_ids,
            batch_size=batch_size,
            device=target_token_ids.device,
        )
        anchor_context = anchor_hidden_state.to(
            device=target_token_ids.device,
            dtype=self.block_position_embeddings.weight.dtype,
        )
        anchor_dropout_probs = self._current_anchor_dropout_probs(
            device=target_token_ids.device,
            global_step=global_step,
        )
        hidden_states = anchor_context
        hidden_states = self.anchor_initializer(hidden_states)
        hidden_states = self._apply_anchor_dropout(
            hidden_states,
            probs=anchor_dropout_probs,
        )
        block_positions = torch.arange(self.block_size, device=hidden_states.device)
        hidden_states = hidden_states + self.block_position_embeddings(block_positions)
        hidden_states = self._apply_anchor_token_conditioning(
            hidden_states,
            anchor_token_ids,
        )
        hidden_states = self._apply_known_token_conditioning(
            hidden_states,
            known_token_ids,
        )

        local_mla_cache_rows = None
        local_mla_cache_valid_mask = None
        if not self.config.anchor_only:
            if mla_cache_rows is None:
                assert mla_cache_rows_packed is not None
                mla_cache_rows = dequantize_fp8_ds_mla_rows(
                    mla_cache_rows_packed,
                    dtype=hidden_states.dtype,
                )
            if mla_cache_valid_mask is not None:
                if mla_cache_valid_mask.shape != mla_cache_rows.shape[:3]:
                    raise ValueError(
                        "mla_cache_valid_mask shape mismatch: got "
                        f"{tuple(mla_cache_valid_mask.shape)}, expected "
                        f"{tuple(mla_cache_rows.shape[:3])}"
                    )
                mla_cache_valid_mask = mla_cache_valid_mask.to(
                    device=mla_cache_rows.device,
                    dtype=torch.bool,
                )
            if verifier_layer_ids is not None:
                if verifier_layer_ids.ndim == 2:
                    verifier_layer_ids = verifier_layer_ids[0]
                available = {
                    int(layer_id.item()): idx
                    for idx, layer_id in enumerate(verifier_layer_ids.to("cpu"))
                }
            else:
                if (
                    mla_cache_rows.shape[1] != len(self.layers)
                    or self.config.local_cross_attention_layer_ids
                ):
                    raise ValueError(
                        f"mla_cache_rows has {mla_cache_rows.shape[1]} layers, "
                        f"expected {len(self.layers)} and no local cross-attention "
                        "layers when verifier_layer_ids is omitted"
                    )
                available = {
                    int(layer_id): idx
                    for idx, layer_id in enumerate(self.config.verifier_kv_layer_ids)
                }

            def select_layer_tensor(
                tensor: torch.Tensor,
                layer_ids: list[int],
                *,
                name: str,
            ) -> torch.Tensor:
                layer_indices: list[int] = []
                for layer_id in layer_ids:
                    idx = available.get(int(layer_id))
                    if idx is None:
                        raise ValueError(
                            f"{name} does not contain requested verifier "
                            f"layer {layer_id}; available layers are "
                            f"{list(available)}"
                        )
                    layer_indices.append(idx)
                index = torch.tensor(layer_indices, device=tensor.device)
                return tensor.index_select(1, index)

            target_topk = int(self.config.sparse_topk)
            if target_topk <= 0:
                raise ValueError(f"sparse_topk must be > 0, got {target_topk}")
            if mla_cache_rows.shape[2] < target_topk:
                raise ValueError(
                    f"mla_cache_rows has topk {mla_cache_rows.shape[2]}, "
                    f"requested {target_topk}"
                )
            if mla_cache_rows.shape[2] > target_topk:
                mla_cache_rows = mla_cache_rows[:, :, :target_topk, :].contiguous()
            if mla_cache_valid_mask is not None:
                if mla_cache_valid_mask.shape[2] < target_topk:
                    raise ValueError(
                        "mla_cache_valid_mask has topk "
                        f"{mla_cache_valid_mask.shape[2]}, requested {target_topk}"
                    )
                if mla_cache_valid_mask.shape[2] > target_topk:
                    mla_cache_valid_mask = mla_cache_valid_mask[
                        :, :, :target_topk
                    ].contiguous()

            sparse_mla_cache_rows = select_layer_tensor(
                mla_cache_rows,
                list(self.config.verifier_kv_layer_ids),
                name="mla_cache_rows",
            )
            sparse_mla_cache_valid_mask = (
                select_layer_tensor(
                    mla_cache_valid_mask,
                    list(self.config.verifier_kv_layer_ids),
                    name="mla_cache_valid_mask",
                )
                if mla_cache_valid_mask is not None
                else None
            )
            if self.config.local_cross_attention_layer_ids:
                local_mla_cache_rows = select_layer_tensor(
                    mla_cache_rows,
                    list(self.config.local_cross_attention_layer_ids),
                    name="mla_cache_rows",
                )
                local_mla_cache_valid_mask = (
                    select_layer_tensor(
                        mla_cache_valid_mask,
                        list(self.config.local_cross_attention_layer_ids),
                        name="mla_cache_valid_mask",
                    )
                    if mla_cache_valid_mask is not None
                    else None
                )
            hidden_states = self._run_sparse_layers(
                hidden_states,
                sparse_mla_cache_rows,
                sparse_mla_cache_valid_mask,
                anchor_hidden_state=anchor_context,
                position_ids=position_ids,
                ablate_sparse_mla_cross_attention=ablate_sparse_mla_cross_attention,
            )
        hidden_states = self._run_local_layers(
            hidden_states,
            local_mla_cache_rows,
            local_mla_cache_valid_mask,
            anchor_hidden_state=anchor_context,
            position_ids=position_ids,
            ablate_sparse_mla_cross_attention=ablate_sparse_mla_cross_attention,
        )

        base_logits = self.lm_head(self.norm(hidden_states))
        base_logits = self._apply_slot1_verifier_head_bypass(
            base_logits,
            anchor_context,
        )
        logits = self._apply_markov_teacher_forcing(
            base_logits,
            target_token_ids,
            known_token_ids,
        )
        position_1_loss_weight = _linear_schedule_value(
            self.config.position_1_loss_weight_start,
            self.config.position_1_loss_weight_end,
            self.config.position_1_loss_weight_steps,
            global_step,
            device=logits.device,
            dtype=torch.float32,
        )
        loss, metrics = compute_sparse_mla_metrics(
            logits,
            target_token_ids,
            self._known_token_loss_mask(loss_mask),
            self.block_size,
            position_1_loss_weight=position_1_loss_weight,
        )
        metrics.update(self._collect_gate_metrics())
        metrics.update(self._collect_branch_metrics())
        metrics.update(self._anchor_dropout_metrics(anchor_dropout_probs))
        if self.markov_head is None:
            draft_tokens = torch.argmax(logits, dim=-1)
        else:
            draft_tokens = target_token_ids.detach().clone()
            draft_tokens[:, 1:] = self._markov_greedy_rollout(
                base_logits,
                (
                    target_token_ids[:, 0]
                    if known_token_ids is None
                    else known_token_ids
                ),
            )
        return draft_tokens, loss, metrics


class ServingSparseMLADraftModel(CanonicalSparseMLADraftModel):
    """Canonical glmflash model with a logits-only serving entry point.

    The architecture and parameter names intentionally come from glmflash, the
    training-side source of truth.  vLLM owns only the live-input partitioning
    and logits-only execution path here.
    """

    class _SharedVerifierLMHead(nn.Module):
        """Parameter-free view of the verifier's tensor-parallel LM head."""

        def __init__(self, vocab_size: int) -> None:
            super().__init__()
            self._head: nn.Module | None = None
            self.logits_processor = LogitsProcessor(vocab_size)
            # CausalCascade executes on every TP rank because the shared head
            # and row lookup both contain TP collectives. Keep the result on
            # every rank so optional iterative/token-fusion paths remain valid.
            self.logits_processor.use_all_gather = True

        def attach(self, head: nn.Module) -> None:
            self._head = head

        def _require_head(self) -> nn.Module:
            if self._head is None:
                raise RuntimeError(
                    "CausalCascade verifier LM head has not been attached"
                )
            return self._head

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            head = self._require_head()
            if isinstance(head, VocabParallelEmbedding):
                logits = self.logits_processor(head, hidden_states)
            else:
                logits = F.linear(hidden_states, head.weight)
            if logits is None:
                raise RuntimeError(
                    "CausalCascade shared LM-head projection returned no logits"
                )
            return logits

        def lookup_rows(self, token_ids: torch.Tensor) -> torch.Tensor:
            # ParallelLMHead deliberately uses its quantization method for the
            # output projection, and some linear quantization methods do not
            # implement the generic embedding API. Gather directly from the
            # existing sharded head weight instead; this is still the same
            # verifier parameter and does not materialize a copied head.
            head = self._require_head()
            if isinstance(head, VocabParallelEmbedding):
                if head.tp_size > 1:
                    masked_input, input_mask = get_masked_input_and_mask(
                        token_ids,
                        head.shard_indices.org_vocab_start_index,
                        head.shard_indices.org_vocab_end_index,
                        head.shard_indices.num_org_vocab_padding,
                        head.shard_indices.added_vocab_start_index,
                        head.shard_indices.added_vocab_end_index,
                    )
                else:
                    masked_input = token_ids
                output_parallel = F.embedding(masked_input.long(), head.weight)
                if head.tp_size > 1:
                    output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
                return tensor_model_parallel_all_reduce(output_parallel)
            return F.embedding(token_ids, head.weight)

        @property
        def weight(self) -> torch.Tensor:
            raise RuntimeError(
                "CausalCascade must access verifier LM-head rows through "
                "_lookup_lm_head_rows; a full replicated weight is not present"
            )

    def _init_sparse_mla_vocab(self, config: CanonicalSparseMLAConfig) -> None:
        """Initialize vocab metadata without allocating a dense LM-head copy."""
        tl_config = config.transformer_layer_config
        self.draft_vocab_size = config.draft_vocab_size
        self.verifier_vocab_size = tl_config.vocab_size
        self.hidden_size = tl_config.hidden_size
        self.use_draft_vocab = self.draft_vocab_size != self.verifier_vocab_size
        t2d: torch.Tensor | None = None
        d2t: torch.Tensor | None = None
        if self.use_draft_vocab:
            t2d = torch.zeros((self.verifier_vocab_size,), dtype=torch.bool)
            d2t = torch.zeros((self.draft_vocab_size,), dtype=torch.long)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)
        self.lm_head = self._SharedVerifierLMHead(self.draft_vocab_size)

    def __init__(self, config: PretrainedConfig) -> None:
        if not isinstance(config, CanonicalSparseMLAConfig):
            config = CanonicalSparseMLAConfig.model_validate(config.to_dict())
        super().__init__(config)

    def attach_verifier_lm_head(self, head: nn.Module) -> None:
        head_vocab_size = int(getattr(head, "org_vocab_size", head.weight.shape[0]))
        if head_vocab_size != int(self.draft_vocab_size):
            raise ValueError(
                "CausalCascade/verifier vocab mismatch: "
                f"draft={self.draft_vocab_size}, verifier={head_vocab_size}"
            )
        self.lm_head.attach(head)

    def _lookup_lm_head_rows(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head.lookup_rows(token_ids)

    def _partition_live_mla_rows(
        self,
        *,
        mla_cache_rows: torch.Tensor | None,
        mla_cache_rows_packed: torch.Tensor | None,
        mla_cache_valid_mask: torch.Tensor | None,
        verifier_layer_ids: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if self.config.anchor_only:
            return (None, None, None, None, None, None)
        if mla_cache_rows is None:
            if mla_cache_rows_packed is None:
                raise ValueError(
                    "CausalCascade requires mla_cache_rows or "
                    "mla_cache_rows_packed unless anchor_only is enabled"
                )
            from glmflash.models.dflash_sparse_mla.core import (
                dequantize_fp8_ds_mla_rows,
            )

            mla_cache_rows = dequantize_fp8_ds_mla_rows(
                mla_cache_rows_packed,
                dtype=dtype,
            )
        if mla_cache_valid_mask is not None:
            if mla_cache_valid_mask.shape != mla_cache_rows.shape[:3]:
                raise ValueError(
                    "mla_cache_valid_mask shape mismatch: got "
                    f"{tuple(mla_cache_valid_mask.shape)}, expected "
                    f"{tuple(mla_cache_rows.shape[:3])}"
                )
            mla_cache_valid_mask = mla_cache_valid_mask.to(
                device=mla_cache_rows.device,
                dtype=torch.bool,
            )

        if verifier_layer_ids is not None:
            if verifier_layer_ids.ndim == 2:
                verifier_layer_ids = verifier_layer_ids[0]
            available = {
                int(layer_id.item()): idx
                for idx, layer_id in enumerate(verifier_layer_ids.to("cpu"))
            }
        else:
            expected_layers = len(self.layers)
            if (
                mla_cache_rows.shape[1] != expected_layers
                or self.config.local_cross_attention_layer_ids
            ):
                raise ValueError(
                    f"mla_cache_rows has {mla_cache_rows.shape[1]} layers, "
                    f"expected {expected_layers} and no local cross-attention "
                    "layers when verifier_layer_ids is omitted"
                )
            available = {
                int(layer_id): idx
                for idx, layer_id in enumerate(self.config.verifier_kv_layer_ids)
            }

        target_topk = int(self.config.sparse_topk)
        if target_topk <= 0:
            raise ValueError(f"sparse_topk must be > 0, got {target_topk}")
        if mla_cache_rows.shape[2] < target_topk:
            raise ValueError(
                f"mla_cache_rows has topk {mla_cache_rows.shape[2]}, "
                f"requested {target_topk}"
            )
        if mla_cache_rows.shape[2] > target_topk:
            mla_cache_rows = mla_cache_rows[:, :, :target_topk, :].contiguous()
        if mla_cache_valid_mask is not None:
            if mla_cache_valid_mask.shape[2] < target_topk:
                raise ValueError(
                    "mla_cache_valid_mask has topk "
                    f"{mla_cache_valid_mask.shape[2]}, requested {target_topk}"
                )
            if mla_cache_valid_mask.shape[2] > target_topk:
                mla_cache_valid_mask = mla_cache_valid_mask[
                    :, :, :target_topk
                ].contiguous()

        def select_layer_tensor(
            tensor: torch.Tensor | None,
            layer_ids: list[int],
            *,
            name: str,
        ) -> torch.Tensor | None:
            if tensor is None:
                return None
            layer_indices: list[int] = []
            for layer_id in layer_ids:
                idx = available.get(int(layer_id))
                if idx is None:
                    raise ValueError(
                        f"{name} does not contain requested verifier layer "
                        f"{layer_id}; available layers are {list(available)}"
                    )
                layer_indices.append(idx)
            index = torch.tensor(layer_indices, device=tensor.device)
            return tensor.index_select(1, index)

        sparse_ids = list(self.config.verifier_kv_layer_ids)
        local_ids = list(self.config.local_cross_attention_layer_ids)
        sparse_rows = select_layer_tensor(
            mla_cache_rows,
            sparse_ids,
            name="mla_cache_rows",
        )
        sparse_mask = select_layer_tensor(
            mla_cache_valid_mask,
            sparse_ids,
            name="mla_cache_valid_mask",
        )
        local_rows = (
            select_layer_tensor(
                mla_cache_rows,
                local_ids,
                name="mla_cache_rows",
            )
            if local_ids
            else None
        )
        local_mask = (
            select_layer_tensor(
                mla_cache_valid_mask,
                local_ids,
                name="mla_cache_valid_mask",
            )
            if local_ids
            else None
        )

        refinement_rows = None
        refinement_mask = None
        if (
            int(getattr(self.config, "iterative_refinement_rounds", 1)) > 1
            and getattr(self.config, "iterative_refinement_mode", "shared") == "untied"
        ):
            refinement_ids = [
                int(getattr(self.config, "iterative_refinement_layer_id", 77))
            ]
            refinement_rows = select_layer_tensor(
                mla_cache_rows,
                refinement_ids,
                name="mla_cache_rows",
            )
            refinement_mask = select_layer_tensor(
                mla_cache_valid_mask,
                refinement_ids,
                name="mla_cache_valid_mask",
            )
        return (
            sparse_rows,
            sparse_mask,
            local_rows,
            local_mask,
            refinement_rows,
            refinement_mask,
        )

    def forward_logits(
        self,
        *,
        anchor_hidden_state: torch.Tensor,
        verifier_head_hidden_state: torch.Tensor | None = None,
        verifier_pre_norm_hidden_state: torch.Tensor | None = None,
        anchor_token_ids: torch.Tensor | None = None,
        mla_cache_rows: torch.Tensor | None = None,
        mla_cache_rows_packed: torch.Tensor | None = None,
        mla_cache_valid_mask: torch.Tensor | None = None,
        verifier_layer_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        known_token_ids: torch.Tensor | None = None,
        ablate_sparse_mla_cross_attention: bool = False,
    ) -> torch.Tensor:
        ablate_sparse_mla_cross_attention = ablate_sparse_mla_cross_attention or bool(
            getattr(
                self.config,
                "ablate_sparse_mla_cross_attention",
                False,
            )
        )
        batch_size = anchor_hidden_state.shape[0]
        hidden_size = self.hidden_size
        if anchor_hidden_state.shape != (batch_size, hidden_size):
            raise ValueError(
                "anchor_hidden_state shape mismatch: got "
                f"{tuple(anchor_hidden_state.shape)}, expected "
                f"{(batch_size, hidden_size)}"
            )
        anchor_token_ids = self._normalize_anchor_token_ids(
            anchor_token_ids,
            target_token_ids=None,
            batch_size=batch_size,
            device=anchor_hidden_state.device,
        )
        anchor_context = anchor_hidden_state.to(
            dtype=self.block_position_embeddings.weight.dtype,
        )
        hidden_states = self.anchor_initializer(anchor_context)
        if self.dual_stream_trunk:
            if verifier_head_hidden_state is None:
                raise ValueError(
                    "verifier_head_hidden_state is required for a dual-stream "
                    "CausalCascade checkpoint"
                )
            if verifier_head_hidden_state.shape != (batch_size, hidden_size):
                raise ValueError(
                    "verifier_head_hidden_state shape mismatch: got "
                    f"{tuple(verifier_head_hidden_state.shape)}, expected "
                    f"{(batch_size, hidden_size)}"
                )
            verifier_context = verifier_head_hidden_state.to(
                device=anchor_context.device,
                dtype=anchor_context.dtype,
            )
            assert self.verifier_anchor_initializer is not None
            hidden_states = torch.cat(
                [
                    hidden_states,
                    self.verifier_anchor_initializer(verifier_context),
                ],
                dim=-1,
            )
            dense_context_for_layers = torch.cat(
                [anchor_context, verifier_context],
                dim=-1,
            )
        else:
            dense_context_for_layers = anchor_context

        if self.verifier_hidden_residual is not None:
            if verifier_pre_norm_hidden_state is None:
                raise ValueError(
                    "verifier_pre_norm_hidden_state is required when "
                    "verifier_hidden_residual_rank > 0"
                )
            hidden_states = hidden_states + self.verifier_hidden_residual(
                verifier_pre_norm_hidden_state.to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            )

        anchor_dropout_probs = self._current_anchor_dropout_probs(
            device=anchor_hidden_state.device,
            global_step=None,
        )
        hidden_states = self._apply_anchor_dropout(
            hidden_states,
            probs=anchor_dropout_probs,
        )
        block_positions = torch.arange(self.block_size, device=hidden_states.device)
        hidden_states = hidden_states + self.block_position_embeddings(block_positions)
        hidden_states = self._apply_anchor_token_conditioning(
            hidden_states,
            anchor_token_ids,
        )
        hidden_states = self._apply_known_token_conditioning(
            hidden_states,
            known_token_ids,
        )

        (
            sparse_rows,
            sparse_mask,
            local_rows,
            local_mask,
            refinement_rows,
            refinement_mask,
        ) = self._partition_live_mla_rows(
            mla_cache_rows=mla_cache_rows,
            mla_cache_rows_packed=mla_cache_rows_packed,
            mla_cache_valid_mask=mla_cache_valid_mask,
            verifier_layer_ids=verifier_layer_ids,
            dtype=hidden_states.dtype,
        )
        if not self.config.anchor_only:
            assert sparse_rows is not None
            hidden_states = self._run_sparse_layers(
                hidden_states,
                sparse_rows,
                sparse_mask,
                anchor_hidden_state=dense_context_for_layers,
                position_ids=position_ids,
                ablate_sparse_mla_cross_attention=(ablate_sparse_mla_cross_attention),
            )
        hidden_states = self._run_local_layers(
            hidden_states,
            local_rows,
            local_mask,
            anchor_hidden_state=dense_context_for_layers,
            position_ids=position_ids,
            ablate_sparse_mla_cross_attention=ablate_sparse_mla_cross_attention,
        )

        base_logits = self._compute_base_logits(hidden_states, anchor_context)
        rounds = int(getattr(self.config, "iterative_refinement_rounds", 1))
        mode = getattr(self.config, "iterative_refinement_mode", "shared")
        for extra_round_idx in range(rounds - 1):
            candidate = self._prediction_conditioned_refinement_input(
                hidden_states,
                base_logits,
            )
            if mode == "shared":
                if not self.config.anchor_only:
                    assert sparse_rows is not None
                    candidate = self._run_sparse_layers(
                        candidate,
                        sparse_rows,
                        sparse_mask,
                        anchor_hidden_state=dense_context_for_layers,
                        position_ids=position_ids,
                        ablate_sparse_mla_cross_attention=(
                            ablate_sparse_mla_cross_attention
                        ),
                        active_start_override=2,
                    )
                candidate = self._run_local_layers(
                    candidate,
                    local_rows,
                    local_mask,
                    anchor_hidden_state=dense_context_for_layers,
                    position_ids=position_ids,
                    ablate_sparse_mla_cross_attention=(
                        ablate_sparse_mla_cross_attention
                    ),
                )
            else:
                if refinement_rows is None:
                    raise RuntimeError(
                        "untied iterative refinement is missing its MLA rows"
                    )
                candidate = self._run_untied_refinement_layer(
                    candidate,
                    self.iterative_refinement_layers[extra_round_idx],
                    refinement_rows[:, 0],
                    (refinement_mask[:, 0] if refinement_mask is not None else None),
                    anchor_hidden_state=dense_context_for_layers,
                    position_ids=position_ids,
                    ablate_sparse_mla_cross_attention=(
                        ablate_sparse_mla_cross_attention
                    ),
                )
            hidden_states, _, _ = self._blend_iterative_refinement(
                hidden_states,
                candidate,
                extra_round_idx=extra_round_idx,
            )
            base_logits = self._compute_base_logits(hidden_states, anchor_context)
        return base_logits


class CausalCascadeForCausalLM(nn.Module):
    """Native vLLM wrapper for the GLM CausalCascade draft model.

    The trainable architecture comes from glmflash so training and serving load
    the same module and parameter names. This wrapper only supplies vLLM's live
    inputs, target-weight population, and model-loader contract.
    """

    packed_modules_mapping = {}

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        del prefix
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.model = ServingSparseMLADraftModel(config)
        self.config = self.model.config
        native_mtp_config = SpeculativeConfig(
            method="mtp",
            num_speculative_tokens=1,
            target_model_config=vllm_config.model_config,
            target_parallel_config=vllm_config.parallel_config,
            quantization=vllm_config.model_config.quantization,
            draft_sample_method="greedy",
        )
        native_mtp_vllm_config = replace(
            vllm_config,
            speculative_config=native_mtp_config,
        )
        from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
            _create_draft_vllm_config,
        )

        native_mtp_vllm_config = _create_draft_vllm_config(native_mtp_vllm_config)
        # The verifier embedding and LM head are attached after both models are
        # loaded. Avoid allocating either shared matrix while constructing the
        # embedded MTP layer; this is the key memory advantage over nesting a
        # second native-MTP speculator.
        from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP

        self.embedded_mtp = DeepSeekMTP(
            vllm_config=native_mtp_vllm_config,
            # Keep the native checkpoint prefix for quantization-policy
            # matching (for example, GLM's BF16 sparse indexer exclusions).
            # The owning attribute still namespaces parameters as
            # ``embedded_mtp.*`` in this wrapper.
            prefix="",
            allocate_shared_weights=False,
        )

    @property
    def block_size(self) -> int:
        return int(self.model.block_size)

    @property
    def target_layer_ids(self) -> list[int]:
        return self.model.target_layer_ids

    @property
    def markov_head_enabled(self) -> bool:
        return self.model.markov_head_enabled

    @property
    def slot1_native_anchor_enabled(self) -> bool:
        return self.model.slot1_native_anchor_enabled

    def apply_markov_head(
        self,
        base_logits: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.apply_markov_head(base_logits, previous_token_ids)

    def forward_logits(
        self,
        *,
        anchor_hidden_state: torch.Tensor,
        verifier_head_hidden_state: torch.Tensor | None = None,
        verifier_pre_norm_hidden_state: torch.Tensor | None = None,
        anchor_token_ids: torch.Tensor | None = None,
        mla_cache_rows: torch.Tensor | None = None,
        mla_cache_rows_packed: torch.Tensor | None = None,
        mla_cache_valid_mask: torch.Tensor | None = None,
        verifier_layer_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        known_token_ids: torch.Tensor | None = None,
        ablate_sparse_mla_cross_attention: bool = False,
    ) -> torch.Tensor:
        return self.model.forward_logits(
            anchor_hidden_state=anchor_hidden_state,
            verifier_head_hidden_state=verifier_head_hidden_state,
            verifier_pre_norm_hidden_state=verifier_pre_norm_hidden_state,
            anchor_token_ids=anchor_token_ids,
            mla_cache_rows=mla_cache_rows,
            mla_cache_rows_packed=mla_cache_rows_packed,
            mla_cache_valid_mask=mla_cache_valid_mask,
            verifier_layer_ids=verifier_layer_ids,
            position_ids=position_ids,
            known_token_ids=known_token_ids,
            ablate_sparse_mla_cross_attention=ablate_sparse_mla_cross_attention,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedded_mtp.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Run the single embedded GLM MTP layer and return its anchor state."""
        return self.embedded_mtp(
            input_ids=input_ids,
            positions=positions,
            hidden_states=hidden_states,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(hidden_states)

    def load_weights(self, weights):
        causal_loader = AutoWeightsLoader(
            self.model,
            ignore_unexpected_prefixes=["verifier_lm_head"],
            ignore_unexpected_suffixes=["inv_freq"],
        )
        causal_loaded: set[str] = set()
        mtp_prefixes = self.embedded_mtp.checkpoint_weight_name_prefixes

        def mtp_weights():
            for raw_name, weight in weights:
                name = raw_name.removeprefix("causal_cascade.")
                mtp_name = name.removeprefix("embedded_mtp.")
                if any(mtp_name.startswith(prefix) for prefix in mtp_prefixes):
                    yield mtp_name, weight
                    continue
                # Historical training checkpoints contain a frozen 1.9GB
                # dense head. The live model intentionally owns no such
                # parameter: logits and token rows use the verifier's head.
                if name == "lm_head.weight":
                    continue
                loaded = causal_loader.load_weights([(name, weight)])
                causal_loaded.update(loaded)

        mtp_loaded = self.embedded_mtp.load_weights(mtp_weights())
        return {
            *(f"model.{name}" for name in causal_loaded),
            *(f"embedded_mtp.{name}" for name in mtp_loaded),
        }

    def attach_target_shared_weights(self, target_model: nn.Module) -> None:
        """Bind verifier-owned embedding/head matrices without copying them."""
        target_language_model = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        target_inner = getattr(target_language_model, "model", target_language_model)
        target_lm_head = getattr(target_language_model, "lm_head", None)
        if target_lm_head is None:
            target_lm_head = getattr(target_model, "lm_head", None)
        target_embed = getattr(target_inner, "embed_tokens", None)
        if target_embed is None:
            target_embed = getattr(target_inner, "embedding", None)
        if target_lm_head is None or target_embed is None:
            raise RuntimeError(
                "CausalCascade could not locate the verifier LM head and input "
                "embedding required by the embedded MTP layer"
            )

        self.model.attach_verifier_lm_head(target_lm_head)
        self.embedded_mtp.model.embed_tokens = target_embed
        for layer in self.embedded_mtp.model.layers.values():
            layer.shared_head.head = target_lm_head

        # Native GLM MTP shares the verifier's sparse-indexer output. Keep the
        # same aliasing when the MTP layer is embedded in CausalCascade.
        target_topk = getattr(target_inner, "topk_indices_buffer", None)
        if target_topk is not None:
            for module in self.embedded_mtp.model.modules():
                if hasattr(module, "topk_indices_buffer"):
                    module.topk_indices_buffer = target_topk

    def populate_target_compatible_mla_weights(self, target_model: nn.Module) -> None:
        """Populate frozen target-compatible MLA tensors from a loaded target.

        The current training checkpoint intentionally omits these frozen verifier
        tensors. This method handles the simple same-shape case and otherwise
        raises a clear error so we do not accidentally serve with random verifier
        query/value projections.
        """
        if self.model.config.cross_attention_impl != "target_compatible":
            return

        target_language_model = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        target_inner = getattr(target_language_model, "model", target_language_model)
        layers = getattr(target_inner, "layers", None)
        if layers is None:
            raise RuntimeError(
                "CausalCascade could not locate target model layers for "
                "target-compatible MLA weight population."
            )

        def materialize_tp_weight(
            *,
            layer_id: int,
            name: str,
            source: torch.Tensor,
            expected: torch.Tensor,
        ) -> torch.Tensor:
            source = source.detach()
            if source.shape == expected.shape:
                return source

            tp_group = get_tp_group()
            tp_size = tp_group.world_size
            if tp_size > 1:
                if (
                    source.ndim == 2
                    and expected.ndim == 2
                    and source.shape[1] == expected.shape[1]
                    and source.shape[0] * tp_size == expected.shape[0]
                ):
                    return tp_group.all_gather(source.contiguous(), dim=0)
                if (
                    source.ndim == 2
                    and expected.ndim == 2
                    and source.shape[0] == expected.shape[0]
                    and source.shape[1] * tp_size == expected.shape[1]
                ):
                    return tp_group.all_gather(source.contiguous(), dim=1)
                if (
                    source.ndim == 1
                    and expected.ndim == 1
                    and source.shape[0] * tp_size == expected.shape[0]
                ):
                    return tp_group.all_gather(source.contiguous(), dim=0)

            raise RuntimeError(
                "CausalCascade target-compatible MLA weight shape mismatch for "
                f"layer {layer_id} {name}: target model has {tuple(source.shape)}, "
                f"draft expects {tuple(expected.shape)}. The source tensor is not "
                f"a recognized TP shard for tp_size={tp_size}."
            )

        modules_by_layer: dict[int, list[CanonicalSparseMLACrossAttention]] = {}
        for module in self.model._iter_cross_attention_modules():
            if module.verifier_layer_id is None:
                continue
            modules_by_layer.setdefault(int(module.verifier_layer_id), []).append(
                module
            )

        for layer_id, modules in modules_by_layer.items():
            attn = layers[layer_id].self_attn
            q_a_weight = None
            if hasattr(attn, "q_a_proj"):
                q_a_weight = attn.q_a_proj.weight
            elif hasattr(attn, "fused_qkv_a_proj"):
                fused = attn.fused_qkv_a_proj.weight
                q_lora_rank = modules[0].q_lora_rank
                q_a_weight = fused[:q_lora_rank]
            if q_a_weight is None:
                raise RuntimeError(
                    "CausalCascade could not locate q_a weight on target layer "
                    f"{layer_id}."
                )

            weights = {
                "q_a_proj.weight": q_a_weight,
                "q_a_layernorm.weight": attn.q_a_layernorm.weight,
                "q_b_proj.weight": attn.q_b_proj.weight,
                "kv_b_proj.weight": attn.kv_b_proj.weight,
                "o_proj.weight": attn.o_proj.weight,
            }
            expected_weights = {
                "q_a_proj.weight": modules[0].target_q_a_proj_weight,
                "q_a_layernorm.weight": modules[0].target_q_a_layernorm_weight,
                "q_b_proj.weight": modules[0].target_q_b_proj_weight,
                "kv_b_proj.weight": modules[0].target_kv_b_proj_weight,
                "o_proj.weight": modules[0].target_o_proj_weight,
            }
            weights = {
                name: materialize_tp_weight(
                    layer_id=layer_id,
                    name=name,
                    source=source,
                    expected=expected_weights[name],
                )
                for name, source in weights.items()
            }
            for module in modules:
                for name, target in (
                    ("q_a_proj.weight", module.target_q_a_proj_weight),
                    ("q_a_layernorm.weight", module.target_q_a_layernorm_weight),
                    ("q_b_proj.weight", module.target_q_b_proj_weight),
                    ("kv_b_proj.weight", module.target_kv_b_proj_weight),
                    ("o_proj.weight", module.target_o_proj_weight),
                ):
                    source = weights[name]
                    if source.shape != target.shape:
                        raise RuntimeError(
                            "CausalCascade target-compatible MLA materialization "
                            f"failed for layer {layer_id} {name}: source "
                            f"has {tuple(source.shape)}, draft expects "
                            f"{tuple(target.shape)}."
                        )
                module.load_target_compatible_weights(weights)
