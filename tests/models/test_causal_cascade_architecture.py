# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

pytest.importorskip("glmflash")

from glmflash.models.dflash_sparse_mla.config import (  # noqa: E402
    DFlashSparseMLASpeculatorConfig,
)

from vllm.model_executor.models.causal_cascade import (  # noqa: E402
    ServingSparseMLADraftModel,
)


def _tiny_latest_architecture_config() -> DFlashSparseMLASpeculatorConfig:
    return DFlashSparseMLASpeculatorConfig(
        transformer_layer_config=Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
        ),
        draft_vocab_size=32,
        block_size=4,
        anchor_hidden_state_source=("native_glm_mtp_post_shared_head_rmsnorm"),
        verifier_kv_layer_ids=[3],
        sparse_topk=5,
        mla_kv_lora_rank=8,
        mla_qk_rope_head_dim=2,
        mla_qk_nope_head_dim=4,
        mla_q_lora_rank=8,
        mla_v_head_dim=4,
        mla_num_heads=2,
        query_ensemble_size=2,
        cross_attention_impl="target_compatible",
        extra_local_layers=1,
        local_cross_attention_layer_ids=[2],
        dual_stream_trunk=True,
        query_position_adapter_rank=4,
        query_position_output_adapter_rank=2,
        known_token_conditioning="lm_head",
        slot1_mtp_residual_rank=3,
        markov_head_rank=3,
        ce_loss_alpha=1.0,
        l1_loss_alpha=0.0,
    )


def test_latest_architecture_serving_logits_match_training_path() -> None:
    model = ServingSparseMLADraftModel(_tiny_latest_architecture_config())
    verifier_lm_head = torch.nn.Linear(16, 32, bias=False)
    model.attach_verifier_lm_head(verifier_lm_head)
    model = model.float()
    model.eval()

    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        # Frozen verifier projections are populated by the target model in
        # serving and deliberately start as NaNs. Give the tiny test finite
        # stand-ins without disturbing already initialized parameters.
        for tensor in [*model.parameters(), *model.buffers()]:
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                replacement = torch.randn(
                    tensor.shape,
                    generator=generator,
                    dtype=tensor.dtype,
                )
                tensor.copy_(torch.where(torch.isfinite(tensor), tensor, replacement))
        assert model.markov_head is not None
        model.markov_head.projection.weight.normal_(generator=generator)

    batch_size = 2
    known_token_ids = torch.tensor([4, 5])
    target_token_ids = torch.randint(0, 32, (batch_size, 4))
    target_token_ids[:, 0] = known_token_ids
    loss_mask = torch.ones_like(target_token_ids, dtype=torch.bool)
    loss_mask[:, 0] = False
    kwargs = {
        "anchor_hidden_state": torch.randn(batch_size, 16),
        "verifier_head_hidden_state": torch.randn(batch_size, 16),
        "verifier_pre_norm_hidden_state": torch.randn(batch_size, 16),
        "anchor_token_ids": known_token_ids,
        "mla_cache_rows": torch.randn(batch_size, 2, 5, 10),
        "verifier_layer_ids": torch.tensor([3, 2]),
        "position_ids": torch.arange(4).expand(batch_size, -1) + 11,
        "known_token_ids": known_token_ids,
    }

    with torch.no_grad():
        base_logits = model.forward_logits(**kwargs)
        _, _, _, training_logits = model(
            **kwargs,
            target_token_ids=target_token_ids,
            loss_mask=loss_mask,
            return_logits=True,
        )
        expected_logits = model._apply_markov_teacher_forcing(
            base_logits,
            target_token_ids,
            known_token_ids,
        )

    torch.testing.assert_close(training_logits, expected_logits)
    # The native MTP residual owns slot 1, so the Markov head begins at slot 2.
    torch.testing.assert_close(expected_logits[:, 1], base_logits[:, 1])
    assert not torch.equal(expected_logits[:, 2:], base_logits[:, 2:])
    assert base_logits.shape == (batch_size, 4, 32)
    assert torch.isfinite(base_logits).all()
    assert model.trunk_hidden_size == 32
    assert model.config.query_ensemble_size == 2
    assert model.slot1_native_anchor_enabled
    assert model.markov_head_enabled
    assert model.lm_head._head is verifier_lm_head
