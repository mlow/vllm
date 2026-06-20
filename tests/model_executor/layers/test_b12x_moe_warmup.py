# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

import vllm.model_executor.layers.fused_moe.b12x_moe as b12x_moe
import vllm.model_executor.layers.fused_moe.runner.moe_runner as moe_runner
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


class _FakePlan:
    @staticmethod
    def scratch_specs():
        return [SimpleNamespace(dtype=torch.uint8, shape=(64,))]


def _make_fake_b12x_experts() -> b12x_moe.B12xExperts:
    num_experts = 8
    experts = object.__new__(b12x_moe.B12xExperts)
    experts.moe_config = SimpleNamespace(
        in_dtype=torch.bfloat16,
        experts_per_token=4,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        swiglu_limit=7.0,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
    )
    experts.quant_config = SimpleNamespace(
        quant_dtype="nvfp4",
        weight_quant_dtype="nvfp4",
        gemm1_clamp_limit=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        a1_gscale=torch.ones(num_experts, dtype=torch.float32),
        a2_gscale=torch.ones(num_experts, dtype=torch.float32),
        w1_scale=torch.ones(num_experts, 16, 16, dtype=torch.float32),
        w2_scale=torch.ones(num_experts, 16, 16, dtype=torch.float32),
        g1_alphas=torch.ones(num_experts, dtype=torch.float32),
        g2_alphas=torch.ones(num_experts, dtype=torch.float32),
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
        per_act_token_quant=False,
        per_out_ch_quant=False,
        w1_zp=None,
        w2_zp=None,
        w1_bias=None,
        w2_bias=None,
    )
    experts._prepared_fp4_moe_by_dtype = {
        torch.bfloat16: SimpleNamespace(
            w1_runtime_alphas=torch.ones(num_experts, dtype=torch.float32),
            w2_runtime_alphas=torch.ones(num_experts, dtype=torch.float32),
            w4a16=None,
            w4a8_tier=None,
        )
    }
    experts._source_params_compacted = False
    experts._unit_scale_by_device = {}
    return experts


def _make_fake_moe_runner(fused_experts: object) -> MoERunner:
    runner = object.__new__(MoERunner)
    runner._shared_experts = None
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(
            moe_kernel=SimpleNamespace(fused_experts=fused_experts)
        )
    )
    return runner


def test_b12x_moe_runner_uses_functional_custom_op(monkeypatch) -> None:
    monkeypatch.setattr(
        moe_runner,
        "current_platform",
        SimpleNamespace(is_tpu=lambda: False, is_cpu=lambda: False),
    )

    runner = _make_fake_moe_runner(object.__new__(b12x_moe.B12xExperts))

    forward_entry = runner._select_forward()

    assert forward_entry._qualified_op_name == "vllm::b12x_moe_forward"


def test_non_b12x_moe_runner_keeps_generic_custom_op(monkeypatch) -> None:
    monkeypatch.setattr(
        moe_runner,
        "current_platform",
        SimpleNamespace(is_tpu=lambda: False, is_cpu=lambda: False),
    )

    runner = _make_fake_moe_runner(object())

    forward_entry = runner._select_forward()

    assert forward_entry._qualified_op_name == "vllm::moe_forward"


def test_b12x_moe_custom_op_matches_generic_mutation_contract() -> None:
    b12x_schema = str(torch.ops.vllm.b12x_moe_forward.default._schema)
    b12x_shared_schema = str(torch.ops.vllm.b12x_moe_forward_shared.default._schema)
    generic_schema = str(torch.ops.vllm.moe_forward.default._schema)
    generic_shared_schema = str(torch.ops.vllm.moe_forward_shared.default._schema)

    assert "Tensor(a0!) hidden_states" in generic_schema
    assert "Tensor(a0!) hidden_states" in generic_shared_schema
    assert "Tensor(a0!) hidden_states" in b12x_schema
    assert "Tensor(a0!) hidden_states" in b12x_shared_schema


def test_b12x_moe_warmup_uses_minimax_swiglu_params(monkeypatch) -> None:
    plan_calls = []
    run_calls = []

    def fake_plan(**kwargs):
        plan_calls.append(kwargs)
        return _FakePlan()

    def fake_run(**kwargs):
        run_calls.append(kwargs)

    monkeypatch.setattr(b12x_moe, "_plan_b12x_moe_fp4_scratch", fake_plan)
    monkeypatch.setattr(b12x_moe, "_run_b12x_moe_fp4", fake_run)
    monkeypatch.setattr(
        b12x_moe,
        "_dynamic_moe_warmup_tokens",
        lambda *, topk, quant_mode, requested_tokens: 7,
    )

    experts = _make_fake_b12x_experts()
    layer = SimpleNamespace(
        w13_weight=torch.empty(8, 32, 32, dtype=torch.uint8),
        w2_weight=torch.empty(8, 64, 8, dtype=torch.uint8),
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        apply_router_weight_on_input=False,
    )

    experts.warmup_dynamic_launch(layer, tokens=3)

    assert len(plan_calls) == 1
    assert len(run_calls) == 1
    assert plan_calls[0]["tokens"] == 7
    assert plan_calls[0]["quant_mode"] == "nvfp4"
    assert plan_calls[0]["source_format"] == "modelopt_nvfp4"
    assert plan_calls[0]["w13_layout"] == "w31"
    assert plan_calls[0]["activation"] == "swigluoai_uninterleave"
    assert plan_calls[0]["swiglu_limit"] == 7.0
    assert plan_calls[0]["swiglu_alpha"] == 1.702
    assert plan_calls[0]["swiglu_beta"] == 1.0

    assert run_calls[0]["a"].shape == (7, 64)
    assert run_calls[0]["output"].shape == (7, 64)
    assert run_calls[0]["topk_ids"].dtype == torch.int32
    assert run_calls[0]["topk_weights"].dtype == torch.float32
    assert run_calls[0]["scratch"].dtype == torch.uint8
    assert run_calls[0]["scratch"].numel() == 64
    assert run_calls[0]["input_scales_are_reciprocal"] is True


def test_b12x_force_a16_nvfp4_selects_w4a16(monkeypatch) -> None:
    monkeypatch.setenv("B12X_MOE_FORCE_A16", "1")

    experts = _make_fake_b12x_experts()

    assert experts._quant_mode() == "w4a16"


def test_b12x_force_a8_mxfp4_prepares_w4a8_tier(monkeypatch) -> None:
    monkeypatch.setenv("B12X_MOE_FORCE_A16", "1")
    monkeypatch.setenv("B12X_FORCE_MOE_A8", "1")

    calls = []
    w4a8_tier = SimpleNamespace(
        num_experts=8,
        hidden_size=256,
        intermediate_size=128,
        params_dtype=torch.bfloat16,
        w13_rp=torch.empty((1,), dtype=torch.uint8),
    )

    def fake_prepare(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            source_format=kwargs["source_format"],
            w13_layout=kwargs["w13_layout"],
            w1_runtime_alphas=None,
            w2_runtime_alphas=None,
            w4a16=None,
            w4a8_tier=w4a8_tier,
        )

    monkeypatch.setattr(b12x_moe, "_prepare_b12x_fp4_moe_weights", fake_prepare)

    experts = _make_fake_b12x_experts()
    experts.quant_config.quant_dtype = "mxfp4"
    experts.quant_config.weight_quant_dtype = "mxfp4"
    experts.quant_config.g1_alphas = None
    experts.quant_config.g2_alphas = None
    experts.quant_config.a1_gscale = None
    experts.quant_config.a2_gscale = None
    experts.quant_config.w1_scale = torch.empty((8, 256, 8), dtype=torch.uint8)
    experts.quant_config.w2_scale = torch.empty((8, 256, 4), dtype=torch.uint8)
    w1 = torch.empty((8, 256, 128), dtype=torch.uint8)
    w2 = torch.empty((8, 256, 64), dtype=torch.uint8)

    prepared = experts._get_or_prepare_fp4_moe_weights(
        w1=w1,
        w2=w2,
        activation=MoEActivation.SILU,
        params_dtype=torch.bfloat16,
    )

    assert experts._quant_mode() == "w4a8_mx"
    assert prepared.w4a8_tier is w4a8_tier
    assert len(calls) == 1
    assert calls[0]["source_format"] == "fp4_e8m0_k32"
    assert calls[0]["w13_layout"] == "w31"
    assert calls[0]["prepare_runtime_alphas"] is False
    assert calls[0]["prepare_w4a16"] is False
    assert calls[0]["prepare_w4a8_tier"] is True
    assert calls[0]["reuse_input_storage"] is True
    assert torch.equal(calls[0]["w1_global_scale"], torch.ones(8))
    assert torch.equal(calls[0]["w2_global_scale"], torch.ones(8))


def test_warmup_b12x_moe_dynamic_dedupes_signatures(monkeypatch) -> None:
    calls = []

    def fake_signature(self, layer):
        return ("same-signature",)

    def fake_warmup(self, layer, *, tokens):
        calls.append((self, layer, tokens))

    monkeypatch.setattr(
        b12x_moe.B12xExperts,
        "warmup_dynamic_signature",
        fake_signature,
    )
    monkeypatch.setattr(
        b12x_moe.B12xExperts,
        "warmup_dynamic_launch",
        fake_warmup,
    )

    experts_0 = object.__new__(b12x_moe.B12xExperts)
    experts_1 = object.__new__(b12x_moe.B12xExperts)
    modules = [
        SimpleNamespace(
            routed_experts=SimpleNamespace(
                quant_method=SimpleNamespace(
                    moe_kernel=SimpleNamespace(fused_experts=experts_0)
                )
            )
        ),
        SimpleNamespace(
            routed_experts=SimpleNamespace(
                quant_method=SimpleNamespace(
                    moe_kernel=SimpleNamespace(fused_experts=experts_1)
                )
            )
        ),
    ]
    model = SimpleNamespace(modules=lambda: iter(modules))

    warmed = b12x_moe.warmup_b12x_moe_dynamic(model, tokens=5)

    assert warmed == 1
    assert len(calls) == 1
    assert calls[0][2] == 5
