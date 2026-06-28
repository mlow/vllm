# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
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
    experts._activation_amax_base_num_layers = None
    experts._activation_amax_state_key = None
    experts._activation_amax_layer_idx = None
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


def test_b12x_activation_amax_registers_stable_vllm_owned_tensor(
    monkeypatch,
) -> None:
    b12x_moe._reset_b12x_moe_activation_amax_for_tests()
    monkeypatch.setenv("B12X_MOE_FORCE_A16", "1")
    monkeypatch.setenv("VLLM_B12X_MOE_ACTIVATION_AMAX", "1")

    experts = _make_fake_b12x_experts()
    experts._register_activation_amax(
        layer=SimpleNamespace(layer_name="model.layers.3.mlp.experts"),
        device=torch.device("cpu"),
        num_experts=8,
    )

    activation_amax, layer_idx = experts._activation_amax_args(
        device=torch.device("cpu"),
        num_experts=8,
    )

    assert activation_amax is not None
    assert activation_amax.shape == (4, 8, 2)
    assert activation_amax.dtype == torch.float32
    assert layer_idx == 3
    data_ptr = activation_amax.data_ptr()

    late_experts = _make_fake_b12x_experts()
    with pytest.raises(RuntimeError, match="would reallocate after use"):
        late_experts._register_activation_amax(
            layer=SimpleNamespace(layer_name="model.layers.4.mlp.experts"),
            device=torch.device("cpu"),
            num_experts=8,
        )
    assert activation_amax.data_ptr() == data_ptr


def test_b12x_activation_amax_is_passed_as_separate_binding_arg(
    monkeypatch,
) -> None:
    b12x_moe._reset_b12x_moe_activation_amax_for_tests()
    monkeypatch.setenv("B12X_MOE_FORCE_A16", "1")
    monkeypatch.setenv("VLLM_B12X_MOE_ACTIVATION_AMAX", "1")

    plan_calls = []
    run_calls = []

    def fake_plan(**kwargs):
        plan_calls.append(kwargs)
        return _FakePlan()

    def fake_run(**kwargs):
        run_calls.append(kwargs)

    monkeypatch.setattr(b12x_moe, "_plan_b12x_moe_fp4_scratch", fake_plan)
    monkeypatch.setattr(b12x_moe, "_run_b12x_moe_fp4", fake_run)

    num_experts = 8
    hidden_size = 16
    experts = _make_fake_b12x_experts()
    experts._prepared_fp4_moe_by_dtype[torch.bfloat16].w4a16 = SimpleNamespace(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=32,
        w13=torch.empty(1),
    )
    experts._register_activation_amax(
        layer=SimpleNamespace(layer_name="model.layers.2.mlp.experts"),
        device=torch.device("cpu"),
        num_experts=num_experts,
    )

    hidden_states = torch.zeros(3, hidden_size, dtype=torch.bfloat16)
    output = torch.empty_like(hidden_states)
    topk_ids = torch.zeros(3, 4, dtype=torch.int32)
    topk_weights = torch.full((3, 4), 0.25, dtype=torch.float32)
    workspace2 = torch.empty(64, dtype=torch.uint8)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=torch.empty(0, dtype=torch.uint8),
        w2=torch.empty(0, dtype=torch.uint8),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        global_num_experts=num_experts,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert len(plan_calls) == 1
    assert len(run_calls) == 1
    assert plan_calls[0]["collect_activation_amax"] is True
    assert run_calls[0]["activation_amax"] is not workspace2
    assert run_calls[0]["activation_amax"].shape == (3, num_experts, 2)
    assert run_calls[0]["layer_idx"] == 2


def test_b12x_activation_amax_save_every_writes_main_and_mtp_files(
    monkeypatch,
    tmp_path,
) -> None:
    b12x_moe._reset_b12x_moe_activation_amax_for_tests()
    monkeypatch.setenv("B12X_MOE_FORCE_A16", "1")
    monkeypatch.setenv("VLLM_B12X_MOE_ACTIVATION_AMAX", "1")
    monkeypatch.setenv("VLLM_B12X_MOE_ACTIVATION_AMAX_SAVE_EVERY", "2")
    monkeypatch.setenv("VLLM_B12X_MOE_ACTIVATION_AMAX_FILE", str(tmp_path / "amax.pt"))

    main = _make_fake_b12x_experts()
    main._activation_amax_base_num_layers = 60
    main._register_activation_amax(
        layer=SimpleNamespace(layer_name="model.layers.3.mlp.experts"),
        device=torch.device("cpu"),
        num_experts=8,
    )
    mtp = _make_fake_b12x_experts()
    mtp._activation_amax_base_num_layers = 60
    mtp._register_activation_amax(
        layer=SimpleNamespace(layer_name="model.layers.60.mlp.experts"),
        device=torch.device("cpu"),
        num_experts=8,
    )

    main_amax, main_layer = main._activation_amax_args(
        device=torch.device("cpu"),
        num_experts=8,
    )
    mtp_amax, mtp_layer = mtp._activation_amax_args(
        device=torch.device("cpu"),
        num_experts=8,
    )
    assert main_amax is not None and mtp_amax is not None
    assert main_layer == 3
    assert mtp_layer == 0
    main_amax[main_layer, 1, 0] = 11.0
    mtp_amax[mtp_layer, 2, 1] = 17.0

    b12x_moe.maybe_save_b12x_moe_activation_amax()
    assert not list(tmp_path.glob("*.pt"))

    b12x_moe.maybe_save_b12x_moe_activation_amax()
    files = sorted(tmp_path.glob("*.pt"))
    assert len(files) == 2
    loaded = [torch.load(path, weights_only=False) for path in files]
    payloads = {payload["model"]: payload for payload in loaded}
    assert set(payloads) == {"main", "mtp"}
    assert payloads["main"]["activation_amax"][3, 1, 0] == 11.0
    assert payloads["mtp"]["activation_amax"][0, 2, 1] == 17.0
    assert payloads["main"]["layers"][3]["prefix"] == "model.layers.3.mlp.experts"
    assert payloads["mtp"]["layers"][0]["external_layer_idx"] == 60


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
