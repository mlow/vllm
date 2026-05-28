# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.mhc as mhc
from vllm.model_executor.layers.mhc import (
    HCHeadOp,
    MHCFusedPostPreOp,
    MHCPostOp,
    MHCPreOp,
)
from vllm.platforms import current_platform

DEVICE = current_platform.device_type


def _clear_cuda_tilelang_failures() -> None:
    failures = getattr(mhc, "_FAILED_CUDA_TILELANG_OPS", None)
    if failures is not None:
        failures.clear()
    verified = getattr(mhc, "_VERIFIED_CUDA_TILELANG_OPS", None)
    if verified is not None:
        verified.clear()
    warmed = getattr(mhc, "_WARMED_CUDA_TILELANG_CONFIGS", None)
    if warmed is not None:
        warmed.clear()


def _mhc_args() -> tuple[torch.Tensor, ...]:
    num_tokens = 1
    hidden_size = 4
    hc_mult = 2
    hc_mult3 = 2 * hc_mult + hc_mult * hc_mult
    residual = torch.zeros((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    x = torch.zeros((num_tokens, hidden_size), dtype=torch.bfloat16)
    post_mix = torch.zeros((num_tokens, hc_mult, 1), dtype=torch.float32)
    comb_mix = torch.zeros((num_tokens, hc_mult, hc_mult), dtype=torch.float32)
    fn = torch.zeros((hc_mult3, hc_mult * hidden_size), dtype=torch.float32)
    hc_scale = torch.zeros((3,), dtype=torch.float32)
    hc_base = torch.zeros((hc_mult3,), dtype=torch.float32)
    return residual, x, post_mix, comb_mix, fn, hc_scale, hc_base


def _hc_head_args() -> tuple[torch.Tensor, ...]:
    hidden_size = 4
    hc_mult = 2
    hidden_states = torch.zeros((1, hc_mult, hidden_size), dtype=torch.bfloat16)
    hc_fn = torch.zeros((hc_mult, hc_mult * hidden_size), dtype=torch.float32)
    hc_scale = torch.zeros((1,), dtype=torch.float32)
    hc_base = torch.zeros((hc_mult,), dtype=torch.float32)
    return hidden_states, hc_fn, hc_scale, hc_base


def test_cuda_dispatch_prefers_tilelang_ops(monkeypatch, default_vllm_config):
    _clear_cuda_tilelang_failures()
    monkeypatch.setattr(mhc, "HAS_TILELANG", True)

    residual, x, post_mix, comb_mix, fn, hc_scale, hc_base = _mhc_args()
    hidden_states, hc_fn, head_scale, head_base = _hc_head_args()
    pre_result = (torch.tensor(1), torch.tensor(2), torch.tensor(3))
    post_result = torch.tensor(4)
    fused_result = (torch.tensor(5), torch.tensor(6), torch.tensor(7), torch.tensor(8))
    calls: list[str] = []

    def fake_mhc_pre_tilelang(*args):
        calls.append("mhc_pre")
        return pre_result

    def fake_mhc_post_tilelang(*args):
        calls.append("mhc_post")
        return post_result

    def fake_mhc_fused_post_pre_tilelang(*args):
        calls.append("mhc_fused_post_pre")
        return fused_result

    def fake_hc_head_tilelang(
        hs_flat,
        hc_fn,
        hc_scale,
        hc_base,
        out,
        hidden_size,
        rms_norm_eps,
        hc_eps,
        hc_mult,
    ):
        calls.append("hc_head")
        out.fill_(9)

    fake_ops = SimpleNamespace(
        mhc_pre_tilelang=fake_mhc_pre_tilelang,
        mhc_post_tilelang=fake_mhc_post_tilelang,
        mhc_fused_post_pre_tilelang=fake_mhc_fused_post_pre_tilelang,
        hc_head_fused_kernel_tilelang=fake_hc_head_tilelang,
    )
    monkeypatch.setattr(mhc.torch.ops, "vllm", fake_ops, raising=False)
    monkeypatch.setattr(
        mhc.mhc_kernels,
        "mhc_pre_torch",
        lambda *args: (_ for _ in ()).throw(AssertionError("native pre called")),
    )
    monkeypatch.setattr(
        mhc.mhc_kernels,
        "mhc_post_torch",
        lambda *args: (_ for _ in ()).throw(AssertionError("native post called")),
    )
    monkeypatch.setattr(
        mhc,
        "_hc_head_cuda_impl",
        lambda *args: (_ for _ in ()).throw(AssertionError("triton head called")),
    )

    assert MHCPreOp().forward_cuda(
        residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 1.0, 20
    ) is pre_result
    assert MHCPostOp().forward_cuda(x, residual, post_mix, comb_mix) is post_result
    assert (
        MHCFusedPostPreOp().forward_cuda(
            x,
            residual,
            post_mix,
            comb_mix,
            fn,
            hc_scale,
            hc_base,
            1e-6,
            1e-6,
            1e-6,
            1.0,
            20,
        )
        is fused_result
    )

    head = HCHeadOp().forward_cuda(
        hidden_states, hc_fn, head_scale, head_base, 1e-6, 1e-6
    )
    assert torch.equal(head, torch.full((1, 4), 9, dtype=torch.bfloat16))
    assert calls == ["mhc_pre", "mhc_post", "mhc_fused_post_pre", "hc_head"]


def test_cuda_dispatch_falls_back_after_tilelang_failure(
    monkeypatch,
    default_vllm_config,
):
    _clear_cuda_tilelang_failures()
    monkeypatch.setattr(mhc, "HAS_TILELANG", True)

    residual, _, _, _, fn, hc_scale, hc_base = _mhc_args()
    post_result = torch.tensor(1)
    comb_result = torch.tensor(2)
    layer_input_result = torch.tensor(3)
    calls = {"tilelang": 0, "native": 0}

    def failing_mhc_pre_tilelang(*args):
        calls["tilelang"] += 1
        raise RuntimeError("tilelang compile failed")

    def fake_forward_native(self, *args):
        calls["native"] += 1
        return post_result, comb_result, layer_input_result

    fake_ops = SimpleNamespace(mhc_pre_tilelang=failing_mhc_pre_tilelang)
    monkeypatch.setattr(mhc.torch.ops, "vllm", fake_ops, raising=False)
    monkeypatch.setattr(MHCPreOp, "forward_native", fake_forward_native)

    first = MHCPreOp().forward_cuda(
        residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 1.0, 20
    )
    second = MHCPreOp().forward_cuda(
        residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 1.0, 20
    )

    assert first[0] is post_result
    assert first[1] is comb_result
    assert first[2] is layer_input_result
    assert second[0] is post_result
    assert second[1] is comb_result
    assert second[2] is layer_input_result
    assert calls == {"tilelang": 1, "native": 2}


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="CUDA required",
)
def test_cuda_tilelang_warmup_requires_complete_mhc_group(
    monkeypatch,
    default_vllm_config,
):
    _clear_cuda_tilelang_failures()
    monkeypatch.setattr(mhc, "HAS_TILELANG", True)

    calls: list[str] = []

    def failing_mhc_pre_tilelang(*args):
        calls.append("mhc_pre")
        raise RuntimeError("tilelang compile failed")

    def fake_mhc_post_tilelang(*args):
        calls.append("mhc_post")
        return torch.empty((1, 2, 4), device=DEVICE, dtype=torch.bfloat16)

    def fake_mhc_fused_post_pre_tilelang(*args):
        calls.append("mhc_fused_post_pre")
        return (
            torch.empty((1, 2, 4), device=DEVICE, dtype=torch.bfloat16),
            torch.empty((1, 2, 1), device=DEVICE, dtype=torch.float32),
            torch.empty((1, 2, 2), device=DEVICE, dtype=torch.float32),
            torch.empty((1, 4), device=DEVICE, dtype=torch.bfloat16),
        )

    def fake_hc_head_tilelang(*args):
        calls.append("hc_head")

    fake_ops = SimpleNamespace(
        mhc_pre_tilelang=failing_mhc_pre_tilelang,
        mhc_post_tilelang=fake_mhc_post_tilelang,
        mhc_fused_post_pre_tilelang=fake_mhc_fused_post_pre_tilelang,
        hc_head_fused_kernel_tilelang=fake_hc_head_tilelang,
    )
    monkeypatch.setattr(mhc.torch.ops, "vllm", fake_ops, raising=False)

    mhc.warm_up_cuda_tilelang_mhc(4, 2, torch.bfloat16, DEVICE)

    assert calls == ["mhc_pre", "mhc_post", "mhc_fused_post_pre", "hc_head"]
    assert set() == mhc._VERIFIED_CUDA_TILELANG_OPS
    assert {
        ("mhc_pre", 4, 2),
        ("mhc_post", 4, 2),
        ("mhc_fused_post_pre", 4, 2),
        ("hc_head", 4, 2),
    } == mhc._FAILED_CUDA_TILELANG_OPS


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="CUDA required",
)
def test_cuda_dispatch_fallback_survives_torch_compile(default_vllm_config):
    _clear_cuda_tilelang_failures()
    torch.set_default_device(DEVICE)

    num_tokens = 1
    hidden_size = 7168
    hc_mult = 4
    hc_mult3 = 2 * hc_mult + hc_mult * hc_mult
    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    fn = torch.randn((hc_mult3, hc_mult * hidden_size), dtype=torch.float32) * 1e-4
    hc_scale = torch.randn((3,), dtype=torch.float32) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float32) * 0.1
    op = MHCPreOp()

    def run(residual, fn, hc_scale, hc_base):
        return op.forward_cuda(
            residual,
            fn,
            hc_scale,
            hc_base,
            1e-6,
            1e-6,
            1e-6,
            1.0,
            20,
        )

    out = torch.compile(
        run, backend=current_platform.simple_compile_backend
    )(residual, fn, hc_scale, hc_base)
    torch.cuda.synchronize()

    assert tuple(out[0].shape) == (num_tokens, hc_mult, 1)
    assert tuple(out[1].shape) == (num_tokens, hc_mult, hc_mult)
    assert tuple(out[2].shape) == (num_tokens, hidden_size)
