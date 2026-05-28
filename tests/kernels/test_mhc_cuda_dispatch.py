# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.mhc as mhc
from vllm.model_executor.layers.mhc import MHCPreOp
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
