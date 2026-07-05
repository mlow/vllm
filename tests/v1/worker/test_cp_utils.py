# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.worker import cp_utils


def _make_config(*, dcp_size: int = 1, pcp_size: int = 1):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
            cp_kv_cache_interleave_size=1,
        ),
        speculative_config=None,
    )


def test_check_attention_cp_compatibility_enables_lse_return(monkeypatch):
    impl = SimpleNamespace(
        can_return_lse_for_decode=True,
        need_to_return_lse_for_decode=False,
        supports_pcp=False,
    )
    layer = SimpleNamespace(impl=impl)

    monkeypatch.setattr(
        cp_utils,
        "get_layers_from_vllm_config",
        lambda vllm_config, layer_type: {"layer": layer},
    )

    cp_utils.check_attention_cp_compatibility(_make_config(dcp_size=2))

    assert impl.need_to_return_lse_for_decode is True


def test_check_attention_cp_compatibility_rejects_no_lse_return(monkeypatch):
    impl = SimpleNamespace(
        can_return_lse_for_decode=False,
        need_to_return_lse_for_decode=False,
        supports_pcp=False,
    )
    layer = SimpleNamespace(impl=impl)

    monkeypatch.setattr(
        cp_utils,
        "get_layers_from_vllm_config",
        lambda vllm_config, layer_type: {"layer": layer},
    )

    with pytest.raises(AssertionError, match="requires attention implementations"):
        cp_utils.check_attention_cp_compatibility(_make_config(dcp_size=2))
