# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

import vllm.model_executor.layers.mla as mla_module
import vllm.model_executor.models.deepseek_v2 as deepseek_v2
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper


class _RecordedModule(torch.nn.Module):
    def __init__(self, name: str, calls: list[str], fn) -> None:
        super().__init__()
        self.name = name
        self.calls = calls
        self.fn = fn

    def forward(self, *args, **kwargs):
        self.calls.append(self.name)
        return self.fn(*args, **kwargs)


def _make_test_wrapper(calls: list[str]) -> MultiHeadLatentAttentionWrapper:
    wrapper = object.__new__(MultiHeadLatentAttentionWrapper)
    torch.nn.Module.__init__(wrapper)

    wrapper.q_lora_rank = 2
    wrapper.kv_lora_rank = 2
    wrapper.qk_nope_head_dim = 1
    wrapper.qk_rope_head_dim = 1
    wrapper.qk_head_dim = 2
    wrapper.v_head_dim = 2
    wrapper.num_heads = 1
    wrapper.is_sparse = True
    wrapper.skip_topk = False
    wrapper.indexer_rope_emb = object()

    wrapper.fused_qkv_a_proj = _RecordedModule(
        "fused_qkv_a",
        calls,
        lambda hidden: (torch.cat([hidden[:, :2], hidden[:, :3]], dim=-1), None),
    )
    wrapper.q_a_layernorm = _RecordedModule("q_norm", calls, lambda q: q)
    wrapper.q_b_proj = _RecordedModule("main_q", calls, lambda q: (q, None))
    wrapper.q_proj = None
    wrapper.kv_a_proj_with_mqa = None
    wrapper.kv_a_layernorm = _RecordedModule("kv_norm", calls, lambda kv: kv)
    wrapper.rotary_emb = _RecordedModule(
        "main_rope", calls, lambda positions, q_pe, k_pe: (q_pe, k_pe)
    )
    wrapper.indexer = _RecordedModule(
        "indexer", calls, lambda hidden, q_c, positions, rotary: None
    )
    wrapper.mla_attn = _RecordedModule(
        "mla",
        calls,
        lambda q, kv, k_pe, output_shape: q.reshape(output_shape),
    )
    wrapper.o_proj = _RecordedModule("o_proj", calls, lambda out: (out, None))

    wrapper.indexer_aux_stream = object()
    wrapper.indexer_start_event = object()
    wrapper.indexer_done_event = object()
    return wrapper


def test_glm_b12x_overlap_enqueues_main_before_indexer(monkeypatch) -> None:
    calls: list[str] = []
    wrapper = _make_test_wrapper(calls)

    def fake_execute_in_parallel(
        default_fn, aux_fn, start_event, done_event, aux_stream
    ):
        assert start_event is wrapper.indexer_start_event
        assert done_event is wrapper.indexer_done_event
        assert aux_stream is wrapper.indexer_aux_stream
        calls.append("fork")
        default_result = default_fn()
        aux_result = aux_fn()
        calls.append("join")
        return default_result, aux_result

    monkeypatch.setattr(
        mla_module, "maybe_execute_in_parallel", fake_execute_in_parallel
    )

    hidden_states = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    output = wrapper(torch.arange(2), hidden_states)

    assert output.shape == (2, 2)
    assert calls == [
        "fused_qkv_a",
        "q_norm",
        "fork",
        "main_q",
        "kv_norm",
        "main_rope",
        "indexer",
        "join",
        "mla",
        "o_proj",
    ]


def test_glm_b12x_overlap_gate_is_backend_and_model_specific(monkeypatch) -> None:
    monkeypatch.setattr(deepseek_v2, "use_b12x_sparse_indexer", lambda: True)

    assert deepseek_v2._should_overlap_glm_b12x_indexer(
        SimpleNamespace(model_type="glm_moe_dsa", index_topk=2048)
    )
    assert not deepseek_v2._should_overlap_glm_b12x_indexer(
        SimpleNamespace(model_type="deepseek_v32", index_topk=2048)
    )
    assert not deepseek_v2._should_overlap_glm_b12x_indexer(
        SimpleNamespace(model_type="glm_moe_dsa", index_topk=0)
    )

    monkeypatch.setattr(deepseek_v2, "use_b12x_sparse_indexer", lambda: False)
    assert not deepseek_v2._should_overlap_glm_b12x_indexer(
        SimpleNamespace(model_type="glm_moe_dsa", index_topk=2048)
    )
