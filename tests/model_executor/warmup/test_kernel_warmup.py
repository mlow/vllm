# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.warmup import kernel_warmup


class _SparseMlaBackend:
    @staticmethod
    def get_name() -> str:
        return "V4_FLASHMLA_SPARSE"


class _FakeModelConfig:
    @staticmethod
    def get_vocab_size() -> int:
        return 128


class _FakeRunner:
    def __init__(self) -> None:
        self.attn_groups = [[SimpleNamespace(backend=_SparseMlaBackend())]]
        self.cache_config = SimpleNamespace(block_size=256)
        self.device = "cuda"
        self.dummy_runs: list[dict] = []
        self.is_pooling_model = False
        self.max_model_len = 2048
        self.max_num_tokens = 4176
        self.model_config = _FakeModelConfig()
        self.num_spec_tokens = 2
        self.speculative_config = SimpleNamespace(method="mtp")
        self.uniform_decode_query_len = 3

    def _dummy_run(self, **kwargs) -> None:
        self.dummy_runs.append(kwargs)


@pytest.fixture
def deepseek_v4_mtp_worker() -> SimpleNamespace:
    runner = _FakeRunner()
    return SimpleNamespace(
        model_runner=runner,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=4176,
            max_num_seqs=2,
        ),
    )


def test_sparse_mla_warmup_covers_mtp_uniform_decode_and_spec_decode_kernels(
    monkeypatch: pytest.MonkeyPatch,
    deepseek_v4_mtp_worker: SimpleNamespace,
) -> None:
    spec_decode_warmups: list[dict] = []

    def fake_spec_decode_warmup(**kwargs) -> None:
        spec_decode_warmups.append(kwargs)

    monkeypatch.setattr(
        kernel_warmup.envs,
        "VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP",
        True,
    )
    monkeypatch.setattr(
        kernel_warmup.current_platform,
        "is_cuda_alike",
        lambda: True,
    )
    monkeypatch.setattr(
        kernel_warmup,
        "_run_deepseek_v4_mtp_spec_decode_warmup_kernels",
        fake_spec_decode_warmup,
        raising=False,
    )
    monkeypatch.setattr(
        kernel_warmup.torch.accelerator,
        "synchronize",
        lambda: None,
    )

    kernel_warmup._deepseek_v4_sparse_mla_attention_warmup(deepseek_v4_mtp_worker)

    dummy_runs = deepseek_v4_mtp_worker.model_runner.dummy_runs
    assert {run["num_tokens"] for run in dummy_runs if run.get("uniform_decode")} == {
        3,
        6,
    }
    assert any(run.get("create_mixed_batch") for run in dummy_runs)
    assert any(run.get("create_single_prefill") for run in dummy_runs)
    assert spec_decode_warmups == [
        {
            "device": "cuda",
            "num_reqs": 1,
            "num_spec_tokens": 2,
            "vocab_size": 128,
            "block_size": 256,
            "max_model_len": 2048,
        },
        {
            "device": "cuda",
            "num_reqs": 2,
            "num_spec_tokens": 2,
            "vocab_size": 128,
            "block_size": 256,
            "max_model_len": 2048,
        },
    ]
