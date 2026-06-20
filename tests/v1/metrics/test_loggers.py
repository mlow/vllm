# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import cast

from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import LoggingStatLogger
from vllm.v1.metrics.stats import IterationStats, PrefillStats, SchedulerStats


def _make_logger() -> LoggingStatLogger:
    config = SimpleNamespace(
        model_config=SimpleNamespace(is_diffusion=False),
        kv_transfer_config=None,
        observability_config=SimpleNamespace(
            cudagraph_metrics=False,
            enable_mfu_metrics=False,
        ),
        cache_config=SimpleNamespace(num_gpu_blocks=0),
    )
    return LoggingStatLogger(cast(VllmConfig, config))


def test_prompt_throughput_tracks_scheduler_only_prefill_chunks():
    stat_logger = _make_logger()

    stat_logger.record(
        scheduler_stats=SchedulerStats(num_scheduled_prompt_tokens=64),
        iteration_stats=None,
    )

    assert stat_logger.num_prompt_tokens == 64
    assert stat_logger.num_generation_tokens == 0


def test_prompt_throughput_does_not_double_count_prefill_output_stats():
    stat_logger = _make_logger()
    iteration_stats = IterationStats()
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=128,
        num_local_cached_tokens=0,
        num_external_cached_tokens=0,
    )
    iteration_stats.prompt_token_stats.update_from_output(prefill_stats)
    iteration_stats.num_generation_tokens = 4

    stat_logger.record(
        scheduler_stats=SchedulerStats(num_scheduled_prompt_tokens=32),
        iteration_stats=iteration_stats,
    )

    assert stat_logger.num_prompt_tokens == 32
    assert stat_logger.num_generation_tokens == 4
