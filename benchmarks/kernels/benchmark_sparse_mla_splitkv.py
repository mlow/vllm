# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark a split-KV sparse MLA prototype against the SM12x matmul path.

This is an experimental benchmark-only prototype inspired by
https://github.com/vllm-project/vllm/pull/38476. It intentionally does not
register a runtime backend. The goal is to test whether splitting the sparse
candidate axis can beat the current materialized-BF16 matmul path for
low-batch, long-candidate DeepSeek V4 decode shapes on SM12x.
"""

import dataclasses
import inspect
import json
import math
import time
from collections.abc import Callable, Iterable

import torch

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    choose_sparse_mla_splitkv_splits,
    matmul_sparse_mla_attention_with_sink,
    splitkv_sparse_mla_attention_with_sink,
)

_DIM = 512
_MERGE_BLOCK_D = 128


@dataclasses.dataclass(frozen=True)
class BenchmarkCase:
    num_tokens: int
    num_candidates: int
    num_heads: int


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    num_tokens: int
    num_candidates: int
    num_heads: int
    num_splits: int
    current_ms: float
    splitkv_ms: float
    speedup: float
    max_abs_diff: float
    mean_abs_diff: float


def choose_num_kv_splits(
    num_tokens: int,
    num_heads: int,
    num_candidates: int,
    sm_count: int,
    head_block_size: int = 16,
) -> int:
    return choose_sparse_mla_splitkv_splits(
        num_tokens=num_tokens,
        num_heads=num_heads,
        num_candidates=num_candidates,
        sm_count=sm_count,
        head_block_size=head_block_size,
    )


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"Expected a comma-separated integer list, got {raw!r}")
    return values


def _parse_splits(raw: str) -> list[int | None]:
    splits: list[int | None] = []
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part == "auto":
            splits.append(None)
        else:
            value = int(part)
            if value <= 0:
                raise ValueError("Split counts must be positive")
            splits.append(value)
    if not splits:
        raise ValueError(f"Expected split counts or auto, got {raw!r}")
    return splits


def _filter_matmul_kwargs(
    fn: Callable[..., object],
    optional_kwargs: dict[str, object],
) -> dict[str, object]:
    parameters = inspect.signature(fn).parameters
    return {
        name: value for name, value in optional_kwargs.items() if name in parameters
    }


def _iter_cases(
    token_counts: Iterable[int],
    candidate_counts: Iterable[int],
    num_heads: int,
) -> Iterable[BenchmarkCase]:
    for num_tokens in token_counts:
        for num_candidates in candidate_counts:
            yield BenchmarkCase(
                num_tokens=num_tokens,
                num_candidates=num_candidates,
                num_heads=num_heads,
            )


def _benchmark_cuda(fn: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.accelerator.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


def _make_inputs(
    case: BenchmarkCase,
    seed: int,
    valid_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q = torch.randn(
        case.num_tokens,
        case.num_heads,
        _DIM,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    kv = torch.randn(
        case.num_tokens,
        case.num_candidates,
        _DIM,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    if valid_fraction >= 1.0:
        valid_tokens = torch.ones(
            case.num_tokens,
            case.num_candidates,
            device="cuda",
            dtype=torch.bool,
        )
    else:
        valid_tokens = (
            torch.rand(
                case.num_tokens,
                case.num_candidates,
                device="cuda",
                generator=generator,
            )
            < valid_fraction
        )
    attn_sink = torch.linspace(
        -0.25,
        0.25,
        case.num_heads,
        device="cuda",
        dtype=torch.float32,
    )
    return q, kv, valid_tokens, attn_sink


def _resolve_splits(
    split_spec: int | None,
    case: BenchmarkCase,
    sm_count: int,
) -> int:
    if split_spec is None:
        return choose_num_kv_splits(
            num_tokens=case.num_tokens,
            num_heads=case.num_heads,
            num_candidates=case.num_candidates,
            sm_count=sm_count,
        )
    return split_spec


def run_case(
    case: BenchmarkCase,
    split_spec: int | None,
    warmup: int,
    iters: int,
    seed: int,
    valid_fraction: float,
    rtol: float,
    atol: float,
) -> BenchmarkResult:
    sm_count = num_compute_units(torch.accelerator.current_device_index())
    num_splits = _resolve_splits(split_spec, case, sm_count)
    q, kv, valid_tokens, attn_sink = _make_inputs(case, seed, valid_fraction)
    scale = 1.0 / math.sqrt(_DIM)
    current_out = torch.empty_like(q)
    splitkv_out = torch.empty_like(q)
    score_buffer = torch.empty(
        case.num_tokens,
        case.num_heads,
        case.num_candidates,
        device="cuda",
        dtype=torch.bfloat16,
    )
    mid = torch.empty(
        case.num_tokens,
        case.num_heads,
        num_splits,
        _DIM + 1,
        device="cuda",
        dtype=torch.float32,
    )

    def run_current() -> None:
        kwargs = _filter_matmul_kwargs(
            matmul_sparse_mla_attention_with_sink,
            {
                "num_heads": case.num_heads,
                "score_buffer": score_buffer,
                "value_block_size": 512,
                "candidate_block_size": 128,
            },
        )
        matmul_sparse_mla_attention_with_sink(
            q,
            kv,
            valid_tokens,
            scale,
            attn_sink,
            current_out,
            **kwargs,
        )

    def run_splitkv() -> None:
        splitkv_sparse_mla_attention_with_sink(
            q,
            kv,
            valid_tokens,
            scale,
            attn_sink,
            splitkv_out,
            mid,
            num_splits,
        )

    run_current()
    run_splitkv()
    torch.accelerator.synchronize()
    finite_diff = (splitkv_out.float() - current_out.float()).abs()
    max_abs_diff = float(finite_diff.max().item())
    mean_abs_diff = float(finite_diff.mean().item())
    torch.testing.assert_close(
        splitkv_out.float(), current_out.float(), rtol=rtol, atol=atol
    )

    current_ms = _benchmark_cuda(run_current, warmup, iters)
    splitkv_ms = _benchmark_cuda(run_splitkv, warmup, iters)
    return BenchmarkResult(
        num_tokens=case.num_tokens,
        num_candidates=case.num_candidates,
        num_heads=case.num_heads,
        num_splits=num_splits,
        current_ms=current_ms,
        splitkv_ms=splitkv_ms,
        speedup=current_ms / splitkv_ms,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
    )


def _print_table(results: list[BenchmarkResult]) -> None:
    print(
        "tokens,candidates,heads,splits,current_ms,splitkv_ms,"
        "speedup,max_abs_diff,mean_abs_diff"
    )
    for result in results:
        print(
            f"{result.num_tokens},{result.num_candidates},{result.num_heads},"
            f"{result.num_splits},{result.current_ms:.4f},"
            f"{result.splitkv_ms:.4f},{result.speedup:.3f},"
            f"{result.max_abs_diff:.6f},{result.mean_abs_diff:.6f}"
        )


def main() -> None:
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="1,2,4,8,16")
    parser.add_argument("--candidates", default="512,1024,2048")
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--splits", default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--valid-fraction", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.accelerator.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if _DIM % _MERGE_BLOCK_D != 0:
        raise RuntimeError("Merge block size must divide the value dimension")

    token_counts = _parse_int_list(args.tokens)
    candidate_counts = _parse_int_list(args.candidates)
    splits = _parse_splits(args.splits)
    results: list[BenchmarkResult] = []
    for case in _iter_cases(token_counts, candidate_counts, args.num_heads):
        for split_spec in splits:
            results.append(
                run_case(
                    case,
                    split_spec,
                    args.warmup,
                    args.iters,
                    args.seed,
                    args.valid_fraction,
                    args.rtol,
                    args.atol,
                )
            )

    if args.json:
        print(json.dumps([dataclasses.asdict(result) for result in results], indent=2))
    else:
        _print_table(results)


if __name__ == "__main__":
    main()
