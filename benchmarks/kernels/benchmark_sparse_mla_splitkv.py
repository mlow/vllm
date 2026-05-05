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
import json
import math
import time
from collections.abc import Callable, Iterable

import torch

from vllm.triton_utils import LOG2E, LOGE2, tl, triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    matmul_sparse_mla_attention_with_sink,
)

_HEAD_BLOCK = 16
_MERGE_HEAD_BLOCK = 1
_DIM = 512
_BLOCK_N = 32
_MERGE_BLOCK_D = 128
_NUM_MERGE_D_TILES = _DIM // _MERGE_BLOCK_D
_MIN_CANDIDATES_PER_SPLIT = 128
_SPLIT_MAX_OCCUPANCY = 4


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


def _next_power_of_2(value: int) -> int:
    return 1 << max(0, value - 1).bit_length()


def choose_num_kv_splits(
    num_tokens: int,
    num_heads: int,
    num_candidates: int,
    sm_count: int,
    head_block_size: int = _HEAD_BLOCK,
) -> int:
    """Pick the split count used by the prototype auto mode.

    Mirrors the split heuristic from PR #38476 but takes `num_heads` directly
    so the benchmark can compare against the current SM12x materialized path.
    """
    num_head_groups = math.ceil(num_heads / min(head_block_size, num_heads))
    baseline = num_tokens * num_head_groups
    if baseline == 0 or baseline * _SPLIT_MAX_OCCUPANCY >= sm_count:
        return 1

    ideal = _next_power_of_2(max(1, num_candidates // _MIN_CANDIDATES_PER_SPLIT))
    max_splits = max(1, sm_count // baseline)
    max_splits = 1 << (max_splits.bit_length() - 1)
    num_splits = min(ideal, max_splits)
    while num_splits > 1 and num_candidates % num_splits != 0:
        num_splits //= 2
    return max(1, num_splits)


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


@triton.jit
def _splitkv_stage1_kernel(
    q_ptr,
    kv_ptr,
    valid_ptr,
    mid_ptr,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kvt: tl.constexpr,
    stride_kvc: tl.constexpr,
    stride_kvd: tl.constexpr,
    stride_vt: tl.constexpr,
    stride_vc: tl.constexpr,
    stride_mt: tl.constexpr,
    stride_mh: tl.constexpr,
    stride_ms: tl.constexpr,
    num_heads: tl.constexpr,
    num_candidates: tl.constexpr,
    scale: tl.constexpr,
    num_splits: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOGE2_VALUE: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_group = tl.program_id(1)
    split_id = tl.program_id(2)

    offs_h = head_group * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    mask_h = offs_h < num_heads
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr
        + token_id * stride_qt
        + offs_h[:, None] * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=mask_h[:, None],
        other=0.0,
    )

    split_size: tl.constexpr = tl.cdiv(num_candidates, num_splits)
    split_start = split_id * split_size
    split_end = tl.minimum(split_start + split_size, num_candidates)

    neg_large = -1.0e30
    e_max = tl.full((HEAD_BLOCK,), neg_large, dtype=tl.float32)
    e_sum = tl.zeros((HEAD_BLOCK,), dtype=tl.float32)
    acc = tl.zeros((HEAD_BLOCK, BLOCK_D), dtype=tl.float32)

    for cand_start in range(split_start, split_end, BLOCK_N):
        offs_c = cand_start + tl.arange(0, BLOCK_N)
        mask_c = offs_c < split_end
        valid = tl.load(
            valid_ptr + token_id * stride_vt + offs_c * stride_vc,
            mask=mask_c,
            other=0,
        )
        mask_kv = mask_c & valid
        k = tl.load(
            kv_ptr
            + token_id * stride_kvt
            + offs_c[:, None] * stride_kvc
            + offs_d[None, :] * stride_kvd,
            mask=mask_kv[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k.to(q.dtype))) * scale
        qk = tl.where(mask_h[:, None] & mask_kv[None, :], qk, neg_large)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(k.dtype), k)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    mid_base = (
        mid_ptr
        + token_id * stride_mt
        + offs_h[:, None] * stride_mh
        + split_id * stride_ms
    )
    tl.store(
        mid_base + offs_d[None, :],
        acc / e_sum_safe[:, None],
        mask=mask_h[:, None],
    )
    tl.store(
        mid_ptr
        + token_id * stride_mt
        + offs_h * stride_mh
        + split_id * stride_ms
        + BLOCK_D,
        (e_max + tl.log2(e_sum)) * LOGE2_VALUE,
        mask=mask_h,
    )


@triton.jit
def _splitkv_merge_kernel(
    mid_ptr,
    sink_ptr,
    output_ptr,
    stride_mt: tl.constexpr,
    stride_mh: tl.constexpr,
    stride_ms: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    num_heads: tl.constexpr,
    num_splits: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_TILE: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_group = tl.program_id(1)
    d_tile = tl.program_id(2)

    offs_h = head_group * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    mask_h = offs_h < num_heads
    offs_d = d_tile * BLOCK_D_TILE + tl.arange(0, BLOCK_D_TILE)
    mask_d = offs_d < BLOCK_D

    e_max = tl.full((HEAD_BLOCK,), -float("inf"), dtype=tl.float32)
    e_sum = tl.zeros((HEAD_BLOCK,), dtype=tl.float32)
    acc = tl.zeros((HEAD_BLOCK, BLOCK_D_TILE), dtype=tl.float32)
    mid_base = mid_ptr + token_id * stride_mt + offs_h[:, None] * stride_mh
    mid_lse = mid_ptr + token_id * stride_mt + offs_h * stride_mh + BLOCK_D

    for split_id in range(num_splits):
        part = tl.load(
            mid_base + split_id * stride_ms + offs_d[None, :],
            mask=mask_h[:, None] & mask_d[None, :],
            other=0.0,
        )
        lse = tl.load(
            mid_lse + split_id * stride_ms,
            mask=mask_h,
            other=-float("inf"),
        )
        n_e_max = tl.maximum(lse, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        part_scale = tl.exp(lse - n_e_max)
        acc = acc * old_scale[:, None] + part * part_scale[:, None]
        e_sum = e_sum * old_scale + part_scale
        e_max = n_e_max

    sink = tl.load(sink_ptr + offs_h, mask=mask_h, other=-float("inf"))
    n_e_max = tl.maximum(sink, e_max)
    value_scale = tl.exp(e_max - n_e_max)
    sink_scale = tl.exp(sink - n_e_max)
    denom = e_sum * value_scale + sink_scale
    denom = tl.where(denom > 0, denom, 1.0)
    merged = acc * value_scale[:, None] / denom[:, None]

    tl.store(
        output_ptr
        + token_id * stride_out_t
        + offs_h[:, None] * stride_oh
        + offs_d[None, :] * stride_od,
        merged.to(tl.bfloat16),
        mask=mask_h[:, None] & mask_d[None, :],
    )


def splitkv_sparse_mla_attention_with_sink(
    q: torch.Tensor,
    kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    mid: torch.Tensor,
    num_splits: int,
) -> None:
    num_tokens, num_heads, head_dim = q.shape
    assert head_dim == _DIM
    assert kv.shape == (num_tokens, valid_tokens.shape[1], _DIM)
    assert output.shape == (num_tokens, num_heads, _DIM)
    assert mid.shape == (num_tokens, num_heads, num_splits, _DIM + 1)
    num_candidates = kv.shape[1]
    num_head_groups = triton.cdiv(num_heads, _HEAD_BLOCK)
    _splitkv_stage1_kernel[(num_tokens, num_head_groups, num_splits)](
        q,
        kv,
        valid_tokens,
        mid,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        valid_tokens.stride(0),
        valid_tokens.stride(1),
        mid.stride(0),
        mid.stride(1),
        mid.stride(2),
        num_heads,
        num_candidates,
        scale * LOG2E,
        num_splits,
        HEAD_BLOCK=_HEAD_BLOCK,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=_DIM,
        LOGE2_VALUE=LOGE2,
        num_warps=4,
    )
    _splitkv_merge_kernel[(num_tokens, num_heads, _NUM_MERGE_D_TILES)](
        mid,
        attn_sink,
        output,
        mid.stride(0),
        mid.stride(1),
        mid.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        num_heads,
        num_splits,
        HEAD_BLOCK=_MERGE_HEAD_BLOCK,
        BLOCK_D=_DIM,
        BLOCK_D_TILE=_MERGE_BLOCK_D,
        num_warps=2,
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
        matmul_sparse_mla_attention_with_sink(
            q,
            kv,
            valid_tokens,
            scale,
            attn_sink,
            current_out,
            num_heads=case.num_heads,
            score_buffer=score_buffer,
            value_block_size=512,
            candidate_block_size=128,
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
