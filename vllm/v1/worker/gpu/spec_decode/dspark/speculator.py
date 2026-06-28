# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 DSpark speculator.

DSpark is a block speculative decoder stored under the DeepSeek V4 checkpoint's
``mtp.*`` namespace, but it is not the serial DeepSeek MTP architecture. The
draft consumes target hidden states from configured target layers and predicts a
noise-token block through DSpark attention, Markov logits and a confidence head.

This file intentionally refuses to fall back to serial MTP. A wrong fallback
would load cleanly but silently measure the wrong algorithm.
"""

import contextlib
import json
import os
import time

import torch
import torch.nn as nn

from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.utils.flashinfer import autotune as flashinfer_autotune
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.spec_decode.dspark.utils import (
    load_deepseek_v4_dspark_model,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

logger = init_logger(__name__)


def _maybe_dump_propose_debug(
    *,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    output_ids: torch.Tensor,
    logits: torch.Tensor,
    confidence: torch.Tensor | None,
) -> None:
    debug_dir = os.getenv("VLLM_DSPARK_DEBUG_DIR")
    if not debug_dir:
        return
    max_dumps = int(os.getenv("VLLM_DSPARK_DEBUG_MAX_DUMPS", "8"))
    path = os.path.join(debug_dir, "propose_debug.jsonl")
    existing = 0
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = sum(1 for _ in f)
    if existing >= max_dumps:
        return

    top_values, top_indices = torch.topk(logits, k=8, dim=-1)
    rows = []
    for req_idx in range(min(int(output_ids.shape[0]), 2)):
        req_rows = []
        for step in range(int(logits.shape[1])):
            req_rows.append({
                "step": step,
                "top_ids": [
                    int(x) for x in top_indices[req_idx, step].detach().cpu()
                ],
                "top_logits": [
                    float(x) for x in top_values[req_idx, step].detach().cpu()
                ],
            })
        rows.append({
            "req": req_idx,
            "input_id": int(input_ids[req_idx].detach().cpu()),
            "position": int(positions[req_idx].detach().cpu()),
            "output_ids": [
                int(x) for x in output_ids[req_idx].detach().cpu().tolist()
            ],
            "logits_top": req_rows,
            "confidence_logits": (
                [
                    float(x)
                    for x in confidence[req_idx].detach().float().cpu().tolist()
                ]
                if confidence is not None
                else None
            ),
            "confidence_probs": (
                [
                    float(x)
                    for x in confidence[req_idx].detach().float().sigmoid().cpu().tolist()
                ]
                if confidence is not None
                else None
            ),
        })
    os.makedirs(debug_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.time(), "rows": rows}) + "\n")


def _maybe_dump_context_debug(
    *,
    input_batch: InputBatch,
    last_token_indices: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected_cpu: list[int],
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    context_ranges: list[tuple[int, int]],
) -> None:
    debug_dir = os.getenv("VLLM_DSPARK_DEBUG_DIR")
    if not debug_dir:
        return
    max_dumps = int(os.getenv("VLLM_DSPARK_DEBUG_MAX_DUMPS", "8"))
    path = os.path.join(debug_dir, "context_debug.jsonl")
    existing = 0
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = sum(1 for _ in f)
    if existing >= max_dumps:
        return

    rows = []
    num_reqs = min(int(input_ids.shape[0]), 2)
    host_input_ids = input_batch.input_ids[:input_batch.num_tokens].detach().cpu()
    host_positions = input_batch.positions[:input_batch.num_tokens].detach().cpu()
    for req_idx in range(num_reqs):
        start, end = context_ranges[req_idx]
        req_state_idx = int(input_batch.idx_mapping[req_idx].detach().cpu())
        rows.append({
            "req": req_idx,
            "req_state_idx": req_state_idx,
            "last_token_index": int(last_token_indices[req_idx].detach().cpu()),
            "current_input_id": int(input_ids[req_idx].detach().cpu()),
            "current_position": int(positions[req_idx].detach().cpu()),
            "num_sampled": int(num_sampled[req_idx].detach().cpu()),
            "num_rejected": int(num_rejected_cpu[req_idx]),
            "last_sampled": int(last_sampled[req_state_idx].detach().cpu()),
            "next_prefill_token": int(
                next_prefill_tokens[req_state_idx].detach().cpu()
            ),
            "query_start": int(input_batch.query_start_loc_np[req_idx]),
            "query_end": int(input_batch.query_start_loc_np[req_idx + 1]),
            "context_start": int(start),
            "context_end": int(end),
            "context_input_ids": [
                int(x) for x in host_input_ids[start:end].tolist()
            ],
            "context_positions": [
                int(x) for x in host_positions[start:end].tolist()
            ],
        })

    os.makedirs(debug_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.time(), "rows": rows}) + "\n")


def _maybe_dump_context_tensor_debug(
    *,
    input_batch: InputBatch,
    main_hidden_all: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    context_ranges: list[tuple[int, int]],
) -> None:
    debug_dir = os.getenv("VLLM_DSPARK_CONTEXT_TENSOR_DEBUG_DIR")
    if not debug_dir:
        return
    max_dumps = int(os.getenv("VLLM_DSPARK_CONTEXT_TENSOR_DEBUG_MAX_DUMPS", "4"))
    os.makedirs(debug_dir, exist_ok=True)
    existing = len([
        name for name in os.listdir(debug_dir) if name.endswith(".pt")
    ])
    if existing >= max_dumps:
        return
    path = os.path.join(debug_dir, f"context_{existing:04d}.pt")
    torch.save(
        {
            "time": time.time(),
            "main_hidden_all": main_hidden_all.detach().cpu(),
            "input_batch_input_ids": input_batch.input_ids[
                : input_batch.num_tokens
            ].detach().cpu(),
            "input_batch_positions": input_batch.positions[
                : input_batch.num_tokens
            ].detach().cpu(),
            "input_ids": input_ids.detach().cpu(),
            "positions": positions.detach().cpu(),
            "context_ranges": list(context_ranges),
        },
        path,
    )


class DSparkSpeculator(DraftModelSpeculator):
    def __init__(self, vllm_config, device: torch.device):
        super().__init__(vllm_config, device)
        self.supports_mm_inputs = False
        parallel_config = vllm_config.parallel_config
        cache_config = vllm_config.cache_config
        if parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError("DSpark currently requires pipeline parallel size 1.")
        if parallel_config.prefill_context_parallel_size > 1:
            raise NotImplementedError("DSpark currently requires prefill context parallel size 1.")
        if parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError("DSpark currently requires decode context parallel size 1.")
        if cache_config.enable_prefix_caching:
            raise NotImplementedError(
                "DSpark currently cannot run with prefix caching because its "
                "private rolling draft KV cache is not prefix-cache aware yet."
            )
        hf_config = self.draft_model_config.hf_config
        self.block_size = int(getattr(hf_config, "dspark_block_size", 0) or 0)
        self.noise_token_id = int(getattr(hf_config, "dspark_noise_token_id", -1))
        self.target_layer_ids = tuple(
            int(i) for i in getattr(hf_config, "dspark_target_layer_ids", ())
        )
        if self.block_size <= 0:
            raise ValueError("DSpark requires dspark_block_size in the model config.")
        if self.noise_token_id < 0:
            raise ValueError("DSpark requires dspark_noise_token_id in the model config.")
        if not self.target_layer_ids:
            raise ValueError(
                "DSpark requires dspark_target_layer_ids in the model config."
            )
        if self.num_speculative_steps > self.block_size:
            raise ValueError(
                "DSpark num_speculative_tokens must be <= dspark_block_size "
                f"({self.num_speculative_steps} > {self.block_size})."
            )
        self.last_token_indices = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int64,
            device=device,
        )
        self.current_draft_step = torch.tensor(0, dtype=torch.int64, device=device)
        self.active_num_reqs = torch.tensor(0, dtype=torch.int32, device=device)
        self.draft_step_cols = torch.arange(
            self.block_size,
            dtype=torch.int64,
            device=device,
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        del target_attn_layer_names
        return load_deepseek_v4_dspark_model(target_model, self.vllm_config)

    def capture(self, attn_states: dict | None = None) -> None:
        del attn_states
        logger.warning_once(
            "DSpark speculator currently runs eager; CUDA graph capture is skipped."
        )
        if os.getenv("VLLM_DSPARK_SKIP_WARMUP") == "1":
            return

        hf_config = self.draft_model_config.hf_config
        main_hidden_size = hf_config.hidden_size * len(self.target_layer_ids)
        input_ids = torch.full(
            (1,),
            self.noise_token_id,
            dtype=torch.long,
            device=self.device,
        )
        context_hidden = torch.zeros(
            (1, main_hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        context_positions = torch.zeros(
            (1,),
            dtype=torch.long,
            device=self.device,
        )
        decode_positions = torch.ones(
            (1,),
            dtype=torch.long,
            device=self.device,
        )

        self.idx_mapping[:1].zero_()
        self.temperature[:1].fill_(1.0)
        self.seeds[:1].zero_()
        self.active_num_reqs.fill_(1)

        batch_descriptor = BatchDescriptor(num_tokens=self.block_size)
        tune_ctx = contextlib.nullcontext()
        if self.vllm_config.kernel_config.enable_flashinfer_autotune:
            # DSpark draft MoE runs with M=block_size (5 for the published
            # checkpoint). FlashInfer's default mapper rounds this to bucket 8,
            # so tune that bucket to make the later non-tuning forward hit cache.
            tune_ctx = flashinfer_autotune(
                True,
                tuning_buckets=(8,),
                round_up=True,
            )
        try:
            with tune_ctx, set_forward_context(
                None,
                self.vllm_config,
                num_tokens=self.block_size,
                batch_descriptor=batch_descriptor,
                slot_mapping=None,
            ):
                self.model.precompute_context_kv(
                    context_hidden,
                    context_positions,
                    [(0, 1)],
                )
                self.model.forward_spec(
                    input_ids,
                    context_hidden,
                    decode_positions,
                    idx_mapping=self.idx_mapping[:1],
                    temperature=self.temperature,
                    seeds=self.seeds,
                    draft_logits=self.draft_logits,
                    draft_step_cols=self.draft_step_cols,
                    active_num_reqs=self.active_num_reqs,
                    use_fp64_gumbel=self.use_fp64_gumbel,
                )
            torch.cuda.synchronize(self.device)
        finally:
            self.model.reset_dspark_kv_cache()
            self.draft_tokens.zero_()
            if self.draft_logits is not None:
                self.draft_logits.zero_()
        logger.info_once("DSpark eager warmup completed.")

    def init_cudagraph_manager(self, cudagraph_mode) -> None:
        return None

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict,
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        del (
            attn_metadata,
            last_hidden_states,
            mm_inputs,
            is_profile,
        )
        num_reqs = input_batch.num_reqs
        if any(
            req_id.startswith(
                ("_warmup_", "_v2_mixed_warmup", "_sparse_mla_v2_warmup")
            )
            for req_id in input_batch.req_ids[:num_reqs]
        ):
            self.draft_tokens[:num_reqs].zero_()
            return self.draft_tokens[:num_reqs]
        if dummy_run and skip_attn_for_dummy_run:
            self.draft_tokens[:num_reqs].zero_()
            return self.draft_tokens[:num_reqs]
        if not aux_hidden_states:
            raise RuntimeError("DSpark requires target auxiliary hidden states.")
        if len(aux_hidden_states) != len(self.target_layer_ids):
            raise RuntimeError(
                "DSpark auxiliary hidden-state count mismatch: "
                f"expected {len(self.target_layer_ids)}, got {len(aux_hidden_states)}."
            )

        with record_function_or_nullcontext("vllm:v2/speculator/dspark/prepare"):
            prepare_prefill_inputs(
                self.last_token_indices,
                self.current_draft_step,
                self.input_buffers,
                input_batch,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                self.max_num_reqs,
            )
            self._copy_request_inputs(
                num_reqs,
                input_batch.idx_mapping,
                temperature,
                seeds,
            )
            self.active_num_reqs.fill_(num_reqs)
            last_token_indices = self.last_token_indices[:num_reqs]
            # prepare_prefill_inputs writes the fresh target token at each
            # request's last_token_index. DSpark must use that same token and
            # position together with the corresponding aux hidden state.
            input_ids = self.input_buffers.input_ids[last_token_indices]
            positions = self.input_buffers.positions[last_token_indices]
            if dummy_run:
                self.draft_tokens[:num_reqs].zero_()
                return self.draft_tokens[:num_reqs]
            main_hidden_all = torch.cat(aux_hidden_states, dim=-1)
            num_rejected_cpu = num_rejected[:num_reqs].detach().cpu().tolist()
            skip_after_rejection = (
                os.getenv("VLLM_DSPARK_SKIP_AFTER_REJECTION") == "1"
            )
            context_ranges = [
                (
                    int(input_batch.query_start_loc_np[req_idx]),
                    int(input_batch.query_start_loc_np[req_idx + 1])
                    - int(num_rejected_cpu[req_idx]),
                )
                for req_idx in range(num_reqs)
            ]
            if skip_after_rejection:
                for req_idx, num_rejected_i in enumerate(num_rejected_cpu):
                    if num_rejected_i > 0:
                        stale_row = int(last_token_indices[req_idx].detach().cpu())
                        start, end = context_ranges[req_idx]
                        context_ranges[req_idx] = (start, min(end, stale_row))
            _maybe_dump_context_debug(
                input_batch=input_batch,
                last_token_indices=last_token_indices,
                input_ids=input_ids,
                positions=positions,
                num_sampled=num_sampled,
                num_rejected_cpu=num_rejected_cpu,
                last_sampled=last_sampled,
                next_prefill_tokens=next_prefill_tokens,
                context_ranges=context_ranges,
            )
            _maybe_dump_context_tensor_debug(
                input_batch=input_batch,
                main_hidden_all=main_hidden_all[: input_batch.num_tokens],
                input_ids=input_ids,
                positions=positions,
                context_ranges=context_ranges,
            )
            self.model.precompute_context_kv(
                main_hidden_all[: input_batch.num_tokens],
                input_batch.positions[: input_batch.num_tokens],
                context_ranges,
            )
            # The initial target prefill only initializes DSpark's rolling
            # context KV.  The public DSpark algorithm starts proposing once a
            # real decode token exists, so position 0 must not run the draft
            # block model.
            if torch.any(positions <= 0).item():
                self.draft_tokens[:num_reqs].zero_()
                return self.draft_tokens[:num_reqs]
            if skip_after_rejection and any(n > 0 for n in num_rejected_cpu):
                return self.draft_tokens[:num_reqs, :0]
            main_hidden = main_hidden_all[last_token_indices]

        with record_function_or_nullcontext("vllm:v2/speculator/dspark/forward"):
            # DSpark uses its own TileLang attention path, but the inherited
            # DeepSeek MoE layers still read vLLM's forward context to resolve
            # static layer state.
            batch_descriptor = BatchDescriptor(num_tokens=num_reqs * self.block_size)
            with set_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_reqs * self.block_size,
                num_tokens_across_dp=num_tokens_across_dp,
                slot_mapping=slot_mappings,
                batch_descriptor=batch_descriptor,
            ):
                output_ids, logits, _confidence = self.model.forward_spec(
                    input_ids,
                    main_hidden,
                    positions,
                    idx_mapping=self.idx_mapping[:num_reqs],
                    temperature=self.temperature,
                    seeds=self.seeds,
                    draft_logits=self.draft_logits,
                    draft_step_cols=self.draft_step_cols,
                    active_num_reqs=self.active_num_reqs,
                    use_fp64_gumbel=self.use_fp64_gumbel,
                )
                _maybe_dump_propose_debug(
                    input_ids=input_ids,
                    positions=positions,
                    output_ids=output_ids,
                    logits=logits,
                    confidence=_confidence,
                )

        steps = self.num_speculative_steps
        self.draft_tokens[:num_reqs, :steps].copy_(output_ids[:, 1 : 1 + steps])
        return self.draft_tokens[:num_reqs]
