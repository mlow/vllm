# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform controls for the portable Triton sparse MLA path."""

import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE = 512
_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE = 256
_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH_ENV = "VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH"
_ENV_TRUE_VALUES = {"1", "true", "yes", "on"}
_ENV_FALSE_VALUES = {"0", "false", "no", "off"}

logger = init_logger(__name__)


def _optional_env_flag(name: str) -> bool | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    value = raw_value.lower()
    if value in _ENV_TRUE_VALUES:
        return True
    if value in _ENV_FALSE_VALUES:
        return False
    return None


def _is_sm12x_device(device: torch.device) -> bool:
    if not torch.cuda.is_available():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(index)[0] == 12


def is_triton_sparse_mla_enabled_for_platform() -> bool:
    return current_platform.is_device_capability_family(120)


def is_triton_sparse_mla_enabled(device: torch.device) -> bool:
    return _is_sm12x_device(device)


def _uses_speculative_decoding(vllm_config) -> bool:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is None:
        return False
    return bool(getattr(speculative_config, "num_speculative_tokens", 0))


def triton_sparse_mla_cudagraphs_allowed(vllm_config=None) -> bool:
    configured = _optional_env_flag(_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH_ENV)
    if configured is not None:
        return configured
    return not _uses_speculative_decoding(vllm_config)


def needs_constant_topk_for_flashmla_cudagraph(vllm_config) -> bool:
    compilation_config = vllm_config.compilation_config
    if not compilation_config.cudagraph_mode.has_full_cudagraphs():
        return False
    # Constant topk lengths are a FlashMLA tile-scheduler requirement. The
    # SM12x Triton sparse MLA path consumes lens directly and keeps dynamic
    # counts even when its experimental cudagraph opt-in is enabled.
    return not is_triton_sparse_mla_enabled_for_platform()


def disable_triton_sparse_mla_cudagraphs_if_enabled(vllm_config) -> None:
    if not is_triton_sparse_mla_enabled_for_platform():
        return

    configured = _optional_env_flag(_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH_ENV)
    if triton_sparse_mla_cudagraphs_allowed(vllm_config):
        logger.warning_once(
            "Keeping the requested vLLM compile and CUDA graph settings for "
            "the DeepSeek V4 Triton sparse MLA path. Set "
            f"{_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH_ENV}=0 to opt out if "
            "a graph-safety issue is found."
        )
        return

    from vllm.config.compilation import CUDAGraphMode

    compilation_config = vllm_config.compilation_config
    if compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        return

    reason = (
        "under speculative decoding. Set "
        f"{_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH_ENV}=1 to opt into the "
        "experimental graph-captured MTP path."
        if configured is None and _uses_speculative_decoding(vllm_config)
        else f"because {_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH_ENV}=0."
    )
    logger.warning_once(
        "Disabling CUDA graphs for the DeepSeek V4 Triton sparse MLA path "
        f"{reason} vLLM compile remains enabled."
    )
    compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    compilation_config.cudagraph_capture_sizes = []
    compilation_config.max_cudagraph_capture_size = 0


def triton_sparse_mla_topk_chunk_size() -> int:
    return _TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE


def triton_sparse_mla_query_chunk_size() -> int:
    return _TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE
