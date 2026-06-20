# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.attention.backend import AttentionBackend, MultipleOf
from vllm.v1.attention.backends.b12x_attn import B12XPagedAttentionBackend
from vllm.v1.worker.utils import select_common_block_size


class _Page128OnlyBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "PAGE128_ONLY"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [128]


def test_b12x_dense_backend_advertises_page128() -> None:
    assert B12XPagedAttentionBackend.get_supported_kernel_block_sizes() == [64, 128]
    assert B12XPagedAttentionBackend.supports_block_size(64)
    assert B12XPagedAttentionBackend.supports_block_size(128)
    assert not B12XPagedAttentionBackend.supports_block_size(32)
    assert B12XPagedAttentionBackend.get_preferred_block_size(16) == 128
    assert B12XPagedAttentionBackend.get_preferred_block_size(64) == 64


def test_b12x_dense_kv_cache_shape_accepts_page128() -> None:
    assert B12XPagedAttentionBackend.get_kv_cache_shape(
        num_blocks=3,
        block_size=128,
        num_kv_heads=4,
        head_size=128,
        cache_dtype_str="bfloat16",
    ) == (3, 2, 128, 4, 128)


def test_b12x_dense_can_share_page128_group() -> None:
    assert (
        select_common_block_size(
            128,
            [B12XPagedAttentionBackend, _Page128OnlyBackend],
        )
        == 128
    )
