# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

# this import will also register the custom ops
# import vllm.model_executor.kernels.mhc  # noqa: F401
import vllm.model_executor.kernels.mhc as mhc_kernels
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_tilelang

HAS_TILELANG = has_tilelang()
logger = init_logger(__name__)

_CUDA_TILELANG_OP = tuple[str, int, int]
_FAILED_CUDA_TILELANG_OPS: set[_CUDA_TILELANG_OP] = set()
_VERIFIED_CUDA_TILELANG_OPS: set[_CUDA_TILELANG_OP] = set()
_WARMED_CUDA_TILELANG_CONFIGS: set[tuple[int, int, torch.dtype, str]] = set()


def _cuda_tilelang_op_key(
    op_name: str,
    hidden_size: int,
    hc_mult: int,
) -> _CUDA_TILELANG_OP:
    return op_name, hidden_size, hc_mult


def _summarize_tilelang_exception(exc: Exception) -> str:
    lines = [line.strip() for line in str(exc).splitlines()]
    if "Compilation error:" in lines:
        lines = lines[lines.index("Compilation error:") + 1 :]
    summary = next((line for line in lines if line), repr(exc))
    if len(summary) > 240:
        summary = f"{summary[:237]}..."
    return f"{type(exc).__name__}: {summary}"


def _should_try_cuda_tilelang(op_key: _CUDA_TILELANG_OP) -> bool:
    if not HAS_TILELANG or op_key in _FAILED_CUDA_TILELANG_OPS:
        return False
    return (
        not torch.compiler.is_compiling()
        or op_key in _VERIFIED_CUDA_TILELANG_OPS
    )


def _mark_cuda_tilelang_verified(op_key: _CUDA_TILELANG_OP) -> None:
    _VERIFIED_CUDA_TILELANG_OPS.add(op_key)


def _disable_cuda_tilelang(op_key: _CUDA_TILELANG_OP, exc: Exception) -> None:
    _FAILED_CUDA_TILELANG_OPS.add(op_key)
    op_name, hidden_size, hc_mult = op_key
    logger.warning_once(
        "CUDA TileLang op %s failed for hidden_size=%s hc_mult=%s; "
        "falling back for this process. Failure: %s",
        op_name,
        hidden_size,
        hc_mult,
        _summarize_tilelang_exception(exc),
    )
    logger.debug(
        "CUDA TileLang op %s failure details",
        op_name,
        exc_info=exc,
    )


def _probe_cuda_tilelang_op(op_key: _CUDA_TILELANG_OP, call) -> bool:
    if not HAS_TILELANG or op_key in _FAILED_CUDA_TILELANG_OPS:
        return False
    try:
        call()
        torch.cuda.synchronize()
    except Exception as exc:
        _disable_cuda_tilelang(op_key, exc)
        return False
    else:
        return True


def warm_up_cuda_tilelang_mhc(
    hidden_size: int,
    hc_mult: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> None:
    """Probe CUDA TileLang MHC kernels outside torch.compile.

    The CUDA TileLang kernels are faster on some SM120 systems but currently
    fail to compile on others. DeepSeek V4 calls this once at model
    construction time so CUDA graph capture and any compiled callers see a
    stable verified-or-fallback decision.
    """
    device = torch.device(device)
    if (
        not HAS_TILELANG
        or not current_platform.is_cuda()
        or device.type != "cuda"
        or torch.compiler.is_compiling()
    ):
        return

    warm_key = (hidden_size, hc_mult, dtype, str(device))
    if warm_key in _WARMED_CUDA_TILELANG_CONFIGS:
        return
    _WARMED_CUDA_TILELANG_CONFIGS.add(warm_key)

    num_tokens = 1
    hc_mult3 = 2 * hc_mult + hc_mult * hc_mult
    with torch.inference_mode():
        residual = torch.zeros(
            (num_tokens, hc_mult, hidden_size), dtype=dtype, device=device
        )
        x = torch.zeros((num_tokens, hidden_size), dtype=dtype, device=device)
        post_mix = torch.zeros(
            (num_tokens, hc_mult, 1), dtype=torch.float32, device=device
        )
        comb_mix = torch.zeros(
            (num_tokens, hc_mult, hc_mult), dtype=torch.float32, device=device
        )
        fn = torch.zeros(
            (hc_mult3, hc_mult * hidden_size), dtype=torch.float32, device=device
        )
        hc_scale = torch.zeros((3,), dtype=torch.float32, device=device)
        hc_base = torch.zeros((hc_mult3,), dtype=torch.float32, device=device)
        hc_head_fn = torch.zeros(
            (hc_mult, hc_mult * hidden_size), dtype=torch.float32, device=device
        )
        hc_head_scale = torch.zeros((1,), dtype=torch.float32, device=device)
        hc_head_base = torch.zeros((hc_mult,), dtype=torch.float32, device=device)
        hc_head_out = torch.empty((num_tokens, hidden_size), dtype=dtype, device=device)

        probes = [
            (
                _cuda_tilelang_op_key("mhc_pre", hidden_size, hc_mult),
                lambda: torch.ops.vllm.mhc_pre_tilelang(
                    residual,
                    fn,
                    hc_scale,
                    hc_base,
                    1e-6,
                    1e-6,
                    1e-6,
                    1.0,
                    20,
                    1,
                    None,
                    0.0,
                ),
            ),
            (
                _cuda_tilelang_op_key("mhc_post", hidden_size, hc_mult),
                lambda: torch.ops.vllm.mhc_post_tilelang(
                    x, residual, post_mix, comb_mix
                ),
            ),
            (
                _cuda_tilelang_op_key("mhc_fused_post_pre", hidden_size, hc_mult),
                lambda: torch.ops.vllm.mhc_fused_post_pre_tilelang(
                    x,
                    residual,
                    post_mix,
                    comb_mix,
                    fn,
                    hc_scale,
                    hc_base,
                    1e-6,
                    1e-6,
                    1e-6,
                    1.0,
                    20,
                    1,
                    1,
                    None,
                    0.0,
                ),
            ),
            (
                _cuda_tilelang_op_key("hc_head", hidden_size, hc_mult),
                lambda: torch.ops.vllm.hc_head_fused_kernel_tilelang(
                    residual,
                    hc_head_fn,
                    hc_head_scale,
                    hc_head_base,
                    hc_head_out,
                    hidden_size,
                    1e-6,
                    1e-6,
                    hc_mult,
                ),
            ),
        ]
        probe_results = [
            (op_key, _probe_cuda_tilelang_op(op_key, call))
            for op_key, call in probes
        ]

        if all(succeeded for _, succeeded in probe_results):
            for op_key, _ in probe_results:
                _mark_cuda_tilelang_verified(op_key)
            return

        for op_key, _ in probe_results:
            _FAILED_CUDA_TILELANG_OPS.add(op_key)
            _VERIFIED_CUDA_TILELANG_OPS.discard(op_key)

        logger.warning_once(
            "CUDA TileLang MHC warmup did not verify all kernels for "
            "hidden_size=%s hc_mult=%s; using the CUDA fallback group for "
            "this process.",
            hidden_size,
            hc_mult,
        )


def _apply_optional_rms_norm(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> torch.Tensor:
    if norm_weight is None:
        return x
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    x_normed = x.float() * torch.rsqrt(variance + norm_eps)
    return (x_normed * norm_weight.float()).to(x.dtype)


# --8<-- [start:mhc_pre]
@CustomOp.register("mhc_pre")
class MHCPreOp(CustomOp):
    """MHC pre block.

    Computes mix logits from RMS-normalized HC residual streams, then
    returns post_mix, comb_mix, and
    layer_input = sum_i pre_mix_i * residual_i.
    """

    # --8<-- [end:mhc_pre]
    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        op_key = _cuda_tilelang_op_key(
            "mhc_pre", residual.shape[-1], residual.shape[-2]
        )

        def tilelang_call():
            return torch.ops.vllm.mhc_pre_tilelang(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                norm_weight,
                norm_eps,
            )

        if _should_try_cuda_tilelang(op_key):
            try:
                result = tilelang_call()
                _mark_cuda_tilelang_verified(op_key)
                return result
            except Exception as exc:
                _disable_cuda_tilelang(op_key, exc)

        post_mix, comb_mix, layer_input = self.forward_native(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
        )
        return post_mix, comb_mix, _apply_optional_rms_norm(
            layer_input, norm_weight, norm_eps
        )

    def forward_hip(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: Reenable aiter after we are at the aiter
        # version that has this bugfix
        # https://github.com/ROCm/aiter/commit/b639cb63bcac4672dce33a731fad042a65cb3649
        # It has accuracy problem at large number of tokens.
        # hidden_size = residual.shape[-1]
        # if hidden_size % 256 == 0:
        #     return torch.ops.vllm.mhc_pre_aiter(
        #         residual,
        #         fn,
        #         hc_scale,
        #         hc_base,
        #         rms_eps,
        #         hc_pre_eps,
        #         hc_sinkhorn_eps,
        #         hc_post_mult_value,
        #         sinkhorn_repeat,
        #     )
        # else:
        if HAS_TILELANG:
            return torch.ops.vllm.mhc_pre_tilelang(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                norm_weight,
                norm_eps,
            )
        else:
            return self.forward_native(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                norm_weight,
                norm_eps,
            )

    def forward_native(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return mhc_kernels.mhc_pre_torch(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )


# --8<-- [start:mhc_post]
@CustomOp.register("mhc_post")
class MHCPostOp(CustomOp):
    """MHC post block.

    Combines the layer output with the HC residual streams:
    out_j = post_layer_mix_j * x + sum_i comb_res_mix_ij * residual_i.
    """

    # --8<-- [end:mhc_post]

    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        op_key = _cuda_tilelang_op_key(
            "mhc_post", residual.shape[-1], residual.shape[-2]
        )

        def tilelang_call():
            return torch.ops.vllm.mhc_post_tilelang(
                x, residual, post_layer_mix, comb_res_mix
            )

        if _should_try_cuda_tilelang(op_key):
            try:
                result = tilelang_call()
                _mark_cuda_tilelang_verified(op_key)
                return result
            except Exception as exc:
                _disable_cuda_tilelang(op_key, exc)

        return self.forward_native(x, residual, post_layer_mix, comb_res_mix)

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: Reenable aiter after we are at the aiter
        # version that has this bugfix
        # https://github.com/ROCm/aiter/commit/b639cb63bcac4672dce33a731fad042a65cb3649
        # It has accuracy problem at large number of tokens.
        # hidden_size = residual.shape[-1]
        # if hidden_size % 256 == 0:
        #     return torch.ops.vllm.mhc_post_aiter(
        #         x,
        #         residual,
        #         post_layer_mix,
        #         comb_res_mix,
        #     )
        # else:
        if HAS_TILELANG:
            return torch.ops.vllm.mhc_post_tilelang(
                x, residual, post_layer_mix, comb_res_mix
            )
        else:
            return self.forward_native(x, residual, post_layer_mix, comb_res_mix)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        return mhc_kernels.mhc_post_torch(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
        )


# Keep the Triton HC-head body as the CUDA fallback. Some SM120 CUDA 13
# environments have observed TileLang compile failures for the DSv4
# hidden_size=7168 shape; other SM120 setups compile the fused TileLang kernel
# and should keep using it for latency. The dispatch path tries TileLang first
# and lands here only after a runtime failure.
@torch.compile(backend=current_platform.simple_compile_backend)
def _hc_head_cuda_impl(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    hc_mult, hidden_size = hidden_states.shape[-2:]
    outer_shape = hidden_states.shape[:-2]
    hs_flat = hidden_states.view(-1, hc_mult, hidden_size)
    num_tokens = hs_flat.shape[0]

    out = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=hidden_states.device
    )
    torch.ops.vllm.hc_head_triton(
        hs_flat,
        hc_fn,
        hc_scale,
        hc_base,
        out,
        hidden_size,
        rms_norm_eps,
        hc_eps,
        hc_mult,
    )
    return out.view(*outer_shape, hidden_size)


def _hc_head_cuda_tilelang_impl(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    hc_mult, hidden_size = hidden_states.shape[-2:]
    outer_shape = hidden_states.shape[:-2]
    hs_flat = hidden_states.view(-1, hc_mult, hidden_size)
    num_tokens = hs_flat.shape[0]

    out = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=hidden_states.device
    )
    torch.ops.vllm.hc_head_fused_kernel_tilelang(
        hs_flat,
        hc_fn,
        hc_scale,
        hc_base,
        out,
        hidden_size,
        rms_norm_eps,
        hc_eps,
        hc_mult,
    )
    return out.view(*outer_shape, hidden_size)


# --8<-- [start:hc_head]
@CustomOp.register("hc_head")
class HCHeadOp(CustomOp):
    """HC head reduction for DeepSeek V4.

    Computes gates from the RMS-normalized flattened HC residual and
    returns out = sum_i gate_i * residual_i, collapsing hc_mult streams
    to one.
    """

    # --8<-- [end:hc_head]
    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        op_key = _cuda_tilelang_op_key(
            "hc_head", hidden_states.shape[-1], hidden_states.shape[-2]
        )

        def tilelang_call():
            return _hc_head_cuda_tilelang_impl(
                hidden_states,
                hc_fn,
                hc_scale,
                hc_base,
                rms_norm_eps,
                hc_eps,
            )

        if _should_try_cuda_tilelang(op_key):
            try:
                result = tilelang_call()
                _mark_cuda_tilelang_verified(op_key)
                return result
            except Exception as exc:
                _disable_cuda_tilelang(op_key, exc)

        return _hc_head_cuda_impl(
            hidden_states,
            hc_fn,
            hc_scale,
            hc_base,
            rms_norm_eps,
            hc_eps,
        )

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        hc_mult, hidden_size = hidden_states.shape[-2:]
        outer_shape = hidden_states.shape[:-2]
        hs_flat = hidden_states.view(-1, hc_mult, hidden_size)
        num_tokens = hs_flat.shape[0]

        out = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=hidden_states.device
        )

        if HAS_TILELANG:
            torch.ops.vllm.hc_head_fused_kernel_tilelang(
                hs_flat,
                hc_fn,
                hc_scale,
                hc_base,
                out,
                hidden_size,
                rms_norm_eps,
                hc_eps,
                hc_mult,
            )
        else:
            torch.ops.vllm.hc_head_triton(
                hs_flat,
                hc_fn,
                hc_scale,
                hc_base,
                out,
                hidden_size,
                rms_norm_eps,
                hc_eps,
                hc_mult,
            )

        return out.view(*outer_shape, hidden_size)

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError("Native implementation of hc_head is not available")


# --8<-- [start:mhc_fused_post_pre]
@CustomOp.register("mhc_fused_post_pre")
class MHCFusedPostPreOp(CustomOp):
    """Fused MHC post block followed by the next MHC pre block.

    Equivalent to applying MHCPostOp and then MHCPreOp to the updated
    residual streams, returning residual_cur, post_mix_cur, comb_mix_cur,
    and layer_input_cur.
    """

    # --8<-- [end:mhc_fused_post_pre]
    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        op_key = _cuda_tilelang_op_key(
            "mhc_fused_post_pre", residual.shape[-1], residual.shape[-2]
        )

        def tilelang_call():
            return torch.ops.vllm.mhc_fused_post_pre_tilelang(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                tile_n,
                norm_weight,
                norm_eps,
            )

        if _should_try_cuda_tilelang(op_key):
            try:
                result = tilelang_call()
                _mark_cuda_tilelang_verified(op_key)
                return result
            except Exception as exc:
                _disable_cuda_tilelang(op_key, exc)

        residual_cur = mhc_kernels.mhc_post_torch(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
        )
        post_mix_cur, comb_mix_cur, layer_input_cur = mhc_kernels.mhc_pre_torch(
            residual_cur,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
        )
        return (
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            _apply_optional_rms_norm(layer_input_cur, norm_weight, norm_eps),
        )

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_weight,
            norm_eps,
        )

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError(
            "Native implementation of mhc_fused_post_pre is not available"
        )
