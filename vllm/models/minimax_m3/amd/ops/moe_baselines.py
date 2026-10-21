# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-quantized baseline copies of the MiniMax-M3.1 NVFP4 routed experts.

The exact NVFP4 weight (code * e4m3 scale / global) is dequantized to fp32
and re-quantized to one of two formats, both executed by aiter.fused_moe
(AITER MoE path only):

  mxfp8     fp8 e4m3 + E8M0 per 32 (round-up scale); handled by the stock
            ModelOpt MXFP8 method (AITER_MXFP8 backend), not here.
  mxfp4     e2m1 + E8M0 per 32 weights, dynamic MXFP4 activations
            (QuantType.per_1x32, a4w4). The E8M0 rule (floor / round-up) is
            chosen by VLLM_M3_NVFP4_MOE_MXFP4_SCALE (default from the op test).

Gate/up stay separated ([gate rows; up rows]) for mxfp4, like the tuned
E128k4 rows.
"""

import os

import torch

BASELINE_FMTS = ("mxfp8", "mxfp4")
# aiter MxScaleRoundMode ints: 0 RoundDown (floor, OCP), 1 RoundUp (ceil(amax/6)).
_MXFP4_SCALE_MODES = {"floor": 0, "roundup": 1}


def aiter_backend_line(fmt: str) -> str:
    """Serve-log marker of the baseline MoE backend (grep 'MoE backend')."""
    q = {"mxfp4": "MXFP4 a4w4 (per_1x32)"}
    return (
        f"Using 'AITER' {q[fmt]} MoE backend for the MiniMax-M3 NVFP4 baseline "
        "copy (aiter.fused_moe, tuned rows from model_configs/*tuned_fmoe*.csv); "
        "the AITER_MXFP8 oracle line above is not used for routed experts"
    )


def mxfp4_scale_mode() -> str:
    m = os.environ.get("VLLM_M3_NVFP4_MOE_MXFP4_SCALE", "roundup")
    if m not in _MXFP4_SCALE_MODES:
        raise ValueError(f"VLLM_M3_NVFP4_MOE_MXFP4_SCALE={m!r}: use floor|roundup")
    return m


def requant_mxfp4(v: torch.Tensor, mode: str):
    """fp32 [e, N, K] -> (packed e2m1 u8 [e, N, K/2], E8M0 u8 [e, N, K/32])."""
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_f4_quant

    e, N, K = v.shape
    q, s = per_1x32_f4_quant(
        v.reshape(e * N, K),
        quant_dtype=dtypes.fp4x2,
        shuffle=False,
        round_mode=_MXFP4_SCALE_MODES[mode],
    )
    return q.view(torch.uint8).view(e, N, K // 2), s.view(torch.uint8).view(
        e, N, K // 32
    )


def dequant_mxfp4(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Inverse of requant_mxfp4 (fp32), for error checks."""
    from aiter.utility import fp4_utils

    e, N, K2 = q.shape
    v = fp4_utils.mxfp4_to_f32(q.reshape(e * N, K2)).view(e, N, K2 * 2 // 32, 32)
    sc = fp4_utils.e8m0_to_f32(s.reshape(e * N, -1)).view(e, N, -1, 1)
    return (v * sc).view(e, N, K2 * 2)


class BaselineWeights:
    """Kernel-format mxfp4 copy: w1 [E, 2I, H/2], w2 [E, H, I/2]."""

    def __init__(self, fmt, w13, s13, w2, s2):
        from aiter import dtypes
        from aiter.ops.shuffle import shuffle_weight
        from aiter.utility import fp4_utils

        if fmt != "mxfp4":
            raise ValueError(f"BaselineWeights: unsupported fmt {fmt!r}")
        self.fmt = fmt
        E = w13.shape[0]
        self.w1 = shuffle_weight(w13.view(dtypes.fp4x2), layout=(16, 16))
        self.w2 = shuffle_weight(w2.view(dtypes.fp4x2), layout=(16, 16))
        self.s1 = fp4_utils.e8m0_shuffle(s13.reshape(E * s13.shape[1], -1)).view(
            dtypes.fp8_e8m0
        )
        self.s2 = fp4_utils.e8m0_shuffle(s2.reshape(E * s2.shape[1], -1)).view(
            dtypes.fp8_e8m0
        )
        self.w1.is_shuffled = True
        self.w2.is_shuffled = True


def baseline_moe(bw: BaselineWeights, x, topk_weights, topk_ids, swiglu_limit):
    """aiter.fused_moe (graph-safe vLLM custom op) on the mxfp4 copy."""
    from aiter import ActivationType, QuantType
    from aiter.ops.flydsl.moe_common import GateMode

    from vllm._aiter_ops import rocm_aiter_ops

    return rocm_aiter_ops.fused_moe(
        x,
        bw.w1,
        bw.w2,
        topk_weights.to(torch.float32),
        topk_ids.to(torch.int32),
        activation_method=ActivationType.Swiglu.value,
        quant_method=QuantType.per_1x32.value,
        doweight_stage1=False,
        w1_scale=bw.s1,
        w2_scale=bw.s2,
        gate_mode=GateMode.SEPARATED.value,
        swiglu_limit=0.0 if swiglu_limit is None else float(swiglu_limit),
        output_dtype=x.dtype,
    )
