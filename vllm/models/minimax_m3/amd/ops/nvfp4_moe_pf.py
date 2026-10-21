# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 FlyDSL MoE (prefill) for the MiniMax-M3.1 NVFP4 routed experts.

Modes (all run inside aiter.fused_moe on the MXFP8 per_1x32 gate/up-interleaved
flow; only the GEMM stages and their weight bytes are swapped):

  nv2    FlyDSL 2-stage MoE with NVFP4 B (e2m1 + e4m3/16), in-kernel decode
         (stage1 _flydsl_moe_stage1_impl b_dtype=nvfp4, stage2 mxfp4_moe_gemm2
         b_dtype=nvfp4), per-expert alpha (1/global) applied in-kernel.
  nv8w   PR #5462 8-wave MoE with NVFP4 B (R7 env: AITER_NV8_BREL=2,
         AITER_NV8_ENC7=1, AITER_NV8_B1SB=1, AITER_8W_STEADY_WAIT=1).
  xdeca  Triton NVFP4 -> fp8 (E8M0 130) decode of all experts into a buffer
         shared by all layers, then the 8-wave fp8 kernels with alpha1 in the
         stage-1 epilogue and alpha2 folded into the routing weights.

The kernels are the vendored ``m3nvpf_fly`` package (copy of the prefill study
aiter/ops/flydsl tree). The image aiter fused_moe is patched once
(install()): the stage entry points dispatch on the current mode and the
mode-specific tuned rows are served by a wrapper around get_2stage_cfgs.
All per-call state is Python-side; nothing syncs with the device.
"""

import os
import threading

import torch

# R7 nv8w defaults; module-level reads in the vendored 8-wave kernels, so they
# must be set before m3nvpf_fly is imported. User values win.
R7_ENV = (
    ("AITER_NV8_BREL", "2"),
    ("AITER_NV8_ENC7", "1"),
    ("AITER_NV8_B1SB", "1"),
    ("AITER_8W_STEADY_WAIT", "1"),
)
for _k, _v in R7_ENV:
    os.environ.setdefault(_k, _v)

# Tuned rows (prefill study r6/cfg/nv_tp4.json, pr8w_tp4.json; TP4 I=768).
# token tier -> (kernelName1, kernelName2, block_m); tiers < 1024 use 1024.
_NV2_S1 = "flydsl_moe1_afp8_wfp8_bf16_t64x128x256_{}_gui_fp8"
_NV2_S2 = "flydsl_moe2_layout_afp8_wfp8_bf16_t64x128x128_atomic{}_sbm64"
NV2_CFG = {
    1024: (_NV2_S1.format("w2"), _NV2_S2.format("_nt"), 64),
    2048: (_NV2_S1.format("w3_bnt0"), _NV2_S2.format(""), 64),
    4096: (_NV2_S1.format("w3_bnt0"), _NV2_S2.format(""), 64),
    8192: (_NV2_S1.format("w2_bnt0"), _NV2_S2.format(""), 64),
    16384: (_NV2_S1.format("w2_bnt0"), _NV2_S2.format(""), 64),
    32768: (_NV2_S1.format("w3_bnt0"), _NV2_S2.format(""), 64),
}
_W8 = (
    "flydsl_moe1_mxfp8_8w_t256x256_xcd3",
    "flydsl_moe2_mxfp8_8w_t256x256_xcd3",
    256,
)
W8_CFG = {t: _W8 for t in NV2_CFG}
MODES = ("nv2", "nv8w", "xdeca")
_TIERS = tuple(1 << i for i in range(16)) + (32768, 131072)  # get_padded_M


def _row_for(cfg, tier):
    for t in sorted(cfg):
        if tier <= t:
            return cfg[t]
    return cfg[max(cfg)]


# ----------------------------------------------------------------- layouts
_CHUNK = 8  # experts per layout chunk (bounds int64 temporaries)


def codes_of(packed):
    """u8 [..., K/2] -> int64 codes [..., K] (even element = low nibble)."""
    return torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2).long()


def perm_pack(packed):
    """Raw packed u8 [..., N, K/2] -> nv2 kernel nibble order (pre-shuffle)."""
    c = codes_of(packed)
    sh = c.shape
    c = c.view(*sh[:-1], sh[-1] // 16, 2, 2, 4)
    by = c[..., 0, :] | (c[..., 1, :] << 4)
    by = by.reshape(*sh[:-1], sh[-1] // 128, 2, 4, 8)
    by = by.transpose(-3, -2).contiguous()
    return by.view(*sh[:-1], sh[-1] // 2).to(torch.uint8)


def nv2_weight_layout(packed, gate_up):
    from aiter.ops.shuffle import shuffle_weight_a16w4

    return shuffle_weight_a16w4(perm_pack(packed), 16, gate_up)


def nv2_scale_layout(scale, gate_up):
    """e4m3 [E, N, K/16] -> u8 dwords [n16][K/256][4 g][16 nintra], gui rows."""
    E, N, KB = scale.shape
    s = scale.view(torch.uint8)
    if gate_up:
        s = s.view(E, 2, N // 32, 16, KB).permute(0, 2, 1, 3, 4)
    kt = KB // 16
    s = s.reshape(E * N // 16, 16, kt, 2, 2, 4).permute(0, 2, 5, 1, 3, 4)
    return s.contiguous().view(-1)


def _gui_rows(t, gate_up):
    if not gate_up:
        return t
    E, N = t.shape[:2]
    return t.view(E, 2, N // 32, 16, *t.shape[2:]).transpose(1, 2).reshape(t.shape)


def nv8w_codes(packed, gate_up):
    """Raw packed u8 [E, N, K/2] -> 8-wave nv layout [E][N/16][K/32][16][16 B]."""
    c = _gui_rows(codes_of(packed), gate_up)
    E, N, K = c.shape
    c = c.view(E, N, K // 16, 2, 2, 4)
    if os.environ.get("AITER_NV8_ENC7", "0") == "1":
        lo, hi = c[..., 0, :], c[..., 1, :]
        by = (
            (lo & 7)
            | ((torch.roll(hi, 1, dims=-1) >> 3) & 1) << 3
            | (hi & 7) << 4
            | ((lo >> 3) & 1) << 7
        )
    else:
        by = c[..., 0, :] | (c[..., 1, :] << 4)
    by = by.reshape(E, N // 16, 16, K // 32, 16).permute(0, 1, 3, 2, 4)
    return by.contiguous().to(torch.uint8).view(E, N, K // 2)


def nv8w_scales(scale, gate_up):
    """e4m3 [E, N, K/16] -> u8 [E][N/16][K/64][16 rows][4 B] (flat)."""
    s = _gui_rows(scale.view(torch.uint8), gate_up)
    E, N, KB = s.shape
    s = s.reshape(E, N // 16, 16, KB // 4, 4).permute(0, 1, 3, 2, 4)
    return s.contiguous().view(-1)


def _chunked(fn, src, out_shape, dtype, *args):
    """Apply a per-expert layout fn in expert chunks into one tensor."""
    E = src[0].shape[0]
    out = torch.empty(out_shape, dtype=dtype, device=src[0].device)
    per = out.numel() // E
    flat = out.view(-1)
    for e0 in range(0, E, _CHUNK):
        e1 = min(E, e0 + _CHUNK)
        r = fn(*[t[e0:e1] for t in src], *args)
        flat[e0 * per : e1 * per].copy_(r.reshape(-1))
        del r
    return out


# --------------------------------------------------------- explicit decode
_xdec_kernel = None


def _get_xdec_kernel():
    global _xdec_kernel
    if _xdec_kernel is not None:
        return _xdec_kernel
    import triton
    import triton.language as tl

    @triton.jit
    def _nv_decode_kernel(p_ptr, s_ptr, o_ptr, per_e, BLOCK: tl.constexpr):
        e = tl.program_id(1)
        base = e.to(tl.int64) * per_e + tl.program_id(0) * BLOCK
        offs = tl.arange(0, BLOCK)
        p = tl.load(p_ptr + base + offs).to(tl.int32)
        s = tl.load(s_ptr.to(tl.pointer_type(tl.float8e4nv)) + (base + offs) // 8)
        # e2m1 nibble -> f16 bits (sign | e2m1 << 9) = value * 2^-14 (exact);
        # 2048 = 2^14 / 8 (fp8 = RNE(code * s / 8), E8M0 130 restores x8).
        s = s.to(tl.float32) * 2048.0
        blo = ((p & 7) << 9) | ((p & 8) << 12)
        bhi = ((p & 0x70) << 5) | ((p & 0x80) << 8)
        vlo = blo.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32) * s
        vhi = bhi.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32) * s
        q = tl.interleave(vlo.to(tl.float8e4nv), vhi.to(tl.float8e4nv))
        o = o_ptr.to(tl.pointer_type(tl.float8e4nv))
        tl.store(o + 2 * base + tl.arange(0, 2 * BLOCK), q)

    _xdec_kernel = _nv_decode_kernel
    return _xdec_kernel


def _chunk_map(N, K, gate_up, dev):
    """Source 16-element chunk of every output chunk of shuffle_weight_a16w4."""
    from aiter.ops.shuffle import shuffle_weight_a16w4

    idx = torch.arange(N * K, dtype=torch.int32, device=dev).view(1, N, K)
    sh = shuffle_weight_a16w4(idx, 16, gate_up).view(-1, 16)
    assert bool((sh[:, 1:] - sh[:, :1] == torch.arange(1, 16, device=dev)).all())
    assert bool((sh[:, 0] % 16 == 0).all())
    return (sh[:, 0] // 16).long()


def xdec_prep(packed, scale, gate_up):
    """Raw [E, N, K/2] u8 + [E, N, K/16] e4m3 -> chunk-ordered packed/scales."""
    E, N, K2 = packed.shape
    m = _chunk_map(N, 2 * K2, gate_up, packed.device)
    p = torch.empty_like(packed)
    s = torch.empty(E, N * K2 // 8, dtype=torch.uint8, device=packed.device)
    for e0 in range(0, E, _CHUNK):
        e1 = min(E, e0 + _CHUNK)
        p[e0:e1] = packed[e0:e1].reshape(e1 - e0, -1, 8)[:, m].view(e1 - e0, N, K2)
        s[e0:e1] = scale[e0:e1].view(torch.uint8).reshape(e1 - e0, -1)[:, m]
    return p, s


def xdec_decode(p, s, out, block=4096):
    E = p.shape[0]
    per_e = p[0].numel()
    assert per_e % block == 0
    _get_xdec_kernel()[(per_e // block, E)](p, s, out, per_e, BLOCK=block,
                                             num_warps=4)
    return out


class _Shared:
    """Per-process objects shared by all layers (keyed by shape)."""

    placeholders: dict = {}
    xdec_bufs: dict = {}


def _placeholders(E, I2, H, dev):
    """Zero-byte fp8 w1/w2 + E8M0 scales: shape/dtype carriers for fused_moe in
    nv2 / nv8w mode (the patched stages substitute the NVFP4 tensors)."""
    key = (E, I2, H, str(dev))
    r = _Shared.placeholders.get(key)
    if r is None:
        from aiter import dtypes

        f8 = torch.zeros(1, 1, 1, dtype=torch.float8_e4m3fn, device=dev)
        sc = torch.full((1, 1), 127, dtype=torch.uint8, device=dev)
        w1 = f8.expand(E, I2, H)
        w2 = f8.expand(E, H, I2 // 2)
        for w in (w1, w2):
            w.is_shuffled = True
        s1 = sc.expand(E * I2, H // 32).view(dtypes.fp8_e8m0)
        s2 = sc.expand(E * H, I2 // 64).view(dtypes.fp8_e8m0)
        r = (w1, w2, s1, s2)
        _Shared.placeholders[key] = r
    return r


def _xdec_buffers(E, I2, H, dev):
    """fp8 decode targets (shuffle_weight_a16w4 layout) + constant E8M0 130."""
    key = (E, I2, H, str(dev))
    r = _Shared.xdec_bufs.get(key)
    if r is None:
        from aiter import dtypes
        from aiter.ops.shuffle import shuffle_scale_a16w4
        from aiter.utility import fp4_utils

        w1 = torch.empty(E, I2, H, dtype=torch.float8_e4m3fn, device=dev)
        w2 = torch.empty(E, H, I2 // 2, dtype=torch.float8_e4m3fn, device=dev)
        for w in (w1, w2):
            w.is_shuffled = True
        s1 = torch.full((E * I2, H // 32), 130, dtype=torch.uint8, device=dev)
        s2 = torch.full((E * H, I2 // 64), 130, dtype=torch.uint8, device=dev)
        s1 = shuffle_scale_a16w4(s1.view(dtypes.fp8_e8m0), E, True)
        s2 = fp4_utils.e8m0_shuffle(s2.view(dtypes.fp8_e8m0))
        r = (w1, w2, s1.view(dtypes.fp8_e8m0), s2.view(dtypes.fp8_e8m0))
        _Shared.xdec_bufs[key] = r
    return r


class NvPrefillWeights:
    """NVFP4 layouts of one layer for the requested modes.

    alpha13 / alpha2: per-expert fp32 dequant multipliers (1 / global scale).
    """

    def __init__(self, w13p, w13s, w2p, w2s, alpha13, alpha2, modes):
        E, N13, H2 = w13p.shape
        H, I = H2 * 2, N13 // 2
        dev = w13p.device
        self.shape = (E, N13, H)
        self.alpha1 = alpha13.float().contiguous()
        self.alpha2 = alpha2.float().contiguous()
        self.modes = tuple(modes)
        u8 = torch.uint8
        if "nv2" in modes:
            self.nv2_w1 = _chunked(nv2_weight_layout, (w13p,), (E, N13, H2), u8, True)
            self.nv2_w2 = _chunked(nv2_weight_layout, (w2p,), (E, H, I // 2), u8, False)
            self.nv2_s1 = _chunked(nv2_scale_layout, (w13s,), (E * N13 * H // 16,), u8, True)
            self.nv2_s2 = _chunked(nv2_scale_layout, (w2s,), (E * H * I // 16,), u8, False)
        if "nv8w" in modes:
            self.nv8_w1 = _chunked(nv8w_codes, (w13p,), (E, N13, H2), u8, True)
            self.nv8_w2 = _chunked(nv8w_codes, (w2p,), (E, H, I // 2), u8, False)
            self.nv8_s1 = _chunked(nv8w_scales, (w13s,), (E * N13 * H // 16,), u8, True)
            self.nv8_s2 = _chunked(nv8w_scales, (w2s,), (E * H * I // 16,), u8, False)
        if "xdeca" in modes:
            self.xd_p1, self.xd_s1 = xdec_prep(w13p, w13s, True)
            self.xd_p2, self.xd_s2 = xdec_prep(w2p, w2s, False)
            _xdec_buffers(E, N13, H, dev)
        _placeholders(E, N13, H, dev)


# ------------------------------------------------------------- aiter patch
class _State(threading.local):
    mode = None
    w = None


_ST = _State()
_PATCH = {}
_ARGN = ("hidden_states", "w1", "w2", "sorted_ids", "sorted_expert_ids",
         "num_valid_ids", "out", "topk")  # fmt: skip


def install():
    """Patch the image aiter fused_moe stage entry points (idempotent)."""
    if _PATCH:
        return
    import aiter.fused_moe as fm
    from aiter.ops.flydsl import moe_kernels as imk
    from aiter.ops.flydsl.kernels import mxmoe_dispatcher as imxd
    from m3nvpf_fly import moe_kernels as vmk
    from m3nvpf_fly.kernels import mxmoe_dispatcher as vmxd

    orig_s1 = imk.flydsl_moe_stage1
    orig_g2 = imxd.mxfp4_moe_gemm2
    orig8_s1 = imk.flydsl_mxfp_moe_stage1
    orig8_s2 = imk.flydsl_mxfp_moe_stage2
    orig_cfgs = fm.get_2stage_cfgs

    def stage1(**kw):
        if _ST.mode == "nv2":
            if kw.get("b_dtype") != "fp8" or kw.get("gate_mode") != "interleave":
                raise RuntimeError(
                    "M3 nv2: expected fp8 gate/up-interleaved stage1, got "
                    f"b_dtype={kw.get('b_dtype')} gate_mode={kw.get('gate_mode')}"
                )
            w = _ST.w
            kw.update(b_dtype="nvfp4", w1=w.nv2_w1, w1_scale=w.nv2_s1,
                      nv_alpha=w.alpha1)  # fmt: skip
            return vmk._flydsl_moe_stage1_impl(**kw)
        return orig_s1(**kw)

    def gemm2(**kw):
        if _ST.mode == "nv2":
            if kw.get("b_dtype") != "fp8" or kw.get("epilog") != "atomic":
                raise RuntimeError(
                    "M3 nv2: expected fp8 atomic stage2, got "
                    f"b_dtype={kw.get('b_dtype')} epilog={kw.get('epilog')}"
                )
            w = _ST.w
            kw.update(b_dtype="nvfp4", w2_u8=w.nv2_w2, w2_scale_u8=w.nv2_s2,
                      nv_alpha=w.alpha2)  # fmt: skip
            return vmxd.mxfp4_moe_gemm2(**kw)
        return orig_g2(**kw)

    def stage1_8w(*args, **kw):
        mode = _ST.mode
        if mode not in ("nv8w", "xdeca"):
            return orig8_s1(*args, **kw)
        kw.update(zip(_ARGN, args))
        w = _ST.w
        if mode == "nv8w":
            kw.update(w1=w.nv8_w1, w1_scale=w.nv8_s1, nv_alpha=w.alpha1)
        else:
            kw.update(alpha=w.alpha1)
        return vmk.flydsl_mxfp_moe_stage1(**kw)

    def stage2_8w(*args, **kw):
        mode = _ST.mode
        if mode not in ("nv8w", "xdeca"):
            return orig8_s2(*args, **kw)
        kw.update(zip(_ARGN, args))
        if mode == "nv8w":
            w = _ST.w
            kw.update(w2=w.nv8_w2, w2_scale=w.nv8_s2, nv_alpha=w.alpha2)
        return vmk.flydsl_mxfp_moe_stage2(**kw)

    for new, old in ((stage1_8w, orig8_s1), (stage2_8w, orig8_s2)):
        new.__dict__.update(getattr(old, "__dict__", {}))
    stage1_8w._is_mxfp8_prefill_stage1 = True
    stage2_8w._is_flydsl_stage2 = True

    tables = {}
    cache = {}

    def mode_tables(mode, args):
        # args = get_2stage_cfgs positional args (token, model_dim, inter_dim,
        # expert, topk, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
        # activation, doweight_stage1, ...); keys built like _lookup_cfg.
        (_, model_dim, inter_dim, expert, topk, dtype, qa, qw, qt, g1u1, act,
         dw) = args[:12]  # fmt: skip
        tail = (model_dim, inter_dim, expert, topk, str(act), str(dtype),
                str(qa), str(qw), str(qt), g1u1, dw)  # fmt: skip
        key = (mode, tail)
        if key in tables:
            return tables[key]
        primary, fallback = fm.cfg_2stages
        head = (fm.get_gfx_runtime(), fm.get_cu_num())
        tmpl = next(iter(primary.values()))
        cfg = NV2_CFG if mode == "nv2" else W8_CFG
        new = dict(primary)
        for tier in _TIERS:
            kn1, kn2, bm = _row_for(cfg, tier)
            row = dict(tmpl)
            row.update(kernelName1=kn1, kernelName2=kn2, block_m=bm, ksplit=0,
                       run_1stage=0)  # fmt: skip
            new[head + (tier,) + tail] = row
        tables[key] = (new, fallback)
        return tables[key]

    def get_2stage_cfgs(*args, **kw):
        mode = _ST.mode
        if mode is None or kw.get("config_file") is not None:
            return orig_cfgs(*args, **kw)
        key = (mode, args, tuple(sorted(kw.items())))
        r = cache.get(key)
        if r is None:
            if fm.cfg_2stages is None:
                orig_cfgs(*args, **kw)  # loads the merged tuned tables
            tab = mode_tables(mode, args)
            saved = fm.cfg_2stages
            fm.cfg_2stages = tab
            try:
                r = orig_cfgs.__wrapped__(*args, **kw)
            finally:
                fm.cfg_2stages = saved
            want = _row_for(NV2_CFG if mode == "nv2" else W8_CFG, args[0])
            got = (
                getattr(r.stage1, "keywords", {}).get("kernelName"),
                getattr(r.stage2, "keywords", {}).get("kernelName"),
                int(r.block_m),
            )
            if got != tuple(want):
                raise RuntimeError(f"M3 {mode}: config selected {got}, want "
                                   f"{want} for {args[:12]}")  # fmt: skip
            cache[key] = r
        return r

    get_2stage_cfgs.cache_clear = orig_cfgs.cache_clear
    get_2stage_cfgs.__wrapped__ = orig_cfgs.__wrapped__

    imk.flydsl_moe_stage1 = stage1
    imxd.mxfp4_moe_gemm2 = gemm2
    imk.flydsl_mxfp_moe_stage1 = stage1_8w
    imk.flydsl_mxfp_moe_stage2 = stage2_8w
    fm.get_2stage_cfgs = get_2stage_cfgs
    _PATCH.update(orig_s1=orig_s1, orig_g2=orig_g2, orig8_s1=orig8_s1,
                  orig8_s2=orig8_s2, orig_cfgs=orig_cfgs)  # fmt: skip


def nv_moe(w: NvPrefillWeights, mode, x, topk_weights, topk_ids, swiglu_limit):
    """Routed-expert MoE of one layer on NVFP4 weights in ``mode``."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    assert mode in w.modes, (mode, w.modes)
    install()
    E, N13, H = w.shape
    tw = topk_weights.to(torch.float32)
    ti = topk_ids.to(torch.int32)
    if mode == "xdeca":
        w1, w2, s1, s2 = _xdec_buffers(E, N13, H, x.device)
        xdec_decode(w.xd_p1, w.xd_s1, w1)
        xdec_decode(w.xd_p2, w.xd_s2, w2)
        tw = tw * w.alpha2[ti.long()]
    else:
        w1, w2, s1, s2 = _placeholders(E, N13, H, x.device)
    _ST.mode, _ST.w = mode, w
    try:
        return fused_moe(
            x,
            w1,
            w2,
            tw,
            ti,
            w1_scale=s1,
            w2_scale=s2,
            quant_type=QuantType.per_1x32,
            activation=ActivationType.Swiglu,
            doweight_stage1=False,
            swiglu_limit=0.0 if swiglu_limit is None else float(swiglu_limit),
            gate_mode="interleave",
        )
    finally:
        _ST.mode, _ST.w = None, None
