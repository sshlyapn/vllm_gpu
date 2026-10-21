# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the NVFP4 routed-expert decode kernels (moe_dec_r8.hip).

The kernels decode NVFP4 weights to f16 in registers (exact), apply the E4M3
block scale and the per-expert global multiplier, and run f16 MFMA on bf16 ->
f16 activations (no activation quant). Three launches per MoE: fused
sort + cast + output zeroing, stage 1 (gate/up + SwiGLU-OAI), stage 2 (down,
weighted reduce into the bf16 output). Graph safe: every launch goes to the
current stream and all buffers come from the torch allocator.

The library is prebuilt in the image (VLLM_M3_NVFP4_MOE_HIP_LIB).
"""

import ctypes
import os

import torch

F_F16 = 2
DEFAULT_LIB = "/opt/m3/nvfp4moe/libmoe_dec_r8.so"
# Largest T * topk the fused sort kernel handles (SORT_THREADS * SORT_PT).
MAX_SORT_ROWS = 4096

# Measured TP4 (E=128, I=768) best (stage1, stage2) configs by T (r8 round,
# results/nvfp4_moe/decode/logs/r8/final_tab_tp4.txt). A T between two rows
# uses the next larger row.
TP4_DISPATCH = (
    (1, 207, 720),
    (2, 240, 720),
    (4, 207, 720),
    (64, 240, 720),
    (256, 240, 721),
    (512, 249, 723),
    (1024, 1200, 739),
)


class _Params(ctypes.Structure):
    _fields_ = (
        [
            (n, ctypes.c_void_p)
            for n in (
                "a",
                "a_sc",
                "w",
                "w_sc",
                "inv_gs",
                "sorted_ids",
                "sorted_w",
                "sorted_eids",
                "num_valid",
                "out",
                "out_sc",
            )
        ]
        + [(n, ctypes.c_int) for n in ("T", "topk", "K", "ntiles", "n_out", "mx8_e")]
        + [(n, ctypes.c_void_p) for n in ("w8", "w8_sc", "zo")]
        + [("zn16", ctypes.c_int)]
    )


class _PrepParams(ctypes.Structure):
    _fields_ = [
        (n, ctypes.c_void_p)
        for n in (
            "x",
            "q",
            "sc",
            "out",
            "ids",
            "w",
            "sorted_ids",
            "sorted_w",
            "sorted_eids",
            "num_valid",
        )
    ] + [(n, ctypes.c_int) for n in ("T", "H", "topk", "E", "BM", "nblk")]


_LIBS: dict[int, ctypes.CDLL] = {}
_CFGS: dict[int, dict] | None = None


def _fp8_lut(device) -> torch.Tensor:
    """[m 0..7][shift 0..15] -> 8 e4m3 bytes = |E2M1| * (1 + m/8) * 2^-shift.

    Only the fp8-LUT kernel family reads it; the library requires it set.
    """
    mags = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)
    m = torch.arange(8, dtype=torch.float64).view(8, 1, 1)
    sh = torch.arange(16, dtype=torch.float64).view(1, 16, 1)
    v = (mags.view(1, 1, 8) * (1 + m / 8) * torch.exp2(-sh)).float()
    b = v.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    return b.view(torch.int32).view(-1).contiguous().to(device)


def lib() -> ctypes.CDLL:
    """Load the prebuilt library once per process and device."""
    dev = torch.cuda.current_device()
    if dev in _LIBS:
        return _LIBS[dev]
    path = os.environ.get("VLLM_M3_NVFP4_MOE_HIP_LIB", DEFAULT_LIB)
    if not _LIBS:
        handle = ctypes.CDLL(path)
    else:
        handle = next(iter(_LIBS.values()))
    lut = _fp8_lut(torch.device("cuda", dev))
    rc = handle.dec_set_fp8_lut(ctypes.c_void_p(lut.data_ptr()))
    torch.cuda.synchronize()
    if rc != 0:
        raise RuntimeError(f"dec_set_fp8_lut failed: {rc}")
    _LIBS[dev] = handle
    return handle


def cfgs() -> dict[int, dict]:
    global _CFGS
    if _CFGS is not None:
        return _CFGS
    L = lib()
    out = {}
    buf = (ctypes.c_int * 8)()
    for i in range(L.dec_num_cfgs()):
        L.dec_cfg_info(i, buf)
        cid, fmt, s1, mt, nt, wk, wn, kc = list(buf)
        lx = bool(kc >> 16 & 1)
        sx = bool(kc >> 16 & 2)
        out[cid] = dict(
            fmt=fmt,
            s1=bool(s1),
            mt=mt,
            nt=nt,
            wk=wk,
            wn=wn,
            kc=kc & 255,
            lx=lx,
            sx=sx,
            d=kc >> 20 & 15,
            ns=kc >> 24 & 15,
            sg=bool(kc >> 24 & 4) if sx else (not lx and bool(kc & 1024)),
        )
    _CFGS = out
    return out


def pick_cfg(T: int) -> tuple[int, int]:
    for t, c1, c2 in TP4_DISPATCH:
        if T <= t:
            return c1, c2
    return TP4_DISPATCH[-1][1], TP4_DISPATCH[-1][2]


def _ptr(t: torch.Tensor | None):
    return None if t is None else t.data_ptr()


# ------------------------------------------------------------------ layouts
def tile_w(packed: torch.Tensor) -> torch.Tensor:
    E, N, K2 = packed.shape
    return (
        packed.view(E, N // 16, 16, K2 // 64, 4, 16)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
    )


def tile_sc_nv(sc: torch.Tensor) -> torch.Tensor:
    E, N, KS = sc.shape
    return (
        sc.view(E, N // 16, 16, KS // 16, 2, 4, 2)
        .permute(0, 1, 3, 5, 2, 4, 6)
        .contiguous()
    )


def sg_sc(ts: torch.Tensor) -> torch.Tensor:
    """Pair layout [E][NT][KT/2][64][4 B] -> SG layout [E][NT][KT/6][64][12 B]."""
    E, NT = ts.shape[:2]
    return ts.reshape(E, NT, -1, 3, 64, 4).permute(0, 1, 2, 4, 3, 5).contiguous()


class NvFp4DecodeWeights:
    """One MoE layer in the F16 kernel layout.

    w13: packed uint8 [E, 2I, H/2] (gate rows then up rows), sc13 E4M3 bytes
    [E, 2I, H/16]; w2: [E, H, I/2], sc2 [E, H, I/16]; g13 [E, 2] and g2 [E]
    fp32 multipliers (1 / weight_global_scale).
    """

    def __init__(self, w13, sc13, w2, sc2, g13, g2):
        self.E, N13, H2 = w13.shape
        self.I, self.H = N13 // 2, H2 * 2
        self.g13 = g13.float().contiguous()
        self.g2 = g2.float().contiguous()
        sc13 = sc13.view(torch.uint8)
        sc2 = sc2.view(torch.uint8)
        self.w13, self.w2 = tile_w(w13), tile_w(w2)
        self.s13, self.s2 = tile_sc_nv(sc13), tile_sc_nv(sc2)
        self.s13g, self.s2g = sg_sc(self.s13), sg_sc(self.s2)


def _launch(cfg_id, cfg, a, wt, w_sc, inv_gs, sort, out, T, topk, K, ntiles, n_out, E):
    sorted_ids, sorted_w, sorted_eids, num_valid = sort
    p = _Params(
        _ptr(a), None, _ptr(wt), _ptr(w_sc), _ptr(inv_gs), _ptr(sorted_ids),
        _ptr(sorted_w), _ptr(sorted_eids), _ptr(num_valid), _ptr(out), None,
        T, topk, K, ntiles, n_out, -1, None, None, None, 0,
    )  # fmt: skip
    BM = 16 * cfg["mt"]
    gy = min(E, T * topk) + (T * topk + BM - 1) // BM
    gy = (gy + 7) // 8 * 8
    if cfg["s1"]:
        gx = (n_out // 16) // (cfg["nt"] // 2 * cfg["wn"])
    else:
        gx = (n_out // 16) // (cfg["nt"] * cfg["wn"])
    st = torch.cuda.current_stream().cuda_stream
    r = lib().dec_launch(cfg_id, ctypes.byref(p), gx, gy, ctypes.c_void_p(st))
    if r != 0:
        raise RuntimeError(f"NVFP4 MoE decode launch {cfg_id} failed: {r}")


def nvfp4_moe_decode(
    W: NvFp4DecodeWeights,
    x: torch.Tensor,  # bf16 [T, H]
    topk_ids: torch.Tensor,  # [T, topk]
    topk_w: torch.Tensor,  # [T, topk]
) -> torch.Tensor:
    """Routed-expert sum (no routed scaling) as bf16 [T, H]."""
    T, H = x.shape
    topk = topk_ids.shape[1]
    if T * topk > MAX_SORT_ROWS:
        raise ValueError(f"NVFP4 MoE decode supports T * topk <= {MAX_SORT_ROWS}")
    c1, c2 = pick_cfg(T)
    cs = cfgs()
    cf1, cf2 = cs[c1], cs[c2]
    BM = 16 * cf1["mt"]
    dev = x.device
    x = x.contiguous()
    n = T * topk
    nblk = min(W.E, n) + (n + BM - 1) // BM
    sids = torch.empty(nblk * BM, dtype=torch.int32, device=dev)
    sw = torch.empty(nblk * BM, dtype=torch.float32, device=dev)
    seids = torch.empty((nblk + 7) // 8 * 8, dtype=torch.int32, device=dev)
    nv = torch.empty(1, dtype=torch.int32, device=dev)
    out = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    xq = torch.empty(T, H, dtype=torch.float16, device=dev)
    ids = topk_ids.to(torch.int32).contiguous()
    tw = topk_w.float().contiguous()
    pp = _PrepParams(
        _ptr(x), _ptr(xq), None, _ptr(out), _ptr(ids), _ptr(tw), _ptr(sids),
        _ptr(sw), _ptr(seids), _ptr(nv), T, H, topk, W.E, BM, nblk,
    )  # fmt: skip
    st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    r = lib().dec_sort_quant(ctypes.byref(pp), 1, st)
    if r != 0:
        raise RuntimeError(f"NVFP4 MoE decode sort failed: {r}")
    sort = (sids, sw, seids, nv)
    a2 = torch.empty(n, W.I, dtype=torch.float16, device=dev)
    s13 = W.s13g if cf1["sg"] else W.s13
    _launch(c1, cf1, xq, W.w13, s13, W.g13, sort, a2, T, topk, W.H,
            2 * W.I // 16, W.I, W.E)  # fmt: skip
    s2 = W.s2g if cf2["sg"] else W.s2
    _launch(c2, cf2, a2, W.w2, s2, W.g2, sort, out, T, topk, W.I, W.H // 16, W.H, W.E)
    return out
