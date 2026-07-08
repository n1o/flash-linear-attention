# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

r"""
Forward-only recurrent Triton kernel for GDN2-Parallax.

This is the inference-time counterpart of the PyTorch reference recurrence in
``naive_recurrent_gdn2_parallax``.  Training still uses the reference path until
a chunked backward kernel exists.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.utils import input_guard


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_gdn2_parallax_fwd_kernel(
    q,
    k,
    v,
    g,
    b,
    w,
    rho,
    gamma,
    o,
    h0,
    c0,
    mk0,
    mv0,
    ht,
    ct,
    mkt,
    mvt,
    cu_seqlens,
    scale: tl.constexpr,
    N: tl.int64,
    T: tl.int64,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    mean_decay: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_n = i_nh // H
    i_h = i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n * T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_g = g + (bos * H + i_h) * K + o_k
    p_b = b + (bos * H + i_h) * K + o_k
    p_rho = rho + (bos * H + i_h) * K + o_k
    p_v = v + (bos * H + i_h) * V + o_v
    p_w = w + (bos * H + i_h) * V + o_v
    p_o = o + (bos * H + i_h) * V + o_v

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    b_c = tl.zeros([BK, BV], dtype=tl.float32)
    b_mk = tl.zeros([BK], dtype=tl.float32)
    b_mv = tl.zeros([BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + (i_n * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
        p_c0 = c0 + (i_n * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
        p_mk0 = mk0 + (i_n * H + i_h) * K + o_k
        p_mv0 = mv0 + (i_n * H + i_h) * V + o_v
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
        b_c += tl.load(p_c0, mask=mask_h, other=0).to(tl.float32)
        b_mk += tl.load(p_mk0, mask=mask_k, other=0).to(tl.float32)
        b_mv += tl.load(p_mv0, mask=mask_v, other=0).to(tl.float32)

    b_gamma = tl.load(gamma + i_h).to(tl.float32)
    update_rate = 1.0 - mean_decay

    for _ in tl.range(0, T, num_stages=num_stages):
        b_q = tl.load(p_q, mask=mask_k, other=0, eviction_policy="evict_last").to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0, eviction_policy="evict_last").to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0, eviction_policy="evict_first").to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q *= scale

        b_g = tl.load(p_g, mask=mask_k, other=0, eviction_policy="evict_last").to(tl.float32)
        b_decay = exp(b_g)

        b_h *= b_decay[:, None]
        b_b = tl.load(p_b, mask=mask_k, other=0, eviction_policy="evict_last").to(tl.float32)
        b_bk = b_b * b_k
        b_erase_h = tl.sum(b_h * b_bk[:, None], 0)

        b_w = tl.load(p_w, mask=mask_v, other=0, eviction_policy="evict_first").to(tl.float32)
        b_write_h = b_w * b_v - b_erase_h
        b_h += b_k[:, None] * b_write_h[None, :]
        b_base = tl.sum(b_h * b_q[:, None], 0)

        b_kc = b_k - b_mk
        b_vc = b_v - b_mv
        b_c *= b_decay[:, None]
        b_erase_c = tl.sum(b_c * (b_b * b_kc)[:, None], 0)
        b_write_c = b_w * b_vc - b_erase_c
        b_c += b_kc[:, None] * b_write_c[None, :]

        b_rho = tl.load(p_rho, mask=mask_k, other=0, eviction_policy="evict_last").to(tl.float32)
        b_corr = tl.sum(b_c * b_rho[:, None], 0)
        b_o = b_base - b_gamma * b_corr
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v, eviction_policy="evict_first")

        b_mk = mean_decay * b_mk + update_rate * b_k
        b_mv = mean_decay * b_mv + update_rate * b_v

        p_q += H * K
        p_k += H * K
        p_g += H * K
        p_b += H * K
        p_rho += H * K
        p_v += H * V
        p_w += H * V
        p_o += H * V

    if STORE_FINAL_STATE:
        p_ht = ht + (i_n * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
        p_ct = ct + (i_n * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
        p_mkt = mkt + (i_n * H + i_h) * K + o_k
        p_mvt = mvt + (i_n * H + i_h) * V + o_v
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        tl.store(p_ct, b_c.to(p_ct.dtype.element_ty), mask=mask_h)
        tl.store(p_mkt, b_mk.to(p_mkt.dtype.element_ty), mask=mask_k & (i_v == 0))
        tl.store(p_mvt, b_mv.to(p_mvt.dtype.element_ty), mask=mask_v)


@torch.compiler.disable
def fused_recurrent_gdn2_parallax_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    rho: torch.Tensor,
    gamma: torch.Tensor,
    scale: float | None = None,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    mean_decay: float = 0.99,
    use_qk_l2norm_in_kernel: bool = False,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    if scale is None:
        scale = k.shape[-1] ** -0.5

    B, T, H, K = q.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = 16

    if BK > 256:
        raise ValueError(f"fused_recurrent_gdn2_parallax supports head_dim <= 256, got {K}.")
    if out is None:
        out = torch.empty_like(v)
    elif out.shape != v.shape:
        raise ValueError(f"out must match v shape; got {tuple(out.shape)} vs {tuple(v.shape)}.")

    if initial_state is None:
        h0 = c0 = mk0 = mv0 = None
    else:
        if not isinstance(initial_state, (tuple, list)) or len(initial_state) != 4:
            raise ValueError("initial_state must be a tuple/list of (h, c, mk, mv).")
        h0, c0, mk0, mv0 = initial_state

    if output_final_state:
        ht = q.new_empty(N, H, K, V, dtype=torch.float32)
        ct = q.new_empty(N, H, K, V, dtype=torch.float32)
        mkt = q.new_empty(N, H, K, dtype=torch.float32)
        mvt = q.new_empty(N, H, V, dtype=torch.float32)
        final_state = (ht, ct, mkt, mvt)
    else:
        ht = ct = mkt = mvt = None
        final_state = None

    grid = (triton.cdiv(V, BV) * N * H,)
    fused_recurrent_gdn2_parallax_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        rho=rho,
        gamma=gamma,
        o=out,
        h0=h0,
        c0=c0,
        mk0=mk0,
        mv0=mv0,
        ht=ht,
        ct=ct,
        mkt=mkt,
        mvt=mvt,
        cu_seqlens=cu_seqlens,
        scale=scale,
        N=N,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        mean_decay=mean_decay,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=4,
        num_stages=2,
    )
    return out, final_state


@input_guard
def fused_recurrent_gdn2_parallax(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    rho: torch.Tensor,
    gamma: torch.Tensor,
    scale: float | None = None,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    mean_decay: float = 0.99,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    r"""
    Token-by-token fused forward for GDN2-Parallax. Inference-only.

    Args:
        q: queries of shape ``[B, T, H, K]``.
        k: keys of shape ``[B, T, H, K]``.
        v: values of shape ``[B, T, H, V]``.
        g: log-decay of shape ``[B, T, H, K]``.
        b: erase gate of shape ``[B, T, H, K]``.
        w: write gate of shape ``[B, T, H, V]``.
        rho: Parallax probe of shape ``[B, T, H, K]``.
        gamma: per-head correction scale of shape ``[H]``.
        scale: attention scale, defaults to ``1/sqrt(K)``.
        initial_state: optional tuple ``(h, c, mk, mv)``.
        output_final_state: whether to return the final recurrent states.
        cu_seqlens: optional packed-sequence offsets. Requires batch size 1.
        mean_decay: EMA decay for the running key/value centers.
        use_qk_l2norm_in_kernel: L2-normalize q and k inside the kernel.

    Returns:
        ``(o, final_state)`` where ``o`` has shape ``[B, T, H, V]``.
    """
    if not (q.shape == k.shape == g.shape):
        raise ValueError("q, k, and g must have matching shapes.")
    if b.shape != q.shape:
        raise ValueError(f"b must match q shape; got {tuple(b.shape)} vs {tuple(q.shape)}.")
    if rho.shape != q.shape:
        raise ValueError(f"rho must match q shape; got {tuple(rho.shape)} vs {tuple(q.shape)}.")
    if w.shape != v.shape:
        raise ValueError(f"w must match v shape; got {tuple(w.shape)} vs {tuple(v.shape)}.")
    if v.shape[:3] != q.shape[:3]:
        raise ValueError(f"v must match q on batch, sequence, and head axes; got {tuple(v.shape)} vs {tuple(q.shape)}.")
    if gamma.shape != (q.shape[2],):
        raise ValueError(f"gamma must have shape [{q.shape[2]}], got {tuple(gamma.shape)}.")
    if not 0.0 <= mean_decay < 1.0:
        raise ValueError(f"mean_decay must be in [0, 1), got {mean_decay}.")
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError("cu_seqlens requires batch size 1 with packed sequences.")
        if initial_state is not None and initial_state[0].shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"initial_state has {initial_state[0].shape[0]} sequences, "
                f"but cu_seqlens describes {len(cu_seqlens) - 1}."
            )

    return fused_recurrent_gdn2_parallax_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        rho=rho,
        gamma=gamma,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        mean_decay=mean_decay,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


__all__ = [
    "fused_recurrent_gdn2_parallax",
    "fused_recurrent_gdn2_parallax_fwd",
    "fused_recurrent_gdn2_parallax_fwd_kernel",
]
