# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

r"""
Chunkwise training path for GDN2-Parallax.

The base GDN-2 recurrence is delegated to ``chunk_gdn2``.  The Parallax
correction is another GDN-2 recurrence over centered keys/values:

    C_t = GDN2State(k_t - m^k_{t-1}, v_t - m^v_{t-1})
    corr_t = C_t^T rho_t

This file is intentionally a correct composite first step.  A future fused
kernel can merge the two state recurrences and move the EMA centering into
Triton once the math and gradient tests are locked.
"""

from __future__ import annotations

import math

import torch

from fla.ops.gdn2 import chunk_gdn2
from fla.utils import input_guard


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return (x.float() / torch.sqrt(torch.sum(x.float() * x.float(), dim=-1, keepdim=True) + 1e-6)).to(x.dtype)


def _split_state(
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if initial_state is None:
        return None, None, None, None
    if not isinstance(initial_state, (tuple, list)) or len(initial_state) != 4:
        raise ValueError("initial_state must be a tuple/list of (h, c, mk, mv).")
    return initial_state[0], initial_state[1], initial_state[2], initial_state[3]


def _center_dense_loop(
    x: torch.Tensor,
    mean: torch.Tensor,
    mean_decay: float,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    centered = []
    update_rate = 1.0 - mean_decay
    T = x.shape[1]
    for t in range(T):
        x_t = x[:, t].float()
        centered.append((x_t - mean).to(output_dtype))
        mean = mean_decay * mean + update_rate * x_t.detach()
    return torch.stack(centered, dim=1), mean


def _center_dense(
    x: torch.Tensor,
    initial_mean: torch.Tensor | None,
    mean_decay: float,
    num_sequences: int,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, D = x.shape
    if initial_mean is None:
        mean = torch.zeros(num_sequences, H, D, device=x.device, dtype=torch.float32)
    else:
        mean = initial_mean.float()
    if mean.shape != (B, H, D):
        raise ValueError(f"initial mean must have shape {(B, H, D)}, got {tuple(mean.shape)}.")
    if T == 0:
        return x.new_empty(B, T, H, D, dtype=output_dtype), mean
    if mean_decay == 0.0:
        x_detached = x.detach().float()
        previous = torch.cat([mean[:, None], x_detached[:, :-1]], dim=1)
        return (x.float() - previous).to(output_dtype), x_detached[:, -1]
    if -(T - 1) * math.log(mean_decay) > 80.0:
        return _center_dense_loop(x, mean, mean_decay, output_dtype)

    idx = torch.arange(T, device=x.device, dtype=torch.float32)
    pow_pos = torch.pow(mean_decay, idx)
    pow_neg = torch.pow(mean_decay, -idx)
    weighted_prefix = (x.detach().float() * pow_neg.view(1, T, 1, 1)).cumsum(dim=1)
    exclusive_prefix = torch.cat([torch.zeros_like(weighted_prefix[:, :1]), weighted_prefix[:, :-1]], dim=1)

    tail = torch.cat([torch.zeros(1, device=x.device, dtype=torch.float32), torch.pow(mean_decay, idx[1:] - 1.0)])
    previous = (
        pow_pos.view(1, T, 1, 1) * mean[:, None]
        + (1.0 - mean_decay) * tail.view(1, T, 1, 1) * exclusive_prefix
    )
    final_mean = (
        (mean_decay ** T) * mean
        + (1.0 - mean_decay) * (mean_decay ** (T - 1)) * weighted_prefix[:, -1]
    )
    return (x.float() - previous).to(output_dtype), final_mean


def _center_varlen(
    x: torch.Tensor,
    initial_mean: torch.Tensor | None,
    mean_decay: float,
    cu_seqlens: torch.LongTensor,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    boundaries = cu_seqlens.detach().cpu().tolist()
    num_sequences = len(boundaries) - 1
    _, _, H, D = x.shape
    if initial_mean is None:
        initial_mean = torch.zeros(num_sequences, H, D, device=x.device, dtype=torch.float32)
    elif initial_mean.shape != (num_sequences, H, D):
        raise ValueError(f"initial mean must have shape {(num_sequences, H, D)}, got {tuple(initial_mean.shape)}.")

    centered_segments = []
    final_means = []
    for i, (bos, eos) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=False)):
        centered_i, mean_i = _center_dense(
            x=x[:, bos:eos],
            initial_mean=initial_mean[i:i + 1],
            mean_decay=mean_decay,
            num_sequences=1,
            output_dtype=output_dtype,
        )
        centered_segments.append(centered_i)
        final_means.append(mean_i)
    return torch.cat(centered_segments, dim=1), torch.cat(final_means, dim=0)


def _center_inputs(
    k: torch.Tensor,
    v: torch.Tensor,
    mk0: torch.Tensor | None,
    mv0: torch.Tensor | None,
    mean_decay: float,
    cu_seqlens: torch.LongTensor | None,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if cu_seqlens is None:
        kc, mkt = _center_dense(k, mk0, mean_decay, k.shape[0], output_dtype)
        vc, mvt = _center_dense(v, mv0, mean_decay, v.shape[0], output_dtype)
    else:
        if k.shape[0] != 1:
            raise ValueError("cu_seqlens requires batch size 1 with packed sequences.")
        kc, mkt = _center_varlen(k, mk0, mean_decay, cu_seqlens, output_dtype)
        vc, mvt = _center_varlen(v, mv0, mean_decay, cu_seqlens, output_dtype)
    return kc, vc, mkt, mvt


@torch.compiler.disable
@input_guard
def chunk_gdn2_parallax(
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
    cu_seqlens_cpu: torch.LongTensor | None = None,
    mean_decay: float = 0.99,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
    disable_recompute: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    r"""
    Chunkwise GDN2-Parallax forward/backward composite.

    Args:
        q: queries of shape ``[B, T, H, K]``.
        k: keys of shape ``[B, T, H, K]``.
        v: values of shape ``[B, T, H, V]``.
        g: log-decay of shape ``[B, T, H, K]``.
        b: erase gate of shape ``[B, T, H, K]``.
        w: write gate of shape ``[B, T, H, V]``.
        rho: Parallax probe of shape ``[B, T, H, K]``.
        gamma: per-head correction scale of shape ``[H]``.
        scale: attention scale. Default: ``1 / sqrt(K)``.
        initial_state: optional tuple ``(h, c, mk, mv)``.
        output_final_state: whether to return final recurrent states.
        cu_seqlens: optional packed-sequence offsets. Requires batch size 1.
        cu_seqlens_cpu: optional CPU mirror forwarded to ``chunk_gdn2``.
        mean_decay: EMA decay for detached running key/value centers. Default: 0.99.
        use_qk_l2norm_in_kernel: L2-normalize q/k before both base and centered correction.
        chunk_size: chunk size forwarded to ``chunk_gdn2``. Default: 64.
        disable_recompute: retain forward intermediates for faster backward at extra memory cost. Default: `False`.

    Returns:
        ``(o, final_state)`` where ``final_state`` is ``(h, c, mk, mv)`` when requested.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
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

    h0, c0, mk0, mv0 = _split_state(initial_state)
    if use_qk_l2norm_in_kernel:
        q_base = _l2norm(q)
        k_base = _l2norm(k)
    else:
        q_base, k_base = q, k
    corr_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.float32

    kc, vc, mkt, mvt = _center_inputs(
        k=k_base,
        v=v,
        mk0=mk0,
        mv0=mv0,
        mean_decay=mean_decay,
        cu_seqlens=cu_seqlens,
        output_dtype=corr_dtype,
    )

    base, ht = chunk_gdn2(
        q=q_base,
        k=k_base,
        v=v,
        g=g,
        b=b,
        w=w,
        scale=scale,
        initial_state=h0,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_size=chunk_size,
        disable_recompute=disable_recompute,
    )
    corr, ct = chunk_gdn2(
        q=rho.to(corr_dtype),
        k=kc,
        v=vc,
        g=g.to(corr_dtype),
        b=b.to(corr_dtype),
        w=w.to(corr_dtype),
        scale=1.0,
        initial_state=c0,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_size=chunk_size,
        disable_recompute=disable_recompute,
    )

    o = (base.float() - gamma.float().view(1, 1, -1, 1) * corr.float()).to(v.dtype)
    if not output_final_state:
        return o, None
    return o, (ht, ct, mkt, mvt)


__all__ = ["chunk_gdn2_parallax"]
