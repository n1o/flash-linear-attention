# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + 1e-6)


def _empty_state(
    q: torch.Tensor,
    v: torch.Tensor,
    num_sequences: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, H, K = q.shape
    V = v.shape[-1]
    h = torch.zeros(num_sequences, H, K, V, device=q.device, dtype=torch.float32)
    c = torch.zeros_like(h)
    mk = torch.zeros(num_sequences, H, K, device=q.device, dtype=torch.float32)
    mv = torch.zeros(num_sequences, H, V, device=q.device, dtype=torch.float32)
    return h, c, mk, mv


def _prepare_state(
    q: torch.Tensor,
    v: torch.Tensor,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    num_sequences: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if initial_state is None:
        return _empty_state(q, v, num_sequences)
    if not isinstance(initial_state, (tuple, list)) or len(initial_state) != 4:
        raise ValueError("initial_state must be a tuple/list of (h, c, mk, mv).")
    h, c, mk, mv = initial_state
    return h.float().clone(), c.float().clone(), mk.float().clone(), mv.float().clone()


def _run_segment(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    rho: torch.Tensor,
    gamma: torch.Tensor,
    scale: float,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    output_final_state: bool,
    mean_decay: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    B, T, H, K = q.shape
    V = v.shape[-1]
    h, c, mk, mv = _prepare_state(q, v, initial_state, B)

    out_dtype = v.dtype
    q = q.float() * scale
    k = k.float()
    v = v.float()
    g = g.float()
    b = b.float()
    w = w.float()
    rho = rho.float()
    gamma = gamma.float().reshape(1, H, 1)

    o = torch.empty(B, T, H, V, device=v.device, dtype=torch.float32)
    update_rate = 1.0 - mean_decay

    for t in range(T):
        q_t = q[:, t]              # [B, H, K]
        k_t = k[:, t]              # [B, H, K]
        v_t = v[:, t]              # [B, H, V]
        g_t = g[:, t]              # [B, H, K]
        b_t = b[:, t]              # [B, H, K]
        w_t = w[:, t]              # [B, H, V]
        rho_t = rho[:, t]          # [B, H, K]

        decay = g_t.exp().unsqueeze(-1)

        h = h * decay
        erase_h = ((b_t * k_t).unsqueeze(-1) * h).sum(-2)
        write_h = w_t * v_t - erase_h
        h = h + k_t.unsqueeze(-1) * write_h.unsqueeze(-2)
        base = (q_t.unsqueeze(-1) * h).sum(-2)

        k_centered = k_t - mk
        v_centered = v_t - mv
        c = c * decay
        erase_c = ((b_t * k_centered).unsqueeze(-1) * c).sum(-2)
        write_c = w_t * v_centered - erase_c
        c = c + k_centered.unsqueeze(-1) * write_c.unsqueeze(-2)
        corr = (rho_t.unsqueeze(-1) * c).sum(-2)

        o[:, t] = base - gamma * corr

        mk = mean_decay * mk + update_rate * k_t.detach()
        mv = mean_decay * mv + update_rate * v_t.detach()

    final_state = (h, c, mk, mv) if output_final_state else None
    return o.to(out_dtype), final_state


def naive_recurrent_gdn2_parallax(
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
    use_qk_l2norm: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    r"""
    Token-by-token PyTorch reference for GDN2 with a Parallax-inspired correction.

    The base state follows GDN-2 exactly. An extra state ``C`` stores a centered
    key/value delta-rule memory and contributes ``gamma * C^T rho`` to the readout.

    Args:
        q: queries of shape ``[B, T, H, K]``.
        k: keys of shape ``[B, T, H, K]``.
        v: values of shape ``[B, T, H, V]``.
        g: log-decay of shape ``[B, T, H, K]``.
        b: channel-wise erase gate of shape ``[B, T, H, K]``.
        w: channel-wise write gate of shape ``[B, T, H, V]``.
        rho: Parallax probe of shape ``[B, T, H, K]``.
        gamma: per-head correction scale of shape ``[H]``.
        scale: attention scale. Default: ``1 / sqrt(K)``.
        initial_state: optional tuple ``(h, c, mk, mv)``.
        output_final_state: whether to return final recurrent states.
        cu_seqlens: optional packed-sequence offsets. Requires batch size 1.
        mean_decay: EMA decay for detached running key/value centers.
        use_qk_l2norm: L2-normalize q and k before the recurrence.

    Returns:
        ``(o, final_state)`` where ``o`` has shape ``[B, T, H, V]``.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
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

    if use_qk_l2norm:
        q = _l2norm(q.float()).to(q.dtype)
        k = _l2norm(k.float()).to(k.dtype)

    if cu_seqlens is None:
        return _run_segment(
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
            mean_decay=mean_decay,
        )

    if q.shape[0] != 1:
        raise ValueError("cu_seqlens requires batch size 1 with packed sequences.")
    boundaries = cu_seqlens.detach().cpu().tolist()
    num_sequences = len(boundaries) - 1
    if initial_state is not None and initial_state[0].shape[0] != num_sequences:
        raise ValueError(
            f"initial_state has {initial_state[0].shape[0]} sequences, "
            f"but cu_seqlens describes {num_sequences}."
        )

    outputs = []
    final_states = []
    for i, (bos, eos) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=False)):
        state_i = None
        if initial_state is not None:
            state_i = tuple(s[i:i + 1] for s in initial_state)
        out_i, final_i = _run_segment(
            q=q[:, bos:eos],
            k=k[:, bos:eos],
            v=v[:, bos:eos],
            g=g[:, bos:eos],
            b=b[:, bos:eos],
            w=w[:, bos:eos],
            rho=rho[:, bos:eos],
            gamma=gamma,
            scale=scale,
            initial_state=state_i,
            output_final_state=output_final_state,
            mean_decay=mean_decay,
        )
        outputs.append(out_i)
        if output_final_state:
            final_states.append(final_i)

    o = torch.cat(outputs, dim=1)
    if not output_final_state:
        return o, None

    h, c, mk, mv = (torch.cat([state[j] for state in final_states], dim=0) for j in range(4))
    return o, (h, c, mk, mv)
