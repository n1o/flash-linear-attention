# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch


def naive_recurrent_qgdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    lq: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Token-by-token QGDN2 reference.

    QGDN2 is GDN2 with a query-conditioned erase direction
    ``x_t = k_t + lambda_t * q_t``. The write direction remains ``k_t`` and
    the output read uses the scaled query.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q, k, v, g, b, w = (
        x.transpose(1, 2).contiguous().float()
        for x in (q, k, v, g, b, w)
    )
    lq = lq.transpose(1, 2).contiguous().float()
    B, H, T, K = k.shape
    V = v.shape[-1]
    o = torch.zeros(B, H, T, V, device=v.device, dtype=torch.float32)
    h = torch.zeros(B, H, K, V, device=v.device, dtype=torch.float32)
    if initial_state is not None:
        h = initial_state.to(torch.float32).clone()
    q_scaled = q * scale

    for t in range(T):
        b_q = q[:, :, t]
        b_k = k[:, :, t]
        b_v = v[:, :, t]
        b_g = g[:, :, t]
        b_b = b[:, :, t]
        b_w = w[:, :, t]
        b_lq = lq[:, :, t].unsqueeze(-1)

        x = b_k + b_lq * b_q
        h = h * b_g.exp().unsqueeze(-1)
        erase = ((b_b * x).unsqueeze(-1) * h).sum(-2)
        v_new = b_w * b_v - erase
        h = h + b_k.unsqueeze(-1) * v_new.unsqueeze(-2)
        o[:, :, t] = (q_scaled[:, :, t].unsqueeze(-1) * h).sum(-2)

    o = o.transpose(1, 2).contiguous().to(v.dtype)
    if not output_final_state:
        h = None
    return o, h


__all__ = ["naive_recurrent_qgdn2"]
