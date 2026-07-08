# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, get_unpad_data, index_first_axis, pad_input, update_layer_cache
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.gdn2_parallax import chunk_gdn2_parallax, fused_recurrent_gdn2_parallax, naive_recurrent_gdn2_parallax

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class GDN2Parallax(nn.Module):
    r"""
    GDN-2 with a Parallax-inspired recurrent covariance readout.

    The base GDN-2 state is unchanged. A second state stores a centered
    key/value delta-rule memory and contributes a bounded, zero-initialized
    correction to the output read:

        o = H^T q / sqrt(K) - gamma * C^T rho

    Training uses a chunked GDN2-based composite path.  Evaluation uses a
    forward-only Triton recurrent kernel when available.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dimension. Default: 1.0.
        head_dim (int, Optional):
            The dimension of each QK head. Default: 128.
        num_heads (int, Optional):
            The number of QK heads. Default: 16.
        num_v_heads (int, Optional):
            The number of value heads, equal to `num_heads` if `None`. Default: `None`.
        mode (str, Optional):
            One of `chunk`, `recurrent`, `fused_recurrent`, or `reference`. Default: `chunk`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        allow_neg_eigval (bool, Optional):
            Whether to multiply the erase gate by 2. Default: `False`.
        conv_size (int, Optional):
            The short convolution kernel size. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in short convolutions. Default: `False`.
        layer_idx (int, Optional):
            The cache layer index. Default: `None`.
        norm_eps (float, Optional):
            The epsilon for normalization layers. Default: 1e-5.
        parallax_mean_decay (float, Optional):
            EMA decay for detached running centers. Default: 0.99.
        parallax_rho_max (float, Optional):
            Bound for `rho_max * tanh(rho_proj(...))`. Default: 1.0.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1.0,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
        mode: str = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        parallax_mean_decay: float = 0.99,
        parallax_rho_max: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()

        if mode not in ("chunk", "recurrent", "fused_recurrent", "reference"):
            raise ValueError(
                f"GDN2Parallax only supports `chunk`, `recurrent`, `fused_recurrent`, or `reference` mode, got `{mode}`."
            )
        if not 0.0 <= parallax_mean_decay < 1.0:
            raise ValueError(f"parallax_mean_decay must be in [0, 1), got {parallax_mean_decay}.")

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.parallax_mean_decay = parallax_mean_decay
        self.parallax_rho_max = parallax_rho_max

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.rho_dim = int(self.num_v_heads * self.head_k_dim)
        self.layer_idx = layer_idx

        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by "
                f"num_v_heads*head_dim={self.num_v_heads * self.head_dim}."
            )
        if self.num_v_heads < self.num_heads:
            raise ValueError(f"num_v_heads={self.num_v_heads} must be >= num_heads={self.num_heads}.")
        if self.num_v_heads % self.num_heads != 0:
            raise ValueError(f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.")
        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}."
            )

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )

        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        self.b_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.w_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.rho_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.rho_proj = nn.Linear(hidden_size, self.rho_dim, bias=False)
        self.parallax_gamma = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.float32))

        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        dt = torch.exp(
            torch.rand(self.key_dim, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = FusedRMSNormGated(self.head_v_dim, activation="sigmoid", eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a [batch_size, seq_len] 0/1 padding mask."
            )

        batch_size, q_len, _ = hidden_states.shape
        last_state = get_layer_cache(self, past_key_values)

        cu_seqlens = kwargs.get("cu_seqlens")
        if cu_seqlens is None and attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        g = F.softplus(self.f_proj(hidden_states).float() + self.dt_bias)
        b = self.b_proj(hidden_states).sigmoid()
        w = self.w_proj(hidden_states).sigmoid()
        rho = self.parallax_rho_max * torch.tanh(self.rho_proj(self.rho_norm(hidden_states)))

        q, k, g = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k, g))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        b = rearrange(b, "... (h d) -> ... h d", d=self.head_k_dim)
        w = rearrange(w, "... (h d) -> ... h d", d=self.head_v_dim)
        rho = rearrange(rho, "... (h d) -> ... h d", d=self.head_k_dim)
        g = -self.A_log.float().exp().unsqueeze(-1) * g

        if self.num_v_heads > self.num_heads:
            q, k, g, b = (
                repeat(x, "... h d -> ... (h g) d", g=self.num_v_heads // self.num_heads)
                for x in (q, k, g, b)
            )

        if self.allow_neg_eigval:
            b = b * 2.0

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        grad_enabled = torch.is_grad_enabled() and any(
            x.requires_grad for x in (q, k, v, g, b, w, rho, self.parallax_gamma)
        )
        mode = "fused_recurrent" if (q_len <= 64 and not self.training and not grad_enabled) else self.mode
        use_chunk = (
            mode == "chunk"
            and q.is_cuda
            and q.shape[-1] <= 256
        )
        use_fused_recurrent = (
            mode in ("chunk", "fused_recurrent")
            and not self.training
            and not grad_enabled
            and q.is_cuda
            and q.shape[-1] <= 256
        )
        if use_fused_recurrent:
            recurrent_fn = fused_recurrent_gdn2_parallax
        elif use_chunk:
            recurrent_fn = chunk_gdn2_parallax
        else:
            recurrent_fn = naive_recurrent_gdn2_parallax
        recurrent_kwargs = {
            "q": q,
            "k": k,
            "v": v,
            "g": g,
            "b": b,
            "w": w,
            "rho": rho,
            "gamma": self.parallax_gamma,
            "initial_state": recurrent_state,
            "output_final_state": use_cache,
            "cu_seqlens": cu_seqlens,
            "mean_decay": self.parallax_mean_decay,
        }
        if use_fused_recurrent:
            recurrent_kwargs["use_qk_l2norm_in_kernel"] = True
        elif use_chunk:
            recurrent_kwargs["use_qk_l2norm_in_kernel"] = True
            recurrent_kwargs["disable_recompute"] = self.training
        else:
            recurrent_kwargs["use_qk_l2norm"] = True
        o, recurrent_state = recurrent_fn(**recurrent_kwargs)

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            offset=q_len,
        )

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
