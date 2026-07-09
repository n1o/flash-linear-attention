# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gdn2 import naive_recurrent_gdn2
from fla.ops.qgdn2 import chunk_qgdn2, naive_recurrent_qgdn2
from fla.utils import assert_close, device


def _rand_inputs(B: int, T: int, H: int, K: int, V: int):
    torch.manual_seed(42)
    scale = K ** -0.5
    q = (torch.randn(B, T, H, K, device=device, dtype=torch.float32) * scale).requires_grad_(True)
    k = (torch.randn(B, T, H, K, device=device, dtype=torch.float32) * scale).requires_grad_(True)
    v = (torch.randn(B, T, H, V, device=device, dtype=torch.float32) * scale).requires_grad_(True)
    g = (-torch.rand(B, T, H, K, device=device, dtype=torch.float32)).requires_grad_(True)
    b = torch.rand(B, T, H, K, device=device, dtype=torch.float32, requires_grad=True)
    w = torch.rand(B, T, H, V, device=device, dtype=torch.float32, requires_grad=True)
    lq = torch.rand(B, T, H, device=device, dtype=torch.float32, requires_grad=True)
    return q, k, v, g, b, w, lq


def _clone_with_grad(xs):
    return tuple(x.detach().clone().requires_grad_(True) for x in xs)


def test_lambda_zero_matches_naive_gdn2():
    B, T, H, K, V = 2, 9, 3, 16, 12
    q, k, v, g, b, w, _ = _rand_inputs(B, T, H, K, V)
    lq = torch.zeros(B, T, H, device=device, dtype=torch.float32)

    ref, _ = naive_recurrent_gdn2(q.detach(), k.detach(), v.detach(), g.detach(), b.detach(), w.detach())
    tri, _ = naive_recurrent_qgdn2(q.detach(), k.detach(), v.detach(), g.detach(), b.detach(), w.detach(), lq)
    assert_close("o", ref, tri, 1e-6)


@pytest.mark.skipif(getattr(device, "type", str(device)) != "cuda", reason="chunk kernel requires CUDA")
def test_chunk_matches_naive_forward_backward():
    B, T, H, K, V = 1, 64, 2, 32, 32
    inputs = tuple(x.detach() for x in _rand_inputs(B, T, H, K, V))

    q, k, v, g, b, w, lq = _clone_with_grad(inputs)
    ref, ref_state = naive_recurrent_qgdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        lq=lq,
        output_final_state=True,
    )
    do = torch.randn_like(ref)
    dstate = torch.randn_like(ref_state)
    (ref * do).sum().add((ref_state * dstate).sum()).backward()
    ref_grads = {
        "q": q.grad.clone(),
        "k": k.grad.clone(),
        "v": v.grad.clone(),
        "g": g.grad.clone(),
        "b": b.grad.clone(),
        "w": w.grad.clone(),
        "lq": lq.grad.clone(),
    }

    q, k, v, g, b, w, lq = _clone_with_grad(inputs)
    tri, tri_state = chunk_qgdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        lq=lq,
        output_final_state=True,
    )
    (tri * do).sum().add((tri_state * dstate).sum()).backward()
    tri_grads = {
        "q": q.grad.clone(),
        "k": k.grad.clone(),
        "v": v.grad.clone(),
        "g": g.grad.clone(),
        "b": b.grad.clone(),
        "w": w.grad.clone(),
        "lq": lq.grad.clone(),
    }

    assert_close("o", ref, tri, 0.006)
    assert_close("state", ref_state, tri_state, 0.006)
    for name, ref_grad in ref_grads.items():
        assert_close(name, ref_grad, tri_grads[name], 0.05)


@pytest.mark.skipif(getattr(device, "type", str(device)) != "cuda", reason="chunk kernel requires CUDA")
def test_chunk_matches_naive_with_l2norm():
    B, T, H, K, V = 1, 65, 2, 32, 32
    q, k, v, g, b, w, lq = (x.detach() for x in _rand_inputs(B, T, H, K, V))

    ref, _ = naive_recurrent_qgdn2(
        q=F.normalize(q.float(), dim=-1),
        k=F.normalize(k.float(), dim=-1),
        v=v,
        g=g,
        b=b,
        w=w,
        lq=lq,
    )
    tri, _ = chunk_qgdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        lq=lq,
        use_qk_l2norm_in_kernel=True,
    )
    assert_close("o", ref, tri, 0.006)
