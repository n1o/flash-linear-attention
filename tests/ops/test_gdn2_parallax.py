# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.gdn2 import naive_recurrent_gdn2
from fla.ops.gdn2_parallax import chunk_gdn2_parallax, fused_recurrent_gdn2_parallax, naive_recurrent_gdn2_parallax
from fla.utils import assert_close, device


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + 1e-6)


def _rand_inputs(B: int, T: int, H: int, K: int, V: int):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, dtype=torch.float32, requires_grad=True)
    g = (-torch.rand(B, T, H, K, device=device, dtype=torch.float32)).requires_grad_(True)
    b = torch.rand(B, T, H, K, device=device, dtype=torch.float32, requires_grad=True)
    w = torch.rand(B, T, H, V, device=device, dtype=torch.float32, requires_grad=True)
    rho = torch.randn(B, T, H, K, device=device, dtype=torch.float32, requires_grad=True)
    return q, k, v, g, b, w, rho


def _clone_with_grad(xs):
    return tuple(x.detach().clone().requires_grad_(True) for x in xs)


def _clone_state_with_grad(state):
    return tuple(x.detach().clone().requires_grad_(True) for x in state)


def test_gamma_zero_matches_gdn2_forward_backward():
    B, T, H, K, V = 2, 7, 3, 16, 12
    inputs = _rand_inputs(B, T, H, K, V)
    q, k, v, g, b, w, rho = _clone_with_grad(inputs)
    q_ref, k_ref, v_ref, g_ref, b_ref, w_ref = _clone_with_grad(inputs[:6])
    gamma = torch.zeros(H, device=device, dtype=torch.float32, requires_grad=True)

    ref, ref_h = naive_recurrent_gdn2(
        q=q_ref,
        k=k_ref,
        v=v_ref,
        g=g_ref,
        b=b_ref,
        w=w_ref,
        output_final_state=True,
    )
    tri, (tri_h, tri_c, tri_mk, tri_mv) = naive_recurrent_gdn2_parallax(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        rho=rho,
        gamma=gamma,
        output_final_state=True,
    )

    assert_close("o", ref, tri, 1e-6)
    assert_close("h", ref_h, tri_h, 1e-6)
    assert torch.isfinite(tri_c).all()
    assert torch.isfinite(tri_mk).all()
    assert torch.isfinite(tri_mv).all()

    do = torch.randn_like(ref)
    ref.backward(do)
    tri.backward(do)

    for name, ref_grad, tri_grad in [
        ("dq", q_ref.grad, q.grad),
        ("dk", k_ref.grad, k.grad),
        ("dv", v_ref.grad, v.grad),
        ("dg", g_ref.grad, g.grad),
        ("db", b_ref.grad, b.grad),
        ("dw", w_ref.grad, w.grad),
    ]:
        assert_close(name, ref_grad, tri_grad, 1e-6)
    assert rho.grad is not None and torch.isfinite(rho.grad).all()
    assert gamma.grad is not None and torch.isfinite(gamma.grad).all()


def test_qk_l2norm_gamma_zero_matches_normalized_gdn2():
    B, T, H, K, V = 1, 5, 2, 8, 6
    q, k, v, g, b, w, rho = _rand_inputs(B, T, H, K, V)
    gamma = torch.zeros(H, device=device, dtype=torch.float32)

    ref, _ = naive_recurrent_gdn2(
        q=_l2norm(q.detach()),
        k=_l2norm(k.detach()),
        v=v.detach(),
        g=g.detach(),
        b=b.detach(),
        w=w.detach(),
    )
    tri, _ = naive_recurrent_gdn2_parallax(
        q=q.detach(),
        k=k.detach(),
        v=v.detach(),
        g=g.detach(),
        b=b.detach(),
        w=w.detach(),
        rho=rho.detach(),
        gamma=gamma,
        use_qk_l2norm=True,
    )
    assert_close("o", ref, tri, 1e-6)


def test_varlen_matches_segmented_dense_with_initial_state():
    B, T, H, K, V = 1, 9, 4, 8, 5
    q, k, v, g, b, w, rho = _rand_inputs(B, T, H, K, V)
    gamma = torch.randn(H, device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 2, 5, 9], device=device, dtype=torch.int32)
    initial_state = (
        torch.randn(3, H, K, V, device=device, dtype=torch.float32),
        torch.randn(3, H, K, V, device=device, dtype=torch.float32),
        torch.randn(3, H, K, device=device, dtype=torch.float32),
        torch.randn(3, H, V, device=device, dtype=torch.float32),
    )

    var, var_state = naive_recurrent_gdn2_parallax(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        rho=rho,
        gamma=gamma,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        mean_decay=0.7,
    )

    dense_outputs = []
    dense_states = []
    boundaries = cu_seqlens.cpu().tolist()
    for i, (bos, eos) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=False)):
        out_i, state_i = naive_recurrent_gdn2_parallax(
            q=q[:, bos:eos],
            k=k[:, bos:eos],
            v=v[:, bos:eos],
            g=g[:, bos:eos],
            b=b[:, bos:eos],
            w=w[:, bos:eos],
            rho=rho[:, bos:eos],
            gamma=gamma,
            initial_state=tuple(s[i:i + 1] for s in initial_state),
            output_final_state=True,
            mean_decay=0.7,
        )
        dense_outputs.append(out_i)
        dense_states.append(state_i)

    ref = torch.cat(dense_outputs, dim=1)
    assert_close("var", ref, var, 1e-6)
    for i, (ref_part, var_part) in enumerate(
        zip((torch.cat([s[j] for s in dense_states], dim=0) for j in range(4)), var_state, strict=False)
    ):
        assert_close(f"state{i}", ref_part, var_part, 1e-6)


def test_zero_covariance_state_removes_correction():
    B, T, H, K, V = 2, 6, 2, 8, 7
    q, k, _, g, b, w, rho = _rand_inputs(B, T, H, K, V)
    v = torch.zeros(B, T, H, V, device=device, dtype=torch.float32)
    gamma = torch.randn(H, device=device, dtype=torch.float32)

    corrected, state = naive_recurrent_gdn2_parallax(
        q=q.detach(),
        k=k.detach(),
        v=v,
        g=g.detach(),
        b=b.detach(),
        w=w.detach(),
        rho=rho.detach(),
        gamma=gamma,
        output_final_state=True,
    )
    baseline, _ = naive_recurrent_gdn2_parallax(
        q=q.detach(),
        k=k.detach(),
        v=v,
        g=g.detach(),
        b=b.detach(),
        w=w.detach(),
        rho=rho.detach(),
        gamma=torch.zeros_like(gamma),
    )

    assert_close("o", baseline, corrected, 1e-6)
    assert_close("c", torch.zeros_like(state[1]), state[1], 1e-6)


@pytest.mark.skipif(getattr(device, "type", str(device)) != "cuda", reason="chunk kernel requires CUDA")
def test_chunk_matches_naive_forward_backward_with_initial_state():
    B, T, H, K, V = 1, 64, 2, 32, 32
    inputs = tuple(x.detach() for x in _rand_inputs(B, T, H, K, V))
    gamma_init = torch.randn(H, device=device, dtype=torch.float32)
    state_init = (
        torch.randn(B, H, K, V, device=device, dtype=torch.float32),
        torch.randn(B, H, K, V, device=device, dtype=torch.float32),
        torch.randn(B, H, K, device=device, dtype=torch.float32),
        torch.randn(B, H, V, device=device, dtype=torch.float32),
    )

    q, k, v, g, b, w, rho = _clone_with_grad(inputs)
    gamma = gamma_init.detach().clone().requires_grad_(True)
    initial_state = _clone_state_with_grad(state_init)
    ref, ref_state = naive_recurrent_gdn2_parallax(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        rho=rho,
        gamma=gamma,
        initial_state=initial_state,
        output_final_state=True,
        mean_decay=0.81,
        use_qk_l2norm=True,
    )
    do = torch.randn_like(ref)
    dstate = tuple(torch.randn_like(s) for s in ref_state)
    (ref * do).sum().add(sum((s * ds).sum() for s, ds in zip(ref_state, dstate, strict=False))).backward()
    ref_grads = {
        "q": q.grad.clone(),
        "k": k.grad.clone(),
        "v": v.grad.clone(),
        "g": g.grad.clone(),
        "b": b.grad.clone(),
        "w": w.grad.clone(),
        "rho": rho.grad.clone(),
        "gamma": gamma.grad.clone(),
        "h0": initial_state[0].grad.clone(),
        "c0": initial_state[1].grad.clone(),
        "mk0": initial_state[2].grad.clone(),
        "mv0": initial_state[3].grad.clone(),
    }

    q, k, v, g, b, w, rho = _clone_with_grad(inputs)
    gamma = gamma_init.detach().clone().requires_grad_(True)
    initial_state = _clone_state_with_grad(state_init)
    tri, tri_state = chunk_gdn2_parallax(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        rho=rho,
        gamma=gamma,
        initial_state=initial_state,
        output_final_state=True,
        mean_decay=0.81,
        use_qk_l2norm_in_kernel=True,
    )
    (tri * do).sum().add(sum((s * ds).sum() for s, ds in zip(tri_state, dstate, strict=False))).backward()
    tri_grads = {
        "q": q.grad.clone(),
        "k": k.grad.clone(),
        "v": v.grad.clone(),
        "g": g.grad.clone(),
        "b": b.grad.clone(),
        "w": w.grad.clone(),
        "rho": rho.grad.clone(),
        "gamma": gamma.grad.clone(),
        "h0": initial_state[0].grad.clone(),
        "c0": initial_state[1].grad.clone(),
        "mk0": initial_state[2].grad.clone(),
        "mv0": initial_state[3].grad.clone(),
    }

    assert_close("o", ref, tri, 0.006)
    for i, (ref_part, tri_part) in enumerate(zip(ref_state, tri_state, strict=False)):
        assert_close(f"state{i}", ref_part, tri_part, 0.006)
    for name, ref_grad in ref_grads.items():
        assert_close(name, ref_grad, tri_grads[name], 0.03)


@pytest.mark.skipif(getattr(device, "type", str(device)) != "cuda", reason="chunk kernel requires CUDA")
def test_chunk_matches_naive_varlen_forward_with_initial_state():
    B, T, H, K, V = 1, 96, 2, 32, 32
    q, k, v, g, b, w, rho = (x.detach() for x in _rand_inputs(B, T, H, K, V))
    gamma = torch.randn(H, device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 10, 64, 96], device=device, dtype=torch.int32)
    cu_seqlens_cpu = cu_seqlens.cpu()
    initial_state = (
        torch.randn(3, H, K, V, device=device, dtype=torch.float32),
        torch.randn(3, H, K, V, device=device, dtype=torch.float32),
        torch.randn(3, H, K, device=device, dtype=torch.float32),
        torch.randn(3, H, V, device=device, dtype=torch.float32),
    )

    ref, ref_state = naive_recurrent_gdn2_parallax(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        rho=rho,
        gamma=gamma,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        mean_decay=0.83,
        use_qk_l2norm=True,
    )
    tri, tri_state = chunk_gdn2_parallax(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        rho=rho,
        gamma=gamma,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        mean_decay=0.83,
        use_qk_l2norm_in_kernel=True,
    )

    assert_close("o", ref, tri, 0.006)
    for i, (ref_part, tri_part) in enumerate(zip(ref_state, tri_state, strict=False)):
        assert_close(f"state{i}", ref_part, tri_part, 0.006)


@pytest.mark.skipif(getattr(device, "type", str(device)) != "cuda", reason="fused recurrent kernel requires CUDA")
def test_fused_recurrent_matches_naive_dense_with_initial_state():
    B, T, H, K, V = 2, 9, 2, 32, 20
    q, k, v, g, b, w, rho = (x.detach() for x in _rand_inputs(B, T, H, K, V))
    gamma = torch.randn(H, device=device, dtype=torch.float32)
    initial_state = (
        torch.randn(B, H, K, V, device=device, dtype=torch.float32),
        torch.randn(B, H, K, V, device=device, dtype=torch.float32),
        torch.randn(B, H, K, device=device, dtype=torch.float32),
        torch.randn(B, H, V, device=device, dtype=torch.float32),
    )

    ref, ref_state = naive_recurrent_gdn2_parallax(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        rho=rho,
        gamma=gamma,
        initial_state=initial_state,
        output_final_state=True,
        mean_decay=0.73,
        use_qk_l2norm=True,
    )
    tri, tri_state = fused_recurrent_gdn2_parallax(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        rho=rho,
        gamma=gamma,
        initial_state=initial_state,
        output_final_state=True,
        mean_decay=0.73,
        use_qk_l2norm_in_kernel=True,
    )

    assert_close("o", ref, tri, 2e-5)
    for i, (ref_part, tri_part) in enumerate(zip(ref_state, tri_state, strict=False)):
        assert_close(f"state{i}", ref_part, tri_part, 2e-5)


@pytest.mark.skipif(getattr(device, "type", str(device)) != "cuda", reason="fused recurrent kernel requires CUDA")
def test_fused_recurrent_matches_naive_varlen():
    B, T, H, K, V = 1, 10, 3, 16, 13
    q, k, v, g, b, w, rho = (x.detach() for x in _rand_inputs(B, T, H, K, V))
    gamma = torch.randn(H, device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 3, 4, 10], device=device, dtype=torch.int32)
    initial_state = (
        torch.randn(3, H, K, V, device=device, dtype=torch.float32),
        torch.randn(3, H, K, V, device=device, dtype=torch.float32),
        torch.randn(3, H, K, device=device, dtype=torch.float32),
        torch.randn(3, H, V, device=device, dtype=torch.float32),
    )

    ref, ref_state = naive_recurrent_gdn2_parallax(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        rho=rho,
        gamma=gamma,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        mean_decay=0.8,
    )
    tri, tri_state = fused_recurrent_gdn2_parallax(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        rho=rho,
        gamma=gamma,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        mean_decay=0.8,
    )

    assert_close("o", ref, tri, 2e-5)
    for i, (ref_part, tri_part) in enumerate(zip(ref_state, tri_state, strict=False)):
        assert_close(f"state{i}", ref_part, tri_part, 2e-5)
