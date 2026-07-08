# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
from transformers import AutoModelForCausalLM

from fla.layers import GatedDeltaNet2, GDN2Parallax
from fla.models import GDN2ParallaxConfig, GDN2ParallaxForCausalLM
from fla.utils import assert_close, device


def _make_config(**kwargs):
    return GDN2ParallaxConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=2,
        head_dim=32,
        expand_v=1.0,
        use_short_conv=False,
        vocab_size=128,
        hidden_ratio=2,
        fuse_cross_entropy=False,
        fuse_linear_cross_entropy=False,
        use_cache=True,
        **kwargs,
    )


def _make_model(**kwargs):
    torch.manual_seed(42)
    config = _make_config(**kwargs)
    model = AutoModelForCausalLM.from_config(config).to(device)
    return model, config


def test_auto_model_registration():
    model, _ = _make_model()
    assert isinstance(model, GDN2ParallaxForCausalLM)


def test_model_forward_backward():
    model, config = _make_model()
    model.train()
    input_ids = torch.randint(0, config.vocab_size, (2, 12), device=device)

    out = model(input_ids=input_ids, labels=input_ids)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    out.loss.backward()

    layer = model.model.layers[0].attn
    assert layer.parallax_gamma.grad is not None
    assert torch.isfinite(layer.parallax_gamma.grad).all()
    assert layer.rho_proj.weight.grad is not None
    assert torch.isfinite(layer.rho_proj.weight.grad).all()


def test_model_varlen_matches_dense():
    model, config = _make_model()
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 8), device=device)
    cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)

    with torch.no_grad():
        dense = model(input_ids=input_ids, output_hidden_states=True, use_cache=False).hidden_states[-1]
        varlen = model(
            input_ids=input_ids.reshape(1, -1),
            output_hidden_states=True,
            use_cache=False,
            cu_seqlens=cu_seqlens,
        ).hidden_states[-1]

    assert_close("hidden", dense.reshape(1, -1, config.hidden_size), varlen, 1e-5)


def test_model_cache_matches_full_sequence():
    model, config = _make_model()
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 12), device=device)

    with torch.no_grad():
        ref = model(input_ids=input_ids, use_cache=False).logits
        out = model(input_ids=input_ids[:, :4], use_cache=True)
        logits = [out.logits]
        past_key_values = out.past_key_values
        for t in range(4, input_ids.shape[1]):
            out = model(
                input_ids=input_ids[:, t:t + 1],
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits.append(out.logits)
            past_key_values = out.past_key_values
        cached = torch.cat(logits, dim=1)

    assert_close("logits", ref, cached, 1e-5)


def test_layer_eval_with_grad_uses_reference_backward():
    torch.manual_seed(42)
    layer = GDN2Parallax(
        hidden_size=32,
        expand_v=1.0,
        head_dim=8,
        num_heads=2,
        num_v_heads=2,
        mode="fused_recurrent",
        use_short_conv=False,
    ).to(device).to(torch.float32).eval()
    layer.parallax_gamma.data.normal_(0.0, 0.1)
    x = torch.randn(1, 4, 32, device=device, dtype=torch.float32, requires_grad=True)

    y, _, _ = layer(x, use_cache=False)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.parallax_gamma.grad is not None and torch.isfinite(layer.parallax_gamma.grad).all()
    assert layer.rho_proj.weight.grad is not None and torch.isfinite(layer.rho_proj.weight.grad).all()


@pytest.mark.skipif(getattr(device, "type", str(device)) != "cuda", reason="GatedDeltaNet2 baseline kernels require CUDA")
def test_layer_gamma_zero_matches_gdn2():
    torch.manual_seed(42)
    kwargs = dict(
        hidden_size=64,
        expand_v=1.0,
        head_dim=16,
        num_heads=2,
        num_v_heads=2,
        use_short_conv=False,
    )
    base = GatedDeltaNet2(**kwargs).to(device).to(torch.float32).eval()
    layer = GDN2Parallax(**kwargs).to(device).to(torch.float32).eval()
    layer.load_state_dict(
        {
            name: value
            for name, value in base.state_dict().items()
            if name in layer.state_dict() and layer.state_dict()[name].shape == value.shape
        },
        strict=False,
    )
    layer.parallax_gamma.data.zero_()
    x = torch.randn(2, 6, kwargs["hidden_size"], device=device, dtype=torch.float32)

    with torch.no_grad():
        ref, _, _ = base(x, use_cache=False)
        tri, _, _ = layer(x, use_cache=False)

    assert_close("o", ref, tri, 2e-3)
