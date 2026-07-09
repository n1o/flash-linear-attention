# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.layers import QGatedDeltaNet2
from fla.utils import device


@pytest.mark.parametrize("disable_recompute", [False, True])
@pytest.mark.skipif(getattr(device, "type", str(device)) != "cuda", reason="QGDN2 chunk kernel requires CUDA")
def test_layer_forward_backward(disable_recompute: bool):
    torch.manual_seed(42)
    layer = QGatedDeltaNet2(
        hidden_size=64,
        num_heads=2,
        num_v_heads=2,
        head_dim=32,
        use_short_conv=False,
        mode="chunk",
        disable_recompute=disable_recompute,
    ).to(device).to(torch.float32)
    x = torch.randn(2, 12, 64, device=device, dtype=torch.float32, requires_grad=True)

    y, _, _ = layer(x, use_cache=False)
    assert y.shape == x.shape
    loss = y.float().square().mean()
    loss.backward()

    assert layer.lamb_proj.weight.grad is not None
    assert torch.isfinite(layer.lamb_proj.weight.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
