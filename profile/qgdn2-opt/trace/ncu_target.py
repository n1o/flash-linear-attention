"""Minimal NCU target: warmup (compile) then ONE fwd+bwd of chunk_qgdn2 on D64.
NCU captures kernel invocations. Use with -c 1 to grab one call per kernel."""
import torch
from fla.ops.qgdn2 import chunk_qgdn2
from benchmarks.ops.registry import get_op, generate_inputs

cfg = get_op('chunk_qgdn2')
B, T, H, D = 4, 4096, 4, 64

def build():
    torch.manual_seed(42)
    return generate_inputs(cfg, B=B, T=T, H=H, D=D, dtype=torch.bfloat16, device='cuda')

# warmup (compile + autotune)
for _ in range(3):
    inp = build()
    out = chunk_qgdn2(**inp, use_qk_l2norm_in_kernel=True)
    out[0].backward(torch.randn_like(out[0]))
torch.cuda.synchronize()

# single profiled iteration
inp = build()
out = chunk_qgdn2(**inp, use_qk_l2norm_in_kernel=True)
out[0].backward(torch.randn_like(out[0]))
torch.cuda.synchronize()
