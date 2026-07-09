"""Profiling harness for chunk_qgdn2 — mirrors run.py's EXACT call path:
  inputs from registry generate_inputs, chunk_qgdn2(**inputs, use_qk_l2norm_in_kernel=True)
  with output_final_state=False (default), then t.backward(do) on `o` only.
Runs fwd+backward under torch.profiler. Prints top kernels by GPU time.

Usage: python profile/qgdn2-opt/trace/prof.py <shape_name>
  shape_name: D64 (B4_T4096_H4_D64) or D32 (B8_T2048_H2_D32)
"""
import sys
import torch
from fla.ops.qgdn2 import chunk_qgdn2
from benchmarks.ops.registry import get_op, generate_inputs

SHAPES = {
    'D32': dict(B=8, T=2048, H=2, D=32),
    'D64': dict(B=4, T=4096, H=4, D=64),
}

def build(B, T, H, D, dtype=torch.bfloat16, device='cuda'):
    torch.manual_seed(42)
    cfg = get_op('chunk_qgdn2')
    inputs = generate_inputs(cfg, B=B, T=T, H=H, D=D, dtype=dtype, device=device)
    return inputs

def step(inputs):
    # exactly like run.py: output_final_state default False, backward on o only
    out = chunk_qgdn2(**inputs, use_qk_l2norm_in_kernel=True)
    o = out[0]
    do = torch.randn_like(o)
    o.backward(do)
    return o

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'D64'
    sh = SHAPES[name]
    B, T, H, D = sh['B'], sh['T'], sh['H'], sh['D']
    print(f"=== shape {name}: B={B} T={T} H={H} D={D} ===")

    inputs = build(B, T, H, D)
    for nm, t in inputs.items():
        print(f"  {nm}: shape={tuple(t.shape)} dtype={t.dtype} req_grad={t.requires_grad}")

    # warmup (compile + autotune)
    for _ in range(3):
        step(build(B, T, H, D))
    torch.cuda.synchronize()

    # sanity: finite outputs & grads on a fresh build
    inputs = build(B, T, H, D)
    out = chunk_qgdn2(**inputs, use_qk_l2norm_in_kernel=True)
    o = out[0]
    do = torch.randn_like(o)
    o.backward(do)
    torch.cuda.synchronize()
    for nm in ['q', 'k', 'v', 'g', 'b', 'w', 'lq']:
        t = inputs[nm]
        if t.grad is not None:
            assert torch.isfinite(t.grad).all(), f"NON-FINITE {nm}.grad"
    assert torch.isfinite(o).all(), "NON-FINITE o"
    print("sanity: all outputs/grads finite OK")

    # profile
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(5):
            step(build(B, T, H, D))
        torch.cuda.synchronize()

    print("\n--- top kernels by CUDA time (5 iters) ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

if __name__ == "__main__":
    main()
