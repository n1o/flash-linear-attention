# Optimization log — chunk_qgdn2

Env: Machine: NVIDIA B300 SXM6 AC | CUDA 13.0 | PyTorch 2.14.0.dev20260707+cu130 | Triton 3.8.0
       (no git in this checkout — OPT_LOG row is the commit record; diffs kept minimal/reversible)
Baseline: chunk_qgdn2 (recompute, default) — see iter 0 numbers below
Target: faster fwdbwd on B300, full gate green (fwd-only is launch-bound, secondary)

Clocks: UNLOCKED (idle 120MHz / max 2032MHz, no sudo to lock). Per TRAPS.md clock-drift:
iteration signal ranks by solution's own runtime in one run; verdicts re-measure
baseline+candidate back-to-back in one session. Never sum deltas.

Gate: tests/ops/test_qgdn2.py — 3 tests, small shapes only (T=64, T=65, K=V=32).
NOTE: gate does NOT cover large benchmark shapes (T=2048/4096). After kernel changes,
run a manual finite-output sanity check on a benchmark shape (see TRAPS).

## Baseline run (run.py, 3 ops, both modes, back-to-back in one session)

fwd (ms)  — launch-overhead bound, ~0.5ms, all ops ~identical; NOT a target
  B4_T4096_H4_D64: gdn2 0.517 | qgdn2 0.520 | qgdn2_save 0.520
  B8_T2048_H2_D32: gdn2 0.522 | qgdn2 0.501 | qgdn2_save 0.494

fwdbwd (ms)  — THE target, ~6ms
  B4_T4096_H4_D64: gdn2 6.056 | qgdn2 6.078 | qgdn2_save 5.414
  B8_T2048_H2_D32: gdn2 6.008 | qgdn2 5.984 | qgdn2_save 5.300

Readings:
- qgdn2 ~= gdn2  -> the lq erase branch `x = k + lq*q` adds ~no runtime cost; not a hot spot on its own.
- qgdn2_save (disable_recompute=True) is ~11% faster fwdbwd on BOTH shapes.
  => recomputing fwd intermediates in bwd costs ~0.6ms. Save trades memory (B300 has 275GB) for speed.
  => candidate #1 (plan) is a measured win with ZERO kernel change (just a kwarg / dispatch decision).

## Summary

| Iter | Direction (one line)                                | Gate | Bench (median ms)        | vs best | Status |
| ---- | --------------------------------------------------- | ---- | ------------------------ | ------- | ------ |
| 0    | baseline (chunk_qgdn2, recompute)                   | pass | D64 6.078 / D32 5.984    | —       | base   |
| 0b   | measure disable_recompute (chunk_qgdn2_save)        | pass | D64 5.414 / D32 5.300    | ~+11%   | note   |
| 1    | WY bwd kernel autotune: 1->8 configs +K key         | pass | D64 6.064 / D32 5.914    | ~0 (worse) | revert |
| 2    | recompute_w_u_fwd kernel autotune: 1->5 w/s configs | pass | D64 6.116 / D32 5.930    | ~0 (worse) | revert |
| 3    | bwd_intra kernel autotune: 1->5 w/s configs         | FAIL | —                        | —       | drop   |
| 4    | WY bwd: remove tl.debug_barrier() (sync cost)      | pass | D64 5.991 / D32 5.871    | ~0      | revert |

Status values: keep / revert / drop / floor / note
"note" = a measured alternative, not a kernel change (dispatch/usage decision; not yet promoted).

## Profiling pass (D64, 5 iters fwd+bwd, output_final_state=False, registry inputs)

torch.profiler, top kernels by CUDA time (kernel compute only, no gaps):
  recompute_w_u_fwd_gdn2_kernel          2.567ms  35%   (513us/iter)  [SKIPPED by disable_recompute]
  chunk_gdn2_bwd_kernel_intra            1.174ms  16%   (235us/iter)
  chunk_gated_delta_rule_fwd_kernel_h    0.812ms  11%   (162us/iter)  [in recompute path; SKIPPED by disable_recompute]
  chunk_gdn2_bwd_kernel_wy_dqkg_fused    0.435ms   6%   ( 87us/iter)  [QGDN2-specific; SINGLE autotune config]
  chunk_gated_delta_rule_bwd_kernel_dhu  0.406ms 5.6%   ( 81us/iter)  [shared with KDA]
  chunk_gdn2_fwd_kernel_inter_solve_fused 0.384ms 5.3% ( 77us/iter)
  chunk_gdn2_fwd_kernel_intra_token_parallel 0.299ms 4.1% (60us/iter)
  chunk_gla_fwd_kernel_o                0.085ms  1.2%  ( 17us/iter)
  (cumsum/elementwise/l2norm/bwd_dAv: small)

Kernel compute total ~1.45ms/iter; do_bench wall ~6.0ms/iter; direct event ~3.5ms/iter.
=> WALL-CLOCK IS LAUNCH-OVERHEAD-BOUND (many small kernels, gaps between them).
   ~26 kernel launches per fwdbwd iter. Strategy: reducing kernel COUNT (fusion / skip
   recompute) beats single-kernel autotune tuning for wall-clock. Autotune tuning is
   low-risk but low wall-clock yield; fusion is higher-risk, higher-yield.

## Memory check (peak torch.cuda.max_memory_allocated)

  D32 recompute:  99.2 MB | D32 save:  99.2 MB   (identical)
  D64 recompute: 370.3 MB | D64 save: 370.3 MB  (identical)
=> disable_recompute=True costs ZERO extra peak memory on target shapes.
   It is a STRICT WIN: ~11% faster fwdbwd, same peak memory. (On larger shapes the
   retained intermediates may raise peak; not checked — out of plan scope.)

## Decision: candidate #1 (disable_recompute) is a strict win on plan shapes.
   B300 has 275GB HBM; memory is not the limiter. Recommendation: use
   chunk_qgdn2_save (disable_recompute=True) for training. This is a usage/dispatch
   decision, NOT a kernel change — the registry already exposes it as chunk_qgdn2_save.
   Kernel optimization continues on top of BOTH paths (recompute is still the default op).

## Candidate #2 — expand Triton autotune configs

### Iter 1 — WY backward kernel `chunk_gdn2_bwd_kernel_wy_dqkg_fused` (chunk_bwd.py)
Delta: expanded single config {BK:32,BV:32,w4,s2} to 8 configs
      (BK,BV in {32,64}x{32,64}, w4/8, s2/3) + added 'K' to autotune key so
      D32 (K=32) and D64 (K=64) tune independently.
Gate: pass (3/3) — but gate time 1.6s -> 12.5s (8-config autotune overhead).
Large-shape finite sanity: D32 OK, D64 OK.
Bench (run.py, all 3 ops, one session) — use chunk_gdn2 as within-run control (clock drift):
  gdn2 (control) drifted -2.5% (D64) / -3.3% (D32) vs baseline run -> absolute invalid.
  ratio qgdn2/gdn2:      D64 1.004 -> 1.027 | D32 0.996 -> 1.018  (slightly worse)
  ratio qgdn2_save/gdn2: D64 0.894 -> 0.905 | D32 0.882 -> 0.903  (slightly worse)
Verdict: NO measurable wall-clock win (within clock-drift noise; slight regression).
   Matches launch-bound prediction: WY backward is only 6% of kernel time (87us/iter);
   tuning it cannot move the 3.5-6ms wall-clock. Autotune optimizes kernel latency, not
   end-to-end, and adds gate overhead. Status: revert (restored single config).

### Iter 2 — recompute kernel `recompute_w_u_fwd_gdn2_kernel` (wy_fast.py, 35% of kernel time)
Delta: expanded {w2,s2} to 5 configs (w2/4/8 x s2/3). BK/BV hardcoded 64 by caller
      (not autotunable without restructuring). Key already includes K,V,BK,BV -> per-shape tune.
Gate: pass (3/3, 5.56s). Large-shape finite: OK.
Bench (gdn2 control, drift -2.4/-2.9%): ratio qgdn2/gdn2 D64 1.004->1.035, D32 0.996->1.017
      (worse). qgdn2_save (skips this kernel) unchanged within noise.
Verdict: NO wall-clock win. This kernel is the biggest (513us/iter) but skipped by save, and
   wall-clock is launch-bound. Status: revert.

### Iter 3 — bwd_intra kernel `chunk_gdn2_bwd_kernel_intra` (chunk_intra.py, 16%)
Delta: expanded {w4,s3} to 5 configs (w4/8/2 x s2/3).
Gate: FAIL — test_chunk_matches_naive_forward_backward exceeded grad tolerance (0.05).
      Large-shape finite check PASSED (no NaN) -> numerical drift, not a crash.
      Cause: different num_warps/num_stages reassociate the dot/reduction -> rounding changes
      exceed the frozen tolerance. Autotune optimizes speed, not accuracy, so a faster config
      can break the gate.
Verdict: DROP (gate is the frozen contract; correctness first, no bench on red). Status: revert.

### Candidate #2 summary — NO-GO
Three representative kernels tried (the 35%, 16%, and 6% kernels):
  - iter1, iter2: gate green but NO measurable wall-clock win (launch-bound).
  - iter3: gate FAIL (numerical sensitivity to warps/stages).
Conclusion: expanding autotune is low-yield (launch-bound) AND can break the gate via
  numerical drift. The single-config kernels are single-config for a reason. Candidate #2
  is a no-go in this regime. Move to candidate #3 (erase branch) and #4 (backward fusion).

## Candidate #3 — inspect QGDN2 erase branch `x = k + lq * q`

Inspected all 3 kernels that compute x (no code change):
  - fwd intra (chunk_intra_token_parallel.py:110-125): q,lq loaded once/token, x inline.
  - recompute (wy_fast.py:180-189): q loaded once/i_k (reused for x + qg), lq once/i_k, x inline.
  - WY bwd (chunk_bwd.py:221-226): q loaded once/i_k (reused for x, dlq, dg), lq once/i_k, x inline.
Evidence: NO redundant q/lq loads in any kernel; x is one FMA. The WY bwd kernel (where x
  lives in backward) is 6% of kernel time (87us/iter). Regime is launch-bound -> materializing
  x would add HBM traffic + launches = strictly worse (the plan's own warning).
Verdict: KEEP INLINE. No code change (evidence-based no-op).

## Candidate #4 — backward kernel focus

Sub-items from the plan vs evidence:
  - repeated q/k/lq/g/b/w_gate loads: NONE found (see candidate #3 inspection).
  - dot-product precision: already bf16-in / fp32-accum; no safe lever (changing risks gate).
  - BK/BV tile balance: tested in iter1 (no wall-clock win; gate ok for WY kernel).
  - register pressure / occupancy: NCU UNAVAILABLE (ERR_NVGPUCTRPERM, no sudo to enable
    perf counters). torch.profiler kernel-time breakdown used instead.
  - extra synchronization / barrier cost: tested in iter4 (debug_barrier removal).

### Iter 4 — remove `tl.debug_barrier()` in WY bwd kernel (chunk_bwd.py:193)
Delta: removed the hazard barrier between the b_dq/b_dk/b_dw_flow dots and the i_k==0 block.
Gate: pass 3x deterministically (no race surfaced). Large-shape finite: OK.
Bench (gdn2 control, drift -2.3/-3.4%): ratio qgdn2/gdn2 D64 1.004->1.013, D32 0.996->1.012
      (no win, within noise).
Verdict: NO wall-clock win. Removing a hazard safety barrier for zero benefit is strictly
  worse (loses a correctness safety margin; a race could surface on other HW/driver).
  Status: revert.

### Candidate #4 summary — NO-GO for kernel-internal micro-optimization
Three kernel-internal attempts (iter1 tiling, iter2 recompute autotune, iter4 barrier)
all show NO wall-clock win in the launch-bound regime (kernel compute 1.45ms/iter vs
wall 3.5-6ms; ~26 launches/iter, gaps dominate). The backward's real lever is LAUNCH-COUNT
reduction, which candidate #1 (disable_recompute) already captures (removes the recompute
kernel + fwd_h = 2 launches + 46% of kernel compute, for ~11% wall-clock). Further
kernel-internal changes are low-yield and high-risk (the gate is numerically sensitive to
config changes per iter3). NCU unavailable for occupancy/SOL, but torch.profiler evidence
is sufficient to establish the launch-bound regime.

## Tooling note
NCU (Nsight Compute) is installed but UNUSABLE on this setup: ERR_NVGPUCTRPERM — the user
lacks permission to GPU performance counters, and there is no sudo to enable them.
The plan's DoD asks for an NCU summary of the winning kernel; the only winning change is
candidate #1 (disable_recompute), which is a kwarg/dispatch change, NOT a kernel change,
so a per-kernel NCU summary is not the right evidence for it. The benchmark numbers
(run.py, back-to-back, gdn2-controlled) are the evidence of record.

## Candidate #5 — shape dispatch (only after evidence)

Plan: add dispatch only when benchmarks show different kernels win different buckets.
Evidence across all runs, both shapes (ratio vs gdn2 control):
  D64: qgdn2_save 0.894-0.908 | qgdn2 1.004-1.035
  D32: qgdn2_save 0.882-0.905 | qgdn2 0.996-1.018
Ranking is CONSISTENT on both shapes: chunk_qgdn2_save < chunk_gdn2 ~ chunk_qgdn2.
No kernel change won on either shape; no shape bucket needs a different path.
The only divergence is recompute-vs-SAVE (a MODE choice, candidate #1), not a shape choice.
Verdict: NO shape dispatch. dispatch.md NOT created (Phase 3 specialization not justified;
  the plan requires a per-bucket win to earn a row, and no bucket diverges).

## FINAL SUMMARY

Loop result: candidates #2/#3/#4/#5 are NO-GO with evidence; candidate #1 is the win.

  #1 disable_recompute (chunk_qgdn2_save): ~11% faster fwdbwd, identical peak memory.
       Strict win on plan shapes. Recommendation: use for training. (kwarg, not a kernel change.)
  #2 expand autotune: NO-GO. Launch-bound (no wall-clock win on 35%/16% kernels) AND
       breaks the gate via numerical drift on bwd_intra (iter3).
  #3 erase branch x=k+lq*q: NO-GO (keep inline). No redundant loads; precomputing adds
       traffic+launches in a launch-bound regime.
  #4 backward kernel focus: NO-GO for kernel-internal micro-opt. 3 attempts (tiling,
       recompute autotune, barrier removal) -> no wall-clock win; gate is numerically
       sensitive. The real backward lever is launch-count reduction = candidate #1.
  #5 shape dispatch: NO-GO. No shape divergence; both shapes favor the same path.

Regime finding (the key discovery): chunk_qgdn2 fwdbwd is LAUNCH-OVERHEAD-BOUND, not
  compute-bound. Kernel compute ~1.45ms/iter; wall-clock 3.5-6ms; ~26 kernel launches/iter.
  This caps the yield of any single-kernel optimization and makes launch-count reduction
  (disable_recompute) the dominant lever.

NO kernel code change is promoted — every candidate kernel change was reverted (gate green
  throughout, kernel back to iter0). The actionable outcome is a USAGE recommendation
  (disable_recompute=True for training), already exposed by the registry as chunk_qgdn2_save.
  Therefore there is no PR/diff to open for a kernel change. If a code change is desired, it
  would be to make disable_recompute=True the DEFAULT for training (a behavior change, needs
  a separate justification + the memory tradeoff documented for larger shapes).

## Final verification (reverted iter0 kernel state, fwdbwd, gdn2-controlled)

Run A (fwd+fwdbwd): D64 gdn2 5.871 | qgdn2 5.919 | save 5.249
                     D32 gdn2 5.781 | qgdn2 4.251* | save 5.139   (* anomaly, see TRAPS)
Run B (fwdbwd only, re-run after anomaly): 
  D64: gdn2 5.849 | qgdn2 5.974 | qgdn2_save 5.330
  D32: gdn2 5.790 | qgdn2 5.817 | qgdn2_save 5.178
qgdn2_save ~11% faster than qgdn2 on BOTH shapes; identical peak memory (99/370 MB).

## Follow-up integration

Added a `disable_recompute` constructor flag to `QGatedDeltaNet2` and threaded it to
`chunk_qgdn2(..., disable_recompute=...)` only during training. This preserves the
default behavior while exposing the measured `chunk_qgdn2_save` path to model-level
training code. The C4 benchmark now has a `qgdn2_save` model choice for direct loss
and throughput comparison against `qgdn2`.

Augur09 A100 validation:
  - `python -m pytest tests/models/test_modeling_qgdn2.py -q` passed both
    `disable_recompute=False` and `disable_recompute=True` cases in the CUDA 13 container.
  - `python -m benchmarks.ops.verify --op chunk_qgdn2_save --modes fwdbwd` passed the
    frozen `tests/ops/test_qgdn2.py` gate (3/3), then timing was discarded because a Ray
    worker started using GPU 3 during the benchmark phase.
Gate: 3/3 green (1.60s). Kernel back to iter0 (all changes reverted). Verdict confirmed.

DoD status:
  - full tests/ops/test_qgdn2.py gate passes: YES
  - chunk_qgdn2_save repeatable fwdbwd speedup (~11% vs chunk_qgdn2, same session): YES
  - chunk_gdn2 comparison included for context: YES
  - profiler output explains the win (recompute path = 46% of kernel time, skipped by save): YES
  - no default-shape regression (no kernel changed): YES
  - NCU summary: UNAVAILABLE (ERR_NVGPUCTRPERM); win is a kwarg change, not a kernel -> NCU
    not the right evidence; benchmark numbers are the evidence of record.




