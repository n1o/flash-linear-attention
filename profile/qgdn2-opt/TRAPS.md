# Optimization Traps — chunk_qgdn2 (session-local)

Seeded from references/TRAPS.md. Add new ones hit this session.

## Session-specific

### Gate is small-shape only — large-T correctness NOT covered

**Fact:** `tests/ops/test_qgdn2.py` runs only `T=64`, `T=65`, `K=V=32` (3 tests).
The benchmark shapes are `T=2048` (D32) and `T=4096` (D64).
A kernel change that corrupts large-T output (e.g. int64 address-overflow at large T,
a tail-block partial write) can pass the gate while breaking the benchmark shapes.

**Why:** run.py only *times*; it does not assert correctness on benchmark shapes.
conftest NaN-poisoning applies to `tests/ops/`, not to the benchmark runner.

**How to apply:** After every kernel change, besides the green gate, run a manual
finite-output sanity check on a benchmark shape:
```python
o, s = chunk_qgdn2(q,k,v,g,b,w,lq, output_final_state=True)
(o.sum() + s.sum()).backward()
assert torch.isfinite(o).all() and torch.isfinite(q.grad).all()  # and k,v,g,b,w,lq
```
Run this BEFORE trusting a benchmark speedup. An implausible speedup here = silent skip.

### No git in this checkout

**Fact:** `git status` fails ("not a git repository"). No commits, no `--base main`.

**How to apply:** The OPT_LOG row IS the commit record. Keep kernel diffs minimal
and reversible (copy parent kernel to scratch before editing if needed).
Re-measure baseline+candidate back-to-back for verdicts (can't diff against a ref).

### Clocks unlocked, no sudo to lock

**Fact:** B300 idle 120MHz / max 2032MHz; `sudo nvidia-smi -lgc` unavailable (no sudo);
`--query-gpu=clocks.applications.gr` is deprecated on this driver.

**How to apply:** iteration signal = solution's own runtime in one run;
verdict = baseline+candidate back-to-back in one session; never sum deltas.

### Expanding autotune configs can break the gate via numerical drift

**Fact:** Adding `num_warps`/`num_stages` variants to a kernel that previously had a
single config can make the GATE FAIL on `assert_close`, even though outputs stay finite
(no NaN) — a numerical drift, not a crash.

**Why:** Different `num_stages` reorders software-pipelined loads; different `num_warps`
changes the reduction/dot accumulation tree. Both reassociate floating-point ops and shift
rounding. Autotune optimizes SPEED only (it does not know the tolerance), so it picks the
fastest config, which may be the least accurate. For a kernel whose grads sit near the
frozen tolerance (qgdn2 grad tol = 0.05), the shift tips it over.

**How to apply:** Treat the single-config autotune as deliberate (numerically pinned).
If you expand configs, the gate is the judge — a red gate means drop, do NOT loosen the
tolerance. Confirm the gate (full, no `-k`) after every config-space change, not just the
finite check (finite != within-tolerance). Observed on `chunk_gdn2_bwd_kernel_intra`.

### Finite != within-tolerance

**Fact:** The large-shape finite sanity check can PASS while the small-shape gate FAILS.

**Why:** The finite check only guards NaN/Inf (partial writes, overflow). The gate's
`assert_close` is a relative-tolerance check that finite-but-drifting results fail.

**How to apply:** The finite check is a NECESSARY extra guard for large-T (the gate doesn't
cover large shapes), but it is NOT a substitute for the gate. Both must pass.

### do_bench can return a structurally-impossible outlier (re-run before trusting)

**Fact:** `triton.testing.do_bench` (used by run.py) returned `chunk_qgdn2 D32 = 4.251ms`
while `chunk_gdn2 = 5.781ms` and `chunk_qgdn2_save = 5.139ms` in the SAME run. qgdn2 being
faster than qgdn2_save is structurally impossible (qgdn2 does the recompute that save skips).

**Why:** do_bench's quantile timing can glitch on the first timed call after autotune
warmup (clock state, cold cache, or a one-off scheduling artifact), producing an outlier
that the median doesn't fully reject on a small sample.

**How to apply:** If a number contradicts the structural ordering of the ops (a superset
op faster than its subset), do NOT trust it — re-run. Treat any single implausible number
as the "implausible speedup = silent skip" trap and re-measure before recording it.


