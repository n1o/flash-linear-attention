# Task contract — chunk_qgdn2 optimization

## Op + entry point
- Op: `chunk_qgdn2` (= `chunk_gdn2` with `lq` erase branch) — see `fla/ops/qgdn2/__init__.py`.
- Entry point: `fla.ops.qgdn2.chunk.chunk_gdn2` in `fla/ops/qgdn2/chunk.py`.
- Pipeline (from plan): decay cumsum → intra-chunk score + triangular solve (`chunk_intra.py`)
  → inter-chunk state recurrence → output composition (`chunk_fwd.py`)
  → backward through WY / state / score (`chunk_bwd.py`, `wy_fast.py`).
- QGDN2-specific extra work: erase branch `x = k + lq * q` (unscaled q).

## Target
- Shapes (from `benchmarks/ops/registry.py` `_gdn2_shapes`):
  - `B8_T2048_H2_D32`
  - `B4_T4096_H4_D64`
- Primary target: **`fwdbwd`** speedup (training throughput). Forward-only is secondary.
- No numeric target yet — establish baseline first, then set a target.

## Allowed languages
- Triton (current implementation). No vendor-lib substitution; `torch` only as glue.

## Validation command (frozen gate)
```bash
python -m benchmarks.ops.verify --op chunk_qgdn2 --modes fwd fwdbwd
python -m benchmarks.ops.verify --op chunk_qgdn2_save --modes fwd fwdbwd
```
Gate test file: `tests/ops/test_qgdn2.py` (frozen — do NOT edit).
Naive reference: `fla/ops/qgdn2/naive.py` (frozen).

NOTE: the gate's pytest shapes are SMALL (`T=64`, `T=65`, `K=32`, `V=32`).
They do NOT cover the large benchmark shapes (`B4_T4096_H4_D64`).
Per the int64-overflow / large-T trap, any change must also be sanity-checked
on the benchmark shapes — but the frozen gate alone won't catch large-T corruption,
so we rely on benchmark runs themselves not erroring and producing sane timings.

## Benchmark command
```bash
python -m benchmarks.ops.run --op chunk_gdn2 chunk_qgdn2 chunk_qgdn2_save --modes fwd fwdbwd
```
No `--base main`: this checkout is NOT a git repo, and the plan says QGDN2 may not be on `main`.
We compare QGDN2 against current GDN2 (sibling op) and against the `disable_recompute=True`
saved-intermediate variant (`chunk_qgdn2_save`).

## Promotion criteria
- Full green gate on `tests/ops/test_qgdn2.py` (no `-k`, no `--no-gate`).
- Repeatable `fwdbwd` speedup on BOTH default shapes, measured in one session against
  the recorded baseline (clock-drift guard — see TRAPS below).
- A profiler / roofline reading that explains the win (NCU if available, else torch.profiler).
- No default-shape regression without an explicit, recorded trade-off.

## Frozen scope (do not touch during this loop)
- `tests/ops/test_qgdn2.py`
- `fla/ops/qgdn2/naive.py`
- `assert_close` tolerances
- public `chunk_qgdn2` / `chunk_gdn2` signature
- numeric flags must stay symmetric (no one-sided TF32 / precision relaxation)

## Candidate directions (from plan, ranked by expected benefit vs risk)
1. Measure `disable_recompute=True` (`chunk_qgdn2_save`) vs recompute — memory vs speed.
2. Backward kernel focus (most important for training): repeated q/k/lq/g/b/w_gate loads,
   dot-product precision, BK/BV tile balance, register pressure, sync/barrier cost.
3. QGDN2 erase branch `x = k + lq * q`: keep inline vs precompute — only with profiler evidence.
4. Expand Triton autotune (BK/BV 32/64, warps 4/8, stages 2/3), tuned per-shape.
5. Shape dispatch — only if benchmarks show divergent winners; record in `dispatch.md`.

## What not to do
- Don't replace the intra-chunk triangular solve with a plain cumsum (it has row
  dependencies through matrix products, not a prefix sum).
- Don't loosen tolerances, skip gradients, or bench after a red gate.
- Don't compare against `main` (QGDN2 not there).

## Iteration protocol (adapted: no git in this checkout)
- One kernel change per iteration.
- Run `verify.py` gate (full, no `-k`) — must stay green.
- Append one row to `OPT_LOG.md` (keep / revert / drop / floor).
- No git commit available — the OPT_LOG row IS the commit record. Keep the diff
  minimal and reversible by keeping a copy of the parent kernel when needed.

## TRAPS specific to this environment
- **Clock drift**: B300 unlocked, idle 120MHz / max 2032MHz, no sudo to lock.
  For iteration signal rank by the solution's own runtime in one run; for the verdict
  re-measure baseline and candidate back-to-back in one session. Never sum deltas.
- **Autotune cache staleness**: clear/rewarm after a config-space change.
- **Gate is small-shape only**: large-T correctness is NOT covered by the frozen gate.
- **Implausible speedup = silent skip**: confirm the shape actually ran.

## Definition of Done (from plan)
- full `tests/ops/test_qgdn2.py` gate passes
- `chunk_qgdn2` and/or `chunk_qgdn2_save` shows repeatable `fwdbwd` speedup
- `chunk_gdn2` comparison included for context
- profiler output explains the win
- no default-shape regression without explicit trade-off
- one short NCU summary for the winning kernel
