# QGDN2 Kernel Optimization Plan

This note is a practical follow-up plan for optimizing the QGDN2 chunk kernel on a B300/Kubeflow setup.
The goal is to make the training kernel faster without weakening correctness or changing the public operator contract.

## Scope

- Operator: `chunk_qgdn2`
- Module: `fla.ops.qgdn2`
- Main entry point: `fla/ops/qgdn2/chunk.py`
- Benchmark registry entry: `benchmarks/ops/registry.py`
- Correctness gate: `tests/ops/test_qgdn2.py`
- Comparison operators:
  - `chunk_gdn2`
  - `chunk_qgdn2`
  - `chunk_qgdn2_save`

The QGDN2 chunk implementation is a pipeline, not a single monolithic kernel:

1. Per-chunk decay cumsum.
2. Intra-chunk score construction and triangular solve.
3. Inter-chunk state recurrence.
4. Output composition.
5. Backward through the WY, state, and score paths.

The likely hot areas are the intra-chunk solve and the fused backward path, especially the QGDN2 erase branch:

```text
x = k + lq * q
```

## Correctness Rule

Do not edit or relax:

- `tests/ops/test_qgdn2.py`
- `fla/ops/qgdn2/naive.py`
- `assert_close` tolerances
- the public `chunk_qgdn2` signature

Every candidate must pass:

```bash
python -m benchmarks.ops.verify --op chunk_qgdn2 --modes fwd fwdbwd
```

Use subset gates only for fast signals. Promote only after the full gate passes.

## Baseline Commands

Run these inside the Kubeflow notebook terminal:

```bash
cd ~/flash-linear-attention

export FLA_BENCH_OP_WARMUP_ITERS=10
export FLA_BENCH_WARMUP_MS=100
export FLA_BENCH_REP_MS=500

python -m benchmarks.ops.verify --op chunk_qgdn2 --modes fwd fwdbwd
python -m benchmarks.ops.verify --op chunk_qgdn2_save --modes fwd fwdbwd

python -m benchmarks.ops.run \
  --op chunk_gdn2 chunk_qgdn2 chunk_qgdn2_save \
  --modes fwd fwdbwd
```

Do not use `--base main` for QGDN2 unless `main` already contains QGDN2.
For now, compare QGDN2 against current GDN2 and against the saved-intermediate variant.

## First Profiling Pass

Use the repo benchmark profiler first:

```bash
python -m benchmarks.ops.verify --op chunk_qgdn2 --profile --modes fwdbwd
```

If Nsight Compute is available, collect one representative full profile and one source profile.
Keep reports under `profile/qgdn2-opt/`, not in git.

Example:

```bash
mkdir -p profile/qgdn2-opt/reports

ncu --set full --section PmSampling --section PmSampling_WarpStates \
  -k "regex:.*qgdn2.*|.*gdn2.*" -c 1 \
  -o profile/qgdn2-opt/reports/full_qgdn2_fwdbwd \
  python -m benchmarks.ops.run --op chunk_qgdn2 --modes fwdbwd

ncu --set source --section SourceCounters \
  -k "regex:.*qgdn2.*|.*gdn2.*" -c 1 \
  -o profile/qgdn2-opt/reports/source_qgdn2_fwdbwd \
  python -m benchmarks.ops.run --op chunk_qgdn2 --modes fwdbwd
```

## Optimization Candidates

### 1. Measure `disable_recompute=True`

`chunk_qgdn2_save` keeps intermediates for backward. This can be faster if B300 memory is not the limiting factor.

Decision:

- If `chunk_qgdn2_save` improves `fwdbwd` materially and memory remains acceptable, keep it as the default training recommendation or add a training-mode dispatch.
- If it only improves tiny shapes or increases memory too much, keep it optional.

### 2. Expand Triton Autotune Configs

Current QGDN2 config spaces are narrow. Try more tilings around:

- `BK = 32, 64`
- `BV = 32, 64`
- `num_warps = 4, 8`
- `num_stages = 2, 3`

Tune separately for the default QGDN2 shapes:

```text
B8_T2048_H2_D32
B4_T4096_H4_D64
```

Do not claim a win until both shapes are measured. A config that wins only one shape may need dispatch.

### 3. Inspect the QGDN2 Erase Branch

The QGDN2-specific extra work is:

```text
x = k + lq * q
```

Possible directions:

- Keep `x` computed inline if compute is cheap and memory is the bottleneck.
- Precompute `x` only if profiling shows repeated `q/lq` loads dominate.
- Consider precomputing `x * exp(g)` only if it reduces more traffic than it adds.

This needs profiler evidence. Precomputing blindly can make the kernel slower.

### 4. Backward Kernel Focus

The backward path is likely the most important for training throughput.
Inspect:

- repeated `q`, `k`, `lq`, `g`, `b`, and `w_gate` loads
- dot-product precision choices
- `BK/BV` tile balance
- register pressure and occupancy
- any extra synchronization or barrier cost

The first target should be `fwdbwd`, not forward-only speed.

### 5. Shape Dispatch Only After Evidence

Potential buckets:

- `D = 32`
- `D = 64`
- short context
- long context
- saved-intermediate vs recompute

Only add dispatch when benchmarks show different kernels win different buckets.
Record the bucket, chosen path, baseline time, candidate time, speedup, and reason in `profile/qgdn2-opt/dispatch.md`.

## What Not To Do

Do not replace the intra-chunk triangular solve with a plain `cumsum`.
The gate cumsum is already handled separately.
The triangular inverse has row dependencies through matrix products, so it is not a simple prefix sum.

Do not loosen correctness tolerances, skip gradients, or benchmark a candidate after a red gate.

Do not compare against `main` for QGDN2 until `main` has the same operator registered.

## Iteration Protocol

For each optimization iteration:

1. Make one kernel change.
2. Run the correctness-gated benchmark:

   ```bash
   python -m benchmarks.ops.verify --op chunk_qgdn2 --modes fwd fwdbwd
   ```

3. Append a row to `profile/qgdn2-opt/OPT_LOG.md`.
4. Commit the candidate if it is worth preserving.

Suggested log layout:

```markdown
# Optimization log - chunk_qgdn2

Env: <Machine line from verify.py>
Baseline: <median ms on target shapes>
Target: faster fwdbwd on B300 with full gate green

| Iter | Direction | Gate | Bench | vs best | Status |
| ---- | --------- | ---- | ----- | ------- | ------ |
| 0 | baseline | pass | TBD | - | base |
| 1 | measure disable_recompute | TBD | TBD | TBD | TBD |
```

## Definition Of Done

A QGDN2 kernel change is promotable when:

- full `tests/ops/test_qgdn2.py` gate passes
- `chunk_qgdn2` and/or `chunk_qgdn2_save` shows repeatable `fwdbwd` speedup
- `chunk_gdn2` comparison is included for context
- profiler output explains the win
- no shape in the default QGDN2 benchmark set regresses without an explicit trade-off

Final evidence should include:

```bash
python -m benchmarks.ops.verify --op chunk_qgdn2 --modes fwd fwdbwd
python -m benchmarks.ops.verify --op chunk_qgdn2_save --modes fwd fwdbwd
python -m benchmarks.ops.run --op chunk_gdn2 chunk_qgdn2 chunk_qgdn2_save --modes fwd fwdbwd
```

And one short Nsight Compute summary for the winning kernel:

- GPU model
- CUDA version
- PyTorch version
- Triton version
- occupancy or SOL summary
- memory throughput summary
- top hot instructions or source lines
- final decision: keep, dispatch, or no-go
