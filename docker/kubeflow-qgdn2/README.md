# QGDN2 Kubeflow Notebook Image

This image is for continuing the GDN2 / QGDN2 / Parallax experiments from a Kubeflow JupyterLab notebook.

It includes:

- CUDA PyTorch base image with Triton support
- CUDA 13.1 devel toolchain for B300 / Blackwell experiments
- this FLA worktree copied into `/opt/flash-linear-attention`
- editable FLA install with `cuda`, `test`, and `benchmark` extras
- Triton, TileLang, and CUDA Python bindings
- JupyterLab with a `Python (FLA QGDN2)` kernel
- `codex` from `@openai/codex`
- `opencode` from `opencode-ai`
- sanitized opencode config for the `ESET-inhouse/glm52` provider, without the Warp plugin
- ESET internal CA certificates for `*.kubeflow.hq.eset.com` endpoints

## Build

From the repo root:

```bash
scripts/build_kubeflow_qgdn2_image.sh
```

This builds and pushes:

```text
ba-docker-registry01.hq.eset.com/crd_incident_creator/ai-gatekeeper/fla-qgdn2-kubeflow:<git-sha>
ba-docker-registry01.hq.eset.com/crd_incident_creator/ai-gatekeeper/fla-qgdn2-kubeflow:latest
```

Useful overrides:

```bash
TAG=my-test \
BASE_IMAGE=nvidia/cuda:13.1.1-devel-ubuntu24.04 \
CODEX_NPM_PACKAGE='@openai/codex@0.143.0' \
OPENCODE_NPM_PACKAGE='opencode-ai@1.17.16' \
scripts/build_kubeflow_qgdn2_image.sh
```

## Kubeflow

Use the pushed image as the notebook image. Expose port `8888`.

The image is self-contained for code, Python dependencies, JupyterLab, Codex CLI, and opencode CLI.
API keys and other credentials are intentionally not baked into the image.
Set them at runtime or log in interactively from the notebook terminal.

The baked opencode config is stored at `/opt/opencode-config`. On notebook startup, `start-notebook` copies it to `/home/jovyan/.config/opencode` only when no opencode config exists there already. `auth.json`, opencode databases, and session state are not copied into the image.

FlashAttention-4 CUDA 13 is not installed in this image because the current `flash-attn-4[cu13]` wheels require `apache-tvm-ffi>=0.1.12`, while `tilelang==0.1.12` requires `apache-tvm-ffi<=0.1.11`. This image keeps TileLang and a clean `pip check` environment for QGDN2 kernel work.

Kubeflow usually sets `NB_PREFIX` for notebook routing. The entrypoint forwards it to JupyterLab.

On first startup, the image copies `/opt/flash-linear-attention` to:

```text
/home/jovyan/flash-linear-attention
```

That gives the notebook a writable workspace on the user PVC when `/home/jovyan` is mounted by Kubeflow.

## Smoke Checks

In a notebook terminal:

```bash
codex --version
opencode --version
opencode --pure providers list
python -m pytest tests/ops/test_qgdn2.py -q -s
```

Run the Triton op benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 \
FLA_BENCH_OP_WARMUP_ITERS=3 \
FLA_BENCH_WARMUP_MS=100 \
FLA_BENCH_REP_MS=500 \
python -m benchmarks.ops.run \
  --op chunk_gdn2 chunk_gdn2_save chunk_qgdn2 chunk_qgdn2_save \
  --modes fwdbwd
```

Run the C4/local JSONL quality benchmark if the data files are mounted:

```bash
python benchmarks/benchmark_gdn2_parallax_c4.py \
  --train-file /path/to/train.jsonl \
  --validation-file /path/to/validation.jsonl \
  --models gdn2 qgdn2 gdn2_parallax \
  --seq-len 2048 \
  --batch-size 8 \
  --hidden-size 128 \
  --num-layers 2 \
  --head-dim 32 \
  --num-heads 2 \
  --num-v-heads 2 \
  --max-train-tokens 10000000 \
  --max-val-tokens 65536 \
  --dtype float32 \
  --device cuda \
  --output-json /home/jovyan/qgdn2_compare.json \
  --log-every-steps 100
```
