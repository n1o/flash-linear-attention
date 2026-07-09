# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import argparse
import fnmatch
import gzip
import json
import math
import os
from pathlib import Path
import time

import httpx
import torch
from datasets import load_dataset
from huggingface_hub.utils import _http as hf_http
from torch.nn import functional as F

from benchmarks.benchmark_gdn2_parallax_quality import TinyFLALM, copy_shared_state, gamma_norm, sizeof_fmt
from fla.layers import GatedDeltaNet2, GDN2Parallax, QGatedDeltaNet2


def configure_ssl(args: argparse.Namespace) -> None:
    if not args.disable_ssl_verify:
        return
    os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    hf_http.set_client_factory(
        lambda: httpx.Client(
            event_hooks={"request": [hf_http.hf_request_event_hook]},
            follow_redirects=True,
            timeout=None,
            verify=False,
        )
    )
    hf_http.close_session = lambda: None


def load_c4_stream(args: argparse.Namespace, split: str, shuffle: bool):
    kwargs = {
        "split": split,
        "streaming": True,
    }
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    try:
        if args.dataset_config:
            dataset = load_dataset(args.dataset_name, args.dataset_config, **kwargs)
        else:
            dataset = load_dataset(args.dataset_name, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Failed to open the C4 stream. If this is an HTTPS certificate issue in the benchmark container, "
            "retry with `--disable-ssl-verify` or fix the container CA bundle."
        ) from exc
    if shuffle and args.shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    return dataset


def iter_text_file(path: str | Path, text_field: str):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = {text_field: line}
            if text_field in row:
                yield row


def encode_bytes(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="ignore"))


def materialize_byte_batches(
    dataset,
    text_field: str,
    batch_size: int,
    seq_len: int,
    num_batches: int,
) -> list[torch.Tensor]:
    block_len = seq_len + 1
    token_buffer: list[int] = []
    blocks: list[torch.Tensor] = []
    batches: list[torch.Tensor] = []
    for row in dataset:
        text = row.get(text_field)
        if not text:
            continue
        token_buffer.extend(encode_bytes(text))
        token_buffer.append(10)
        while len(token_buffer) >= block_len:
            block = torch.tensor(token_buffer[:block_len], dtype=torch.long)
            del token_buffer[:block_len]
            blocks.append(block)
            if len(blocks) == batch_size:
                batches.append(torch.stack(blocks, dim=0))
                blocks = []
                if len(batches) == num_batches:
                    return batches
    raise RuntimeError(
        f"C4 stream ended before {num_batches} batches could be materialized. "
        f"Collected {len(batches)} batches."
    )


def batch_loss(model: torch.nn.Module, batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    batch = batch.to(device=device, non_blocking=True)
    logits = model(batch[:, :-1])
    targets = batch[:, 1:]
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))


def append_jsonl(path: str | None, payload: dict) -> None:
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True))
        f.write("\n")


class OptimizerGroup:

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict]:
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()


def stamp_initial_lrs(optimizer: torch.optim.Optimizer | OptimizerGroup) -> None:
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])


def is_norm_param(name: str) -> bool:
    return any(part == "norm" or part.endswith("_norm") for part in name.split("."))


def is_decay_param(name: str) -> bool:
    return "A_log" in name or "dt_bias" in name


def should_use_muon(name: str, param: torch.nn.Parameter, args: argparse.Namespace) -> bool:
    if param.ndim != 2:
        return False
    if args.muon_include is not None and not any(
        fnmatch.fnmatchcase(name, pattern) for pattern in args.muon_include
    ):
        return False
    if name.startswith(("embed.", "lm_head.")):
        return False
    if name.endswith(".bias") or is_norm_param(name) or is_decay_param(name) or "parallax_gamma" in name:
        return False
    if args.muon_exclude_decay_proj and ".f_proj." in name:
        return False
    return True


def adamw_fallback_groups(
    named_params: list[tuple[str, torch.nn.Parameter]],
    args: argparse.Namespace,
) -> list[dict]:
    fallback_lr = args.fallback_lr if args.fallback_lr is not None else args.muon_lr
    groups_by_key: dict[tuple[float, float], dict] = {}
    for name, param in named_params:
        lr = fallback_lr
        if name.startswith(("embed.", "lm_head.")):
            lr *= args.muon_embed_lr_mult
        elif is_norm_param(name):
            lr *= args.muon_norm_lr_mult

        weight_decay = args.fallback_weight_decay
        if param.ndim <= 1 or name.endswith(".bias") or is_norm_param(name) or getattr(param, "_no_weight_decay", False):
            weight_decay = 0.0

        key = (lr, weight_decay)
        if key not in groups_by_key:
            groups_by_key[key] = {"params": [], "lr": lr, "weight_decay": weight_decay}
        groups_by_key[key]["params"].append(param)
    return list(groups_by_key.values())


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer | OptimizerGroup:
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
        stamp_initial_lrs(optimizer)
        return optimizer
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("torch.optim.Muon is not available in this PyTorch build.")

    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if should_use_muon(name, param, args):
            muon_params.append((name, param))
        else:
            adamw_params.append((name, param))
    optimizers: list[torch.optim.Optimizer] = []
    if muon_params:
        optimizers.append(
            torch.optim.Muon(
                muon_params,
                lr=args.muon_lr,
                weight_decay=args.muon_weight_decay,
                momentum=args.muon_momentum,
                nesterov=not args.muon_no_nesterov,
                ns_steps=args.muon_ns_steps,
            )
        )
    if adamw_params:
        optimizers.append(
            torch.optim.AdamW(
                adamw_fallback_groups(adamw_params, args),
                betas=(args.fallback_beta1, args.fallback_beta2),
                eps=args.fallback_eps,
            )
        )
    optimizer = OptimizerGroup(optimizers)
    stamp_initial_lrs(optimizer)
    print(
        f"optimizer=muon muon_tensors={len(muon_params)} fallback_tensors={len(adamw_params)} "
        f"muon_lr={args.muon_lr} fallback_lr={args.fallback_lr if args.fallback_lr is not None else args.muon_lr} "
        f"embed_lr_mult={args.muon_embed_lr_mult} norm_lr_mult={args.muon_norm_lr_mult}",
        flush=True,
    )
    print(f"muon_examples={[name for name, _ in muon_params[:8]]}", flush=True)
    print(f"muon_rho_proj={[name for name, _ in muon_params if 'rho_proj' in name]}", flush=True)
    return optimizer


def lr_schedule_scale(step: int, total_steps: int, args: argparse.Namespace) -> float:
    if args.scheduler == "constant":
        return 1.0
    if total_steps <= 1:
        return args.lr_final_scale
    progress = step / (total_steps - 1)
    if args.lr_warmup_ratio > 0.0 and progress < args.lr_warmup_ratio:
        return progress / args.lr_warmup_ratio
    if progress < args.lr_decay_start_ratio:
        return 1.0
    decay_span = max(1e-12, 1.0 - args.lr_decay_start_ratio)
    decay_progress = min(1.0, max(0.0, (progress - args.lr_decay_start_ratio) / decay_span))
    return 1.0 + (args.lr_final_scale - 1.0) * decay_progress


def set_scheduled_lrs(
    optimizer: torch.optim.Optimizer | OptimizerGroup,
    step: int,
    total_steps: int,
    args: argparse.Namespace,
) -> float:
    scale = lr_schedule_scale(step, total_steps, args)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * scale
    return scale


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    device: torch.device,
) -> tuple[float, float]:
    was_training = model.training
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    for batch in batches:
        batch = batch.to(device=device, non_blocking=True)
        logits = model(batch[:, :-1])
        targets = batch[:, 1:]
        total_loss += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)).item()
        predictions = logits.argmax(dim=-1)
        total_correct += (predictions == targets).sum().item()
        total_tokens += targets.numel()
    model.train(was_training)
    return total_loss / len(batches), total_correct / total_tokens


def prewarm(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    device: torch.device,
    warmup_steps: int,
) -> None:
    model.train()
    for batch in batches[:warmup_steps]:
        loss = batch_loss(model, batch, device)
        loss.backward()
        model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_one(
    name: str,
    model: torch.nn.Module,
    train_batches: list[torch.Tensor],
    val_batches: list[torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
    grad_clip: float | None,
    warmup_steps: int,
    max_seconds: float | None,
    log_every_steps: int,
    progress_jsonl: str | None,
) -> dict[str, float | None]:
    prewarm(model, train_batches, device, warmup_steps)
    optimizer = build_optimizer(args, model)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    train_loss = math.nan
    steps = 0
    tokens = 0
    log_loss = 0.0
    log_steps = 0
    for batch in train_batches:
        if max_seconds is not None and steps > 0 and time.perf_counter() - start >= max_seconds:
            break
        lr_scale = set_scheduled_lrs(optimizer, steps, len(train_batches), args)
        loss = batch_loss(model, batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        train_loss = loss.item()
        steps += 1
        tokens += batch.shape[0] * (batch.shape[1] - 1)
        log_loss += train_loss
        log_steps += 1
        if log_every_steps > 0 and steps % log_every_steps == 0:
            elapsed = time.perf_counter() - start
            avg_loss = log_loss / log_steps
            progress = {
                "event": "train_progress",
                "model": name,
                "step": steps,
                "tokens": tokens,
                "train_loss": train_loss,
                "avg_train_loss": avg_loss,
                "elapsed_s": elapsed,
                "tokens_per_second": tokens / elapsed,
                "lr_scale": lr_scale,
            }
            print(
                f"{name:14s} step={steps:6d} tokens={tokens:9d} "
                f"loss={train_loss:.4f} avg_loss={avg_loss:.4f} "
                f"tok/s={tokens / elapsed:.1f} lr_scale={lr_scale:.4f}",
                flush=True,
            )
            append_jsonl(progress_jsonl, progress)
            log_loss = 0.0
            log_steps = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    val_loss, val_acc = evaluate(model, val_batches, device)
    peak_memory_mib = (
        torch.cuda.max_memory_allocated(device) / 1024 / 1024
        if device.type == "cuda" else 0.0
    )
    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_ppl": math.exp(min(val_loss, 20.0)),
        "val_acc": val_acc,
        "steps": float(steps),
        "tokens": float(tokens),
        "ms_per_step": elapsed / max(steps, 1) * 1000,
        "tokens_per_second": tokens / elapsed,
        "peak_memory_mib": peak_memory_mib,
        "gamma_norm": gamma_norm(model),
    }
    gamma = "n/a" if metrics["gamma_norm"] is None else f"{metrics['gamma_norm']:.4f}"
    print(
        f"{name:14s} steps={steps:4d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"ppl={metrics['val_ppl']:.2f} acc={val_acc * 100:5.2f}% "
        f"step={metrics['ms_per_step']:.2f} ms tok/s={metrics['tokens_per_second']:.1f} "
        f"peak={peak_memory_mib:.1f} MiB gamma={gamma}",
        flush=True,
    )
    append_jsonl(progress_jsonl, {"event": "model_final", "model": name, **metrics})
    return metrics


def build_models(args: argparse.Namespace, device: torch.device) -> dict[str, torch.nn.Module]:
    kwargs = dict(
        vocab_size=256,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        head_dim=args.head_dim,
        num_heads=args.num_heads,
        num_v_heads=args.num_v_heads,
        mlp_ratio=args.mlp_ratio,
        use_short_conv=args.use_short_conv,
    )
    torch.manual_seed(args.seed)
    gdn2 = TinyFLALM(GatedDeltaNet2, **kwargs)
    torch.manual_seed(args.seed)
    qgdn2 = TinyFLALM(QGatedDeltaNet2, **kwargs)
    torch.manual_seed(args.seed)
    parallax = TinyFLALM(GDN2Parallax, **kwargs)
    torch.manual_seed(args.seed)
    last_parallax_classes = [GatedDeltaNet2] * max(args.num_layers - 1, 0) + [GDN2Parallax]
    parallax_last = TinyFLALM(last_parallax_classes, **kwargs)
    shared_q = copy_shared_state(gdn2, qgdn2)
    shared = copy_shared_state(gdn2, parallax)
    shared_last = copy_shared_state(gdn2, parallax_last)
    print(f"copied {shared_q} shared tensors from gdn2 init into qgdn2")
    print(f"copied {shared} shared tensors from gdn2 init into gdn2_parallax")
    print(f"copied {shared_last} shared tensors from gdn2 init into gdn2_parallax_last")
    dtype = getattr(torch, args.dtype)
    all_models = {
        "gdn2": gdn2.to(device=device, dtype=dtype),
        "qgdn2": qgdn2.to(device=device, dtype=dtype),
        "gdn2_parallax": parallax.to(device=device, dtype=dtype),
        "gdn2_parallax_last": parallax_last.to(device=device, dtype=dtype),
    }
    return {name: all_models[name] for name in args.models}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Budgeted C4 byte-level LM benchmark for GDN2 vs GDN2-Parallax.")
    parser.add_argument("--dataset-name", default="allenai/c4")
    parser.add_argument("--dataset-config", default="en")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--validation-file", default=None)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-train-tokens", type=int, default=262_144)
    parser.add_argument("--max-val-tokens", type=int, default=65_536)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("gdn2", "qgdn2", "gdn2_parallax", "gdn2_parallax_last"),
        default=["gdn2", "gdn2_parallax"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-v-heads", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=int, default=2)
    parser.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--muon-lr", type=float, default=5e-3)
    parser.add_argument("--muon-weight-decay", type=float, default=0.1)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-no-nesterov", action="store_true")
    parser.add_argument("--muon-include", nargs="+", default=None)
    parser.add_argument("--muon-embed-lr-mult", type=float, default=0.3)
    parser.add_argument("--muon-norm-lr-mult", type=float, default=0.015)
    parser.add_argument("--muon-exclude-decay-proj", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback-lr", type=float, default=None)
    parser.add_argument("--fallback-weight-decay", type=float, default=0.0)
    parser.add_argument("--fallback-beta1", type=float, default=0.8)
    parser.add_argument("--fallback-beta2", type=float, default=0.95)
    parser.add_argument("--fallback-eps", type=float, default=1e-7)
    parser.add_argument("--scheduler", choices=("constant", "wsd"), default="constant")
    parser.add_argument("--lr-warmup-ratio", type=float, default=0.0)
    parser.add_argument("--lr-decay-start-ratio", type=float, default=0.8)
    parser.add_argument("--lr-final-scale", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-short-conv", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-ssl-verify", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--progress-jsonl", default=None)
    parser.add_argument("--log-every-steps", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_ssl(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.num_v_heads < args.num_heads or args.num_v_heads % args.num_heads != 0:
        raise ValueError("num_v_heads must be >= num_heads and divisible by num_heads.")
    if (args.train_file is None) != (args.validation_file is None):
        raise ValueError("Pass both --train-file and --validation-file, or neither.")
    if not 0.0 <= args.lr_warmup_ratio <= args.lr_decay_start_ratio <= 1.0:
        raise ValueError("--lr-warmup-ratio must be <= --lr-decay-start-ratio and both must be in [0, 1].")
    if not 0.0 <= args.lr_final_scale <= 1.0:
        raise ValueError("--lr-final-scale must be in [0, 1].")
    train_batches = max(1, args.max_train_tokens // (args.batch_size * args.seq_len))
    val_batches = max(1, args.max_val_tokens // (args.batch_size * args.seq_len))
    if args.progress_jsonl is None and args.output_json is not None:
        output = Path(args.output_json)
        args.progress_jsonl = str(output.with_name(f"{output.stem}.progress.jsonl"))
    data_source = (
        f"local train_file={args.train_file} validation_file={args.validation_file}"
        if args.train_file is not None
        else f"dataset={args.dataset_name}/{args.dataset_config} train={args.train_split} validation={args.validation_split}"
    )

    print("== benchmark_gdn2_parallax_c4 ==")
    print(
        f"{data_source} byte_vocab=256 seq_len={args.seq_len} batch={args.batch_size}"
    )
    print(
        f"budget train_tokens<={train_batches * args.batch_size * args.seq_len} "
        f"val_tokens<={val_batches * args.batch_size * args.seq_len} max_seconds={args.max_seconds}"
    )
    print(
        f"model hidden={args.hidden_size} layers={args.num_layers} heads={args.num_heads} "
        f"v_heads={args.num_v_heads} head_dim={args.head_dim} dtype={args.dtype} device={device} models={args.models}"
    )
    print(
        f"optimizer={args.optimizer} scheduler={args.scheduler} lr={args.lr} weight_decay={args.weight_decay} "
        f"muon_lr={args.muon_lr} muon_weight_decay={args.muon_weight_decay} "
        f"lr_warmup_ratio={args.lr_warmup_ratio} lr_decay_start_ratio={args.lr_decay_start_ratio} "
        f"lr_final_scale={args.lr_final_scale}"
    )
    if args.progress_jsonl is not None:
        progress = Path(args.progress_jsonl)
        progress.parent.mkdir(parents=True, exist_ok=True)
        progress.write_text("", encoding="utf-8")
        append_jsonl(
            args.progress_jsonl,
            {
                "event": "run_start",
                "args": vars(args),
                "train_batches": train_batches,
                "validation_batches": val_batches,
                "train_tokens_budget": train_batches * args.batch_size * args.seq_len,
                "validation_tokens_budget": val_batches * args.batch_size * args.seq_len,
            },
        )
        print(f"progress_jsonl={args.progress_jsonl}", flush=True)

    load_start = time.perf_counter()
    if args.train_file is not None:
        train_dataset = iter_text_file(args.train_file, args.text_field)
        val_dataset = iter_text_file(args.validation_file, args.text_field)
    else:
        train_dataset = load_c4_stream(args, args.train_split, shuffle=True)
        val_dataset = load_c4_stream(args, args.validation_split, shuffle=False)
    train_data = materialize_byte_batches(
        dataset=train_dataset,
        text_field=args.text_field,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_batches=train_batches,
    )
    val_data = materialize_byte_batches(
        dataset=val_dataset,
        text_field=args.text_field,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_batches=val_batches,
    )
    print(f"materialized data in {time.perf_counter() - load_start:.2f}s")

    models = build_models(args, device)
    results = {
        name: train_one(
            name=name,
            model=model,
            train_batches=train_data,
            val_batches=val_data,
            device=device,
            args=args,
            grad_clip=args.grad_clip,
            warmup_steps=args.warmup_steps,
            max_seconds=args.max_seconds,
            log_every_steps=args.log_every_steps,
            progress_jsonl=args.progress_jsonl,
        )
        for name, model in models.items()
    }

    print("\nsummary")
    print("model          steps  tokens    val_loss  ppl      val_acc  ms/step   tokens/s   peak_mem   gamma_norm")
    for name, metrics in results.items():
        gamma = "n/a" if metrics["gamma_norm"] is None else f"{metrics['gamma_norm']:.4f}"
        print(
            f"{name:14s} {metrics['steps']:5.0f} {metrics['tokens']:7.0f} "
            f"{metrics['val_loss']:9.4f} {metrics['val_ppl']:7.2f} "
            f"{metrics['val_acc'] * 100:7.2f}% {metrics['ms_per_step']:8.2f} "
            f"{metrics['tokens_per_second']:10.1f} "
            f"{sizeof_fmt(metrics['peak_memory_mib'] * 1024 * 1024):>10s} {gamma:>10s}"
        )
    if args.output_json is not None:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "args": vars(args),
                    "train_batches": train_batches,
                    "validation_batches": val_batches,
                    "train_tokens_budget": train_batches * args.batch_size * args.seq_len,
                    "validation_tokens_budget": val_batches * args.batch_size * args.seq_len,
                    "results": results,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"\nwrote metrics to {output}")


if __name__ == "__main__":
    main()
