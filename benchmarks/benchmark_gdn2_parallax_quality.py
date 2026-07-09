# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from fla.layers import GatedDeltaNet2, GDN2Parallax


@dataclass
class Metrics:
    train_loss: float
    val_loss: float
    val_acc: float
    ms_per_step: float
    tokens_per_second: float
    peak_memory_mib: float
    gamma_norm: float | None


class TinyBlock(nn.Module):

    def __init__(
        self,
        mixer_cls: type[nn.Module],
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_v_heads: int,
        mlp_ratio: int,
        use_short_conv: bool,
    ) -> None:
        super().__init__()
        self.mixer_norm = nn.RMSNorm(hidden_size)
        self.mixer = mixer_cls(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_v_heads=num_v_heads,
            use_short_conv=use_short_conv,
            mode="chunk",
        )
        intermediate_size = hidden_size * mlp_ratio
        self.mlp_norm = nn.RMSNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(intermediate_size, hidden_size, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _, _ = self.mixer(self.mixer_norm(x), use_cache=False)
        x = x + y
        return x + self.mlp(self.mlp_norm(x))


class TinyFLALM(nn.Module):

    def __init__(
        self,
        mixer_cls: type[nn.Module] | list[type[nn.Module]],
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        head_dim: int,
        num_heads: int,
        num_v_heads: int,
        mlp_ratio: int,
        use_short_conv: bool,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        mixer_classes = [mixer_cls] * num_layers if isinstance(mixer_cls, type) else mixer_cls
        if len(mixer_classes) != num_layers:
            raise ValueError(f"Expected {num_layers} mixer classes, got {len(mixer_classes)}.")
        self.blocks = nn.ModuleList([
            TinyBlock(
                mixer_cls=mixer_classes[i],
                hidden_size=hidden_size,
                head_dim=head_dim,
                num_heads=num_heads,
                num_v_heads=num_v_heads,
                mlp_ratio=mlp_ratio,
                use_short_conv=use_short_conv,
            )
            for i in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))


def sizeof_fmt(num: float, suffix: str = 'B') -> str:
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:.2f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.2f}Yi{suffix}'


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    return generator


def make_copy_batch(
    batch_size: int,
    copy_span: int,
    vocab_size: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if vocab_size < 3:
        raise ValueError("vocab_size must be at least 3 so the separator has its own token.")
    generator = make_generator(device, seed)
    prefix = torch.randint(
        low=0,
        high=vocab_size - 1,
        size=(batch_size, copy_span),
        generator=generator,
        device=device,
    )
    separator = torch.full((batch_size, 1), vocab_size - 1, device=device, dtype=torch.long)
    tokens = torch.cat([prefix, separator, prefix], dim=1)

    target_mask = torch.zeros(batch_size, tokens.shape[1] - 1, device=device, dtype=torch.bool)
    target_mask[:, copy_span:] = True
    return tokens, target_mask


def copy_shared_state(source: nn.Module, target: nn.Module) -> int:
    source_state = source.state_dict()
    target_state = target.state_dict()
    shared = {
        key: value
        for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    target.load_state_dict(shared, strict=False)
    return len(shared)


def masked_lm_loss(logits: torch.Tensor, tokens: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    next_logits = logits[:, :-1]
    targets = tokens[:, 1:]
    return F.cross_entropy(next_logits[target_mask].float(), targets[target_mask])


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batch_size: int,
    copy_span: int,
    vocab_size: int,
    device: torch.device,
    eval_batches: int,
    eval_seed: int,
) -> tuple[float, float]:
    was_training = model.training
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    for i in range(eval_batches):
        tokens, target_mask = make_copy_batch(
            batch_size=batch_size,
            copy_span=copy_span,
            vocab_size=vocab_size,
            device=device,
            seed=eval_seed + i,
        )
        logits = model(tokens)
        loss = masked_lm_loss(logits, tokens, target_mask)
        predictions = logits[:, :-1].argmax(dim=-1)
        targets = tokens[:, 1:]
        total_loss += loss.item()
        total_correct += (predictions[target_mask] == targets[target_mask]).sum().item()
        total_tokens += target_mask.sum().item()
    model.train(was_training)
    return total_loss / eval_batches, total_correct / total_tokens


def gamma_norm(model: nn.Module) -> float | None:
    values = [
        module.parallax_gamma.detach().float().norm()
        for module in model.modules()
        if isinstance(module, GDN2Parallax)
    ]
    if not values:
        return None
    return torch.stack(values).norm().item()


def train_one(
    name: str,
    model: nn.Module,
    batch_size: int,
    copy_span: int,
    vocab_size: int,
    device: torch.device,
    steps: int,
    eval_batches: int,
    lr: float,
    weight_decay: float,
    train_seed: int,
    eval_seed: int,
    grad_clip: float | None,
) -> Metrics:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    train_loss = math.nan
    for step in range(steps):
        tokens, target_mask = make_copy_batch(
            batch_size=batch_size,
            copy_span=copy_span,
            vocab_size=vocab_size,
            device=device,
            seed=train_seed + step,
        )
        logits = model(tokens)
        loss = masked_lm_loss(logits, tokens, target_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        train_loss = loss.item()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    val_loss, val_acc = evaluate(
        model=model,
        batch_size=batch_size,
        copy_span=copy_span,
        vocab_size=vocab_size,
        device=device,
        eval_batches=eval_batches,
        eval_seed=eval_seed,
    )
    sequence_length = copy_span * 2 + 1
    tokens_per_second = steps * batch_size * sequence_length / elapsed
    peak_memory_mib = (
        torch.cuda.max_memory_allocated(device) / 1024 / 1024
        if device.type == "cuda" else 0.0
    )
    print(
        f"{name:14s} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"val_acc={val_acc * 100:6.2f}% step={elapsed / steps * 1000:.2f} ms "
        f"tokens/s={tokens_per_second:.1f} peak={peak_memory_mib:.1f} MiB"
    )
    return Metrics(
        train_loss=train_loss,
        val_loss=val_loss,
        val_acc=val_acc,
        ms_per_step=elapsed / steps * 1000,
        tokens_per_second=tokens_per_second,
        peak_memory_mib=peak_memory_mib,
        gamma_norm=gamma_norm(model),
    )


def build_models(args: argparse.Namespace, device: torch.device) -> dict[str, nn.Module]:
    kwargs = dict(
        vocab_size=args.vocab_size,
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
    parallax = TinyFLALM(GDN2Parallax, **kwargs)
    shared = copy_shared_state(gdn2, parallax)
    print(f"copied {shared} shared tensors from gdn2 init into gdn2_parallax")
    dtype = getattr(torch, args.dtype)
    return {
        "gdn2": gdn2.to(device=device, dtype=dtype),
        "gdn2_parallax": parallax.to(device=device, dtype=dtype),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small quality benchmark for GDN2 vs GDN2-Parallax.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--copy-span", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-v-heads", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=10_000)
    parser.add_argument("--eval-seed", type=int, default=20_000)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-short-conv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.num_v_heads < args.num_heads or args.num_v_heads % args.num_heads != 0:
        raise ValueError("num_v_heads must be >= num_heads and divisible by num_heads.")

    print("== benchmark_gdn2_parallax_quality ==")
    print(
        f"task=copy span={args.copy_span} seq_len={args.copy_span * 2 + 1} vocab={args.vocab_size} "
        f"batch={args.batch_size} steps={args.steps} eval_batches={args.eval_batches}"
    )
    print(
        f"model hidden={args.hidden_size} layers={args.num_layers} heads={args.num_heads} "
        f"v_heads={args.num_v_heads} head_dim={args.head_dim} dtype={args.dtype} device={device}"
    )

    models = build_models(args, device)
    results = {
        name: train_one(
            name=name,
            model=model,
            batch_size=args.batch_size,
            copy_span=args.copy_span,
            vocab_size=args.vocab_size,
            device=device,
            steps=args.steps,
            eval_batches=args.eval_batches,
            lr=args.lr,
            weight_decay=args.weight_decay,
            train_seed=args.train_seed,
            eval_seed=args.eval_seed,
            grad_clip=args.grad_clip,
        )
        for name, model in models.items()
    }

    print("\nsummary")
    print("model          val_loss  val_acc    ms/step   tokens/s   peak_mem   gamma_norm")
    for name, metrics in results.items():
        gamma = "n/a" if metrics.gamma_norm is None else f"{metrics.gamma_norm:.4f}"
        print(
            f"{name:14s} {metrics.val_loss:8.4f} {metrics.val_acc * 100:7.2f}% "
            f"{metrics.ms_per_step:9.2f} {metrics.tokens_per_second:10.1f} "
            f"{sizeof_fmt(metrics.peak_memory_mib * 1024 * 1024):>10s} {gamma:>10s}"
        )


if __name__ == "__main__":
    main()
