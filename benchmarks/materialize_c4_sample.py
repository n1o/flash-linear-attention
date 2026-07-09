# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import argparse
import gzip
import json
import os
import ssl
import urllib.request
from pathlib import Path


def configure_ssl(disable_ssl_verify: bool) -> None:
    if not disable_ssl_verify:
        return
    os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    import httpx
    from huggingface_hub.utils import _http as hf_http

    hf_http.set_client_factory(
        lambda: httpx.Client(
            event_hooks={"request": [hf_http.hf_request_event_hook]},
            follow_redirects=True,
            timeout=None,
            verify=False,
        )
    )
    hf_http.close_session = lambda: None


def load_stream(args: argparse.Namespace, split: str, shuffle: bool):
    from datasets import load_dataset

    kwargs = {
        "split": split,
        "streaming": True,
    }
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config, **kwargs)
    else:
        dataset = load_dataset(args.dataset_name, **kwargs)
    if shuffle and args.shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    return dataset


def iter_json_gz_url(url: str, text_field: str, disable_ssl_verify: bool):
    context = ssl._create_unverified_context() if disable_ssl_verify else None
    with urllib.request.urlopen(url, context=context, timeout=120) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            for raw_line in gz:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                row = json.loads(line)
                if text_field in row:
                    yield row


def write_sample(dataset, output: Path, text_field: str, max_text_bytes: int) -> tuple[int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    docs = 0
    text_bytes = 0
    with gzip.open(output, "wt", encoding="utf-8") as f:
        for row in dataset:
            text = row.get(text_field)
            if not text:
                continue
            encoded_len = len(text.encode("utf-8", errors="ignore"))
            f.write(json.dumps({text_field: text}, ensure_ascii=False))
            f.write("\n")
            docs += 1
            text_bytes += encoded_len
            if text_bytes >= max_text_bytes:
                return docs, text_bytes
    raise RuntimeError(f"dataset ended before {max_text_bytes} text bytes could be written to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize small C4 JSONL.gz samples for local-file benchmarks.")
    parser.add_argument("--dataset-name", default="allenai/c4")
    parser.add_argument("--dataset-config", default="en")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--train-output", default="/tmp/gdn2p-c4-train.jsonl.gz")
    parser.add_argument("--validation-output", default="/tmp/gdn2p-c4-validation.jsonl.gz")
    parser.add_argument(
        "--train-url",
        default="https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz",
    )
    parser.add_argument(
        "--validation-url",
        default="https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-validation.00000-of-00008.json.gz",
    )
    parser.add_argument("--use-datasets", action="store_true")
    parser.add_argument("--train-bytes", type=int, default=1_500_000)
    parser.add_argument("--validation-bytes", type=int, default=500_000)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-ssl-verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_datasets:
        configure_ssl(args.disable_ssl_verify)
        train_dataset = load_stream(args, args.train_split, shuffle=True)
        validation_dataset = load_stream(args, args.validation_split, shuffle=False)
    else:
        train_dataset = iter_json_gz_url(args.train_url, args.text_field, args.disable_ssl_verify)
        validation_dataset = iter_json_gz_url(args.validation_url, args.text_field, args.disable_ssl_verify)
    train_docs, train_bytes = write_sample(
        dataset=train_dataset,
        output=Path(args.train_output),
        text_field=args.text_field,
        max_text_bytes=args.train_bytes,
    )
    validation_docs, validation_bytes = write_sample(
        dataset=validation_dataset,
        output=Path(args.validation_output),
        text_field=args.text_field,
        max_text_bytes=args.validation_bytes,
    )
    print(f"train: {args.train_output} docs={train_docs} text_bytes={train_bytes}")
    print(f"validation: {args.validation_output} docs={validation_docs} text_bytes={validation_bytes}")


if __name__ == "__main__":
    main()
