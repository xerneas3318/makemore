"""
FineWeb-Edu data prep (for GPT-2 style pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

Downloads FineWeb-Edu, tokenizes it with the GPT-2 BPE tokenizer, and writes the
tokens to disk as fixed-size .npy shards (uint16). Shard 0 is the validation
split, the rest are training.

Run:
    uv run python gpts/fineweb.py                       # defaults: sample-10BT, 100M-token shards
    uv run python gpts/fineweb.py --shard-size 50000000 # smaller shards
    uv run python gpts/fineweb.py --output-dir ../data/edu_fineweb10B

Needs the `datasets` package:  uv add datasets
"""

import os

# default data location: a `data/` folder at the repo root (one level up from
# this script's `gpts/` dir), resolved absolutely so it works from any cwd.
DEFAULT_DATA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "data")
)

# Route ALL Hugging Face caches onto the big data disk *before* importing
# datasets/huggingface_hub. Otherwise HF downloads default to ~/.cache on the
# small root filesystem and fill it (that caused OSError: No space left on
# device mid-download). HF_HOME covers the hub download cache, which
# load_dataset's own cache_dir does NOT control.
os.environ.setdefault("HF_HOME", os.path.join(DEFAULT_DATA_ROOT, "hf_home"))

# Xet streaming was stalling in this environment; force plain HTTPS transfers.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import time
import argparse
import multiprocessing as mp

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


# the tokenizer is created once at module scope so every worker process
# (fork *or* spawn) has it available without re-passing it through the pool.
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # <|endoftext|> delimits documents


def tokenize(doc):
    """Tokenize a single document -> np.uint16 array, prefixed with the EOT token."""
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token id too large for uint16"
    return tokens_np.astype(np.uint16)


def parse_args():
    p = argparse.ArgumentParser(description="Download + tokenize FineWeb-Edu into .npy shards.")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu", help="HF dataset id")
    p.add_argument("--name", default="sample-10BT", help="dataset config / subset name")
    p.add_argument("--shard-size", type=int, default=int(1e8), help="tokens per shard (default 100M)")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                   help="base data dir; shards + HF cache go under here "
                        "(default: <repo>/data)")
    p.add_argument("--output-dir", default=None,
                   help="where to write shards (default: <data-root>/edu_fineweb10B)")
    p.add_argument("--cache-dir", default=None,
                   help="HF datasets download cache (default: <data-root>/hf_cache)")
    p.add_argument("--nprocs", type=int, default=None,
                   help="tokenizer worker processes (default: cpu_count // 2)")
    return p.parse_args()


def main():
    args = parse_args()

    # everything big lands under the data dir: both the tokenized shards and the
    # raw HF download cache (~27 GB) which otherwise defaults to ~/.cache.
    output_dir = args.output_dir or os.path.join(args.data_root, "edu_fineweb10B")
    cache_dir = args.cache_dir or os.path.join(args.data_root, "hf_cache")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    nprocs = args.nprocs or max(1, os.cpu_count() // 2)
    shard_size = args.shard_size

    print(f"dataset:     {args.dataset} ({args.name})")
    print(f"output dir:  {output_dir}")
    print(f"cache dir:   {cache_dir}")
    print(f"shard size:  {shard_size:,} tokens")
    print(f"workers:     {nprocs}")

    # streaming load: pull + tokenize documents on the fly instead of staging the
    # whole dataset to disk as parquet + arrow (~70 GB) first. This keeps disk use
    # low (only the output shards) and avoids the slow arrow-build step. Trade-off:
    # len(fw) is unknown, so the progress bar counts docs without a total/ETA.
    fw = load_dataset(args.dataset, name=args.name, split="train",
                      cache_dir=cache_dir, streaming=True)

    def save_shard(idx, n_tokens):
        split = "val" if idx == 0 else "train"
        path = os.path.join(output_dir, f"edufineweb_{split}_{idx:06d}")
        np.save(path, buffer[:n_tokens])
        return path

    buffer = np.empty((shard_size,), dtype=np.uint16)  # preallocated current shard
    count = 0            # tokens currently in the buffer
    total_tokens = 0     # tokens seen across the whole run
    shard_index = 0
    start = time.time()

    with mp.Pool(nprocs) as pool:
        bar = tqdm(unit="doc", desc="tokenizing",
                   dynamic_ncols=True, smoothing=0.05)

        for tokens in pool.imap_unordered(tokenize, fw, chunksize=256):
            total_tokens += len(tokens)

            if count + len(tokens) < shard_size:
                # fits in the current shard
                buffer[count:count + len(tokens)] = tokens
                count += len(tokens)
            else:
                # fill the rest of this shard, flush it, carry the leftover into the next
                remainder = shard_size - count
                buffer[count:count + remainder] = tokens[:remainder]
                save_shard(shard_index, shard_size)
                shard_index += 1
                leftover = len(tokens) - remainder
                buffer[0:leftover] = tokens[remainder:]
                count = leftover

            bar.update(1)
            if bar.n % 2000 == 0:
                elapsed = time.time() - start
                bar.set_postfix(
                    shards=shard_index,
                    tokens=f"{total_tokens / 1e9:.2f}B",
                    rate=f"{total_tokens / max(elapsed, 1e-9) / 1e6:.1f}M tok/s",
                )

        # flush whatever is left in the buffer as the final shard
        if count != 0:
            save_shard(shard_index, count)
            shard_index += 1

        bar.close()

    elapsed = time.time() - start
    print(f"\ndone: {shard_index} shards, {total_tokens:,} tokens in {elapsed / 60:.1f} min "
          f"({total_tokens / max(elapsed, 1e-9) / 1e6:.2f}M tok/s average)")


if __name__ == "__main__":
    main()
