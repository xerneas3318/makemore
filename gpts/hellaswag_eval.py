"""
Standalone HellaSwag evaluation for a trained GPT-2 checkpoint.

Scores a base language model on the HellaSwag validation set by the standard
completion-loss method: for each of the 4 candidate endings, run the model over
(context + ending), take the length-normalized cross-entropy over just the
ending tokens, and predict the ending with the lowest loss. Accuracy is the
fraction of examples where that prediction matches the labeled answer.

Random baseline is 25%. GPT-2 124M lands around 29-30%.

Usage:
    python hellaswag_eval.py                       # latest checkpoint, full val set
    python hellaswag_eval.py --ckpt path/to.pt     # a specific checkpoint
    python hellaswag_eval.py --limit 1000          # quick noisy estimate
"""

import os
import glob
import json
import math
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import requests
from tqdm import tqdm

# -----------------------------------------------------------------------------
# paths (mirror train-gpt-2.py: data/ lives one level up from gpts/)

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, "data"))
CKPT_DIR = os.path.join(DATA_ROOT, "checkpoints")
DATA_CACHE_DIR = os.path.join(HERE, "hellaswag")

HELLASWAG_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"

enc = tiktoken.get_encoding("gpt2")

# -----------------------------------------------------------------------------
# model (identical architecture to train-gpt-2.py so the checkpoint loads)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B, nh, T, hs
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B, nh, T, hs
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B, nh, T, hs
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # B, T, C
        out = self.c_proj(out)
        return out


class MLP(nn.Module):  # feedforward
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# -----------------------------------------------------------------------------
# hellaswag data


def download(fname):
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    if os.path.exists(fname):
        return
    print(f"downloading HellaSwag val set -> {fname}")
    resp = requests.get(HELLASWAG_URL, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as f, tqdm(total=total, unit="iB", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            bar.update(f.write(chunk))


def iterate_examples():
    fname = os.path.join(DATA_CACHE_DIR, "hellaswag_val.jsonl")
    download(fname)
    with open(fname, "r") as f:
        for line in f:
            yield json.loads(line)


def render_example(example):
    """Return (tokens, mask, label). tokens/mask are (4, T); mask==1 marks ending tokens."""
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    ctx_tokens = enc.encode(ctx)
    tok_rows, mask_rows = [], []
    for end in endings:
        end_tokens = enc.encode(" " + end)  # leading space: GPT-2 BPE convention
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (t, m) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(t)] = torch.tensor(t)
        mask[i, :len(m)] = torch.tensor(m)
    return tokens, mask, label


@torch.no_grad()
def get_most_likely_row(tokens, mask, logits):
    # autoregressive loss at every position, then keep only the ending region
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_tokens.view(-1),
        reduction="none",
    ).view(tokens.size(0), -1)

    shift_mask = mask[..., 1:].contiguous()  # mask must shift the same way as tokens
    masked = losses * shift_mask
    avg_loss = masked.sum(dim=1) / shift_mask.sum(dim=1)  # length-normalized
    return avg_loss.argmin().item()  # index of the lowest-loss ending


# -----------------------------------------------------------------------------
# checkpoint loading + eval


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    vocab_size = ckpt.get("vocab_size", 50304)
    model = GPT(GPT2Config(vocab_size=vocab_size))
    # older checkpoints carry a dead causal-mask buffer (transformer.h.*.attn.bias)
    # that the fixed model no longer registers; drop it so the rest loads strictly.
    state = {k: v for k, v in ckpt["model"].items() if not k.endswith(".attn.bias")}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    step = ckpt.get("step", "?")
    print(f"loaded {ckpt_path} (step {step}, vocab_size {vocab_size})")
    return model


def evaluate(model, device, device_type, limit=None):
    num_correct, num_total = 0, 0
    for i, example in enumerate(tqdm(iterate_examples(), desc="hellaswag")):
        if limit is not None and i >= limit:
            break
        tokens, mask, label = render_example(example)
        tokens, mask = tokens.to(device), mask.to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                                enabled=(device_type == "cuda")):
                logits, _ = model(tokens)
            # score on-device; only the single int result comes back to the host
            pred = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct += int(pred == label)
    return num_correct, num_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="checkpoint path (default: latest in data/checkpoints)")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu (default: auto)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only evaluate the first N examples (default: all)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.set_float32_matmul_precision("high")

    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_*.pt")))
        assert ckpts, f"no checkpoints found in {CKPT_DIR}"
        ckpt_path = ckpts[-1]

    model = load_model(ckpt_path, device)
    num_correct, num_total = evaluate(model, device, device_type, limit=args.limit)
    acc = num_correct / num_total
    print(f"HellaSwag accuracy: {num_correct}/{num_total} = {acc:.4f}")


if __name__ == "__main__":
    main()
