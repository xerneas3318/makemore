"""Text generation from a trained GPT-2 checkpoint.

Loads a checkpoint saved by train-gpt-2.py and generates samples autoregressively.
Unlike the quick generate block in the training script, this primes the model with
the real prompt tokens (no zero/`!` padding) and grows the context as it decodes,
with temperature + top-k sampling.

Usage:
    python gpts/inference.py --prompt "Once upon a time" --num-samples 5 --max-new-tokens 100
"""
from dataclasses import dataclass
import argparse
import glob
import math
import os

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# model (mirrors the architecture in train-gpt-2.py)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

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
    vocab_size: int = 50304
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

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits


# -----------------------------------------------------------------------------
# generation

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None,
             block_size=1024, eot_token=None):
    """Autoregressively extend idx (B, T) by max_new_tokens, growing the context."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]                 # never exceed the context window
        logits = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
        if eot_token is not None and (next_id == eot_token).all():
            break
    return idx


def latest_checkpoint(ckpt_dir):
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no ckpt_*.pt in {ckpt_dir}")
    return ckpts[-1]


def main():
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data"))
    default_ckpt = latest_checkpoint(os.path.join(data_root, "checkpoints"))

    p = argparse.ArgumentParser(description="Generate text from a trained GPT-2 checkpoint")
    p.add_argument("--ckpt", default=default_ckpt, help="path to checkpoint .pt")
    p.add_argument("--prompt", default="Once upon a time", help="text prompt to condition on")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(args.seed)
    if device_type == "cuda":
        torch.cuda.manual_seed(args.seed)

    print(f"device: {device}")
    if device_type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    vocab_size = ckpt.get("vocab_size", 50304)
    print(f"loading {args.ckpt} (step {ckpt.get('step')}, vocab_size {vocab_size})")

    model = GPT(GPT2Config(vocab_size=vocab_size))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # 50256, GPT-2 <|endoftext|>
    prompt_ids = enc.encode(args.prompt)
    print(f"prompt: {args.prompt!r} -> {len(prompt_ids)} tokens\n")

    idx = torch.tensor([prompt_ids] * args.num_samples, dtype=torch.long, device=device)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device_type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with autocast_ctx:
        out = generate(
            model, idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            block_size=model.config.block_size,
            eot_token=eot,
        )

    for i in range(args.num_samples):
        text = enc.decode(out[i].tolist())
        print(f"===== sample {i + 1} =====")
        print(text)
        print()


if __name__ == "__main__":
    main()
