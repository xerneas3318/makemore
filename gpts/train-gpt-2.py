from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import tqdm
import os
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# -----------------------------------------------------------------------------
# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1   # torchrun sets RANK; -1 means plain run
if ddp:
    assert torch.cuda.is_available(), "DDP needs CUDA"
    init_process_group(backend='nccl')
    ddp_rank       = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0            # only rank 0 prints / saves
else:
    ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

device_type = 'cuda' if device.startswith('cuda') else 'cpu'
if master_process:
    print(f"device: {device}, ddp: {ddp}, ddp_rank: {ddp_rank}, ddp_local_rank: {ddp_local_rank}, ddp_world_size: {ddp_world_size}")
# -----------------------------------------------------------------------------
# data


with open('input.txt', 'r') as f:
    text = f.read()
data = text

enc = tiktoken.get_encoding("gpt2")
tr_tokens = enc.encode(data[:1000000])
tst_tokens = enc.encode(data[1000000:])

torch.set_float32_matmul_precision('high')

T = 1024
B = 8

total_batch_size = 524288 
assert total_batch_size % (B * T * ddp_world_size) == 0

grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total_batch_size: {total_batch_size}, B: {B}, T: {T}, ddp_world_size: {ddp_world_size}, grad_accum_steps: {grad_accum_steps}")


# buf = torch.tensor(tr_tokens, device=device, dtype=torch.long)
# N = (len(buf) - 1) // T
# x = buf[:N*T].view(N, T)
# y = buf[1:1+N*T].view(N, T)
# tr_ds = TensorDataset(x, y)
# if ddp:
#     tr_sampler = DistributedSampler(tr_ds)
#     tr_loader = DataLoader(tr_ds, batch_size=B, sampler=tr_sampler)
# else:
#     tr_loader = DataLoader(tr_ds, batch_size=B, shuffle=True)

# buf2 = torch.tensor(tst_tokens, device=device, dtype=torch.long)
# N2 = (len(buf2) - 1) // T
# x2 = buf2[:N2*T].view(N2, T)
# y2 = buf2[1:1+N2*T].view(N2, T)
# tst_ds = TensorDataset(x2, y2)
# if ddp:
#     tst_sampler = DistributedSampler(tst_ds)
#     tst_loader = DataLoader(tst_ds, batch_size=B, sampler=tst_sampler)
# else:
#     tst_loader = DataLoader(tst_ds, batch_size=B, shuffle=True)



def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split,
                 data_root="/mnt/datasets/edu_fineweb10B"):
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        shards = sorted(s for s in os.listdir(data_root) if split in s)
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert len(self.shards) > 0, f"no shards for split {split} in {data_root}"
        if master_process:
            print(f"found {len(self.shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank,
                              num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank,
                            num_processes=ddp_world_size, split="val")


# -----------------------------------------------------------------------------
# optimizer

max_lr = 6e-4 * 3


def configure_optimizers(model, weight_decay, lr):
    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    decay   = [p for p in param_dict.values() if p.dim() >= 2]   # weights, embeddings
    nodecay = [p for p in param_dict.values() if p.dim() <  2]   # biases, layernorms
    groups = [
        {'params': decay,   'weight_decay': weight_decay},
        {'params': nodecay, 'weight_decay': 0.0},
    ]
    if device_type == 'cuda':
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    else:
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


# -----------------------------------------------------------------------------
# model

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02/math.sqrt(2 * self.n_layer))

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
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # bias does not init to 0 by default in torch
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)   # token + position embeddings
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                                    # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# -----------------------------------------------------------------------------
# train

model = GPT(GPT2Config(vocab_size=50304))
model.to(device)

max_steps = 19073

optimizer = configure_optimizers(model, weight_decay=0.1, lr=max_lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    total_steps=max_steps,
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=25,
    final_div_factor=1e4,
)

model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model
tr_loss = []
model.train()

for step in tqdm(range(max_steps), disable=not master_process):
    last_step = (step == max_steps - 1)

    # val
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_accum = 0.0
            val_steps = 20
            for _ in range(val_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                val_accum += (loss / val_steps).detach()
        if ddp:
            dist.all_reduce(val_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"step {step:5d} | val loss {val_accum.item():.4f}")

    # 
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro == grad_accum_steps - 1)
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if master_process:
        tr_loss.append(loss_accum.item())
        if step % 100 == 0:
            print(f"step {step:5d} | loss {loss_accum.item():.4f} | norm {norm:.2f}") 
    

# -----------------------------------------------------------------------------
# eval
# if master_process:
#     raw_model.eval()
#     with torch.no_grad():
#         tr_loss_eval = []
#         tst_loss_eval = []
#         for x, y in tr_loader:
#             logits, loss = raw_model(x.to(device), y.to(device))
#             tr_loss_eval.append(loss.item())
#         for x, y in tst_loader:
#             logits, loss = raw_model(x.to(device), y.to(device))
#             tst_loss_eval.append(loss.item())

#         print(f"Train Loss: {sum(tr_loss_eval) / len(tr_loss_eval)}")
#         print(f"Test Loss: {sum(tst_loss_eval) / len(tst_loss_eval)}")

if master_process:
    plt.figure(figsize=(10, 5))

    plt.plot(tr_loss, color='lightsteelblue', linewidth=0.8, label='loss (per step)')

    k = 50
    #smoothy smoothy
    if len(tr_loss) >= k:
        smooth = np.convolve(tr_loss, np.ones(k) / k, mode='valid')
        x = np.arange(k - 1, len(tr_loss))
        plt.plot(x, smooth, color='crimson', linewidth=2, label=f'{k}-step average')

    plt.xlabel('optimizer step')
    plt.ylabel('train loss')
    plt.title('Training loss over time')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('saved loss.png')


# -----------------------------------------------------------------------------
# generate
if master_process:
    raw_model.eval()
    inp = 'Shakespeare:'
    tok = enc.encode(inp)
    for i in range(10):
        out = []
        context = tok + [0] * (32 - len(tok))
        for _ in range(50):
            logits, loss = raw_model(torch.tensor([context], device=device))
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            out.append(ix)
        print(enc.decode(out))

if ddp:
    destroy_process_group()