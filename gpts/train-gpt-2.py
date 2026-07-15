from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# data

with open('input.txt', 'r') as f:
    text = f.read()
data = text

enc = tiktoken.get_encoding("gpt2")
tr_tokens = enc.encode(data[:1000000])
tst_tokens = enc.encode(data[1000000:])

torch.set_float32_matmul_precision('high')

T = 32
B = 16

device = 'cuda' if torch.cuda.is_available() else 'cpu'

buf = torch.tensor(tr_tokens, device=device, dtype=torch.long)
N = (len(buf) - 1) // T
x = buf[:N*T].view(N, T)
y = buf[1:1+N*T].view(N, T)
tr_ds = TensorDataset(x, y)
tr_loader = DataLoader(tr_ds, batch_size=B, shuffle=True)

buf2 = torch.tensor(tst_tokens, device=device, dtype=torch.long)
N2 = (len(buf2) - 1) // T
x2 = buf2[:N2*T].view(N2, T)
y2 = buf2[1:1+N2*T].view(N2, T)
tst_ds = TensorDataset(x2, y2)
tst_loader = DataLoader(tst_ds, batch_size=B, shuffle=True)


# -----------------------------------------------------------------------------
# optimizer

max_lr = 6e-4
min_lr = max_lr * 0.1


def configure_optimizers(model, weight_decay, lr):
    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    decay   = [p for p in param_dict.values() if p.dim() >= 2]   # weights, embeddings
    nodecay = [p for p in param_dict.values() if p.dim() <  2]   # biases, layernorms
    groups = [
        {'params': decay,   'weight_decay': weight_decay},
        {'params': nodecay, 'weight_decay': 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=True)


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

epochs = 10
step = 0

optimizer = configure_optimizers(model, weight_decay=0.1, lr=max_lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    total_steps=epochs * len(tr_loader),
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=25,
    final_div_factor=1e4,
)

model = torch.compile(model)
model.train()

for epoch in range(epochs):
    e_loss = []
    for x, y in tr_loader:
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x.to(device), y.to(device))
        loss.backward()
        e_loss.append(loss.item())
        optimizer.step()
        scheduler.step()
        step += 1
    print(f"Epoch {epoch+1}/{epochs} - Loss: {sum(e_loss)/len(e_loss)}")


# -----------------------------------------------------------------------------
# eval

model.eval()
with torch.no_grad():
    tr_loss = []
    tst_loss = []
    for x, y in tr_loader:
        logits, loss = model(x.to(device), y.to(device))
        tr_loss.append(loss.item())
    for x, y in tst_loader:
        logits, loss = model(x.to(device), y.to(device))
        tst_loss.append(loss.item())

    print(f"Train Loss: {sum(tr_loss) / len(tr_loss)}")
    print(f"Test Loss: {sum(tst_loss) / len(tst_loss)}")


# -----------------------------------------------------------------------------
# generate

model.eval()
inp = 'Shakespeare:'
tok = enc.encode(inp)

for i in range(10):
    out = []
    context = tok + [0] * (32 - len(tok))
    for _ in range(50):
        logits, loss = model(torch.tensor([context], device=device))
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)
    print(enc.decode(out))
