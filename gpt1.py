"""
gpt1 — script version of gpt1.ipynb.

Logic is identical to the notebook. The only additions are at the end:
  - saves model weights      -> gpt1_weights.pt
  - saves generated samples  -> gpt1_output.txt   (also includes train/val loss)
  - saves the training graph -> gpt1_training_curve.png

"""

import torch
import torch.nn.functional as F
import requests
import os
import random
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import tqdm
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- output file paths ----
WEIGHTS_PATH = "gpt1_weights.pt"
OUTPUT_PATH = "gpt1_output.txt"
GRAPH_PATH = "gpt1_training_curve.png"

# ---------------------------------------------------------------------------
# data download / load
# ---------------------------------------------------------------------------
path = "input.txt"
if not os.path.exists(path):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("Content-Length", 0))
    with open(path, "wb") as f:
        bar = tqdm(total=total, unit="B", unit_scale=True, desc="input.txt")
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
        bar.close()

with open(path) as f:
    text = f.read()

# print(len(text)
# print(text[:500])

# ---------------------------------------------------------------------------
# device + hyperparameters
# ---------------------------------------------------------------------------
if (device := torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")):
    print(f"Using device: {device}")

block_size = 64
epochs = 5000
iters = 5000
eval_iters = 500
lr = 0.0001
embd = 384
head = 100
batch = 1024
dropout = 0.2
num_heads = 8
num_blocks = 6

# ---------------------------------------------------------------------------
# vocab
# ---------------------------------------------------------------------------
chars = sorted(set(text))
vocab_size = len(chars)
print(chars)
def stoi(x):
    return chars.index(x) if isinstance(x, str) else chars[x]
_stoi = {c: i for i, c in enumerate(chars)}
def itos(x):
    return _stoi[x] if isinstance(x, str) else chars[x]

print("itos:", itos(1), itos(2), itos(3), itos(0), "\n")
print("stoi:", stoi('.'), stoi('a'), stoi('b'), stoi('c'), "/n")

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def build_dataset(data):
    X, Y = [], []
    for i in range(len(data) - block_size):
        X.append(data[i : i + block_size])
        Y.append(data[i+1 : i + block_size + 1])
    X = torch.stack(X)
    Y = torch.stack(Y)
    return X, Y

data = torch.tensor([stoi(c) for c in text])
n = int(0.9 * len(data))
train_data = data[:n].to(device)     # ~8.5 MB, trivial — keep on GPU
val_data   = data[n:].to(device)

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch,))
    x = torch.stack([d[i   : i+block_size]   for i in ix])
    y = torch.stack([d[i+1 : i+block_size+1] for i in ix])
    return x, y

# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, head_size, embd):
        super().__init__()
        self.key = nn.Linear(embd, head_size, bias=False)
        self.query = nn.Linear(embd, head_size, bias=False)
        self.value = nn.Linear(embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5 #BTC @ BCT => BTT
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.drop(wei)
        out = wei @ v
        return out


class multihead(nn.Module):
    def __init__(self, num_heads, head_size, embd):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, embd) for _ in range(num_heads)])
        self.proj = nn.Linear(embd, embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        stack = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.drop(self.proj(stack))


class feedForward(nn.Module):
    def __init__(self, embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embd, embd),
            nn.ReLU(),
            nn.Linear(embd, embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class attenB(nn.Module):
    def __init__(self, embd, nhead):
        super().__init__()
        self.ffw = feedForward(embd)
        self.sa = multihead(nhead, embd//nhead, embd)
        self.ln1 = nn.LayerNorm(embd)
        self.ln2 = nn.LayerNorm(embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embd)
        self.position_embedding = nn.Embedding(block_size, embd)
        #to change
        self.atten = nn.Sequential(
            *[attenB(embd, num_heads) for _ in range(num_blocks)]
        )
        self.ln = nn.LayerNorm(embd)
        self.lm = nn.Linear(embd, vocab_size)
        self.ffw = feedForward(embd)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.atten(x)
        return self.lm(self.ln(x))


model = LM().to(device)
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99995)

# ---------------------------------------------------------------------------
# eval helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            Xb, Yb = get_batch(split)
            logits = model(Xb)
            losses[k] = F.cross_entropy(logits.view(-1, vocab_size), Yb.view(-1))
        out[split] = losses.mean().item()
    model.train()
    return out

# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
model.train()
pbar = tqdm(range(iters), desc="train")
t_loss = []
for step in pbar:
    Xb, Yb = get_batch('train')
    logits = model(Xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), Yb.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    t_loss.append(loss.item())
    pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    if step % eval_iters == 0:
        print(step, loss.item(), scheduler.get_last_lr()[0])

# ---------------------------------------------------------------------------
# save weights
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), WEIGHTS_PATH)
print(f"saved weights -> {WEIGHTS_PATH}")

# ---------------------------------------------------------------------------
# save training curve
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(len(t_loss)), t_loss, label='train', linewidth=1)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training curve')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(GRAPH_PATH, dpi=150, bbox_inches='tight')
print(f"saved graph -> {GRAPH_PATH}")

# ---------------------------------------------------------------------------
# eval + generate, save output text
# ---------------------------------------------------------------------------
losses = estimate_loss()
print("train loss:", losses['train'], "val loss:", losses['val'])

samples = []
model.eval()
for i in range(10):
    out = []
    context = [0] * block_size
    for _ in range(50):
        logits = model(torch.tensor([context], device=device))
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(itos(ix))
    s = ''.join(out)
    print(s)
    samples.append(s)

with open(OUTPUT_PATH, "w") as f:
    f.write(f"train loss: {losses['train']}\n")
    f.write(f"val loss: {losses['val']}\n")
    f.write("\n--- samples ---\n")
    for s in samples:
        f.write(s + "\n")
print(f"saved output -> {OUTPUT_PATH}")
