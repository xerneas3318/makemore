"""Self-contained Gradio demo for the mlp2-gpu makemore model.

Run with:
    uv run python demo.py            # local + public share URL
    uv run python demo.py --no-share # local only
"""
from __future__ import annotations
import argparse
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F


BLOCK_SIZE = 7
VOCAB = ["."] + list("abcdefghijklmnopqrstuvwxyz")
VOCAB_SIZE = len(VOCAB)
EMBED = 10
HIDDEN = 200

stoi = {c: i for i, c in enumerate(VOCAB)}
itos = {i: c for i, c in enumerate(VOCAB)}


class MLP(torch.nn.Module):
    """Same arch as mlp2-gpu.ipynb: emb(27,10) -> 70 -> tanh(200) -> 27."""
    def __init__(self) -> None:
        super().__init__()
        self.C = torch.nn.Parameter(torch.randn((VOCAB_SIZE, EMBED)))
        self.W1 = torch.nn.Parameter(torch.randn((BLOCK_SIZE * EMBED, HIDDEN)) / (5 / 3) * (BLOCK_SIZE * EMBED) ** 0.5)
        self.b1 = torch.nn.Parameter(torch.randn(HIDDEN) * 0.1)
        self.W2 = torch.nn.Parameter(torch.randn((HIDDEN, VOCAB_SIZE)) * 0.01)
        self.b2 = torch.nn.Parameter(torch.zeros(VOCAB_SIZE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.C[x].view(-1, BLOCK_SIZE * EMBED)
        h = torch.tanh(emb @ self.W1 + self.b1)
        return h @ self.W2 + self.b2


@torch.no_grad()
def sample(model: MLP, n: int, temperature: float, seed: int | None) -> list[str]:
    model.eval()
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    names: list[str] = []
    for _ in range(n):
        ctx = [0] * BLOCK_SIZE
        out: list[str] = []
        while True:
            logits = model(torch.tensor([ctx]))
            probs = F.softmax(logits / max(temperature, 1e-3), dim=1)
            ix = int(torch.multinomial(probs, num_samples=1, generator=gen).item())
            ctx = ctx[1:] + [ix]
            if ix == 0 or len(out) > 30:
                break
            out.append(itos[ix])
        names.append("".join(out))
    return names


HERE = Path(__file__).resolve().parent
WEIGHTS_CANDIDATES = [
    HERE / "mlp.pt",
    HERE.parent / "makemore-space" / "mlp.pt",
]
WEIGHTS = next((p for p in WEIGHTS_CANDIDATES if p.exists()), None)
if WEIGHTS is None:
    raise SystemExit(
        "mlp.pt not found. Searched:\n  " +
        "\n  ".join(str(p) for p in WEIGHTS_CANDIDATES)
    )

model = MLP()
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
model.eval()
print(f"loaded weights: {WEIGHTS}")


def generate(n: int, temperature: float, seed: int) -> str:
    n = max(1, min(int(n), 100))
    temperature = max(0.1, min(float(temperature), 2.0))
    s = None if int(seed) == 0 else int(seed)
    return "\n".join(sample(model, n=n, temperature=temperature, seed=s))


with gr.Blocks(title="makemore — name generator") as demo:
    gr.Markdown(
        "# makemore — name generator\n"
        "Character-level MLP (block 7, embed 10, hidden 200) trained on ~32k names. "
        "Dev loss ~2.21."
    )
    with gr.Row():
        n_in = gr.Slider(1, 50, value=10, step=1, label="how many")
        t_in = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="temperature")
        s_in = gr.Number(value=0, precision=0, label="seed (0 = random)")
    out = gr.Textbox(label="names", lines=12)
    btn = gr.Button("generate", variant="primary")
    btn.click(generate, inputs=[n_in, t_in, s_in], outputs=out)
    demo.load(generate, inputs=[n_in, t_in, s_in], outputs=out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-share", action="store_true", help="local only, no public URL")
    args = p.parse_args()
    demo.launch(share=not args.no_share)
