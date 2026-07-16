"""Gradio demo for the trained GPT-2 124M model.

Loads the checkpoint once and serves an interactive text-generation UI.

    python3 gpts/app.py                 # local, http://0.0.0.0:7860
    python3 gpts/app.py --share         # public gradio.live link
"""
import argparse
import os

import gradio as gr
import tiktoken
import torch

from inference import GPT, GPT2Config, generate, latest_checkpoint

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data"))

# -----------------------------------------------------------------------------
# load model once at startup

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = latest_checkpoint(os.path.join(DATA_ROOT, "checkpoints"))

print(f"loading {CKPT_PATH} on {DEVICE} ...")
_ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
_vocab = _ckpt.get("vocab_size", 50304)
MODEL = GPT(GPT2Config(vocab_size=_vocab))
MODEL.load_state_dict(_ckpt["model"])
MODEL.to(DEVICE).eval()
ENC = tiktoken.get_encoding("gpt2")
STEP = _ckpt.get("step")
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
print(f"ready (step {STEP}, device {DEVICE})")


def run(prompt, max_new_tokens, temperature, top_k, num_samples, seed):
    if not prompt.strip():
        prompt = "<|endoftext|>"
    torch.manual_seed(int(seed))
    if DEVICE == "cuda":
        torch.cuda.manual_seed(int(seed))

    prompt_ids = ENC.encode(prompt, allowed_special={"<|endoftext|>"})
    idx = torch.tensor([prompt_ids] * int(num_samples), dtype=torch.long, device=DEVICE)

    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if DEVICE == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with ctx:
        out = generate(
            MODEL, idx,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k) if int(top_k) > 0 else None,
            block_size=MODEL.config.block_size,
            eot_token=ENC.eot_token,
        )

    samples = [ENC.decode(out[i].tolist()) for i in range(int(num_samples))]
    return "\n\n" + ("\n\n" + "─" * 60 + "\n\n").join(
        f"▌ Sample {i + 1}\n\n{s}" for i, s in enumerate(samples)
    )


with gr.Blocks(title="GPT-2 124M — edu-FineWeb 10B") as demo:
    gr.Markdown(
        f"""
        # GPT-2 124M — edu-FineWeb 10B
        A 124M-parameter GPT-2 trained from scratch on 10B tokens of edu-FineWeb.
        **Checkpoint:** step {STEP} · val loss ≈ 3.10 · running on **{GPU_NAME}**.

        Enter a prompt and the model continues it. This is a small base model
        (not instruction-tuned), so it *continues* text rather than answering questions.
        """
    )
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt", value="Once upon a time", lines=3,
                placeholder="Type a prompt for the model to continue...",
            )
            with gr.Row():
                max_new_tokens = gr.Slider(8, 512, value=120, step=8, label="Max new tokens")
                num_samples = gr.Slider(1, 6, value=2, step=1, label="Samples")
            with gr.Row():
                temperature = gr.Slider(0.1, 1.5, value=0.9, step=0.05, label="Temperature")
                top_k = gr.Slider(0, 200, value=50, step=5, label="Top-k (0 = off)")
            seed = gr.Number(value=1337, label="Seed", precision=0)
            btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=3):
            output = gr.Textbox(label="Generated text", lines=22)

    gr.Examples(
        examples=[
            ["Once upon a time", 120, 0.9, 50, 2, 1337],
            ["The most important scientific discovery of the century was", 120, 0.8, 50, 2, 42],
            ["Here is a simple recipe for bread:", 120, 0.85, 50, 2, 7],
            ["In the year 2050, cities will", 120, 1.0, 50, 2, 100],
        ],
        inputs=[prompt, max_new_tokens, temperature, top_k, num_samples, seed],
    )

    btn.click(run, [prompt, max_new_tokens, temperature, top_k, num_samples, seed], output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true", help="create a public gradio.live link")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
