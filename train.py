"""Train and compare baseline vs low-rank transformer on the Python code corpus."""

import argparse

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from data import (
    PROMPT_COSINE,
    PROMPT_NEWTON,
    estimate_loss,
    get_batch,
    load_data,
)
from models import LowRankTransformerLM, TransformerLM

# ── Hyperparameters ────────────────────────────────────────────────────────────
CONTEXT_WINDOW = 256
EMBED_SIZE = 384
NUM_HEADS = 6
N_LAYERS = 6
LEARNING_RATE = 1e-4
TRAIN_ITERS = 2000
EVAL_INTERVAL = 200
EVAL_ITERS = 200
DEFAULT_RANK = int(EMBED_SIZE * 1.5)  # 576; break-even is 4 * EMBED_SIZE = 1536
# ──────────────────────────────────────────────────────────────────────────────


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, train_data, val_data, device, iters=TRAIN_ITERS):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    history = []

    for it in tqdm(range(iters)):
        if it % EVAL_INTERVAL == 0 or it == iters - 1:
            losses = estimate_loss(
                model, train_data, val_data, EVAL_ITERS, CONTEXT_WINDOW, device
            )
            tqdm.write(
                f"step {it:4d}: train={losses['train']:.4f}  val={losses['val']:.4f}"
            )

        xb, yb = get_batch("train", train_data, val_data, CONTEXT_WINDOW, device)
        _, loss = model(xb, yb)
        history.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return history


def evaluate(model, train_data, val_data, device):
    losses = estimate_loss(
        model, train_data, val_data, EVAL_ITERS, CONTEXT_WINDOW, device
    )
    n_params = count_parameters(model)
    size_mb = n_params * 4 / (1024**2)
    print(f"  Parameters : {n_params:,}  ({size_mb:.1f} MB)")
    print(f"  Train PPL  : {torch.exp(losses['train']).item():.2f}")
    print(f"  Val PPL    : {torch.exp(losses['val']).item():.2f}")
    return losses


def generate_samples(model, tokenizer, from_codebert, to_codebert, device):
    model.eval()

    def encode(text):
        ids = [from_codebert[t] for t in tokenizer.encode(text)[:-1]]
        return torch.tensor(ids, device=device).unsqueeze(0)

    def decode(ids):
        return tokenizer.decode([to_codebert[i] for i in ids])

    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\n--- Unconditional ---")
    print(decode(model.generate(start, max_new_tokens=CONTEXT_WINDOW)[0].tolist()))

    for name, prompt in [("newton", PROMPT_NEWTON), ("cosine", PROMPT_COSINE)]:
        ctx = encode(prompt)
        print(f"\n--- Conditional ({name}) ---")
        print(decode(model.generate(ctx, max_new_tokens=CONTEXT_WINDOW)[0].tolist()))

    model.train()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["baseline", "lowrank", "both"],
        default="both",
        help="which model(s) to train",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=DEFAULT_RANK,
        help="bottleneck rank for low-rank feedforward layers",
    )
    parser.add_argument("--iters", type=int, default=TRAIN_ITERS)
    parser.add_argument(
        "--generate", action="store_true", help="generate text samples after training"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading data…")
    tokenizer, train_data, val_data, vocab_size, from_codebert, to_codebert = load_data()
    train_data, val_data = train_data.to(device), val_data.to(device)

    results = {}

    if args.model in ("baseline", "both"):
        print("\n=== Baseline TransformerLM ===")
        model = TransformerLM(
            vocab_size, CONTEXT_WINDOW, EMBED_SIZE, NUM_HEADS, N_LAYERS
        ).to(device)
        history = train(model, train_data, val_data, device, args.iters)
        losses = evaluate(model, train_data, val_data, device)
        results["baseline"] = {"history": history, "losses": losses, "model": model}

    if args.model in ("lowrank", "both"):
        print(f"\n=== LowRankTransformerLM (rank={args.rank}) ===")
        model = LowRankTransformerLM(
            vocab_size, CONTEXT_WINDOW, args.rank, EMBED_SIZE, NUM_HEADS, N_LAYERS
        ).to(device)
        history = train(model, train_data, val_data, device, args.iters)
        losses = evaluate(model, train_data, val_data, device)
        results["lowrank"] = {"history": history, "losses": losses, "model": model}

    # Loss curves
    plt.figure(figsize=(8, 4))
    for name, res in results.items():
        plt.plot(res["history"], label=name, alpha=0.8)
    plt.xlabel("Iteration")
    plt.ylabel("Training loss (cross-entropy)")
    plt.title("Baseline vs low-rank transformer")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=150)
    print("\nSaved loss_curves.png")

    if args.generate and results:
        name, res = next(iter(results.items()))
        print(f"\nGenerating samples from {name} model…")
        generate_samples(
            res["model"], tokenizer, from_codebert, to_codebert, device
        )


if __name__ == "__main__":
    main()
