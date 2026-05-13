"""Knowledge distillation: train a student TransformerLM to mimic codeparrot-small.

Loss (Hinton et al., 2015):
    L = α * L_NLL  +  (1 - α) * T² * KL( softmax(s/T) ‖ softmax(t/T) )

where s and t are the student and teacher logits respectively.
"""

import argparse
from itertools import product

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import TransformerLM

# ── Hyperparameters ────────────────────────────────────────────────────────────
CONTEXT_WINDOW = 256
EMBED_SIZE = 384
NUM_HEADS = 6
N_LAYERS = 6
LEARNING_RATE = 1e-4
TRAIN_ITERS = 1000
EVAL_INTERVAL = 200
EVAL_ITERS = 200

TEACHER_NAME = "codeparrot/codeparrot-small"
TRAIN_SIZE = 50_000
VAL_SIZE = 10_000
# ──────────────────────────────────────────────────────────────────────────────


def load_data():
    """Load and tokenize code_search_net using the CodeParrot tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    dataset = load_dataset("code_search_net", "python")

    def tokenize(rows):
        ids = [tokenizer.convert_tokens_to_ids(row) for row in rows]
        flat = [x for xs in ids for x in xs]
        return torch.tensor(flat)

    train_tokens = tokenize(dataset["train"]["func_code_tokens"][:TRAIN_SIZE])
    val_tokens = tokenize(dataset["validation"]["func_code_tokens"][:VAL_SIZE])
    vocab_size = tokenizer.vocab_size

    print(
        f"Train tokens: {len(train_tokens):,} | "
        f"Val tokens: {len(val_tokens):,} | "
        f"Vocab: {vocab_size:,}"
    )
    return tokenizer, train_tokens, val_tokens, vocab_size


def get_batch(split, train_data, val_data, device, batch_size=32):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - CONTEXT_WINDOW, (batch_size,))
    x = torch.stack([data[i : i + CONTEXT_WINDOW] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + CONTEXT_WINDOW + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, device):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split, train_data, val_data, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def distillation_loss(student_logits, teacher_logits, targets, alpha, temperature):
    """Return (combined teaching loss, raw student NLL loss)."""
    nll = F.cross_entropy(
        student_logits.view(-1, student_logits.shape[-1]), targets.view(-1)
    )
    kl = nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
    )
    combined = alpha * nll + (1.0 - alpha) * (temperature**2) * kl
    return combined, nll


def train_one(vocab_size, train_data, val_data, teacher, alpha, temperature, device):
    student = TransformerLM(
        vocab_size, CONTEXT_WINDOW, EMBED_SIZE, NUM_HEADS, N_LAYERS
    ).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE)

    stu_history, teach_history = [], []

    for it in tqdm(range(TRAIN_ITERS), desc=f"α={alpha} T={temperature}"):
        if it % EVAL_INTERVAL == 0 or it == TRAIN_ITERS - 1:
            losses = estimate_loss(student, train_data, val_data, device)
            tqdm.write(
                f"  step {it:4d}: train={losses['train']:.4f}  val={losses['val']:.4f}"
            )

        xb, yb = get_batch("train", train_data, val_data, device)
        student_logits, _ = student(xb, yb)

        with torch.no_grad():
            teacher_logits = teacher(xb).logits

        loss, nll = distillation_loss(student_logits, teacher_logits, yb, alpha, temperature)
        stu_history.append(nll.item())
        teach_history.append(loss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final = estimate_loss(student, train_data, val_data, device)
    return student, stu_history, teach_history, final["val"].item()


def plot_run(stu_history, teach_history, alpha, temperature, path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(stu_history, color="blue")
    axs[0].set(
        title=f"Student loss  α={alpha} T={temperature}",
        xlabel="Iteration",
        ylabel="NLL loss",
    )
    axs[1].plot(teach_history, color="green")
    axs[1].set(
        title=f"Teaching loss  α={alpha} T={temperature}",
        xlabel="Iteration",
        ylabel="Teaching loss",
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7],
        help="weight on NLL loss (1 = no distillation)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        nargs="+",
        default=[0.8, 1.0, 2.0, 3.0],
        help="softmax temperature for distillation",
    )
    parser.add_argument("--iters", type=int, default=TRAIN_ITERS)
    args = parser.parse_args()

    global TRAIN_ITERS
    TRAIN_ITERS = args.iters

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading data…")
    tokenizer, train_data, val_data, vocab_size = load_data()
    train_data, val_data = train_data.to(device), val_data.to(device)

    print(f"Loading teacher ({TEACHER_NAME})…")
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_NAME).to(device)
    teacher.eval()

    results = {}

    for alpha, temperature in product(args.alpha, args.temperature):
        print(f"\n=== α={alpha}  T={temperature} ===")
        student, stu_hist, teach_hist, val_loss = train_one(
            vocab_size, train_data, val_data, teacher, alpha, temperature, device
        )
        results[(alpha, temperature)] = val_loss
        plot_run(stu_hist, teach_hist, alpha, temperature, f"distill_a{alpha}_t{temperature}.png")
        print(f"  Final val loss: {val_loss:.4f}")
        student.cpu()
        torch.cuda.empty_cache()

    # Summary table
    print("\n=== Validation Loss Grid ===")
    temps = sorted({t for _, t in results})
    alphas = sorted({a for a, _ in results})
    header = f"{'α \\ T':>8}" + "".join(f"{t:>10.1f}" for t in temps)
    print(header)
    for a in alphas:
        row = f"{a:>8.1f}" + "".join(f"{results[(a, t)]:>10.4f}" for t in temps)
        print(row)


if __name__ == "__main__":
    main()
