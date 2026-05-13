import warnings
import pandas as pd
import torch
from transformers import AutoTokenizer

_CHUNK_SIZE = 512
_DATA_URL = (
    "https://raw.githubusercontent.com/slinderman/stats305b/winter2024"
    "/assignments/hw4/python_corpus_4M.csv"
)

PROMPT_NEWTON = """\
def newton(eta, N, X, y, gamma, beta=None):
  \"\"\"
  Performs Newton's method on the negative average log likelihood with an
  l2 regularization term

  beta: torch.Tensor, of shape (teams)
  X: torch.Tensor, the covariate matrix, of shape (-1, teams)
  y: torch.Tensor, the response vector, of shape (teams)
  gamma: float, the scale parameter for the regularization
  beta: torch.Tensor, the starting point for gradient descent, if specified
  \"\"\"

  if beta is None:
    beta = torch.randn(X.shape[1])
  else:
    beta = torch.clone(beta)

  loss = []

  for i in tqdm(range(N)):"""

PROMPT_COSINE = """\
import torch
import torch.nn.functional as F


def normalize(x, axis=-1):
    \"\"\"Performs L2-Norm.\"\"\"
    num = x
    denom = torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12
    return num / denom

def euclidean_dist(x, y):
    \"\"\"Computes Euclidean distance.\"\"\"
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(x, 2).sum(1, keepdim=True).expand(m, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def cosine_dist(x, y):"""


def _chunk_string(string, size):
    return [string[i : i + size] for i in range(0, len(string), size)]


def load_data(train_frac=0.9):
    """Download the Python corpus, tokenize it, and return train/val splits.

    Returns:
        tokenizer, train_data, val_data, vocab_size, from_codebert, to_codebert
    """
    tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeBERT-base")

    print("Downloading corpus…")
    raw_data = pd.read_csv(_DATA_URL, header=None)

    warnings.filterwarnings("ignore")
    tokens = torch.tensor([], dtype=torch.long)
    for _, row in raw_data.iterrows():
        text = row[0]
        chunks = _chunk_string(text, _CHUNK_SIZE)
        n = len(chunks)
        for idx, chunk in enumerate(chunks):
            new_tokens = torch.tensor(
                tokenizer.encode(chunk, add_special_tokens=True, truncation=True)
            )
            if idx == 0:
                tokens = torch.cat([tokens, new_tokens[:-1]])
            elif idx == n - 1:
                tokens = torch.cat([tokens, new_tokens[1:]])
            else:
                tokens = torch.cat([tokens, new_tokens[1:-1]])

    # Build a compact vocabulary that includes all tokens from the eval prompts.
    prompt_tokens = torch.cat([
        torch.tensor(tokenizer.encode(PROMPT_NEWTON, add_special_tokens=True)),
        torch.tensor(tokenizer.encode(PROMPT_COSINE, add_special_tokens=True)),
    ])
    unique_tokens = torch.unique(torch.cat([tokens, prompt_tokens]))
    from_codebert = {elem.item(): idx for idx, elem in enumerate(unique_tokens)}
    to_codebert = {idx: elem.item() for elem, idx in from_codebert.items()}
    vocab_size = len(unique_tokens)

    tokens_remapped = torch.tensor([from_codebert[t.item()] for t in tokens])
    n = int(train_frac * len(tokens_remapped))
    train_data = tokens_remapped[:n]
    val_data = tokens_remapped[n:]

    print(
        f"Loaded {len(tokens_remapped):,} tokens | "
        f"vocab={vocab_size:,} | train={len(train_data):,} | val={len(val_data):,}"
    )
    return tokenizer, train_data, val_data, vocab_size, from_codebert, to_codebert


def get_batch(split, train_data, val_data, context_window_size, device, batch_size=32):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - context_window_size, (batch_size,))
    x = torch.stack([data[i : i + context_window_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + context_window_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, context_window_size, device):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, train_data, val_data, context_window_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
