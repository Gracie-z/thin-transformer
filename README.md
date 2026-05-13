# Low-Rank Transformers for Code Generation

Experiments comparing a standard GPT-style transformer against a variant where the feedforward sublayer uses a low-rank weight factorization, trained on a Python code corpus.

## Method

Standard transformer feedforward blocks use a two-layer MLP with a 4× hidden expansion (8D² parameters per layer). We replace this with a rank-r bottleneck:

```
output = relu(x W₁) W₂,   W₁ ∈ ℝ^{D×r},  W₂ ∈ ℝ^{r×D}
```

Parameter cost: 2Dr vs 8D². Break-even at r = 4D; below that, the low-rank layer is smaller.

With D = 384 and r = 576 (default), each feedforward block uses **37% fewer parameters** than the standard version.

## Setup

```bash
pip install -r requirements.txt
```

Requires a GPU (CUDA). Training on the full corpus (~4 M tokens) takes roughly 30 min per model on a T4.

## Usage

Train and compare both models:

```bash
python train.py --model both --iters 2000
```

Train only the low-rank model with a custom rank:

```bash
python train.py --model lowrank --rank 300 --iters 2000
```

Add `--generate` to print text samples after training.

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `both` | `baseline`, `lowrank`, or `both` |
| `--rank` | `576` | Bottleneck rank for low-rank FFN |
| `--iters` | `2000` | Training iterations |
| `--generate` | off | Generate text samples after training |

Outputs a `loss_curves.png` with training loss for each model.

## Files

| File | Description |
|------|-------------|
| `data.py` | Data loading, tokenization, vocabulary construction, data utilities |
| `models.py` | `TransformerLM`, `LowRankTransformerLM`, and shared components |
| `train.py` | Training loop, evaluation, and text generation |
