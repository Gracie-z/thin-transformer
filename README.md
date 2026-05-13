# Slimming the Transformer

Class project (Stats 305B, Stanford) extending the transformer baseline from
[Linderman, HW4](https://slinderman.github.io/stats305b/assignments/hw4/hw4.html).

Experiments compressing a GPT-style transformer for Python code generation via two techniques: **low-rank feedforward approximation** and **knowledge distillation** from `codeparrot-small`.

## Methods

### Low-rank approximation
The standard feedforward sublayer (two linear layers, 8D² parameters) is replaced with a rank-r bottleneck:

```
output = relu(x W₁) W₂,   W₁ ∈ ℝ^{D×r},  W₂ ∈ ℝ^{r×D}
```

Parameter count: 2Dr vs 8D². At D=384 and r=576 (default) each feedforward layer retains 37.5% of its original parameters (**62.5% fewer FFN params**), reducing total model size from ~31M to ~26.7M parameters (~14% overall).

### Knowledge distillation
The student (our small transformer) is trained to mimic `codeparrot-small` using a combined loss (Hinton et al., 2015):

```
L = α · L_NLL  +  (1 − α) · T² · KL( softmax(s/T) ‖ softmax(t/T) )
```

where s and t are student and teacher logits, α controls the teacher/ground-truth trade-off, and T is the softmax temperature.

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU. Training each model takes roughly 5–10 min on a T4.

## Usage

### Low-rank approximation

Train and compare baseline vs low-rank transformer on the Python corpus:

```bash
python train.py --model both --iters 2000
```

Train only the low-rank model with a custom rank:

```bash
python train.py --model lowrank --rank 300 --iters 2000
```

Add `--generate` to print text samples after training. Saves `loss_curves.png`.

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `both` | `baseline`, `lowrank`, or `both` |
| `--rank` | `576` | Bottleneck rank for low-rank FFN |
| `--iters` | `2000` | Training iterations |
| `--generate` | off | Generate text samples after training |

### Knowledge distillation

Run the full α × T grid search (16 runs):

```bash
python distill.py
```

Run a specific configuration:

```bash
python distill.py --alpha 0.7 --temperature 0.8 --iters 1000
```

Saves per-run loss plots (`distill_a{α}_t{T}.png`) and prints a validation loss table.

| Argument | Default | Description |
|----------|---------|-------------|
| `--alpha` | `0.1 0.3 0.5 0.7` | NLL weight(s); 1.0 = no distillation |
| `--temperature` | `0.8 1.0 2.0 3.0` | Softmax temperature(s) |
| `--iters` | `1000` | Training iterations per run |

## Credits

Transformer baseline adapted from [Linderman, Stats 305B HW4](https://slinderman.github.io/stats305b/assignments/hw4/hw4.html).

## Files

| File | Description |
|------|-------------|
| `data.py` | Data loading for the low-rank experiment (CodeBERT tokenizer, Python corpus CSV) |
| `models.py` | `TransformerLM`, `LowRankTransformerLM`, and shared components |
| `train.py` | Low-rank experiment: training, evaluation, text generation |
| `distill.py` | Knowledge distillation experiment: grid search over α and T |
