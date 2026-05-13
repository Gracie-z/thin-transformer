import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """Single head of causal self-attention."""

    def __init__(self, head_size, context_window_size, embed_size=384):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(context_window_size, context_window_size))
        )

    def forward(self, x):
        T = x.shape[1]
        scores = self.query(x) @ self.key(x).transpose(-2, -1) / (self.head_size**0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        return F.softmax(scores, dim=-1) @ self.value(x)


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention (outputs are summed across heads)."""

    def __init__(self, num_heads, head_size, context_window_size, embed_size=384):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, context_window_size, embed_size) for _ in range(num_heads)
        ])

    def forward(self, x):
        return sum(h(x) for h in self.heads)


class FeedForward(nn.Module):
    """Standard two-layer feedforward block with 4× expansion."""

    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )

    def forward(self, x):
        return self.net(x)


class LowRankFeedForward(nn.Module):
    """Feedforward block with low-rank weight factorization: output = relu(x W1) W2.

    Parameter count: 2 * embed_size * rank  vs  8 * embed_size^2 for standard FFN.
    Break-even rank: 4 * embed_size.
    """

    def __init__(self, embed_size, rank):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(embed_size, rank))
        self.W2 = nn.Parameter(torch.randn(rank, embed_size))

    def forward(self, x):
        return F.relu(x @ self.W1) @ self.W2


class _TransformerBlock(nn.Module):
    """Transformer block: causal multi-head attention + feedforward, both with residuals."""

    def __init__(self, context_window_size, ff_layer, embed_size=384, num_heads=6):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.attn = MultiHeadAttention(
            num_heads, embed_size // num_heads, context_window_size, embed_size
        )
        self.ff = ff_layer

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class _TransformerLMBase(nn.Module):
    """Shared forward pass and generation logic for both model variants."""

    def __init__(self, vocab_size, context_window_size, blocks, embed_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_window_size = context_window_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(context_window_size, embed_size)
        self.blocks = nn.Sequential(*blocks)
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, targets=None):
        B, T = token_ids.shape
        x = self.token_embedding(token_ids) + self.position_embedding(
            torch.arange(T, device=token_ids.device)
        )
        logits = self.lm_head(self.ln_f(self.blocks(x)))  # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens):
        for _ in range(max_new_tokens):
            ctx = token_ids[:, -self.context_window_size :]
            logits, _ = self(ctx)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids


class TransformerLM(_TransformerLMBase):
    """GPT-style autoregressive language model with standard feedforward layers."""

    def __init__(
        self,
        vocab_size,
        context_window_size,
        embed_size=384,
        num_heads=6,
        n_layers=6,
    ):
        blocks = [
            _TransformerBlock(
                context_window_size, FeedForward(embed_size), embed_size, num_heads
            )
            for _ in range(n_layers)
        ]
        super().__init__(vocab_size, context_window_size, blocks, embed_size)


class LowRankTransformerLM(_TransformerLMBase):
    """GPT-style autoregressive language model with low-rank feedforward layers."""

    def __init__(
        self,
        vocab_size,
        context_window_size,
        rank,
        embed_size=384,
        num_heads=6,
        n_layers=6,
    ):
        blocks = [
            _TransformerBlock(
                context_window_size,
                LowRankFeedForward(embed_size, rank),
                embed_size,
                num_heads,
            )
            for _ in range(n_layers)
        ]
        super().__init__(vocab_size, context_window_size, blocks, embed_size)
