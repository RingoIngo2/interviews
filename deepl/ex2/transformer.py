import torch.nn as nn
import torch
import math


class Transformer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int,
        padding_idx: int,
        n_classes: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        # TODO: Positional encoding
        self.layers = nn.ModuleList(
            TransformerBlock(embedding_dim, hidden_dim) for _ in range(n_layers)
        )
        self.classification_head = ClassificationHead(embedding_dim, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.classification_head(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.layer_norm_1 = LayerNorm(embedding_dim)
        self.layer_norm_2 = LayerNorm(embedding_dim)
        self.attention_layer = SelfAttentionLayer(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):  # input shape: [bs, L, d]
        x = self.layer_norm_1(x)
        x = x + self.attention_layer(x)
        x = self.layer_norm_2(x)
        return x + self.ff(x)


class SelfAttentionLayer(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.w_K = nn.Linear(embedding_dim, embedding_dim)
        self.w_Q = nn.Linear(embedding_dim, embedding_dim)
        self.w_V = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):  # input shape: [bs, L, d]
        K = self.w_K(x)  # input shape: [bs, L, d]
        Q = self.w_Q(x)
        V = self.w_V(x)
        scores = (
            K @ Q.transpose(-1, -2) / math.sqrt(self.embedding_dim)
        )  # input shape: [bs, L, L]
        mask = torch.triu(
            torch.full_like(scores, -torch.inf, device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        scores += mask  # input shape: [bs, L, L]
        weights = torch.softmax(scores, -1)  # input shape: [bs, L, L]
        return weights @ V  # input shape: [bs, L, d]


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):  # input shape: [bs, L, d]
        m = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return ((x - m) / std + self.eps) * self.gamma + self.beta


class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim: int, n_classes: int):
        super().__init__()
        self.ff = nn.Linear(embedding_dim, n_classes)

    def forward(self, x):  # input shape: [bs, L, d]
        x = torch.mean(x, dim=-2)  # shape: [bs, d]
        return torch.softmax(self.ff(x), -1)  # shape: [bs, n_classes]


class MixtureOfExperts(nn.Module):
    def __init__(self, embedding_dim: int, n_experts: int, hidden_dim: int, top_k: int):
        self.ffs = nn.ModuleList(
            nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim),
            )
            for _ in range(n_experts)
        )
        self.selector = nn.Linear(embedding_dim, n_experts)
        self.top_k = top_k

    def forward(self, x):
        scores = self.selector(x)
        weights = torch.softmax(scores, dim=-1)
        argmax = 0
        return


if __name__ == "__main__":
    bs, L, d, h = 4, 10, 5, 20
    x = torch.randn(bs, L, d)
    y = SelfAttentionLayer(d)(x)
    print(y.shape)
    z = TransformerBlock(d, h)(x)
    print(z.shape)

    num_embeddings = 1000
    input = torch.randint(num_embeddings, (bs, L))
    out = Transformer(num_embeddings, d, h, 6, 0, 2)(input)
    print(out.shape)
