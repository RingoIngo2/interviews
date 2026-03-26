import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Transformer(nn.Module):
    def __init__(
        self, n_layers: int, embedding_dim: int, n_classes: int, num_embeddings: int
    ):
        super().__init__()
        # TODO: pos encoding
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.layers = nn.ModuleList(
            TransformerBlock(embedding_dim) for _ in range(n_layers)
        )
        self.classification_head = ClassificationHead(n_classes, embedding_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.classification_head(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(
            embedding_dim
        )  # shape: bs, seq_length, embedding_dim
        self.attention = Attention()
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        x = self.layer_norm_1(x)
        x = x + self.attention(x)
        x = self.layer_norm_2(x)
        return x + self.ff(x)


class MoE(nn.Module):
    def __init__(self, n_experts: int, embedding_dim: int, hidden_dim: int, n_active: int):
        super().__init__()
        self.n_active = n_active
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim),
            )
            for _ in range(n_experts)
        )
        self.router = nn.Linear(embedding_dim, n_experts)

    def forward(self, x):
        weights = F.softmax(self.router(x), dim=-1)
        # select n_active
        return


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim: int):
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):  # shape: [bs, seq_length, embedding_dim]
        m = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return ((x - m) / (std + 1e-5)) * self.gamma + self.beta


class Attention(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.W_K = nn.Linear(embedding_dim, embedding_dim)
        self.Q_K = nn.Linear(embedding_dim, embedding_dim)
        self.V_K = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        K = self.W_K(x)
        Q = self.W_K(x)
        V = self.V_K(x)
        scores = K @ Q.transpose(-1, -2)
        scores = scores / math.sqrt(self.embedding_dim)
        mask = torch.triu(
            torch.full_like(scores, -torch.inf, device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        scores += mask
        weights = F.softmax(scores, dim=-1)
        return weights @ V


class ClassificationHead(nn.Module):
    def __init__(self, n_classes: int, embedding_dim: int):
        super().__init__()
        self.ff = nn.Linear(embedding_dim, n_classes)

    def forward(self, x):
        x = self.ff(x)
        return F.softmax(x, dim=-1)
