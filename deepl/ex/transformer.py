import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Transformer(nn.Module):
    def __init__(
        self,
        ff_dim: int,
        embedding_dim: int,
        n_layers: int,
        num_embeddings: int,
        padding_idx: int,
        n_output_classes: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(ff_dim, embedding_dim) for _ in range(n_layers)]
        )
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            padding_idx=padding_idx,
            embedding_dim=embedding_dim,
        )
        self.classification_head = nn.Linear(embedding_dim, n_output_classes)

    def forward(self, x):
        x = self.embedding(x)
        # TODO: positional encoding
        for layer in self.layers:
            x = layer(x)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # add batch dimension → (1, seq_len, d_model)
        # return F.softmax(self.classification_head(x[:, -1, :]), dim=-1)
        return self.classification_head(x[:, -1, :])


class TransformerBlock(nn.Module):

    def __init__(self, ff_dim: int, embedding_dim: int):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim),
        )
        self.attention = Attention(embedding_dim=embedding_dim)
        self.layer_norm_1 = LayerNorm(embedding_dim)
        self.layer_norm_2 = LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.layer_norm_1(x)
        x = x + self.attention(x)
        x = self.layer_norm_2(x)
        x = x + self.ff(x)
        return x


class Attention(nn.Module):

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.W_K = nn.Linear(embedding_dim, embedding_dim)
        self.W_Q = nn.Linear(embedding_dim, embedding_dim)
        self.W_V = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):  # shape x = (bs, L, d_embedding)
        K = self.W_K(x)  # shape K = (bs, L, d_embedding)
        Q = self.W_Q(x)
        V = self.W_V(x)
        L = x.shape[-2]

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(
            self.embedding_dim
        )  # shape: (bs, L, L)
        mask = torch.triu(
            torch.full((L, L), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        scores = F.softmax(scores + mask, dim=-1)
        return scores @ V  # shape (bs, s, d_embedding)


class LayerNorm(nn.Module):
    def __init__(self, feat_dim: int, eps=1e-5):
        super().__init__()
        # register as parameters to have grad
        self.gamma = nn.Parameter(torch.ones(feat_dim))
        self.beta = nn.Parameter(torch.zeros(feat_dim))
        self.eps = eps

    def forward(self, x):
        m = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, unbiased=False, keepdim=True)
        return ((x - m) / (std + self.eps)) * self.gamma + self.beta


if __name__ == "__main__":
    bs, sl, embedding_dim, ff_dim = 2, 5, 7, 45
    x = torch.ones(2, 5, 7)
    print(f"shape x: {x.shape}")
    att = Attention(embedding_dim=embedding_dim)
    y = att(x)
    print(f"shape y: {y.shape}")
    transformer_block = TransformerBlock(ff_dim=ff_dim, embedding_dim=embedding_dim)
    y = transformer_block(x)
    print(f"shape y: {y.shape}")
