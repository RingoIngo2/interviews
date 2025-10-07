# where did I do the padding in my time series project?

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from datasets import load_dataset
from collections import Counter
from typing import Optional
from functools import cached_property


class AttentionLayer(nn.Module):
    def __init__(self, feat_dim: int, att_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.att_dim = att_dim
        self.W_K = nn.Linear(feat_dim, att_dim)
        self.W_Q = nn.Linear(feat_dim, att_dim)
        self.W_V = nn.Linear(feat_dim, att_dim)

    def forward(self, x):
        K = self.W_K(x)  # dim: (bs, L, att_dim)
        Q = self.W_Q(x)
        V = self.W_V(x)
        L = x.shape[-2]
        # note the normalization!
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(
            self.att_dim
        )  # dim (bs, L, L)

        mask = torch.triu(
            torch.full((L, L), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        scores += mask
        weights = F.softmax(scores, dim=-1)
        return weights @ V


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


class TransformerBlock(nn.Module):
    def __init__(self, feat_dim: int, ff_dim: int):
        super().__init__()
        self.attention_layer = AttentionLayer(feat_dim, feat_dim)

        # Note: we need two layer norm instances
        # self.norm1 = nn.LayerNorm(feat_dim)
        self.norm1 = LayerNorm(feat_dim)
        self.norm2 = LayerNorm(feat_dim)
        self.ff = nn.Sequential(
            nn.Linear(feat_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, feat_dim)
        )

    def forward(self, x):
        x = x + self.attention_layer(self.norm1(x))
        return x + self.ff(self.norm2(x))


class SinosoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, seq_len=5000):
        super().__init__()
        N = 10000
        i = torch.arange(0, d_model // 2)
        div_term = N ** ((2 * i) / d_model)
        position = torch.arange(seq_len).unsqueeze(1)

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as a buffer so it moves with the model (e.g. to GPU) but is not trainable
        # Any attribute that is a torch.nn.Parameter is automatically registered as a model parameter.
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe


class ClassificationTransformer(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        ff_dim: int,
        n_classes: int,
        num_embeddings: int,
        padding_idx: int,
        seq_len: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, feat_dim, padding_idx=padding_idx)
        self.TBlock = TransformerBlock(feat_dim, ff_dim)
        self.classification_head = nn.Linear(feat_dim, n_classes)
        self.sinosoidal_embedding = SinosoidalPositionalEmbedding(feat_dim, seq_len)

    def forward(self, x):
        x = self.embedding(x)
        x = self.sinosoidal_embedding(x)
        x = self.TBlock(x)
        # pool over sequence dim
        x = torch.mean(x, dim=1)
        return self.classification_head(x)


class Tokenizer:
    end_of_sequence = "<EOS>"
    unknown = "<UNKNOWN>"
    padding = "<PADDING>"

    def __init__(self, corpus: list[str], max_vocab: int = 10000):
        self.id_to_token = self.init_id_to_token(corpus, max_vocab - 3)
        self.token_to_id = self.revert_dict(self.id_to_token)

    def init_id_to_token(self, corpus, max_vocab):
        tokens = Counter()
        for s in corpus:
            tokens_s = self.tokenize(s)
            tokens.update(tokens_s)
        most_common = tokens.most_common(max_vocab)
        id_to_token = {0: self.padding, 1: self.end_of_sequence, 2: self.unknown}
        id_to_token.update({i + 3: t for i, (t, _) in enumerate(most_common)})
        return id_to_token

    @cached_property
    def id_padding(self) -> int:
        return self.token_to_id[self.padding]

    @cached_property
    def id_eos(self) -> int:
        return self.token_to_id[self.end_of_sequence]

    @cached_property
    def id_unknown(self) -> int:
        return self.token_to_id[self.unknown]

    @staticmethod
    def revert_dict(d):
        return {i: t for t, i in d.items()}

    def __call__(self, s: str, length: Optional[int]) -> list[int]:
        token_ids = [
            self.token_to_id.get(t, self.id_unknown) for t in self.tokenize(s)
        ] + [self.id_eos]
        if not length:
            return token_ids
        if length < len(token_ids):
            return token_ids[:length]
        return token_ids + [self.id_padding] * (length - len(token_ids))

    @staticmethod
    def tokenize(s: str) -> list[str]:
        return s.lower().split()


class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, max_tokens, max_vocab):
        self.tokenizer = Tokenizer(corpus=texts, max_vocab=max_vocab)
        self.texts = texts
        self.labels = labels
        self.max_tokens = max_tokens
        self.max_vocab = max_vocab

    def __len__(self):
        return len(self.texts)

    @staticmethod
    def p_to_one_hot(p: int):
        if p:
            return torch.tensor([0.0, 1.0])
        return torch.tensor([1.0, 0.0])

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx], self.max_tokens)
        return torch.tensor(tokens), self.p_to_one_hot(self.labels[idx])


def train_model(
    data_loader: DataLoader, n_epochs: int, model: nn.Module, learning_rate: float
):
    model.train()
    for i in range(n_epochs):
        for x_b, y_b in data_loader:
            loss = nn.CrossEntropyLoss()(model(x_b), y_b)
            print(f"ep: {i}, loss: {loss.item()}")
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param
            model.zero_grad()


if __name__ == "__main__":
    seq_length = 128
    dataset = load_dataset("imdb")
    train_ds = IMDBDataset(
        texts=dataset["train"]["text"][:2000],
        labels=dataset["train"]["label"][:2000],
        max_vocab=10000,
        max_tokens=seq_length,
    )
    # embedded tokens
    feat_dim, att_dim = 512, 64
    batch_size = 256
    ff_dim = 254
    model = ClassificationTransformer(
        feat_dim=feat_dim,
        ff_dim=ff_dim,
        n_classes=2,
        num_embeddings=train_ds.max_vocab,
        padding_idx=train_ds.tokenizer.id_padding,
        seq_len=seq_length,
    )
    x_0, y_0 = train_ds[0]
    x_1, y_1 = train_ds[1]
    output = model(torch.cat([x_0.unsqueeze(0), x_1.unsqueeze(0)]))
    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size)
    print(output.shape)
    train_model(train_dl, n_epochs=200, model=model, learning_rate=0.001)
