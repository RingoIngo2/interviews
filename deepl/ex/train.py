import torch
from ex.transformer import Transformer
from ex.data import IMDBDataset
from datasets import load_dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader


def gradient_descent(
    lr: float, n_epochs: int, dl: DataLoader, model: Transformer
):
    for e in range(n_epochs):
        print(f"in epoch: {e}")
        for x, y in dl:
            loss = F.cross_entropy(model(x), y)
            print(f"loss: {loss.item()}")
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    param -= lr * param.grad
            model.zero_grad()


if __name__ == "__main__":
    seq_length = 10
    bs = 3
    embedding_dim, ff_dim, n_layers = 12, 24, 5
    max_number_tokens = 1000
    dataset = load_dataset("imdb")
    train_ds = IMDBDataset(
        texts=dataset["train"]["text"][:2000],
        labels=dataset["train"]["label"][:2000],
        max_vocab=max_number_tokens,
        max_seq_length=seq_length,
    )
    train_dl = DataLoader(train_ds, batch_size=bs)
    transformer = Transformer(
        ff_dim=ff_dim,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        num_embeddings=max_number_tokens,
        padding_idx=0,
        n_output_classes=2,
    )
    gradient_descent(lr=0.001, n_epochs=2, dl=train_dl, model=transformer)
