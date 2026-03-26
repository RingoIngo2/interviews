import torch.nn
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ex3.data import IMDBDataset
from ex3.transfomer import Transformer


def train(model: torch.nn.Module, lr, n_epochs, dl):
    for _ in range(n_epochs):
        for x, y in dl:
            # TODO: with model.train()
            y_hat = model(x)
            loss = F.cross_entropy(y_hat, y)
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    param -= lr * param.grad
            model.zero_grad()



if __name__ == "__main__":
    bs = 10
    n_epochs = 20
    lr = 0.001
    num_embeddings = 1000
    embedding_dim = 512
    hidden_dim = 1024
    n_layers = 5
    n_classes = 2

    data = load_dataset("imdb")["train"]
    dataset = IMDBDataset(list(data["text"]), list(data["label"]), num_embeddings)
    dl = DataLoader(dataset, bs)
    model = Transformer(
        num_embeddings,
        embedding_dim,
        hidden_dim,
        n_layers,
        0,
        # dataset.tokenizer.padding_idx(),
        n_classes,
    )
