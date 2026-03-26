import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import logging

# Write a simple training and validation script
# Define a simple model, generate dummy data
# Define all the objects needed to train a model in PyTorch
# Finally, implement a validation script


class ClassificationDataset(Dataset):
    def __init__(self, c1, y1, c2, y2):
        self.x = torch.cat([c1, c2])
        self.y = torch.cat([y1, y2])

    def __getitem__(self, i):
        return self.x[i, :], self.y[i, :]

    def __len__(self):
        return self.x.shape[0]



class Classifier(nn.Module):
    def __init__(self, feature_dim: int, n_classes: int, hidden_dim: int):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_classes))

    def forward(self, x):
        x = self.ff(x)
        return F.softmax(x, dim=-1)


def train(model, loader, optimizer, device):
    for i, (x, y) in enumerate(loader):
        # print(f"Processing batch {i} in train")
        x.to(device)
        y.to(device)
        y_hat = model(x)
        # Input: Shape (C), (N, C)
        loss = F.cross_entropy(y, y_hat)
        loss.backward()
        # print(f"Loss in it {i}: {loss.item()}")
        with torch.no_grad():
            optimizer.step()
        optimizer.zero_grad()


def validate(model, loader, device):
    mean_loss = 0
    for i, (x, y) in enumerate(loader):
        # print(f"Processing batch {i} in train")
        x.to(device)
        y.to(device)
        with torch.no_grad():
            y_hat = model(x)
            # Input: Shape (C), (N, C)
            loss = F.cross_entropy(y, y_hat)
            print(f"Loss in it {i}: {loss.item()}")
            mean_loss += loss.item()
    return mean_loss / len(loader)


def create_data(n_samples: int, feature_dim: int):
    m1 = torch.zeros(feature_dim)
    m2 = torch.ones(feature_dim)
    c1 = torch.randn((n_samples, feature_dim)) + m1
    c2 = torch.randn((n_samples, feature_dim)) + m2
    y1 = torch.zeros(n_samples, 2)
    y2 = torch.zeros(n_samples, 2)
    y1[:, 0] = 1
    y2[:, 1] = 1
    return c1, y1, c2, y2



n_samples = 500
feature_dim = 32
batch_size = 8
n_classes = 2
hidden_dim = 64
device = torch.device("cpu")
c1, y1, c2, y2 = create_data(n_samples, feature_dim)
dataset = ClassificationDataset(c1, y1, c2, y2)
train_loader = DataLoader(dataset, batch_size)

c1_val, y1_val, c2_val, y2_val = create_data(n_samples, feature_dim)
dataset_val = ClassificationDataset(c1_val, y1_val, c2_val, y2_val)
val_loader = DataLoader(dataset_val, batch_size)

model = Classifier(feature_dim, n_classes, hidden_dim)

optimizer = AdamW(params=model.parameters())

val_loss_before = validate(model, val_loader, device)
print(f"val loss before: {val_loss_before}")
train(model, train_loader, optimizer, device)
val_loss_after = validate(model, val_loader, device)
print(f"val loss after: {val_loss_after}")

