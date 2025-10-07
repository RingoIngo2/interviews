# Follow https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html
# Use F.cross_entropy which is similar to logsigmoid
# use torch.Dataset
# use torch.Dataloader (would abstract away the minibatch logic, not implemented here)
# use minibatch gradient descent with opt.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cpu")


def generate_ring_data(d: int, n: int, dtype=torch.float32, device="cpu"):
    assert d == 2, "Ring dataset works for 2D only."

    # ----- Inner Gaussian -----
    inner_mean = torch.zeros(d, dtype=dtype, device=device)
    inner_cov = torch.diag(torch.tensor([0.2, 0.2], dtype=dtype, device=device))
    C1 = torch.distributions.MultivariateNormal(inner_mean, inner_cov).sample((n,))

    # ----- Outer Ring -----
    angles = 2 * torch.pi * torch.rand(n, dtype=dtype, device=device)
    radius = 2.5 + 0.2 * torch.randn(n, dtype=dtype, device=device)  # radius with noise
    x_outer = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
    C2 = x_outer

    # ----- Combine -----
    X = torch.cat([C1, C2])  # shape (2n, 2)

    y = torch.cat([
        torch.zeros(n, dtype=dtype, device=device),
        torch.ones(n, dtype=dtype, device=device)
    ])

    print(X.shape, y.shape)
    return X, y


def compute_loss(X, y, model):
    # torch Xentropy takes logits for stability
    # in contrast to earlier implementation, we don't need any knowledge about the loss implementaiton anymore
    return nn.CrossEntropyLoss()(model(X), y.long())


def minibatch_gradient_descent(train_ds: TensorDataset, model, n_epochs, learning_rate, batchsize):
    # https://docs.pytorch.org/tutorials/beginner/examples_autograd/polynomial_autograd.html
    #  always call model.train() before training, and model.eval() before inference,
    #  because these are used by layers such as nn.BatchNorm2d and nn.Dropout
    #  to ensure appropriate behavior for these different phases.
    model.train()
    for i in range(n_epochs):
        for j in range(len(train_ds) // batchsize):
            lo, hi = j * batchsize, min(len(X), (j + 1) * batchsize)
            print(f"Epoch: {i}, batch: {j}, lo: {lo}, hi: {hi}")
            xb, yb = train_ds[lo:hi]
            loss = compute_loss(xb, yb, model)

            # Use autograd to compute the backward pass. This call will compute the
            # gradient of loss with respect to all Tensors with requires_grad=True.
            # After this call w.grad will be Tensors holding
            # the gradient of the loss with respect to w.
            loss.backward()
            print(f"Loss in it {i}: {loss.item()}")

            # Manually update weights using gradient descent. Wrap in torch.no_grad()
            # because weights have requires_grad=True, but we don't need to track this
            # in autograd.
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad
                    # Manually zero the gradients after updating weights
            model.zero_grad()


def plot_classification(ax, X, y, model):
    model.eval()
    with torch.no_grad():
        y_hat = F.softmax(model(X)).argmax(dim=1)
    colors = ['b', 'orange']
    markers = ['*', 'o']
    for y_target, y_hat_target in [(0, 0), (0, 1), (1, 1), (1, 0)]:
        x = X[(y == y_target) & (y_hat == y_hat_target), :]
        ax.scatter(x[:, 0], x[:, 1], label="C1", color=colors[y_target], marker=markers[y_hat_target])


if __name__ == '__main__':
    d = 2  # feature embedding dimension
    n = 100  # number of samples per class
    X, y = generate_ring_data(d, n)
    train_ds = TensorDataset(X, y)

    embedding_d = 10
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # nn.Sequential is a subclass of nn.Module that makes simple feed-forward architectures easier to write.
    model = nn.Sequential(
        nn.Linear(d, embedding_d),
        nn.ReLU(),
        nn.Linear(embedding_d, 2)
    )
    plot_classification(axes[0], X, y, model)

    print(f"init loss: {compute_loss(X, y, model)}")

    minibatch_gradient_descent(train_ds, model, n_epochs=1000, learning_rate=0.01, batchsize=10)

    plot_classification(axes[1], X, y, model)
    plt.tight_layout()
    plt.show()
