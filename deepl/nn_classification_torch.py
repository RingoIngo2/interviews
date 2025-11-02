import torch
import torch.nn.functional as F
import math

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
    x_outer = torch.stack(
        [radius * torch.cos(angles), radius * torch.sin(angles)], dim=1
    )
    C2 = x_outer

    # ----- Combine -----
    X = torch.cat([C1, C2], dim=0).T  # shape (2, 2n)
    X = torch.cat(
        [X, torch.ones((1, 2 * n), dtype=dtype, device=device)]
    )  # add bias term

    y = torch.cat(
        [
            torch.ones(n, dtype=dtype, device=device),
            -torch.ones(n, dtype=dtype, device=device),
        ]
    )

    print(X.shape, y.shape)
    return X, y


def sigmoid(z):
    return 1 / (1 + torch.exp(-z))


def compute_loss(X, y, A, w):
    # loss function stability logsigmoid!
    return -torch.sum(F.logsigmoid(y * forward_logit(X, A, w)))


def gradient_descent(X, y, A, w_init, n_steps, step_size):
    # https://docs.pytorch.org/tutorials/beginner/examples_autograd/polynomial_autograd.html
    w = w_init
    for i in range(n_steps):
        loss = compute_loss(X, y, A, w)

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
            A -= step_size * A.grad
            w -= step_size * w.grad

            # Manually zero the gradients after updating weights
            w.grad = None
    return w


def forward_logit(X, A, w):
    return w.T @ F.relu(A @ X)


def to_np(t):
    return t.detach().cpu().numpy()


def plot_classification(ax, C1, C2, A, w):
    for c in C1:
        c = torch.cat([c, torch.ones((1,))])
        p = sigmoid(forward_logit(c, A, w))
        class_symbol = "*" if p >= 0.5 else "o"
        ax.scatter(c[0], c[1], label="C1", color="b", marker=class_symbol)
    for c in C2:
        c = torch.cat([c, torch.ones((1,))])
        p = sigmoid(forward_logit(c, A, w))
        class_symbol = "*" if p >= 0.5 else "o"
        ax.scatter(c[0], c[1], label="C1", color="orange", marker=class_symbol)


if __name__ == "__main__":
    d = 2  # feature embedding dimension
    n = 100  # number of samples per class
    X, y = generate_ring_data(d, n)  # data per class

    embedding_d = 10
    A = torch.randn(
        (embedding_d, d + 1), requires_grad=True, device=device, dtype=dtype
    )  # / math.sqrt(embedding_d * (d + 1))
    w_0 = torch.rand(embedding_d, requires_grad=True)  # start point for w

    print(f"init loss: {compute_loss(X, y, A, w_0)}")

    w = gradient_descent(X, y, A, w_0, 3000, 0.0015)

    fig, axes = plt.subplots(1, 1, figsize=(12, 5), sharex=True, sharey=True)
    plot_classification(axes, X[0:2, y == 1].T, X[0:2, y == -1].T, A, w)
    plt.tight_layout()
    plt.show()
