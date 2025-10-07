import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cpu")


def generate_data(d: int, size: int):
    cov1 = torch.diag(torch.tensor([0.5, 0.2], dtype=dtype, device=device))
    C1_0 = MultivariateNormal(loc=torch.rand(d), covariance_matrix=cov1).sample((size //2,))
    C1_1 = MultivariateNormal(loc=2 + torch.rand(d), covariance_matrix=cov1).sample((size //2,))
    C1 = torch.cat([C1_0, C1_1])
    cov2 = torch.diag(torch.tensor([0.1, .2], dtype=dtype, device=device))
    C2 = MultivariateNormal(loc=torch.rand(d), covariance_matrix=cov2).sample((size,))
    return C1, C2


# w in R^d
# forward: sigma(w.T x)
# loss: cross entropy(y^, y) = - (log(y^)y + log(1-y^)(1-y))
# in our case with sigmoid: - (log(s(w.T x)y + log(1 - s(w.T x))(1-y))
# 1-s(x) = s(-x)
# ---> = - (log(s(w.T x))y + log(s(-w.T x))(1-y)) = - log( s(y* w.T x))
# where now y= 1/ -1


def sigmoid(z):
    return 1 / (1 + torch.exp(- z))


def compute_loss(X, y, w):
    # loss function stability logsigmoid!
    return - torch.sum(F.logsigmoid(y * (w.T @ X)))


def gradient_descent(X, y, w_init, n_steps, step_size):
    # https://docs.pytorch.org/tutorials/beginner/examples_autograd/polynomial_autograd.html
    w = w_init
    for i in range(n_steps):
        loss = compute_loss(X, y, w)

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
            w -= step_size * w.grad

            # Manually zero the gradients after updating weights
            w.grad = None
    return w


def to_np(t):
    return t.detach().cpu().numpy()

def plot_scatter(ax, C1, C2, w, w_name: str):
    x = torch.linspace(-2, 2, 200)
    w1, w2, b = w
    y = -(w1 * x + b) / w2
    x = to_np(x)
    y = to_np(y)
    ax.plot(x, y, label=f"{w1.item():.3f}*x + {w2.item():.3f}*y + {b.item():.3f} = 0")
    ax.scatter(C1[:, 0], C1[:, 1], label="C1")
    ax.scatter(C2[:, 0], C2[:, 1], label="C2")
    ax.legend()
    ax.set_title(f"Line from {w_name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_classification(ax, C1, C2, w):
    for c in C1:
        c = torch.cat([c, torch.ones((1, ))])
        p = sigmoid(w.dot(c))
        class_symbol = "*" if p >= 0.5 else "o"
        ax.scatter(c[0], c[1], label="C1", color='b', marker=class_symbol)
    for c in C2:
        c = torch.cat([c, torch.ones((1,))])
        p = sigmoid(w.dot(c))
        class_symbol = "*" if p >= 0.5 else "o"
        ax.scatter(c[0], c[1], label="C1", color='orange', marker=class_symbol)


if __name__ == '__main__':
    d = 2  # feature embedding dimension
    n = 100  # number of samples per class
    C1, C2 = generate_data(d, n)  # data per class
    w_0 = torch.rand(d + 1, requires_grad=True)  # start point for w

    X = torch.cat([C1, C2]).T.to(device)  # join into one dataset
    X = torch.cat([X, torch.ones((1, 2 * n), dtype=dtype, device=device)])  # add bias term
    y = torch.cat(
        [torch.ones(n, dtype=dtype, device=device), - torch.ones(n, dtype=dtype, device=device)])  # class labels
    print(X.shape, y.shape)
    print(f"init loss: {compute_loss(X, y, w_0)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    plot_scatter(axes[0], C1, C2, w_0, 'w_0')

    w = gradient_descent(X, y, w_0, 2000, 0.001)
    # the set {x: w.T x + b = 0} is the set where s(w.T x) = 0.5, i.e. where the class prob is equal

    # plot_scatter(axes[1], C1, C2, w, 'w_final')
    plot_classification(axes[1], C1, C2, w)
    plt.tight_layout()
    plt.show()
