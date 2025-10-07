import numpy as np
import matplotlib.pyplot as plt


def generate_data(d: int, size: int):
    C1 = np.random.multivariate_normal(np.random.uniform(-1, 1, d), np.diag([0.5, 0.2]), size)
    C2 = np.random.multivariate_normal(np.random.uniform(-1, 1, d), np.diag([0.1, .2]), size)
    return C1, C2


# w in R^d
# forward: sigma(w.T x)
# loss: cross entropy(y^, y) = - (log(y^)y + log(1-y^)(1-y))
# in our case with sigmoid: - (log(s(w.T x)y + log(1 - s(w.T x))(1-y))
# 1-s(x) = s(-x)
# ---> = - (log(s(w.T x))y + log(s(-w.T x))(1-y)) = - log( s(y* w.T x))
# where now y= 1/ -1


def sigmoid(z):
    return 1 / (1 + np.exp(- z))


def loss(X, y, w):
    return - np.sum(np.log(sigmoid(y * (w.T @ X))))


def gradient(X, y, w):
    delta = - (y * (1 - sigmoid(y * (w.T @ X))))  # error term (think why!)
    return X @ delta  # X times error term. Same as linear regression


def gradient_descent(X, y, w_init, n_steps, step_size):
    w = w_init
    for i in range(n_steps):
        print(f"Loss in it {i}: {loss(X, y, w)}")
        w -= step_size * gradient(X, y, w)
    return w


def plot_scatter(ax, C1, C2, w, w_name: str):
    x = np.linspace(-2, 2, 200)
    w1, w2, b = w
    y = -(w1 * x + b) / w2
    ax.plot(x, y, label=f"{round(w1, 3)}*x + {round(w2, 3)}*y + {round(b, 3)} = 0")
    ax.scatter(C1[:, 0], C1[:, 1], label="C1")
    ax.scatter(C2[:, 0], C2[:, 1], label="C2")
    ax.legend()
    ax.set_title(f"Line from {w_name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


if __name__ == '__main__':
    d = 2  # feature embedding dimension
    n = 100  # number of samples per class
    C1, C2 = generate_data(d, n)  # data per class
    w_0 = np.random.uniform(-1, 1, d + 1)  # start point for w

    X = np.concat([C1, C2]).T  # join into one dataset
    X = np.concat([X, np.ones((1, 2 * n))])  # add bias term
    y = np.concat([np.ones(n), - np.ones(n)])  # class labels
    print(X.shape, y.shape)
    print(f"init loss: {loss(X, y, w_0)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    plot_scatter(axes[0], C1, C2, w_0, 'w_0')

    w = gradient_descent(X, y, w_0, 2000, 0.001)
    # the set {x: w.T x + b = 0} is the set where s(w.T x) = 0.5, i.e. where the class prob is equal

    plot_scatter(axes[1], C1, C2, w, 'w_final')
    plt.tight_layout()
    plt.show()
