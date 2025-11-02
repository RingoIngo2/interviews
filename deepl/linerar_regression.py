import numpy as np

# we model the data as y = w.T * x + eps,
# x, w in R^d, eps ~ N(0, sigma), we also choose w randomly
# n = number of data points
# setup
d = 10
n = 100
w_gt = np.random.uniform(-1, 1, d)
X = np.random.uniform(-1, 1, d * n).reshape([d, n])
eps = np.random.normal(0, 0.1, n)
y = np.matmul(w_gt.T, X)

# compute w_gt
# argmin w: np.sum(np.square(np.matmul(w.T, X) - y))
# = (w.T * X - y.T) * (w.T * X - y.T).T
# = w.T X X.T w - 2 y.T X w - y.T y
# derivative != 0
# 2  X X.T w - 2 X y = 0     <==> X(X.T w - y) != 0
#  X X.T w= X y
# w = (X X.T)-1 X y

# w = np.linalg.inv(X @ X.T) @ X @ y


def gradient_descent(
    step_size: float, n_steps: int, X: np.ndarray, y: np.ndarray, w_init: np.ndarray
):
    w = w_init
    for i in range(n_steps):
        delta = X.T @ w - y  # error term
        derivative = X @ delta  # X times error term (same for logistic regression)
        w -= step_size * derivative
        diff = np.mean(np.square(w - w_gt))
        print(f"Diff in it {i}: {diff}")
    return w


if __name__ == "__main__":
    gradient_descent(0.02, 50, X, y, np.ones(d))
    # print(np.linalg.norm(w - w_gt))
